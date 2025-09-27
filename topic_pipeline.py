import os
import json
import shutil
import argparse
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as SPACY_STOP

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# BERTopic et sentence-transformers (nécessaires si on veut bertopic)
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# mes utilitaires de valeurs par défaut (heuristique)
from utils_defaults import default_n_topics, default_ber_min_topic_size

# chemins utilisés
RAW_CORPUS = "data/raw/articles.csv"
CLEAN_CORPUS = "data/corpus_clean.csv"
OUTPUT_JSON = "results/topics.json"
BACKUP_DIR = "results/backups/"

# motif pour tokenisation basique (pour la conversion des stopwords spaCy)
_TOKEN_PATTERN = r"(?u)\b\w\w+\b"


# ---------------- Utilitaires ----------------
def now_iso() -> str:
    """Timestamp ISO UTC sans microsecondes."""
    return datetime.utcnow().replace(microsecond=0).isoformat()


def backup_file(path: str, backup_dir: str = BACKUP_DIR) -> str:
    """
    Sauvegarde un backup du fichier path dans backup_dir (nom incrementation).
    Renvoie le chemin du backup.
    """
    os.makedirs(backup_dir, exist_ok=True)
    if not os.path.exists(path):
        return ""
    base = os.path.basename(path)
    idx = 0
    while True:
        dest = os.path.join(backup_dir, f"{base}.bak{idx}.json")
        if not os.path.exists(dest):
            shutil.copy2(path, dest)
            return dest
        idx += 1


def save_partial(output_path: str, key: str, payload: Any, meta_update: Optional[Dict[str,str]] = None):
    """
    Met à jour (ou crée) results/topics.json :
    - charge le fichier si existant, sinon crée une structure vide
    - remplace data[key] par payload
    - met à jour _meta avec meta_update (timestamps)
    - écrit le fichier
    Cela permet d'écrire les résultats par étapes (LDA -> save -> NMF -> save -> BERTopic -> save)
    """
    data = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    # assurer la structure minimale
    data.setdefault("lda", [])
    data.setdefault("nmf", [])
    data.setdefault("bertopic", [])
    data.setdefault("coherence", {})
    data.setdefault("_meta", {})
    data[key] = payload
    if meta_update:
        for mk, mv in meta_update.items():
            data["_meta"][mk] = mv
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def load_clean_corpus(path: str = CLEAN_CORPUS) -> pd.DataFrame:
    """
    Charger data/corpus_clean.csv ; lève si le fichier n'est pas conforme.
    Je m'attends à des colonnes : id, title, text_clean
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus propre introuvable : {path}")
    df = pd.read_csv(path)
    if "text_clean" not in df.columns:
        raise ValueError("Le CSV propre doit contenir la colonne 'text_clean'.")
    if "id" in df.columns:
        df["id"] = df["id"].astype(int)
    else:
        df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].astype(int) + 1
    if "title" not in df.columns:
        df["title"] = ""
    return df[["id", "title", "text_clean"]]


# ---------------- Prétraitement ----------------
def ensure_spacy_model():
    """
    Charge le pipeline spaCy fr_core_news_sm.
    On suppose que l'utilisateur a installé le modèle au préalable (readme).
    """
    return spacy.load("fr_core_news_sm")


def tokenize_spacy(text: str, nlp) -> str:
    """
    Tokenize + lemmatisation via spaCy :
    - minuscule
    - enlever stopwords spaCy
    - ne garder que tokens alphabétiques > 2 chars
    """
    if pd.isna(text):
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if (not token.is_stop) and token.is_alpha and len(token.text) > 2]
    return " ".join(tokens)


def preprocess_raw_to_clean(raw_path: str = RAW_CORPUS, out_path: str = CLEAN_CORPUS, force: bool = False):
    """
    Génère data/corpus_clean.csv depuis data/raw/articles.csv si besoin.
    - si out_path existe et force=False : skip
    - sinon effectue le nettoyage spaCy et écrit le CSV
    """
    if os.path.exists(out_path) and not force:
        print(f"{out_path} exists, skipping (use --preprocess to force).")
        return
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    print("Prétraitement: chargement spaCy et nettoyage...")
    nlp = ensure_spacy_model()
    df = pd.read_csv(raw_path)
    # travailler avec colonne titre/texte possibles (français/anglais)
    if "titre" in df.columns:
        title_col = "titre"
    elif "title" in df.columns:
        title_col = "title"
    else:
        title_col = None
    if "texte" in df.columns:
        text_col = "texte"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError("Raw CSV must contain 'titre'/'texte' or 'title'/'text'.")
    # appliquer le nettoyage
    df["title"] = df[title_col].apply(lambda t: tokenize_spacy(t, nlp))
    df["text_clean"] = df[text_col].apply(lambda t: tokenize_spacy(t, nlp))
    # garantir la colonne id
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].astype(int) + 1
    else:
        df["id"] = df["id"].astype(int)
    out = df[["id", "title", "text_clean"]]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Corpus propre sauvegardé -> {out_path} ({len(out)} docs)")


# ---------------- Stopwords conversion ----------------
def get_spacy_stop_tokens(token_pattern: str = _TOKEN_PATTERN) -> List[str]:
    """
    SpaCy fournit des stopwords sous forme de strings qui peuvent contenir des caractères
    non alphanumériques. sklearn attend une liste de tokens compatbiles.
    Ici je tokenise les stopwords spaCy pour fournir une liste utilisable par CountVectorizer/TfidfVectorizer.
    """
    pattern = re.compile(token_pattern)
    tokenized = set()
    for sw in SPACY_STOP:
        if not isinstance(sw, str):
            continue
        s = sw.lower()
        found = pattern.findall(s)
        if found:
            tokenized.update(found)
        else:
            tokenized.add(s)
    return sorted(tokenized)


# ---------------- Helpers topics ----------------
def get_top_words(components: np.ndarray, feature_names: List[str], top_n: int) -> List[List[str]]:
    """Récupère top_n mots pour chaque composante (topic)."""
    out = []
    for comp in components:
        idx = comp.argsort()[::-1][:top_n]
        out.append([feature_names[i] for i in idx])
    return out


def assign_documents(doc_topic_matrix: np.ndarray) -> Dict[int, List[int]]:
    """
    Assigne chaque document au topic le plus probable (argmax).
    Renvoie mapping topic_index -> [doc_index,...]
    """
    assignments: Dict[int, List[int]] = {}
    top_topic = np.argmax(doc_topic_matrix, axis=1)
    for doc_idx, t in enumerate(top_topic):
        assignments.setdefault(int(t), []).append(doc_idx)
    return assignments


# ---------------- LDA / NMF ----------------
def run_lda(df: pd.DataFrame, n_topics: int, top_n_words: int, max_features: int, min_df: int, stop_words: List[str], random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Entraîne un LDA avec CountVectorizer.
    Retourne la liste de topics au format attendu pour topics.json.
    """
    texts = df["text_clean"].astype(str).tolist()
    doc_ids = df["id"].astype(int).tolist()
    n_topics = min(n_topics, max(1, len(texts)))
    vect = CountVectorizer(max_df=0.95, min_df=min_df, max_features=max_features, stop_words=stop_words, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    feat = vect.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, learning_method="batch")
    doc_topic = lda.fit_transform(X)
    comps = lda.components_
    top_words = get_top_words(comps, feat, top_n_words)
    assigns = assign_documents(doc_topic)
    out = []
    for tid, words in enumerate(top_words):
        docs_idx = assigns.get(tid, [])
        mapped = [int(doc_ids[i]) for i in docs_idx]
        out.append({"topic_id": int(tid), "keywords": words, "documents": sorted(mapped), "label": " ".join(words[:3])})
    return out


def run_nmf(df: pd.DataFrame, n_topics: int, top_n_words: int, max_features: int, min_df: int, stop_words: List[str], tfidf_matrix: Optional[np.ndarray] = None, tfidf_vectorizer = None, random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Entraîne un NMF sur une matrice TF-IDF (optionnellement fournie).
    Renvoie la liste de topics compatible avec topics.json.
    """
    texts = df["text_clean"].astype(str).tolist()
    doc_ids = df["id"].astype(int).tolist()
    n_topics = min(n_topics, max(1, len(texts)))
    if tfidf_matrix is None or tfidf_vectorizer is None:
        vect = TfidfVectorizer(max_df=0.95, min_df=min_df, max_features=max_features, stop_words=stop_words, ngram_range=(1,2))
        X = vect.fit_transform(texts)
    else:
        X = tfidf_matrix
        vect = tfidf_vectorizer
    feat = vect.get_feature_names_out()
    nmf = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_
    top_words = get_top_words(H, feat, top_n_words)
    assigns = assign_documents(W)
    out = []
    for tid, words in enumerate(top_words):
        docs_idx = assigns.get(tid, [])
        mapped = [int(doc_ids[i]) for i in docs_idx]
        out.append({"topic_id": int(tid), "keywords": words, "documents": sorted(mapped), "label": " ".join(words[:3])})
    return out


def run_lda_nmf(df: pd.DataFrame, n_topics_lda: int, n_topics_nmf: int, top_n_words: int, max_features: int, min_df: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Wrapper pour exécuter LDA puis NMF (on réutilise la matrice TF-IDF pour NMF).
    On renvoie dict {'lda': [...], 'nmf': [...]}
    """
    stop_tokens = get_spacy_stop_tokens()
    lda_res = run_lda(df, n_topics=n_topics_lda, top_n_words=top_n_words, max_features=max_features, min_df=min_df, stop_words=stop_tokens)
    vect = TfidfVectorizer(max_df=0.95, min_df=min_df, max_features=max_features, stop_words=stop_tokens, ngram_range=(1,2))
    X_tfidf = vect.fit_transform(df["text_clean"].astype(str).tolist())
    nmf_res = run_nmf(df, n_topics=n_topics_nmf, top_n_words=top_n_words, max_features=max_features, min_df=min_df, stop_words=stop_tokens, tfidf_matrix=X_tfidf, tfidf_vectorizer=vect)
    return {"lda": lda_res, "nmf": nmf_res}


# ---------------- BERTopic ----------------
def run_bertopic(df: pd.DataFrame, top_n_words: int, embedding_model_name: str, nr_topics: str, min_topic_size: int) -> List[Dict[str, Any]]:
    """
    Exécute BERTopic :
    - crée embeddings via sentence-transformers
    - fit_transform sur les textes
    - récupère topics (on ignore topic -1 s'il y en a)
    Attention : cette étape peut être lourde (RAM + temps).
    """
    texts = df["text_clean"].astype(str).tolist()
    doc_ids = df["id"].astype(int).tolist()
    embed = SentenceTransformer(embedding_model_name)
    embeddings = embed.encode(texts, show_progress_bar=True)
    model = BERTopic(embedding_model=embed, nr_topics=nr_topics, min_topic_size=int(min_topic_size))
    topics, probs = model.fit_transform(texts, embeddings)
    unique_topics = sorted([t for t in set(topics) if t != -1])
    out = []
    for t in unique_topics:
        words = [w for w, _ in model.get_topic(t)]
        doc_idxs = [i for i, top in enumerate(topics) if top == t]
        mapped = [int(doc_ids[i]) for i in doc_idxs]
        out.append({"topic_id": int(t), "keywords": words[:top_n_words], "documents": sorted(mapped), "label": " ".join(words[:3])})
    return out


# ---------------- Cohérence (optimisé) ----------------
def compute_coherence(topic_lists: Dict[str, List[Dict[str,Any]]], texts_tokenized: List[List[str]], top_n_for_coherence: int = 10, max_topics: int = 200) -> Dict[str, float]:
    """
    Calcule la cohérence c_v pour chaque méthode en une passe.
    - topic_lists : {'lda': [...], 'nmf': [...], 'bertopic': [...]}
    - texts_tokenized : list de tokens par doc (prétraités)
    On initialise un seul CoherenceModel par méthode avec la liste de topics : plus rapide
    que de construire un CoherenceModel par topic.
    """
    result: Dict[str, float] = {}
    dictionary = Dictionary(texts_tokenized)
    for method, topics in topic_lists.items():
        if not topics:
            result[method] = float("nan")
            continue
        topics_words = []
        for t in topics[:max_topics]:
            words = t.get("keywords", [])[:top_n_for_coherence]
            if words:
                topics_words.append(list(words))
        if not topics_words:
            result[method] = float("nan")
            continue
        cm = CoherenceModel(topics=topics_words, texts=texts_tokenized, dictionary=dictionary, coherence="c_v")
        try:
            score = float(cm.get_coherence())
        except Exception:
            score = float("nan")
        result[method] = score
    return result


# ---------------- Validation topics.json ----------------
def validate_topics_json(path: str = OUTPUT_JSON) -> Dict[str, Any]:
    """
    Vérifie que results/topics.json existe et contient les clés minimales.
    Renvoie la structure chargée, ou lève FileNotFoundError si absent.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("lda", [])
    data.setdefault("nmf", [])
    data.setdefault("bertopic", [])
    data.setdefault("coherence", {})
    data.setdefault("_meta", {})
    return data


# ---------------- main : CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Topic pipeline with progressive save.")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--run", choices=["all", "lda", "nmf", "lda_nmf", "bertopic", "coherence"], default="all")
    parser.add_argument("--n_topics_lda", type=int, default=None)
    parser.add_argument("--n_topics_nmf", type=int, default=None)
    parser.add_argument("--ber_min_topic_size", type=int, default=None)
    parser.add_argument("--top_n_words", type=int, default=10)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=10000)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--nr_topics", type=str, default="auto")
    parser.add_argument("--compute-coherence", action="store_true", dest="compute_coherence")
    parser.add_argument("--output", type=str, default=OUTPUT_JSON)
    args = parser.parse_args()

    # si on demande prétraitement ou si le corpus nettoyé est absent -> générer
    if not os.path.exists(CLEAN_CORPUS) or args.preprocess:
        preprocess_raw_to_clean(RAW_CORPUS, CLEAN_CORPUS, force=args.preprocess)

    df = load_clean_corpus(CLEAN_CORPUS)
    n_docs = len(df)

    # décider des valeurs par défaut si non fournies
    n_topics_lda = args.n_topics_lda if args.n_topics_lda is not None else default_n_topics(n_docs)
    n_topics_nmf = args.n_topics_nmf if args.n_topics_nmf is not None else default_n_topics(n_docs)
    ber_min = args.ber_min_topic_size if args.ber_min_topic_size is not None else default_ber_min_topic_size(n_docs)

    print(f"n_docs={n_docs} -> n_topics_lda={n_topics_lda}, n_topics_nmf={n_topics_nmf}, ber_min={ber_min}")

    # garantir que le fichier de sortie existe avec une structure minimale
    if not os.path.exists(args.output):
        save_partial(args.output, "lda", [], meta_update={"lda": now_iso(), "nmf": now_iso(), "bertopic": now_iso(), "coherence": now_iso()})

    # LDA (et sauvegarde immédiatement)
    if args.run in ("all", "lda", "lda_nmf"):
        print("Running LDA...")
        stop_tokens = get_spacy_stop_tokens()
        lda_res = run_lda(df, n_topics=n_topics_lda, top_n_words=args.top_n_words, max_features=args.max_features, min_df=args.min_df, stop_words=stop_tokens)
        save_partial(args.output, "lda", lda_res, meta_update={"lda": now_iso()})

    # NMF (et sauvegarde)
    if args.run in ("all", "nmf", "lda_nmf"):
        print("Running NMF...")
        stop_tokens = get_spacy_stop_tokens()
        vect = TfidfVectorizer(max_df=0.95, min_df=args.min_df, max_features=args.max_features, stop_words=stop_tokens, ngram_range=(1,2))
        X_tfidf = vect.fit_transform(df["text_clean"].astype(str).tolist())
        nmf_res = run_nmf(df, n_topics=n_topics_nmf, top_n_words=args.top_n_words, max_features=args.max_features, min_df=args.min_df, stop_words=stop_tokens, tfidf_matrix=X_tfidf, tfidf_vectorizer=vect)
        save_partial(args.output, "nmf", nmf_res, meta_update={"nmf": now_iso()})

    # BERTopic (et sauvegarde)
    if args.run in ("all", "bertopic"):
        print("Running BERTopic...")
        bres = run_bertopic(df, top_n_words=args.top_n_words, embedding_model_name=args.embedding_model, nr_topics=args.nr_topics, min_topic_size=ber_min)
        save_partial(args.output, "bertopic", bres, meta_update={"bertopic": now_iso()})

    # cohérence
    if args.run == "coherence" or args.compute_coherence:
        print("Validating topics.json and computing coherence...")
        topics = validate_topics_json(args.output)
        tokenized = [t.split() for t in df["text_clean"].astype(str).tolist()]
        coh = compute_coherence({"lda": topics.get("lda", []), "nmf": topics.get("nmf", []), "bertopic": topics.get("bertopic", [])}, tokenized)
        save_partial(args.output, "coherence", coh, meta_update={"coherence": now_iso()})

    print("Done.")


if __name__ == "__main__":
    main()
