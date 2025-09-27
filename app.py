import os
import sys
import json
import shutil
import subprocess
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from utils_defaults import default_n_topics, default_ber_min_topic_size

# chemins utilisés dans l'app (relatifs au répertoire du projet)
TOPICS_PATH = "results/topics.json"
CORPUS_PATH = "data/corpus_clean.csv"
BACKUP_DIR = "results/backups/"
PIPELINE_SCRIPT = "topic_pipeline.py"

st.set_page_config(page_title="Topic Explorer (éditable)", layout="wide")


# ---------------- Helpers utilitaires ----------------
def load_json(path: str) -> Dict[str, Any]:
    """
    Charger le JSON results/topics.json si présent,
    sinon retourner une structure par défaut vide.
    """
    if not os.path.exists(path):
        return {"lda": [], "nmf": [], "bertopic": [], "coherence": {}, "_meta": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]):
    """
    Sauvegarder le JSON sur disque (écrase).
    Je n'effectue pas de backup ici — on utilise backup_file() avant si besoin.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def backup_file(path: str) -> str:
    """
    Créer un backup simple de TOPICS_PATH dans BACKUP_DIR.
    Retourne le chemin du backup créé (ou chaîne vide si pas de fichier source).
    """
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if not os.path.exists(path):
        return ""
    base = os.path.basename(path)
    idx = 0
    while True:
        dest = os.path.join(BACKUP_DIR, f"{base}.bak{idx}.json")
        if not os.path.exists(dest):
            shutil.copy2(path, dest)
            return dest
        idx += 1


def load_corpus(path: str = CORPUS_PATH) -> pd.DataFrame:
    """
    Charger le corpus nettoyé (data/corpus_clean.csv).
    S'il n'existe pas, retourner un DataFrame vide avec les colonnes attendues.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["id", "title", "text_clean"])
    df = pd.read_csv(path)
    # garantir la colonne id entière
    if "id" in df.columns:
        try:
            df["id"] = df["id"].astype(int)
        except Exception:
            df = df.reset_index().rename(columns={"index": "id"})
            df["id"] = df["id"].astype(int) + 1
    else:
        df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].astype(int) + 1
    if "title" not in df.columns:
        df["title"] = ""
    return df[["id", "title", "text_clean"]]


def build_topic_map(lst: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Transformer la liste [{'topic_id':.., 'keywords':.., 'documents':..}] en dict {topic_id: obj}.
    Utile pour lookup rapide et édition.
    """
    m: Dict[int, Dict[str, Any]] = {}
    for t in lst:
        tid = int(t.get("topic_id", -1))
        m[tid] = dict(t)
        m[tid].setdefault("label", f"Topic {tid}")
        m[tid].setdefault("documents", [])
        m[tid].setdefault("keywords", [])
    return m


def topic_list_from_map(m: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    L'inverse de build_topic_map : reconstruire la liste triée par topic_id.
    On utilise ça avant de sauvegarder dans topics_data.
    """
    out: List[Dict[str, Any]] = []
    for tid in sorted(m.keys()):
        obj = m[tid]
        out.append(
            {
                "topic_id": int(tid),
                "keywords": obj.get("keywords", []),
                "documents": obj.get("documents", []),
                "label": obj.get("label", f"Topic {tid}"),
            }
        )
    return out


def model_key_from_label(lbl: str) -> str:
    """
    Convertit l'étiquette du modèle (BERTopic/LDA/NMF) en clé minuscule attendue
    dans topics_data ('bertopic','lda','nmf').
    """
    return lbl.lower()


# ---------------- state initial ----------------
# Je charge les données une fois dans la session pour pouvoir les modifier sans écrire immédiatement.
if "topics_data" not in st.session_state:
    st.session_state["topics_data"] = load_json(TOPICS_PATH)
if "corpus_df" not in st.session_state:
    st.session_state["corpus_df"] = load_corpus()
if "selected_model_label" not in st.session_state:
    st.session_state["selected_model_label"] = "LDA"  # choix par défaut pour l'UI


# raccourcis pour accéder/update le state
def get_topics_data() -> Dict[str, Any]:
    return st.session_state["topics_data"]


def set_topics_data(data: Dict[str, Any]):
    st.session_state["topics_data"] = data


def get_corpus_df() -> pd.DataFrame:
    return st.session_state["corpus_df"]


# -------------- opérations mutantes (vraiment modifient session_state) --------------
def rename_topic(model_label: str, topic_id: int, new_label: str) -> bool:
    """
    Renomme un topic pour le modèle donné en mémoire (session_state).
    Retourne True si ok, False sinon.
    """
    key = model_key_from_label(model_label)
    td = get_topics_data()
    if key not in td:
        return False
    topic_map = build_topic_map(td[key])
    if topic_id not in topic_map:
        return False
    topic_map[topic_id]["label"] = new_label.strip() or topic_map[topic_id].get("label", f"Topic {topic_id}")
    td[key] = topic_list_from_map(topic_map)
    set_topics_data(td)
    return True


def move_documents_within_model(model_label: str, doc_ids: List[int], target_topic_id: int) -> Dict[str, int]:
    """
    Déplace une liste de documents (ids) vers un topic cible au sein du même modèle.
    - Supprime les ids des topics sources
    - Ajoute les ids au topic cible sans duplication
    Renvoie un dict avec counts : {'removed': X, 'added': Y}
    """
    key = model_key_from_label(model_label)
    td = get_topics_data()
    if key not in td:
        return {"removed": 0, "added": 0}
    topic_map = build_topic_map(td[key])

    # docs qui existent réellement dans ce modèle
    all_docs = sorted({d for obj in topic_map.values() for d in obj.get("documents", [])})
    docs_to_move = [d for d in doc_ids if d in all_docs]
    if not docs_to_move:
        return {"removed": 0, "added": 0}

    removed = 0
    # on retire des topics sources
    for tid, obj in list(topic_map.items()):
        if tid == target_topic_id:
            continue
        cur = obj.get("documents", [])
        new_cur = [d for d in cur if d not in docs_to_move]
        removed += len(cur) - len(new_cur)
        topic_map[tid]["documents"] = new_cur

    # on ajoute au topic cible (set pour éviter doublons)
    target_docs = set(topic_map[target_topic_id].get("documents", []))
    before = len(target_docs)
    target_docs.update(docs_to_move)
    topic_map[target_topic_id]["documents"] = sorted(list(target_docs))
    added = len(target_docs) - before

    td[key] = topic_list_from_map(topic_map)
    set_topics_data(td)
    return {"removed": removed, "added": added}


def merge_topics_within_model(model_label: str, selected_topic_ids: List[int]) -> bool:
    """
    Fusionne plusieurs topics dans le même modèle : tout est fusionné dans le premier id passé.
    Retourne True si réussite.
    """
    if not selected_topic_ids or len(selected_topic_ids) < 2:
        return False
    key = model_key_from_label(model_label)
    td = get_topics_data()
    if key not in td:
        return False
    topic_map = build_topic_map(td[key])
    target = selected_topic_ids[0]
    if target not in topic_map:
        return False
    for o in selected_topic_ids[1:]:
        if o not in topic_map:
            continue
        oobj = topic_map[o]
        # mots-clés : on ajoute sans dupliquer, en gardant l'ordre
        for kw in oobj.get("keywords", []):
            if kw not in topic_map[target]["keywords"]:
                topic_map[target]["keywords"].append(kw)
        # documents : union
        topic_map[target]["documents"] = sorted(list(set(topic_map[target].get("documents", [])) | set(oobj.get("documents", []))))
        # suppression du topic fusionné
        del topic_map[o]
    td[key] = topic_list_from_map(topic_map)
    set_topics_data(td)
    return True


# helper safe_rerun : essaye d'appeler st.experimental_rerun() si disponible
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        # certaines versions de streamlit n'exposent pas experimental_rerun;
        # tant que les modifications sont dans session_state, l'UI se mettra à jour
        # au prochain événement utilisateur.
        return


# ---------------- UI layout ----------------
st.title("Topic Explorer — édition (renommer / déplacer / fusionner)")
st.write("Les modifications sont gardées en session. Appuie sur 'Sauvegarder modifications' pour écrire results/topics.json.")


# Sidebar : paramètres pipeline (optionnel)
st.sidebar.header("Pipeline & paramètres (optionnel)")
n_docs = len(get_corpus_df())
lda_default = default_n_topics(n_docs)
nmf_default = default_n_topics(n_docs)
ber_default = default_ber_min_topic_size(n_docs)

run_mode = st.sidebar.selectbox("Mode d'exécution", ["all", "lda", "nmf", "lda_nmf", "bertopic", "coherence"])
lda_n = st.sidebar.number_input("n_topics LDA", min_value=1, max_value=200, value=lda_default)
nmf_n = st.sidebar.number_input("n_topics NMF", min_value=1, max_value=200, value=nmf_default)
top_n = st.sidebar.number_input("Top N mots", min_value=3, max_value=50, value=10)
ber_min = st.sidebar.number_input("BERTopic min_topic_size", min_value=1, max_value=100, value=ber_default)
compute_coh = st.sidebar.checkbox("Activer calcul de la cohérence (coûteux en ressources/temps)", value=False)

python_exec = sys.executable or "python"
if st.sidebar.button("Relancer pipeline (local)"):
    # relance topic_pipeline.py en local avec les paramètres choisis
    cmd = [python_exec, PIPELINE_SCRIPT, "--run", run_mode, "--n_topics_lda", str(lda_n), "--n_topics_nmf", str(nmf_n), "--top_n_words", str(top_n), "--ber_min_topic_size", str(ber_min)]
    if compute_coh:
        cmd.append("--compute-coherence")
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        st.sidebar.text_area("Pipeline stdout/stderr", value=(completed.stdout + "\n\n" + completed.stderr), height=300)
        # recharge depuis le disque pour avoir la dernière version
        st.session_state["topics_data"] = load_json(TOPICS_PATH)
        st.session_state["corpus_df"] = load_corpus()
        st.success("Pipeline terminé et données rechargées.")
        safe_rerun()
    except Exception as e:
        st.error(f"Impossible d'exécuter le pipeline local: {e}")


if st.sidebar.button("Actualiser la cohérence (local)"):
    cmd = [python_exec, PIPELINE_SCRIPT, "--run", "coherence"]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        st.sidebar.text_area("Coherence stdout/stderr", value=(completed.stdout + "\n\n" + completed.stderr), height=300)
        st.session_state["topics_data"] = load_json(TOPICS_PATH)
        st.success("Cohérence recalculée.")
        safe_rerun()
    except Exception as e:
        st.error(f"Erreur en calculant la cohérence: {e}")


# actions disponibles
st.sidebar.markdown("---")
st.sidebar.header("Actions")
action = st.sidebar.radio("Action", ["Visualiser", "Renommer topic", "Déplacer documents", "Fusionner topics", "Sauvegarder & Export"])


# ---------------- Visualiser ----------------
if action == "Visualiser":
    st.header("Visualiser — choisir le modèle et le topic")
    # le selectbox du modèle est ici (zone principale) — c'est plus ergonomique.
    model_label = st.selectbox("Modèle à visualiser", ["BERTopic", "LDA", "NMF"], index=["BERTopic", "LDA", "NMF"].index(st.session_state.get("selected_model_label", "LDA")))
    st.session_state["selected_model_label"] = model_label
    key = model_key_from_label(model_label)
    topic_list = get_topics_data().get(key, [])
    topic_map = build_topic_map(topic_list)

    if not topic_map:
        st.warning("Aucun topic à afficher pour ce modèle.")
    else:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            sel_topic = st.selectbox("Choisir topic id", sorted(topic_map.keys()))
            t = topic_map[sel_topic]
            st.header(f"{t.get('label')}  (id={sel_topic})")
            st.subheader("Mots-clés")
            kws = t.get("keywords", [])
            if kws:
                dfkw = pd.DataFrame({"keyword": kws, "rank": list(range(len(kws), 0, -1))})
                chart = alt.Chart(dfkw).mark_bar().encode(x="rank:Q", y=alt.Y("keyword:N", sort="-x"))
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("Pas de mots-clés.")
            st.subheader("Wordcloud")
            if kws:
                wc = WordCloud(collocations=False).generate(" ".join(kws))
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            st.subheader("Documents associés")
            docs = t.get("documents", [])
            if docs:
                found = get_corpus_df()[get_corpus_df()["id"].isin(docs)]
                if not found.empty:
                    q = st.text_input("Filtrer texte (mot)")
                    display = found
                    if q:
                        display = found[found["text_clean"].str.contains(q, case=False, na=False)]
                    for _, r in display.iterrows():
                        st.markdown(f"**{int(r['id'])} — {r.get('title','(no title)')}**")
                        st.text(r.get("text_clean", ""))
                        st.markdown("---")
                else:
                    st.warning("Aucun document trouvé dans data/corpus_clean.csv pour ces IDs.")
            else:
                st.write("Aucun document listé pour ce topic.")
        with col_right:
            st.subheader("Résumé des topics")
            summary = []
            for tid, obj in topic_map.items():
                summary.append({"topic_id": tid, "label": obj.get("label", ""), "n_docs": len(obj.get("documents", []))})
            if summary:
                st.dataframe(pd.DataFrame(summary).sort_values("topic_id").reset_index(drop=True), height=400)
            else:
                st.info("Aucun topic à afficher.")


# ---------------- Renommer ----------------
elif action == "Renommer topic":
    st.header("Renommer un topic")
    model_label = st.selectbox("Sélectionner le modèle", ["BERTopic", "LDA", "NMF"])
    key = model_key_from_label(model_label)
    topic_list = get_topics_data().get(key, [])
    if not topic_list:
        st.info("Aucun topic disponible pour ce modèle.")
    else:
        topic_map = build_topic_map(topic_list)
        sel_topic = st.selectbox("Topic à renommer", sorted(topic_map.keys()))
        cur_label = topic_map[sel_topic].get("label", f"Topic {sel_topic}")
        new_label = st.text_input("Nouveau label", value=cur_label)
        if st.button("Appliquer le label"):
            ok = rename_topic(model_label, sel_topic, new_label)
            if ok:
                st.success("Renommé.")
                safe_rerun()
            else:
                st.error("Erreur : impossible de renommer (topic non trouvé).")


# ---------------- Déplacer documents ----------------
elif action == "Déplacer documents":
    st.header("Déplacer documents (dans le même modèle)")
    model_label = st.selectbox("Sélectionner le modèle", ["BERTopic", "LDA", "NMF"])
    key = model_key_from_label(model_label)
    topic_list = get_topics_data().get(key, [])
    if not topic_list:
        st.info("Aucun topic pour ce modèle.")
    else:
        topic_map = build_topic_map(topic_list)
        all_docs = sorted({d for obj in topic_map.values() for d in obj.get("documents", [])})
        if not all_docs:
            st.info("Aucun document listé pour ce modèle.")
        else:
            sel_docs = st.multiselect("Documents (IDs) à déplacer", all_docs)
            tgt = st.selectbox("Topic cible", sorted(topic_map.keys()))
            if st.button("Déplacer"):
                if not sel_docs:
                    st.warning("Sélectionnez au moins un document.")
                else:
                    res = move_documents_within_model(model_label, sel_docs, tgt)
                    st.success(f"{res['removed']} doc(s) retiré(s) des topics sources, {res['added']} doc(s) ajoutés au topic {tgt}.")
                    safe_rerun()


# ---------------- Fusionner topics ----------------
elif action == "Fusionner topics":
    st.header("Fusionner topics (dans le même modèle)")
    model_label = st.selectbox("Sélectionner le modèle", ["BERTopic", "LDA", "NMF"])
    key = model_key_from_label(model_label)
    topic_list = get_topics_data().get(key, [])
    if not topic_list or len(topic_list) < 2:
        st.info("Besoin d'au moins 2 topics pour fusionner.")
    else:
        topic_map = build_topic_map(topic_list)
        sel = st.multiselect("Topics à fusionner (premier = cible)", sorted(topic_map.keys()))
        if len(sel) >= 2 and st.button("Fusionner"):
            ok = merge_topics_within_model(model_label, sel)
            if ok:
                st.success(f"Fusionné {sel[1:]} -> {sel[0]}")
                safe_rerun()
            else:
                st.error("Erreur lors de la fusion.")


# ---------------- Sauvegarder & Export ----------------
elif action == "Sauvegarder & Export":
    st.header("Sauvegarder & Exporter")
    st.markdown("Sauvegarde les modifications stockées en session vers results/topics.json (backup automatique).")
    if st.button("Créer backup maintenant"):
        b = backup_file(TOPICS_PATH)
        st.success(f"Backup: {b}" if b else "Aucun backup créé (fichier absent).")
    if st.button("Sauvegarder modifications (écrase results/topics.json)"):
        b = backup_file(TOPICS_PATH)
        save_json(TOPICS_PATH, get_topics_data())
        st.success("Sauvegardé. Backup: " + (b or "none"))
        safe_rerun()
    st.download_button("Télécharger topics.json", data=json.dumps(get_topics_data(), ensure_ascii=False, indent=2), file_name="topics_edited.json", mime="application/json")


# ---------------- Bas de page : cohérence & comparaison ----------------
st.markdown("---")
st.subheader("État de la cohérence (si présent)")
coh = get_topics_data().get("coherence", {})
if not coh:
    st.info("Aucune cohérence calculée.")
else:
    st.table(pd.DataFrame([{"method": k.upper(), "avg_coherence": v} for k, v in coh.items()]))

st.markdown("---")
st.subheader("Comparaison rapide des méthodes")
methods = ["bertopic", "lda", "nmf"]
comp = []
td = get_topics_data()
for m in methods:
    lst = td.get(m, [])
    comp.append({"method": m.upper(), "n_topics": len(lst), "n_documents_assigned": sum(len(t.get("documents", [])) for t in (lst or []))})
st.table(pd.DataFrame(comp).set_index("method"))

st.caption("Remarque : les modifications sont stockées en mémoire (session). Appuie sur 'Sauvegarder modifications' pour écrire results/topics.json.")
