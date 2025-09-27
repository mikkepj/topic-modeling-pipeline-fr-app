# Modélisation de sujets (LDA / NMF / BERTopic) + Visualisationn (Streamlit)

[Voir le dépôt sur GitHub](https://github.com/saiken77/topic-modeling-pipeline-fr-app)

Résumé
Ce dépôt contient un pipeline complet pour **prétraiter** un corpus d'articles, **extraire des sujets** avec trois méthodes (LDA, NMF, BERTopic), **comparer** les résultats et **visualiser / éditer** les topics via une application Streamlit. Le produit final est un unique fichier JSON `results/topics.json` mis à jour **progressivement** (après LDA, après NMF, après BERTopic, puis après calcul de cohérence si demandé).

---

## Arborescence du dépôt (raccourci)

```
topic-modeling-pipeline-fr-app/
│
├── data/
│   ├── raw/                  # corpus brut (articles.csv attendu)
│   └── corpus_clean.csv      # corpus nettoyé (généré par le pipeline)
│
├── results/
│   ├── topics.json           # fichier unique de sortie (lda,nmf,bertopic,coherence,_meta)
│   └── backups/              # backups automatiques (écrasement safe)
│
├── app.py                    # application Streamlit pour visualiser / éditer / relancer pipeline
├── topic_pipeline.py         # script principal CLI (prétraitement + modèles + sauvegarde progressive)
├── utils_defaults.py         # fonctions default_n_topics / default_ber_min_topic_size
└── requirements.txt          # (recommandé) dépendances Python
```

---

## Prérequis

Recommandé : créer un environnement virtuel (conda / venv). Exemple (conda) :

```bash
conda create -n nlp39 python=3.9 -y
conda activate nlp39
```

Installer les dépendances (adapté selon ton gestionnaire) :

```bash
pip install -r requirements.txt
```

**Packages clés (exemples)** :

* `spacy`
* `fr-core-news-sm` (modèle spaCy français)
* `scikit-learn`
* `gensim`
* `bertopic`
* `sentence-transformers`
* `pandas`, `numpy`
* `streamlit`
* `wordcloud`, `matplotlib`, `altair`

Installer le modèle spaCy (obligatoire) :

```bash
python -m spacy download fr_core_news_sm
```

**Remarques :**

* BERTopic et sentence-transformers téléchargent/pèsent des modèles ; prévoir de la RAM et un peu de temps.
* Si tu utilises un GPU, configurer les packages compatibles (optionnel).

---

## Fichiers importants & rôle

* `topic_pipeline.py`
  Script CLI principal. Il :

  * génère `data/corpus_clean.csv` à partir de `data/raw/articles.csv` si absent (option `--preprocess` pour forcer),
  * exécute LDA, NMF, BERTopic (modes `--run`),
  * sauvegarde **progressivement** `results/topics.json` après chaque modèle,
  * calcule la cohérence `c_v` uniquement si `--compute-coherence` est passé (ou en mode `--run coherence`),
  * met à jour `_meta` dans `results/topics.json` (timestamps ISO) pour indiquer la fraîcheur.
    Voir section « Utilisation » pour la liste complète d'options.

* `utils_defaults.py`
  Contient :

  * `default_n_topics(n_docs)` — heuristique dynamique pour `n_topics` (LDA/NMF),
  * `default_ber_min_topic_size(n_docs)` — heuristique pour `min_topic_size` de BERTopic.
    Ces fonctions sont utilisées par le pipeline et par l'interface Streamlit pour fixer les valeurs par défaut dynamiques.

* `app.py`
  Application web interactive pour :

  * visualiser topics (BERTopic / LDA / NMF),
  * éditer (renommer topics, déplacer documents entre topics du même modèle, fusionner topics),
  * relancer le pipeline localement (en passant des paramètres),
  * recalculer uniquement la cohérence,
  * sauvegarder les changements (backup automatique).

* `results/topics.json` (format attendu) :
  Exemple simplifié :

  ```json
  {
    "lda": [
      {"topic_id": 0, "keywords": ["mot1","mot2"], "documents": [1,5], "label":"mot1 mot2 mot3"},
      ...
    ],
    "nmf": [ ... ],
    "bertopic": [ ... ],
    "coherence": {"lda": 0.32, "nmf": 0.28, "bertopic": 0.35},
    "_meta": {"lda": "2025-09-26T12:00:00", "nmf": "...", "bertopic":"...", "coherence": "..."}
  }
  ```

---

## Utilisation — commandes essentielles

> **Préparation** : assure-toi que `data/raw/articles.csv` existe et contient au minimum les colonnes `titre`/`texte` (ou `title`/`text`).

### 1) Générer le corpus propre (si absent ou pour forcer)

```bash
python topic_pipeline.py --preprocess
```

Ce script va :

* charger `data/raw/articles.csv`,
* appliquer le nettoyage / lemmatisation via spaCy (`fr_core_news_sm`),
* produire `data/corpus_clean.csv`.

### 2) Exécuter pipeline (sans cohérence — plus rapide)

```bash
python topic_pipeline.py --run all
```

### 3) Exécuter pipeline et calculer la cohérence (plus long)

```bash
python topic_pipeline.py --run all --compute-coherence
# ou pour n'exécuter QUE la cohérence (après avoir généré topics.json) :
python topic_pipeline.py --run coherence
```

### 4) Exécution ciblée

* LDA uniquement :

  ```bash
  python topic_pipeline.py --run lda --n_topics_lda 8
  ```
* NMF uniquement :

  ```bash
  python topic_pipeline.py --run nmf --n_topics_nmf 6
  ```
* LDA + NMF (lda_nmf) :

  ```bash
  python topic_pipeline.py --run lda_nmf
  ```
* BERTopic (nécessite `--ber_min_topic_size` entier, sinon la valeur par défaut heuristique est utilisée) :

  ```bash
  python topic_pipeline.py --run bertopic --ber_min_topic_size 5
  ```

### 5) Lancer l'interface Streamlit

```bash
streamlit run app.py
```

Interface :

* sélectionner le modèle (BERTopic / LDA / NMF),
* visualiser les topics, mots-clés, wordcloud et documents,
* renommer / déplacer / fusionner topics (modifications en session, puis `Sauvegarder modifications` pour persister),
* relancer le pipeline local via la sidebar (paramètres dynamiques calculés depuis `utils_defaults`),
* bouton « Calculer uniquement la cohérence » si besoin.

---

## Exemple d'exécution rapide (step-by-step)

```bash
# 1) Créer l'environnement & installer dépendances
conda create -n nlp39 python=3.9 -y
conda activate nlp39
pip install -r requirements.txt
python -m spacy download fr_core_news_sm

# 2) Générer corpus propre (si nécessaire)
python topic_pipeline.py --preprocess

# 3) Lancer pipeline (rapide sans cohérence)
python topic_pipeline.py --run all

# 4) Lancer Streamlit
streamlit run app.py
```

---

## Dépannage / FAQ

* **Erreur `fr_core_news_sm` introuvable**
  → exécuter : `python -m spacy download fr_core_news_sm`

* **Longtemps bloqué au calcul de la cohérence**
  → relancer sans cohérence : `python topic_pipeline.py --run all --no-coherence` (ou `--run all` sans `--compute-coherence`), puis lancer la cohérence séparément lorsqu'on est prêt : `python topic_pipeline.py --run coherence`.

* **Problèmes avec BERTopic / sentence-transformers**

  * Ces étapes nécessitent souvent plus de mémoire et parfois l’accès internet la première fois (téléchargement de modèles). Si l’étape bloque, vérifier la RAM/disque et relancer avec `--run bertopic` seul pour mieux suivre l’avancement.

* **Streamlit ne montre pas les modifications immédiates**

  * L’app stocke les modifications en `st.session_state`. Clique sur `Sauvegarder modifications` pour persister sur disque. Si tu veux annuler, recharger l’app ou utiliser le bouton pour recharger le pipeline.

* **Paramètre `stop_words='french'` (erreur sklearn)**

  * Le pipeline utilise la conversion explicite des stopwords spaCy en liste compatible sklearn pour éviter ce problème.

---

## Auteurs

Projet réalisé dans le cadre du module **Natural Language Processing (NLP)** — M1/S2 Fouilles de Données & Intelligence Artificielle (Université Virtuelle du Burkina Faso).

- **SAWADOGO Abdel Saïd Najib**
- **COULIBALY Cheick Ahmed**
- **BAZIE Dureel**

---

## Encadrement

- **Dr. Rodrique KAFANDO** — Responsable pédagogique du module NLP

---

## 📜 Licence

Ce projet est sous licence MIT. Voir [LICENSE](LICENCE) pour plus de détails.