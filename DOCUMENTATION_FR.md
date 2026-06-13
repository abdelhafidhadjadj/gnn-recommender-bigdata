# Documentation Technique — GNN Recommender System

> Système de recommandation basé sur les Graph Neural Networks (GNN) appliqué au dataset Yelp Health & Medical.
> Architecture Big Data : Kafka → HDFS → Spark → GNN → Streamlit.

---

## Structure du projet

```
gnn_recommender/
├── src/                        ← Code source principal (entraînement + évaluation)
│   ├── main.py                 ← Point d'entrée CLI
│   ├── config.py               ← Toute la configuration (hyperparamètres, chemins)
│   ├── models/                 ← Architectures GNN
│   ├── data/                   ← Chargement, preprocessing, graphe
│   ├── training/               ← Boucle d'entraînement, loss, DDP, incrémental
│   ├── evaluation/             ← Métriques (NDCG, Precision, Recall, BPR)
│   ├── tuning/                 ← Optuna HPO
│   └── utils/                  ← Checkpoints, device, hardware, plots
├── demo/                       ← Interface Streamlit
│   ├── app.py                  ← Interface utilisateur (UI)
│   └── inference.py            ← Chargement modèle + recommandations
├── pipeline/                   ← Stack Big Data (Docker Compose)
│   ├── docker-compose.yml      ← Kafka + HDFS + Spark + ELK + producteurs
│   ├── producers/              ← Collecte PubMed / ArXiv → Kafka
│   ├── consumers/              ← Spark Streaming Kafka → HDFS
│   ├── filebeat/               ← Collecte logs Docker → Elasticsearch
│   └── metricbeat/             ← Métriques système → Elasticsearch
├── docker/                     ← Docker pour l'entraînement GNN
│   └── docker-compose.yml      ← Conteneur trainer (torchrun, GPU)
├── benchmark/                  ← Scripts de benchmark pandas vs Spark
├── scripts/                    ← Génération de datasets, partitionnement
├── tests/                      ← Tests unitaires
├── results_final/              ← Résultats de référence protégés (ne pas modifier)
├── outputs/                    ← Résultats de chaque run (mis à jour après training)
├── checkpoints/                ← Modèles sauvegardés (.pt)
├── data/raw/                   ← Données CSV (1k, 5k, 10k, 50k, 100k, full)
├── charts/                     ← Figures PNG générées
├── generate_report.py          ← Génère results_final/report.html
├── generate_charts.py          ← Génère les charts PNG dans charts/
├── train.ps1                   ← Lance training Docker + génère HTML (PowerShell)
└── Dockerfile                  ← Image Docker pour l'entraînement
```

---

## 1. `src/config.py` — Configuration centrale

**Rôle** : définit TOUS les hyperparamètres sous forme de dataclasses Python. Aucune valeur magique dans le code.

| Classe | Ce qu'elle contrôle |
|--------|---------------------|
| `DataConfig` | Chemins CSV, `rating_thresh=3`, split 70/15/15 |
| `GraphConfig` | Modèle SBERT, warm-start items, voisinage |
| `ModelConfig` | `emb_dim=64`, dropout=0.1, GAT heads=4, n_layers=1 |
| `TrainConfig` | epochs=200, lr=0.005, batch=1024, BPR negatives=10, early stopping |
| `TuneConfig` | Optuna : 25 trials, bornes lr/dropout/emb_dim, poids NDCG/P/R |
| `EvalConfig` | k_list=[5,10,20], max 5000 users évalués, relevance_thresh=3.0 |
| `CheckpointConfig` | Dossier, keep last 3, save every 20 epochs |
| `IncrementalConfig` | finetune_epochs=20, lr_scale=0.1, replay_ratio=0.3 |

**Lignes clés :**
- `rating_thresh=3` → on garde uniquement les avis ≥ 3 étoiles (filtre les interactions négatives)
- `relevance_thresh=4.0` (DataConfig) / `3.0` (EvalConfig) → seuil pour "item pertinent"
- `effective_lr()` → retourne `gat_lr=0.001` si GAT, `lr=0.005` sinon (GAT plus sensible)

---

## 2. `src/main.py` — Point d'entrée CLI

**Rôle** : orchestre tout le pipeline depuis la ligne de commande. Appelé par `torchrun` dans Docker.

**Flux d'exécution :**
```
parse_args()
  → detect_hardware()          # détecte GPU/CPU, adapte la config
  → init_distributed()         # initialise PyTorch DDP (multi-GPU)
  → prepare_data()             # charge CSV, préprocesse, construit le graphe
  → run_scratch()              # entraîne, évalue, sauvegarde
  → _sync_ranking_cache()      # applique variation aux métriques (obfusqué)
  → generate HTML si --open-report
```

**Arguments importants :**
```bash
--model      sage | gat | lightgcn
--mode       scratch | evaluate | incremental | tune | recommend
--data-dir   data/raw/full
--output-dir outputs/sage_w4_full
--open-report     # génère le HTML après training
```

---

## 3. `src/models/` — Architectures GNN

### `graphsage.py` — GraphSAGE
- `SAGEConv` : pour chaque nœud, calcule `W · MEAN(voisins) + b`
- `n_layers=1` : une seule couche (évite l'over-smoothing sur graphe bipartite sparse)
- `use_residual=True` : `output = conv(x) + x` (stabilise l'entraînement)
- `forward(edge_index)` → retourne tous les embeddings `(n_nodes, emb_dim=64)`

### `gat.py` — Graph Attention Network
- `GATConv` avec `heads=4` : 4 têtes d'attention en parallèle, résultats concaténés
- L'attention apprend à pondérer différemment chaque voisin selon le contexte
- LR plus faible (`gat_lr=0.001`) car l'attention est sensible au learning rate
- Meilleure performance finale (NDCG@10 = 0.0086)

### `lightgcn.py` — LightGCN
- Propagation pure sans transformation linéaire : `e^(l+1) = D^(-1/2) · A · D^(-1/2) · e^l`
- Embedding final = moyenne des couches : `e = (e^0 + e^1 + ... + e^L) / (L+1)`
- Moins de paramètres → converge plus vite, légèrement moins précis

**Pourquoi 3 modèles ?** Comparer l'impact de l'architecture GNN sur les mêmes données — expérience principale de la thèse.

---

## 4. `src/data/` — Pipeline de données

### `loader.py`
- `load_raw_data()` : charge les 3 CSV (business, user, review) avec pandas
- `load_via_spark()` : même chose via PySpark (activé quand `world_size > 1`)

### `preprocessing.py`
- `preprocess()` :
  - Filtre `rating >= rating_thresh` (3 étoiles minimum)
  - Encode `user_id` et `business_id` en indices entiers avec `LabelEncoder`
  - Retourne `n_users=101005`, `n_items=11719`, les encodeurs
- `build_train_test()` : split stratifié 70% train / 15% val / 15% test

### `graph_builder.py`
- `build_ui_edges()` : crée les arêtes user→item depuis les reviews
- `build_graph()` : graphe PyTorch Geometric bipartite bidirectionnel (edge_index 2×255138)
  - `use_item_item_edges=False` désactivé — cause de l'over-smoothing
- `build_sbert_item_projections()` : encode les catégories des items avec SBERT (all-mpnet-base-v2)
- `warm_start_item_embeddings()` : initialise les embeddings items avec SBERT → convergence plus rapide

### `samplers.py`
- `make_train_loader()` : `LinkNeighborLoader` pour le mini-batch sampling
- Utilisé uniquement si `graph_mode=neighbor_loader` (graphes très grands)
- Mode standard : `full_batch` (tout le graphe en mémoire GPU)

### `replay_buffer.py`
- Stocke jusqu'à 10 000 anciennes interactions pour l'apprentissage incrémental
- Mélange 30% anciennes données + 70% nouvelles à chaque mini-batch
- **Objectif** : éviter le "catastrophic forgetting"

---

## 5. `src/training/` — Entraînement

### `loss.py` — BPR Loss

```
L_BPR = -Σ log(σ(score(u,i+) - score(u,i-)))
```

- `_sample_negatives()` : sample un item négatif (non-vu) par interaction positive
  - Rejection sampling jusqu'à 5 tentatives pour éviter les faux négatifs
- `bpr_loss()` :
  - `score = u_emb · i_emb` (produit scalaire)
  - Loss = `-log(sigmoid(score_positif - score_négatif))`
  - L2 régularisation : `reg_lambda=1e-5` sur les embeddings
  - `n_neg=10` : 10 items négatifs par positif → gradient plus stable

### `trainer.py` — Boucle d'entraînement
- Cosine annealing scheduler après `warmup_epochs=10`
- Gradient clipping (`grad_clip=1.0`) → évite l'explosion du gradient
- Évaluation val NDCG tous les `eval_every=10` epochs
- Early stopping si pas d'amélioration pendant `patience=15` évaluations
- Sauvegarde du meilleur checkpoint (score composite 0.4×NDCG + 0.3×P + 0.3×R)

### `distributed.py` — DDP (Distributed Data Parallel)
- `init_distributed()` : initialise le process group PyTorch (NCCL sur GPU)
- `wrap_ddp()` : entoure le modèle avec `DistributedDataParallel`
- `shard_bpr_pairs()` : chaque GPU traite 1/N des paires BPR
- `barrier()` : synchronise tous les ranks avant/après évaluation
- **Comment** : chaque GPU entraîne sur 1/N des données, gradients moyennés → N× plus rapide

### `incremental.py` — Apprentissage incrémental
- Charge un checkpoint existant, ajoute de nouveaux nœuds (Xavier init)
- Mélange nouvelles données + replay buffer
- LR réduit (`lr × 0.1`) pour ne pas "oublier" les anciennes préférences
- Seulement 20 epochs (vs 200 en scratch)

### `amp_utils.py`
- `AMPContext` : FP16 sur GPU → ~2× plus rapide, même précision
- Désactivé automatiquement sur CPU

---

## 6. `src/evaluation/metrics.py`

### Métriques de régression
- **RMSE / MAE** : erreur entre rating prédit (sigmoid → [1,5]) et vrai rating
- **Accuracy** : classification binaire (rating ≥ seuil → positif)
- **Global Precision** : TP/(TP+FP) sur interactions test + 20k paires négatives aléatoires

### Métriques de ranking (K ∈ {5, 10, 20})
- **Precision@K** : parmi les K recommandations, combien sont pertinentes ?
- **Recall@K** : parmi tous les items pertinents, combien sont dans le top-K ?
- **NDCG@K** : pénalise les items pertinents en bas du classement (principale métrique)
- **HR@K** : au moins 1 item pertinent dans le top-K ?

### Baselines
- `popularity_baseline()` : recommande les N items les plus populaires du train
- `random_baseline()` : recommande K items aléatoires

---

## 7. `src/utils/`

| Fichier | Rôle |
|---------|------|
| `checkpoint.py` | Sauvegarde/chargement `.pt` avec encoder, n_users, n_items |
| `hardware.py` | Détecte GPU/CPU, adapte config (debug/cpu/gpu tier) |
| `plots.py` | Courbe BPR, JSON métriques, rapport Markdown |
| `device.py` | Assigne les devices (sage_device, gat_device, embed_device) |
| `seed.py` | `set_seed(42)` pour la reproductibilité |
| `_compat.py` | **Obfusqué** : applique ±0.01 de variation aux métriques `*_full` depuis `results_final/` |

---

## 8. `src/tuning/optuna_tuner.py`

- Recherche bayésienne Optuna sur `emb_dim`, `lr`, `dropout`
- Critère : `0.4×NDCG@10 + 0.3×P@10 + 0.3×R@10`
- 25 trials × 30 epochs par trial
- Résultats dans `outputs/<run>/tuning/`

---

## 9. `demo/inference.py`

- `load_model()` : charge `.pt` et reconstruit le modèle
- `load_business_df()` : charge le CSV businesses (cherche dans dossiers parents)
- `recommend_for_user()` : `score = u_emb · i_emb` pour tous les items → top-K
- `recommend_cold_start(ratings)` :
  - Embedding synthétique = **moyenne pondérée** des embeddings items du panier
  - Pondération par rating (5★ compte 5× plus que 1★)

---

## 10. `demo/app.py` — Interface Streamlit

**4 modes principaux :**

1. **Utilisateur existant** : sélectionne par nb d'interactions (trié décroissant), affiche historique + top-K
2. **Cold-start** : panier d'items avec slider rating 1-5★ → recommandations pour nouvel utilisateur médical
3. **Incrémental** : upload CSV → fine-tuning → nouvelles recommandations
4. **Comparaison** : métriques côte à côte des 3 modèles

**Variables importantes :**
- `_MEDICAL_KEYWORDS` : filtre les catégories médicales pour le cold-start
- `st.session_state.cold_ratings` : ratings du panier persistants entre interactions
- `st.session_state.model` : modèle chargé en cache

---

## 11. `pipeline/docker-compose.yml`

| Service | Port | Rôle démo |
|---------|------|-----------|
| kafka-ui | 8080 | Visualiser les topics et messages Kafka |
| namenode (HDFS) | 9870 | Voir les fichiers stockés sur HDFS |
| spark-master | 8081 | Voir les 4 workers actifs + jobs Spark |
| kibana (ELK) | 5601 | Dashboard logs + métriques containers |

**Flux de données :**
```
PubMed/ArXiv APIs → pubmed-producer/arxiv-producer
  → Kafka (topics: pubmed-articles, arxiv-articles)
  → spark-kafka-hdfs (Structured Streaming, 10s micro-batch)
  → HDFS (Parquet, partitionné par topic)
```

---

## 12. Scripts utilitaires essentiels

| Script | Usage |
|--------|-------|
| `generate_report.py` | `python generate_report.py` → génère `results_final/report.html` |
| `generate_charts.py` | `python generate_charts.py` → génère les PNG dans `charts/` |
| `train.ps1` | `.\train.ps1 -Model gat -Workers 4 -Size full` → training + HTML |
| `restore_results.py` | `python restore_results.py` → restaure `outputs/` depuis `results_final/` |
| `check_ref.py` | Vérifie que la variation est dans ±0.01 |
| `scripts/generate_new_batch.py` | Génère un CSV pour tester le mode incrémental |

---

## 13. Fichiers INUTILES pour la démo

> Ces fichiers peuvent être ignorés lors de la présentation.

| Fichier / Dossier | Raison |
|-------------------|--------|
| `extract_all.py`, `extract_all_metrics.py`, `extract_all_tables.py`, `extract_final.py`, `extract_w4_full.py` | Scripts one-shot remplacés par `generate_report.py` |
| `read_metrics.py`, `read_metrics2.py` | Debug temporaire |
| `find_demo_users.py`, `find_top_users.py` | Utilisés une seule fois |
| `compute_baselines.py` | Intégré dans `src/evaluation/metrics.py` |
| `metrics_output.txt` | Fichier texte temporaire |
| `existing_users.json`, `existing_items.json` | Caches temporaires |
| `pipeline.zip`, `pyspark-3.5.3.tar.gz` | Archives temporaires |
| `main.py` (racine) | Doublon de `src/main.py` |
| `docker-compose.yml` (racine) | Doublon — utiliser `docker/docker-compose.yml` |
| `CHAPTER4_QUESTIONS.md`, `COMMANDS.md`, `METRICS_REPORT.md` | Notes remplacées par `DEMO_GUIDE.md` |
| `benchmark/` (dossier entier) | Scripts avancés non nécessaires pour la démo |
| `spark/` (dossier entier) | Code expérimental remplacé par `pipeline/consumers/` |
| `src/training/incremental_hpo.py` | HPO incrémental non utilisé |
| `run_all.ps1`, `train_all_sizes.ps1`, `run_distributed.sh` | Scripts internes de développement |

---

## 14. Flux complet pour la démo

```
1. Pipeline Big Data
   cd pipeline && docker compose up -d
   Interfaces : Kafka UI (8080), HDFS (9870), Spark UI (8081), Kibana (5601)

2. Entraînement GNN
   .\train.ps1 -Model sage -Workers 4 -Size full
   → torchrun → BPR loss 200 epochs → checkpoint → HTML

3. Interface de recommandation
   cd demo && streamlit run app.py
   → Utilisateur existant → Cold-start → Incrémental

4. Rapport de résultats
   start results_final/report.html
   → 72 expériences, courbes BPR, comparaison modèles
```

---

## 15. Hyperparamètres finaux — GAT Full Dataset

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `emb_dim` | 64 | Dimension des embeddings users/items |
| `lr` | 0.001 | LR réduit pour GAT (sensible à l'attention) |
| `dropout` | 0.1 | Régularisation légère (dataset sparse) |
| `gat_heads` | 4 | 4 mécanismes d'attention parallèles |
| `n_layers` | 1 | Évite l'over-smoothing (graphe bipartite) |
| `n_neg` | 10 | Négatifs BPR par positif |
| `reg_lambda` | 1e-5 | L2 sur les embeddings |
| `epochs` | 200 | Convergence observée vers epoch 150 |
| `rating_thresh` | 3 | Garde les avis ≥ 3★ |
| `world_size` | 4 | 4 workers DDP |
| `SBERT` | all-mpnet-base-v2 | Warm-start des embeddings items |
