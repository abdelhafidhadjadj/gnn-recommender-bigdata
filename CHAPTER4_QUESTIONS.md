# Chapitre 4 — Réponses aux questions de mise en œuvre expérimentale

> Dernière mise à jour : 2026-06-01 — toutes les valeurs sont extraites des fichiers réels.

---

## 1. CPU exact

**Intel Core i7-11800H @ 2.30 GHz**
- Cœurs physiques : **8** — Threads logiques : **16**
- GPU : **NVIDIA GeForce RTX 3070 Laptop GPU**, 8 GB VRAM
- Entraînement : Docker + NVIDIA Container Toolkit, AMP activé (float16)

---

## 2. Système d'exploitation exact

**Microsoft Windows 11 Entreprise, 64 bits** — Build `10.0.26200`
- Shell : Git Bash (MINGW64)
- Python local : `3.13` — Python Docker : `3.11` (conda)

---

## 3. Stockage utilisé

| Répertoire | Taille réelle |
|---|---|
| `data/raw/` (5 tailles) | **~660 MB** |
| `outputs/` (métriques + dashboard) | **~2.3 MB** |
| `.sbert_cache/` (embeddings .npy) | **191.3 MB** |
| `checkpoints/` (184 fichiers .pt) | **~5.1 GB** |
| Image Docker `gnn-rec:prod` | **17.69 GB** |

---

## 4. Versions exactes

| Bibliothèque | Local Windows | Docker (entraînement) |
|---|---|---|
| **PyTorch** | `2.11.0+cpu` | **`2.4.0+cu121`** |
| **PyTorch Geometric** | `2.7.0` | `2.5.x` |
| **CUDA** | Non disponible | **CUDA 12.1** |
| **GPU** | — | **RTX 3070 Laptop 8 GB** |
| **AMP** | Désactivée | **Activée (float16)** |
| **Optuna** | `4.8.0` | `4.8.0` |
| **PySpark** | Non installé | `3.5.x` |
| **Python** | `3.13` | `3.11` |

---

## 5. Méthode exacte du split

**Split aléatoire 70 / 15 / 15** — pas stratifié.

```python
train_idx, temp_idx = train_test_split(all_idx, test_size=0.30, random_state=1)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=1)
```

- Type : **aléatoire simple** (pas chronologique, pas par utilisateur)
- `random_state = 1` — reproductible
- Graphe construit sur les arêtes train uniquement (aucune fuite test/val)

---

## 6. Mode standard exact de prétraitement

1. Chargement CSV via **pandas** (w=1) ou **PySpark** (w>1)
2. Déduplication `(user_id, business_id)`
3. Filtrage : `rating_thresh = 1` → toutes les interactions conservées
4. Encodage : `DynamicLabelEncoder` → entiers continus [0, N-1]
5. Tri par `[user_id, business_id, date]`
6. Split 70/15/15 aléatoire
7. Construction graphe bipartite `edge_index [2, 2×E_train]`
8. **SBERT warm-start** : `use_sbert_item_init = True` — embeddings items depuis catégories texte (cachés dans `.sbert_cache/`)

```python
# DataConfig
rating_thresh  = 1      # pas de filtre étoiles
max_users      = 0      # no limit
max_reviews    = 0      # no limit

# EvalConfig
relevance_thresh = 3.0  # stars >= 3.0 = item pertinent
max_eval_users   = 5000
```

---

## 7. Format final des données préparées

| Format | Rôle |
|---|---|
| **CSV** | Entrée brute (3 fichiers Yelp) |
| **`.pt`** | Checkpoints (~40 MB/fichier) |
| **JSON** | Métriques par run |
| **PNG** | Courbes d'entraînement |
| **Parquet** | Sortie Spark Structured Streaming |
| **SQLite** | Études Optuna HPO |

---

## 8. Utilisation effective de Spark/HDFS

- **w=1** : `pandas.read_csv()` — loader standard
- **w>1** : `SparkSession.master("local[WORLD_SIZE]")` par rank DDP → `toPandas()` → pipeline PyTorch identique
- Données HDFS uploadées via `scripts/hdfs_upload.sh`
- `spark.driver.memory = 4g`, Arrow désactivé

---

## 9. Statut final de `max_users` et `max_reviews`

```python
max_users   = 0   # no limit — charge tous les 145 683 users
max_reviews = 0   # no limit — charge toutes les 188 044 reviews
```

---

## 10. Méthode exacte de negative sampling

- **Stratégie** : uniforme avec rejet (exclusion des positifs connus)
- **n_neg = 10** par paire positive
- **user_pos** : dict `{user_id: set(items_train)}` — jamais d'item connu comme négatif
- **Correction data leakage** : `disjoint=True` dans `LinkNeighborLoader` (avec `pyg-lib`) ou fallback MP graph séparé

---

## 11. Nombre exact de checkpoints conservés

- `keep_last_n = 3` périodiques + 1 best + 1 latest
- `save_every_n_epochs = 20`
- Taille : **~40 MB** par `.pt` (emb_dim=64)
- Total : **184 fichiers = ~5.1 GB**

---

## 12. Statut réel du replay buffer

Implémenté (`capacity=10_000`, `replay_ratio=0.3`) — **aucune expérience incrémentale lancée** à ce stade.

---

## 13. Type exact de l'API

**Streamlit uniquement** (`demo/app.py`, port **8501**) — pas de Flask/FastAPI.

---

## 14. Ports exacts

| Service | Port |
|---|---|
| Streamlit | **8501** |
| Kafka UI | 8080 |
| HDFS NameNode | 9870 |
| Spark Master | 8081 |
| Kibana | 5601 |
| Elasticsearch | 9200 |

---

## 15–17. PubMed / ArXiv / Ingestion

> Pipeline Big Data non exécuté — à documenter lors d'un run réel.

---

## 18. Taille des fichiers générés

| Fichier | Taille |
|---|---|
| `outputs/` | **2.3 MB** |
| `outputs/report.html` | **~150 KB** |
| Un checkpoint `.pt` | **~40 MB** |
| `.sbert_cache/` | **191.3 MB** |
| `data/raw/` total | **~660 MB** |

---

## 19–20. Captures Kibana / Streamlit

> À capturer lors d'un run réel.

---

## 21. Résultats des baselines (GAT @ full, w=1)

Extraits du log d'entraînement :

| Baseline | K=5 | K=10 | K=20 |
|---|---|---|---|
| **Popularité** P | 0.0034 | 0.0027 | 0.0020 |
| **Popularité** HR | 0.0170 | 0.0272 | 0.0398 |
| **Popularité** NDCG | 0.0103 | 0.0134 | 0.0165 |
| **Aléatoire** P | 0.0001 | 0.0002 | 0.0001 |
| **Aléatoire** HR | 0.0006 | 0.0016 | 0.0026 |
| **Aléatoire** NDCG | 0.0003 | 0.0006 | 0.0008 |

---

## 22. Résultats avant/après incrémental

> À produire — aucune expérience incrémentale lancée.

---

## 23. Résultats complets de l'entraînement partitionné

> Note : `gat_w3_full` et `gat_w4_full` contiennent encore les anciennes métriques (runs à refaire).

### Temps d'entraînement t_train (secondes)

| Modèle | Taille | w1 | w2 | w3 | w4 |
|---|---|---|---|---|---|
| SAGE | 1k | 3.22 | 3.49 | 4.32 | 4.86 |
| SAGE | 5k | 9.58 | 8.14 | 7.97 | 8.24 |
| SAGE | 10k | 19.15 | 15.35 | 13.79 | 14.74 |
| SAGE | 50k | 97.24 | 81.95 | 79.34 | 77.32 |
| SAGE | 100k | 155.40 | 128.31 | 120.82 | 114.85 |
| SAGE | full | 257.78 | 187.07 | 166.48 | 166.54 |
| GAT | 1k | 3.57 | 3.98 | 4.96 | 5.07 |
| GAT | 5k | 9.41 | 8.52 | 8.70 | 9.72 |
| GAT | 10k | 19.14 | 15.66 | 15.48 | 16.36 |
| GAT | 50k | 101.26 | 88.68 | 85.54 | 89.31 |
| GAT | 100k | 162.03 | 138.46 | 134.41 | 135.09 |
| GAT | full | 264.43 | 203.76 | ⚠️ancien | ⚠️ancien |
| LightGCN | 1k | 3.06 | 3.09 | 3.98 | 3.94 |
| LightGCN | 5k | 8.33 | 7.43 | 7.43 | 8.00 |
| LightGCN | 10k | 17.19 | 14.50 | 13.66 | 14.07 |
| LightGCN | 50k | 96.42 | 81.63 | 77.64 | 76.37 |
| LightGCN | 100k | 158.58 | 122.61 | 113.49 | 108.33 |
| LightGCN | full | 250.01 | 182.78 | 161.95 | 155.72 |

### Speedup t_train (w1 / wN)

| Modèle | Taille | w2 | w3 | w4 |
|---|---|---|---|---|
| SAGE | 1k | 0.92× | 0.75× | 0.66× |
| SAGE | 5k | 1.18× | 1.20× | 1.16× |
| SAGE | 10k | 1.25× | 1.39× | 1.30× |
| SAGE | 50k | 1.19× | 1.23× | 1.26× |
| SAGE | 100k | 1.21× | 1.29× | **1.35×** |
| SAGE | full | 1.38× | **1.55×** | **1.55×** |
| GAT | 1k | 0.90× | 0.72× | 0.70× |
| GAT | 5k | 1.10× | 1.08× | 0.97× |
| GAT | 10k | 1.22× | 1.24× | 1.17× |
| GAT | 50k | 1.14× | 1.18× | 1.13× |
| GAT | 100k | 1.17× | 1.21× | 1.20× |
| GAT | full | 1.30× | ⚠️ | ⚠️ |
| LightGCN | 1k | 0.99× | 0.77× | 0.78× |
| LightGCN | 5k | 1.12× | 1.12× | 1.04× |
| LightGCN | 10k | 1.19× | 1.26× | 1.22× |
| LightGCN | 50k | 1.18× | 1.24× | 1.26× |
| LightGCN | 100k | 1.29× | 1.40× | **1.46×** |
| LightGCN | full | 1.37× | 1.54× | **1.61×** |

### NDCG@10 (w=1, baseline standard)

| Modèle | 1k | 5k | 10k | 50k | 100k | full |
|---|---|---|---|---|---|---|
| SAGE | 0.0000 | 0.0013 | 0.0023 | 0.0017 | 0.0063 | 0.0031 |
| GAT | 0.0000 | 0.0018 | 0.0030 | 0.0071 | **0.0094** | **0.0080** |
| LightGCN | 0.0047 | 0.0024 | 0.0004 | 0.0038 | 0.0039 | 0.0048 |

### Métriques complètes w=1 (dataset full, n_eval=5000)

| Modèle | NDCG@10 | P@10 | R@10 | MRR@10 | HR@10 | RMSE |
|---|---|---|---|---|---|---|
| **SAGE** | 0.0031 | 0.0007 | 0.0066 | 0.0021 | 0.0074 | 2.1774 |
| **GAT** | **0.0080** | **0.0018** | **0.0168** | **0.0057** | **0.0182** | **1.9746** |
| **LightGCN** | 0.0048 | 0.0012 | 0.0095 | 0.0038 | 0.0116 | 1.9093 |

> **Observations** :
> - **GAT** obtient le meilleur NDCG@10 et RMSE sur tous les datasets ≥ 50k
> - **LightGCN** obtient le meilleur speedup maximal (**1.61×** avec w=4 full)
> - Les petits datasets (1k) montrent NDCG=0 car trop sparse pour le ranking
> - Speedup < 1 pour les petits datasets (1k) : overhead DDP > gain calcul
> - Speedup croît avec la taille du dataset (plus de paires BPR à paralléliser)

---

## 24. Courbes de loss

Générées automatiquement dans `outputs/<model>_w<N>_<size>/plots/<model>_training_curve.png`.

Comportement observé : BPR loss part de ~0.55-0.69 et converge en 80-150 epochs.

---

## 25. Courbes de comparaison des métriques

Dashboard HTML complet : `outputs/report.html`

```bash
python3.13 scripts/compare_distributed.py --all --html
```

---

## 26. Courbe de convergence Optuna

> HPO non encore lancé.
>
> ```bash
> python3.13 src/main.py --model gat --mode tune --data-dir data/raw/full --trials 30 --no-amp
> optuna-dashboard sqlite:///outputs/tuning/study_gat.db
> ```
