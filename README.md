# GNN Recommender System

Système de recommandation basé sur des **Graph Neural Networks** appliqué aux données de santé Yelp.  
Supporte **GraphSAGE**, **GAT** et **LightGCN** avec entraînement scratch, apprentissage incrémental, tuning Optuna et interface Streamlit.

---

## Modèles supportés

| Modèle | Architecture | NDCG@10 (full, w=1) | RMSE |
|--------|-------------|---------------------|------|
| **GraphSAGE** | 2 couches SAGEConv, agrégation mean | 0.0031 | 2.1774 |
| **GAT** | GATConv multi-head (4 têtes) | **0.0080** | **1.9746** |
| **LightGCN** | GCN simplifié, filtrage collaboratif | 0.0048 | 1.9093 |

---

## Structure du projet

```
gnn_recommender/
├── src/                          # Code source
│   ├── main.py                   # Point d'entrée CLI
│   ├── config.py                 # Configuration (dataclasses)
│   ├── models/                   # GraphSAGE, GAT, LightGCN
│   ├── data/                     # Loader, preprocessing, graph builder
│   ├── training/                 # Trainer, BPR loss, DDP, incrémental
│   ├── evaluation/               # Métriques ranking
│   ├── tuning/                   # HPO Optuna
│   └── utils/                    # Checkpoint, hardware, plots
├── demo/
│   ├── app.py                    # Interface Streamlit (port 8501)
│   └── inference.py
├── pipeline/                     # Kafka + Spark + HDFS + ELK
├── docker/
│   └── docker-compose.yml        # Entraînement distribué
├── scripts/
│   ├── compare_distributed.py    # Dashboard HTML résultats
│   └── hdfs_upload.sh            # Upload données HDFS
├── data/raw/                     # 1k, 5k, 10k, 50k, 100k, full
├── Dockerfile                    # Image GPU (PyTorch 2.4.0+cu121)
├── COMMANDS.md                   # Référence complète des commandes
├── DOCUMENTATION_FR.md           # Documentation technique
└── CHAPTER4_QUESTIONS.md         # Réponses chapitre 4
```

---

## Environnement

| | Local (Windows) | Docker (entraînement) |
|---|---|---|
| Python | 3.13 | 3.11 |
| PyTorch | 2.11.0+cpu | **2.4.0+cu121** |
| GPU | — | **RTX 3070 Laptop 8 GB** |
| CUDA | — | 12.1 |

---

## Installation rapide

```powershell
# Vérifier l'environnement local
python3.13 -c "import torch; print(torch.__version__)"

# Construire l'image Docker (une seule fois)
docker build -t gnn-rec:prod .
```

---

## Datasets

| Taille | Reviews | Users | Items |
|--------|---------|-------|-------|
| `1k` | 1 000 | 783 | ~500 |
| `5k` | 5 000 | 3 870 | ~2k |
| `10k` | 10 000 | 7 738 | ~4k |
| `50k` | 50 000 | 38 883 | ~9k |
| `100k` | 100 000 | 77 640 | 11 813 |
| `full` | **188 044** | **145 683** | **11 890** |

Format : 3 fichiers CSV Yelp Health & Medical par taille (`reviews`, `users`, `business`).

---

## Lancer l'entraînement

```bash
# Standard (pandas, w=1)
WORLD_SIZE=1 MODEL_TYPE=sage DATA_DIR=data/raw/100k SIZE=100k \
  docker compose -f docker/docker-compose.yml up

# Bigdata (Spark+HDFS, w=4)
WORLD_SIZE=4 MODEL_TYPE=sage DATA_DIR=data/raw/100k SIZE=100k \
  docker compose -f docker/docker-compose.yml up

# Dataset full (full_batch, w=1)
WORLD_SIZE=1 MODEL_TYPE=gat DATA_DIR=data/raw/full SIZE=full \
  docker compose -f docker/docker-compose.yml up
```

> Variables obligatoires : `WORLD_SIZE` + `MODEL_TYPE` + `DATA_DIR` + `SIZE`

---

## Résultats — Speedup t_train

| Modèle | Taille | w2 | w3 | w4 |
|--------|--------|----|----|-----|
| SAGE | full | 1.38× | 1.55× | **1.55×** |
| GAT | full | 1.30× | — | — |
| LightGCN | full | 1.37× | 1.54× | **1.61×** |
| LightGCN | 100k | 1.29× | 1.40× | **1.46×** |

> Petits datasets (1k) : speedup < 1 — overhead DDP > gain calcul.

---

## VRAM par worker

| Condition | VRAM |
|---|---|
| `SIZE=full` + `w=1` | 4 GB |
| `GAT` + `w=1` + `SIZE=100k` | 2.5 GB |
| Tout le reste | **2 GB** |

---

## Dashboard résultats

```bash
python3.13 scripts/compare_distributed.py --all --html
# Ouvre outputs/report.html dans le navigateur
```

---

## Interface Streamlit

```powershell
.\run_app.ps1
# → http://localhost:8501
```

3 onglets : Recommandations · Nouvel Utilisateur (cold-start) · Apprentissage Incrémental

---

## Pipeline Big Data (optionnel)

```bash
cd pipeline && docker compose up -d
python3.13 pipeline/producers/pubmed_producer.py   # Terminal 1
python3.13 pipeline/producers/arxiv_producer.py    # Terminal 2
.\pipeline\run_consumer.ps1                         # Spark consumer
```

Services : Kafka UI (8080) · HDFS (9870) · Spark (8081) · Kibana (5601)

---

## Tests

```bash
python3.13 -m pytest tests/ -v
```

---

## Documentation

| Fichier | Contenu |
|---|---|
| `COMMANDS.md` | Toutes les commandes (entraînement, Docker, HDFS) |
| `DOCUMENTATION_FR.md` | Architecture, modèles, pipeline |
| `CHAPTER4_QUESTIONS.md` | Réponses détaillées chapitre 4 avec métriques réelles |

---

## Références

- **GraphSAGE** : Hamilton et al., NeurIPS 2017
- **GAT** : Veličković et al., ICLR 2018
- **LightGCN** : He et al., SIGIR 2020
- **BPR** : Rendle et al., UAI 2009
