# GNN Recommender — Référence des commandes

> **Répertoire de travail** : toujours depuis la racine `gnn_recommender/`  
> **Python local** : `python3.13`  
> **Entraînement** : Docker (`gnn-rec:prod`) avec GPU RTX 3070

---

## 1. Prérequis

```powershell
# Vérifier l'environnement local
python3.13 -c "import torch; print(torch.__version__)"
python3.13 -c "import torch_geometric; print('PyG OK')"
python3.13 -c "import streamlit; print('Streamlit OK')"

# Construire l'image Docker (une seule fois, ~10 min)
docker build -t gnn-rec:prod .
```

---

## 2. Datasets disponibles

| Dossier | Reviews | Users | Items | Usage |
|---------|---------|-------|-------|-------|
| `data/raw/1k` | 1 000 | 783 | ~500 | Test rapide |
| `data/raw/5k` | 5 000 | 3 870 | ~2k | Debug |
| `data/raw/10k` | 10 000 | 7 738 | ~4k | Test |
| `data/raw/50k` | 50 000 | 38 883 | ~9k | Entraînement |
| `data/raw/100k` | 100 000 | 77 640 | 11 813 | **Comparaison principale** |
| `data/raw/full` | **188 044** | **145 683** | **11 890** | **Dataset complet** |

---

## 3. Entraînement Docker — règles VRAM

| Condition | VRAM/worker |
|---|---|
| `SIZE=full` + `WORLD_SIZE=1` | **4 GB** |
| `MODEL_TYPE=gat` + `WORLD_SIZE=1` + `SIZE=100k` | **2.5 GB** |
| Tout le reste | **2 GB** |

> Toujours passer les 4 variables ensemble : `WORLD_SIZE` + `MODEL_TYPE` + `DATA_DIR` + `SIZE`

---

## 4. Commandes d'entraînement — tous les runs

### SAGE

```bash
# 1k
WORLD_SIZE=1 MODEL_TYPE=sage DATA_DIR=data/raw/1k SIZE=1k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=sage DATA_DIR=data/raw/1k SIZE=1k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=sage DATA_DIR=data/raw/1k SIZE=1k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=sage DATA_DIR=data/raw/1k SIZE=1k docker compose -f docker/docker-compose.yml up

# 5k
WORLD_SIZE=1 MODEL_TYPE=sage DATA_DIR=data/raw/5k SIZE=5k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=sage DATA_DIR=data/raw/5k SIZE=5k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=sage DATA_DIR=data/raw/5k SIZE=5k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=sage DATA_DIR=data/raw/5k SIZE=5k docker compose -f docker/docker-compose.yml up

# 10k
WORLD_SIZE=1 MODEL_TYPE=sage DATA_DIR=data/raw/10k SIZE=10k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=sage DATA_DIR=data/raw/10k SIZE=10k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=sage DATA_DIR=data/raw/10k SIZE=10k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=sage DATA_DIR=data/raw/10k SIZE=10k docker compose -f docker/docker-compose.yml up

# 50k
WORLD_SIZE=1 MODEL_TYPE=sage DATA_DIR=data/raw/50k SIZE=50k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=sage DATA_DIR=data/raw/50k SIZE=50k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=sage DATA_DIR=data/raw/50k SIZE=50k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=sage DATA_DIR=data/raw/50k SIZE=50k docker compose -f docker/docker-compose.yml up

# 100k
WORLD_SIZE=1 MODEL_TYPE=sage DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=sage DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=sage DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=sage DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up

# full
WORLD_SIZE=1 MODEL_TYPE=sage DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=sage DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=sage DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=sage DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
```

### GAT

```bash
# 1k → 5k → 10k → 50k (même pattern que SAGE)
WORLD_SIZE=1 MODEL_TYPE=gat DATA_DIR=data/raw/1k SIZE=1k docker compose -f docker/docker-compose.yml up
# ... (idem pour 5k, 10k, 50k)

# 100k — w=1 nécessite 2.5 GB (détecté automatiquement via SIZE=100k)
WORLD_SIZE=1 MODEL_TYPE=gat DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=gat DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=gat DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=gat DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up

# full — w=1 nécessite 4 GB (détecté automatiquement via SIZE=full)
WORLD_SIZE=1 MODEL_TYPE=gat DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=gat DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=gat DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=gat DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
```

### LightGCN

```bash
# Même pattern que SAGE — 2 GB pour tous les runs
WORLD_SIZE=1 MODEL_TYPE=lightgcn DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=lightgcn DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=lightgcn DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=lightgcn DATA_DIR=data/raw/100k SIZE=100k docker compose -f docker/docker-compose.yml up

WORLD_SIZE=1 MODEL_TYPE=lightgcn DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
WORLD_SIZE=2 MODEL_TYPE=lightgcn DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
WORLD_SIZE=3 MODEL_TYPE=lightgcn DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
WORLD_SIZE=4 MODEL_TYPE=lightgcn DATA_DIR=data/raw/full SIZE=full docker compose -f docker/docker-compose.yml up
```

---

## 5. Dashboard résultats

```bash
# Génère outputs/report.html et l'ouvre dans le navigateur
python3.13 scripts/compare_distributed.py --all --html

# Sans ouvrir le navigateur
python3.13 scripts/compare_distributed.py --all --html --no-browser

# Un modèle / une taille
python3.13 scripts/compare_distributed.py --model gat --size full --html
```

---

## 6. Pipeline Big Data

```bash
# Démarrer infrastructure
cd pipeline && docker compose up -d && cd ..

# Upload données HDFS (Git Bash — désactiver conversion chemins)
export MSYS_NO_PATHCONV=1
bash scripts/hdfs_upload.sh --size 100k
bash scripts/hdfs_upload.sh --size full

# Producteurs
python3.13 pipeline/producers/pubmed_producer.py   # Terminal 1
python3.13 pipeline/producers/arxiv_producer.py    # Terminal 2

# Consumer Spark
cd pipeline
.\run_consumer.ps1           # HDFS (défaut)
.\run_consumer.ps1 -Console  # Affichage console
.\run_consumer.ps1 -Status   # Logs
.\run_consumer.ps1 -Stop     # Arrêt

# Vérifier HDFS
docker exec namenode hdfs dfs -ls /data/
docker exec namenode hdfs dfsck /user/spark/streaming -files -blocks -locations
docker exec namenode hdfs dfsadmin -safemode leave
```

Services disponibles :

| Service | URL |
|---------|-----|
| Kafka UI | http://localhost:8080 |
| HDFS NameNode | http://localhost:9870 |
| Spark Master | http://localhost:8081 |
| Kibana | http://localhost:5601 |
| Elasticsearch | http://localhost:9200 |

---

## 7. Interface Streamlit

```powershell
.\run_app.ps1                    # Lance + ouvre http://localhost:8501
.\run_app.ps1 -Port 8502         # Port alternatif
.\run_app.ps1 -Stop              # Arrêt propre
python3.13 -m streamlit run demo/app.py  # Manuel
```

---

## 8. Apprentissage incrémental

```bash
# Générer dataset de test (3 fichiers Yelp)
python3.13 generate_incremental_dataset.py

# Via CLI
python3.13 src/main.py \
    --model sage --mode incremental \
    --ckpt checkpoints/sage_w1_full/sage_best.pt \
    --new-data data/incremental/incremental_reviews.csv \
    --ckpt-dir checkpoints/sage_incremental \
    --no-amp --finetune-epochs 30 --finetune-lr-scale 0.1 --replay-ratio 0.3
```

---

## 9. Optimisation Optuna (HPO)

```bash
python3.13 src/main.py --model gat --mode tune \
    --data-dir data/raw/full --trials 30 --no-amp

# Visualiser
optuna-dashboard sqlite:///outputs/tuning/study_gat.db
```

---

## 10. Tests

```bash
python3.13 -m pytest tests/ -v
python3.13 -m pytest tests/ -v -k "metrics"
python3.13 -m pytest tests/ -v -k "checkpoint"
```

---

## 11. Gestion des checkpoints

```powershell
# Lister les checkpoints
Get-ChildItem checkpoints\ -Recurse -Filter "*.pt" |
  Select Name, LastWriteTime, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
```

---

## 12. Nettoyage Docker

```bash
docker system prune -f          # Build cache + containers arrêtés
docker system prune -a -f       # Tout (requier rebuild de l'image)
docker builder prune -f         # Build cache uniquement
```

---

## 13. Dépannage

| Erreur | Cause | Solution |
|--------|-------|----------|
| `ModuleNotFoundError: torch` | Mauvais Python | Utiliser `python3.13` |
| `OOM` full_batch | VRAM insuffisante | Variables SIZE/MODEL_TYPE correctement définies |
| `No FileSystem for scheme C` | Git Bash conversion chemins | `export MSYS_NO_PATHCONV=1` |
| `OutOfMemoryError Java heap` | Spark driver trop petit | `spark.driver.memory=4g` (déjà configuré) |
| `BPR Loss: 0.0002` dès epoch 1 | Data leakage neighbor_loader | `disjoint=True` (requiert pyg-lib) |
| `N/A` dans Streamlit | Cache périmé | Redémarrer `.\run_app.ps1` |
| `HDFS Safe Mode` | Datanodes reconnectés | `docker exec namenode hdfs dfsadmin -safemode leave` |
