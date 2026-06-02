# Documentation technique — GNN Recommender System

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du système](#2-architecture-du-système)
3. [Modèles GNN](#3-modèles-gnn)
4. [Pipeline de données](#4-pipeline-de-données)
5. [Entraînement](#5-entraînement)
6. [Apprentissage incrémental](#6-apprentissage-incrémental)
7. [Optimisation des hyperparamètres (HPO)](#7-optimisation-des-hyperparamètres-hpo)
8. [Évaluation et métriques](#8-évaluation-et-métriques)
9. [Interface Streamlit](#9-interface-streamlit)
10. [Pipeline Big Data](#10-pipeline-big-data)
11. [Configuration](#11-configuration)
12. [Checkpoints](#12-checkpoints)
13. [Benchmark](#13-benchmark)
14. [Tests](#14-tests)
15. [Dépannage](#15-dépannage)

---

## 1. Vue d'ensemble

Ce système de recommandation applique des **Graph Neural Networks** à des données de santé issues de la plateforme Yelp. Il modélise les interactions utilisateur-item comme un **graphe bipartite** où les nœuds sont les utilisateurs et les businesses, et les arêtes représentent les reviews.

### Résultats expérimentaux (dataset full, 188 044 reviews)

| Modèle | NDCG@10 | RMSE | Speedup max |
|---|---|---|---|
| GraphSAGE | 0.0031 | 2.1774 | 1.55× (w=3,4) |
| **GAT** | **0.0080** | **1.9746** | 1.30× (w=2) |
| LightGCN | 0.0048 | 1.9093 | **1.61×** (w=4) |

### Caractéristiques principales

- **3 architectures GNN** : GraphSAGE, GAT, LightGCN
- **Apprentissage incrémental** : fine-tuning sans réentraînement complet, avec replay buffer anti-oubli
- **HPO composite** : critère eq. 3.21 — `0.4×NDCG@K + 0.3×Precision@K + 0.3×Recall@K`
- **Interface Streamlit** avec upload multi-fichiers (format Yelp : reviews + users + business)
- **Pipeline Big Data** : Kafka + Spark + HDFS (2 datanodes) + ELK
- **Entraînement distribué** : DDP via torchrun jusqu'à 4 GPUs

---

## 2. Architecture du système

### Graphe bipartite user-item

```
Utilisateurs (noeuds 0..N_users-1)
     |  |  |
     |  |  +---- review (stars >= rating_thresh) ----> Business i
     |  +-------- review ----------------------------------------> Business j
     +------------ review ----------------------------------------> Business k
```

Les embeddings `emb_u` et `emb_i` (vecteurs de dimension d) sont appris conjointement.
Score de recommandation : `score(u, i) = emb_u . emb_i` (produit scalaire)

### Composants principaux

```
src/
+-- data/
|   +-- loader.py              # Chargement CSV Yelp (3 fichiers)
|   +-- preprocessing.py       # DynamicLabelEncoder (encodeurs extensibles)
|   +-- graph_builder.py       # Construction edge_index PyTorch Geometric
|   +-- samplers.py            # Mini-batch / full-batch PyG Data
|   +-- replay_buffer.py       # Stockage interactions historiques
|
+-- models/
|   +-- graphsage.py           # 2 couches SAGEConv + residus
|   +-- gat.py                 # GATConv multi-tetes
|   +-- lightgcn.py            # LGConv empilees (sans transformation non-lineaire)
|
+-- training/
|   +-- trainer.py             # Boucle principale + BPR + early stopping
|   +-- loss.py                # BPR loss avec echantillonnage negatif
|   +-- incremental.py         # Extension embeddings + fine-tuning
|   +-- incremental_hpo.py     # HPO incremental (Optuna, critere composite)
|   +-- distributed.py         # Wrapper DDP torchrun
|
+-- evaluation/
|   +-- metrics.py             # compute_ranking_metrics(), baselines
|
+-- tuning/
    +-- optuna_tuner.py        # HPO scratch (critere composite)
```

---

## 3. Modèles GNN

### 3.1 GraphSAGE

**Principe :** aggregation inductif des voisins par moyenne (SampleAndAGGregatE).

```
h_v^(k) = sigma( W^k . CONCAT(h_v^(k-1), MEAN({h_u : u in N(v)})) )
```

- 2 couches SAGEConv
- Connexions residuelles si `use_residual=True`
- Dropout entre les couches

**Usage recommande :** baseline robuste, usage general, bonne generalisation sur de nouvelles entites.

### 3.2 GAT (Graph Attention Network)

**Principe :** ponderation adaptative des voisins par mecanisme d'attention multi-tetes.

```
h_v^(k) = sigma( sum_{u in N(v)} alpha_vu . W^k . h_u^(k-1) )

alpha_vu = softmax( LeakyReLU( a^T [Wh_v || Wh_u] ) )
```

- Nombre de tetes configurable (`gat_heads=4` par defaut)
- Interpretabilite : les poids d'attention revelent les voisins influents

**Usage recommande :** quand la ponderation differenciee des voisins est pertinente.

### 3.3 LightGCN

**Principe :** GCN simplifie — supprime la transformation non-lineaire, conserve uniquement la propagation de signal.

```
e_u^(k) = sum_{i in N(u)} (1/sqrt(|N(u)|*|N(i)|)) * e_i^(k-1)

e_u_final = (1/(K+1)) * sum_{k=0}^{K} e_u^(k)   (moyenne des couches)
```

**Usage recommande :** filtrage collaboratif pur, entrainement le plus rapide.

### 3.4 Comparaison

| | GraphSAGE | GAT | LightGCN |
|--|-----------|-----|----------|
| **Complexite** | Moyenne | Haute | Faible |
| **Vitesse** | Rapide | Lente | Tres rapide |
| **Nouvelles entites** | Excellent | Bon | Bon |
| **Interpretabilite** | Faible | Haute | Moyenne |
| **Parametres** | emb_dim, n_layers, dropout | emb_dim, gat_heads, dropout | emb_dim, n_layers |

---

## 4. Pipeline de données

### 4.1 Format des fichiers

**Reviews** (obligatoire pour l'entrainement) :
```
user_id, review_id, business_id, stars, date, text, useful, funny, cool
```

**Users** :
```
user_id, name, review_count, yelping_since, average_stars, fans, useful, funny, cool
```

**Business** :
```
business_id, name, address, city, state, postal_code, stars, review_count, is_open, categories
```

### 4.2 Pretraitement

`DynamicLabelEncoder` (dans `preprocessing.py`) :
- Encode les identifiants string vers entiers
- **Extensible** : la methode `extend(new_ids)` ajoute de nouveaux IDs sans reencoder les anciens
- Utilise pour les users et les items

Construction du graphe (`graph_builder.py`) :
- Noeuds : `[0, N_users-1]` = users, `[N_users, N_users+N_items-1]` = items
- Aretes : review avec `stars >= rating_thresh` (defaut : 3.0)
- `edge_index` de shape `[2, 2*E]` (non-oriente, aretes dans les deux sens)

### 4.3 Generation de donnees synthetiques

Le script `generate_incremental_dataset.py` genere 3 fichiers au format Yelp :

```
data/incremental/
  incremental_reviews.csv    # user_id, review_id, business_id, stars, date, text, ...
  incremental_users.csv      # user_id, name, review_count, yelping_since, average_stars, ...
  incremental_business.csv   # business_id, name, address, city, state, postal_code, stars, ...
```

Caracteristiques du dataset genere :
- 80 users (60 existants + 20 nouveaux), 50 businesses (38 existants + 12 nouveaux)
- 1 356 reviews, sparsite 66%, note moyenne 3.89
- 5 groupes de preferences coherents
- Modele de notation : `mu = 2.5 + 2.5 * group_match`, sigma=0.9

---

## 5. Entraînement

### 5.1 BPR Loss (Bayesian Personalized Ranking)

```
L_BPR = -sum_{(u,i,j)} log( sigma(score(u,i) - score(u,j)) ) + lambda * ||Theta||^2

ou :
  i = item positif (stars >= threshold)
  j = item negatif (echantillonne aleatoirement)
  lambda = reg_lambda (regularisation L2)
```

### 5.2 Boucle d'entraînement

```
Pour chaque epoch :
  1. Propagation GNN : embs = model(edge_index)
  2. Calcul BPR loss avec n_neg negatifs par positif
  3. Retropropagation + gradient clipping
  4. LR scheduler (cosine ou plateau)
  5. Validation tous les eval_every epochs
  6. Sauvegarde si meilleur score validation (critere composite)
  7. Early stopping si pas d'amelioration depuis patience epochs
```

### 5.3 Mixed Precision (AMP)

- Active automatiquement sur GPU (`device.type == "cuda"`)
- Desactive avec `--no-amp` (CPU, debug)
- Utilise `torch.cuda.amp.GradScaler` pour la stabilite numerique

### 5.4 Detection automatique du materiel

| Tier | Condition | emb_dim | epochs | AMP |
|------|-----------|---------|--------|-----|
| `debug` | `--debug` | 16 | 3 | off |
| `cpu` | Pas de GPU | 64 | 100 | off |
| `single_gpu` | 1 GPU | 128 | 200 | on |
| `multi_gpu` | 2+ GPUs | 128 | 200 | on |

### 5.5 Entraînement distribué — Docker + Spark + HDFS

Le système compare deux environnements :

| Mode | `WORLD_SIZE` | Loader | Backend DDP |
|------|-------------|--------|-------------|
| Standard | 1 | pandas CSV local | — |
| Bigdata | 2, 3, 4 | PySpark local[N] + HDFS | gloo (simulation multi-process) |

**Architecture bigdata :**
```
HDFS (namenode + datanode1 + datanode2)
    │
    └── Spark local[N]  ←── chaque DDP rank lance son propre SparkSession
            │
            └── toPandas()  →  pipeline PyTorch identique au mode standard
```

**Démarrage complet :**

```bash
# 1. Infrastructure HDFS + Spark
cd pipeline && docker compose up -d namenode datanode1 datanode2 spark-master spark-worker && cd ..

# 2. Upload données sur HDFS (Git Bash : désactiver conversion chemins)
export MSYS_NO_PATHCONV=1
bash scripts/hdfs_upload.sh --size 100k

# 3. Entraînement (toujours les 4 variables ensemble)
WORLD_SIZE=1 MODEL_TYPE=sage DATA_DIR=data/raw/100k SIZE=100k \
  docker compose -f docker/docker-compose.yml up

WORLD_SIZE=4 MODEL_TYPE=sage DATA_DIR=data/raw/100k SIZE=100k \
  docker compose -f docker/docker-compose.yml up
```

**Métriques comparées :**
- `t_load` : temps de chargement (pandas ≪ Spark dû à l'overhead JVM)
- `t_train` : temps d'entraînement (diminue avec world_size grâce à DDP)
- `t_sbert` : quasi-nul après le premier run (cache `.sbert_cache/`)
- `NDCG@K`, `Precision@K`, `Recall@K` : qualité des recommandations

- Sauvegarde uniquement sur `rank=0`

---

## 6. Apprentissage incrémental

### 6.1 Principe

L'apprentissage incremental permet d'integrer de nouvelles donnees sans reentrainer le modele depuis zero.

**Les 3 composants geres automatiquement :**

| Composant | Graphe | Modele |
|-----------|--------|--------|
| Nouveaux users | Nouveaux noeuds user | Lignes ajoutees dans emb_table (Xavier init) |
| Nouveaux items | Nouveaux noeuds item | Lignes ajoutees dans emb_table (Xavier init) |
| Nouvelles interactions | Nouvelles aretes user-item | Poids affines par fine-tuning BPR |

### 6.2 Extension des embeddings

```python
# Dans incremental.py -- extend_model_embeddings()
old_weight = model.emb.weight.data          # [N_old, emb_dim]
new_rows = torch.empty(n_new, emb_dim)
torch.nn.init.xavier_uniform_(new_rows)     # Init Xavier pour nouveaux noeuds
new_weight = torch.cat([old_weight, new_rows], dim=0)
model.emb = nn.Embedding(N_old + n_new, emb_dim)
model.emb.weight.data = new_weight          # Anciens embeddings preserves
```

### 6.3 Replay buffer

Le `ReplayBuffer` (capacite 10 000 par defaut) stocke les interactions passees pour eviter l'**oubli catastrophique** :

```
Fine-tune set = nouvelles interactions + replay_ratio * anciennes interactions

Exemple (replay_ratio=0.3, 1000 nouvelles interactions) :
  n_replay = 1000 * 0.3 / 0.7 = 429 anciennes interactions
  Dataset total : ~1 429 interactions
```

### 6.4 Configuration fine-tuning

```
finetune_epochs   : 20-50 epochs (vs 200 pour scratch)
finetune_lr_scale : 0.1   -> LR = base_lr * 0.1
replay_ratio      : 0.3   -> 30% d'anciennes interactions
warmup_epochs     : max(1, finetune_epochs // 10)
```

### 6.5 HPO incremental (Optuna)

La fonction `tune_incremental_hparams()` dans `incremental_hpo.py` :

**Espace de recherche :**
```python
finetune_epochs   : [5, 10, 20, 30, 50]
finetune_lr_scale : [0.01, 0.05, 0.1, 0.2]
replay_ratio      : [0.0, 0.1, 0.2, 0.3, 0.5]
```

**Objectif maximise (eq. 3.21) :**
```
Score = alpha * Composite(nouvelles_donnees) + (1-alpha) * Composite(anciennes_donnees)

Composite(data) = 0.4 * NDCG@10 + 0.3 * P@10 + 0.3 * R@10

alpha = 0.6  (favorise les nouvelles donnees, penalise l'oubli)
```

Les poids `ndcg_w`, `prec_w`, `rec_w` et `alpha` sont parametrables depuis `tune_incremental_hparams()`.

---

## 7. Optimisation des hyperparamètres (HPO)

### 7.1 Critere composite (eq. 3.21)

Les deux modules HPO (scratch et incremental) utilisent le meme critere :

```
Score = ndcg_w * NDCG@K + prec_w * P@K + rec_w * R@K
      = 0.4 * NDCG@K + 0.3 * Precision@K + 0.3 * Recall@K
```

Configurable dans `TuneConfig` (`config.py`) :
```python
ndcg_w = 0.4   # Poids NDCG
prec_w = 0.3   # Poids Precision
rec_w  = 0.3   # Poids Recall
```

### 7.2 Espaces de recherche

**GraphSAGE :** emb_dim [32, 64, 128], n_layers [1, 2], dropout [0, 0.4], lr [1e-4, 1e-2], reg_lambda [1e-6, 1e-3]

**GAT :** emb_dim [64, 128], gat_heads [2, 4], dropout [0, 0.4], lr [1e-4, 5e-3], reg_lambda [1e-6, 1e-3]

**LightGCN :** emb_dim [64, 128, 256], n_layers [1, 2, 3], lr [1e-4, 1e-2], reg_lambda [1e-6, 1e-3]

### 7.3 Sampler et pruner

- **Sampler :** `TPESampler(seed=cfg.seed)` — Tree-structured Parzen Estimator (bayesien)
- **Pruner :** `MedianPruner(n_startup_trials=3)` — elagage des trials sous-performants
- **Stockage :** SQLite `outputs/tuning/study_{model}.db` (resumable a tout moment)

---

## 8. Évaluation et métriques

### 8.1 Metriques de ranking

Pour chaque utilisateur avec au moins 1 item positif dans le test :

| Metrique | Description |
|----------|-------------|
| **Precision@K** | Fraction des top-K qui sont pertinents : hits(K) / K |
| **Recall@K** | Fraction des positifs retrouves : hits(K) / \|pos\| |
| **F1@K** | Moyenne harmonique P et R |
| **NDCG@K** | Precision avec ponderation positionnelle (log2) |
| **MAP@K** | Precision moyenne cumulee sur les rangs des hits |
| **MRR@K** | Reciprocal Rank du premier hit |
| **HR@K** | Hit Rate : 1 si au moins un hit dans top-K |

### 8.2 NDCG@K

```
DCG@K  = sum_{r=1}^{K} hit_r / log2(r+1)
IDCG@K = sum_{r=1}^{min(|pos|,K)} 1 / log2(r+1)
NDCG@K = DCG@K / IDCG@K
```

### 8.3 Critere composite de selection (eq. 3.21)

Utilise pour selectionner le meilleur checkpoint et optimiser le HPO :
```
val_score = 0.4 * NDCG@10 + 0.3 * P@10 + 0.3 * R@10
```

### 8.4 Baselines

- **Popularite :** recommande les items les plus vus dans le train
- **Aleatoire :** selection uniforme parmi tous les items

---

## 9. Interface Streamlit

### 9.1 Onglet 1 — Recommandations

- Selection d'un utilisateur parmi les 200 premiers connus du modele
- Affichage historique (noms businesses + notes en etoiles)
- Top-K avec graphique Plotly (barres horizontales, score colore)
- Items deja vus exclus automatiquement

### 9.2 Onglet 2 — Nouvel Utilisateur (Cold-Start)

- Construction d'un profil par panier multi-categories (max 10 items)
- Profil = `MEAN(emb_i pour i dans panier)`
- Score = `profil . emb_j` pour chaque item candidat
- Pas de fine-tuning requis — inference pure

### 9.3 Onglet 3 — Apprentissage Incremental

**Mode Upload (3 fichiers Yelp) :**
```
Uploader 1 : Reviews  (obligatoire) -- user_id, business_id, stars, date
Uploader 2 : Users    (optionnel)   -- enrichit les noms des nouveaux users
Uploader 3 : Business (optionnel)   -- enrichit les noms dans les recs
```

Analyse croisee :
- Detection automatique nouveaux users / nouveaux items inconnus du modele
- 6 metriques : reviews, users uniques, nouveaux users, items, nouveaux items, fichiers uploades
- Apercu en 3 onglets (Reviews + histogramme, Users, Business)
- Noms affiches depuis les fichiers optionnels

Apres fine-tuning :
- Les nouveaux businesses sont merges dans `biz_df` pour afficher les noms dans les recs APRES
- Comparaison avant/apres : nouvelles recs apparues, recs conservees

**Mode Manuel :** saisie de 1 a 20 interactions avec selection d'items depuis le catalogue.

**Parametres d'entrainement :**
- HPO auto (Optuna 5-30 trials) avec critere composite eq. 3.21
- Ou manual : epochs, LR scale, replay ratio

### 9.4 Lancement

```powershell
.\run_app.ps1              # Lance + ouvre le navigateur
.\run_app.ps1 -Port 8502   # Port alternatif
.\run_app.ps1 -Stop        # Arret propre
```

---

## 10. Pipeline Big Data — Architecture complète

### 10.1 Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COUCHE INGESTION                                │
│                                                                     │
│  PubMed API (NCBI Entrez)                                           │
│       │  pubmed_producer.py                                         │
│       │  → topic: pubmed-articles                                   │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                        │
│  │              KAFKA BROKER               │  port 9092             │
│  │   topic: pubmed-articles (partition=1)  │                        │
│  │   topic: arxiv-papers    (partition=1)  │                        │
│  └──────────────────┬──────────────────────┘                        │
│                     │ consumer group                                │
│  ArXiv API          │                                               │
│       │  arxiv_producer.py                 ┌─────────────────────┐  │
│       │  → topic: arxiv-papers             │    LOGSTASH          │  │
│       └────────────────────────────────────► port 5044           │  │
│                                            └────────┬────────────┘  │
└─────────────────────────────────────────────────────┼───────────────┘
                                                       │
┌─────────────────────────────────────────────────────┼───────────────┐
│                     COUCHE TRAITEMENT                │               │
│                                                      ▼               │
│  ┌───────────────────────────────────┐  ┌─────────────────────────┐ │
│  │   SPARK STRUCTURED STREAMING      │  │   ELASTICSEARCH          │ │
│  │   spark_consumer.py               │  │   port 9200              │ │
│  │   - Lit les 2 topics Kafka        │  │   Index: articles-*      │ │
│  │   - Schema StructType (PySpark 3.5│  └──────────┬──────────────┘ │
│  │   - Batch interval: 30s           │             │                │
│  │   - Ecriture Parquet partitionne  │  ┌──────────▼──────────────┐ │
│  └──────────────┬────────────────────┘  │   KIBANA                 │ │
│                 │                       │   port 5601              │ │
└─────────────────┼───────────────────────┴──────────────────────────┘─┘
                  │
┌─────────────────┼──────────────────────────────────────────────────────┐
│                 ▼     COUCHE STOCKAGE — HADOOP HDFS                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      NAMENODE (port 9870/9000)                   │  │
│  │   - Gere le namespace HDFS (metadonnees)                         │  │
│  │   - Repertoire: /data/{size}/  et  /user/spark/streaming/        │  │
│  │   - Replication factor = 2                                        │  │
│  │   - Block size = 128 MB                                           │  │
│  └────────────────────┬─────────────────────────────────────────────┘  │
│                        │  replique chaque bloc sur 2 datanodes          │
│            ┌───────────┴───────────┐                                    │
│            ▼                       ▼                                    │
│  ┌──────────────────┐   ┌──────────────────┐                           │
│  │   DATANODE 1     │   │   DATANODE 2     │                           │
│  │   hadoop_dn1     │   │   hadoop_dn2     │                           │
│  │   (volume Docker)│   │   (volume Docker)│                           │
│  │                  │   │                  │                           │
│  │  /data/100k/*.csv│   │  /data/100k/*.csv│  ← meme bloc              │
│  │  /data/full/*.csv│   │  /data/full/*.csv│    sur les 2 nodes         │
│  │  Parquet batch1  │   │  Parquet batch1  │                           │
│  └──────────────────┘   └──────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 10.2 Comparaison Standard vs Bigdata

#### Couche de chargement des données Yelp

| Aspect | Standard (w=1, pandas) | Bigdata (w>1, Spark+HDFS) |
|--------|------------------------|---------------------------|
| **Loader** | `pandas.read_csv()` | `SparkSession.read.csv()` |
| **Stockage source** | Fichier CSV local | CSV sur HDFS (`hdfs://namenode:9000/data/{size}/`) |
| **Parallelisme lecture** | Mono-thread | Chaque rank DDP lance sa propre `SparkSession local[N]` |
| **Conversion finale** | `DataFrame` direct | `.toPandas()` → identique ensuite |
| **Fault tolerance** | Aucune (fichier unique) | **Replication factor=2** — perte d'un datanode = 0 perte de donnees |
| **t_load (100k)** | **2.3s** | 15-21s (overhead JVM Spark) |
| **Scalabilite** | Limite par RAM machine | Horizontal — ajouter des datanodes |
| **Format sortie** | DataFrame pandas | DataFrame pandas (meme API) |

#### Avantages de HDFS sur les fichiers CSV locaux

```
1. TOLERRANCE AUX PANNES
   CSV local : un seul fichier → perte = 100% des donnees
   HDFS      : chaque bloc replique sur 2 datanodes
               → perte d'un datanode = donnees toujours disponibles
               → NameNode detecte et re-replique automatiquement

2. LOCALITE DES DONNEES (Data Locality)
   CSV local : donnees sur le disque du master, compute sur worker
   HDFS      : "move computation to data" — Spark lit directement
               depuis le datanode local au worker

3. SCALABILITE HORIZONTALE
   CSV local : limite par le disque d'une machine
   HDFS      : ajouter un datanode = augmenter le stockage lineairement
               sans changer le code applicatif

4. GESTION DES GROS FICHIERS
   CSV local : un seul fichier de 350 MB (full) = goulot d'etranglement
   HDFS      : decoupe en blocs de 128 MB distribues
               → lectures paralleles par plusieurs workers

5. AUDIT ET METADATA
   CSV local : pas de metadata systeme
   HDFS      : NameNode trace chaque bloc (taille, checksum, localisation)
               → verifiable via hdfs dfsck
```

#### Performance comparee (dataset full)

| Metrique | Standard w=1 | Bigdata w=2 | Bigdata w=3 | Bigdata w=4 |
|----------|-------------|-------------|-------------|-------------|
| `t_load` | **4.6s** (pandas) | 21s (Spark) | 27s (Spark) | 35s (Spark) |
| `t_train` | **257.8s** | 187.1s | 166.5s | **166.5s** |
| **Speedup** | 1.00× | 1.38× | **1.55×** | **1.55×** |
| Loader | pandas CSV | Spark HDFS | Spark HDFS | Spark HDFS |
| VRAM/worker | 4 GB | 2 GB | 2 GB | 2 GB |

> Note : `t_load` plus eleve en bigdata (overhead JVM Spark + connexion HDFS)
> mais `t_train` accelere grace au sharding BPR entre workers DDP.

---

### 10.3 Architecture HDFS detaillee

```
HDFS Namespace (/data/)
├── /data/1k/
│   ├── yelp_academic_dataset_review_healthandmedical.csv     (2.2 MB)
│   ├── yelp_academic_dataset_user_healthandmedical.csv
│   └── yelp_academic_dataset_business_healthandmedical.csv
├── /data/5k/  ...
├── /data/10k/ ...
├── /data/50k/ ...
├── /data/100k/
│   └── yelp_academic_dataset_review_healthandmedical.csv    (189.6 MB)
├── /data/full/
│   └── yelp_academic_dataset_review_healthandmedical.csv    (350.6 MB)
└── /user/spark/streaming/
    ├── pubmed-articles/   (Parquet partitionne par batch 30s)
    └── arxiv-papers/      (Parquet partitionne par batch 30s)
```

Composants :

| Composant | Role | Port |
|-----------|------|------|
| **NameNode** | Gere le namespace, metadonnees, replication | 9870 (UI), 9000 (RPC) |
| **DataNode 1** | Stocke les blocs de donnees (volume Docker) | — |
| **DataNode 2** | Replica des memes blocs (volume Docker) | — |

Configuration cle (`hadoop.env`) :
```
HDFS_CONF_dfs_replication=2
HDFS_CONF_dfs_blocksize=134217728   # 128 MB
```

Verification de sante :
```bash
# Etat du cluster
docker exec namenode hdfs dfsck /data/full -files -blocks -locations
# Attendu : Live_repl=2, bloc present sur datanode1 ET datanode2

# Quitter le safe mode si necessaire
docker exec namenode hdfs dfsadmin -safemode leave

# Lister les fichiers
docker exec namenode hdfs dfs -ls /data/
```

---

### 10.4 Couche ingestion — Kafka + Producteurs

#### Architecture Kafka

```
PubMed API (NCBI Entrez)                    ArXiv API (export.arxiv.org)
      │                                           │
      │  pubmed_producer.py                       │  arxiv_producer.py
      │  - Entrezpy / Biopython                   │  - requests + XML parsing
      │  - pmid, title, abstract, authors         │  - arxiv_id, categories
      │  - Batch de 50 articles / appel            │  - Atom/XML → JSON
      │                                           │
      ▼                                           ▼
┌─────────────────────────────────────────────────────────┐
│                  KAFKA BROKER  (port 9092)              │
│                                                         │
│  Topic: pubmed-articles   (1 partition, retention=7j)   │
│  ┌──┬──┬──┬──┬──┐                                      │
│  │m1│m2│m3│m4│m5│ ...  offset 0 → N                    │
│  └──┴──┴──┴──┴──┘                                      │
│                                                         │
│  Topic: arxiv-papers      (1 partition, retention=7j)   │
│  ┌──┬──┬──┬──┬──┐                                      │
│  │a1│a2│a3│a4│a5│ ...                                  │
│  └──┴──┴──┴──┴──┘                                      │
│                                                         │
│  Zookeeper (port 2181) : gestion des offsets/leaders    │
└─────────────────────────────────────────────────────────┘
      │                              │
      ▼ Spark consumer               ▼ Logstash consumer
```

#### Schema des messages Kafka

**Topic `pubmed-articles`** :
```json
{
  "source":   "pubmed",
  "pmid":     "12345678",
  "title":    "Graph Neural Networks for...",
  "abstract": "...",
  "authors":  ["Smith J", "Doe A"],
  "date":     "2024-01-15"
}
```

**Topic `arxiv-papers`** :
```json
{
  "source":     "arxiv",
  "arxiv_id":   "2401.12345",
  "title":      "...",
  "abstract":   "...",
  "authors":    ["..."],
  "categories": ["cs.LG", "cs.IR"],
  "doi":        "...",
  "journal_ref": "..."
}
```

---

### 10.5 Spark Structured Streaming — Consumer

```python
# pipeline/consumers/spark_consumer.py

spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "pubmed-articles,arxiv-papers")
    .load()
    # Schema StructType programmatique (pas de fromJson — compatibilite PySpark 3.5)
    # Ecriture Parquet partitionne toutes les 30s
    # → hdfs://namenode:9000/user/spark/streaming/{topic}/
```

Avantages du format **Parquet** vs CSV :

| Aspect | CSV | Parquet |
|--------|-----|---------|
| **Compression** | Aucune | Snappy/Gzip — 3-5× moins de place |
| **Lecture selective** | Lecture complete | Columnar — lit uniquement les colonnes utiles |
| **Schema** | Infera a la lecture | Stocke avec les donnees |
| **Performance** | Lente sur gros fichiers | Optimisee pour analytics |
| **Compatibilite** | Universel | Spark, Hive, BigQuery, Pandas |

Gestion via PowerShell :
```powershell
.\run_consumer.ps1           # Mode HDFS (defaut)
.\run_consumer.ps1 -Console  # Affichage console
.\run_consumer.ps1 -Status   # Derniers logs
.\run_consumer.ps1 -Stop     # Arret
```

---

### 10.6 Monitoring ELK — Elasticsearch + Logstash + Kibana

Le stack ELK est entièrement configuré et fonctionnel. Il consomme les mêmes topics Kafka que Spark, en parallèle, et indexe les articles dans Elasticsearch pour visualisation dans Kibana.

#### Architecture ELK

```
Kafka Topics
  pubmed-articles  ──┐
  arxiv-articles   ──┤──► LOGSTASH ──► ELASTICSEARCH ──► KIBANA
                      │   port 5044    port 9200          port 5601
                      │
                      │  Pipeline: kafka-to-elastic.conf
                      │  consumer_group: logstash-elk-consumer
                      │  (independant du consumer Spark)
```

#### Pipeline Logstash (`pipeline/logstash/pipeline/kafka-to-elastic.conf`)

**INPUT — Lecture Kafka :**
```ruby
input {
  kafka {
    bootstrap_servers => "kafka:29092"
    topics            => ["pubmed-articles", "arxiv-articles"]
    codec             => json { charset => "UTF-8" }
    group_id          => "logstash-elk-consumer"
    auto_offset_reset => "earliest"     # relit depuis le debut si nouveau consumer
    consumer_threads  => 2
    decorate_events   => true           # ajoute kafka.topic, kafka.partition, kafka.offset
  }
}
```

**FILTER — Enrichissement :**
```ruby
filter {
  # 1. Tag selon la source (pubmed ou arxiv)
  if [source] == "pubmed" { mutate { add_tag => ["pubmed"] } }
  else if [source] == "arxiv" { mutate { add_tag => ["arxiv"] } }

  # 2. Parse les dates en format Elasticsearch
  date { match => ["published_at", "yyyy-MM-dd"] target => "published_at" }
  date { match => ["ingested_at",  "ISO8601"]     target => "ingested_at"  }

  # 3. Calcule des metriques utiles
  ruby { code => "event.set('abstract_length', event.get('abstract').to_s.length)" }
  ruby { code => "event.set('authors_count', event.get('authors').is_a?(Array) ? event.get('authors').length : 0)" }

  # 4. Ajoute metadata pipeline
  mutate {
    add_field => { "pipeline_version" => "1.0"  "indexed_at" => "%{+ISO8601}" }
    remove_field => ["@version", "event"]
  }
}
```

**OUTPUT — Index Elasticsearch par source et par mois :**
```ruby
output {
  if "pubmed" in [tags] {
    elasticsearch {
      hosts       => ["elasticsearch:9200"]
      index       => "pubmed-articles-%{+YYYY.MM}"   # ex: pubmed-articles-2026.06
      document_id => "%{pmid}"                        # deduplication par PMID
      action      => "index"
    }
  }
  else if "arxiv" in [tags] {
    elasticsearch {
      hosts       => ["elasticsearch:9200"]
      index       => "arxiv-articles-%{+YYYY.MM}"
      document_id => "%{arxiv_id}"
      action      => "index"
    }
  }
  stdout { codec => rubydebug { metadata => false } }  # debug logs
}
```

#### Index Elasticsearch créés

| Index | Document ID | Champs indexés |
|-------|-------------|----------------|
| `pubmed-articles-YYYY.MM` | `pmid` | title, abstract, authors, published_at, abstract_length, authors_count |
| `arxiv-articles-YYYY.MM` | `arxiv_id` | title, abstract, authors, categories, doi, published_at |

Déduplication native : si un article est réingéré, `document_id` identique → `action=index` met à jour sans dupliquer.

#### Démarrage et vérification

```bash
# Démarrer le stack complet
cd pipeline && docker compose up -d

# Vérifier Elasticsearch
curl http://localhost:9200/_cluster/health?pretty
# Attendu : "status": "green" ou "yellow"

# Lister les index créés
curl http://localhost:9200/_cat/indices?v

# Compter les documents indexés
curl http://localhost:9200/pubmed-articles-*/_count
curl http://localhost:9200/arxiv-articles-*/_count

# Vérifier que Logstash consomme bien
docker logs logstash --tail=20
```

#### Kibana — Dashboards disponibles

URL : **http://localhost:5601**

Configuration initiale (une seule fois) :
```
1. Menu → Stack Management → Data Views
2. Créer une Data View :
   - Name: "Articles scientifiques"
   - Index pattern: *-articles-*
   - Timestamp field: published_at
3. Menu → Discover → sélectionner la Data View
```

Métriques visualisables dans Kibana :

| Visualisation | Type | Champ |
|---------------|------|-------|
| Articles par source | Pie chart | `source.keyword` |
| Articles ingérés / jour | Time series | `ingested_at` |
| Articles publiés / mois | Time series | `published_at` |
| Longueur moyenne des abstracts | Metric | `abstract_length` (avg) |
| Nombre d'auteurs | Histogram | `authors_count` |
| Top catégories ArXiv | Bar chart | `categories.keyword` |
| Flux temps réel | Discover | tri par `ingested_at` desc |

#### Statut actuel

Le stack ELK est **configuré et fonctionnel** mais non lancé en ce moment.  
Pour l'activer :
```bash
cd pipeline
docker compose up -d elasticsearch logstash kibana
# Attendre ~60s que Elasticsearch soit healthy
docker compose ps   # vérifier que tous sont "healthy"
```

---

### 10.7 Services et ports complets

| Service | Port | URL | Role |
|---------|------|-----|------|
| **Kafka Broker** | 9092 | (interne) | Message broker |
| **Zookeeper** | 2181 | (interne) | Coordination Kafka |
| **Kafka UI** | 8080 | http://localhost:8080 | Monitoring topics/offsets |
| **HDFS NameNode UI** | 9870 | http://localhost:9870 | Etat cluster HDFS |
| **HDFS NameNode RPC** | 9000 | (interne) | API Hadoop |
| **Spark Master UI** | 8081 | http://localhost:8081 | Jobs Spark |
| **Spark Worker UI** | 8082 | http://localhost:8082 | Worker details |
| **Logstash Beats** | 5044 | (interne) | Ingestion logs |
| **Elasticsearch** | 9200 | http://localhost:9200 | Index full-text |
| **Kibana** | 5601 | http://localhost:5601 | Dashboards |

---

## 10.bis Avantage HDFS pour le stockage — Analyse réelle sur le dataset Full

> Mesures effectuées sur le dataset Yelp Health & Medical complet.  
> Outil : Python `pyarrow` + `gzip` + `snappy`.

---

### Taille brute du dataset Full (CSV)

| Fichier | Lignes | Taille CSV |
|---------|--------|-----------|
| `yelp_academic_dataset_business_healthandmedical.csv` | 11 890 businesses | **5.16 MB** |
| `yelp_academic_dataset_review_healthandmedical.csv` | 188 044 reviews | **136.68 MB** |
| `yelp_academic_dataset_user_healthandmedical.csv` | 145 683 utilisateurs | **208.74 MB** |
| **TOTAL** | **345 617 lignes** | **350.58 MB** |

---

### CSV vs Parquet+Snappy (stockage HDFS) — Mesures réelles

| Fichier | CSV | Parquet (sans compr.) | Parquet + Snappy | Gain |
|---------|-----|-----------------------|-----------------|------|
| business | 5.16 MB | 3.73 MB | **1.52 MB** | **−70.5 %** |
| review | 136.68 MB | 133.55 MB | **82.87 MB** | **−39.4 %** |
| user | 208.74 MB | 205.57 MB | **202.93 MB** | **−2.8 %** |
| **TOTAL** | **350.58 MB** | 342.86 MB | **287.33 MB** | **−18.0 %** |

**Avec Snappy : on passe de 350.58 MB à 287.33 MB → 63.25 MB économisés.**

---

### Pourquoi les gains sont si différents selon le fichier ?

**Business (-70.5%) — Excellent**  
Les colonnes `name`, `categories`, `city`, `state` ont une **faible cardinalité** : beaucoup de valeurs qui se répètent (`"Health & Medical"`, `"CA"`, `"Dentists"`…). Parquet stocke ces colonnes en encodage dictionnaire : au lieu de répéter la chaîne, il stocke un entier + une table de traduction. Résultat : compression massive.

**Review (-39.4%) — Bon**  
Les colonnes `stars` (1–5), `useful`, `funny`, `cool` sont des entiers à faible plage → très bien compressés. Seul `text` (le texte libre de la review) résiste à la compression car le langage naturel est entropique. Il représente ~60% de la taille du fichier.

**User (-2.8%) — Faible**  
Le coupable : la colonne `friends` contient des listes de `user_id` séparés par des virgules, avec une longueur moyenne de **2 448 caractères par cellule**. Les `user_id` Yelp sont des chaînes de 22 caractères aléatoires (`1McG5Rn_UDkmlkZOrsdptg`) → entropie maximale → incompressible. Cette colonne seule représente ~80% du fichier user.

```
Exemple colonne friends :
"1McG5Rn_UDkmlkZOrsdptg, 4oTPkH5t9a0TaKiCGl3ibA, q6lcNi3GpE1oBm0P7xFp7g, ..."
(moyenne : 2448 caractères = ~112 user_ids par utilisateur)
```

---

### Impact de la réplication HDFS

HDFS réplique chaque bloc sur plusieurs DataNodes pour la tolérance aux pannes.

| Mode | Facteur de réplication | Taille totale (dataset full) |
|------|------------------------|------------------------------|
| CSV local (1 copie) | ×1 | 350.58 MB |
| HDFS par défaut (production) | **×3** | ~862 MB (Parquet+Snappy) |
| HDFS ce projet (demo) | **×1** | **287.33 MB** ← configuré dans docker-compose |

> Dans ce projet : `HDFS_CONF_dfs_replication=1` → pas de réplication, car un seul DataNode (environnement demo). En production avec 3 nœuds, la tolérance aux pannes vaut le surcoût ×3.

---

### Le vrai avantage : la vitesse de lecture (pas juste la taille)

La taille n'est que la partie visible. L'avantage principal de Parquet+HDFS sur CSV est la **vitesse de requête**.

**Exemple concret : construction du graphe GNN**

Pour construire le graphe bipartite, le pipeline GNN n'a besoin que de 3 colonnes sur 8 dans le fichier review :

```
Colonnes review : user_id, business_id, stars, date, review_id, text, useful, funny, cool
                  ───────   ───────────  ─────
                  ✓ utiles  ✓ utile      ✓ utile   ✗ ignorées (5 colonnes)
```

| Format | Ce qui est lu depuis le disque |
|--------|-------------------------------|
| **CSV** | 100% du fichier — toutes les colonnes, même `text` (gros) |
| **Parquet** | Seulement les 3 colonnes demandées — lecture **columnar** |

**En CSV** : Spark lit 136.68 MB entièrement, puis filtre en mémoire.  
**En Parquet+HDFS** : Spark lit uniquement les colonnes `user_id`, `business_id`, `stars` → ~**12 MB** réellement lus depuis le disque.

**Gain de lecture : ×11 sur le fichier review pour notre pipeline GNN.**

---

### Parallélisme HDFS — Blocs et workers Spark

HDFS découpe les fichiers en blocs de 128 MB. Chaque bloc est traité par un worker Spark indépendamment.

```
Dataset full (350 MB) → HDFS découpe en blocs :
  Bloc 1 : reviews 1–140k      → spark-worker-1
  Bloc 2 : reviews 140k–188k   → spark-worker-2
  Bloc 3 : users 1–145k        → spark-worker-3
  Bloc 4 : businesses           → spark-worker-4 (petit, fusionné)

→ 4 workers traitent en parallèle → temps divisé par ~4
```

Avec des CSV locaux, Spark peut paralléliser via des partitions mais le **bottleneck est le disque** (un seul fichier, un seul accès séquentiel). Avec HDFS, chaque bloc est sur un DataNode différent → **I/O parallèle physique**.

---

### Résumé comparatif

| Critère | CSV local | HDFS + Parquet/Snappy |
|---------|-----------|----------------------|
| Taille stockage | 350.58 MB | **287.33 MB (−18%)** |
| Colonnes lues (GNN) | 136.68 MB (tout) | **~12 MB (×11 moins)** |
| Tolérance aux pannes | ❌ Aucune | ✅ Réplication configurable |
| Lecture parallèle | Partielle | ✅ Blocs distribués |
| Filtrage avant chargement | ❌ Toujours tout charger | ✅ Predicate pushdown |
| Schema validation | ❌ Au runtime | ✅ Schema enforced |
| Compression adaptative | ❌ Uniforme | ✅ Par colonne selon type |

---

## 11. Configuration

### 11.1 Parametres principaux (config.py)

```python
# DataConfig
rating_thresh    = 1      # garder toutes les interactions
relevance_thresh = 4.0    # non utilise pour le graphe (heritage)
max_users        = 0      # no limit (145 683 users sur full)
max_reviews      = 0      # no limit (188 044 reviews sur full)

# EvalConfig
relevance_thresh = 3.0    # Stars >= 3.0 = item pertinent (metriques ranking)
max_eval_users   = 5000   # utilisateurs evalues par run

# Entrainement
num_epochs = 200
lr = 0.005              # SAGE/LightGCN
gat_lr = 0.001          # GAT (sensible au LR)
reg_lambda = 1e-5
emb_dim = 64
n_neg = 10              # Negatifs par positif (BPR)
min_epochs = 80
patience = 15

# HPO
ndcg_w = 0.4            # Critere composite eq. 3.21
prec_w = 0.3
rec_w  = 0.3
n_trials = 20
optuna_epochs = 50

# Evaluation
k_list = [5, 10, 20]
```

### 11.2 Arguments CLI

```
--model       {sage, gat, lightgcn}
--mode        {scratch, incremental, evaluate, tune, recommend}
--data-dir    Dossier contenant les 3 CSV Yelp
--ckpt        Checkpoint source (incremental, evaluate)
--ckpt-dir    Dossier de sauvegarde
--new-data    CSV reviews pour incremental
--epochs      Epochs d'entrainement
--emb-dim     Dimension embeddings
--lr          Learning rate
--trials      Trials Optuna (mode tune)
--finetune-epochs    Epochs fine-tuning
--finetune-lr-scale  Multiplicateur LR (defaut: 0.1)
--replay-ratio       Fraction replay buffer (defaut: 0.3)
--debug       Mode debug (emb_dim=16, epochs=3, AMP off)
--no-amp      Desactiver Mixed Precision
--config      Fichier YAML de config
```

---

## 12. Checkpoints

### 12.1 Nomenclature

```
checkpoints/sage/
  sage_best.pt          <- meilleur val_score composite
  sage_v001_e0020.pt    <- periodique epoch 20
  sage_v002_e0040.pt    <- periodique epoch 40
  sage_latest.pt        <- dernier etat
```

### 12.2 Contenu

Chaque `.pt` contient :
- `model_state_dict` : poids du modele (DDP-safe, prefixe `module.` supprime)
- `optimizer_state` / `scheduler_state` / `scaler_state`
- `epoch`, `val_score`
- `model_config` : architecture complete pour reconstruction sans CLI
- `user_encoder` / `item_encoder` : `DynamicLabelEncoder` serialise
- `extra.train_interactions` : IDs et ratings du train (reconstruction graphe incremental)
- `extra.replay_buffer` : buffer d'interactions historiques

### 12.3 API

```python
from utils.checkpoint import CheckpointManager

# Charger
ckpt = CheckpointManager.load("checkpoints/sage/sage_best.pt", device)

# Reconstruire le modele
model = CheckpointManager.build_model_from_ckpt(ckpt, device, build_model)

# Infos
n_users = len(ckpt["user_encoder"].classes_)
n_items = len(ckpt["item_encoder"].classes_)
```

---

## 13. Benchmark

### 13.1 Structure

```
benchmark/
  runner.py            # Orchestre preprocess -> train -> evaluate
  train_one.py         # Run unique avec monitoring ressources
  train_distributed.py # Run DDP avec isolation memoire
  preprocess_pandas.py # Pretraitement standard
  reporter.py          # Agregation + graphiques
  resource_monitor.py  # CPU/GPU/RAM en temps reel
```

### 13.2 Sortie

Un JSON par run dans `outputs/benchmark/` :
```json
{
  "model": "sage",
  "dataset": "medium",
  "ndcg@10": 0.1234,
  "precision@10": 0.0891,
  "train_time_s": 142.3,
  "memory_peak_mb": 2048
}
```

---

## 14. Tests

```
tests/
  conftest.py            # Fixtures : mini-dataset (100 reviews), mini-modele
  test_checkpointing.py  # Sauvegarde / chargement cycle complet
  test_data_loading.py   # CSV -> encodage -> edge_index
  test_graph_builder.py  # Proprietes du graphe bipartite
  test_metrics.py        # NDCG, Precision, Recall (valeurs connues)
  test_training_debug.py # 3 epochs debug, convergence BPR
```

```bash
python3.13 -m pytest tests/ -v
python3.13 -m pytest tests/test_metrics.py -v
python3.13 -m pytest tests/ -v -k "checkpoint"
```

---

## 15. Dépannage

### ModuleNotFoundError: torch
Cause : Python 3.10 (defaut Windows) n'a pas torch.
Solution : Utiliser `python3.13` systematiquement.

### FileNotFoundError: sage_best.pt
Cause : L'entrainement n'a pas produit de checkpoint.
Solution : Verifier "[Checkpoint] New best -> sage_best.pt" dans les logs. Utiliser --epochs 100+.

### HDFS Safe Mode
Cause : NameNode attend la reconnexion des datanodes.
Solution : `docker exec namenode hdfs dfsadmin -safemode leave`

### OOM CUDA out of memory
Solution : Reduire --emb-dim (64 -> 32) ou ajouter --no-amp.

### BPR loss ne converge pas
Cause : Trop peu d'interactions ou LR trop faible.
Solution : --finetune-epochs 50, activer HPO dans Streamlit, utiliser data/medium.

### Cache Streamlit perime
Cause : Apres run incremental, les caches pointent sur l'ancien checkpoint.
Solution : Redemarrer `.\run_app.ps1`.

### KeyError: metadata (PySpark)
Deja corrige : le consumer utilise des StructType programmatiques.

### No FileSystem for scheme "C" (HDFS upload)
Cause : Git Bash sur Windows convertit `/data/1k` → `C:/data/1k`.
Solution : `export MSYS_NO_PATHCONV=1 && export MSYS2_ARG_CONV_EXCL="*"` avant toute commande `docker exec`.

### OutOfMemoryError: Java heap space (Spark, full dataset)
Cause : `spark.driver.memory=1g` insuffisant pour 188k reviews.
Solution : Deja corrige a `4g` + `spark.driver.maxResultSize=2g` dans `src/data/loader.py`.

### PytorchStreamWriter failed / file write failed
Cause : Disque Docker Desktop plein (fichier `.vhdx` sature).
Solution : `docker system prune -f` + Docker Desktop → Settings → Resources → Disk image size.

### Outputs enregistres dans `*_50k` au lieu de `*_full`
Cause : Variable `SIZE` non passee dans la commande.
Solution : Toujours passer les 4 variables : `WORLD_SIZE`, `MODEL_TYPE`, `DATA_DIR`, `SIZE`.

### Connection closed by peer (gloo, ranks 1/2/3)
Cause : Rank 0 a crashe → les autres tombent en cascade.
Solution : Lire l'erreur du rank 0 (toujours la root cause), pas des ranks 1-3.

### Spark Ivy cache FileNotFoundError
```bash
docker exec -u root spark-master bash -c \
  "mkdir -p /home/spark/.ivy2/cache && chown -R spark:spark /home/spark/.ivy2"
```
