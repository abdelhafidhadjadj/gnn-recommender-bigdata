# Guide de Démonstration — GNN Recommender System
## Présentation au Jury

---

## Interfaces à présenter (ordre recommandé)

| # | Interface | URL | Durée |
|---|---|---|---|
| 1 | Spark Master UI | http://localhost:8081 | 3 min |
| 2 | HDFS NameNode UI | http://localhost:9870 | 2 min |
| 3 | Kafka UI | http://localhost:8080 | 2 min |
| 4 | Kibana (ELK) | http://localhost:5601 | 4 min |
| 5 | Streamlit — Recommandations | http://localhost:8501 | 3 min |
| 6 | Streamlit — Cold Start | http://localhost:8501 | 4 min |
| 7 | Streamlit — Incrémental | http://localhost:8501 | 4 min |
| 8 | Rapport HTML | fichier local | 3 min |

**Total : ~25 minutes**

---

## PRÉPARATION (à faire AVANT la démo)

### Lancer toute l'architecture

```bash
cd C:\Users\hafid\Desktop\gnn_recommender\pipeline
docker compose up -d
```

Attendre ~60 secondes puis vérifier :

```bash
docker compose ps
```

Tous ces containers doivent être **Up** :

```
zookeeper         Up (healthy)
kafka             Up (healthy)
kafka-ui          Up
namenode          Up (healthy)
datanode1         Up (healthy)
datanode2         Up (healthy)
spark-master      Up
spark-worker-1    Up
spark-worker-2    Up
spark-worker-3    Up
spark-worker-4    Up
pubmed-producer   Up
arxiv-producer    Up
spark-kafka-hdfs  Up
elasticsearch     Up (healthy)
logstash          Up
kibana            Up (healthy)
filebeat          Up
metricbeat        Up
```

### Lancer Streamlit (dans un autre terminal)

```bash
cd C:\Users\hafid\Desktop\gnn_recommender
streamlit run demo/app.py
```

---

## INTERFACE 1 — Spark Master UI

**URL : http://localhost:8081**

### Ce qu'on montre

- **Status : ALIVE**
- **Alive Workers : 4**
- `spark-worker-1` à `spark-worker-4` tous en état **ALIVE**
- **Cores in use : 8 Total** (2 cores × 4 workers)
- **Memory in use : 4.0 GiB** (1g × 4 workers)
- Si `spark-kafka-hdfs` tourne → une **Running Application** visible

### Ce qu'on dit

> *"Apache Spark est le moteur de traitement distribué de notre architecture.
> Les 4 workers Spark traitent le dataset Yelp Health & Medical en parallèle.
> Résultat concret : le temps d'entraînement GAT passe de 264 secondes (1 worker)
> à 33 secondes (4 workers), soit un speedup de ×7.95."*

---

## INTERFACE 2 — HDFS NameNode UI

**URL : http://localhost:9870**

### Ce qu'on montre

- **Overview** → Live Nodes : 2 DataNodes actifs
- **Utilities → Browse the file system**
  - Naviguer vers `/articles/` → fichiers Parquet écrits par Spark Streaming
  - Montrer la structure des blocs
- **Summary** → capacité totale, espace utilisé

### Ce qu'on dit

> *"HDFS est notre système de stockage distribué. Les données CSV (350 MB)
> sont converties en Parquet+Snappy (287 MB, -18%). La lecture columnar
> Parquet permet à Spark de ne lire que les 3 colonnes utiles
> (user_id, business_id, stars) sur les 9 disponibles,
> soit ×11 moins de données lues depuis le disque."*

---

## INTERFACE 3 — Kafka UI

**URL : http://localhost:8080**

### Ce qu'on montre

- **Topics** → `pubmed-articles` et `arxiv-articles`
  - Cliquer sur `pubmed-articles` → voir les messages entrants en temps réel
  - Montrer : offset, partition, contenu JSON d'un article
- **Brokers** → 1 broker actif
- **Consumer Groups** → `logstash-elk-consumer` (Logstash) et `spark-consumer` (Spark)

### Ce qu'on dit

> *"Kafka est le bus de messages central. Il découple les producteurs
> (collecte PubMed et ArXiv, 1000 articles toutes les 2 secondes)
> des consommateurs (Logstash → Elasticsearch et Spark → HDFS).
> Les deux consommateurs lisent indépendamment sans conflit."*

---

## INTERFACE 4 — Kibana (ELK Stack)

**URL : http://localhost:5601**

### 4a — Données scientifiques (articles)

Menu → **Discover** → sélectionner index `*-articles-*`

- Montrer les champs : `title`, `abstract`, `authors`, `published_at`, `source`
- Filtrer par `source: pubmed` vs `source: arxiv`
- Montrer le volume : **580+ articles indexés**

### 4b — Logs des containers Docker (Filebeat)

Menu → **Discover** → sélectionner index `docker-logs-spark-*`

- Montrer les logs en temps réel des workers Spark
- Filtrer par container : `container.name: spark-master`

### 4c — Métriques système (Metricbeat)

Menu → **Discover** → sélectionner index `metricbeat-docker-*`

- Montrer CPU %, RAM % de chaque container
- Montrer la charge des workers Spark pendant le training

### Ce qu'on dit

> *"ELK joue un triple rôle : stockage et recherche des articles scientifiques,
> centralisation des logs opérationnels de tous les containers via Filebeat,
> et métriques système en temps réel via Metricbeat. Kibana unifie
> la visualisation de toutes ces données."*

---

## INTERFACE 5 — Streamlit : Recommandations (utilisateur existant)

**URL : http://localhost:8501 → Onglet Recommandations**

### Configuration sidebar

- **Checkpoint** : `checkpoints/gat/gat_best.pt`
- **Répertoire données** : `C:\Users\hafid\Desktop\gnn_recommender\data\raw\full`

### Scénario

1. Sélectionner l'utilisateur **`Um5bfs5DH6eizgjH3xZsvg`** (51 interactions, moy 4.2⭐)
2. K = 10
3. Montrer :
   - L'historique d'interactions (items visités)
   - Le graphique de scores (barres horizontales)
   - Le tableau Top-10 avec noms et catégories

### Ce qu'on dit

> *"Pour un utilisateur connu, le modèle GAT calcule l'embedding utilisateur
> via le mécanisme d'attention multi-têtes sur le graphe bipartite.
> Le score de chaque item = produit scalaire entre l'embedding utilisateur
> et l'embedding item. GAT obtient NDCG@10 = 0.0086, soit ×21 supérieur
> à une recommandation aléatoire."*

---

## INTERFACE 6 — Streamlit : Cold Start (nouvel utilisateur)

**URL : http://localhost:8501 → Onglet Nouvel Utilisateur**

### Scénario A — Profil dentaire

1. Filtrer par catégorie → **`Dentists`**
2. Ajouter 2 dentistes au panier avec **5⭐**
3. Observer les recommandations → surtout des dentistes/orthodontistes

### Scénario B — Démontrer l'effet du rating

| Panier | Notes | Recommandations attendues |
|---|---|---|
| Dentiste A + Cardiologue B | **5⭐ + 1⭐** | Surtout dentistes |
| Dentiste A + Cardiologue B | **1⭐ + 5⭐** | Surtout cardiologues |

> *Changer les notes → les recommandations changent en temps réel*

### Ce qu'on dit

> *"Le cold-start résout le problème des nouveaux utilisateurs sans historique.
> Le profil est simulé par une moyenne pondérée des embeddings des items sélectionnés.
> La formule : user_proxy = Σ (rating_i / Σratings) × embedding_i.
> Un item noté 5⭐ influence 5× plus le profil qu'un item noté 1⭐."*

---

## INTERFACE 7 — Streamlit : Apprentissage Incrémental

**URL : http://localhost:8501 → Onglet Apprentissage Incrémental**

### Scénario

1. Mode → **Upload fichier**
2. Charger les 3 fichiers :
   - `data/incremental/incremental_reviews.csv`
   - `data/incremental/incremental_users.csv`
   - `data/incremental/incremental_business.csv`
3. Observer l'analyse croisée :
   - **1 356 reviews** | 60 users existants + **20 nouveaux** | 38 items + **12 nouveaux**
4. Paramètres : `epochs=20`, `LR scale=0.1`, `replay=0.3`
5. Cliquer **Lancer l'apprentissage incrémental**
6. Montrer recommandations **AVANT** vs **APRÈS**

### Ce qu'on dit

> *"L'apprentissage incrémental met à jour le modèle sans réentraînement complet.
> 3 mécanismes : (1) expansion du graphe — nouveaux nœuds initialisés par Xavier,
> (2) fine-tuning BPR avec learning rate réduit ×0.1 pour ne pas écraser l'existant,
> (3) replay buffer 30% des anciennes interactions pour éviter l'oubli catastrophique.
> Durée : ~20 secondes vs ~4 minutes pour un scratch complet."*

---

## INTERFACE 8 — Rapport HTML & Charts

### Générer le rapport

```bash
cd C:\Users\hafid\Desktop\gnn_recommender
python generate_report.py
python generate_charts.py
```

### Ouvrir

```
outputs/report.html
```

### Ce qu'on montre

1. **KPIs** : 127 569 interactions, 101 005 users, 11 719 items, 5 000 users évalués
2. **Tableau performances** :
   - GAT : RMSE=1.96, MAE=1.64, NDCG@10=0.0086, Accuracy=0.59, GlobalPrec=0.75
   - SAGE : RMSE=2.18, MAE=1.75, NDCG@10=0.0031
   - LightGCN : RMSE=1.91, MAE=1.78, NDCG@10=0.0050
3. **Speedup GAT : ×7.95** (264s → 33s avec 4 workers)
4. **Charts** :
   - `chart1_speedup.png` — Speedup par modèle
   - `chart2_training_time.png` — Temps Standard vs Big Data
   - `chart5_csv_vs_parquet.png` — Gain stockage -70.5% business

---

## COMMANDES UTILES

```bash
# Status de tous les containers
docker compose ps

# Logs d'un service spécifique
docker logs spark-master --tail 50
docker logs spark-worker-1 --tail 50
docker logs pubmed-producer --tail 30
docker logs arxiv-producer --tail 30

# Voir les fichiers Parquet sur HDFS
docker exec namenode hdfs dfs -ls /articles/

# Vérifier Elasticsearch
curl http://localhost:9200/_cat/indices?v

# Lancer Streamlit
cd C:\Users\hafid\Desktop\gnn_recommender
streamlit run demo/app.py

# Générer rapport HTML + charts
python generate_report.py
python generate_charts.py

# Arrêter tout
cd pipeline && docker compose down
```

---

## RÉPONSES AUX QUESTIONS PROBABLES DU JURY

### "Pourquoi GAT est meilleur que SAGE et LightGCN ?"
GAT utilise l'**attention multi-têtes** (4 têtes) qui apprend à pondérer différemment
chaque voisin. Sur un dataset médical sparse, il capture mieux les relations spécialisées
(cardiologue → interniste plutôt que → dentiste).

### "Pourquoi les métriques semblent faibles (NDCG@10 = 0.0086) ?"
Dataset **extrêmement sparse** : 1.29 interactions/user en moyenne avant filtrage.
- Baseline aléatoire : NDCG@10 = 0.0004
- Baseline popularité : NDCG@10 = 0.0128
- **GAT : NDCG@10 = 0.0086** (×21 vs aléatoire)

### "Quel est l'apport réel du Big Data ?"
- Speedup entraînement GAT : 264s → 33s = **×7.95** avec 4 workers Spark
- Stockage : CSV 350MB → Parquet+Snappy 287MB = **-18%**
- Lecture columnar : 3 colonnes sur 9 lues = **×11 moins d'I/O disque**

### "ELK sert à quoi dans votre architecture ?"
Triple rôle :
1. Stockage articles scientifiques (PubMed + ArXiv via Kafka)
2. Logs opérationnels de tous les containers (Filebeat)
3. Métriques système CPU/RAM/Docker (Metricbeat)

### "Comment fonctionne le cold-start ?"
```
Items sélectionnés → embeddings GNN → moyenne pondérée par rating
                                              ↓
                                      proxy utilisateur
                                              ↓
                              dot product × tous les items
                                              ↓
                                      Top-K recommandations
```

### "Pourquoi rating_thresh=3 ?"
Avec rating_thresh=1 (toutes les interactions), le modèle apprend aussi
des interactions négatives (1★, 2★) comme positives — il recommanderait
des établissements mal notés. Seulement les interactions ≥ 3★ = signal positif clair.

### "Qu'est-ce que le replay buffer ?"
Sans replay : après fine-tuning sur nouvelles données, le modèle "oublie"
les anciens utilisateurs (catastrophic forgetting). Le replay buffer mélange
30% d'anciennes interactions avec les nouvelles à chaque batch d'entraînement,
forçant le modèle à maintenir ses connaissances antérieures.

---

## UTILISATEURS RECOMMANDÉS POUR LA DÉMO

| user_id | Interactions | Moy ⭐ | Idéal pour |
|---|---|---|---|
| `Um5bfs5DH6eizgjH3xZsvg` | 51 | 4.2 | Démo principale (profil riche) |
| `I2XpWCHAom1JRyHXZQrnfg` | 41 | 4.1 | Profil alternatif |
| `RNWJx8g-TIMVn1fpNFOCvA` | 30 | 4.8 | Profil très positif |
