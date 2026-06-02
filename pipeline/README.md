# Pipeline Big Data — Ingestion de données scientifiques

Pipeline de streaming independant du GNN Recommender pour collecter des articles PubMed et ArXiv.

```
PubMed API --+                              +--> HDFS Parquet
             +--> Kafka --> Spark -----------+    datanode1 (172.x.x.12:9866)
ArXiv API  --+      |       (streaming)     +--> datanode2 (172.x.x.8:9866)
                    |       replication=2
                    |
                    +--> Logstash --> Elasticsearch --> Kibana
```

---

## Interfaces web

| Service | URL | Description |
|---------|-----|-------------|
| Kafka UI | http://localhost:8080 | Visualiser les topics et messages |
| HDFS NameNode | http://localhost:9870 | Etat du cluster HDFS |
| Spark Master | http://localhost:8081 | Jobs Spark en cours |
| Spark Worker | http://localhost:8082 | Metriques worker |
| Kibana | http://localhost:5601 | Dashboards ELK |
| Elasticsearch | http://localhost:9200 | API REST index |

---

## 1. Prerequis

- **Docker Desktop** installe et demarre
- **python3.13** avec les dependances pipeline

```bash
python3.13 -m pip install -r pipeline/requirements.txt
```

---

## 2. Demarrer l'infrastructure

```bash
cd pipeline

# Demarrer tous les services
docker compose up -d

# Verifier l'etat
docker compose ps

# Suivre les logs
docker compose logs -f kafka
docker compose logs -f namenode
docker compose logs -f elasticsearch
```

> Premier demarrage : 3-5 minutes (telechargement des images)

---

## 3. Verifier les services

```bash
# Kafka pret ?
curl http://localhost:8080

# Elasticsearch pret ?
curl http://localhost:9200/_cluster/health

# HDFS pret ? (doit afficher 2 datanodes live)
curl http://localhost:9870

# Verifier les 2 datanodes
docker exec namenode hdfs dfsadmin -report | grep "Live datanodes"
```

---

## 4. Lancer les Producteurs

Ouvrir deux terminaux depuis la racine du projet.

### Terminal 1 — PubMed

```bash
# 100 articles sur un sujet (une fois)
python3.13 pipeline/producers/pubmed_producer.py --query "graph neural network" --max 100

# En boucle toutes les 5 minutes
python3.13 pipeline/producers/pubmed_producer.py --query "deep learning health" --max 200 --loop --interval 300

# Autres sujets
python3.13 pipeline/producers/pubmed_producer.py --query "recommendation system" --max 50
python3.13 pipeline/producers/pubmed_producer.py --query "medical imaging AI" --max 100
```

### Terminal 2 — ArXiv

```bash
# Par categories (cs.LG = Machine Learning, cs.AI = Artificial Intelligence)
python3.13 pipeline/producers/arxiv_producer.py --category cs.LG cs.AI --max 100

# Par mot-cle
python3.13 pipeline/producers/arxiv_producer.py --query "graph neural network" --max 150

# En boucle
python3.13 pipeline/producers/arxiv_producer.py --loop --interval 120 --max 200
```

---

## 5. Lancer le Consumer Spark

Utiliser le script PowerShell depuis le dossier `pipeline/` :

```powershell
cd pipeline

# Mode HDFS (stockage distribue sur 2 datanodes)
.\run_consumer.ps1

# Mode console (debug — affiche sans sauvegarder)
.\run_consumer.ps1 -Console

# Voir les logs en direct
.\run_consumer.ps1 -Status

# Arreter
.\run_consumer.ps1 -Stop
```

Le script gere automatiquement :
- Copie des fichiers dans le container `spark-master`
- Verification des permissions Ivy cache
- Lancement via `spark-submit` dans le container
- Surveillance du demarrage

---

## 6. Stockage HDFS Distribue

### Architecture 2 DataNodes

Le pipeline utilise **2 DataNodes** avec **replication factor = 2** :
- Chaque bloc Parquet est stocke sur `datanode1` ET `datanode2`
- Haute disponibilite : la perte d'un datanode ne cause pas de perte de donnees

### Chemins HDFS

```
hdfs://namenode:9000/user/spark/streaming/
  pubmed/    -- articles PubMed en Parquet
  arxiv/     -- articles ArXiv en Parquet
```

### Verifier la replication

```bash
# Sante complete du stockage
docker exec namenode hdfs dfsck /user/spark/streaming -files -blocks -locations

# Chaque bloc doit afficher :
# Live_repl=2   (replique sur 2 datanodes)
# datanode1:9866 + datanode2:9866
```

### Quitter le Safe Mode si necessaire

```bash
# Apres redemarrage des datanodes, le NameNode peut entrer en safe mode
docker exec namenode hdfs dfsadmin -safemode get

# Forcer la sortie
docker exec namenode hdfs dfsadmin -safemode leave
```

### Lister les fichiers HDFS

```bash
docker exec namenode hdfs dfs -ls /user/spark/streaming/pubmed/
docker exec namenode hdfs dfs -ls /user/spark/streaming/arxiv/

# Taille totale
docker exec namenode hdfs dfs -du -h /user/spark/streaming/
```

---

## 7. Topics Kafka

| Topic | Producteur | Champs principaux |
|-------|-----------|-------------------|
| `pubmed-articles` | pubmed_producer.py | source, pmid, title, abstract, authors, journal, published_at |
| `arxiv-papers` | arxiv_producer.py | source, arxiv_id, title, abstract, categories, pdf_url |

### Verifier dans Kafka UI

1. Ouvrir http://localhost:8080
2. Cluster : `scientific-pipeline`
3. Topics -> `pubmed-articles` -> voir les messages
4. Consumer Groups -> voir la progression de Logstash et Spark

---

## 8. Visualiser dans Kibana

1. Ouvrir http://localhost:5601
2. Menu **Stack Management -> Index Patterns**
3. Creer deux patterns : `pubmed-articles-*` et `arxiv-articles-*`
4. **Discover** -> selectionner un pattern -> explorer les articles
5. **Dashboard** -> creer des visualisations :
   - Histogramme des publications par mois
   - Top auteurs (terms aggregation)
   - Distribution des categories ArXiv
   - Volume d'ingestion en temps reel

---

## 9. Schemas de donnees

### Article PubMed
```json
{
  "source": "pubmed",
  "pmid": "12345678",
  "doi": "10.1000/xyz123",
  "title": "Graph Neural Networks for Medical Recommendation",
  "abstract": "In this paper we propose...",
  "authors": ["Smith, John", "Doe, Jane"],
  "journal": "Nature Machine Intelligence",
  "keywords": ["GNN", "deep learning", "health"],
  "published_at": "2023-06-15",
  "ingested_at": "2024-01-01T10:00:00"
}
```

### Article ArXiv
```json
{
  "source": "arxiv",
  "arxiv_id": "2301.12345",
  "title": "LightGCN: Simplifying Graph Convolution...",
  "abstract": "We propose LightGCN...",
  "authors": ["He, Xiangnan", "Deng, Kuan"],
  "categories": ["cs.IR", "cs.LG"],
  "primary_category": "cs.IR",
  "pdf_url": "https://arxiv.org/pdf/2301.12345",
  "published_at": "2023-01-30",
  "ingested_at": "2024-01-01T10:00:00"
}
```

---

## 10. Arreter l'infrastructure

```bash
cd pipeline

# Arreter sans supprimer les donnees
docker compose stop

# Redemarrer
docker compose start

# Arreter et supprimer les volumes (reset complet)
docker compose down -v
```

---

## 11. Structure du dossier

```
pipeline/
+-- docker-compose.yml          # Infrastructure complete (Kafka, HDFS, Spark, ELK)
+-- hadoop.env                  # Config HDFS (replication=2, 2 datanodes)
+-- requirements.txt            # kafka-python, pyspark, requests, feedparser, ...
+-- run_consumer.ps1            # Lanceur Spark consumer (Windows PowerShell)
|
+-- config/
|   +-- settings.py             # Topics, URLs, parametres centralises
|
+-- producers/
|   +-- pubmed_producer.py      # PubMed API -> Kafka (topic: pubmed-articles)
|   +-- arxiv_producer.py       # ArXiv API  -> Kafka (topic: arxiv-papers)
|
+-- consumers/
|   +-- spark_consumer.py       # Kafka -> Spark Streaming -> Parquet/HDFS
|
+-- logstash/
    +-- pipeline/
        +-- kafka-to-elastic.conf  # Kafka -> Elasticsearch
```

---

## 12. Depannage

### Safe Mode HDFS apres redemarrage
```bash
docker exec namenode hdfs dfsadmin -safemode leave
```

### Ivy cache Spark (FileNotFoundError)
```bash
docker exec -u root spark-master bash -c \
  "mkdir -p /home/spark/.ivy2/cache && chown -R spark:spark /home/spark/.ivy2"
```

### Consumer Spark ne demarre pas
```bash
# Voir les logs du container
docker logs spark-master --tail 50

# Verifier que Kafka est accessible depuis le container
docker exec spark-master curl -s kafka:9092
```

### Elasticsearch refuse les connexions
```bash
# Verifier l'etat
curl http://localhost:9200/_cluster/health

# Logs
docker logs elasticsearch --tail 30
```

### DataNode ne rejoint pas le cluster
```bash
# Voir les logs
docker logs datanode1 --tail 20
docker logs datanode2 --tail 20

# Reinitialiser les volumes si necessaire
docker compose down -v
docker compose up -d
```
