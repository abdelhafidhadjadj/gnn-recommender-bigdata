# Documentation — Pipeline d'Ingestion de Données Scientifiques

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture détaillée](#2-architecture-détaillée)
3. [Composants](#3-composants)
   - 3.1 [Sources de données](#31-sources-de-données)
   - 3.2 [Apache Kafka](#32-apache-kafka)
   - 3.3 [Kafka UI](#33-kafka-ui)
   - 3.4 [Apache Hadoop / HDFS](#34-apache-hadoop--hdfs)
   - 3.5 [Apache Spark](#35-apache-spark)
   - 3.6 [Stack ELK](#36-stack-elk)
4. [Flux de données](#4-flux-de-données)
5. [Schémas des données](#5-schémas-des-données)
6. [Configuration](#6-configuration)
7. [Déploiement Docker](#7-déploiement-docker)
8. [Producers](#8-producers)
9. [Consumer Spark](#9-consumer-spark)
10. [Monitoring avec ELK](#10-monitoring-avec-elk)
11. [Isolation par rapport au GNN Recommender](#11-isolation-par-rapport-au-gnn-recommender)
12. [Avantages et limites](#12-avantages-et-limites)
13. [Références](#13-références)

---

## 1. Vue d'ensemble

Ce pipeline constitue la **couche d'ingestion de données** du projet. Il est entièrement isolé de l'architecture du GNN Recommender (entraîné sur Yelp) et sert à démontrer qu'une infrastructure de collecte et de streaming de données scientifiques peut coexister avec un système de recommandation.

### Objectif

Simuler un pipeline de données réel en production capable de :
- **Collecter** des articles scientifiques depuis des APIs publiques (PubMed, ArXiv)
- **Streamer** les données en temps réel via un message broker (Kafka)
- **Stocker** les données brutes de façon distribuée (HDFS)
- **Transformer** et structurer les données (Spark Structured Streaming)
- **Monitorer** et visualiser le flux d'ingestion (ELK Stack)

### Ce que ce pipeline NE fait PAS

Ce pipeline ne produit pas de données au format user-item et n'alimente pas le GNN Recommender. Son rôle est exclusivement de démontrer la couche d'ingestion.

---

## 2. Architecture détaillée

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SOURCES DE DONNÉES                               │
│                                                                         │
│   ┌─────────────────┐              ┌─────────────────┐                  │
│   │   PubMed API    │              │    ArXiv API    │                  │
│   │  (NCBI Entrez)  │              │  (Atom/XML)     │                  │
│   │  ~35M articles  │              │  ~2.3M articles │                  │
│   └────────┬────────┘              └────────┬────────┘                  │
└────────────┼───────────────────────────────┼────────────────────────────┘
             │ Python Producer               │ Python Producer
             │ (pubmed_producer.py)          │ (arxiv_producer.py)
             ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          KAFKA CLUSTER                                  │
│                                                                         │
│   ┌──────────────┐     ┌────────────────────┐    ┌──────────────────┐  │
│   │  Zookeeper   │────►│     Kafka Broker   │    │    Kafka UI      │  │
│   │  Port: 2181  │     │    Port: 9092      │    │   Port: 8080     │  │
│   └──────────────┘     │                    │    └──────────────────┘  │
│                         │  Topics:           │                          │
│                         │  • pubmed-articles │                          │
│                         │  • arxiv-articles  │                          │
│                         │  • dead-letter-    │                          │
│                         │    queue           │                          │
│                         └────────┬───────────┘                          │
└──────────────────────────────────┼──────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
┌─────────────────────────┐             ┌──────────────────────────────┐
│     SPARK STREAMING     │             │         ELK STACK            │
│                         │             │                              │
│  ┌───────────────────┐  │             │  ┌──────────┐  Kafka Input  │
│  │   Spark Master    │  │             │  │ Logstash │◄──────────────│
│  │   Port: 8081      │  │             │  └────┬─────┘               │
│  └────────┬──────────┘  │             │       │                     │
│           │             │             │       ▼                     │
│  ┌────────▼──────────┐  │             │  ┌──────────────┐          │
│  │   Spark Worker    │  │             │  │Elasticsearch │          │
│  │   Port: 8082      │  │             │  │  Port: 9200  │          │
│  └────────┬──────────┘  │             │  └──────┬───────┘          │
│           │  Structured │             │         │                   │
│           │  Streaming  │             │  ┌──────▼───────┐          │
└───────────┼─────────────┘             │  │    Kibana    │          │
            │                           │  │  Port: 5601  │          │
            ▼                           │  └──────────────┘          │
┌─────────────────────────┐             └──────────────────────────────┘
│     HADOOP / HDFS       │
│                         │
│  ┌───────────────────┐  │
│  │     NameNode      │  │
│  │   Port: 9870/9000 │  │
│  └────────┬──────────┘  │
│           │             │
│  ┌────────▼──────────┐  │
│  │     DataNode      │  │
│  │  /data/scientific │  │
│  │  ├── pubmed/      │  │
│  │  └── arxiv/       │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

---

## 3. Composants

### 3.1 Sources de données

#### PubMed (NCBI Entrez)

PubMed est la base de données biomédicale de référence gérée par le National Center for Biotechnology Information (NCBI). Elle contient plus de 35 millions de citations d'articles scientifiques.

| Paramètre | Valeur |
|---|---|
| API | NCBI Entrez E-utilities |
| URL | https://eutils.ncbi.nlm.nih.gov/entrez/eutils |
| Authentification | Aucune (clé optionnelle pour +10 req/s) |
| Rate limit | 3 requêtes/seconde sans clé API |
| Format retour | XML (efetch) / JSON (esearch) |
| Endpoints utilisés | `esearch.fcgi` (recherche), `efetch.fcgi` (contenu) |

**Données collectées par article :**
- PMID (identifiant unique PubMed)
- Titre
- Abstract
- Liste des auteurs
- Journal de publication
- Date de publication
- Mots-clés MeSH
- DOI

#### ArXiv

ArXiv est une archive ouverte de prépublications scientifiques gérée par Cornell University. Elle contient plus de 2.3 millions d'articles en mathématiques, physique, informatique, etc.

| Paramètre | Valeur |
|---|---|
| API | ArXiv API (Atom/XML) |
| URL | http://export.arxiv.org/api/query |
| Authentification | Aucune |
| Rate limit | 1 requête / 3 secondes |
| Format retour | Atom XML |
| Tri disponible | `submittedDate`, `lastUpdatedDate`, `relevance` |

**Catégories collectées :**
- `cs.LG` — Machine Learning
- `cs.AI` — Artificial Intelligence
- `cs.IR` — Information Retrieval
- `stat.ML` — Statistics / Machine Learning

**Données collectées par article :**
- ArXiv ID
- Titre
- Abstract
- Auteurs
- Catégories (primaire + secondaires)
- Lien PDF
- Date de soumission
- DOI et référence journal (si disponibles)

---

### 3.2 Apache Kafka

Kafka est le message broker central du pipeline. Il découple les producers (collecte) des consumers (traitement) et garantit la durabilité des messages.

#### Rôle dans le pipeline

```
Producer → [Kafka Topic] → Consumer(s)
```

Plusieurs consumers peuvent lire le même topic indépendamment et à leur propre rythme. Spark et Logstash sont deux consumers distincts qui lisent les mêmes messages sans interférence.

#### Topics configurés

| Topic | Description | Rétention |
|---|---|---|
| `pubmed-articles` | Articles collectés depuis PubMed | 24h |
| `arxiv-articles` | Articles collectés depuis ArXiv | 24h |
| `dead-letter-queue` | Messages en erreur (parsing échoué) | 24h |

#### Configuration réseau

Kafka expose deux listeners pour gérer les connexions internes (Docker) et externes (machine hôte) :

| Listener | Adresse | Usage |
|---|---|---|
| `INTERNAL` | `kafka:29092` | Communication inter-containers Docker |
| `EXTERNAL` | `localhost:9092` | Connexion depuis la machine hôte |

#### Propriétés clés

```properties
replication.factor = 1          # Single broker (démo)
auto.create.topics = true       # Création automatique des topics
log.retention.hours = 24        # Conservation 24h
offsets.topic.replication = 1
```

---

### 3.3 Kafka UI

Interface web open-source (Provectus) pour visualiser et gérer le cluster Kafka.

**Accès :** http://localhost:8080

**Fonctionnalités disponibles :**
- Liste et contenu des topics (messages, offsets, partitions)
- Consumer groups et lag de consommation
- Statistiques du broker (throughput, taille des messages)
- Création/suppression de topics
- Visualisation des messages en temps réel

---

### 3.4 Apache Hadoop / HDFS

HDFS (Hadoop Distributed File System) est le système de fichiers distribué qui stocke les données brutes sous forme de fichiers Parquet.

#### Architecture

```
Client (Spark) ──► NameNode (métadonnées) ──► DataNode (données réelles)
                   Port 9870 (Web UI)          /hadoop/dfs/data
                   Port 9000 (RPC)
```

**NameNode** — gère les métadonnées du système de fichiers (noms des fichiers, blocs, emplacements). Ne stocke pas les données réelles.

**DataNode** — stocke les blocs de données. En production il y en a plusieurs pour la redondance. Ici un seul DataNode suffit pour la démonstration.

#### Structure des données dans HDFS

```
/data/scientific/
├── pubmed/
│   ├── source=pubmed/
│   │   ├── part-00000-xxx.parquet
│   │   └── part-00001-xxx.parquet
│   └── _SUCCESS
└── arxiv/
    ├── source=arxiv/
    │   ├── part-00000-xxx.parquet
    │   └── part-00001-xxx.parquet
    └── _SUCCESS
```

**Accès Web UI :** http://localhost:9870 — visualiser les fichiers, l'espace disque, l'état des DataNodes.

#### Configuration (hadoop.env)

```properties
CORE_CONF_fs_defaultFS=hdfs://namenode:9000
HDFS_CONF_dfs_webhdfs_enabled=true
HDFS_CONF_dfs_permissions_enabled=false
HDFS_CONF_dfs_replication=1
```

---

### 3.5 Apache Spark

Spark est utilisé en **Structured Streaming** pour consommer les messages Kafka, les transformer et les écrire en Parquet sur HDFS.

#### Mode de traitement : Structured Streaming

Spark Structured Streaming traite les données comme un flux continu de micro-batches. Toutes les N secondes (configurable), il lit les nouveaux messages Kafka, applique les transformations et écrit les résultats.

```
Kafka Topic
    │
    ▼
readStream (Kafka source)
    │
    ▼
from_json()  → parse le JSON
    │
    ▼
Transformations :
  • trim(title, abstract)
  • to_timestamp(published_at)
  • size(authors) → n_authors
  • length(abstract) → abstract_length
  • filter(title != null)
    │
    ▼
writeStream (Parquet sink)
  • partitionBy("source")
  • trigger(30 seconds)
  • checkpointLocation
```

#### Cluster

| Composant | Port Web | Rôle |
|---|---|---|
| Spark Master | 8081 | Coordinateur, allocation des ressources |
| Spark Worker | 8082 | Exécuteur des tâches |

#### Packages requis pour Kafka

```
org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0
org.apache.hadoop:hadoop-client:3.3.0
```

---

### 3.6 Stack ELK

La stack ELK (Elasticsearch + Logstash + Kibana) fournit le monitoring et la visualisation du pipeline en temps réel.

#### Logstash

Logstash lit les messages Kafka (consumer group `logstash-elk-consumer`) et les indexe dans Elasticsearch.

**Pipeline Logstash :**

```
Input (Kafka)
  topics: pubmed-articles, arxiv-articles
  bootstrap_servers: kafka:29092
  codec: json
        │
        ▼
Filters
  • Tagging par source (pubmed / arxiv)
  • Parsing des dates (published_at, ingested_at)
  • Calcul abstract_length
  • Calcul authors_count
        │
        ▼
Output (Elasticsearch)
  pubmed → index: pubmed-articles-YYYY.MM
  arxiv  → index: arxiv-articles-YYYY.MM
  document_id: pmid / arxiv_id (déduplication)
```

#### Elasticsearch

Moteur d'indexation et de recherche full-text. Stocke les documents JSON pour permettre des requêtes analytiques rapides.

**Accès :** http://localhost:9200

```bash
# Vérifier la santé du cluster
curl http://localhost:9200/_cluster/health?pretty

# Voir les index créés
curl http://localhost:9200/_cat/indices?v

# Compter les documents PubMed
curl http://localhost:9200/pubmed-articles-*/_count

# Exemple de recherche full-text
curl -X GET "http://localhost:9200/arxiv-articles-*/_search" \
  -H "Content-Type: application/json" \
  -d '{"query": {"match": {"title": "graph neural network"}}}'
```

#### Kibana

Interface de visualisation pour Elasticsearch.

**Accès :** http://localhost:5601

**Visualisations suggérées :**
- Histogramme des publications par mois (`published_at`)
- Top 20 auteurs les plus productifs
- Distribution des catégories ArXiv
- Volume d'ingestion en temps réel (`ingested_at`)
- Longueur moyenne des abstracts par source
- Carte des journaux (PubMed)

---

## 4. Flux de données

### Flux complet étape par étape

```
Étape 1 — Collecte
  pubmed_producer.py appelle esearch.fcgi → obtient liste de PMIDs
  pubmed_producer.py appelle efetch.fcgi  → obtient XML des articles
  Parsing XML → extraction des champs → dict Python

Étape 2 — Sérialisation et publication
  dict Python → json.dumps() → bytes UTF-8
  KafkaProducer.send(topic="pubmed-articles", key=pmid, value=bytes)
  → Message stocké dans le log Kafka (partition 0)

Étape 3A — Consumer Spark
  Spark readStream lit les offsets depuis le checkpoint
  Nouveaux messages décodés : bytes → JSON → DataFrame
  Transformations appliquées (nettoyage, typage, enrichissement)
  Micro-batch écrit en Parquet sur HDFS toutes les 30 secondes
  Checkpoint mis à jour (offset Kafka)

Étape 3B — Consumer Logstash (parallèle)
  Logstash lit le même topic depuis son offset indépendant
  Filtres appliqués (date parsing, comptage, tagging)
  Document indexé dans Elasticsearch
  Document_id = pmid → déduplication automatique

Étape 4 — Visualisation
  Kibana interroge Elasticsearch
  Dashboards mis à jour en temps réel
```

### Garanties de livraison

| Garantie | Kafka | Spark | Logstash |
|---|---|---|---|
| At-least-once | ✅ (`acks=all`) | ✅ (checkpoints) | ✅ |
| Exactly-once | ✅ (idempotent producer) | ✅ (document_id ES) | ✅ (document_id ES) |
| Ordering | ✅ (par partition) | N/A (batch) | N/A |

---

## 5. Schémas des données

### Article PubMed (topic `pubmed-articles`)

| Champ | Type | Description |
|---|---|---|
| `source` | string | Toujours `"pubmed"` |
| `pmid` | string | Identifiant unique PubMed |
| `doi` | string | Digital Object Identifier (si disponible) |
| `title` | string | Titre de l'article |
| `abstract` | string | Résumé (max 2000 caractères) |
| `authors` | array[string] | Liste "Nom, Prénom" |
| `journal` | string | Nom du journal |
| `keywords` | array[string] | Mots-clés MeSH |
| `published_at` | string | Date ISO `YYYY-MM-DD` |
| `ingested_at` | string | Timestamp UTC d'ingestion |

### Article ArXiv (topic `arxiv-articles`)

| Champ | Type | Description |
|---|---|---|
| `source` | string | Toujours `"arxiv"` |
| `arxiv_id` | string | Identifiant ArXiv (ex: `2301.12345`) |
| `doi` | string | DOI si publié dans un journal |
| `title` | string | Titre de l'article |
| `abstract` | string | Résumé (max 2000 caractères) |
| `authors` | array[string] | Liste des auteurs |
| `categories` | array[string] | Catégories ArXiv (ex: `["cs.LG", "cs.AI"]`) |
| `primary_category` | string | Catégorie principale |
| `journal_ref` | string | Référence journal si accepté |
| `pdf_url` | string | URL du PDF |
| `published_at` | string | Date de soumission `YYYY-MM-DD` |
| `ingested_at` | string | Timestamp UTC d'ingestion |

### Champs ajoutés par Spark (enrichissement)

| Champ | Type | Description |
|---|---|---|
| `title_lower` | string | Titre en minuscules (pour recherche) |
| `has_abstract` | boolean | Abstract non vide |
| `n_authors` | integer | Nombre d'auteurs |
| `processed_at` | timestamp | Timestamp de traitement Spark |
| `kafka_timestamp` | timestamp | Timestamp Kafka du message |
| `offset` | long | Offset Kafka |
| `partition` | integer | Partition Kafka |

---

## 6. Configuration

Tous les paramètres sont centralisés dans `config/settings.py`.

### Kafka

```python
KAFKA_BOOTSTRAP_SERVERS          = "localhost:9092"    # hôte → Kafka
KAFKA_BOOTSTRAP_SERVERS_INTERNAL = "kafka:29092"       # Docker interne
TOPIC_PUBMED                     = "pubmed-articles"
TOPIC_ARXIV                      = "arxiv-articles"
TOPIC_DLQ                        = "dead-letter-queue"
```

### PubMed

```python
PUBMED_BASE_URL    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_BATCH_SIZE  = 50        # articles par appel API
PUBMED_EMAIL       = "demo@pipeline.io"   # requis par NCBI
```

### ArXiv

```python
ARXIV_BASE_URL     = "http://export.arxiv.org/api/query"
ARXIV_BATCH_SIZE   = 50
ARXIV_CATEGORIES   = ["cs.LG", "cs.AI", "cs.IR", "stat.ML"]
```

### HDFS

```python
HDFS_OUTPUT_PATH   = "/data/scientific"
HDFS_PUBMED_PATH   = "/data/scientific/pubmed"
HDFS_ARXIV_PATH    = "/data/scientific/arxiv"
```

---

## 7. Déploiement Docker

### Services et ressources

| Service | Image | CPU | RAM | Port(s) |
|---|---|---|---|---|
| zookeeper | confluentinc/cp-zookeeper:7.5.0 | faible | 256MB | 2181 |
| kafka | confluentinc/cp-kafka:7.5.0 | moyen | 512MB | 9092 |
| kafka-ui | provectus/kafka-ui | faible | 256MB | 8080 |
| namenode | bde2020/hadoop-namenode:3.2.1 | faible | 512MB | 9870, 9000 |
| datanode | bde2020/hadoop-datanode:3.2.1 | faible | 256MB | — |
| spark-master | bde2020/spark-master:3.3.0 | moyen | 512MB | 8081, 7077 |
| spark-worker | bde2020/spark-worker:3.3.0 | élevé | 1GB | 8082 |
| elasticsearch | elasticsearch:8.11.0 | moyen | 1GB | 9200 |
| logstash | logstash:8.11.0 | moyen | 512MB | 5044 |
| kibana | kibana:8.11.0 | faible | 512MB | 5601 |
| **Total** | | | **~5.5 GB** | |

> ⚠️ Allouer au minimum **8 GB RAM** à Docker Desktop.
> Dans Docker Desktop → Settings → Resources → Memory → 8 GB

### Ordre de démarrage (géré par `depends_on`)

```
zookeeper → kafka → kafka-ui
namenode → datanode
namenode → spark-master → spark-worker
elasticsearch → logstash
elasticsearch → kibana
kafka + elasticsearch → logstash
```

### Volumes persistants

```yaml
hadoop_namenode:    # métadonnées HDFS
hadoop_datanode:    # blocs de données HDFS
elasticsearch_data: # index Elasticsearch
```

---

## 8. Producers

### pubmed_producer.py

Le producer PubMed fonctionne en deux phases :

**Phase 1 — Recherche (esearch)**
```
GET /esearch.fcgi?db=pubmed&term=<query>&retmax=<n>&retmode=json
→ {"esearchresult": {"idlist": ["37123456", "37123457", ...]}}
```

**Phase 2 — Récupération par batch (efetch)**
```
GET /efetch.fcgi?db=pubmed&id=37123456,37123457,...&retmode=xml
→ XML avec <PubmedArticleSet><PubmedArticle>...</PubmedArticle></PubmedArticleSet>
```

**Gestion des erreurs :**
- Timeout API → retry automatique (3 tentatives)
- Message Kafka non livré → envoi dans `dead-letter-queue`
- Article sans titre → ignoré silencieusement

### arxiv_producer.py

L'API ArXiv retourne directement les métadonnées complètes en une seule requête (pas besoin d'une phase de recherche séparée).

```
GET /api/query?search_query=all:graph+neural+network&start=0&max_results=50
→ Atom XML avec <entry>...</entry> pour chaque article
```

**Déduplication :** le `arxiv_id` est utilisé comme clé Kafka — si le même article est envoyé deux fois, Kafka le stocke une fois par offset mais Elasticsearch le déduplique via `document_id`.

---

## 9. Consumer Spark

### Structured Streaming vs DStream

Ce pipeline utilise **Structured Streaming** (DataFrame API) plutôt que l'ancien DStream. Avantages :
- API SQL familière (`select`, `filter`, `withColumn`)
- Gestion automatique des offsets et des checkpoints
- Meilleure tolérance aux pannes
- Support natif de Kafka comme source

### Trigger et latence

```python
.trigger(processingTime="30 seconds")
```

Toutes les 30 secondes, Spark lit les nouveaux messages Kafka depuis le dernier offset sauvegardé et écrit un nouveau fichier Parquet. La latence de bout-en-bout est donc de 0-30 secondes.

### Checkpoints

Les checkpoints HDFS permettent la reprise après panne sans perte ni duplication de données :

```
hdfs://namenode:9000/checkpoints/pubmed/
├── commits/
│   └── 0, 1, 2, ...      ← micro-batch IDs
├── offsets/
│   └── 0, 1, 2, ...      ← derniers offsets Kafka lus
└── sources/
    └── 0/                 ← metadata source Kafka
```

### Partitionnement Parquet

Les fichiers sont partitionnés par `source` pour permettre des lectures filtrées efficaces :

```
/data/scientific/pubmed/source=pubmed/part-xxxx.parquet
/data/scientific/arxiv/source=arxiv/part-xxxx.parquet
```

---

## 10. Monitoring avec ELK

### Créer les index patterns dans Kibana

1. Ouvrir http://localhost:5601
2. **Stack Management** → **Index Patterns** → **Create index pattern**
3. Pattern : `pubmed-articles-*` → champ de date : `published_at` → **Create**
4. Répéter pour `arxiv-articles-*`

### Dashboards recommandés

#### Volume d'ingestion
- Visualisation : **Area chart**
- X-axis : `ingested_at` (date histogram, intervalle 1 minute)
- Y-axis : Count
- Permet de voir le débit du pipeline en temps réel

#### Distribution des catégories (ArXiv)
- Visualisation : **Pie chart**
- Aggregation : Terms sur `primary_category.keyword`
- Top 10 catégories

#### Top auteurs
- Visualisation : **Horizontal bar**
- Aggregation : Terms sur `authors.keyword`
- Top 20 auteurs

#### Timeline des publications
- Visualisation : **Line chart**
- X-axis : `published_at` (date histogram, intervalle 1 mois)
- Permet d'observer la distribution temporelle des articles collectés

### Requêtes Elasticsearch utiles

```bash
# Nombre total d'articles indexés
curl http://localhost:9200/_cat/indices/pubmed-articles-*,arxiv-articles-*?v

# Recherche full-text dans les abstracts
curl -X GET "http://localhost:9200/arxiv-articles-*/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {"match": {"abstract": "graph neural network recommendation"}},
    "size": 5,
    "_source": ["title", "authors", "published_at"]
  }'

# Articles les plus récents
curl -X GET "http://localhost:9200/pubmed-articles-*/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{
    "sort": [{"published_at": "desc"}],
    "size": 5,
    "_source": ["title", "journal", "published_at"]
  }'

# Statistiques sur les longueurs d'abstracts
curl -X GET "http://localhost:9200/arxiv-articles-*/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{
    "size": 0,
    "aggs": {
      "avg_abstract_length": {"avg": {"field": "abstract_length"}},
      "max_abstract_length": {"max": {"field": "abstract_length"}}
    }
  }'
```

---

## 11. Isolation par rapport au GNN Recommender

Ce pipeline est **entièrement isolé** du code du GNN Recommender :

| Aspect | GNN Recommender | Pipeline d'ingestion |
|---|---|---|
| Répertoire | `src/`, `demo/` | `pipeline/` |
| Données | Yelp (user-item) | PubMed, ArXiv (articles) |
| Format | CSV/Parquet Yelp | JSON Kafka / Parquet HDFS |
| Infrastructure | Local (Python) | Docker (Kafka, Spark, ELK) |
| Objectif | Recommandation | Collecte et streaming |
| Schéma | user_id, item_id, rating | title, abstract, authors, ... |

Les deux composants peuvent coexister sur la même machine sans interférence. Aucun fichier du répertoire `src/`, `demo/` ou `benchmark/` n'est modifié.

---

## 12. Avantages et limites

### Avantages

| Avantage | Détail |
|---|---|
| **Découplage total** | Producers et consumers évoluent indépendamment |
| **Scalabilité horizontale** | Ajouter des partitions Kafka, des Workers Spark, des DataNodes |
| **Tolérance aux pannes** | Kafka réplication, Spark checkpoints, HDFS replication |
| **Monitoring intégré** | ELK fournit visibilité complète sur le flux |
| **Sources multiples** | Facile d'ajouter IEEE Xplore, Semantic Scholar, etc. |
| **Déduplication** | `document_id` Elasticsearch, clé Kafka, offsets Spark |

### Limites (contexte démo)

| Limite | Cause | Solution en production |
|---|---|---|
| HDFS single DataNode | Démo locale | Ajouter N DataNodes Docker |
| Kafka single broker | Démo locale | Cluster 3+ brokers |
| Pas de schéma registry | Simplicité | Confluent Schema Registry |
| Rate limits APIs | NCBI/ArXiv | Clé API NCBI (10 req/s), ArXiv bulk S3 |
| RAM requise (~6GB) | Tous les services sur une machine | Distribuer sur plusieurs machines |

---

## 13. Références

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Apache Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- [Apache Hadoop HDFS Architecture](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)
- [NCBI Entrez API](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [ArXiv API Documentation](https://arxiv.org/help/api/index)
- [Elasticsearch Reference](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Logstash Kafka Input Plugin](https://www.elastic.co/guide/en/logstash/current/plugins-inputs-kafka.html)
- [Kafka UI (Provectus)](https://github.com/provectus/kafka-ui)
- [bde2020 Docker Hadoop](https://github.com/big-data-europe/docker-hadoop)
