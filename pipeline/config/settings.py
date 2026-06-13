"""
Centralized configuration for the scientific data ingestion pipeline.
"""
import os

# ── Kafka ─────────────────────────────────────────────────────────────────────
# Lire depuis env var si défini (Docker), sinon localhost (host machine)
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_BOOTSTRAP_SERVERS_INTERNAL = "kafka:29092"    # inside Docker network

TOPIC_PUBMED  = "pubmed-articles"
TOPIC_ARXIV   = "arxiv-articles"
TOPIC_DLQ     = "dead-letter-queue"                 # failed messages

KAFKA_PRODUCER_CONFIG = {
    "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS,
    "value_serializer": None,   # set at runtime (JSON bytes)
    "acks": "all",
    "retries": 3,
    "max_block_ms": 10_000,
}

# ── PubMed (NCBI Entrez API) ──────────────────────────────────────────────────
PUBMED_BASE_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_EMAIL      = "demo@pipeline.io"              # required by NCBI (no key needed)
PUBMED_TOOL       = "gnn-scientific-pipeline"
PUBMED_BATCH_SIZE = 50                              # articles per API call (max 500)
PUBMED_DEFAULT_QUERY = (
    "machine learning[Title/Abstract] OR "
    "deep learning[Title/Abstract] OR "
    "graph neural network[Title/Abstract]"
)

# ── ArXiv API ─────────────────────────────────────────────────────────────────
ARXIV_BASE_URL    = "http://export.arxiv.org/api/query"
ARXIV_BATCH_SIZE  = 50
ARXIV_DEFAULT_QUERY = "graph neural network"
ARXIV_CATEGORIES  = ["cs.LG", "cs.AI", "cs.IR", "stat.ML"]

# ── HDFS ──────────────────────────────────────────────────────────────────────
HDFS_NAMENODE_URL   = "http://localhost:9870"
HDFS_OUTPUT_PATH    = "/data/scientific"
HDFS_PUBMED_PATH    = f"{HDFS_OUTPUT_PATH}/pubmed"
HDFS_ARXIV_PATH     = f"{HDFS_OUTPUT_PATH}/arxiv"

# ── Elasticsearch ─────────────────────────────────────────────────────────────
ES_HOST           = "http://localhost:9200"
ES_INDEX_PUBMED   = "pubmed-articles"
ES_INDEX_ARXIV    = "arxiv-articles"

# ── Spark ─────────────────────────────────────────────────────────────────────
SPARK_MASTER      = "spark://localhost:7077"
SPARK_APP_NAME    = "ScientificArticleConsumer"
SPARK_KAFKA_PKG   = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0,"
    "org.apache.hadoop:hadoop-client:3.3.0"
)
SPARK_OUTPUT_MODE = "parquet"                       
LOCAL_OUTPUT_PATH = "../outputs/pipeline"           # fallback when HDFS not available
