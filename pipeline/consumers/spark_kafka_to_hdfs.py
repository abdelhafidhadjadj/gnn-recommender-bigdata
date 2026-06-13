"""
Spark Structured Streaming — Kafka → HDFS
==========================================
Consomme les topics pubmed-articles et arxiv-articles depuis Kafka
et écrit en Parquet sur HDFS, distribué sur les 4 workers Spark.

Usage (depuis le container spark-master ou en local) :
    spark-submit \
        --master spark://spark-master:7077 \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
        consumers/spark_kafka_to_hdfs.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, TimestampType
)

# ── Schema articles ────────────────────────────────────────────────────────────
ARTICLE_SCHEMA = StructType([
    StructField("title",        StringType(), True),
    StructField("abstract",     StringType(), True),
    StructField("authors",      ArrayType(StringType()), True),
    StructField("published_at", StringType(), True),
    StructField("source",       StringType(), True),
    StructField("pmid",         StringType(), True),
    StructField("arxiv_id",     StringType(), True),
    StructField("url",          StringType(), True),
    StructField("ingested_at",  StringType(), True),
])

KAFKA_SERVERS = "kafka:29092"
HDFS_BASE     = "hdfs://namenode:9000/articles"
CHECKPOINT    = "hdfs://namenode:9000/checkpoints/spark-streaming"

def main():
    spark = (
        SparkSession.builder
        .appName("KafkaToHDFS")
        .config("spark.sql.shuffle.partitions", "4")    # 1 partition par worker
        .config("spark.default.parallelism", "4")       # forcer 4 partitions
        .config("spark.executor.instances", "4")        # 1 executor par worker
        .config("spark.executor.cores", "1")
        .config("spark.executor.memory", "512m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ── Lire depuis Kafka (les 2 topics) ──────────────────────────────────────
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", "pubmed-articles,arxiv-articles")
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 500)
        .load()
    )

    # ── Parser JSON ───────────────────────────────────────────────────────────
    articles = (
        raw
        .select(
            col("topic").alias("kafka_topic"),
            col("partition"),
            col("offset"),
            col("timestamp").alias("kafka_ts"),
            from_json(col("value").cast("string"), ARTICLE_SCHEMA).alias("data")
        )
        .select(
            "kafka_topic", "partition", "offset", "kafka_ts",
            col("data.title"),
            col("data.abstract"),
            col("data.authors"),
            col("data.published_at"),
            col("data.source"),
            col("data.pmid"),
            col("data.arxiv_id"),
            col("data.url"),
            col("data.ingested_at"),
            current_timestamp().alias("hdfs_written_at"),
        )
    )

    # ── Écrire sur HDFS en Parquet (partitionné par source + date) ────────────
    query = (
        articles.writeStream
        .format("parquet")
        .option("path", HDFS_BASE)
        .option("checkpointLocation", f"{CHECKPOINT}/articles")
        .partitionBy("kafka_topic")
        .trigger(processingTime="10 seconds")   # micro-batch toutes les 10s
        .outputMode("append")
        .start()
    )

    print(f"[KafkaToHDFS] Streaming en cours -> {HDFS_BASE}")
    print(f"  Topics  : pubmed-articles, arxiv-articles")
    print(f"  Workers : 4 Spark workers")
    print(f"  Output  : Parquet partitionné sur HDFS")
    query.awaitTermination()

if __name__ == "__main__":
    main()
