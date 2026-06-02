"""
Spark Structured Streaming Consumer
====================================
Lit les topics Kafka `pubmed-articles` et `arxiv-articles`,
parse les messages JSON, nettoie les données
et les écrit en Parquet sur HDFS (ou en local en fallback).

Lancement via spark-submit:
    spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 \
      consumers/spark_consumer.py \
      --mode hdfs

Ou directement (mode local):
    python consumers/spark_consumer.py --mode local
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_BOOTSTRAP_SERVERS_INTERNAL,
    TOPIC_PUBMED, TOPIC_ARXIV,
    SPARK_APP_NAME, SPARK_KAFKA_PKG,
    HDFS_PUBMED_PATH, HDFS_ARXIV_PATH, LOCAL_OUTPUT_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SparkConsumer] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spark_consumer")


# ── Schemas ───────────────────────────────────────────────────────────────────
# Built programmatically to avoid PySpark version differences with JSON schema

def _build_pubmed_schema():
    from pyspark.sql.types import StructType, StructField, StringType, ArrayType
    return StructType([
        StructField("source",       StringType(), True),
        StructField("pmid",         StringType(), True),
        StructField("doi",          StringType(), True),
        StructField("title",        StringType(), True),
        StructField("abstract",     StringType(), True),
        StructField("authors",      ArrayType(StringType(), True), True),
        StructField("journal",      StringType(), True),
        StructField("keywords",     ArrayType(StringType(), True), True),
        StructField("published_at", StringType(), True),
        StructField("ingested_at",  StringType(), True),
    ])


def _build_arxiv_schema():
    from pyspark.sql.types import StructType, StructField, StringType, ArrayType
    return StructType([
        StructField("source",           StringType(), True),
        StructField("arxiv_id",         StringType(), True),
        StructField("doi",              StringType(), True),
        StructField("title",            StringType(), True),
        StructField("abstract",         StringType(), True),
        StructField("authors",          ArrayType(StringType(), True), True),
        StructField("categories",       ArrayType(StringType(), True), True),
        StructField("primary_category", StringType(), True),
        StructField("journal_ref",      StringType(), True),
        StructField("pdf_url",          StringType(), True),
        StructField("published_at",     StringType(), True),
        StructField("ingested_at",      StringType(), True),
    ])


# ── Spark session ─────────────────────────────────────────────────────────────

def build_spark(mode: str):
    from pyspark.sql import SparkSession

    # Kafka connector + Hadoop client packages
    kafka_pkg  = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0"
    hadoop_pkg = "org.apache.hadoop:hadoop-client:3.3.0"
    packages   = f"{kafka_pkg},{hadoop_pkg}"

    builder = (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .config("spark.jars.packages", packages)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        # Avoid winutils.exe warning on Windows
        .config("spark.hadoop.hadoop.security.authentication", "simple")
    )

    if mode == "local":
        builder = builder.master("local[2]")
    elif mode == "hdfs":
        builder = (
            builder
            .master("spark://spark-master:7077")
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
            # Replication factor = 2 (matches 2 datanodes)
            .config("spark.hadoop.dfs.replication", "2")
        )

    return builder.getOrCreate()


# ── Stream builder ────────────────────────────────────────────────────────────

def read_kafka_stream(spark, bootstrap_servers: str, topics: str):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topics)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 1000)
        .load()
    )


def process_topic(spark, stream_df, schema_fn, topic_name: str,
                  output_path: str, checkpoint_path: str):
    """Parse, clean and write one topic's stream to Parquet."""
    from pyspark.sql.functions import (
        col, from_json, to_timestamp, trim, lower,
        size, when, lit, current_timestamp,
    )

    schema = schema_fn()

    parsed = (
        stream_df
        .filter(col("topic") == topic_name)
        .select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_timestamp"),
            col("offset"),
            col("partition"),
        )
        .select("data.*", "kafka_timestamp", "offset", "partition")
    )

    # Cleaning transforms
    cleaned = (
        parsed
        .withColumn("title",    trim(col("title")))
        .withColumn("abstract", trim(col("abstract")))
        .withColumn("title_lower", lower(col("title")))
        .withColumn("has_abstract", when(
            col("abstract").isNotNull() & (col("abstract") != ""), True
        ).otherwise(False))
        .withColumn("n_authors", when(
            col("authors").isNotNull(), size(col("authors"))
        ).otherwise(lit(0)))
        .withColumn("published_at", to_timestamp(col("published_at"), "yyyy-MM-dd"))
        .withColumn("processed_at", current_timestamp())
        # Drop rows with no title
        .filter(col("title").isNotNull() & (col("title") != ""))
    )

    # Write stream
    query = (
        cleaned.writeStream
        .format("parquet")
        .option("path", output_path)
        .option("checkpointLocation", checkpoint_path)
        .outputMode("append")
        .partitionBy("source")
        .trigger(processingTime="30 seconds")
        .start()
    )
    log.info(f"Stream started for topic '{topic_name}' → {output_path}")
    return query


# ── Console mode (debug) ──────────────────────────────────────────────────────

def process_console(spark, stream_df, schema_fn, topic_name: str):
    """Print parsed messages to console (debug mode)."""
    from pyspark.sql.functions import col, from_json

    schema = schema_fn()

    parsed = (
        stream_df
        .filter(col("topic") == topic_name)
        .select(from_json(col("value").cast("string"), schema).alias("data"))
        .select("data.title", "data.source", "data.published_at", "data.ingested_at")
    )

    return (
        parsed.writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", "false")
        .option("numRows", 5)
        .trigger(processingTime="10 seconds")
        .start()
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run(mode: str, console_only: bool) -> None:
    log.info(f"Starting Spark consumer (mode={mode})")

    spark = build_spark(mode)
    spark.sparkContext.setLogLevel("WARN")

    # Choose bootstrap servers
    bootstrap = (
        KAFKA_BOOTSTRAP_SERVERS_INTERNAL if mode == "hdfs"
        else KAFKA_BOOTSTRAP_SERVERS
    )
    log.info(f"Kafka bootstrap: {bootstrap}")

    # Read both topics in one stream
    topics = f"{TOPIC_PUBMED},{TOPIC_ARXIV}"
    stream_df = read_kafka_stream(spark, bootstrap, topics)

    queries = []

    if console_only:
        # Debug: just print to console
        q1 = process_console(spark, stream_df, _build_pubmed_schema, TOPIC_PUBMED)
        q2 = process_console(spark, stream_df, _build_arxiv_schema,  TOPIC_ARXIV)
        queries = [q1, q2]
    else:
        # Production: write to HDFS or local Parquet
        if mode == "hdfs":
            pubmed_out = HDFS_PUBMED_PATH
            arxiv_out  = HDFS_ARXIV_PATH
            pubmed_ckpt = "hdfs://namenode:9000/checkpoints/pubmed"
            arxiv_ckpt  = "hdfs://namenode:9000/checkpoints/arxiv"
        else:
            os.makedirs(f"{LOCAL_OUTPUT_PATH}/pubmed", exist_ok=True)
            os.makedirs(f"{LOCAL_OUTPUT_PATH}/arxiv",  exist_ok=True)
            pubmed_out  = f"{LOCAL_OUTPUT_PATH}/pubmed"
            arxiv_out   = f"{LOCAL_OUTPUT_PATH}/arxiv"
            pubmed_ckpt = f"{LOCAL_OUTPUT_PATH}/checkpoints/pubmed"
            arxiv_ckpt  = f"{LOCAL_OUTPUT_PATH}/checkpoints/arxiv"

        q1 = process_topic(spark, stream_df, _build_pubmed_schema, TOPIC_PUBMED,
                            pubmed_out, pubmed_ckpt)
        q2 = process_topic(spark, stream_df, _build_arxiv_schema,  TOPIC_ARXIV,
                            arxiv_out,  arxiv_ckpt)
        queries = [q1, q2]

    log.info(f"Streaming {len(queries)} topic(s). Press Ctrl+C to stop.")
    try:
        for q in queries:
            q.awaitTermination()
    except KeyboardInterrupt:
        log.info("Stopping streams…")
        for q in queries:
            q.stop()
    finally:
        spark.stop()
        log.info("Spark session stopped.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kafka → Spark → HDFS/Parquet consumer")
    p.add_argument("--mode",    choices=["local", "hdfs"], default="local",
                   help="local=spark local mode, hdfs=cluster mode")
    p.add_argument("--console", action="store_true",
                   help="Print to console only (debug, no file output)")
    args = p.parse_args()
    run(mode=args.mode, console_only=args.console)
