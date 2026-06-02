"""
Distributed preprocessing with PySpark.

Reads raw Yelp CSV files from HDFS (or local mount),
applies filtering / encoding / splitting, and writes parquet outputs:

  /processed/{size}/edges.parquet      — (user_idx, item_idx, rating, date, split)
  /processed/{size}/businesses.parquet — (item_idx, categories, business_id)
  /processed/{size}/meta.json          — stats + timing

Usage (spark-submit):
  spark-submit \\
    --master spark://spark-master:7077 \\
    /opt/spark-apps/preprocessing_spark.py \\
    --input-dir /data/1k \\
    --output-dir /processed/1k \\
    --size-tag 1k
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer


# ── Spark session ──────────────────────────────────────────────────────────────

def get_spark(app_name: str, master: str | None = None) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    hdfs_url = os.environ.get("HDFS_URL", "hdfs://namenode:9000")
    builder = (
        builder
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Ne pas définir fs.defaultFS globalement : on utilise des URLs
        # explicites pour HDFS et file:// pour les outputs locaux
        .config("spark.hadoop.fs.hdfs.impl",
                "org.apache.hadoop.hdfs.DistributedFileSystem")
        .config("spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.LocalFileSystem")
    )
    return builder.getOrCreate()


# ── Readers ────────────────────────────────────────────────────────────────────

def read_businesses(spark: SparkSession, path: str) -> DataFrame:
    df = spark.read.option("header", "true").csv(path)
    cols = [c for c in ["business_id", "name", "categories", "city", "state",
                         "stars", "review_count"] if c in df.columns]
    return df.select(cols)


def read_reviews(spark: SparkSession, path: str) -> DataFrame:
    df = spark.read.option("header", "true").csv(path)
    cols = [c for c in ["review_id", "user_id", "business_id", "stars", "date"]
            if c in df.columns]
    df = df.select(cols).withColumnRenamed("stars", "rating")
    return df.withColumn("rating", df["rating"].cast(IntegerType()))


def read_users(spark: SparkSession, path: str) -> DataFrame:
    df = spark.read.option("header", "true").csv(path)
    cols = [c for c in ["user_id", "name", "review_count", "average_stars"]
            if c in df.columns]
    return df.select(cols)


# ── Preprocessing steps ────────────────────────────────────────────────────────

def filter_health_medical(businesses: DataFrame) -> DataFrame:
    return businesses.filter(
        F.col("categories").isNotNull() &
        (F.lower(F.col("categories")).contains("health") |
         F.lower(F.col("categories")).contains("medical"))
    )


def join_and_clean(reviews: DataFrame, businesses: DataFrame,
                   rating_thresh: int = 3) -> DataFrame:
    joined = reviews.join(
        businesses.select("business_id").distinct(),
        on="business_id", how="inner"
    )
    return joined.filter(F.col("rating") >= rating_thresh).dropna(
        subset=["user_id", "business_id", "rating"]
    )


def encode_ids(df: DataFrame) -> tuple[DataFrame, dict]:
    """StringIndexer encode user_id → user_idx, business_id → item_idx."""
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx",
                                 handleInvalid="skip")
    item_indexer = StringIndexer(inputCol="business_id", outputCol="item_idx",
                                 handleInvalid="skip")

    user_model = user_indexer.fit(df)
    df = user_model.transform(df)
    item_model = item_indexer.fit(df)
    df = item_model.transform(df)

    df = df.withColumn("user_idx", F.col("user_idx").cast(IntegerType()))
    df = df.withColumn("item_idx", F.col("item_idx").cast(IntegerType()))

    meta = {
        "n_users": len(user_model.labels),
        "n_items": len(item_model.labels),
        "user_labels": user_model.labels[:100],   # first 100 for debug
        "item_labels": item_model.labels[:100],
    }
    return df, meta


def train_val_test_split(df: DataFrame,
                         ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
                         ) -> DataFrame:
    """Temporal split: sort by date, assign split column."""
    df = df.withColumn(
        "date_ts",
        F.to_timestamp(F.col("date")).cast("long")
    )
    # Use row_number over ordered date as proxy for temporal ordering
    from pyspark.sql.window import Window
    w = Window.orderBy("date_ts")
    df = df.withColumn("row_num", F.row_number().over(w))
    total = df.count()
    train_end = int(total * ratios[0])
    val_end   = int(total * (ratios[0] + ratios[1]))

    df = df.withColumn(
        "split",
        F.when(F.col("row_num") <= train_end, F.lit("train"))
         .when(F.col("row_num") <= val_end,   F.lit("val"))
         .otherwise(F.lit("test"))
    )
    return df.drop("row_num", "date_ts")


# ── Writers ────────────────────────────────────────────────────────────────────

def _local_path(output_dir: str, filename: str) -> str:
    """Retourne un chemin file:// pour forcer l'écriture locale (pas HDFS)."""
    return "file://" + os.path.join(output_dir, filename)


def write_edges(df: DataFrame, output_dir: str) -> None:
    edges = df.select("user_idx", "item_idx", "rating", "date", "split")
    edges.coalesce(4).write.mode("overwrite").parquet(
        _local_path(output_dir, "edges.parquet")
    )


def write_businesses(df: DataFrame, businesses: DataFrame,
                     output_dir: str) -> None:
    item_map = df.select("business_id", "item_idx").distinct()
    biz_out  = businesses.join(item_map, on="business_id", how="inner")
    biz_out  = biz_out.select("item_idx", "categories", "business_id")
    biz_out.coalesce(1).write.mode("overwrite").parquet(
        _local_path(output_dir, "businesses.parquet")
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size-tag",   default="full")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Max reviews to keep (None = all). Applied after join.")
    parser.add_argument("--rating-thresh", type=int, default=3)
    parser.add_argument("--master",     default=None)
    args = parser.parse_args()

    timings: dict[str, float] = {}
    spark = get_spark(f"GNN-Preprocess-{args.size_tag}", args.master)
    spark.sparkContext.setLogLevel("WARN")

    # ── t_load ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    hdfs_url = os.environ.get("HDFS_URL", "hdfs://namenode:9000")
    # URLs HDFS explicites pour la lecture
    biz_path = f"{hdfs_url}{args.input_dir}/yelp_academic_dataset_business_healthandmedical.csv"
    rev_path = f"{hdfs_url}{args.input_dir}/yelp_academic_dataset_review_healthandmedical.csv"
    usr_path = f"{hdfs_url}{args.input_dir}/yelp_academic_dataset_user_healthandmedical.csv"

    businesses = read_businesses(spark, biz_path)
    reviews    = read_reviews(spark, rev_path)
    _users     = read_users(spark, usr_path)
    businesses.cache()
    reviews.cache()
    businesses.count()   # trigger load
    reviews.count()
    timings["t_load"] = round(time.perf_counter() - t0, 3)

    # ── t_filter + join ────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    hm_biz  = filter_health_medical(businesses)
    cleaned = join_and_clean(reviews, hm_biz, args.rating_thresh)

    # Apply row limit AFTER join (Spark samples from real interactions)
    if args.limit is not None:
        cleaned = cleaned.limit(args.limit)
        cleaned.cache()
        cleaned.count()   # materialize the limit

    timings["t_filter"] = round(time.perf_counter() - t1, 3)

    # ── t_graph (encode + edge construction) ───────────────────────────────────
    t2 = time.perf_counter()
    encoded, id_meta = encode_ids(cleaned)
    timings["t_encode"] = round(time.perf_counter() - t2, 3)

    t3 = time.perf_counter()
    with_splits = train_val_test_split(encoded)
    with_splits.cache()
    n_edges = with_splits.count()
    timings["t_graph"] = round(time.perf_counter() - t3, 3)

    # ── Write outputs ──────────────────────────────────────────────────────────
    t4 = time.perf_counter()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_edges(with_splits, args.output_dir)
    write_businesses(with_splits, hm_biz, args.output_dir)
    timings["t_write"] = round(time.perf_counter() - t4, 3)

    timings["t_total_spark"] = round(time.perf_counter() - t0, 3)

    # ── Write meta.json ────────────────────────────────────────────────────────
    split_counts = (
        with_splits.groupBy("split").count()
        .rdd.collectAsMap()
    )
    meta = {
        "size_tag": args.size_tag,
        "n_users":  id_meta["n_users"],
        "n_items":  id_meta["n_items"],
        "n_edges":  n_edges,
        "split_counts": split_counts,
        "timings":  timings,
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Spark] Preprocessing done — {args.size_tag}")
    print(f"  n_users={id_meta['n_users']}  n_items={id_meta['n_items']}  n_edges={n_edges}")
    print(f"  timings: {timings}")
    spark.stop()


if __name__ == "__main__":
    main()
