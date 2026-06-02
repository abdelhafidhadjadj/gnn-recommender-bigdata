"""
Standard (non-Spark) preprocessing — pandas only.

Reads CSV files directly from the local mount, applies the same logic
as preprocessing_spark.py, writes the same parquet format:

  {output_dir}/edges.parquet
  {output_dir}/businesses.parquet
  {output_dir}/meta.json

Used for the standard baseline (n_partitions=1, no big-data tools).

Usage (inside trainer container):
  python benchmark/preprocess_pandas.py \\
    --input-dir /workspace/data/medium \\
    --output-dir /workspace/processed/standard/1k \\
    --size-tag 1k \\
    --limit 1000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


LIMIT_MAP = {
    "1k":   1_000,
    "5k":   5_000,
    "10k":  10_000,
    "full": None,
}


def read_csvs(input_dir: str):
    d = Path(input_dir)
    biz = pd.read_csv(d / "yelp_academic_dataset_business_healthandmedical.csv",
                      usecols=lambda c: c in
                      ["business_id", "name", "categories", "stars", "review_count"])
    rev = pd.read_csv(d / "yelp_academic_dataset_review_healthandmedical.csv",
                      usecols=lambda c: c in
                      ["review_id", "user_id", "business_id", "stars", "date"])
    rev = rev.rename(columns={"stars": "rating"})
    return biz, rev


def filter_health_medical(biz: pd.DataFrame) -> pd.DataFrame:
    cats = biz["categories"].fillna("")
    mask = (cats.str.lower().str.contains("health") |
            cats.str.lower().str.contains("medical"))
    return biz[mask].copy()


def join_and_clean(rev: pd.DataFrame, hm_biz: pd.DataFrame,
                   rating_thresh: int = 3) -> pd.DataFrame:
    joined = rev.merge(hm_biz[["business_id"]], on="business_id", how="inner")
    joined = joined[joined["rating"] >= rating_thresh].dropna(
        subset=["user_id", "business_id", "rating"]
    )
    return joined.reset_index(drop=True)


def encode_ids(df: pd.DataFrame):
    user_enc = LabelEncoder().fit(df["user_id"])
    item_enc = LabelEncoder().fit(df["business_id"])
    df = df.copy()
    df["user_idx"] = user_enc.transform(df["user_id"]).astype(np.int32)
    df["item_idx"] = item_enc.transform(df["business_id"]).astype(np.int32)
    meta = {
        "n_users": len(user_enc.classes_),
        "n_items": len(item_enc.classes_),
    }
    return df, meta


def temporal_split(df: pd.DataFrame,
                   ratios=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    df = df.copy()
    df["date_ts"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date_ts").reset_index(drop=True)
    n = len(df)
    train_end = int(n * ratios[0])
    val_end   = int(n * (ratios[0] + ratios[1]))
    splits = np.array(["train"] * n, dtype=object)
    splits[train_end:val_end] = "val"
    splits[val_end:]          = "test"
    df["split"] = splits
    return df.drop(columns=["date_ts"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",    required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--size-tag",     default="full")
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--rating-thresh", type=int, default=3)
    args = parser.parse_args()

    limit = args.limit or LIMIT_MAP.get(args.size_tag)

    timings: dict[str, float] = {}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── t_load ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    biz, rev = read_csvs(args.input_dir)
    timings["t_load"] = round(time.perf_counter() - t0, 3)

    # ── t_filter ───────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    hm_biz  = filter_health_medical(biz)
    cleaned = join_and_clean(rev, hm_biz, args.rating_thresh)
    if limit is not None:
        cleaned = cleaned.head(limit).copy()
    timings["t_filter"] = round(time.perf_counter() - t1, 3)

    # ── t_encode ───────────────────────────────────────────────────────────────
    t2 = time.perf_counter()
    encoded, id_meta = encode_ids(cleaned)
    timings["t_encode"] = round(time.perf_counter() - t2, 3)

    # ── t_graph (split) ────────────────────────────────────────────────────────
    t3 = time.perf_counter()
    with_splits = temporal_split(encoded)
    n_edges = len(with_splits)
    timings["t_graph"] = round(time.perf_counter() - t3, 3)

    # ── t_write ────────────────────────────────────────────────────────────────
    t4 = time.perf_counter()

    edges_out = with_splits[["user_idx", "item_idx", "rating", "date", "split"]]
    edges_out.to_parquet(out_dir / "edges.parquet", index=False)

    item_map = with_splits[["business_id", "item_idx"]].drop_duplicates()
    biz_out  = hm_biz.merge(item_map, on="business_id", how="inner")
    biz_out  = biz_out[["item_idx", "categories", "business_id"]]
    biz_out.to_parquet(out_dir / "businesses.parquet", index=False)

    timings["t_write"] = round(time.perf_counter() - t4, 3)
    timings["t_total_pandas"] = round(sum(timings.values()), 3)

    split_counts = with_splits["split"].value_counts().to_dict()
    meta = {
        "size_tag":    args.size_tag,
        "n_users":     id_meta["n_users"],
        "n_items":     id_meta["n_items"],
        "n_edges":     n_edges,
        "split_counts": split_counts,
        "timings":     timings,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[pandas] Preprocessing done — {args.size_tag}")
    print(f"  n_users={id_meta['n_users']}  n_items={id_meta['n_items']}"
          f"  n_edges={n_edges}")
    print(f"  timings: {timings}")


if __name__ == "__main__":
    main()
