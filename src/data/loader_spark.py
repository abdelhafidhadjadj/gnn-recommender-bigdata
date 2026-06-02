"""
Load preprocessed Spark parquet output into the data structures
expected by the existing trainer (same as preprocessing.preprocess output).

Parquet layout produced by spark/preprocessing_spark.py:
  {processed_dir}/
    edges.parquet       — user_idx, item_idx, rating, date, split
    businesses.parquet  — item_idx, categories, business_id
    meta.json           — n_users, n_items, n_edges, timings

Returns a dict:
  {
    "train_df":     pd.DataFrame(user_id, item_id, rating)  # item_id offset by n_users
    "val_df":       pd.DataFrame(user_id, item_id, rating)
    "test_df":      pd.DataFrame(user_id, item_id, rating)
    "business_df":  pd.DataFrame sorted by item_idx (for SBERT)
    "n_users":      int
    "n_items":      int
    "n_edges":      int
  }
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_from_parquet(processed_dir: str) -> dict:
    root = Path(processed_dir)

    # ── meta ──────────────────────────────────────────────────────────────────
    meta_path = root / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        n_users = meta["n_users"]
        n_items = meta["n_items"]
        n_edges = meta["n_edges"]
    else:
        n_users = n_items = n_edges = None

    # ── edges ─────────────────────────────────────────────────────────────────
    edges_path = root / "edges.parquet"
    edges = pd.read_parquet(edges_path)

    # Apply n_users offset: item_id in trainer space = item_idx + n_users
    if n_users is None:
        n_users = int(edges["user_idx"].max()) + 1
    edges["item_id"] = edges["item_idx"] + n_users

    def _split_df(split_name: str) -> pd.DataFrame:
        sub = edges[edges["split"] == split_name].copy()
        return sub.rename(columns={"user_idx": "user_id"})[
            ["user_id", "item_id", "rating"]
        ].reset_index(drop=True)

    train_df = _split_df("train")
    val_df   = _split_df("val")
    test_df  = _split_df("test")

    # Also expose a full_df with date for temporal edge weights
    train_full = edges[edges["split"] == "train"].copy()
    train_full = train_full.rename(columns={"user_idx": "user_id",
                                             "item_idx": "item_id_raw"})
    train_full["item_id"] = train_full["item_id_raw"] + n_users

    if n_items is None:
        n_items = int(edges["item_idx"].max()) + 1
    if n_edges is None:
        n_edges = len(edges)

    # ── businesses (for SBERT) ────────────────────────────────────────────────
    biz_path = root / "businesses.parquet"
    if biz_path.exists():
        business_df = pd.read_parquet(biz_path)
        business_df = business_df.sort_values("item_idx").reset_index(drop=True)
        # Ensure 'categories' column exists (SBERT needs it)
        if "categories" not in business_df.columns:
            business_df["categories"] = ""
    else:
        # Fallback: empty df with correct structure
        business_df = pd.DataFrame({"item_idx": range(n_items),
                                    "categories": [""] * n_items,
                                    "business_id": [""] * n_items})

    return {
        "train_df":    train_df,
        "train_full":  train_full,   # with 'date' column for temporal weights
        "val_df":      val_df,
        "test_df":     test_df,
        "business_df": business_df,
        "n_users":     n_users,
        "n_items":     n_items,
        "n_edges":     n_edges,
    }
