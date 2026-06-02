"""
Inference utilities for the GNN Recommender demo.
Provides model loading, embedding computation, and recommendation functions.
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models import build_model
from utils.checkpoint import CheckpointManager


# ── Minimal edge builder (avoids faiss/SBERT imports from graph_builder) ──────

def _build_ui_edges(df: pd.DataFrame, rating_thresh: float = 3.0) -> torch.Tensor:
    """Build bidirectional user-item edge_index from an interaction DataFrame."""
    mask = df["rating"].values >= rating_thresh
    users = df["user_id"].values[mask].astype(int)
    items = df["item_id"].values[mask].astype(int)
    # bidirectional
    src = np.concatenate([users, items])
    dst = np.concatenate([items, users])
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, device: torch.device = torch.device("cpu")) -> dict:
    return CheckpointManager.load(ckpt_path, device)


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    model = CheckpointManager.build_model_from_ckpt(ckpt, device, build_model)
    model.eval()
    return model


# ── Graph reconstruction from checkpoint ─────────────────────────────────────

def build_edge_index_from_ckpt(ckpt: dict) -> tuple[torch.Tensor | None, int, int]:
    """Rebuild edge_index from the training interactions stored in the checkpoint."""
    extra = ckpt.get("extra", {})
    n_users = extra.get("n_users")
    n_items = extra.get("n_items")
    train_inter = extra.get("train_interactions", {})

    # Fallback 1: derive from encoders (most accurate)
    user_enc = ckpt.get("user_encoder")
    item_enc = ckpt.get("item_encoder")
    mc = ckpt["model_config"]

    if n_users is None:
        if user_enc is not None and hasattr(user_enc, "classes_"):
            n_users = len(user_enc.classes_)
        else:
            n_users = mc["num_nodes"] // 2

    if n_items is None:
        if item_enc is not None and hasattr(item_enc, "classes_"):
            n_items = len(item_enc.classes_)
        else:
            n_items = mc["num_nodes"] - n_users

    if not train_inter:
        return None, n_users, n_items

    df = pd.DataFrame({
        "user_id": train_inter["user_ids"].astype(int),
        "item_id": train_inter["item_ids_local"].astype(int) + n_users,
        "rating":  train_inter["ratings"].astype(float),
    })

    edge_index = _build_ui_edges(df)
    return edge_index, n_users, n_items


# ── Embeddings ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_embeddings(model: torch.nn.Module, edge_index: torch.Tensor) -> np.ndarray:
    embs = model(edge_index)
    return embs.cpu().numpy()


# ── Recommendation ────────────────────────────────────────────────────────────

def recommend(
    user_idx: int,
    embeddings: np.ndarray,
    n_users: int,
    n_items: int,
    seen_item_indices: set | None = None,
    k: int = 10,
) -> list[tuple[int, float]]:
    """
    Return top-K (item_idx, score) pairs for a given user index.
    Excludes items in `seen_item_indices`.
    """
    user_emb  = embeddings[user_idx]
    item_embs = embeddings[n_users: n_users + n_items]
    scores    = item_embs @ user_emb

    if seen_item_indices:
        for idx in seen_item_indices:
            if 0 <= idx < n_items:
                scores[idx] = -np.inf

    top_k = np.argsort(scores)[::-1][:k + len(seen_item_indices or [])]
    result = []
    for i in top_k:
        if len(result) >= k:
            break
        if scores[i] > -np.inf:
            result.append((int(i), float(scores[i])))
    return result


def recommend_cold_start(
    liked_item_indices: list[int],
    embeddings: np.ndarray,
    n_users: int,
    n_items: int,
    k: int = 10,
) -> list[tuple[int, float]]:
    """
    Cold-start: simulate a new user as the average of liked items' embeddings.
    """
    if not liked_item_indices:
        return []

    item_embs = embeddings[n_users: n_users + n_items]
    liked_vecs = np.stack([item_embs[i] for i in liked_item_indices if i < n_items])
    user_proxy = liked_vecs.mean(axis=0)

    scores = item_embs @ user_proxy
    for i in liked_item_indices:
        if i < n_items:
            scores[i] = -np.inf

    top_k = np.argsort(scores)[::-1]
    return [(int(i), float(scores[i])) for i in top_k[:k] if scores[i] > -np.inf]


# ── User history ──────────────────────────────────────────────────────────────

def get_user_history(user_idx: int, ckpt: dict) -> pd.DataFrame:
    """Return training interactions for a given user_idx."""
    extra = ckpt.get("extra", {})
    train_inter = extra.get("train_interactions", {})
    if not train_inter:
        return pd.DataFrame()

    n_users = extra.get("n_users", 0)
    mask = train_inter["user_ids"] == user_idx
    return pd.DataFrame({
        "item_idx": train_inter["item_ids_local"][mask].astype(int),
        "rating":   train_inter["ratings"][mask].astype(float),
    })


# ── Business catalogue ────────────────────────────────────────────────────────

def load_business_df(ckpt: dict, data_dir: str) -> pd.DataFrame:
    """
    Load business metadata from parquet or CSV.
    Falls back to a minimal DataFrame built from item_encoder if no file is found.
    """
    item_enc = ckpt.get("item_encoder")

    # 1. Try parquet (from benchmark preprocessing)
    for candidate in [
        Path(data_dir) / "businesses.parquet",
        Path(data_dir) / "full" / "businesses.parquet",
    ]:
        if candidate.exists():
            try:
                return pd.read_parquet(candidate)
            except Exception:
                pass

    # 2. Try CSV from raw data
    candidates = list(Path(data_dir).rglob("*business*.csv"))
    # try sibling sizes one level up (e.g. data/raw/100k/ when data_dir=data/raw/full)
    raw_root = Path(data_dir).parent
    for size in ["full", "100k", "50k", "40k", "10k", "5k", "1k"]:
        extra = raw_root / size / "yelp_academic_dataset_business_healthandmedical.csv"
        if extra.exists() and extra not in candidates:
            candidates.append(extra)
    # also try two levels up (e.g. data/1k/ when data_dir=data/raw/full)
    data_root = raw_root.parent
    for size in ["full", "100k", "50k", "40k", "10k", "5k", "1k"]:
        extra = data_root / size / "yelp_academic_dataset_business_healthandmedical.csv"
        if extra.exists() and extra not in candidates:
            candidates.append(extra)

    for biz_csv in candidates:
        try:
            df = pd.read_csv(biz_csv, encoding="utf-8", low_memory=False)
            # normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            if "business_id" not in df.columns:
                continue
            keep = [c for c in ["business_id", "name", "categories"] if c in df.columns]
            df = df[keep]
            if "name" not in df.columns:
                df["name"] = df["business_id"]
            if "categories" not in df.columns:
                df["categories"] = "N/A"
            df = df.dropna(subset=["business_id"])
            df["business_id"] = df["business_id"].astype(str).str.strip()
            return df
        except Exception:
            pass

    # 3. Build minimal DataFrame from item encoder
    if item_enc is not None:
        n_items = len(item_enc.classes_)
        return pd.DataFrame({
            "item_idx":    range(n_items),
            "business_id": item_enc.classes_,
            "name":        [f"Business {i}" for i in range(n_items)],
            "categories":  ["N/A"] * n_items,
        })

    return pd.DataFrame(columns=["item_idx", "business_id", "name", "categories"])


def enrich_recommendations(
    recs: list[tuple[int, float]],
    item_enc,
    business_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert (item_idx, score) list to a display DataFrame with business info."""
    rows = []

    # Build lookup dict {business_id: {name, categories}} — handles duplicates
    biz_lookup: dict = {}
    if not business_df.empty and "business_id" in business_df.columns:
        for _, r in business_df.drop_duplicates("business_id").iterrows():
            bid = str(r["business_id"]).strip()
            biz_lookup[bid] = {
                "name":       str(r.get("name", bid) or bid).strip(),
                "categories": str(r.get("categories", "N/A") or "N/A")[:120],
            }

    for rank, (item_idx, score) in enumerate(recs, start=1):
        biz_id = (
            str(item_enc.classes_[item_idx]).strip()
            if item_enc and item_idx < len(item_enc.classes_)
            else f"item_{item_idx}"
        )
        meta  = biz_lookup.get(biz_id, {})
        name  = meta.get("name", biz_id)
        cats  = meta.get("categories", "N/A")

        rows.append({
            "Rang":        rank,
            "Nom":         name,
            "Catégories":  cats,
            "Score":       round(score, 4),
            "business_id": biz_id,
        })
    return pd.DataFrame(rows)


# ── Available checkpoints ─────────────────────────────────────────────────────

def find_checkpoints(root: str | Path = ".") -> list[Path]:
    root = Path(root)
    ckpts = sorted(root.rglob("*_best.pt")) + sorted(root.rglob("*_v*.pt"))
    seen, unique = set(), []
    for p in ckpts:
        if str(p) not in seen:
            seen.add(str(p))
            unique.append(p)
    return unique
