"""
Graph construction — leakage-free design.

Key fixes vs. original:
  F2: build_graph receives pre-split TRAIN edges only (no test leakage).
  F5: business_df_ordered must already be sorted to match LabelEncoder order
      so that SBERT embedding index i == encoded item id i.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from config import GraphConfig, DataConfig


# ---------------------------------------------------------------------------
# UI edge construction  (call this BEFORE splitting)
# ---------------------------------------------------------------------------

def load_edge_csv(df: pd.DataFrame, src_col: str, dst_col: str,
                  rating_col: str, rating_thresh: int):
    edge_attr = torch.from_numpy(df[rating_col].values).view(-1, 1).to(torch.long) >= rating_thresh
    edge_index = [[], []]
    edge_values = []
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(df[src_col].iloc[i])
            edge_index[1].append(df[dst_col].iloc[i])
            edge_values.append(df[rating_col].iloc[i])
    return edge_index, edge_values


def build_ui_edges(review_df: pd.DataFrame, cfg: DataConfig):
    """Build ALL user-item edges before splitting."""
    edge_index_list, edge_values_list = load_edge_csv(
        review_df, 'user_id', 'item_id', 'rating', cfg.rating_thresh
    )
    ui_edge_index = torch.LongTensor(edge_index_list)
    ui_edge_values = torch.tensor(edge_values_list, dtype=torch.float)
    return ui_edge_index, ui_edge_values


# ---------------------------------------------------------------------------
# Temporal edge weights (train interactions only)
# ---------------------------------------------------------------------------

def build_temporal_weights(review_df_full: pd.DataFrame,
                           edge_values: torch.Tensor,
                           cfg: GraphConfig) -> torch.Tensor:
    if 'date' in review_df_full.columns:
        today = pd.Timestamp('today')
        days_since = (
            (today - pd.to_datetime(review_df_full['date'], errors='coerce'))
            .dt.days.fillna(365).values.astype(float)
        )
        time_decay = np.exp(-days_since / cfg.time_decay_days)
        stars_norm = (review_df_full['rating'].values - 1.0) / 4.0
        return torch.tensor((stars_norm * time_decay).astype(np.float32), dtype=torch.float)
    return edge_values / 5.0


# ---------------------------------------------------------------------------
# SBERT item embeddings
# ---------------------------------------------------------------------------

def build_item_embeddings(business_df_ordered: pd.DataFrame,
                          device: torch.device,
                          cfg: GraphConfig) -> np.ndarray:
    """
    business_df_ordered MUST be sorted to match LabelEncoder order
    (caller: business_df.set_index('business_id').loc[item_enc.classes_].reset_index())
    so that embedding row i == encoded item i.
    """
    sbert = SentenceTransformer(cfg.sbert_model)
    sbert = sbert.to(device)
    categories = business_df_ordered['categories'].fillna('').tolist()
    embeddings = sbert.encode(
        categories,
        batch_size=cfg.sbert_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=str(device),
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# SBERT warm-start for item embedding rows in the user-item graph
# ---------------------------------------------------------------------------

def build_sbert_item_projections(
    business_df_ordered: pd.DataFrame,
    emb_dim: int,
    embed_device: torch.device,
    cfg: GraphConfig,
    seed: int = 42,
) -> torch.Tensor:
    """
    Run SBERT on item category text and project from 768-dim to emb_dim.

    Returns a CPU float32 tensor of shape (n_items, emb_dim) that can be
    used to warm-start item embedding rows in nn.Embedding:

        model.embeddings.weight.data[n_users:] = build_sbert_item_projections(...)

    Items start content-aware (semantically grouped by category).
    BPR fine-tunes all rows during training.

    Args:
        business_df_ordered: business DataFrame sorted to LabelEncoder order.
        emb_dim:             target embedding dimension (matches model emb_dim).
        embed_device:        device for SBERT encoding (GPU speeds this up).
        cfg:                 GraphConfig (sbert_model, sbert_batch_size).
        seed:                RNG seed for the projection matrix (reproducibility).
    """
    # 1. Encode categories with SBERT  →  (n_items, 768)
    raw_emb = build_item_embeddings(business_df_ordered, embed_device, cfg)
    raw_tensor = torch.tensor(raw_emb, dtype=torch.float32)   # CPU

    # 2. Project 768 → emb_dim with a fixed-seed linear layer
    torch.manual_seed(seed)
    proj = torch.nn.Linear(raw_tensor.shape[1], emb_dim, bias=False)
    torch.nn.init.xavier_uniform_(proj.weight)

    with torch.no_grad():
        projected = proj(raw_tensor)          # (n_items, emb_dim)

    return projected.cpu()                    # always return on CPU


def warm_start_item_embeddings(
    model: torch.nn.Module,
    item_projections: torch.Tensor,
    n_users: int,
) -> None:
    """
    Copy SBERT-projected vectors into the item rows of model.embeddings.
    User rows are left unchanged (Xavier random).

    Safe to call after model.to(device) — copies to whichever device
    the embedding is on.
    """
    device = model.embeddings.weight.device
    with torch.no_grad():
        model.embeddings.weight.data[n_users:] = item_projections.to(device)


# ---------------------------------------------------------------------------
# Item-item FAISS edges
# ---------------------------------------------------------------------------

def build_item_item_edges(item_embeddings: np.ndarray, n_users: int,
                          k: int, device: torch.device) -> torch.Tensor:
    emb = item_embeddings.copy()
    faiss.normalize_L2(emb)

    n_items, dim = emb.shape
    if device.type == 'cuda':
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, dim)
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(emb)
    _, indices = index.search(emb, k + 1)  # +1 to exclude self

    src_list, dst_list = [], []
    for i in range(n_items):
        for j in indices[i]:
            if j != i and j < n_items:
                src_list.append(i + n_users)
                dst_list.append(j + n_users)

    ii_edge_index = torch.tensor(
        [src_list + dst_list, dst_list + src_list], dtype=torch.long
    )
    return ii_edge_index


# ---------------------------------------------------------------------------
# Unified training graph  (F2: train edges only)
# ---------------------------------------------------------------------------

def build_graph(train_edge_index: torch.Tensor,
                train_edge_values: torch.Tensor,
                train_review_df_full: pd.DataFrame,
                business_df_ordered: pd.DataFrame | None,
                n_users: int,
                embed_device: torch.device,
                graph_device: torch.device,
                graph_cfg: GraphConfig):
    """
    Build the unified training graph from TRAINING interactions only.

    Args:
        train_edge_index      – (2, N_train)
        train_edge_values     – (N_train,)
        train_review_df_full  – training subset of review_df_full (with 'date')
        business_df_ordered   – business_df in LabelEncoder order; may be None
                                when graph_cfg.use_item_item_edges is False
        n_users               – number of users
        embed_device          – device for SBERT / FAISS
        graph_device          – device for the returned edge tensors
        graph_cfg             – GraphConfig (use_item_item_edges flag lives here)

    Returns:
        full_edge_index  – (2, E) on graph_device
        full_edge_weight – (E,)  on graph_device
    """
    temporal_weights = build_temporal_weights(
        train_review_df_full, train_edge_values, graph_cfg
    )

    # Bidirectional UI edges (train only)
    ui_bidir   = torch.cat([
        train_edge_index,
        torch.stack([train_edge_index[1], train_edge_index[0]])
    ], dim=1)
    ui_bidir_w = torch.cat([temporal_weights, temporal_weights])

    if graph_cfg.use_item_item_edges:
        if business_df_ordered is None:
            raise ValueError(
                "build_graph: business_df_ordered must not be None "
                "when use_item_item_edges=True"
            )
        item_embeddings = build_item_embeddings(business_df_ordered, embed_device, graph_cfg)
        ii_edge_index   = build_item_item_edges(
            item_embeddings, n_users, graph_cfg.k_neighbors, embed_device
        )
        ii_w            = torch.ones(ii_edge_index.shape[1])
        full_edge_index  = torch.cat([ui_bidir, ii_edge_index], dim=1).to(graph_device)
        full_edge_weight = torch.cat([ui_bidir_w, ii_w]).to(graph_device)
    else:
        # UI-only graph — fast, no SBERT, no over-smoothing from content edges
        full_edge_index  = ui_bidir.to(graph_device)
        full_edge_weight = ui_bidir_w.to(graph_device)

    return full_edge_index, full_edge_weight
