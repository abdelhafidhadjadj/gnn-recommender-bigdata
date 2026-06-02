"""
DataLoader factories for mini-batch GNN training (Phase 4).

graph_mode = "full_batch"
    Caller uses full_edge_index directly; no loader needed.
    fast for small graphs (< 20 % of free VRAM).

graph_mode = "neighbor_loader"
    Returns a LinkNeighborLoader from PyTorch Geometric.
    Samples k-hop subgraphs for each supervision (positive) edge.
    Scales to graphs that do not fit in VRAM.

Key design choice for BPR in mini-batch mode
─────────────────────────────────────────────
  Positive embeddings  : GNN-propagated (full neighborhood context, from batch)
  Negative embeddings  : raw from embedding table (no GNN propagation)

This asymmetry is a known, widely-used approximation in mini-batch CF training
(used in NGCF, KGCN, many others). At evaluation time, full GNN propagation
is used for all nodes, so the approximation only affects training gradients.

Distributed note (Phase 5):
  make_train_loader accepts rank / world_size. When world_size > 1, a
  DistributedSampler wrapper will be added here. For now they are
  ignored — single-process only.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from utils.hardware import HardwareProfile, recommended_workers


def neighbor_sampler_available() -> bool:
    """
    Return True if pyg-lib or torch-sparse is installed.

    LinkNeighborLoader requires one of these C++ backends to run the
    neighbour-sampling kernel.  Without them the loader can be constructed
    but fails at the first __next__() call.

    On the CPU dev laptop this will be False; on the production server
    (which has a full PyG + CUDA installation) it will be True.
    """
    for pkg in ("pyg_lib", "torch_sparse"):
        try:
            __import__(pkg)
            return True
        except ImportError:
            pass
    return False


# ── Graph data wrapper ────────────────────────────────────────────────────────

def build_pyg_data(edge_index: torch.Tensor, num_nodes: int) -> Data:
    """
    Wrap edge_index in a PyG Data object for use with LinkNeighborLoader.

    The tensor is kept on CPU — the loader automatically moves each mini-batch
    to the target device.
    """
    return Data(
        num_nodes  = num_nodes,
        edge_index = edge_index.cpu(),
    )


# ── Loader factory ────────────────────────────────────────────────────────────

def make_train_loader(
    pyg_data:          Data,
    train_edge_index:  torch.Tensor,
    batch_size:        int,
    num_neighbors:     list[int],
    profile:           HardwareProfile,
) -> LinkNeighborLoader:
    # rank / world_size added in Phase 5 (DistributedSampler for DDP)
    """
    Build a LinkNeighborLoader for mini-batch BPR training.

    Args:
        pyg_data:          PyG Data with the full training message-passing graph.
        train_edge_index:  (2, N_train) supervision edges (user -> item,
                           unidirectional). These are the edges whose embeddings
                           BPR will train on.
        batch_size:        Number of supervision edges per mini-batch.
        num_neighbors:     Neighbor sampling depth per GNN layer, e.g. [15, 10].
        profile:           HardwareProfile — drives num_workers + pin_memory.
        rank, world_size:  DDP rank/size (Phase 5 will add DistributedSampler).

    Returns:
        A ready-to-iterate LinkNeighborLoader.
    """
    num_workers = recommended_workers(profile)
    pin_memory  = torch.cuda.is_available()

    loader_kwargs: dict = dict(
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        persistent_workers = (num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    # disjoint=True évite le data leakage mais requiert pyg-lib.
    # Fallback : on passe un graphe MP sans les arêtes de supervision
    # → même effet sans pyg-lib.
    try:
        return LinkNeighborLoader(
            data               = pyg_data,
            num_neighbors      = num_neighbors,
            edge_label_index   = train_edge_index.cpu(),
            neg_sampling_ratio = 0.0,
            batch_size         = batch_size,
            shuffle            = True,
            disjoint           = True,          # requiert pyg-lib
            **loader_kwargs,
        )
    except (TypeError, ValueError):
        # pyg-lib absent — disjoint non supporté.
        # Solution : retenir 80 % des arêtes pour le message-passing,
        # utiliser les 20 % restants comme arêtes de supervision.
        # Les arêtes de supervision ne sont donc jamais dans le graphe MP → pas de leakage.
        n_total   = pyg_data.num_nodes
        n_sup     = train_edge_index.size(1)
        perm      = torch.randperm(n_sup)
        n_mp      = int(0.80 * n_sup)
        mp_idx    = perm[:n_mp]                      # 80 % pour MP
        sup_idx   = perm[n_mp:]                      # 20 % pour supervision BPR

        mp_edges  = train_edge_index[:, mp_idx]
        sup_edges = train_edge_index[:, sup_idx]

        # Graphe MP bidirectionnel sur les 80 % d'arêtes
        mp_edge_index = torch.cat([mp_edges, mp_edges.flip(0)], dim=1)

        from torch_geometric.data import Data
        mp_data = Data(edge_index=mp_edge_index.cpu(), num_nodes=n_total)

        return LinkNeighborLoader(
            data               = mp_data,
            num_neighbors      = num_neighbors,
            edge_label_index   = sup_edges.cpu(),    # supervision = 20 % sans leakage
            neg_sampling_ratio = 0.0,
            batch_size         = batch_size,
            shuffle            = True,
            **loader_kwargs,
        )
