"""
LightGCN — He et al., 2020 (https://arxiv.org/abs/2002.02126)
Supports both full-batch and mini-batch (Phase 4).

Key differences from GraphSAGE / GAT:
  - No feature transformation (W)
  - No non-linear activation
  - Propagation = degree-normalised weighted sum only
  - Final embedding = mean of all layer outputs (including layer 0)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class _LGConv(MessagePassing):
    """Stateless D^{-1/2} A D^{-1/2} aggregation (LightGCN propagation)."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int | None = None) -> torch.Tensor:
        n = num_nodes or x.size(0)
        row, col = edge_index
        deg  = degree(col, num_nodes=n, dtype=x.dtype).clamp(min=1.0)
        norm = (deg[row] * deg[col]).pow(-0.5)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.unsqueeze(-1) * x_j


class LightGCN_Recommender(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes, emb_dim)
        self.n_layers   = n_layers
        self.conv       = _LGConv()
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, edge_index: torch.Tensor,
                n_id: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            edge_index: (2, E) — subgraph or full graph edges.
            n_id:       (N,)  — global node indices; None for full-batch.

        In mini-batch mode, num_nodes for degree computation is inferred from
        x.size(0) (batch size). This is an approximation — degrees are computed
        on the subgraph, not the full graph. Acceptable for training; evaluation
        always uses full-batch (n_id=None).
        """
        x  = self.embeddings(n_id) if n_id is not None else self.embeddings.weight
        xs = [x]
        for _ in range(self.n_layers):
            x = self.conv(x, edge_index)
            xs.append(x)
        return torch.stack(xs, dim=0).mean(dim=0)

    def predict(self, user_indices: torch.Tensor, item_indices: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        all_emb = self.forward(edge_index)
        return (all_emb[user_indices] * all_emb[item_indices]).sum(dim=1)
