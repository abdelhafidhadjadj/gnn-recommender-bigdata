"""
GAT recommender — supports both full-batch and mini-batch (Phase 4).

concat=False throughout so every layer keeps emb_dim — required for
residual connections and consistent mini-batch indexing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT_Recommender(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 64, heads: int = 4,
                 dropout: float = 0.1, n_layers: int = 1, use_residual: bool = True):
        super().__init__()
        self.embeddings   = nn.Embedding(num_nodes, emb_dim)
        self.convs         = nn.ModuleList([
            GATConv(emb_dim, emb_dim, heads=heads, dropout=dropout, concat=False)
            for _ in range(n_layers)
        ])
        self.dropout      = nn.Dropout(dropout)
        self.n_layers     = n_layers
        self.use_residual = use_residual
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, edge_index: torch.Tensor,
                n_id: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            edge_index: (2, E) — subgraph edges.
            n_id:       (N,)  — global node indices; None for full-batch.
        """
        x = self.embeddings(n_id) if n_id is not None else self.embeddings.weight

        for i, conv in enumerate(self.convs):
            h    = conv(x, edge_index)
            last = (i == len(self.convs) - 1)
            if not last:
                h = F.elu(h)
                h = self.dropout(h)
            if self.use_residual:
                h = h + x
            x = h
        return x

    def predict(self, user_indices: torch.Tensor, item_indices: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        all_emb = self.forward(edge_index)
        return (all_emb[user_indices] * all_emb[item_indices]).sum(dim=1)
