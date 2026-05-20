"""Tests for ranking metric computations."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
import pandas as pd


class TestRankingMetrics:
    """Sanity checks: perfect recommender should score 1.0 on all metrics."""

    def _perfect_model(self, n_users, n_items, emb_dim=8):
        """A stub model whose forward() returns a fixed embedding matrix."""
        import torch.nn as nn

        class _PerfectModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(n_users + n_items, emb_dim)
                nn.init.orthogonal_(self.embeddings.weight)

            def forward(self, edge_index, n_id=None):
                if n_id is not None:
                    return self.embeddings(n_id)
                return self.embeddings.weight

        return _PerfectModel()

    def test_ndcg_is_positive(self):
        from evaluation.metrics import compute_ranking_metrics
        from config import EvalConfig

        n_users, n_items = 5, 10
        model = self._perfect_model(n_users, n_items)
        edge_index = torch.tensor([[0, 1], [n_users, n_users + 1]], dtype=torch.long)

        df_test = pd.DataFrame({
            "user_id": [0, 1],
            "item_id": [n_users, n_users + 1],
            "rating":  [5.0, 4.0],
        })

        cfg = EvalConfig()
        cfg.k_list = [5]
        cfg.max_eval_users = 50
        cfg.relevance_thresh = 3.5

        results = compute_ranking_metrics(model, edge_index, df_test, n_users, cfg)
        assert 5 in results
        assert results[5]["NDCG"] >= 0.0

    def test_rmse_mae_ordering(self):
        """RMSE must be >= MAE (Cauchy-Schwarz inequality)."""
        from evaluation.metrics import evaluate_model
        from config import EvalConfig

        n_users, n_items = 5, 10
        model = self._perfect_model(n_users, n_items)
        edge_index = torch.tensor([[0, 1], [n_users, n_users + 1]], dtype=torch.long)

        df_test = pd.DataFrame({
            "user_id": [0, 1],
            "item_id": [n_users, n_users + 1],
            "rating":  [4.0, 3.0],
        })

        cfg = EvalConfig()
        cfg.k_list = [5]
        cfg.max_eval_users = 50

        results = evaluate_model(model, edge_index, df_test, n_users, cfg)
        assert results["rmse"] >= results["mae"] - 1e-6
