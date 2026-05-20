"""Tests for graph construction and embedding extension (incremental mode)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pandas as pd

from config import Config, DataConfig
from data.graph_builder import build_ui_edges
from training.incremental import extend_model_embeddings
from models import build_model


# ── build_ui_edges ────────────────────────────────────────────────────────────

class TestBuildUiEdges:
    def _make_df(self, n_users=5, n_items=4, n_ratings=10):
        rng = __import__("numpy").random.default_rng(0)
        return pd.DataFrame({
            "user_id": rng.integers(0, n_users, n_ratings),
            "item_id": rng.integers(n_users, n_users + n_items, n_ratings),
            "rating":  rng.uniform(1, 5, n_ratings),
        })

    def test_returns_two_tensors(self):
        df = self._make_df()
        cfg = DataConfig(rating_thresh=1)
        edge_index, edge_weight = build_ui_edges(df, cfg)
        assert edge_index.shape[0] == 2
        assert edge_weight.shape[0] == edge_index.shape[1]

    def test_edge_index_dtype(self):
        df = self._make_df()
        cfg = DataConfig(rating_thresh=1)
        edge_index, _ = build_ui_edges(df, cfg)
        assert edge_index.dtype == torch.long

    def test_directed_user_to_item(self):
        """build_ui_edges produces directed user→item edges (one per interaction)."""
        df = pd.DataFrame({
            "user_id": [0, 1],
            "item_id": [5, 6],
            "rating":  [5.0, 4.0],
        })
        cfg = DataConfig(rating_thresh=1)
        edge_index, _ = build_ui_edges(df, cfg)
        # 2 interactions → 2 edges; sources are users, destinations are items
        assert edge_index.shape[1] == 2
        assert set(edge_index[0].tolist()) <= {0, 1}   # sources are users
        assert set(edge_index[1].tolist()) <= {5, 6}   # destinations are items

    def test_rating_threshold_filters(self):
        """Interactions below rating_thresh must be excluded."""
        df = pd.DataFrame({
            "user_id": [0, 1, 2],
            "item_id": [3, 3, 3],
            "rating":  [1.0, 2.0, 5.0],
        })
        cfg_strict = DataConfig(rating_thresh=4)
        ei, _ = build_ui_edges(df, cfg_strict)
        # Only the 5-star interaction should survive → 1 directed edge
        assert ei.shape[1] == 1


# ── extend_model_embeddings ───────────────────────────────────────────────────

class TestExtendModelEmbeddings:
    def _base_model(self, n_users=10, n_items=8, emb_dim=16):
        return build_model("sage", n_users + n_items, emb_dim, 0.0, 1)

    def test_no_extension_returns_original(self):
        model = self._base_model()
        cfg = Config()
        cfg.model_type = "sage"
        cfg.model.emb_dim = 16
        cfg.model.dropout = 0.0
        cfg.model.gat_heads = 1
        cfg.model.n_layers = 1
        cfg.model.use_residual = False

        new_model, n_u, n_i = extend_model_embeddings(
            model, n_new_users=0, n_new_items=0, n_old_users=10, cfg=cfg
        )
        assert new_model is model            # unchanged reference
        assert n_u == 10
        assert n_i == 8

    def test_user_embeddings_preserved(self):
        n_old_u, n_old_i = 10, 8
        model = self._base_model(n_old_u, n_old_i)
        old_user0 = model.embeddings.weight.data[0].clone()

        cfg = Config()
        cfg.model_type = "sage"
        cfg.model.emb_dim = 16
        cfg.model.dropout = 0.0
        cfg.model.gat_heads = 1
        cfg.model.n_layers = 1
        cfg.model.use_residual = False

        new_model, _, _ = extend_model_embeddings(
            model, n_new_users=3, n_new_items=2, n_old_users=n_old_u, cfg=cfg
        )
        assert torch.allclose(new_model.embeddings.weight.data[0], old_user0), \
            "Old user embedding was modified"

    def test_item_embeddings_shifted(self):
        n_old_u, n_old_i = 10, 8
        model = self._base_model(n_old_u, n_old_i)
        old_item0 = model.embeddings.weight.data[n_old_u].clone()

        cfg = Config()
        cfg.model_type = "sage"
        cfg.model.emb_dim = 16
        cfg.model.dropout = 0.0
        cfg.model.gat_heads = 1
        cfg.model.n_layers = 1
        cfg.model.use_residual = False

        n_new_u = 3
        new_model, n_u_ext, _ = extend_model_embeddings(
            model, n_new_users=n_new_u, n_new_items=2,
            n_old_users=n_old_u, cfg=cfg
        )
        # Old item 0 should now be at position n_u_ext (shifted by n_new_u)
        assert torch.allclose(
            new_model.embeddings.weight.data[n_u_ext], old_item0
        ), "Old item embedding not correctly shifted"

    def test_table_size_correct(self):
        n_old_u, n_old_i = 10, 8
        model = self._base_model(n_old_u, n_old_i)
        cfg = Config()
        cfg.model_type = "sage"
        cfg.model.emb_dim = 16
        cfg.model.dropout = 0.0
        cfg.model.gat_heads = 1
        cfg.model.n_layers = 1
        cfg.model.use_residual = False

        new_model, n_u, n_i = extend_model_embeddings(
            model, n_new_users=5, n_new_items=3, n_old_users=n_old_u, cfg=cfg
        )
        assert n_u == 15
        assert n_i == 11
        assert new_model.embeddings.num_embeddings == 26
