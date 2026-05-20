"""Shared pytest fixtures for the GNN recommender test suite."""
import sys, os
import pytest
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config


@pytest.fixture
def debug_cfg():
    cfg = Config()
    cfg.model.emb_dim = 16
    cfg.train.num_epochs = 2
    cfg.train.min_epochs = 0
    cfg.train.eval_every = 1
    cfg.train.patience = 1
    cfg.graph.use_sbert_item_init = False
    cfg.data.data_dir = "data/test"
    return cfg


@pytest.fixture
def small_review_df():
    return pd.DataFrame({
        "user_id":     [0, 0, 1, 1, 2],
        "item_id":     [3, 4, 3, 5, 4],
        "rating":      [5.0, 4.0, 3.0, 5.0, 4.0],
    })


@pytest.fixture
def device():
    return torch.device("cpu")
