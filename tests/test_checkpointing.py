"""Tests for CheckpointManager save / load round-trip."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
import tempfile
from pathlib import Path

from utils.checkpoint import CheckpointManager
from models import build_model
from config import Config


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    return str(tmp_path / "ckpts")


def _make_model(emb_dim=16):
    return build_model("sage", num_nodes=50, emb_dim=emb_dim,
                       dropout=0.0, gat_heads=1)


def test_save_and_load_round_trip(tmp_ckpt_dir):
    cfg = Config()
    cfg.model.emb_dim = 16
    mgr = CheckpointManager(tmp_ckpt_dir, "sage", keep_last_n=3)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    path = mgr.save(
        model=model, optimizer=optimizer,
        epoch=5, val_score=0.42,
        cfg=cfg, num_nodes=50,
        rank=0,
    )
    assert path is not None and path.exists()

    ckpt = CheckpointManager.load(path, torch.device("cpu"))
    assert ckpt["epoch"] == 5
    assert abs(ckpt["val_score"] - 0.42) < 1e-6
    assert "model_state" in ckpt
    assert "model_config" in ckpt


def test_best_checkpoint_written(tmp_ckpt_dir):
    cfg = Config()
    cfg.model.emb_dim = 16
    mgr = CheckpointManager(tmp_ckpt_dir, "sage", keep_last_n=3)
    model = _make_model()
    opt = torch.optim.Adam(model.parameters())

    mgr.save(model=model, optimizer=opt, epoch=1, val_score=0.1,
             cfg=cfg, num_nodes=50, rank=0)
    mgr.save(model=model, optimizer=opt, epoch=2, val_score=0.5,
             cfg=cfg, num_nodes=50, rank=0)

    best = Path(tmp_ckpt_dir) / "sage_best.pt"
    assert best.exists()
    ckpt = CheckpointManager.load(best, torch.device("cpu"))
    assert abs(ckpt["val_score"] - 0.5) < 1e-6


def test_build_model_from_ckpt(tmp_ckpt_dir):
    cfg = Config()
    cfg.model.emb_dim = 16
    mgr = CheckpointManager(tmp_ckpt_dir, "sage", keep_last_n=3)
    model = _make_model()
    opt = torch.optim.Adam(model.parameters())

    path = mgr.save(model=model, optimizer=opt, epoch=1, val_score=0.1,
                    cfg=cfg, num_nodes=50, rank=0)
    ckpt = CheckpointManager.load(path, torch.device("cpu"))
    rebuilt = CheckpointManager.build_model_from_ckpt(
        ckpt, torch.device("cpu"), build_model
    )
    assert rebuilt.embeddings.num_embeddings == 50
