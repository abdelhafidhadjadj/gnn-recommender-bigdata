"""
Reproducible seeding — works in single-process and DDP modes.

DDP strategy: each rank gets `seed + rank` so that
  - Model weights are initialised identically on rank 0 (seed + 0)
    then DDP broadcasts rank-0 weights to all ranks anyway.
  - DataLoader worker shuffling DIFFERS per rank, which is what we want
    (each GPU sees different mini-batches).
"""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int, rank: int = 0, make_deterministic: bool = False) -> None:
    """
    Set all RNG seeds for full reproducibility.

    Args:
        seed:              Master seed value (from Config.seed)
        rank:              DDP process rank (0 for non-DDP)
        make_deterministic: Force CUDA deterministic algorithms.
                            Slower but produces bit-exact results.
                            Set True only when debugging precision issues.
    """
    effective = seed + rank
    random.seed(effective)
    os.environ["PYTHONHASHSEED"] = str(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)

    if make_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True lets cuDNN pick the fastest conv algorithm
        # for fixed input sizes — beneficial for GNN training
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def worker_init_fn(worker_id: int, base_seed: int = 0) -> None:
    """
    Pass to DataLoader(worker_init_fn=...) for reproducible worker seeds.
    Each worker gets a unique seed derived from base_seed.

    Usage:
        from functools import partial
        init = partial(worker_init_fn, base_seed=cfg.seed)
        DataLoader(dataset, worker_init_fn=init, ...)
    """
    set_seed(base_seed + worker_id)
