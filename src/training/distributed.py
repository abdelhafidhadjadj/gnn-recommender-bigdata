"""
DDP utilities — torchrun-compatible (Phase 5).

Replaces the old mp.spawn approach.

Why torchrun instead of mp.spawn:
  - torchrun is the production standard since PyTorch 1.9
  - handles process spawning, elastic training, and fault tolerance externally
  - works natively with Docker: just set env vars in docker-compose.yml
  - no code change needed to switch from 1 GPU to 4 GPUs

How it works:
  torchrun --nproc_per_node=4 main.py [args]
  ↓
  torchrun spawns 4 processes and sets:
    RANK        = 0,1,2,3   (global rank)
    LOCAL_RANK  = 0,1,2,3   (GPU index on this machine)
    WORLD_SIZE  = 4
    MASTER_ADDR = localhost
    MASTER_PORT = 29500
  ↓
  each process calls init_distributed() → reads env vars → sets cuda device
  ↓
  model wrapped in DDP(model, device_ids=[local_rank])
  ↓
  BPR pairs sharded: rank k processes pairs [k::world_size]
  ↓
  loss.backward() → DDP allreduce averages gradients across all ranks
  ↓
  all ranks apply the averaged gradient via optimizer.step()
  ↓
  rank 0 saves checkpoint and runs final evaluation

DDP + full-batch GNN:
  Each GPU holds the FULL graph (full_edge_index replicated).
  Different BPR pair shards → different loss signals → averaged gradients.
  Effective batch = world_size × pairs_per_rank.
  For 4 GPUs: effective batch ≈ 4× compared to single GPU.

NCCL on PCIe (no NVLink on RTX 2080 Ti):
  allreduce bandwidth ≈ 16 GB/s.  For our model (~45k params × 64 dim = 2.9 MB),
  one allreduce takes < 1 ms — negligible vs. forward/backward.
"""
from __future__ import annotations
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ── Process group lifecycle ───────────────────────────────────────────────────

def init_distributed() -> tuple[bool, int, int, int]:
    """
    Initialize NCCL process group if torchrun has set the required env vars.

    Safe to call when NOT running under torchrun — returns (False, 0, 0, 1)
    so the rest of the code can treat the single-process case uniformly.

    Returns:
        (is_ddp, rank, local_rank, world_size)
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1

    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Sensible defaults in case they were not set by the launcher
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Destroy the process group (call at end of main on every rank)."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Model wrapping ────────────────────────────────────────────────────────────

def wrap_ddp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    """
    Wrap model in DistributedDataParallel.

    find_unused_parameters=False: our models use all parameters in every
    forward pass — setting True adds unnecessary overhead.
    """
    return DDP(
        model,
        device_ids          = [local_rank],
        output_device       = local_rank,
        find_unused_parameters = False,
    )


# ── BPR pair sharding ─────────────────────────────────────────────────────────

def shard_bpr_pairs(
    train_u:   torch.Tensor,
    train_pos: torch.Tensor,
    rank:      int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Each rank receives a strided slice of the training pairs:
      rank 0 → [0, 4, 8, ...]
      rank 1 → [1, 5, 9, ...]
      ...

    This ensures the model sees all training interactions every epoch
    (distributed across ranks) and gradients are averaged by DDP allreduce.
    """
    if world_size <= 1:
        return train_u, train_pos
    return train_u[rank::world_size], train_pos[rank::world_size]


# ── Synchronization helpers ───────────────────────────────────────────────────

def barrier() -> None:
    """All-ranks synchronization point."""
    if dist.is_initialized():
        dist.barrier()


def is_main_process(rank: int = 0) -> bool:
    """True only for rank 0 — used to gate logging, saving, evaluation."""
    return rank == 0
