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
    Initialize process group if torchrun has set the required env vars.

    Safe to call when NOT running under torchrun — returns (False, 0, 0, 1)
    so the rest of the code can treat the single-process case uniformly.

    Single-GPU simulation (1 physical GPU, N logical workers):
      torchrun --nproc_per_node=N main.py ...
      → detects 1 physical GPU → uses gloo backend → all ranks on cuda:0
      → BPR pairs still sharded across ranks → effective batch × N

    Multi-GPU (N physical GPUs):
      torchrun --nproc_per_node=N main.py ...
      → uses nccl backend → each rank on its own cuda:local_rank

    Returns:
        (is_ddp, rank, physical_local_rank, world_size)
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1

    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Sensible defaults in case they were not set by the launcher
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    n_physical_gpus = torch.cuda.device_count()

    if n_physical_gpus == 1:
        # ── Single-GPU : toujours 2 GB fixe par worker ────────────────────────
        # Comparaison équitable entre toutes les configurations :
        #   standard   : 1 worker  × 2 GB  (torchrun --nproc_per_node=1)
        #   bigdata x2 : 2 workers × 2 GB  (torchrun --nproc_per_node=2)
        #   bigdata x4 : 4 workers × 2 GB  (torchrun --nproc_per_node=4)
        #
        # gloo backend : supporte N processus sur le même device.
        backend        = "gloo"
        physical_local = 0

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        size  = os.environ.get("SIZE", "")
        model = os.environ.get("MODEL_TYPE", "")

        if world_size == 1 and size == "full":
            VRAM_PER_WORKER_GB = 4.0   # full w=1 : 131k paires × 10 neg ≈ 2.1 GB
        elif model == "gat" and world_size == 1 and size == "100k":
            VRAM_PER_WORKER_GB = 2.5   # GAT w=1 100k uniquement
        else:
            # Tous les autres cas : 2 GB (comparaison équitable)
            VRAM_PER_WORKER_GB = 2.0

        vram_fraction = min(VRAM_PER_WORKER_GB / total_vram_gb, 1.0)

        torch.cuda.set_device(physical_local)
        torch.cuda.set_per_process_memory_fraction(vram_fraction, physical_local)

        mode_label = f"standard (1 worker × {VRAM_PER_WORKER_GB} GB)" if world_size == 1 else f"bigdata ({world_size} workers × 2 GB)"
        if rank == 0:
            print(
                f"\n[DDP] Simulation single-GPU — {mode_label}\n"
                f"      VRAM total   : {total_vram_gb:.0f} GB\n"
                f"      Par worker   : {VRAM_PER_WORKER_GB:.0f} GB  "
                f"({vram_fraction*100:.0f}% du VRAM)\n"
                f"      VRAM utilisé : {world_size * VRAM_PER_WORKER_GB:.0f} GB / {total_vram_gb:.0f} GB\n"
                f"      Backend      : gloo"
            )
    else:
        # ── Multi-GPU réel ────────────────────────────────────────────────────
        backend        = "nccl"
        physical_local = local_rank
        torch.cuda.set_device(physical_local)

    dist.init_process_group(backend=backend)

    return True, rank, physical_local, world_size


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

    Pour la simulation single-GPU, local_rank=0 pour tous les processus
    donc device_ids=[0] est correct dans tous les cas.
    """
    return DDP(
        model,
        device_ids             = [local_rank],
        output_device          = local_rank,
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
