"""
Hardware detection and adaptive configuration.

Detects the hardware tier (debug / cpu / single_gpu / multi_gpu) and
automatically adjusts the training Config to match available resources.

Design rules:
  1. CLI overrides always win  — call build_adaptive_config() BEFORE applying
     argparse overrides in main.py.
  2. Conservative estimates   — use 20 % of free VRAM as the threshold for
     full-batch vs NeighborLoader (graph_mode).
  3. No hard dependencies     — psutil is optional; graceful fallback if absent.
  4. Idempotent               — calling build_adaptive_config() twice is safe.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List

import torch


# ── Hardware profile ──────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    # GPU info
    num_gpus:         int
    gpu_names:        List[str]
    vram_free_gb:     List[float]   # free VRAM per GPU at detection time
    vram_total_gb:    List[float]   # total VRAM per GPU
    # CPU / RAM info
    cpu_cores:        int
    ram_available_gb: float
    # Mode flag
    is_debug:         bool          # True -> always apply debug settings

    @property
    def tier(self) -> str:
        """
        Four hardware tiers drive config adaptation:
          debug       — CPU or forced debug (--debug flag)
          cpu         — CPU only, normal mode
          single_gpu  — exactly 1 GPU
          multi_gpu   — 2+ GPUs
        """
        if self.is_debug:
            return "debug"
        if self.num_gpus == 0:
            return "cpu"
        if self.num_gpus == 1:
            return "single_gpu"
        return "multi_gpu"

    @property
    def min_free_vram_gb(self) -> float:
        """Smallest free VRAM across all GPUs (bottleneck GPU)."""
        return min(self.vram_free_gb) if self.vram_free_gb else 0.0

    @property
    def min_total_vram_gb(self) -> float:
        """Smallest total VRAM across all GPUs."""
        return min(self.vram_total_gb) if self.vram_total_gb else 0.0

    def summary(self) -> str:
        lines = [
            f"  Hardware tier  : {self.tier}",
            f"  GPUs           : {self.num_gpus}",
        ]
        for i, (name, free, total) in enumerate(
            zip(self.gpu_names, self.vram_free_gb, self.vram_total_gb)
        ):
            lines.append(f"  GPU {i}          : {name}  ({free:.1f}/{total:.0f} GB free)")
        lines += [
            f"  CPU cores      : {self.cpu_cores}",
            f"  RAM available  : {self.ram_available_gb:.1f} GB",
        ]
        return "\n".join(lines)


# ── Detection ─────────────────────────────────────────────────────────────────

def _get_gpu_info() -> tuple[list, list, list]:
    """Return (names, free_gb, total_gb) — one entry per visible GPU."""
    names, free_gb, total_gb = [], [], []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        names.append(props.name)
        try:
            free, total = torch.cuda.mem_get_info(i)
            free_gb.append(free  / 1024 ** 3)
            total_gb.append(total / 1024 ** 3)
        except Exception:
            free_gb.append(0.0)
            total_gb.append(props.total_memory / 1024 ** 3)
    return names, free_gb, total_gb


def detect_hardware(force_debug: bool = False) -> HardwareProfile:
    """
    Inspect available compute resources and return a HardwareProfile.

    Args:
        force_debug: Always set tier to 'debug' (maps to the --debug CLI flag).
    """
    cpu_cores = os.cpu_count() or 1
    ram_gb    = 16.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / 1024 ** 3
    except ImportError:
        pass   # psutil is optional; 16 GB is a safe default for a laptop

    names, free_gb, total_gb = _get_gpu_info()

    return HardwareProfile(
        num_gpus         = len(names),
        gpu_names        = names,
        vram_free_gb     = free_gb,
        vram_total_gb    = total_gb,
        cpu_cores        = cpu_cores,
        ram_available_gb = ram_gb,
        is_debug         = force_debug,
    )


# ── Adaptive config ───────────────────────────────────────────────────────────

def build_adaptive_config(cfg, profile: HardwareProfile):
    """
    Adjust cfg in-place to match the detected hardware tier.

    IMPORTANT: call this BEFORE applying CLI arg overrides so that
    explicit user flags (--epochs, --lr, …) always take precedence.

    Tier settings
    ─────────────
    debug      -> emb_dim=16, epochs=3, batch=32, n_neg=2, workers=0,
                 no AMP, skip SBERT, min_epochs=1, patience=1
    cpu        -> batch=128, workers=2, n_neg=10, n_trials=10
    single_gpu -> VRAM-scaled batch, workers=4, n_trials=25, AMP ready
    multi_gpu  -> per-GPU batch=1024, workers=4/GPU, n_trials=50, AMP ready
    """
    tier = profile.tier

    if tier == "debug":
        cfg.model.emb_dim         = 16
        cfg.train.num_epochs      = 3
        cfg.train.final_epochs    = 3
        cfg.train.warmup_epochs   = 1
        cfg.train.batch_size      = 32
        cfg.train.n_neg           = 2
        cfg.train.min_epochs      = 1
        cfg.train.eval_every      = 1
        cfg.train.patience        = 1
        cfg.train.use_scheduler   = False
        cfg.tune.n_trials             = 2
        cfg.tune.optuna_epochs        = 2
        cfg.ckpt.save_every_n_epochs  = 9999
        # SBERT warm-start takes minutes on large datasets — skip in debug
        cfg.graph.use_sbert_item_init = False

    elif tier == "cpu":
        cfg.train.batch_size      = 128
        cfg.train.n_neg           = 10
        cfg.tune.n_trials         = 10

    elif tier == "single_gpu":
        vram = profile.min_total_vram_gb
        if vram < 8:
            cfg.train.batch_size  = 512
        elif vram < 16:
            cfg.train.batch_size  = 1024
        else:
            cfg.train.batch_size  = 2048
        cfg.tune.n_trials         = 25

    elif tier == "multi_gpu":
        # Per-GPU batch size; effective batch = batch_size × world_size
        cfg.train.batch_size      = 1024
        cfg.tune.n_trials         = 50

    return cfg


# ── Worker count ──────────────────────────────────────────────────────────────

def recommended_workers(profile: HardwareProfile) -> int:
    """
    DataLoader num_workers recommendation.

    Rules:
      debug / cpu  -> 0 or 2 (avoid spawning overhead for small workloads)
      single GPU   -> 4
      multi GPU    -> 4 per GPU (capped by CPU cores / num_gpus)
    """
    if profile.tier == "debug":
        return 0
    if profile.tier == "cpu":
        return min(2, max(1, profile.cpu_cores // 2))
    # GPU modes
    workers_per_gpu = max(1, profile.cpu_cores // max(1, profile.num_gpus))
    return min(4, workers_per_gpu)


# ── Graph mode resolver ───────────────────────────────────────────────────────

def resolve_graph_mode(
    profile: HardwareProfile,
    cfg,
    n_nodes: int,
    emb_dim: int,
) -> str:
    """
    Resolve cfg.graph_mode = "auto" to an actual mode.

    Returns "full_batch" or "neighbor_loader".

    Auto logic:
      - CPU or debug -> full_batch  (no VRAM constraint)
      - GPU          -> full_batch if embedding table < 20 % of free VRAM;
                       else neighbor_loader
    """
    explicit = getattr(cfg, "graph_mode", "auto")
    if explicit != "auto":
        return explicit   # user pinned a specific mode

    if profile.num_gpus == 0 or profile.tier == "debug":
        return "full_batch"

    emb_bytes    = n_nodes * emb_dim * 4   # fp32
    free_bytes   = profile.min_free_vram_gb * 1024 ** 3
    threshold    = 0.20 * free_bytes       # 20 % safety margin

    return "full_batch" if emb_bytes < threshold else "neighbor_loader"


# ── Console report ────────────────────────────────────────────────────────────

def print_hardware_report(profile: HardwareProfile, cfg=None) -> None:
    """Print a formatted hardware summary, optionally with key config values."""
    print("\n" + "=" * 55)
    print("HARDWARE PROFILE")
    print("=" * 55)
    print(profile.summary())
    if cfg is not None:
        print(f"  -> tier           : {profile.tier}")
        print(f"  -> batch_size     : {cfg.train.batch_size}")
        print(f"  -> emb_dim        : {cfg.model.emb_dim}")
        print(f"  -> n_neg          : {cfg.train.n_neg}")
        print(f"  -> num_epochs     : {cfg.train.num_epochs}")
        print(f"  -> Optuna trials  : {cfg.tune.n_trials}")
        print(f"  -> AMP            : {profile.num_gpus > 0}")
        wkrs = recommended_workers(profile)
        print(f"  -> DataLoader workers: {wkrs}")
    print("=" * 55 + "\n")
