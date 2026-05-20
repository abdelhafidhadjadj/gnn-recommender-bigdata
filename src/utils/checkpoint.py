"""
Versioned, full-state checkpoint management.

Every checkpoint stores:
  - model weights  (DDP-safe: saves model.module.state_dict when wrapped)
  - optimizer state
  - LR scheduler state
  - AMP GradScaler state
  - full model architecture config  → rebuild without CLI args
  - num_nodes                       → required for Embedding table size
  - serialised DynamicLabelEncoders → required for incremental mode (Phase 6)
  - training statistics

Checkpoint naming: {model_type}_v{version:03d}_e{epoch:04d}.pt
Best model copy:   {model_type}_best.pt  (always the highest val_score seen)

Only rank-0 writes to disk; all other DDP ranks are silently skipped.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        model_type: str,
        keep_last_n: int = 3,
    ) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.keep_last_n = keep_last_n
        self._history: list[Path] = []
        self.best_path: Optional[Path] = None
        self._best_score: float = -float("inf")

    # ── save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_score: float,
        cfg,                      # full Config object
        num_nodes: int,
        scheduler=None,
        scaler=None,              # AMPContext or GradScaler with .state_dict()
        user_encoder=None,
        item_encoder=None,
        extra: dict | None = None,
        rank: int = 0,
    ) -> Optional[Path]:
        """
        Save a complete checkpoint. Non-zero DDP ranks return None immediately.
        """
        if rank != 0:
            return None

        version = len(self._history) + 1
        fname = f"{self.model_type}_v{version:03d}_e{epoch:04d}.pt"
        path = self.dir / fname

        # DDP wraps the model in DistributedDataParallel; weights are in .module
        model_state = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        state: dict = {
            # ── identity ──────────────────────────────────────────────────────
            "version": version,
            "epoch": epoch,
            "val_score": val_score,
            "training_mode": getattr(cfg, "training_mode", "scratch"),

            # ── model ─────────────────────────────────────────────────────────
            "model_state": model_state,
            "model_config": {
                "model_type":   cfg.model_type,
                "num_nodes":    num_nodes,
                "emb_dim":      cfg.model.emb_dim,
                "dropout":      cfg.model.dropout,
                "gat_heads":    cfg.model.gat_heads,
                "n_layers":     cfg.model.n_layers,
                "use_residual": cfg.model.use_residual,
            },

            # ── optimiser / scheduler / AMP ───────────────────────────────────
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict() if scheduler is not None else None,
            "scaler_state":     scaler.state_dict()    if scaler    is not None else None,

            # ── encoders (needed for incremental mode, Phase 6) ───────────────
            "user_encoder": pickle.dumps(user_encoder) if user_encoder is not None else None,
            "item_encoder": pickle.dumps(item_encoder) if item_encoder is not None else None,

            # ── caller-supplied extras (nested dict, not spread) ──────────────
            # Access via: ckpt.get("extra", {}).get("key")
            "extra": extra or {},
        }

        torch.save(state, path)
        print(f"[Checkpoint] Saved  {path.name}  (val={val_score:.4f})")

        self._history.append(path)

        # Always keep the best model in a stable location
        if val_score > self._best_score:
            self._best_score = val_score
            best_path = self.dir / f"{self.model_type}_best.pt"
            torch.save(state, best_path)
            self.best_path = best_path
            print(f"[Checkpoint] New best  ->  {best_path.name}")

        self._evict_old()
        return path

    def _evict_old(self) -> None:
        """Remove oldest periodic checkpoints; always keep best + last N."""
        protected = {self.best_path}
        if len(self._history) > self.keep_last_n:
            for old in self._history[: -self.keep_last_n]:
                if old not in protected and old.exists():
                    old.unlink()
            self._history = self._history[-self.keep_last_n :]

    # ── load ──────────────────────────────────────────────────────────────────

    @staticmethod
    def load(path: str | Path, device: torch.device) -> dict:
        """
        Load a checkpoint from disk.
        Deserialises pickled encoders back to Python objects.
        Raises RuntimeError if the file is in the old ad-hoc format.
        """
        # weights_only=False required: checkpoints contain numpy arrays,
        # pickled DynamicLabelEncoder, and ReplayBuffer (trusted source).
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise RuntimeError(
                f"Checkpoint '{path}' is in the old format.\n"
                "Re-train with training_mode=scratch to produce a v2 checkpoint."
            )
        if ckpt.get("user_encoder") is not None:
            ckpt["user_encoder"] = pickle.loads(ckpt["user_encoder"])
        if ckpt.get("item_encoder") is not None:
            ckpt["item_encoder"] = pickle.loads(ckpt["item_encoder"])
        return ckpt

    @staticmethod
    def build_model_from_ckpt(
        ckpt: dict, device: torch.device, build_model_fn
    ) -> nn.Module:
        """
        Reconstruct the model from checkpoint metadata and load its weights.
        build_model_fn must accept (model_type, num_nodes, emb_dim, dropout,
                                    gat_heads, n_layers, use_residual).
        """
        mc = ckpt["model_config"]
        model = build_model_fn(
            mc["model_type"],
            mc["num_nodes"],
            mc["emb_dim"],
            mc["dropout"],
            mc.get("gat_heads",    4),
            mc.get("n_layers",     1),
            mc.get("use_residual", True),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def latest_path(self) -> Optional[Path]:
        return self._history[-1] if self._history else None
