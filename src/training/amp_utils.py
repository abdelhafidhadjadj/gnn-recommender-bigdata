"""
Automatic Mixed Precision (AMP) context — CPU-safe.

Uses the current PyTorch 2.x API:
  torch.amp.autocast(device_type=...)
  torch.amp.GradScaler(device=...)

When CUDA is unavailable: autocast() is a no-op, GradScaler performs no scaling.

Usage:
    amp = AMPContext(enabled=torch.cuda.is_available())

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with amp.forward_ctx():          # fp16 forward pass on GPU / no-op on CPU
        out  = model(edge_index)
        loss = bpr_loss(...)

    amp.backward(loss, optimizer, model, grad_clip=1.0)
    # scale -> backward -> unscale -> clip -> step -> scaler.update
"""
from __future__ import annotations
from contextlib import contextmanager

import torch
import torch.nn as nn


class AMPContext:
    def __init__(self, enabled: bool = True) -> None:
        # Never enable AMP on CPU — autocast on CPU requires extra config
        self.enabled      = enabled and torch.cuda.is_available()
        self.device_type  = "cuda" if self.enabled else "cpu"
        # PyTorch 2.x API: torch.amp.GradScaler(device=...)
        self.scaler = torch.amp.GradScaler(device=self.device_type, enabled=self.enabled)

    # ── forward ───────────────────────────────────────────────────────────────

    @contextmanager
    def forward_ctx(self):
        """Wrap the model forward + loss computation for fp16 on GPU."""
        with torch.amp.autocast(device_type=self.device_type, enabled=self.enabled):
            yield

    # ── backward ──────────────────────────────────────────────────────────────

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        grad_clip: float = 1.0,
    ) -> float:
        """
        Full backward step:
          1. scaler.scale(loss).backward()
          2. scaler.unscale_(optimizer)
          3. clip_grad_norm_
          4. scaler.step(optimizer)
          5. scaler.update()

        Caller must call optimizer.zero_grad(set_to_none=True) BEFORE forward_ctx().
        Returns: scalar loss value for logging.
        """
        loss_val = loss.item()
        self.scaler.scale(loss).backward()
        if grad_clip > 0.0:
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
        return loss_val

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return self.scaler.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.scaler.load_state_dict(state)

    def __repr__(self) -> str:
        scale = self.scaler.get_scale() if self.enabled else 1.0
        return f"AMPContext(enabled={self.enabled}, scale={scale:.0f})"
