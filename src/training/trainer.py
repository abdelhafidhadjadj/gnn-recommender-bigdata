"""
GNN training loop — Phase 4: auto full-batch / mini-batch dispatch.

graph_mode = "full_batch" (default for small graphs)
  model(full_edge_index) called ONCE per epoch — original fast path.

graph_mode = "neighbor_loader"
  LinkNeighborLoader feeds mini-batches; model(batch.edge_index, n_id=batch.n_id)
  processes a k-hop subgraph per batch. Scales to large graphs that exceed VRAM.

  Negative embedding approximation (standard in mini-batch CF):
    - Positive item embeddings : GNN-propagated (full neighborhood context)
    - Negative item embeddings : raw from embedding table (no GNN propagation)

Evaluation always uses full-batch (n_id=None) regardless of training mode.

Preserved from earlier phases:
  F4  — ONE forward pass per epoch (full-batch) / one forward per batch (mini-batch)
  F6  — positive exclusion via user_pos dict
  F8  — L2 regularisation in bpr_loss
  F9  — cosine annealing LR scheduler after warmup
  F10 — early stopping guarded by min_epochs
  AMP — AMPContext (no-op on CPU)
  Ckpt — CheckpointManager saves periodic + best
"""
from __future__ import annotations
import copy
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainConfig, CheckpointConfig, Config
from training.loss import bpr_loss, _sample_negatives
from training.amp_utils import AMPContext
from utils.checkpoint import CheckpointManager


# ── helpers ───────────────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
    lr_override: float | None = None,
) -> torch.optim.Optimizer:
    lr = lr_override if lr_override is not None else cfg.lr
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    return torch.optim.Adam(model.parameters(), lr=lr)


def _build_user_pos(
    train_u: torch.Tensor, train_pos: torch.Tensor
) -> Dict[int, Set[int]]:
    """Map each user to the set of positive item-local indices in training."""
    user_pos: Dict[int, Set[int]] = defaultdict(set)
    for u, p in zip(train_u.tolist(), train_pos.tolist()):
        user_pos[int(u)].add(int(p))
    return dict(user_pos)


# ── mini-batch epoch helper ───────────────────────────────────────────────────

def _train_minibatch_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,                              # LinkNeighborLoader
    n_users: int,
    n_items: int,
    cfg: TrainConfig,
    amp_ctx: AMPContext,
    user_pos: Dict[int, Set[int]],
) -> float:
    """
    One epoch of mini-batch BPR training via LinkNeighborLoader.

    Positive embeddings  : GNN-propagated (full neighborhood context).
    Negative embeddings  : raw from model.embeddings (no GNN propagation).
    This asymmetry is a standard approximation in mini-batch CF (NGCF, LightGCN).
    """
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    device = next(model.parameters()).device

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Supervision edge local indices inside the batch
        src = batch.edge_label_index[0]   # local user positions in batch
        pos = batch.edge_label_index[1]   # local item positions in batch

        if src.numel() == 0:
            continue

        with amp_ctx.forward_ctx():
            # GNN on sampled subgraph — only batch nodes processed
            all_emb = model(batch.edge_index, n_id=batch.n_id)

            user_emb_batch = all_emb[src]   # GNN-propagated user embeddings
            pos_emb_batch  = all_emb[pos]   # GNN-propagated positive item emb.

            # Negative sampling from global item space
            # Map batch-local src → global user idx for positive exclusion
            u_global = batch.n_id[src]
            if cfg.n_neg > 1:
                u_global  = u_global.repeat_interleave(cfg.n_neg)
                user_emb  = user_emb_batch.repeat_interleave(cfg.n_neg, dim=0)
                pos_emb   = pos_emb_batch.repeat_interleave(cfg.n_neg, dim=0)
            else:
                user_emb = user_emb_batch
                pos_emb  = pos_emb_batch

            # Sample negatives (global item-local indices, 0-based)
            neg_local = _sample_negatives(u_global, n_items, user_pos)
            # Raw embedding for negatives (no GNN propagation — standard approx)
            emb_module = model.module.embeddings if hasattr(model, "module") else model.embeddings
            neg_emb    = emb_module(neg_local.to(device) + n_users)

            pos_s = (user_emb * pos_emb).sum(dim=1)
            neg_s = (user_emb * neg_emb).sum(dim=1)
            loss  = -F.logsigmoid(pos_s - neg_s).mean()

            if cfg.reg_lambda > 0.0:
                reg = cfg.reg_lambda * (
                    user_emb.norm(dim=1).pow(2).mean()
                    + pos_emb.norm(dim=1).pow(2).mean()
                    + neg_emb.norm(dim=1).pow(2).mean()
                )
                loss = loss + reg

        amp_ctx.backward(loss, optimizer, model, cfg.grad_clip)
        epoch_loss += loss.item()
        n_batches  += 1

    return epoch_loss / max(n_batches, 1)


# ── main training function ────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    full_edge_index: torch.Tensor,
    train_u: torch.Tensor,
    train_pos: torch.Tensor,
    n_users: int,
    n_items: int,
    cfg: TrainConfig,
    *,
    # Phase 4: pass a LinkNeighborLoader to enable mini-batch training
    loader=None,                         # None → full-batch path
    # Phase 1 additions (all optional for backward compatibility)
    amp_ctx: AMPContext | None = None,
    ckpt_manager: CheckpointManager | None = None,
    ckpt_cfg: CheckpointConfig | None = None,
    full_cfg: Config | None = None,       # needed for checkpoint metadata
    ckpt_extra: dict | None = None,       # passed as-is to every ckpt_manager.save()
    num_nodes: int | None = None,
    user_encoder=None,
    item_encoder=None,
    # existing callbacks
    rank: int = 0,
    verbose: bool = True,
    epoch_callback: Callable[[int, float], None] | None = None,
    eval_fn: Callable[[nn.Module], float] | None = None,
) -> List[float]:
    """
    Train for cfg.num_epochs with BPR loss.

    Args:
        amp_ctx:      AMPContext for mixed precision (pass None → no-op on CPU).
        ckpt_manager: CheckpointManager; if None, no periodic saves are made.
        ckpt_cfg:     CheckpointConfig (save_every_n_epochs).
        full_cfg:     Full Config object stored inside each checkpoint.
        num_nodes:    n_users + n_items; stored in checkpoint for model rebuild.
        eval_fn:      callable(model) → float; called every cfg.eval_every epochs.
                      Return value is treated as validation score (higher = better).
        rank:         DDP process rank; only rank 0 prints and saves.

    Returns:
        List of per-epoch BPR loss values.
    """
    # Default AMPContext — no-op when CUDA absent
    if amp_ctx is None:
        amp_ctx = AMPContext(enabled=False)

    history: List[float] = []
    base_lr = optimizer.param_groups[0]["lr"]

    # Build positive lookup ONCE before the loop (F6)
    user_pos = _build_user_pos(train_u, train_pos)

    # Cosine annealing scheduler activates after warmup (F9)
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if cfg.use_scheduler:
        t_max = max(1, cfg.num_epochs - cfg.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=base_lr * 0.01
        )

    # Early-stopping state (F10)
    best_val_score: float = -float("inf")
    patience_counter: int = 0
    best_state: Optional[dict] = None

    # Checkpoint cadence
    save_every = (
        ckpt_cfg.save_every_n_epochs
        if ckpt_cfg is not None
        else 9999  # effectively disabled
    )

    for epoch in range(1, cfg.num_epochs + 1):

        # ── LR warmup ─────────────────────────────────────────────────────────
        if epoch <= cfg.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * epoch / cfg.warmup_epochs
        elif epoch == cfg.warmup_epochs + 1 and scheduler is None:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr

        # ── train one epoch — dispatches on loader ────────────────────────────
        if loader is not None:
            # ── Mini-batch path (Phase 4): LinkNeighborLoader ─────────────────
            # Scheduler step and LR warmup are handled here too, but the actual
            # forward/backward is delegated to _train_minibatch_epoch.
            loss_val = _train_minibatch_epoch(
                model, optimizer, loader,
                n_users, n_items, cfg, amp_ctx, user_pos,
            )
        else:
            # ── Full-batch path (Phase 1): ONE forward per epoch (F4) ─────────
            model.train()
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx.forward_ctx():
                all_emb  = model(full_edge_index)
                user_emb = all_emb[:n_users]
                item_emb = all_emb[n_users:]

                n_train = len(train_u)
                if cfg.use_all_pairs or n_train <= 50_000:
                    u_b, pos_b = train_u, train_pos
                else:
                    perm  = torch.randperm(n_train, device=train_u.device)[: cfg.batch_size]
                    u_b   = train_u[perm]
                    pos_b = train_pos[perm]

                loss = bpr_loss(
                    user_emb, item_emb, u_b, pos_b,
                    n_items, cfg.n_neg, user_pos, cfg.reg_lambda,
                )

            loss_val = amp_ctx.backward(loss, optimizer, model, cfg.grad_clip)

        # Step scheduler after warmup
        if scheduler is not None and epoch > cfg.warmup_epochs:
            scheduler.step()

        history.append(loss_val)
        current_lr = optimizer.param_groups[0]["lr"]

        if verbose and rank == 0 and epoch % 10 == 0:
            print(
                f"  Epoch {epoch:4d}/{cfg.num_epochs}"
                f" | BPR Loss: {loss_val:.4f}"
                f" | LR: {current_lr:.2e}"
                f" | AMP: {amp_ctx.enabled}"
            )

        if epoch_callback is not None:
            epoch_callback(epoch, loss_val)

        # ── periodic checkpoint ────────────────────────────────────────────────
        if (
            ckpt_manager is not None
            and full_cfg is not None
            and rank == 0
            and epoch % save_every == 0
        ):
            ckpt_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_score=best_val_score,
                cfg=full_cfg,
                num_nodes=num_nodes or (n_users + n_items),
                scheduler=scheduler,
                scaler=amp_ctx,
                user_encoder=user_encoder,
                item_encoder=item_encoder,
                extra=ckpt_extra,
                rank=rank,
            )

        # ── early stopping (guarded by min_epochs) (F10) ──────────────────────
        past_min = epoch >= cfg.min_epochs
        if eval_fn is not None and past_min and epoch % cfg.eval_every == 0:
            val_score = eval_fn(model)
            model.train()

            if val_score > best_val_score + 1e-5:
                best_val_score = val_score
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
                # Save best checkpoint immediately
                if ckpt_manager is not None and full_cfg is not None and rank == 0:
                    ckpt_manager.save(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        val_score=val_score,
                        cfg=full_cfg,
                        num_nodes=num_nodes or (n_users + n_items),
                        scheduler=scheduler,
                        scaler=amp_ctx,
                        user_encoder=user_encoder,
                        item_encoder=item_encoder,
                        extra=ckpt_extra,
                        rank=rank,
                    )
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    if verbose and rank == 0:
                        print(
                            f"  [Early stop] epoch {epoch}"
                            f" — best val NDCG = {best_val_score:.4f}"
                        )
                    break

    # Restore best weights if early stopping improved them
    if best_state is not None:
        model.load_state_dict(best_state)

    return history
