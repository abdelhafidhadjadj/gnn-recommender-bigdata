"""
BPR pairwise ranking loss.

F6: positive exclusion — known training positives are never sampled as negatives.
F8: L2 regularisation on embeddings added via reg_lambda.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Set


def _sample_negatives(u_indices: torch.Tensor, n_items: int,
                      user_pos: Dict[int, Set[int]] | None,
                      max_retries: int = 5) -> torch.Tensor:
    """
    Sample one negative item per entry in u_indices.
    If user_pos is supplied, re-sample any candidate that is a known positive
    (up to max_retries rounds of rejection sampling).
    """
    device = u_indices.device
    u_arr = u_indices.cpu().numpy()
    neg = np.random.randint(0, n_items, size=len(u_arr))

    if user_pos is not None:
        for _ in range(max_retries):
            false_neg = np.array(
                [neg[j] in user_pos.get(int(u_arr[j]), set()) for j in range(len(u_arr))],
                dtype=bool,
            )
            if not false_neg.any():
                break
            neg[false_neg] = np.random.randint(0, n_items, size=int(false_neg.sum()))

    return torch.tensor(neg, dtype=torch.long, device=device)


def bpr_loss(user_emb: torch.Tensor, item_emb: torch.Tensor,
             u_idx: torch.Tensor, pos_idx: torch.Tensor,
             n_items: int,
             n_neg: int = 4,
             user_pos: Dict[int, Set[int]] | None = None,
             reg_lambda: float = 0.0) -> torch.Tensor:
    """
    BPR loss with optional positive exclusion and L2 regularisation.

    Args:
        user_emb   – (n_users, d) convolved user embeddings
        item_emb   – (n_items, d) convolved item embeddings
        u_idx      – (B,) user indices for this batch
        pos_idx    – (B,) positive item indices (local, 0-based)
        n_items    – total number of items
        n_neg      – negatives sampled per positive
        user_pos   – dict {user_id -> set of positive item ids} for exclusion
        reg_lambda – L2 weight (0 = disabled)
    """
    if n_neg > 1:
        u_e   = u_idx.repeat_interleave(n_neg)
        pos_e = pos_idx.repeat_interleave(n_neg)
    else:
        u_e   = u_idx
        pos_e = pos_idx

    neg_e = _sample_negatives(u_e, n_items, user_pos)

    pos_s = (user_emb[u_e] * item_emb[pos_e]).sum(dim=1)
    neg_s = (user_emb[u_e] * item_emb[neg_e]).sum(dim=1)

    loss = -F.logsigmoid(pos_s - neg_s).mean()

    if reg_lambda > 0.0:
        reg = reg_lambda * (
            user_emb[u_e].norm(dim=1).pow(2).mean()
            + item_emb[pos_e].norm(dim=1).pow(2).mean()
            + item_emb[neg_e].norm(dim=1).pow(2).mean()
        )
        loss = loss + reg

    return loss
