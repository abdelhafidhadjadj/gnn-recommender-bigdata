"""
Replay buffer for incremental training — prevents catastrophic forgetting.

Design:
  - Stores past interactions as (user_enc, item_enc_local, rating).
  - item_enc_local is the 0-based encoded item ID with NO n_users offset.
    This is critical: when n_users grows after adding new users, the replay
    buffer remains valid because we apply the CURRENT n_users offset at
    sample time (not at store time).
  - Ring buffer (deque with maxlen) — old interactions are silently dropped
    when capacity is exceeded.
  - Serialisable via pickle — stored inside the checkpoint for incremental runs.

Usage:
    buf = ReplayBuffer(capacity=10_000)
    buf.add(df_train_encoded, n_users=300)         # store after scratch training
    # later, during incremental fine-tuning:
    replay_df = buf.sample(n=1000, current_n_users=310)  # applies new offset
"""
from __future__ import annotations
import random
from collections import deque
import pandas as pd


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self._buf: deque = deque(maxlen=capacity)

    # ── add interactions ──────────────────────────────────────────────────────

    def add(self, df: pd.DataFrame, n_users: int) -> None:
        """
        Store interactions from df.

        Args:
            df:      DataFrame with 'user_id' (encoded), 'item_id' (global,
                     i.e. encoded_item + n_users), 'rating' columns.
            n_users: current n_users (used to derive item_enc_local).
        """
        for row in df.itertuples(index=False):
            self._buf.append({
                "user_enc":      int(row.user_id),
                "item_enc_local": int(row.item_id) - n_users,  # strip offset
                "rating":         float(row.rating),
            })

    # ── sample interactions ───────────────────────────────────────────────────

    def sample(self, n: int, current_n_users: int) -> pd.DataFrame:
        """
        Draw a random sample of past interactions.

        Args:
            n:               how many interactions to sample (capped at len).
            current_n_users: current n_users AFTER any new-user additions.
                             Applied as item_id = item_enc_local + current_n_users.

        Returns:
            DataFrame with columns: user_id, item_id (global), rating.
        """
        n = min(n, len(self._buf))
        if n == 0:
            return pd.DataFrame(columns=["user_id", "item_id", "rating"])

        indices = random.sample(range(len(self._buf)), n)
        rows = [self._buf[i] for i in indices]
        df = pd.DataFrame(rows)
        df["user_id"] = df["user_enc"]
        df["item_id"] = df["item_enc_local"] + current_n_users
        return df[["user_id", "item_id", "rating"]].reset_index(drop=True)

    # ── properties ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._buf.maxlen

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"
