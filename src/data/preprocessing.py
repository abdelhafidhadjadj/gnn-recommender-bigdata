"""
Encoding, filtering, deduplication, and 70/15/15 train/val/test split.

Phase 6 addition: DynamicLabelEncoder
  - Same API as sklearn LabelEncoder (.classes_, .transform(), .fit())
  - Adds .add_new(values) to extend with new entities without refitting
  - Existing mappings are ALWAYS preserved after add_new()
  - Serialisable via pickle (stored inside checkpoint for incremental mode)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from config import DataConfig  # used by build_train_test


# ── Dynamic Label Encoder ─────────────────────────────────────────────────────

class DynamicLabelEncoder:
    """
    sklearn-compatible LabelEncoder that can be extended with new classes.

    Invariant: existing class → integer mappings NEVER change after add_new().
    New classes are always appended at the end, preserving old model embedding
    indices without any reindexing.

    API mirrors sklearn LabelEncoder:
        enc.fit(values)
        enc.transform(values)
        enc.classes_          — np.ndarray of all known classes (ordered)
        enc.is_known(value)   — True if value was seen in fit() or add_new()
    """

    def __init__(self):
        self.classes_: np.ndarray = np.array([], dtype=object)
        self._cls_to_idx: dict = {}

    # ── initial fit ───────────────────────────────────────────────────────────

    def fit(self, values) -> "DynamicLabelEncoder":
        """Fit on an iterable of values (discards any previous state)."""
        unique = sorted(set(values))
        self.classes_    = np.array(unique, dtype=object)
        self._cls_to_idx = {v: i for i, v in enumerate(unique)}
        return self

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, values) -> np.ndarray:
        """Encode values to integers. Raises KeyError for unknown values."""
        return np.array([self._cls_to_idx[v] for v in values], dtype=np.int64)

    def fit_transform(self, values) -> np.ndarray:
        return self.fit(values).transform(values)

    # ── incremental extension ────────────────────────────────────────────────

    def add_new(self, new_values) -> int:
        """
        Extend the encoder with new classes.
        Existing mappings are untouched; new classes receive the next integers.
        Returns the number of classes actually added.
        """
        added = 0
        for v in sorted(set(new_values)):
            if v not in self._cls_to_idx:
                idx = len(self.classes_)
                self._cls_to_idx[v] = idx
                self.classes_ = np.append(self.classes_, v)
                added += 1
        return added

    # ── queries ───────────────────────────────────────────────────────────────

    def is_known(self, value) -> bool:
        return value in self._cls_to_idx

    def __len__(self) -> int:
        return len(self.classes_)

    def __repr__(self) -> str:
        return f"DynamicLabelEncoder(n_classes={len(self.classes_)})"


# ── Internal helper (uses DynamicLabelEncoder, not sklearn) ───────────────────

def _encode_data(df: pd.DataFrame, column_name: str,
                 review_df: pd.DataFrame) -> tuple:
    """
    Fit a DynamicLabelEncoder on entities that appear in review_df,
    then transform both DataFrames in-place.
    Returns (review_df, DynamicLabelEncoder).
    """
    nodes    = review_df[column_name].unique()
    mask     = df[column_name].isin(nodes)
    nodes_df = df[mask].sort_values(by=column_name).reset_index(drop=True)

    enc = DynamicLabelEncoder()
    enc.fit(nodes_df[column_name].values)

    review_df = review_df.copy()
    review_df[column_name] = enc.transform(review_df[column_name].values)
    return review_df, enc


# ── Public preprocessing function ─────────────────────────────────────────────

def preprocess(business_df: pd.DataFrame, user_df: pd.DataFrame,
               review_df: pd.DataFrame):
    """
    Returns:
        review_df       – interaction table (user_id, item_id [offset], rating)
        review_df_full  – same + date column (for temporal edge weights)
        user_enc        – DynamicLabelEncoder for users
        item_enc        – DynamicLabelEncoder for items
        n_users, n_items
    """
    review_df = review_df[
        review_df['user_id'].isin(user_df['user_id']) &
        review_df['business_id'].isin(business_df['business_id'])
    ].copy()

    review_df, user_enc = _encode_data(user_df,     'user_id',     review_df)
    review_df, item_enc = _encode_data(business_df, 'business_id', review_df)

    # Shift item IDs into [n_users, n_users+n_items) node space
    review_df['business_id'] = review_df['business_id'] + len(user_enc.classes_)
    review_df = review_df.drop(columns=['review_id'], errors='ignore')

    sort_cols = [c for c in ['user_id', 'business_id', 'date'] if c in review_df.columns]
    review_df = review_df.sort_values(sort_cols)
    review_df = review_df.drop_duplicates(subset=['user_id', 'business_id'], keep='last')
    review_df = review_df.reset_index(drop=True)

    rating_col = 'stars' if 'stars' in review_df.columns else 'rating'
    review_df_full = review_df.rename(
        columns={rating_col: 'rating', 'business_id': 'item_id'}
    ).copy()
    review_df = review_df[['user_id', 'business_id', rating_col]].copy()
    review_df = review_df.rename(columns={rating_col: 'rating', 'business_id': 'item_id'})

    n_users = review_df['user_id'].nunique()
    n_items = review_df['item_id'].nunique()
    return review_df, review_df_full, user_enc, item_enc, n_users, n_items


def build_train_test(ui_edge_index: torch.Tensor, ui_edge_values: torch.Tensor,
                     n_users: int, cfg: DataConfig):
    """
    70 / 15 / 15 stratified split.

    Returns:
        train_u, train_pos, df_val, df_test, train_idx
    """
    n = ui_edge_index.shape[1]
    all_idx = list(range(n))

    temp_frac = cfg.val_size + cfg.test_size
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=temp_frac, random_state=cfg.random_state
    )
    test_ratio = cfg.test_size / temp_frac
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio, random_state=cfg.random_state
    )

    def _make_df(indices):
        return pd.DataFrame({
            'user_id': ui_edge_index[0, indices].numpy(),
            'item_id': ui_edge_index[1, indices].numpy(),
            'rating':  ui_edge_values[indices].numpy(),
        })

    train_edge_index = ui_edge_index[:, train_idx]
    df_val  = _make_df(val_idx)
    df_test = _make_df(test_idx)

    train_u   = train_edge_index[0]
    train_pos = train_edge_index[1] - n_users

    return train_u, train_pos, df_val, df_test, train_idx
