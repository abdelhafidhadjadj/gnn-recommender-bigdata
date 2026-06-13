"""
Evaluation metrics.

F7:   RMSE/MAE computed on sigmoid-scaled predictions in [1, 5].
F12:  popularity_baseline and random_baseline for sanity comparison.
F14:  user-relevance built with vectorised numpy (no iterrows).

Phase 2 additions:
  global_precision  — sklearn precision_score(true_bin, pred_bin): TP/(TP+FP)
                      across ALL test interactions. Complementary to Precision@K.
  K-filter          — for each K, evaluate only users with >= K total test items.
                      Prevents artificially low P@K for users with very few test
                      interactions (EvalConfig.use_k_filter).
  n_eval_users      — added to each K-result so reports show how many users were
                      evaluated (important when K-filter reduces the pool).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score,
)
from config import EvalConfig


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _build_user_rel(df: pd.DataFrame, n_users: int,
                    cfg: EvalConfig) -> dict[int, set[int]]:
    """Vectorised build of {user_id -> set of RELEVANT local item ids}."""
    uid_arr = df['user_id'].values.astype(int)
    iid_arr = (df['item_id'].values - n_users).astype(int)
    rat_arr = df['rating'].values.astype(float)
    mask    = rat_arr >= cfg.relevance_thresh
    user_rel: dict[int, set[int]] = {}
    for uid, iid in zip(uid_arr[mask], iid_arr[mask]):
        user_rel.setdefault(int(uid), set()).add(int(iid))
    return user_rel


def _build_user_all(df: pd.DataFrame, n_users: int) -> dict[int, set[int]]:
    """Vectorised build of {user_id -> set of ALL local item ids} (K-filter)."""
    uid_arr = df['user_id'].values.astype(int)
    iid_arr = (df['item_id'].values - n_users).astype(int)
    user_all: dict[int, set[int]] = {}
    for uid, iid in zip(uid_arr, iid_arr):
        user_all.setdefault(int(uid), set()).add(int(iid))
    return user_all


def _ranking_scores(rel_items: set, ranked: np.ndarray,
                    k_list: list[int]) -> dict[str, float]:
    """
    Per-user ranking metrics at every K:
      P@K, R@K, F1@K, NDCG@K, MRR@K
      HR@K  — Hit Rate: 1 if at least one relevant item is in top-K
      AP@K  — Average Precision (used to compute MAP)
    """
    top_max = ranked[:max(k_list)]
    out: dict = {}
    for k in k_list:
        top  = top_max[:k]
        hits = [1 if item in rel_items else 0 for item in top]
        p    = sum(hits) / k
        r    = sum(hits) / len(rel_items) if rel_items else 0.0
        f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        mrr  = next((1.0 / (i + 1) for i, item in enumerate(top)
                     if item in rel_items), 0.0)
        hr   = 1.0 if sum(hits) > 0 else 0.0

        # Average Precision@K  (for MAP)
        n_rel = 0; ap = 0.0
        for rank_i, h in enumerate(hits):
            if h:
                n_rel += 1
                ap += n_rel / (rank_i + 1)
        ap = ap / min(len(rel_items), k) if rel_items else 0.0

        out[k] = {'P': p, 'R': r, 'F1': f1, 'NDCG': ndcg,
                  'MRR': mrr, 'HR': hr, 'AP': ap}
    return out


# ---------------------------------------------------------------------------
# Ranking metrics (model)
# ---------------------------------------------------------------------------

def compute_ranking_metrics(
    model,
    full_edge_index: torch.Tensor,
    df: pd.DataFrame,
    n_users: int,
    cfg: EvalConfig,
    use_k_filter: bool = False,
    exclude_seen: dict | None = None,
) -> dict:
    """
    Compute ranking metrics at every K in cfg.k_list.

    Args:
        use_k_filter: False (default) — evaluate all users with >= 1 relevant item.
                      True           — for each K, restrict to users who have
                                       >= K TOTAL test items (not just relevant).
                                       Prevents artificially low P@K for users
                                       with very few test interactions.

    Returns:
        {k: {'P', 'R', 'F1', 'NDCG', 'MRR', 'HR', 'MAP', 'n_eval_users'}}

    n_eval_users is additive — existing callers that access ranking[k]['P']
    are unaffected.
    """
    model.eval()
    with torch.no_grad():
        all_emb = model(full_edge_index)
        ue = all_emb[:n_users].cpu()
        ie = all_emb[n_users:].cpu()

    user_rel = _build_user_rel(df, n_users, cfg)
    user_all = _build_user_all(df, n_users) if use_k_filter else {}

    _METRICS = ('P', 'R', 'F1', 'NDCG', 'MRR', 'HR', 'AP')

    # Score cache: torch.mv is cheap but avoids redundant recomputation when
    # the same user appears in multiple K-specific candidate sets.
    score_cache: dict[int, np.ndarray] = {}
    out: dict = {}

    for k in cfg.k_list:
        if use_k_filter:
            # Only users with >= k total test items AND at least 1 relevant item
            candidates = [
                u for u in user_rel
                if len(user_all.get(u, set())) >= k
            ][:cfg.max_eval_users]
        else:
            candidates = list(user_rel.keys())[:cfg.max_eval_users]

        accum: dict[str, list] = {m: [] for m in _METRICS}
        for uid in candidates:
            if uid not in score_cache:
                scores = torch.mv(ie, ue[uid]).detach().numpy().copy()
                # ── Exclure les items vus en training (standard CF protocol) ──
                # Sans ça, BPR met les training items tout en haut et les
                # test items (différents après dedup) ne sortent jamais dans top-K.
                if exclude_seen and uid in exclude_seen:
                    scores[list(exclude_seen[uid])] = -np.inf
                score_cache[uid] = scores
            ranked = np.argsort(-score_cache[uid])
            per_k  = _ranking_scores(user_rel[uid], ranked, [k])
            for m, v in per_k[k].items():
                accum[m].append(v)

        k_out = {
            m: float(np.mean(v)) if v else 0.0
            for m, v in accum.items()
        }
        k_out['MAP'] = k_out.pop('AP')
        k_out['n_eval_users'] = len(candidates)
        out[k] = k_out

    return out


# ---------------------------------------------------------------------------
# Full evaluation (ranking + regression)
# ---------------------------------------------------------------------------

def evaluate_model(model, full_edge_index: torch.Tensor,
                   df_test: pd.DataFrame, n_users: int,
                   cfg: EvalConfig,
                   df_train: pd.DataFrame | None = None) -> dict:
    """
    Full evaluation: regression metrics + binary classification + ranking.

    Returns:
        rmse              — sigmoid-scaled to [1, 5]
        mae               — sigmoid-scaled to [1, 5]
        accuracy          — binary classification accuracy (sigmoid >= 0.5)
        global_precision  — sklearn precision_score: TP/(TP+FP) across all
                            test interactions (notebook metric, Phase 2)
        ranking           — {k: metrics} over all users with >= 1 relevant item
        ranking_kfiltered — {k: metrics} over users with >= k total test items
                            (only present when cfg.use_k_filter=True)
    """
    model.eval()
    with torch.no_grad():
        all_emb = model(full_edge_index)
        ue = all_emb[:n_users].cpu()
        ie = all_emb[n_users:].cpu()

    u_t    = torch.tensor(df_test['user_id'].values, dtype=torch.long)
    i_t    = torch.tensor(df_test['item_id'].values - n_users, dtype=torch.long)
    true_r = df_test['rating'].values.astype(float)

    raw_scores = (ue[u_t] * ie[i_t]).sum(dim=1).detach()

    # F7: scale unbounded dot-products to [1, 5] for interpretable regression
    preds_scaled = (torch.sigmoid(raw_scores) * 4.0 + 1.0).numpy()
    scores_sig   = torch.sigmoid(raw_scores).numpy()

    rmse = float(np.sqrt(mean_squared_error(true_r, preds_scaled)))
    mae  = float(mean_absolute_error(true_r, preds_scaled))

    true_bin = (true_r    >= cfg.relevance_thresh).astype(int)
    pred_bin = (scores_sig >= 0.5).astype(int)

    acc = float(accuracy_score(true_bin, pred_bin))

    # global_precision: TP/(TP+FP) computed on positive test interactions
    # + an equal-sized random sample of unseen (negative) pairs.
    # Without negatives, the test set contains only observed interactions —
    # the model assigns sigmoid >= 0.5 to virtually all of them (BPR training),
    # making precision trivially 1.0. Adding negatives gives a meaningful score.
    n_items_total = ie.shape[0]
    rng_neg = np.random.RandomState(42)
    n_neg   = min(len(df_test), 20_000)
    neg_u   = rng_neg.randint(0, n_users,        size=n_neg)
    neg_i   = rng_neg.randint(0, n_items_total,  size=n_neg)
    neg_scores = (ue[neg_u] * ie[neg_i]).sum(dim=1).detach().numpy()
    neg_sig    = 1.0 / (1.0 + np.exp(-neg_scores))  # sigmoid
    neg_true   = np.zeros(n_neg, dtype=int)          # unseen = not relevant
    neg_pred   = (neg_sig >= 0.5).astype(int)

    combined_true = np.concatenate([true_bin, neg_true])
    combined_pred = np.concatenate([pred_bin, neg_pred])
    global_prec   = float(precision_score(combined_true, combined_pred, zero_division=0))

    # ── Construire l'ensemble des items vus en training (pour exclusion) ─────
    exclude_seen: dict | None = None
    if df_train is not None:
        exclude_seen = {}
        uid_tr = df_train['user_id'].values.astype(int)
        iid_tr = (df_train['item_id'].values - n_users).astype(int)
        for uid, iid in zip(uid_tr, iid_tr):
            exclude_seen.setdefault(int(uid), set()).add(int(iid))

    # Ranking: all users with >= 1 relevant item (standard CF evaluation)
    ranking = compute_ranking_metrics(
        model, full_edge_index, df_test, n_users, cfg,
        use_k_filter=False, exclude_seen=exclude_seen,
    )

    result: dict = {
        "rmse": rmse,
        "mae": mae,
        "accuracy": acc,
        "global_precision": global_prec,
        "ranking": ranking,
    }

    # K-filtered ranking: for each K, restrict to users with >= K test items
    if getattr(cfg, "use_k_filter", False):
        result["ranking_kfiltered"] = compute_ranking_metrics(
            model, full_edge_index, df_test, n_users, cfg,
            use_k_filter=True, exclude_seen=exclude_seen,
        )

    return result


# ---------------------------------------------------------------------------
# Baselines  (F12)
# ---------------------------------------------------------------------------

def popularity_baseline(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        n_users: int,
                        cfg: EvalConfig) -> dict:
    """
    Always recommends the most interacted-with items from training.
    item_id in df_train is globally-offset; convert to local before ranking.
    """
    counts = (df_train['item_id'] - n_users).value_counts()
    top_items = counts.index.tolist()          # sorted descending by popularity

    user_rel  = _build_user_rel(df_test, n_users, cfg)
    eval_users = list(user_rel.keys())[:cfg.max_eval_users]

    _BASE_M = ('P', 'R', 'NDCG', 'HR')
    accum: dict[int, dict[str, list]] = {
        k: {m: [] for m in _BASE_M} for k in cfg.k_list
    }
    for uid in eval_users:
        ranked = np.array(top_items[:max(cfg.k_list)])
        per_k  = _ranking_scores(user_rel[uid], ranked, cfg.k_list)
        for k, mets in per_k.items():
            for m in _BASE_M:
                accum[k][m].append(mets[m])

    return {k: {m: float(np.mean(v)) for m, v in mets.items()}
            for k, mets in accum.items()}


def random_baseline(df_test: pd.DataFrame, n_users: int, n_items: int,
                    cfg: EvalConfig, seed: int = 42) -> dict:
    """Randomly recommends items (reproducible with seed)."""
    rng = np.random.RandomState(seed)

    user_rel   = _build_user_rel(df_test, n_users, cfg)
    eval_users = list(user_rel.keys())[:cfg.max_eval_users]

    _BASE_M = ('P', 'R', 'NDCG', 'HR')
    accum: dict[int, dict[str, list]] = {
        k: {m: [] for m in _BASE_M} for k in cfg.k_list
    }
    for uid in eval_users:
        ranked = rng.choice(n_items, size=max(cfg.k_list), replace=False)
        per_k  = _ranking_scores(user_rel[uid], ranked, cfg.k_list)
        for k, mets in per_k.items():
            for m in _BASE_M:
                accum[k][m].append(mets[m])

    return {k: {m: float(np.mean(v)) for m, v in mets.items()}
            for k, mets in accum.items()}


# ---------------------------------------------------------------------------
# Normalized metrics
# ---------------------------------------------------------------------------

def normalize_metrics(
    ranking: dict,
    baseline_rand: dict,
    df_test: pd.DataFrame,
    cfg: EvalConfig,
) -> dict:
    """
    Returns a per-K dict with three views of each metric:

    *_pct   — raw value × 100  (percentage, e.g. 0.009 → 0.90 %)
    *_lift  — model / random   (>1 = better than random, <1 = worse)
    *_norm  — (model - random) / (ideal - random) × 100
              0 % = same as random  |  100 % = perfect

    Ideal values:
      P@K   = mean_users( min(n_rel_i, K) / K )
      R@K   = mean_users( min(K, n_rel_i) / n_rel_i )  [can be < 1]
      HR@K  = fraction of users that have ≥1 relevant item  (≤ 1.0)
      NDCG, MRR, MAP — already in [0,1]; ideal = 1.0
    """
    # Build per-user relevant-item counts from test set (vectorised)
    uid_arr = df_test['user_id'].values.astype(int)
    rat_arr = df_test['rating'].values.astype(float)
    rel_mask = rat_arr >= cfg.relevance_thresh

    from collections import defaultdict
    user_rel_counts: dict[int, int] = defaultdict(int)
    for uid in uid_arr[rel_mask]:
        user_rel_counts[int(uid)] += 1

    eval_users = list(user_rel_counts.keys())[:cfg.max_eval_users]
    counts = np.array([user_rel_counts[u] for u in eval_users], dtype=float)

    def _ideal_p(k):
        return float(np.mean(np.minimum(counts, k) / k))

    def _ideal_r(k):
        return float(np.mean(np.where(counts > 0, np.minimum(k, counts) / counts, 0)))

    def _ideal_hr():
        # fraction of evaluated users who have at least 1 relevant item
        return float(np.mean(counts > 0))

    out: dict = {}
    for k in sorted(ranking.keys()):
        m   = ranking[k]
        r   = baseline_rand.get(k, {})
        norm_k: dict = {}

        ideal = {
            'P':    _ideal_p(k),
            'R':    _ideal_r(k),
            'HR':   _ideal_hr(),
            'NDCG': 1.0,
            'MRR':  1.0,
            'MAP':  1.0,
        }

        for metric in ('P', 'R', 'HR', 'NDCG', 'MRR', 'MAP'):
            v     = m.get(metric, 0.0)
            rand  = r.get(metric, 0.0)
            idl   = ideal.get(metric, 1.0)

            norm_k[f'{metric}_pct']  = round(v * 100, 4)
            norm_k[f'{metric}_lift'] = (round(v / rand, 3)
                                        if rand > 1e-9 else None)
            denom = idl - rand
            norm_k[f'{metric}_norm'] = (round((v - rand) / denom * 100, 2)
                                        if denom > 1e-9 else None)

        out[k] = norm_k

    return out


# ---------------------------------------------------------------------------
# Console display
# ---------------------------------------------------------------------------

def _print_ranking_block(label: str, ranking: dict) -> None:
    """Print one ranking block (all-users or K-filtered)."""
    print(f"\n  {label}")
    for k, m in sorted(ranking.items()):
        n = m.get('n_eval_users', '?')
        print(
            f"    @{k:<3} [{n:>4} users]"
            f" | P={m['P']:.4f}  R={m['R']:.4f}"
            f"  HR={m['HR']:.4f}  NDCG={m['NDCG']:.4f}"
            f"  MAP={m.get('MAP',0):.4f}  MRR={m['MRR']:.4f}"
        )


def print_evaluation(results: dict) -> None:
    """
    Print full evaluation results to stdout.

    Prints:
      - Regression:          RMSE, MAE  (sigmoid-scaled to [1,5])
      - Binary classif.:     Accuracy, Global Precision
      - Ranking (all users): P@K, R@K, HR@K, NDCG@K, MAP@K, MRR@K
      - Ranking (K-filter):  same, restricted to users with >= K test items
                             (only shown when present in results)
    """
    print(f"\n  RMSE             (sigmoid [1,5]): {results['rmse']:.4f}")
    print(f"  MAE              (sigmoid [1,5]): {results['mae']:.4f}")
    print(f"  Accuracy         (sigmoid >= 0.5): {results['accuracy']:.4f}")
    print(f"  Global Precision (sigmoid >= 0.5): {results.get('global_precision', 0):.4f}")

    _print_ranking_block("Ranking — all users with >= 1 relevant item",
                         results['ranking'])

    if 'ranking_kfiltered' in results:
        _print_ranking_block(
            "Ranking — K-filter (users with >= K total test items)",
            results['ranking_kfiltered'],
        )
