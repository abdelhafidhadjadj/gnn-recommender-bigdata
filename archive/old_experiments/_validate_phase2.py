"""Phase 2 validation — metrics overhaul."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

from config import EvalConfig
from evaluation.metrics import (
    compute_ranking_metrics, evaluate_model,
    _build_user_rel, _build_user_all,
    normalize_metrics, popularity_baseline, random_baseline,
)
from models import build_model

print("=" * 60)
print("PHASE 2 VALIDATION — metrics overhaul")
print("=" * 60)

# ── shared fixtures ─────────────────────────────────────────────────────────
n_users, n_items = 20, 30
num_nodes = n_users + n_items
cfg = EvalConfig(k_list=[5, 10], max_eval_users=100,
                 relevance_thresh=4.0, use_k_filter=True)

model = build_model("sage", num_nodes, 32, 0.0, 4, n_layers=1, use_residual=False)
# Random edge_index (users → items, bidirectional)
ei = torch.stack([
    torch.cat([torch.randint(0, n_users, (60,)),
               torch.randint(n_users, num_nodes, (60,))]),
    torch.cat([torch.randint(n_users, num_nodes, (60,)),
               torch.randint(0, n_users, (60,))]),
])

rng = np.random.default_rng(0)
# Build test df: each of 20 users has 3-8 test interactions
rows = []
for uid in range(n_users):
    n_interactions = rng.integers(3, 9)
    iids = rng.choice(range(n_users, num_nodes), size=n_interactions, replace=False)
    for iid in iids:
        rows.append({'user_id': uid, 'item_id': iid,
                     'rating': rng.choice([3.0, 4.0, 5.0])})
df_test = pd.DataFrame(rows)

# ── [1] _build_user_all ──────────────────────────────────────────────────────
print("\n[1] _build_user_all:")
user_all = _build_user_all(df_test, n_users)
user_rel = _build_user_rel(df_test, n_users, cfg)
assert all(isinstance(v, set) for v in user_all.values()), "Expected sets"
print(f"    users in user_all: {len(user_all)}  (all test users)")
print(f"    users in user_rel: {len(user_rel)}  (users with >= 1 relevant item)")
assert len(user_all) >= len(user_rel), "user_all should be superset of user_rel keys"
# All items in user_rel should be in user_all for the same user
for uid, rel in user_rel.items():
    assert rel.issubset(user_all.get(uid, set()))
print("    user_rel is subset of user_all for every user  [PASS]")

# ── [2] K-filter reduces candidate set ──────────────────────────────────────
print("\n[2] K-filter user count reduction:")
rank_no_filter = compute_ranking_metrics(model, ei, df_test, n_users, cfg,
                                          use_k_filter=False)
rank_filtered  = compute_ranking_metrics(model, ei, df_test, n_users, cfg,
                                          use_k_filter=True)

for k in cfg.k_list:
    n_all = rank_no_filter[k]['n_eval_users']
    n_flt = rank_filtered[k]['n_eval_users']
    assert n_flt <= n_all, f"K-filter should reduce pool at K={k}"
    print(f"    @{k}: all_users={n_all}  k_filtered={n_flt}  (<= all: OK)")

# ── [3] n_eval_users key present ────────────────────────────────────────────
print("\n[3] n_eval_users in every K result:")
for k in cfg.k_list:
    assert 'n_eval_users' in rank_no_filter[k], f"Missing n_eval_users at K={k}"
    assert 'n_eval_users' in rank_filtered[k],  f"Missing n_eval_users (filtered) at K={k}"
print("    n_eval_users present in all K results  [PASS]")

# ── [4] evaluate_model returns global_precision ──────────────────────────────
print("\n[4] evaluate_model — new fields:")
results = evaluate_model(model, ei, df_test, n_users, cfg)

assert 'global_precision' in results,      "Missing global_precision"
assert 'ranking'          in results,      "Missing ranking"
assert 'ranking_kfiltered' in results,     "Missing ranking_kfiltered (use_k_filter=True)"

gp = results['global_precision']
acc = results['accuracy']
assert 0.0 <= gp <= 1.0,  f"global_precision out of range: {gp}"
assert 0.0 <= acc <= 1.0, f"accuracy out of range: {acc}"

print(f"    RMSE             = {results['rmse']:.4f}")
print(f"    MAE              = {results['mae']:.4f}")
print(f"    Accuracy         = {acc:.4f}")
print(f"    Global Precision = {gp:.4f}")
print(f"    ranking keys     = {list(results['ranking'].keys())}")
print(f"    kfiltered keys   = {list(results['ranking_kfiltered'].keys())}")
print("    All fields present  [PASS]")

# ── [5] Backward compatibility — old callers still work ─────────────────────
print("\n[5] Backward compatibility:")
# Optuna tuner does: ranking.get(k, {}).get('NDCG', 0.0)
k_sample = cfg.k_list[0]
ndcg = results['ranking'].get(k_sample, {}).get('NDCG', 0.0)
p    = results['ranking'].get(k_sample, {}).get('P', 0.0)
assert isinstance(ndcg, float), "NDCG should be float"
assert isinstance(p,    float), "P should be float"
print(f"    ranking[{k_sample}]['NDCG'] = {ndcg:.4f}  (Optuna access pattern: OK)")

# normalize_metrics still works
nm = normalize_metrics(results['ranking'], {}, df_test, cfg)
assert len(nm) == len(cfg.k_list), "normalize_metrics output length mismatch"
print(f"    normalize_metrics returns {len(nm)} K-entries  [PASS]")

# popularity/random baselines still work
df_train = df_test.copy()  # just for API test
pop  = popularity_baseline(df_train, df_test, n_users, cfg)
rand = random_baseline(df_test, n_users, n_items, cfg)
assert set(pop.keys())  == set(cfg.k_list), "popularity_baseline K mismatch"
assert set(rand.keys()) == set(cfg.k_list), "random_baseline K mismatch"
print(f"    popularity_baseline: {list(pop.keys())}  [PASS]")
print(f"    random_baseline:     {list(rand.keys())}  [PASS]")

# ── [6] MAP field present in all outputs ────────────────────────────────────
print("\n[6] MAP field (AP renamed after averaging):")
for k in cfg.k_list:
    assert 'MAP' in results['ranking'][k],          f"MAP missing at K={k}"
    assert 'AP'  not in results['ranking'][k],      f"AP should be renamed to MAP"
    assert 'MAP' in results['ranking_kfiltered'][k], f"MAP missing in kfiltered at K={k}"
print("    MAP present, AP renamed correctly  [PASS]")

print("\n" + "=" * 60)
print("PHASE 2 VALIDATION  ->  ALL PASSED")
print("=" * 60)
