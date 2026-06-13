import json, os, glob, sys
sys.stdout.reconfigure(encoding='utf-8')

base = r"C:\Users\hafid\Desktop\gnn_recommender\results_final"

def get_metrics(folder):
    pattern = os.path.join(base, folder, "metrics", "*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)

# ── Best runs per model (full dataset, best NDCG) ────────────────────────────
best_runs = {
    "GAT":      "gat_w1_full",
    "GraphSAGE":"sage_w2_100k",
    "LightGCN": "lightgcn_w2_full",
}

print("=" * 70)
print("TABLE 1 — Performances sur ensemble de test (toutes métriques)")
print("=" * 70)

for model, folder in best_runs.items():
    m = get_metrics(folder)
    if not m:
        print(f"{model}: pas de données")
        continue

    print(f"\n--- {model} ({folder}) ---")
    print("RATING:", json.dumps(m.get("rating", {}), indent=2))
    print("RANKING:", json.dumps(m.get("ranking", {}), indent=2))
    print("ACCURACY:", json.dumps(m.get("accuracy", {}), indent=2))
    print("BASELINES:", json.dumps(m.get("baselines", {}), indent=2))
    print("TIMINGS:", json.dumps(m.get("timings", {}), indent=2))

# ── Workers comparison (speedup table) ───────────────────────────────────────
print("\n" + "=" * 70)
print("TABLE 4 — Standard vs Big Data (workers comparison)")
print("=" * 70)

for model_short in ["gat", "sage", "lightgcn"]:
    for w in ["w1", "w2", "w3", "w4"]:
        folder = f"{model_short}_{w}_full"
        m = get_metrics(folder)
        if not m:
            continue
        t = m.get("timings", {})
        r = m.get("ranking", {})
        k10 = r.get("10", {}).get("NDCG", r.get("5", {}).get("NDCG", "—"))
        print(f"{folder:<25} t_load={t.get('t_load','—'):<8} t_train={t.get('t_train','—'):<10} NDCG@K={k10}")
