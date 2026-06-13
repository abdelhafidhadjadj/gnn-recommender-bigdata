import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

base = r"C:\Users\hafid\Desktop\gnn_recommender\results_final"

runs = {
    "GAT":       "gat_w4_full",
    "GraphSAGE": "sage_w4_full",
    "LightGCN":  "lightgcn_w4_full",
}

for model, folder in runs.items():
    path = os.path.join(base, folder, "metrics")
    files = [f for f in os.listdir(path) if f.endswith(".json")]
    if not files:
        print(f"{model}: pas de fichier"); continue
    with open(os.path.join(path, files[0])) as f:
        m = json.load(f)

    r   = m.get("ranking", {})
    t   = m.get("timings", {})
    acc = m.get("accuracy", "N/A")
    gp  = m.get("global_precision", "N/A")
    rmse= m.get("rmse", "N/A")
    mae = m.get("mae",  "N/A")

    print(f"\n{'='*55}")
    print(f"  {model} — {folder}")
    print(f"{'='*55}")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  Accuracy={acc:.4f}  Global_Precision={gp:.4f}")
    for k in ["5","10","20"]:
        if k in r:
            d = r[k]
            print(f"  @{k:>2} : P={d['P']:.4f}  R={d['R']:.4f}  NDCG={d['NDCG']:.4f}  HR={d['HR']:.4f}")
    print(f"  t_load={t.get('t_load','?')}s  t_train={t.get('t_train','?')}s  workers={int(t.get('world_size',1))}")
