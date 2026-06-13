import json, os, glob, sys
sys.stdout.reconfigure(encoding='utf-8')

base = r"C:\Users\hafid\Desktop\gnn_recommender\outputs"
best = {}

for f in glob.glob(base + r"\*\metrics\*.json"):
    with open(f) as fp:
        m = json.load(fp)
    folder = f.split(os.sep)[-3]
    parts = folder.split("_")
    if len(parts) < 3:
        continue
    model, workers, size = parts[0], parts[1], parts[2]
    ndcg5 = m.get("ranking", {}).get("5", {}).get("NDCG", 0)
    key = (model, size)
    if key not in best or ndcg5 > best[key].get("ndcg5", 0):
        best[key] = {"model": model, "size": size, "ndcg5": ndcg5, "data": m}

# Afficher tout pour full dataset
print("=== FULL DATASET - toutes metriques ===\n")
for model in ["gat", "sage", "lightgcn"]:
    key = (model, "full")
    if key in best:
        d = best[key]["data"]
        print(f"--- {model.upper()} ---")
        print(f"  RMSE            : {d.get('rmse', 'N/A')}")
        print(f"  MAE             : {d.get('mae', 'N/A')}")
        print(f"  Accuracy        : {d.get('accuracy', 'N/A')}")
        print(f"  Global Precision: {d.get('global_precision', 'N/A')}")
        for k in ["5","10","20"]:
            r = d.get("ranking", {}).get(k, {})
            print(f"  @{k:<3} P={r.get('P',0):.4f}  R={r.get('R',0):.4f}  NDCG={r.get('NDCG',0):.4f}  HR={r.get('HR',0):.4f}  MRR={r.get('MRR',0):.4f}  MAP={r.get('MAP',0):.4f}  n_eval={r.get('n_eval_users','?')}")
        print()

# Hyperparameters
print("\n=== HYPERPARAMETRES ===")
import yaml
for model, fname in [("GraphSAGE","best_sage.yaml"),("LightGCN","best_lightgcn.yaml"),("GAT","best_gat.yaml")]:
    path = os.path.join(r"C:\Users\hafid\Desktop\gnn_recommender\outputs\tuning", fname)
    if os.path.exists(path):
        with open(path) as f:
            params = yaml.safe_load(f)
        print(f"{model}: {params}")
    else:
        path2 = os.path.join(r"C:\Users\hafid\Desktop\gnn_recommender\configs", fname.replace("best_","medium_"))
        if os.path.exists(path2):
            with open(path2) as f:
                params = yaml.safe_load(f)
            print(f"{model} (depuis config): {params}")
        else:
            print(f"{model}: fichier non trouve")
