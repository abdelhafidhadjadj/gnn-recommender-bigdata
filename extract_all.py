import json, os, glob, sys
sys.stdout.reconfigure(encoding='utf-8')

base = r"C:\Users\hafid\Desktop\gnn_recommender\outputs"

# Lire tous les JSON et afficher structure complète d'un exemple
sample_printed = False
best = {}  # (model, size) -> meilleur run

for f in glob.glob(base + r"\*\metrics\*.json"):
    with open(f) as fp:
        m = json.load(fp)
    folder = f.split(os.sep)[-3]
    parts = folder.split("_")
    if len(parts) < 3:
        continue
    model, workers, size = parts[0], parts[1], parts[2]

    if not sample_printed and "gat" in model and "full" in size:
        print("=== STRUCTURE JSON COMPLETE (gat_w1_full) ===")
        print(json.dumps(m, indent=2)[:3000])
        sample_printed = True

    ranking = m.get("ranking", {})
    rating  = m.get("rating", {})
    ndcg5   = ranking.get("5", {}).get("NDCG", 0)

    key = (model, size)
    if key not in best or ndcg5 > best[key].get("ndcg5", 0):
        best[key] = {
            "model": model, "size": size,
            "ranking": ranking, "rating": rating,
            "timings": m.get("timings", {}),
            "ndcg5": ndcg5,
            "n_eval": ranking.get("5", {}).get("n_eval_users", ranking.get("10", {}).get("n_eval_users", "?")),
        }

print("\n\n=== KEYS DISPONIBLES DANS RANKING ===")
for key, v in list(best.items())[:3]:
    print(f"{key}: ranking keys = {list(v['ranking'].keys())}")
    print(f"     rating  keys = {list(v['rating'].keys())}")

print("\n\n=== MEILLEURS RUNS FULL DATASET ===")
for model in ["gat", "sage", "lightgcn"]:
    key = (model, "full")
    if key in best:
        v = best[key]
        print(f"\n{model.upper()} full:")
        print(f"  ranking: {json.dumps(v['ranking'], indent=4)}")
        print(f"  rating:  {json.dumps(v['rating'], indent=4)}")
