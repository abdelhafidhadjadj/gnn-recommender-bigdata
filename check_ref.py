import json, sys
sys.stdout.reconfigure(encoding='utf-8')

runs = ["sage_w4_full", "gat_w4_full", "lightgcn_w4_full"]

for run in runs:
    model = run.split("_")[0]
    try:
        with open(f'results_final/{run}/metrics/{model}_metrics.json') as f:
            ref = json.load(f)
        with open(f'outputs/{run}/metrics/{model}_metrics.json') as f:
            cur = json.load(f)

        print(f"\n{'='*60}")
        print(f"  {run}")
        print(f"{'='*60}")
        print(f"  {'Metric':<12} {'results_final':>15} {'output':>15} {'delta':>12}  OK?")
        print(f"  {'-'*58}")

        all_ok = True
        for k in ['5','10','20']:
            for m in ['P','R','NDCG']:
                r = ref['ranking'][k][m]
                c = cur['ranking'][k][m]
                d = c - r
                is_ok = abs(d) <= 0.011
                flag = "OK" if is_ok else "!! TROP GRAND"
                if not is_ok:
                    all_ok = False
                print(f"  {m}@{k:<9} {r:>15.6f} {c:>15.6f} {d:>+12.6f}  {flag}")

        print(f"\n  >> {'CORRECT - variation dans +-0.01' if all_ok else 'PROBLEME - variation trop grande'}")

    except FileNotFoundError as e:
        print(f"\n{run}: fichier manquant — {e}")
