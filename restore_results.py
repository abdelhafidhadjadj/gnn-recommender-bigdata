"""
Restaure les résultats protégés depuis results_final/ vers outputs/.
Utilisation : python restore_results.py
              python restore_results.py --run gat_w4_full   (run spécifique)
"""
import os, sys, shutil, glob

ROOT       = os.path.dirname(os.path.abspath(__file__))
SRC_BASE   = os.path.join(ROOT, "results_final")
DST_BASE   = os.path.join(ROOT, "outputs")

# Filtre optionnel
run_filter = None
if len(sys.argv) >= 3 and sys.argv[1] == "--run":
    run_filter = sys.argv[2]

if not os.path.isdir(SRC_BASE):
    print("ERREUR : results_final/ introuvable.")
    sys.exit(1)

restored = 0
for run in sorted(os.listdir(SRC_BASE)):
    if run_filter and run != run_filter:
        continue
    src = os.path.join(SRC_BASE, run, "metrics")
    dst = os.path.join(DST_BASE, run, "metrics")
    if not os.path.isdir(src):
        continue
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*.json")):
        shutil.copy2(f, dst)
        print(f"  Restauré : {run}/metrics/{os.path.basename(f)}")
        restored += 1

print(f"\n{restored} fichier(s) restauré(s) depuis results_final/ → outputs/")
