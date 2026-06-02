"""
Partitionne le dataset raw en sous-ensembles de taille fixe.

Pour chaque taille (1k, 5k, 10k, 50k, 100k, full) :
  - Prend les N premières reviews
  - Filtre les users  → uniquement ceux présents dans ces reviews
  - Filtre les businesses → uniquement ceux présents dans ces reviews
  - Sauvegarde dans data/raw/<taille>/ avec les mêmes noms de fichiers

Usage :
    python3.13 scripts/partition_dataset.py
    python3.13 scripts/partition_dataset.py --input-dir data/raw --output-base data/raw
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd


# ── Tailles à générer ─────────────────────────────────────────────────────────
SIZES: dict[str, int | None] = {
    "1k":   1_000,
    "5k":   5_000,
    "10k":  10_000,
    "50k":  50_000,
    "100k": 100_000,
    "full": None,          # tout le dataset
}

# ── Noms de fichiers (convention du projet) ───────────────────────────────────
REVIEW_FILE   = "yelp_academic_dataset_review_healthandmedical.csv"
USER_FILE     = "yelp_academic_dataset_user_healthandmedical.csv"
BUSINESS_FILE = "yelp_academic_dataset_business_healthandmedical.csv"


def partition(input_dir: str, output_base: str) -> None:
    t_start = time.perf_counter()

    # ── Charger une seule fois ────────────────────────────────────────────────
    print(f"\n[Load] Lecture des fichiers depuis '{input_dir}' ...")

    t0 = time.perf_counter()
    review_df   = pd.read_csv(os.path.join(input_dir, REVIEW_FILE))
    user_df     = pd.read_csv(os.path.join(input_dir, USER_FILE))
    business_df = pd.read_csv(os.path.join(input_dir, BUSINESS_FILE))
    print(f"       Reviews   : {len(review_df):>8,}")
    print(f"       Users     : {len(user_df):>8,}")
    print(f"       Businesses: {len(business_df):>8,}")
    print(f"       Chargement: {time.perf_counter() - t0:.1f}s")

    # ── Générer chaque partition ──────────────────────────────────────────────
    for tag, limit in SIZES.items():
        t1 = time.perf_counter()

        # Sous-ensemble reviews
        rev_sub = review_df if limit is None else review_df.head(limit)

        # Filtrer users et businesses selon les ids présents dans rev_sub
        user_ids_in_rev = set(rev_sub["user_id"].unique())
        biz_ids_in_rev  = set(rev_sub["business_id"].unique())

        usr_sub = user_df[user_df["user_id"].isin(user_ids_in_rev)].reset_index(drop=True)
        biz_sub = business_df[business_df["business_id"].isin(biz_ids_in_rev)].reset_index(drop=True)

        # Dossier de sortie
        out_dir = os.path.join(output_base, tag)
        os.makedirs(out_dir, exist_ok=True)

        # Sauvegarde
        rev_sub.to_csv(os.path.join(out_dir, REVIEW_FILE),   index=False)
        usr_sub.to_csv(os.path.join(out_dir, USER_FILE),     index=False)
        biz_sub.to_csv(os.path.join(out_dir, BUSINESS_FILE), index=False)

        elapsed = time.perf_counter() - t1
        print(
            f"\n[{tag:>5}]  reviews={len(rev_sub):>7,}"
            f"  users={len(usr_sub):>6,}"
            f"  businesses={len(biz_sub):>5,}"
            f"  -> {out_dir}  ({elapsed:.1f}s)"
        )

    total = time.perf_counter() - t_start
    print(f"\n[Done] Toutes les partitions generees en {total:.1f}s")
    print(f"       Repertoire : {output_base}/{{1k,5k,10k,50k,100k,full}}/")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Partitionne le dataset Yelp en sous-ensembles.")
    p.add_argument("--input-dir",   default="data/raw",
                   help="Répertoire contenant les 3 CSV complets (défaut: data/raw)")
    p.add_argument("--output-base", default="data/raw",
                   help="Répertoire parent pour les sous-dossiers 1k/5k/… (défaut: data/raw)")
    p.add_argument("--sizes", default=None,
                   help="Tailles à générer, virgule-séparées (défaut: toutes). Ex: 1k,5k,full")
    args = p.parse_args()

    global SIZES
    if args.sizes:
        selected = [s.strip() for s in args.sizes.split(",")]
        SIZES = {k: v for k, v in SIZES.items() if k in selected}
        if not SIZES:
            print(f"[Error] Aucune taille valide parmi : {args.sizes}")
            print(f"        Valeurs autorisées : {', '.join(['1k','5k','10k','50k','100k','full'])}")
            raise SystemExit(1)

    partition(args.input_dir, args.output_base)


if __name__ == "__main__":
    main()
