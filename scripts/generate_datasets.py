"""
Prepare benchmark datasets from real Yelp data by subsampling.

Source data (already in repo):
  data/test/    — ~4 334 reviews  (Yelp real)
  data/medium/  — ~40 322 reviews (Yelp real)

Outputs (symlinks or copies):
  data/1k/   — 1 000 reviews  (sampled from test)
  data/5k/   — 4 334 reviews  (= full test, no sampling needed)
  data/10k/  — 10 000 reviews (sampled from medium)
  data/40k/  — 40 322 reviews (= full medium, no sampling needed)

Sampling is stratified by user to preserve interaction density.

Usage:
    python scripts/generate_datasets.py
    python scripts/generate_datasets.py --sizes 1k 5k --seed 42
    python scripts/generate_datasets.py --show-stats
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"

BIZ_FILE  = "yelp_academic_dataset_business_healthandmedical.csv"
REV_FILE  = "yelp_academic_dataset_review_healthandmedical.csv"
USR_FILE  = "yelp_academic_dataset_user_healthandmedical.csv"

# Source -> target mapping
SOURCES = {
    "test":   DATA_DIR / "test",
    "medium": DATA_DIR / "medium",
}

SIZE_MAP: dict[str, tuple[str, int | None]] = {
    # tag   : (source_dir, n_reviews or None = keep all)
    "1k":   ("test",   1_000),
    "5k":   ("test",   None),    # full test   ~ 4 334
    "10k":  ("medium", 10_000),
    "full": ("medium", None),    # full medium ~ 40 322
}


def load_source(source: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src = SOURCES[source]
    biz = pd.read_csv(src / BIZ_FILE)
    rev = pd.read_csv(src / REV_FILE)
    usr = pd.read_csv(src / USR_FILE)
    return biz, usr, rev


def subsample_reviews(rev: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows, preserving temporal order and user diversity."""
    if n >= len(rev):
        return rev.copy()
    # Sort by date (if available) so we keep the most recent interactions
    if "date" in rev.columns:
        rev = rev.sort_values("date", ascending=False)
    # Stratified: keep at least 1 interaction per user, fill up to n
    users = rev["user_id"].unique()
    # One per user first
    first = rev.groupby("user_id").first().reset_index()
    remaining = rev[~rev.index.isin(first.index)]
    extra_n = max(0, n - len(first))
    if extra_n > 0 and len(remaining) > 0:
        extra = remaining.sample(n=min(extra_n, len(remaining)), random_state=seed)
        sampled = pd.concat([first, extra])
    else:
        sampled = first.head(n)
    return sampled.head(n).reset_index(drop=True)


def filter_consistent(biz: pd.DataFrame, usr: pd.DataFrame,
                      rev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep only users and businesses present in the sampled reviews."""
    active_biz  = rev["business_id"].unique()
    active_usr  = rev["user_id"].unique()
    biz_out = biz[biz["business_id"].isin(active_biz)].reset_index(drop=True)
    usr_out = usr[usr["user_id"].isin(active_usr)].reset_index(drop=True)
    return biz_out, usr_out, rev


def write_dataset(biz: pd.DataFrame, usr: pd.DataFrame, rev: pd.DataFrame,
                  out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    biz.to_csv(out_dir / BIZ_FILE, index=False)
    usr.to_csv(out_dir / USR_FILE, index=False)
    rev.to_csv(out_dir / REV_FILE, index=False)


def print_stats() -> None:
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  {'Dataset':<12} {'Reviews':>9} {'Users':>8} {'Businesses':>12}")
    print(sep)
    for tag in [*SOURCES.keys(), *SIZE_MAP.keys()]:
        d = DATA_DIR / tag
        rev_f = d / REV_FILE
        usr_f = d / USR_FILE
        biz_f = d / BIZ_FILE
        if rev_f.exists():
            n_rev = len(pd.read_csv(rev_f))
            n_usr = len(pd.read_csv(usr_f)) if usr_f.exists() else "?"
            n_biz = len(pd.read_csv(biz_f)) if biz_f.exists() else "?"
            print(f"  {tag:<12} {n_rev:>9,} {str(n_usr):>8} {str(n_biz):>12}")
        else:
            print(f"  {tag:<12} {'(missing)':>9}")
    print(f"{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=list(SIZE_MAP.keys()),
                        choices=list(SIZE_MAP.keys()))
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--show-stats", action="store_true",
                        help="Print dataset stats and exit")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directories")
    args = parser.parse_args()

    if args.show_stats:
        print_stats()
        return

    # Check sources exist
    for name, path in SOURCES.items():
        if not (path / REV_FILE).exists():
            print(f"[WARNING] Source '{name}' not found at {path}. "
                  f"Some sizes may be skipped.")

    print("\nPreparing benchmark datasets (seed=%d):" % args.seed)
    for tag in args.sizes:
        source_name, n_reviews = SIZE_MAP[tag]
        src_path = SOURCES[source_name]
        out_dir  = DATA_DIR / tag

        if out_dir.exists() and not args.force:
            n_existing = len(pd.read_csv(out_dir / REV_FILE))
            print(f"  {tag:<6} -> already exists ({n_existing:,} reviews) — skip "
                  f"(use --force to overwrite)")
            continue

        if not (src_path / REV_FILE).exists():
            print(f"  {tag:<6} -> source '{source_name}' missing — skip")
            continue

        biz, usr, rev = load_source(source_name)
        if n_reviews is not None:
            rev = subsample_reviews(rev, n_reviews, args.seed)
        biz, usr, rev = filter_consistent(biz, usr, rev)
        write_dataset(biz, usr, rev, out_dir)
        print(f"  {tag:<6} -> {len(rev):>6,} reviews  "
              f"{len(usr):>5,} users  {len(biz):>5,} businesses  -> data/{tag}/")

    print("\nDone. Run --show-stats to verify:")
    print("  python scripts/generate_datasets.py --show-stats")


if __name__ == "__main__":
    main()
