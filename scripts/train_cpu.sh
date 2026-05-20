#!/usr/bin/env bash
# ── CPU training (debug mode) ─────────────────────────────────────────────────
# Runs on the dev laptop (no GPU).  Hardware auto-detection sets:
#   emb_dim=16, epochs=3, batch=32, SBERT skipped.
#
# Usage:
#   bash scripts/train_cpu.sh
#   bash scripts/train_cpu.sh --model gat --data-dir data/raw
set -euo pipefail

cd "$(dirname "$0")/.."

python main.py \
    --mode  scratch \
    --model "${MODEL:-sage}" \
    --data-dir "${DATA_DIR:-data/test}" \
    --debug \
    --no-amp \
    --seed 42 \
    "$@"
