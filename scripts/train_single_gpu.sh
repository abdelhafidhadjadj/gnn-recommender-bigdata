#!/usr/bin/env bash
# ── Single-GPU training ───────────────────────────────────────────────────────
# Runs on any machine with >=1 GPU.
# Hardware auto-detection handles AMP and batch sizing.
#
# Usage:
#   bash scripts/train_single_gpu.sh
#   bash scripts/train_single_gpu.sh --model gat --epochs 300
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"

python main.py \
    --mode     scratch \
    --model    "${MODEL:-sage}" \
    --data-dir "${DATA_DIR:-data/raw}" \
    --seed     "${SEED:-42}" \
    "$@"
