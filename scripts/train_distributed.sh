#!/usr/bin/env bash
# ── 4-GPU distributed training via torchrun ──────────────────────────────────
# Launches 4 worker processes (one per GPU) with NCCL gradient synchronization.
# Equivalent to running inside the Docker container via docker-compose.
#
# Requirements:
#   - 4x GPU visible to the system
#   - NCCL installed (comes with PyTorch CUDA builds)
#   - Port 29500 free on localhost
#
# Usage:
#   bash scripts/train_distributed.sh
#   bash scripts/train_distributed.sh --model gat --epochs 200
#
# Override GPU count:
#   NUM_GPUS=2 bash scripts/train_distributed.sh
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[DDP] Launching ${NUM_GPUS} processes on port ${MASTER_PORT}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --standalone \
    --master_port="${MASTER_PORT}" \
    main.py \
    --mode     scratch \
    --model    "${MODEL:-sage}" \
    --data-dir "${DATA_DIR:-data/raw}" \
    --seed     "${SEED:-42}" \
    "$@"
