#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
MODEL=${1:-sage}
DATA_DIR=${2:-data/raw}
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --standalone \
    --master_port="$MASTER_PORT" \
    src/main.py \
    --model "$MODEL" \
    --mode scratch \
    --data-dir "$DATA_DIR" \
    --ckpt-dir "checkpoints/$MODEL"
