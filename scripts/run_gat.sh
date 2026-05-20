#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
DATA_DIR=${1:-data/medium}
python src/main.py --model gat --mode scratch --data-dir "$DATA_DIR" \
    --epochs 100 --no-amp --ckpt-dir checkpoints/gat
