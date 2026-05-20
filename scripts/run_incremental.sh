#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
MODEL=${1:-sage}
CKPT=${2:-checkpoints/sage/sage_best.pt}
NEW_DATA=${3:-data/new_data/yelp_academic_dataset_review_healthandmedical.csv}
python src/main.py --model "$MODEL" --mode incremental \
    --ckpt "$CKPT" --new-data "$NEW_DATA" \
    --ckpt-dir "checkpoints/$MODEL" --no-amp
