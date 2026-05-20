#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
MODEL=${1:-sage}
TRIALS=${2:-30}
python src/main.py --model "$MODEL" --mode tune \
    --data-dir data/medium --trials "$TRIALS" --no-amp
