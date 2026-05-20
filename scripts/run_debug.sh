#!/usr/bin/env bash
# Quick smoke test — runs in ~30 seconds on CPU
set -e
cd "$(dirname "$0")/.."
python src/main.py --model sage --mode scratch --data-dir data/test --debug --no-amp --ckpt-dir checkpoints/debug
