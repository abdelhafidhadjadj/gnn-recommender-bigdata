#!/usr/bin/env bash
# ── Benchmark entry point ──────────────────────────────────────────────────────
#
# Generates datasets (if missing), then launches the full experiment matrix
# via benchmark/runner.py (which drives Docker compose).
#
# Usage:
#   bash scripts/run_benchmark.sh                    # full 27-run matrix
#   bash scripts/run_benchmark.sh --dry-run          # print plan, no execution
#   bash scripts/run_benchmark.sh --sizes 1k 5k      # subset of sizes
#   bash scripts/run_benchmark.sh --workers 2 3      # subset of workers
#
# Requirements:
#   - Docker Desktop running with NVIDIA runtime enabled
#   - Python + pip (for generate_datasets.py + reporter.py on host)
#   - pip install pyyaml pandas matplotlib seaborn (host-side reporter deps)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Args ───────────────────────────────────────────────────────────────────────
DRY_RUN=false
SIZES=(1k 5k 10k)
WORKERS=(2 3 4)
CONFIG="configs/benchmark.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY_RUN=true; shift ;;
    --sizes)     shift; SIZES=(); while [[ $# -gt 0 && "$1" != --* ]]; do SIZES+=("$1"); shift; done ;;
    --workers)   shift; WORKERS=(); while [[ $# -gt 0 && "$1" != --* ]]; do WORKERS+=("$1"); shift; done ;;
    --config)    CONFIG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  GNN Recommender — Distributed Benchmark"
echo "  Workers : ${WORKERS[*]}"
echo "  Sizes   : ${SIZES[*]}"
echo "  Config  : $CONFIG"
echo "  Dry-run : $DRY_RUN"
echo "============================================================"

# ── Step 1: Generate datasets if missing ──────────────────────────────────────
echo ""
echo "── Step 1: Generating datasets ──────────────────────────────"
MISSING_SIZES=()
for SIZE in "${SIZES[@]}"; do
  if [[ ! -d "data/$SIZE" ]]; then
    MISSING_SIZES+=("$SIZE")
  fi
done

if [[ ${#MISSING_SIZES[@]} -gt 0 ]]; then
  echo "  Generating: ${MISSING_SIZES[*]}"
  python scripts/generate_datasets.py --sizes "${MISSING_SIZES[@]}"
else
  echo "  All datasets present — skipping generation"
fi

# ── Step 2: Build Docker images ───────────────────────────────────────────────
if [[ "$DRY_RUN" == false ]]; then
  echo ""
  echo "── Step 2: Building Docker images ───────────────────────────"
  docker compose -f docker/docker-compose.bigdata.yml build trainer
fi

# ── Step 3: Run experiments ───────────────────────────────────────────────────
echo ""
echo "── Step 3: Running experiments ──────────────────────────────"
START_TIME=$(date +%s)

RUNNER_ARGS="--config $CONFIG"
if [[ "$DRY_RUN" == true ]]; then
  RUNNER_ARGS="$RUNNER_ARGS --dry-run"
fi

python benchmark/runner.py $RUNNER_ARGS

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ── Step 4: Generate report ───────────────────────────────────────────────────
echo ""
echo "── Step 4: Generating report ────────────────────────────────"
if [[ -f "outputs/benchmark/summary.csv" ]]; then
  python benchmark/reporter.py
  echo ""
  echo "  Plots → outputs/benchmark/plots/"
else
  echo "  No summary.csv found — skipping report"
fi

echo ""
echo "============================================================"
printf "  Total benchmark time: %02d:%02d:%02d\n" \
  $((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60))
echo "  Results → outputs/benchmark/"
echo "============================================================"
