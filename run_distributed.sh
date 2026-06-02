#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_distributed.sh — Entraînement simulé : standard vs bigdata
#
# Divise le VRAM logiquement en N partitions égales :
#   1 worker  = env standard    (1 partition = VRAM complet)
#   2 workers = bigdata léger   (2 partitions = VRAM/2 chacun)
#   4 workers = bigdata distribué (4 partitions = VRAM/4 chacun)
#
# Usage :
#   bash run_distributed.sh --workers 1 --model sage --size 50k
#   bash run_distributed.sh --workers 2 --model gat  --size 50k
#   bash run_distributed.sh --workers 4 --model lightgcn --size full
# ─────────────────────────────────────────────────────────────────────────────

# ── Valeurs par défaut ────────────────────────────────────────────────────────
WORKERS=1
MODEL=sage
SIZE=50k
DATA_BASE=data/raw
EXTRA_ARGS=""

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)  WORKERS="$2";   shift 2 ;;
        --model)    MODEL="$2";     shift 2 ;;
        --size)     SIZE="$2";      shift 2 ;;
        --data-base) DATA_BASE="$2"; shift 2 ;;
        *)          EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

DATA_DIR="$DATA_BASE/$SIZE"
CKPT_DIR="checkpoints/${MODEL}_w${WORKERS}_${SIZE}"
OUT_DIR="outputs/${MODEL}_w${WORKERS}_${SIZE}"

# ── Affichage ─────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
if [ "$WORKERS" -eq 1 ]; then
    echo "  MODE : STANDARD  (1 worker, VRAM complet)"
else
    echo "  MODE : BIGDATA   ($WORKERS workers logiques, VRAM/$WORKERS chacun)"
fi
echo "  Modèle   : $MODEL"
echo "  Taille   : $SIZE  ($DATA_DIR)"
echo "  Ckpt     : $CKPT_DIR"
echo "  Outputs  : $OUT_DIR"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Vérifier que la partition existe ─────────────────────────────────────────
if [ ! -f "$DATA_DIR/yelp_academic_dataset_review_healthandmedical.csv" ]; then
    echo "[ERREUR] Partition introuvable : $DATA_DIR"
    echo "         Lancez d'abord : python3.13 scripts/partition_dataset.py"
    exit 1
fi

# ── Lancer l'entraînement ─────────────────────────────────────────────────────
# Toujours utiliser torchrun — même pour 1 worker (standard)
# → active set_per_process_memory_fraction(2GB) dans distributed.py
# → comparaison équitable : chaque worker utilise toujours 2 GB fixe
torchrun \
    --nproc_per_node="$WORKERS" \
    --master_port=29500 \
    main.py \
    --model      "$MODEL" \
    --mode       scratch \
    --data-dir   "$DATA_DIR" \
    --ckpt-dir   "$CKPT_DIR" \
    --output-dir "$OUT_DIR" \
    $EXTRA_ARGS

echo ""
echo "[Done] Checkpoint : $CKPT_DIR/model_best.pt"
echo "       Métriques  : $OUT_DIR/metrics/${MODEL}_metrics.json"
