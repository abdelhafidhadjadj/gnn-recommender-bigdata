#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# hdfs_to_parquet.sh — Convertit les CSV HDFS en Parquet via spark-submit
#
# Lit   : hdfs://namenode:9000/data/{size}/*.csv
# Écrit : hdfs://namenode:9000/parquet/{size}/*.parquet
#
# Prérequis :
#   bash scripts/hdfs_upload.sh --all   (CSV déjà sur HDFS)
#   pipeline spark-master démarré       (cd pipeline && docker compose up -d spark-master spark-worker)
#
# Usage :
#   bash scripts/hdfs_to_parquet.sh --size 1k
#   bash scripts/hdfs_to_parquet.sh --all
# ─────────────────────────────────────────────────────────────────────────────

export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

SIZE=""
ALL=false
SPARK_CONTAINER="spark-master"

while [[ $# -gt 0 ]]; do
    case $1 in
        --size) SIZE="$2"; shift 2 ;;
        --all)  ALL=true;  shift ;;
        *) echo "[WARN] Argument inconnu : $1"; shift ;;
    esac
done

# ── Vérifier que spark-master tourne ─────────────────────────────────────────
if ! docker ps --format "{{.Names}}" | grep -q "^${SPARK_CONTAINER}$"; then
    echo "[ERREUR] spark-master n'est pas démarré."
    echo "  cd pipeline && docker compose up -d spark-master spark-worker"
    exit 1
fi

# ── Copier le script de conversion dans spark-master ─────────────────────────
CONVERT_SCRIPT="/tmp/csv_to_parquet.py"
docker cp spark/preprocessing_spark.py "${SPARK_CONTAINER}:${CONVERT_SCRIPT}"

# ── Fonction conversion d'une taille ─────────────────────────────────────────
convert_size() {
    local size="$1"
    echo ""
    echo "── CSV → Parquet : ${size} ──────────────────────────────────────────"
    echo "   Input  : hdfs://namenode:9000/data/${size}/"
    echo "   Output : hdfs://namenode:9000/parquet/${size}/"

    docker exec "${SPARK_CONTAINER}" \
        /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        "${CONVERT_SCRIPT}" \
        --input-dir  "/data/${size}" \
        --output-dir "/parquet/${size}" \
        --size-tag   "${size}"

    if [ $? -eq 0 ]; then
        echo "   [OK] ${size} converti en Parquet."
    else
        echo "   [ERREUR] Échec conversion ${size}."
    fi
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
if $ALL; then
    for size in 1k 5k 10k 50k 100k full; do
        convert_size "$size"
    done
elif [ -n "$SIZE" ]; then
    convert_size "$SIZE"
else
    echo "Usage :"
    echo "  bash scripts/hdfs_to_parquet.sh --size 1k"
    echo "  bash scripts/hdfs_to_parquet.sh --all"
    exit 1
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Conversion terminée."
echo "  Vérifier : MSYS_NO_PATHCONV=1 docker exec namenode hdfs dfs -ls /parquet/"
echo "══════════════════════════════════════════════════════"
