#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# hdfs_upload.sh — Upload des partitions CSV vers HDFS
#
# Prérequis :
#   cd pipeline && docker compose up -d namenode datanode1 datanode2 spark-master spark-worker
#
# Usage :
#   bash scripts/hdfs_upload.sh --size 1k
#   bash scripts/hdfs_upload.sh --size 50k
#   bash scripts/hdfs_upload.sh --all      # upload toutes les tailles
#
# Structure HDFS créée :
#   hdfs://namenode:9000/data/1k/yelp_academic_dataset_review_healthandmedical.csv
#   hdfs://namenode:9000/data/1k/yelp_academic_dataset_business_healthandmedical.csv
#   hdfs://namenode:9000/data/1k/yelp_academic_dataset_user_healthandmedical.csv
#   hdfs://namenode:9000/data/5k/...
#   ...
# ─────────────────────────────────────────────────────────────────────────────

SIZE=""
ALL=false
DATA_BASE="data/raw"
NAMENODE_CONTAINER="namenode"

# ── Fix Git Bash sur Windows ──────────────────────────────────────────────────
# Git Bash convertit automatiquement /data/1k → C:/data/1k quand le chemin
# est passé à docker exec. MSYS_NO_PATHCONV=1 désactive cette conversion.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

while [[ $# -gt 0 ]]; do
    case $1 in
        --size)  SIZE="$2"; shift 2 ;;
        --all)   ALL=true;  shift ;;
        --data-base) DATA_BASE="$2"; shift 2 ;;
        *) echo "[WARN] Argument inconnu : $1"; shift ;;
    esac
done

# ── Vérifier que le namenode tourne ──────────────────────────────────────────
if ! docker ps --format "{{.Names}}" | grep -q "^${NAMENODE_CONTAINER}$"; then
    echo ""
    echo "[ERREUR] Le container '${NAMENODE_CONTAINER}' n'est pas démarré."
    echo "         Lancez d'abord :"
    echo "           cd pipeline"
    echo "           docker compose up -d namenode datanode1 datanode2 spark-master spark-worker"
    exit 1
fi

# ── Attendre que le namenode soit prêt ───────────────────────────────────────
echo "[HDFS] Attente du namenode..."
for i in $(seq 1 15); do
    if docker exec "${NAMENODE_CONTAINER}" hdfs dfsadmin -report &>/dev/null; then
        echo "[HDFS] Namenode prêt."
        break
    fi
    echo "       ... tentative $i/15"
    sleep 4
done

# ── Fonction upload d'une taille ─────────────────────────────────────────────
upload_size() {
    local size="$1"
    local local_dir="${DATA_BASE}/${size}"

    if [ ! -d "$local_dir" ]; then
        echo "[SKIP] Partition introuvable : ${local_dir}"
        return
    fi

    echo ""
    echo "── Upload ${size} ────────────────────────────────────────────────────"
    echo "   Source : ${local_dir}"
    echo "   HDFS   : /data/${size}/"

    # Créer le répertoire HDFS
    docker exec "${NAMENODE_CONTAINER}" hdfs dfs -mkdir -p "/data/${size}"

    # Upload chaque CSV
    for csv_file in "${local_dir}"/*.csv; do
        fname=$(basename "$csv_file")
        echo "   Uploading ${fname} ..."
        # Copier le fichier dans le container puis sur HDFS
        docker cp "${csv_file}" "${NAMENODE_CONTAINER}:/tmp/${fname}"
        docker exec "${NAMENODE_CONTAINER}" hdfs dfs -put -f "/tmp/${fname}" "/data/${size}/${fname}"
        docker exec "${NAMENODE_CONTAINER}" rm -f "/tmp/${fname}"
    done

    echo "   [OK] ${size} uploadé."
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
if $ALL; then
    for size in 1k 5k 10k 50k 100k full; do
        upload_size "$size"
    done
elif [ -n "$SIZE" ]; then
    upload_size "$SIZE"
else
    echo ""
    echo "Usage :"
    echo "  bash scripts/hdfs_upload.sh --size 1k"
    echo "  bash scripts/hdfs_upload.sh --all"
    exit 1
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  HDFS upload terminé (CSV bruts)."
echo ""
echo "  Optionnel — convertir CSV -> Parquet via Spark :"
echo "  (lecture ~3x plus rapide, typage natif, taille -70%)"
echo ""
echo "    bash scripts/hdfs_to_parquet.sh --size 1k"
echo "    bash scripts/hdfs_to_parquet.sh --all"
echo ""
echo "  Vérifier avec :"
echo "    MSYS_NO_PATHCONV=1 docker exec namenode hdfs dfs -ls /data/"
echo "    MSYS_NO_PATHCONV=1 docker exec namenode hdfs dfs -ls /parquet/"
echo "══════════════════════════════════════════════════════"
