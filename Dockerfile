# ── Production GPU image ──────────────────────────────────────────────────────
# Target hardware: RTX 3070 Laptop GPU (Ampere, Compute Capability 8.6)
# Host driver:     CUDA 13.0 (581.83)
# Container base:  CUDA 12.4 (forward-compatible avec driver 13.0)
#
# Build:
#   docker build -t gnn-rec:prod .
#
# Run standard (1 worker, VRAM complet) :
#   docker run --gpus all --rm \
#       -v $(pwd)/data:/workspace/data \
#       -v $(pwd)/checkpoints:/workspace/checkpoints \
#       -v $(pwd)/outputs:/workspace/outputs \
#       gnn-rec:prod --mode scratch --model sage --data-dir data/raw/50k
#
# Run distribué simulé (4 workers logiques) :
#   docker compose -f docker/docker-compose.yml up
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# RTX 3070 = Ampere (CC 8.6)
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
        openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# PySpark a besoin de JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ── Dépendances du projet ─────────────────────────────────────────────────────
# Copier requirements-docker.txt en premier pour profiter du cache Docker
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

# PySpark — copié depuis l'hôte pour éviter les timeout réseau Docker Desktop.
# Télécharger une fois sur l'hôte avant de builder :
#   curl -L "https://files.pythonhosted.org/packages/9a/90/cb80c8cf194958ab9a3242851c62fa5aef1a0b42f2d9642f1e2eca098005/pyspark-3.5.3.tar.gz" -o pyspark-3.5.3.tar.gz
COPY pyspark-3.5.3.tar.gz /tmp/pyspark-3.5.3.tar.gz
RUN pip install --no-cache-dir /tmp/pyspark-3.5.3.tar.gz \
    && rm /tmp/pyspark-3.5.3.tar.gz

# torch-scatter / torch-sparse (CUDA 12.4 wheels)
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster pyg-lib \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# FAISS : faiss-gpu-cu12 entre en conflit avec libcublas de l'image PyTorch base.
# faiss-cpu est utilisé dans le container — fonctionnellement identique car
# use_item_item_edges=False dans GraphConfig (FAISS n'est pas appelé pendant l'entraînement).
RUN pip install --no-cache-dir faiss-cpu

# gloo backend pour simulation single-GPU (pas besoin de NCCL multi-GPU)
ENV NCCL_P2P_LEVEL=NVL
ENV NCCL_SHM_DISABLE=0
ENV NCCL_SOCKET_IFNAME=eth0

# Cache SBERT dans le container (monté comme volume pour persistance)
ENV SBERT_CACHE_DIR=/workspace/.sbert_cache

WORKDIR /workspace
COPY . /workspace/

ENTRYPOINT ["python", "src/main.py"]
CMD ["--mode", "scratch", "--model", "sage"]
