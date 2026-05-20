# ── Production GPU image ──────────────────────────────────────────────────────
# Target hardware: 4x RTX 2080 Ti (Turing, Compute Capability 7.5)
# Host driver:     CUDA 12.4
# Container base:  CUDA 12.1 (forward-compatible with 12.4 host driver)
#
# Build:  docker build -f docker/Dockerfile.prod -t gnn-rec:prod .
# Run single GPU:
#   docker run --gpus '"device=0"' --rm \
#       -v $(pwd)/data:/workspace/data \
#       -v $(pwd)/checkpoints:/workspace/checkpoints \
#       gnn-rec:prod --mode scratch --model sage
#
# Run 4-GPU DDP (via docker-compose):
#   docker compose -f docker/docker-compose.yml up
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

# RTX 2080 Ti = Turing (CC 7.5) — only compile for this arch to save space
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch Geometric + CUDA 12.1 wheels
# torch-scatter / torch-sparse enable LinkNeighborLoader (Phase 4) on GPU
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# FAISS-GPU (CUDA 12.x compatible)
RUN pip install --no-cache-dir faiss-gpu-cu12

# Project dependencies
RUN pip install --no-cache-dir \
    sentence-transformers \
    optuna \
    pandas numpy scikit-learn

# NCCL tuning for PCIe topology (RTX 2080 Ti has no NVLink)
# P2P access via PCIe; SHM fallback for intra-node
ENV NCCL_P2P_LEVEL=NVL
ENV NCCL_SHM_DISABLE=0
ENV NCCL_SOCKET_IFNAME=eth0

WORKDIR /workspace
COPY . /workspace/

ENTRYPOINT ["python", "src/main.py"]
CMD ["--mode", "scratch", "--model", "sage"]
