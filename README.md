# GNN Recommender System

A production-ready Graph Neural Network recommender system supporting **GraphSAGE**, **GAT**, and **LightGCN**. Designed for Yelp Health & Medical data with scratch training, incremental retraining, hyperparameter tuning, and 4-GPU distributed training.

---

## Supported Models

| Model | Description | Best for |
|-------|-------------|----------|
| **GraphSAGE** | 2-layer inductive GNN with residual connections | General use, robust baseline |
| **GAT** | Graph Attention Network with configurable heads | When attention weighting matters |
| **LightGCN** | Lightweight collaborative filtering GNN | Pure CF signal, fast training |

---

## Project Structure

```
gnn_recommender/
├── src/                    # All source code
│   ├── main.py             # CLI entry point
│   ├── config.py           # Typed dataclass configuration
│   ├── models/             # GraphSAGE, GAT, LightGCN
│   ├── data/               # Loading, preprocessing, graph building
│   ├── training/           # Trainer, BPR loss, DDP, incremental
│   ├── evaluation/         # Ranking metrics (Precision@K, NDCG@K, ...)
│   ├── tuning/             # Optuna hyperparameter search
│   └── utils/              # Checkpoints, hardware, seeding
│
├── configs/                # YAML configuration presets
│   ├── default.yaml
│   ├── cpu_debug.yaml
│   ├── medium_sage.yaml
│   ├── medium_gat.yaml
│   ├── medium_lightgcn.yaml
│   ├── server_4gpu.yaml
│   └── tuning.yaml
│
├── data/
│   ├── test/               # Small dataset (300 users, 150 businesses)
│   ├── medium/             # Medium dataset (2K users, 500 businesses)
│   ├── raw/                # Production data (place Yelp CSVs here)
│   └── new_data/           # New batch for incremental retraining
│
├── scripts/                # Shell scripts for common workflows
│   ├── run_debug.sh
│   ├── run_sage.sh / run_gat.sh / run_lightgcn.sh
│   ├── run_tuning.sh
│   ├── run_incremental.sh
│   └── run_server_4gpu.sh
│
├── tests/                  # pytest test suite
├── checkpoints/            # Saved model checkpoints (gitignored)
├── outputs/                # Logs, metrics, plots (gitignored)
├── docker/                 # Docker build files
├── Dockerfile              # Production GPU image
├── docker-compose.yml      # 4-GPU server deployment
└── archive/                # Old experiments (not production code)
```

---

## Dataset Format

Three CSV files are required:

**businesses** (`yelp_academic_dataset_business_healthandmedical.csv`):
```
business_id, name, address, city, state, postal_code, stars, review_count, is_open, categories
```

**users** (`yelp_academic_dataset_user_healthandmedical.csv`):
```
user_id, name, review_count, yelping_since, average_stars, fans, useful, funny, cool
```

**reviews** (`yelp_academic_dataset_review_healthandmedical.csv`):
```
review_id, user_id, business_id, stars, date, text, useful, funny, cool
```

Place files in `data/raw/` for production or `data/medium/` for local testing.

---

## Installation (Local)

```bash
# 1. Clone the repository
git clone <repo-url>
cd gnn_recommender

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

For GPU support on the server:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

---

## Installation (Docker)

```bash
# Build production GPU image
docker compose build

# Or build dev CPU image
docker build -f docker/Dockerfile.dev -t gnn_recommender_dev .
```

---

## Quick Start

### Debug mode (CPU, ~30 seconds)

```bash
python src/main.py --model sage --mode scratch --data-dir data/test --debug --no-amp
```

Expected output:
```
Hardware tier : debug  (emb_dim=16, epochs=3)
[Train] SAGE  emb_dim=16  lr=0.005  epochs=3
...
RMSE  (sigmoid [1,5]): ~1.0
Precision@5: ~0.01
```

---

## Training

### Train GraphSAGE

```bash
python src/main.py --model sage --mode scratch \
    --data-dir data/medium --epochs 100 \
    --no-amp --ckpt-dir checkpoints/sage
```

### Train GAT

```bash
python src/main.py --model gat --mode scratch \
    --data-dir data/medium --epochs 100 \
    --no-amp --ckpt-dir checkpoints/gat
```

### Train LightGCN

```bash
python src/main.py --model lightgcn --mode scratch \
    --data-dir data/medium --epochs 100 \
    --no-amp --ckpt-dir checkpoints/lightgcn
```

---

## Incremental Retraining

Use when new interactions arrive — no full retraining needed.

**Step 1:** Place new review CSV in `data/new_data/`

**Step 2:** Run incremental fine-tuning:

```bash
python src/main.py --model sage --mode incremental \
    --ckpt checkpoints/sage/sage_best.pt \
    --new-data data/new_data/yelp_academic_dataset_review_healthandmedical.csv \
    --ckpt-dir checkpoints/sage --no-amp
```

What happens:
- Loads the trained model and entity encoders from the checkpoint
- Detects brand-new users and businesses in the new CSV
- Extends the embedding table (old weights preserved)
- Fine-tunes for 20 epochs at 10% of the original learning rate
- Mixes new interactions with a replay buffer (30%) to avoid forgetting
- Saves a new versioned checkpoint

**New-data-only mode** (skip replay buffer and old interactions):

```bash
python src/main.py --model sage --mode incremental \
    --ckpt checkpoints/sage/sage_best.pt \
    --new-data data/new_data/yelp_academic_dataset_review_healthandmedical.csv \
    --ckpt-dir checkpoints/sage --no-amp --new-data-only
```

---

## Hyperparameter Tuning

```bash
# Tune GraphSAGE (30 Optuna trials)
python src/main.py --model sage --mode tune \
    --data-dir data/medium --trials 30 --no-amp

# Tune GAT
python src/main.py --model gat --mode tune \
    --data-dir data/medium --trials 30 --no-amp

# Tune LightGCN
python src/main.py --model lightgcn --mode tune \
    --data-dir data/medium --trials 30 --no-amp
```

Results are saved to:
- `outputs/tuning/best_sage.yaml` — best hyperparameters
- `outputs/tuning/study_sage.db` — full Optuna study
- `outputs/tuning/trials_sage.csv` — trial history

Train with best parameters:
```bash
python src/main.py --model sage --mode scratch \
    --data-dir data/medium --epochs 100 --no-amp \
    --ckpt-dir checkpoints/sage
```

---

## 4-GPU Server Training

### Direct (without Docker)

```bash
torchrun --nproc_per_node=4 --standalone \
    src/main.py --model sage --mode scratch \
    --data-dir data/medium --ckpt-dir checkpoints/sage
```

### With Docker

```bash
# GraphSAGE
docker compose run --rm gnn_train \
    torchrun --nproc_per_node=4 --standalone \
    src/main.py --model sage --mode scratch \
    --data-dir data/medium --ckpt-dir checkpoints/sage

# GAT
docker compose run --rm gnn_train \
    torchrun --nproc_per_node=4 --standalone \
    src/main.py --model gat --mode scratch \
    --data-dir data/medium --ckpt-dir checkpoints/gat

# LightGCN
docker compose run --rm gnn_train \
    torchrun --nproc_per_node=4 --standalone \
    src/main.py --model lightgcn --mode scratch \
    --data-dir data/medium --ckpt-dir checkpoints/lightgcn

# Incremental retraining
docker compose run --rm gnn_train \
    python src/main.py --model sage --mode incremental \
    --ckpt checkpoints/sage/sage_best.pt \
    --new-data data/new_data/yelp_academic_dataset_review_healthandmedical.csv \
    --ckpt-dir checkpoints/sage
```

---

## Hardware Auto-Detection

The system automatically adapts based on available hardware:

| Tier | Detected when | Behaviour |
|------|--------------|-----------|
| `debug` | `--debug` flag | emb_dim=16, epochs=3, batch=32, AMP off |
| `cpu` | No GPU found | batch=128, emb_dim=64, AMP off |
| `single_gpu` | 1 GPU found | CUDA + AMP enabled, batch scaled to VRAM |
| `multi_gpu` | 2+ GPUs found | DDP via torchrun, NCCL backend |

Override any setting with CLI flags (`--epochs`, `--emb-dim`, `--lr`).

---

## Checkpoints

Checkpoints are saved in `checkpoints/<model>/`:

```
checkpoints/sage/
  sage_best.pt          ← best validation score
  sage_v001_e0020.pt    ← periodic save
  sage_v002_e0040.pt
  sage_v003_e0050.pt    ← latest
```

Each checkpoint stores:
- Model weights (DDP-safe)
- Optimizer + scheduler + AMP scaler state
- Full model architecture config (for rebuilding without CLI)
- Entity encoders (DynamicLabelEncoder — needed for incremental mode)
- Training interactions (for incremental graph reconstruction)

Load a checkpoint for evaluation:
```bash
python src/main.py --model sage --mode evaluate \
    --ckpt checkpoints/sage/sage_best.pt
```

---

## Metrics

Final evaluation reports:

**Regression:**
- RMSE and MAE (sigmoid-scaled to [1, 5] rating range)

**Ranking (all users with >= 1 relevant test item):**
- Precision@K, Recall@K, F1@K, NDCG@K, MAP@K, MRR@K, HR@K

**Baselines:**
- Popularity recommender
- Random recommender

**K-filter variant:** only includes users who have at least K test items — a stricter but more reliable evaluation.

---

## Comparing Models

After training all three models, run:

```bash
python src/main.py --model sage --mode evaluate --ckpt checkpoints/sage/sage_best.pt
python src/main.py --model gat  --mode evaluate --ckpt checkpoints/gat/gat_best.pt
python src/main.py --model lightgcn --mode evaluate --ckpt checkpoints/lightgcn/lightgcn_best.pt
```

---

## Generate Synthetic Data

Generate medium dataset (2K users, 500 businesses, ~40K reviews):
```bash
python scripts/generate_medium_data.py
```

Generate an incremental new-batch (200 new users, ~5K reviews):
```bash
python scripts/generate_new_batch.py
```

---

## Run Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Recommended Workflow

```
1.  Debug test locally
    python src/main.py --model sage --mode scratch --data-dir data/test --debug --no-amp

2.  Train all models on medium data
    python src/main.py --model sage       --mode scratch --data-dir data/medium --epochs 100 --no-amp --ckpt-dir checkpoints/sage
    python src/main.py --model gat        --mode scratch --data-dir data/medium --epochs 100 --no-amp --ckpt-dir checkpoints/gat
    python src/main.py --model lightgcn   --mode scratch --data-dir data/medium --epochs 100 --no-amp --ckpt-dir checkpoints/lightgcn

3.  Compare models
    Evaluate all three and compare Precision@K / NDCG@K

4.  Tune best model
    python src/main.py --model sage --mode tune --data-dir data/medium --trials 30 --no-amp

5.  Retrain with best config on full data
    python src/main.py --model sage --mode scratch --data-dir data/raw --epochs 200 --ckpt-dir checkpoints/sage

6.  Deploy to 4-GPU server via Docker
    docker compose run --rm gnn_train torchrun --nproc_per_node=4 --standalone src/main.py ...

7.  When new data arrives: incremental retraining (minutes, not hours)
    python src/main.py --model sage --mode incremental --ckpt checkpoints/sage/sage_best.pt --new-data data/new_data/...
```

---

## Troubleshooting

**`FileNotFoundError: checkpoints/.../sage_best.pt`**
The scratch training must complete before incremental mode. Ensure the training finished and printed `[Checkpoint] New best -> sage_best.pt`.

**`min_epochs > num_epochs` — validation never ran**
Use `--epochs` value higher than the default `min_epochs` (80). For quick tests use `--debug` or `--epochs 50`.

**`RuntimeError: NCCL backend not available`**
NCCL requires NVIDIA GPUs. On CPU-only machines, DDP is not supported — run without `torchrun`.

**`SBERT warning: embeddings.position_ids UNEXPECTED`**
Safe to ignore — this is a model architecture mismatch warning from sentence-transformers, does not affect functionality.

**`OOM (CUDA out of memory)`**
Reduce `--emb-dim` or use `--no-amp` to disable AMP. The system will retry with half batch size automatically.

**Low Precision@K metrics**
- Use more epochs (100+) for meaningful convergence
- Check that the test set is not too sparse (use medium or raw dataset, not test dataset)
- Run baselines to verify GNN beats random recommender

---

## Citation / Credits

Models implemented based on:
- GraphSAGE: Hamilton et al., 2017
- GAT: Veličković et al., 2018
- LightGCN: He et al., 2020
- BPR Loss: Rendle et al., 2009
