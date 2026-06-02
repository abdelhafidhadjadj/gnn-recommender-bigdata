"""
Benchmark orchestrator — runs on the HOST machine.

Two execution backends:

  Docker mode (default):
    preprocessing : spark-submit inside spark-master container (HDFS)
    training      : docker exec trainer python benchmark/train_distributed.py

  No-Docker mode (--no-docker):
    preprocessing : pandas (preprocess_pandas.py) run directly
    training      : python benchmark/train_distributed.py run directly
    No Docker, no HDFS, no Spark needed — pure local execution.

Experiment matrix:
  standard  : pandas preprocessing  + 1  GPU partition x 2 GB
  bigdata N : Spark/pandas preproc  + N  GPU partitions x 2 GB (N=2,3,4)

Usage:
    python benchmark/runner.py --mode standard
    python benchmark/runner.py --mode bigdata
    python benchmark/runner.py --mode bigdata --no-docker
    python benchmark/runner.py --rebuild-csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


COMPOSE_FILE  = "docker/docker-compose.bigdata.yml"
SPARK_APP     = "/opt/spark-apps/preprocessing_spark.py"
SPARK_SUBMIT  = "/opt/spark/bin/spark-submit"
RESULTS_DIR   = Path("outputs/benchmark")

LIMIT_MAP = {
    "1k":   1_000,
    "5k":   5_000,
    "10k":  10_000,
    "50k":  50_000,
    "100k": 100_000,
    "full": None,
}


# ── Docker helpers ─────────────────────────────────────────────────────────────

def compose_up(n_workers: int) -> None:
    cmd = [
        "docker", "compose", "-f", COMPOSE_FILE,
        "up", "-d",
        "--scale", f"spark-worker={n_workers}",
        "--remove-orphans",
    ]
    print(f"  [compose] starting stack with {n_workers} Spark workers…")
    subprocess.run(cmd, check=True)


def compose_down() -> None:
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "down", "--volumes"],
        check=False,
    )


def wait_healthy(service: str, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.check_output(
            ["docker", "inspect", "--format",
             "{{.State.Health.Status}}", service],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if out == "healthy":
            return True
        time.sleep(5)
    return False


# ── Local (no-Docker) helpers ──────────────────────────────────────────────────

def preprocess_pandas_local(size_tag: str, input_dir: str,
                             output_dir: str) -> dict:
    """Run pandas preprocessing directly on the host (no Docker)."""
    print(f"  [pandas-local] preprocessing {size_tag}...")
    limit = LIMIT_MAP.get(size_tag)
    cmd = [sys.executable, "benchmark/preprocess_pandas.py",
           "--input-dir",  input_dir,
           "--output-dir", output_dir,
           "--size-tag",   size_tag]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  [pandas-local] FAILED:\n{result.stderr[-2000:]}")
        return {"error": result.stderr[-500:]}
    print(result.stdout[-300:])
    return {"t_pandas_wall": round(elapsed, 3)}


def train_distributed_local(size_tag: str, model: str, n_workers: int,
                             n_partitions: int, n_epochs: int,
                             run_id: str, processed_dir: str) -> dict:
    """Run train_distributed.py directly on the host (no Docker)."""
    print(f"  [trainer-local] {model} on {size_tag} | "
          f"partitions={n_partitions} epochs={n_epochs}...")
    cmd = [sys.executable, "benchmark/train_distributed.py",
           "--size-tag",      size_tag,
           "--model",         model,
           "--n-partitions",  str(n_partitions),
           "--n-epochs",      str(n_epochs),
           "--n-workers",     str(n_workers),
           "--processed-dir", processed_dir,
           "--run-id",        run_id]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  [trainer-local] FAILED:\n{result.stderr[-2000:]}")
        return {"error": result.stderr[-500:], "t_wall": elapsed}
    print(result.stdout[-1000:])
    return {"t_wall": round(elapsed, 3)}


# ── Docker preprocessing ───────────────────────────────────────────────────────

def preprocess_pandas(size_tag: str, input_dir: str,
                      output_dir: str) -> dict:
    """Run pandas preprocessing inside the trainer container (standard mode)."""
    print(f"  [pandas] preprocessing {size_tag}…")
    limit = LIMIT_MAP.get(size_tag)
    cmd = [
        "docker", "exec", "trainer",
        "python", "benchmark/preprocess_pandas.py",
        "--input-dir",  input_dir,
        "--output-dir", output_dir,
        "--size-tag",   size_tag,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  [pandas] FAILED:\n{result.stderr[-2000:]}")
        return {"error": result.stderr[-500:], "t_pandas_wall": elapsed}
    print(result.stdout[-500:])
    return {"t_pandas_wall": round(elapsed, 3)}


def spark_submit(size_tag: str, input_dir: str,
                 output_dir: str) -> dict:
    """Submit Spark preprocessing job."""
    print(f"  [spark] preprocessing {size_tag}…")
    limit = LIMIT_MAP.get(size_tag)
    cmd = [
        "docker", "exec", "spark-master",
        SPARK_SUBMIT,
        "--master", "spark://spark-master:7077",
        "--deploy-mode", "client",
        "--conf", "spark.executor.memory=2g",
        "--conf", "spark.executor.cores=2",
        SPARK_APP,
        "--input-dir",  input_dir,
        "--output-dir", output_dir,
        "--size-tag",   size_tag,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  [spark] FAILED:\n{result.stderr[-2000:]}")
        return {"error": result.stderr[-500:], "t_spark_wall": elapsed}
    return {"t_spark_wall": round(elapsed, 3)}


# ── Training ───────────────────────────────────────────────────────────────────

def train_distributed(size_tag: str, model: str, n_workers: int,
                      n_partitions: int, n_epochs: int,
                      run_id: str, processed_dir: str) -> dict:
    """Run train_distributed.py inside the trainer container."""
    print(f"  [trainer] {model} on {size_tag} | "
          f"workers={n_workers} partitions={n_partitions} "
          f"epochs={n_epochs}…")
    cmd = [
        "docker", "exec", "trainer",
        "python", "benchmark/train_distributed.py",
        "--size-tag",      size_tag,
        "--model",         model,
        "--n-partitions",  str(n_partitions),
        "--n-epochs",      str(n_epochs),
        "--n-workers",     str(n_workers),
        "--processed-dir", processed_dir,
        "--run-id",        run_id,
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  [trainer] FAILED:\n{result.stderr[-2000:]}")
        return {"error": result.stderr[-500:], "t_wall": elapsed}
    print(result.stdout[-1000:])
    return {"t_wall": round(elapsed, 3)}


# ── Experiment loops ───────────────────────────────────────────────────────────

def run_standard(cfg: dict, dry_run: bool = False,
                 no_docker: bool = False) -> list[dict]:
    """
    Standard baseline: pandas preprocessing + 1 GPU partition.
    no_docker=True: runs everything directly on the host.
    """
    dataset_sizes  = cfg["experiments"]["dataset_sizes"]
    models         = cfg["experiments"]["models"]
    n_epochs       = cfg["experiments"].get("n_epochs", 200)

    if no_docker:
        data_mount     = cfg.get("data_source_local", "data/medium")
        processed_base = "processed/standard"
    else:
        data_mount     = cfg.get("data_mount_in_container",
                                 "/workspace/data/medium")
        processed_base = "/workspace/processed/standard"

    summary: list[dict] = []
    run_counter = 0

    print(f"\n{'='*60}")
    print(f"  MODE: STANDARD (pandas + 1 GPU partition × 2 GB)")
    print(f"{'='*60}")

    for size_tag in dataset_sizes:
        print(f"\n  -- Dataset: {size_tag} --")
        output_dir = f"{processed_base}/{size_tag}"

        if not dry_run:
            preprocess_pandas(size_tag, data_mount, output_dir)

        for model in models:
            run_id = f"std_{run_counter:04d}"
            run_counter += 1
            print(f"\n  ── Model: {model}  run_id={run_id} ──")

            if not dry_run:
                train_distributed(
                    size_tag, model,
                    n_workers=1, n_partitions=1,
                    n_epochs=n_epochs,
                    run_id=run_id,
                    processed_dir="/workspace/processed/standard",
                )

            result_file = (RESULTS_DIR /
                           f"{size_tag}_{model}_w1_p1_{run_id}.json")
            if result_file.exists():
                summary.append(json.loads(result_file.read_text()))
            else:
                summary.append({
                    "size_tag": size_tag, "model": model,
                    "n_workers": 1, "n_partitions": 1,
                })

    return summary


def run_bigdata(cfg: dict, dry_run: bool = False,
                no_docker: bool = False) -> list[dict]:
    """
    Big-data runs: N GPU partitions (n_partitions = n_workers).
    no_docker=True : pandas preprocessing + direct train_distributed.py call.
    no_docker=False: Spark on HDFS + docker exec trainer.
    """
    n_workers_list = cfg["experiments"]["n_workers"]
    dataset_sizes  = cfg["experiments"]["dataset_sizes"]
    models         = cfg["experiments"]["models"]
    n_epochs       = cfg["experiments"].get("n_epochs", 200)
    data_source    = cfg.get("data_source_local", "data/medium")

    summary: list[dict] = []
    run_counter = 0

    for n_workers in n_workers_list:
        n_partitions = n_workers

        print(f"\n{'='*60}")
        print(f"  BIG DATA | partitions={n_partitions} "
              f"({'no-docker' if no_docker else 'docker+spark'})")
        print(f"{'='*60}")

        if not no_docker and not dry_run:
            compose_down()
            time.sleep(5)
            compose_up(n_workers)
            print("  Waiting for services...")
            if not wait_healthy("namenode", timeout=120):
                print("  namenode not healthy — skipping")
                continue
            if not wait_healthy("spark-master", timeout=120):
                print("  spark-master not healthy — skipping")
                continue
            time.sleep(10)

        # HDFS upload once per worker count (outside size_tag loop)
        if not no_docker and not dry_run:
            hdfs_input = "/data/source"
            print(f"  [hdfs] uploading {data_source}...")
            subprocess.run([
                "docker", "exec", "namenode", "bash", "-c",
                f"hdfs dfs -mkdir -p {hdfs_input} && "
                f"hdfs dfs -put -f /input-data/*.csv {hdfs_input}/",
            ], check=False)

        for size_tag in dataset_sizes:
            print(f"\n  -- Dataset: {size_tag} --")

            if no_docker:
                processed_base_spark   = f"processed/bigdata_p{n_partitions}"
                processed_base_trainer = processed_base_spark
                output_dir = f"{processed_base_spark}/{size_tag}"
                if not dry_run:
                    preprocess_pandas_local(size_tag, data_source, output_dir)
            else:
                # Spark container mounts volume at /processed
                # Trainer container mounts same volume at /workspace/processed
                spark_output_dir           = f"/processed/{size_tag}"
                processed_base_trainer     = "/workspace/processed"
                hdfs_input                 = "/data/source"
                if not dry_run:
                    spark_submit(size_tag, hdfs_input, spark_output_dir)

            for model in models:
                run_id = f"bd_{run_counter:04d}"
                run_counter += 1
                print(f"\n  -- Model: {model}  run_id={run_id} --")

                if not dry_run:
                    if no_docker:
                        train_distributed_local(
                            size_tag, model,
                            n_workers=n_workers,
                            n_partitions=n_partitions,
                            n_epochs=n_epochs, run_id=run_id,
                            processed_dir=processed_base_trainer,
                        )
                    else:
                        train_distributed(
                            size_tag, model,
                            n_workers=n_workers,
                            n_partitions=n_partitions,
                            n_epochs=n_epochs, run_id=run_id,
                            processed_dir=processed_base_trainer,
                        )

                result_file = (RESULTS_DIR /
                               f"{size_tag}_{model}"
                               f"_w{n_workers}_p{n_partitions}_{run_id}.json")
                if result_file.exists():
                    summary.append(json.loads(result_file.read_text()))
                else:
                    summary.append({
                        "size_tag": size_tag, "model": model,
                        "n_workers": n_workers, "n_partitions": n_partitions,
                    })

    return summary




# ── Summary CSV ────────────────────────────────────────────────────────────────

def _write_summary_csv(results: list[dict]) -> None:
    import csv
    rows = []
    for r in results:
        t   = r.get("timings", {})
        st  = r.get("spark_timings", {})
        tm  = r.get("test_metrics", {})
        res = r.get("resources", {})

        # ranking metrics are nested: test_metrics["ranking"]["10"]["NDCG"]
        # JSON keys are always strings after json.load()
        ranking    = tm.get("ranking", {})
        ranking_10 = ranking.get("10", ranking.get(10, {}))
        ranking_5  = ranking.get("5",  ranking.get(5,  {}))
        ranking_20 = ranking.get("20", ranking.get(20, {}))

        rows.append({
            "size_tag":          r.get("size_tag"),
            "model":             r.get("model"),
            "n_workers":         r.get("n_workers"),
            "n_partitions":      r.get("n_partitions"),
            "mode":              "standard" if r.get("n_partitions") == 1 else "bigdata",
            "n_users":           r.get("n_users"),
            "n_items":           r.get("n_items"),
            "n_edges":           r.get("n_edges"),
            # Preprocessing timings
            "t_preprocess":      (st.get("t_total_spark") or
                                  st.get("t_total_pandas")),
            "t_load_spark":      st.get("t_load"),
            "t_filter_spark":    st.get("t_filter"),
            "t_graph_spark":     st.get("t_graph"),
            # GNN timings
            "t_load":            t.get("t_load"),
            "t_sbert":           t.get("t_sbert"),
            "t_graph":           t.get("t_graph"),
            "t_train":           t.get("t_train"),
            "t_eval":            t.get("t_eval"),
            "t_total":           t.get("t_total"),
            # GPU resources
            "gpu_util_mean":     res.get("train", {}).get(
                                     "gpu_util_pct", {}).get("mean"),
            "gpu_mem_max_mb":    res.get("train", {}).get(
                                     "gpu_mem_used_mb", {}).get("max"),
            # Model quality — extracted from nested ranking dict
            "ndcg@5":            ranking_5.get("NDCG"),
            "ndcg@10":           ranking_10.get("NDCG"),
            "ndcg@20":           ranking_20.get("NDCG"),
            "recall@10":         ranking_10.get("R"),
            "precision@10":      ranking_10.get("P"),
            "hr@10":             ranking_10.get("HR"),
            "map@10":            ranking_10.get("MAP"),
            "mrr@10":            ranking_10.get("MRR"),
            "rmse":              tm.get("rmse"),
            "mae":               tm.get("mae"),
            "n_eval_users@10":   ranking_10.get("n_eval_users"),
        })

    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[runner] Summary -> {csv_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def rebuild_csv() -> None:
    """Reload all existing JSON results and regenerate summary.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    if results:
        _write_summary_csv(results)
        print(f"[runner] Rebuilt CSV from {len(results)} JSON files.")
    else:
        print("[runner] No JSON files found.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/benchmark.yaml")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rebuild-csv", action="store_true",
                        help="Regenerate summary.csv from existing JSON files "
                             "(no training, no Docker)")
    parser.add_argument("--mode",      default="all",
                        choices=["all", "standard", "bigdata"],
                        help="all=both modes, standard=pandas+1GPU, "
                             "bigdata=N GPU partitions")
    parser.add_argument("--no-docker", action="store_true",
                        help="Run everything locally without Docker/HDFS/Spark. "
                             "Uses pandas preprocessing + direct Python calls.")
    args = parser.parse_args()

    if args.rebuild_csv:
        rebuild_csv()
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exps = cfg["experiments"]
    bd_runs  = (len(exps["n_workers"]) *
                len(exps["dataset_sizes"]) *
                len(exps["models"]))
    std_runs = len(exps["dataset_sizes"]) * len(exps["models"])

    if args.dry_run:
        print("[runner] DRY RUN — experiment plan:")
        print(f"  Mode          : {args.mode}")
        print(f"  Standard runs : {std_runs}  (1 worker, 1 GPU partition)")
        print(f"  BigData runs  : {bd_runs}   "
              f"(workers={exps['n_workers']}, partitions=same)")
        print(f"  Sizes         : {exps['dataset_sizes']}")
        print(f"  Models        : {exps['models']}")
        print(f"  Epochs        : {exps.get('n_epochs', 100)}")
        total = (std_runs if args.mode in ("all", "standard") else 0) + \
                (bd_runs  if args.mode in ("all", "bigdata")  else 0)
        print(f"  TOTAL         : {total} runs")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    if args.mode in ("all", "standard"):
        if not args.no_docker and not args.dry_run:
            compose_down()
            time.sleep(3)
            compose_up(n_workers=1)
            print("  Waiting for trainer...")
            time.sleep(15)
        all_results += run_standard(cfg, args.dry_run, args.no_docker)

    if args.mode in ("all", "bigdata"):
        all_results += run_bigdata(cfg, args.dry_run, args.no_docker)

    if all_results:
        _write_summary_csv(all_results)
    print(f"\n[runner] Done. {len(all_results)} experiments completed.")


if __name__ == "__main__":
    main()
