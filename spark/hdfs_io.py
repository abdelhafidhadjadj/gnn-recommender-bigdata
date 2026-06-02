"""
Utilities for uploading local CSV files to HDFS and downloading results.
Uses the `hdfs` Python client (pip install hdfs).
Falls back to subprocess hadoop commands when hdfs client is unavailable.
"""
from __future__ import annotations

import os
import subprocess
import json
from pathlib import Path


HDFS_URL = os.environ.get("HDFS_URL", "hdfs://namenode:9000")


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _hdfs_cmd(*args: str) -> str:
    cmd = ["hdfs", "dfs"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"hdfs command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def hdfs_mkdir(hdfs_path: str) -> None:
    _hdfs_cmd("-mkdir", "-p", hdfs_path)


def hdfs_put(local_path: str, hdfs_path: str) -> None:
    """Upload a local file or directory to HDFS."""
    _hdfs_cmd("-put", "-f", local_path, hdfs_path)


def hdfs_get(hdfs_path: str, local_path: str) -> None:
    """Download from HDFS to local path."""
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    _hdfs_cmd("-get", "-f", hdfs_path, local_path)


def hdfs_ls(hdfs_path: str) -> list[str]:
    out = _hdfs_cmd("-ls", hdfs_path)
    lines = [l.split()[-1] for l in out.strip().splitlines() if l.startswith("-")]
    return lines


def hdfs_exists(hdfs_path: str) -> bool:
    result = subprocess.run(
        ["hdfs", "dfs", "-test", "-e", hdfs_path],
        capture_output=True
    )
    return result.returncode == 0


# ── Dataset upload ─────────────────────────────────────────────────────────────

def upload_dataset(local_dir: str, size_tag: str) -> str:
    """
    Upload dataset CSVs to HDFS at /data/{size_tag}/.
    Returns the HDFS input path.
    """
    hdfs_input = f"/data/{size_tag}"
    hdfs_mkdir(hdfs_input)
    local = Path(local_dir)
    for csv_file in local.glob("*.csv"):
        print(f"  Uploading {csv_file.name} → HDFS:{hdfs_input}/")
        hdfs_put(str(csv_file), f"{hdfs_input}/{csv_file.name}")
    return hdfs_input


# ── Meta reading ───────────────────────────────────────────────────────────────

def read_meta(processed_dir: str, size_tag: str) -> dict:
    meta_path = os.path.join(processed_dir, size_tag, "meta.json")
    with open(meta_path) as f:
        return json.load(f)
