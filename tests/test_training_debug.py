"""End-to-end smoke test: scratch training on tiny data, all three models."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import subprocess
import pytest

DATA_DIR = "data/test"


def _run(extra_args, tmp_path):
    cmd = [
        sys.executable, "src/main.py",
        "--mode", "scratch",
        "--data-dir", DATA_DIR,
        "--ckpt-dir", str(tmp_path / "ckpts"),
        "--debug", "--no-amp", "--seed", "42",
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        pytest.fail(f"Command failed:\n{result.stderr[-3000:]}")
    return result.stdout


def test_sage_debug(tmp_path):
    out = _run(["--model", "sage"], tmp_path)
    assert "RMSE" in out


def test_gat_debug(tmp_path):
    out = _run(["--model", "gat"], tmp_path)
    assert "RMSE" in out


def test_lightgcn_debug(tmp_path):
    out = _run(["--model", "lightgcn"], tmp_path)
    assert "RMSE" in out
