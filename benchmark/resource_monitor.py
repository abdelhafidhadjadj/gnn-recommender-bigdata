"""
Background resource monitor: GPU (nvidia-smi), CPU, RAM.
Samples every N seconds, stores time-series, reports stats per phase.

Usage:
    monitor = ResourceMonitor(interval=2.0)
    monitor.start()
    monitor.mark("t_train")
    # ... do work ...
    monitor.mark("t_train_end")
    monitor.stop()
    stats = monitor.phase_stats("t_train", "t_train_end")
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sample:
    ts: float
    gpu_util_pct: float       # 0-100
    gpu_mem_used_mb: float
    gpu_mem_total_mb: float
    cpu_pct: float            # system-wide %
    ram_used_gb: float
    ram_total_gb: float


class ResourceMonitor:
    def __init__(self, interval: float = 2.0):
        self.interval   = interval
        self._samples:  list[Sample] = []
        self._markers:  dict[str, float] = {}
        self._stop_evt  = threading.Event()
        self._thread:   Optional[threading.Thread] = None

    # ── Control ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)

    def mark(self, name: str) -> None:
        self._markers[name] = time.time()

    # ── Sampling loop ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            s = self._collect()
            if s:
                self._samples.append(s)
            self._stop_evt.wait(self.interval)

    def _collect(self) -> Optional[Sample]:
        try:
            gpu = self._gpu_stats()
            cpu, ram_used, ram_total = self._cpu_ram_stats()
            return Sample(
                ts=time.time(),
                gpu_util_pct=gpu["util"],
                gpu_mem_used_mb=gpu["mem_used"],
                gpu_mem_total_mb=gpu["mem_total"],
                cpu_pct=cpu,
                ram_used_gb=ram_used,
                ram_total_gb=ram_total,
            )
        except Exception:
            return None

    # ── GPU via nvidia-smi ─────────────────────────────────────────────────────

    @staticmethod
    def _gpu_stats() -> dict:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()[0]
        util, mem_used, mem_total = [float(x.strip()) for x in out.split(",")]
        return {"util": util, "mem_used": mem_used, "mem_total": mem_total}

    # ── CPU / RAM via psutil ───────────────────────────────────────────────────

    @staticmethod
    def _cpu_ram_stats() -> tuple[float, float, float]:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        return cpu, mem.used / 1e9, mem.total / 1e9

    # ── Stats for a time window ────────────────────────────────────────────────

    def phase_stats(self, start_marker: str, end_marker: str) -> dict:
        t0 = self._markers.get(start_marker, 0.0)
        t1 = self._markers.get(end_marker, time.time())
        window = [s for s in self._samples if t0 <= s.ts <= t1]
        if not window:
            return {}

        def _agg(vals: list[float]) -> dict:
            return {
                "mean": round(sum(vals) / len(vals), 2),
                "max":  round(max(vals), 2),
                "min":  round(min(vals), 2),
            }

        utils    = [s.gpu_util_pct    for s in window]
        gpu_mem  = [s.gpu_mem_used_mb for s in window]
        cpu_vals = [s.cpu_pct         for s in window]
        ram_vals = [s.ram_used_gb     for s in window]

        return {
            "gpu_util_pct":     _agg(utils),
            "gpu_mem_used_mb":  _agg(gpu_mem),
            "gpu_mem_total_mb": window[0].gpu_mem_total_mb,
            "cpu_pct":          _agg(cpu_vals),
            "ram_used_gb":      _agg(ram_vals),
            "ram_total_gb":     window[0].ram_total_gb,
            "n_samples":        len(window),
        }

    def all_stats(self) -> dict:
        """Stats across all collected samples."""
        if not self._samples:
            return {}
        return self.phase_stats(
            "__start__", "__end__"
        ) if False else self.phase_stats.__func__(  # type: ignore
            self,
            list(self._markers.keys())[0] if self._markers else "",
            list(self._markers.keys())[-1] if self._markers else "",
        )

    @property
    def markers(self) -> dict[str, float]:
        return dict(self._markers)
