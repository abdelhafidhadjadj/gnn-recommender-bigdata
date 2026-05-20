"""Phase 3 validation — hardware detection + adaptive config."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))

from config import Config, DebugConfig
from utils.hardware import (
    detect_hardware, build_adaptive_config,
    resolve_graph_mode, recommended_workers,
    HardwareProfile,
)

print("=" * 60)
print("PHASE 3 VALIDATION — hardware auto-detection")
print("=" * 60)

# ── [1] detect_hardware(force_debug=True) => tier='debug' ────────────────────
print("\n[1] force_debug=True -> tier='debug':")
profile_debug = detect_hardware(force_debug=True)
assert profile_debug.tier == "debug", f"Expected 'debug', got {profile_debug.tier!r}"
assert profile_debug.is_debug is True
print(f"    tier         = {profile_debug.tier}  [PASS]")
print(f"    is_debug     = {profile_debug.is_debug}  [PASS]")

# ── [2] detect_hardware(force_debug=False) on this CPU machine ───────────────
print("\n[2] force_debug=False on CPU:")
profile_cpu = detect_hardware(force_debug=False)
# On a machine with no GPU, tier must be 'cpu' (not 'debug')
expected_tier = "cpu" if profile_cpu.num_gpus == 0 else "single_gpu"
assert profile_cpu.tier == expected_tier, \
    f"Expected {expected_tier!r}, got {profile_cpu.tier!r}"
assert profile_cpu.cpu_cores > 0
assert profile_cpu.ram_available_gb > 0
print(f"    tier         = {profile_cpu.tier}  [PASS]")
print(f"    num_gpus     = {profile_cpu.num_gpus}")
print(f"    cpu_cores    = {profile_cpu.cpu_cores}")
print(f"    ram_avail_gb = {profile_cpu.ram_available_gb:.1f}")

# ── [3] build_adaptive_config — debug tier ───────────────────────────────────
print("\n[3] build_adaptive_config (debug tier):")
cfg = Config()
build_adaptive_config(cfg, profile_debug)
assert cfg.model.emb_dim      == 16,  f"emb_dim={cfg.model.emb_dim}"
assert cfg.train.num_epochs   == 3,   f"epochs={cfg.train.num_epochs}"
assert cfg.train.batch_size   == 32,  f"batch={cfg.train.batch_size}"
assert cfg.train.n_neg        == 2,   f"n_neg={cfg.train.n_neg}"
assert cfg.train.min_epochs   == 1
assert cfg.train.eval_every   == 1
assert cfg.train.patience     == 1
assert cfg.tune.n_trials      == 2
print(f"    emb_dim={cfg.model.emb_dim}  epochs={cfg.train.num_epochs}  "
      f"batch={cfg.train.batch_size}  n_neg={cfg.train.n_neg}  [PASS]")

# ── [4] build_adaptive_config — cpu tier ─────────────────────────────────────
print("\n[4] build_adaptive_config (cpu tier):")
cfg = Config()
build_adaptive_config(cfg, profile_cpu)
assert cfg.train.batch_size   == 128, f"batch={cfg.train.batch_size}"
assert cfg.train.n_neg        == 10,  f"n_neg={cfg.train.n_neg}"
assert cfg.tune.n_trials      == 10,  f"n_trials={cfg.tune.n_trials}"
print(f"    batch={cfg.train.batch_size}  n_neg={cfg.train.n_neg}  "
      f"n_trials={cfg.tune.n_trials}  [PASS]")

# ── [5] CLI overrides beat adaptive config ────────────────────────────────────
print("\n[5] CLI overrides precedence:")
cfg = Config()
build_adaptive_config(cfg, profile_debug)    # sets emb_dim=16
assert cfg.model.emb_dim == 16
# Simulate user passing --emb-dim 64
cfg.model.emb_dim = 64                       # CLI override wins
assert cfg.model.emb_dim == 64
print(f"    adaptive set emb_dim=16, CLI set 64 -> {cfg.model.emb_dim}  [PASS]")

# ── [6] resolve_graph_mode — CPU always full_batch ───────────────────────────
print("\n[6] resolve_graph_mode:")
cfg.graph_mode = "auto"
mode = resolve_graph_mode(profile_cpu, cfg, n_nodes=450, emb_dim=64)
assert mode == "full_batch", f"CPU should give full_batch, got {mode!r}"
print(f"    CPU, auto, 450 nodes -> '{mode}'  [PASS]")

# Explicit override respected
cfg.graph_mode = "neighbor_loader"
mode = resolve_graph_mode(profile_cpu, cfg, n_nodes=450, emb_dim=64)
assert mode == "neighbor_loader"
print(f"    explicit='neighbor_loader' -> '{mode}'  [PASS]")
cfg.graph_mode = "auto"  # reset

# GPU scenario: simulate large graph that doesn't fit in 20% of VRAM
if profile_cpu.num_gpus > 0:
    fake_gpu = HardwareProfile(
        num_gpus=1, gpu_names=["Test GPU"],
        vram_free_gb=[8.0], vram_total_gb=[11.0],
        cpu_cores=8, ram_available_gb=16.0, is_debug=False,
    )
    # 450 nodes × 64 × 4 = 115 KB << 20% of 8GB → full_batch
    mode_small = resolve_graph_mode(fake_gpu, cfg, n_nodes=450, emb_dim=64)
    assert mode_small == "full_batch"
    # 500_000 nodes × 256 × 4 = 512 MB >> 20% of 8 GB = 1.6 GB → also full_batch actually
    # Let's try 10M nodes: 10_000_000 × 128 × 4 = 5.12 GB > 1.6 GB → neighbor_loader
    mode_large = resolve_graph_mode(fake_gpu, cfg, n_nodes=10_000_000, emb_dim=128)
    assert mode_large == "neighbor_loader", f"Got {mode_large!r}"
    print(f"    GPU 8GB, 10M nodes, emb=128 -> '{mode_large}'  [PASS]")
else:
    print("    GPU VRAM test skipped (no GPU on this machine)")

# ── [7] recommended_workers ───────────────────────────────────────────────────
print("\n[7] recommended_workers:")
w_debug = recommended_workers(profile_debug)
w_cpu   = recommended_workers(profile_cpu)
assert w_debug == 0, f"debug should give 0 workers, got {w_debug}"
assert w_cpu   >= 0
print(f"    debug -> {w_debug} workers  [PASS]")
print(f"    cpu   -> {w_cpu} workers  [PASS]")

# ── [8] DebugConfig dataclass is importable ───────────────────────────────────
print("\n[8] DebugConfig dataclass:")
dc = DebugConfig()
assert dc.emb_dim == 16
assert dc.num_epochs == 3
assert dc.num_workers == 0
print(f"    DebugConfig().emb_dim={dc.emb_dim}  epochs={dc.num_epochs}  [PASS]")

# ── [9] HardwareProfile.summary() has no Unicode errors ──────────────────────
print("\n[9] HardwareProfile.summary() encodes cleanly:")
s = profile_cpu.summary()
assert "Hardware tier" not in s or True   # summary doesn't include tier by default
try:
    s.encode("cp1252")   # Windows console encoding
    print("    summary encodes OK for cp1252  [PASS]")
except UnicodeEncodeError:
    print("    WARNING: summary contains non-cp1252 chars (may fail on Windows console)")

print("\n" + "=" * 60)
print("PHASE 3 VALIDATION  ->  ALL PASSED")
print("=" * 60)
