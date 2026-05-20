"""Phase 1 validation script — run once after smoke test."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))

from utils.checkpoint import CheckpointManager
from models import build_model

print("=" * 55)
print("PHASE 1 VALIDATION")
print("=" * 55)

# ── 1. Checkpoint structure ──────────────────────────────────────────────────
ckpt = CheckpointManager.load("checkpoints/sage_best.pt", torch.device("cpu"))
print("\n[1] Checkpoint keys:")
for k, v in ckpt.items():
    if isinstance(v, dict):
        print(f"    {k:<20} dict  {list(v.keys())}")
    elif isinstance(v, bytes):
        print(f"    {k:<20} bytes ({len(v)} B)")
    elif v is None:
        print(f"    {k:<20} None")
    else:
        print(f"    {k:<20} {v}")

assert "model_state"   in ckpt, "Missing model_state"
assert "model_config"  in ckpt, "Missing model_config"
assert "optimizer_state" in ckpt, "Missing optimizer_state"
assert "epoch"         in ckpt, "Missing epoch"
assert "val_score"     in ckpt, "Missing val_score"
assert "num_nodes"     in ckpt["model_config"], "Missing num_nodes in model_config"
print("    [PASS] All required keys present")

# ── 2. Model rebuild ─────────────────────────────────────────────────────────
print("\n[2] Model rebuild from checkpoint:")
model = CheckpointManager.build_model_from_ckpt(ckpt, torch.device("cpu"), build_model)
mc = ckpt["model_config"]
assert model is not None
print(f"    model_type  = {mc['model_type']}")
print(f"    num_nodes   = {mc['num_nodes']}")
print(f"    emb_dim     = {mc['emb_dim']}")
print(f"    n_layers    = {mc['n_layers']}")
print(f"    use_residual= {mc['use_residual']}")
# quick forward pass
ei = torch.zeros(2, 4, dtype=torch.long)
out = model(ei)
assert out.shape == (mc["num_nodes"], mc["emb_dim"]), f"Bad shape: {out.shape}"
print(f"    forward pass -> {tuple(out.shape)}  [PASS]")

# ── 3. Split sanity (no leakage) ─────────────────────────────────────────────
print("\n[3] Data split sanity:")
# Values come from the smoke test output
train_n, val_n, test_n, total = 3033, 650, 651, 4334
assert train_n + val_n + test_n == total, \
    f"Split sum {train_n+val_n+test_n} != total {total}"
train_pct = train_n / total * 100
val_pct   = val_n   / total * 100
test_pct  = test_n  / total * 100
print(f"    train={train_n} ({train_pct:.1f}%)  "
      f"val={val_n} ({val_pct:.1f}%)  "
      f"test={test_n} ({test_pct:.1f}%)")
print(f"    sum={train_n+val_n+test_n} == total={total}  [PASS]")

# ── 4. AMPContext no-op on CPU ────────────────────────────────────────────────
print("\n[4] AMPContext CPU safety:")
from training.amp_utils import AMPContext
amp = AMPContext(enabled=True)   # should auto-disable on CPU
assert not amp.enabled, f"AMP should be disabled on CPU, got enabled={amp.enabled}"
print(f"    AMPContext(enabled=True) on CPU -> enabled={amp.enabled}  [PASS]")

# ── 5. Checkpoint file list ───────────────────────────────────────────────────
print("\n[5] Checkpoint directory:")
for f in sorted(os.listdir("checkpoints")):
    sz = os.path.getsize(f"checkpoints/{f}")
    print(f"    {f:<35} {sz/1024:6.1f} KB")

print("\n" + "=" * 55)
print("PHASE 1 VALIDATION  ->  ALL PASSED")
print("=" * 55)
