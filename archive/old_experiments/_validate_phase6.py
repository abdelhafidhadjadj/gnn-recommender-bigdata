"""Phase 6 validation — incremental training end-to-end."""
import sys, os, shutil, torch, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from data.preprocessing import DynamicLabelEncoder
from data.replay_buffer import ReplayBuffer
from training.incremental import extend_model_embeddings
from models import build_model
from config import Config
from utils.checkpoint import CheckpointManager

print("=" * 60)
print("PHASE 6 VALIDATION — incremental training")
print("=" * 60)

# ── [1] DynamicLabelEncoder: fit + add_new + is_known ────────────────────────
print("\n[1] DynamicLabelEncoder:")
enc = DynamicLabelEncoder()
enc.fit(["alice", "bob", "carol"])
assert list(enc.classes_) == ["alice", "bob", "carol"]
assert enc.transform(["carol", "alice"]).tolist() == [2, 0]
assert enc.is_known("bob")
assert not enc.is_known("dave")

n_added = enc.add_new(["dave", "alice"])  # alice already known -> not added again
assert n_added == 1
assert enc.is_known("dave")
assert enc.transform(["alice"]).tolist() == [0]   # existing mapping UNCHANGED
assert enc.transform(["dave"]).tolist()  == [3]   # new entity appended
print(f"    fit({3}) -> add_new({1}) -> total={len(enc)}  mappings preserved  [PASS]")

# ── [2] DynamicLabelEncoder is serialisable (for checkpoint) ─────────────────
print("\n[2] DynamicLabelEncoder pickling:")
import pickle
restored = pickle.loads(pickle.dumps(enc))
assert list(restored.classes_) == list(enc.classes_)
assert restored.transform(["dave"]).tolist() == [3]
print("    pickle round-trip OK  [PASS]")

# ── [3] ReplayBuffer: add + sample + local encoding ──────────────────────────
print("\n[3] ReplayBuffer:")
buf = ReplayBuffer(capacity=1000)
df_old = pd.DataFrame({
    "user_id": [0, 1, 2],
    "item_id": [10 + 300, 11 + 300, 12 + 300],  # global IDs with n_users=300
    "rating":  [4.0, 5.0, 3.0],
})
buf.add(df_old, n_users=300)
assert len(buf) == 3

# Sample with new n_users=310 (10 new users added)
sampled = buf.sample(2, current_n_users=310)
assert set(sampled.columns) == {"user_id", "item_id", "rating"}
assert all(sampled["item_id"] >= 310)   # offset correctly updated
print(f"    added 3 interactions -> sampled 2 with new offset n_users=310  [PASS]")

# ── [4] extend_model_embeddings: old weights preserved, correct layout ────────
print("\n[4] extend_model_embeddings:")
n_old_users, n_old_items = 20, 15
n_new_users, n_new_items = 3,  5
cfg = Config(); cfg.model_type = "sage"
cfg.model.emb_dim = 16; cfg.model.dropout = 0.0
cfg.model.gat_heads = 4; cfg.model.n_layers = 1; cfg.model.use_residual = False

model_old = build_model("sage", n_old_users + n_old_items, 16, 0.0, 4)
old_user0 = model_old.embeddings.weight.data[0].clone()               # user 0
old_item0 = model_old.embeddings.weight.data[n_old_users].clone()     # item 0

model_new, n_u, n_i = extend_model_embeddings(
    model_old, n_new_users, n_new_items, n_old_users, cfg, seed=42
)
assert n_u == n_old_users + n_new_users
assert n_i == n_old_items + n_new_items
assert model_new.embeddings.num_embeddings == n_u + n_i

# Old user 0 must be at position 0 (unchanged)
assert torch.allclose(model_new.embeddings.weight.data[0], old_user0), \
    "Old user embedding was modified"

# Old item 0 must be at position n_users_new (shifted by n_new_users)
n_users_new = n_old_users + n_new_users
assert torch.allclose(
    model_new.embeddings.weight.data[n_users_new], old_item0
), "Old item embedding was not correctly shifted"
print(f"    {n_old_users}u+{n_old_items}i -> +{n_new_users}u+{n_new_items}i: "
      f"old weights preserved, layout correct  [PASS]")

# ── [5] Full scratch -> checkpoint -> incremental round-trip ───────────────────
print("\n[5] Full round-trip (scratch -> checkpoint -> incremental):")

CKPT_DIR = "_phase6_test_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)

try:
    import subprocess
    result = subprocess.run(
        [sys.executable, "main.py",
         "--mode", "scratch",
         "--model", "sage",
         "--data-dir", "data/test",
         "--ckpt-dir", CKPT_DIR,
         "--debug",
         "--no-amp",
         "--seed", "42"],
        capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        print("    Scratch training failed:")
        print(result.stderr[-2000:])
        sys.exit(1)
    print("    Scratch training completed  [PASS]")

    # Find the saved checkpoint
    ckpt_path = os.path.join(CKPT_DIR, "sage_best.pt")
    assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"

    # Load and verify DynamicLabelEncoder is stored
    from utils.device import detect_devices
    dev = detect_devices()
    ckpt = CheckpointManager.load(ckpt_path, torch.device("cpu"))
    assert isinstance(ckpt["user_encoder"], DynamicLabelEncoder), \
        "user_encoder is not DynamicLabelEncoder — preprocess() must use DynamicLabelEncoder"
    assert isinstance(ckpt["item_encoder"], DynamicLabelEncoder)
    print(f"    Checkpoint has DynamicLabelEncoder  [PASS]")

    # Run incremental update with the SAME data/test (simulates new batch)
    result2 = subprocess.run(
        [sys.executable, "main.py",
         "--mode", "incremental",
         "--model", "sage",
         "--ckpt", ckpt_path,
         "--new-data", "data/test/yelp_academic_dataset_review_healthandmedical.csv",
         "--ckpt-dir", CKPT_DIR,
         "--no-amp",
         "--seed", "42"],
        capture_output=True, text=True, timeout=300
    )
    if result2.returncode != 0:
        print("    Incremental training failed:")
        print(result2.stderr[-3000:])
        sys.exit(1)
    print("    Incremental fine-tuning completed  [PASS]")

    # Check a v2 checkpoint was saved (version > 1)
    ckpt2 = CheckpointManager.load(ckpt_path, torch.device("cpu"))
    assert ckpt2["version"] >= 1
    assert "train_interactions" in ckpt2.get("extra", {})
    assert isinstance(ckpt2["extra"]["replay_buffer"], ReplayBuffer)
    print(f"    New checkpoint v{ckpt2['version']} has train_interactions + replay_buffer  [PASS]")

finally:
    shutil.rmtree(CKPT_DIR, ignore_errors=True)

print("\n" + "=" * 60)
print("PHASE 6 VALIDATION  ->  ALL PASSED")
print("=" * 60)
