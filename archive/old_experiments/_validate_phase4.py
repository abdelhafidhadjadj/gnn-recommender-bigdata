"""
Phase 4 validation — auto graph-mode + mini-batch dispatch.

Tests are split into two groups:
  A. Backend-independent (run on any machine, no pyg-lib/torch-sparse needed)
  B. Backend-dependent  (skipped gracefully when backends not installed)
"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from utils.hardware import detect_hardware, build_adaptive_config, resolve_graph_mode
from data.samplers import build_pyg_data, neighbor_sampler_available
from models import build_model
from training.amp_utils import AMPContext

print("=" * 60)
print("PHASE 4 VALIDATION — graph-mode auto-select + mini-batch")
print("=" * 60)

# ── A1. n_id parameter accepted by all three models ──────────────────────────
print("\n[A1] Model forward(edge_index, n_id=...) for all architectures:")
n_users, n_items = 20, 30
num_nodes = n_users + n_items
ei = torch.stack([torch.randint(0, n_users, (40,)),
                  torch.randint(n_users, num_nodes, (40,))])

for mtype, extra in [("sage", {}), ("gat", {}), ("lightgcn", {"n_layers": 2})]:
    m = build_model(mtype, num_nodes, 16, 0.0, 4, **extra)
    # Full-batch: n_id=None
    out_full = m(ei, n_id=None)
    assert out_full.shape == (num_nodes, 16), f"{mtype} full-batch shape wrong"
    # Mini-batch: subset of nodes
    n_id = torch.arange(10)          # batch of 10 nodes
    sub_ei = torch.zeros(2, 0, dtype=torch.long)  # empty subgraph
    out_mb = m(sub_ei, n_id=n_id)
    assert out_mb.shape == (10, 16), f"{mtype} mini-batch shape wrong"
    print(f"    {mtype:<10} full={tuple(out_full.shape)}  mini={tuple(out_mb.shape)}  [PASS]")

# ── A2. build_pyg_data wraps edge_index correctly ────────────────────────────
print("\n[A2] build_pyg_data:")
data = build_pyg_data(ei, num_nodes)
assert data.num_nodes == num_nodes
assert data.edge_index.shape == ei.shape
assert data.edge_index.device.type == "cpu",  "edge_index should stay on CPU"
print(f"    num_nodes={data.num_nodes}  edge_index={tuple(data.edge_index.shape)}  [PASS]")

# ── A3. resolve_graph_mode logic ──────────────────────────────────────────────
print("\n[A3] resolve_graph_mode:")
cfg = Config()
profile_cpu = detect_hardware(force_debug=False)

# Auto on CPU → full_batch (no VRAM concern)
cfg.graph_mode = "auto"
mode = resolve_graph_mode(profile_cpu, cfg, n_nodes=num_nodes, emb_dim=16)
assert mode == "full_batch"
print(f"    CPU auto   -> '{mode}'  [PASS]")

# Explicit pin respected
cfg.graph_mode = "full_batch"
assert resolve_graph_mode(profile_cpu, cfg, num_nodes, 16) == "full_batch"
print(f"    explicit full_batch -> 'full_batch'  [PASS]")
cfg.graph_mode = "auto"  # reset

# ── A4. _train_minibatch_epoch with mock loader ───────────────────────────────
print("\n[A4] _train_minibatch_epoch with mock loader:")

from training.trainer import _train_minibatch_epoch, build_optimizer
from training.amp_utils import AMPContext

class _MockBatch:
    """Minimal PyG Batch substitute for testing the mini-batch loop."""
    def __init__(self, n, n_users, n_items):
        self.n_id            = torch.arange(n)
        self.edge_index      = torch.zeros(2, 0, dtype=torch.long)
        # 4 supervision edges: first n_users/2 are users, rest are items
        n_sup = 4
        self.edge_label_index = torch.stack([
            torch.randint(0, n_users, (n_sup,)),          # user local idx
            torch.randint(n_users, n, (n_sup,)),           # item local idx
        ])
    def to(self, device): return self

mock_loader = [_MockBatch(num_nodes, n_users, n_items) for _ in range(3)]

model   = build_model("sage", num_nodes, 16, 0.0, 4, n_layers=1, use_residual=False)
opt     = build_optimizer(model, Config().train, lr_override=0.01)
amp_ctx = AMPContext(enabled=False)
user_pos = {u: {u % n_items} for u in range(n_users)}   # dummy positive map

epoch_loss = _train_minibatch_epoch(
    model, opt, mock_loader, n_users, n_items,
    Config().train, amp_ctx, user_pos,
)
assert math.isfinite(epoch_loss) and epoch_loss > 0, f"Bad loss: {epoch_loss}"
print(f"    epoch_loss = {epoch_loss:.4f}  (finite, > 0)  [PASS]")

# ── A5. train_model dispatches correctly ──────────────────────────────────────
print("\n[A5] train_model loader=None -> full_batch path:")
from training.trainer import train_model

cfg2 = Config()
build_adaptive_config(cfg2, detect_hardware(force_debug=True))
model2 = build_model("sage", num_nodes, 16, 0.0, 4)
opt2   = build_optimizer(model2, cfg2.train, lr_override=0.005)
hist   = train_model(model2, opt2, ei, torch.randint(0, n_users, (10,)),
                     torch.randint(0, n_items, (10,)),
                     n_users, n_items, cfg2.train,
                     loader=None, verbose=False)
assert len(hist) == cfg2.train.num_epochs
assert all(math.isfinite(v) and v > 0 for v in hist)
print(f"    {len(hist)} epochs, all finite  [PASS]")

print("\n[A5] train_model loader=mock -> mini-batch path:")
model3 = build_model("sage", num_nodes, 16, 0.0, 4)
opt3   = build_optimizer(model3, cfg2.train, lr_override=0.005)
hist3  = train_model(model3, opt3, ei, torch.randint(0, n_users, (10,)),
                     torch.randint(0, n_items, (10,)),
                     n_users, n_items, cfg2.train,
                     loader=mock_loader, verbose=False)
assert len(hist3) == cfg2.train.num_epochs
assert all(math.isfinite(v) and v > 0 for v in hist3)
print(f"    {len(hist3)} epochs, all finite  [PASS]")

# ── A6. neighbor_sampler_available() reports correctly ────────────────────────
print("\n[A6] neighbor_sampler_available():")
available = neighbor_sampler_available()
print(f"    pyg-lib / torch-sparse installed: {available}")
if not available:
    print("    (expected on CPU dev laptop — GPU server will have them)")
print("    [PASS]")

# ── B1. Backend-dependent: actual LinkNeighborLoader ─────────────────────────
print("\n[B1] LinkNeighborLoader (skipped if backends absent):")
if not available:
    print("    SKIPPED — install pyg-lib or torch-sparse on GPU server")
else:
    from data.samplers import make_train_loader
    train_ei = torch.stack([torch.randint(0, n_users, (20,)),
                            torch.randint(n_users, num_nodes, (20,))])
    ldr = make_train_loader(data, train_ei, batch_size=8,
                            num_neighbors=[3, 2], profile=profile_cpu)
    batch = next(iter(ldr))
    assert hasattr(batch, "n_id")
    assert hasattr(batch, "edge_label_index")
    print(f"    batch.n_id={tuple(batch.n_id.shape)}  "
          f"edge_label_index={tuple(batch.edge_label_index.shape)}  [PASS]")

print("\n" + "=" * 60)
print("PHASE 4 VALIDATION  ->  ALL PASSED")
print("=" * 60)
