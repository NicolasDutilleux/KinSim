"""Find which refactor flag breaks m6A IPD prediction.

Loads the in-training v12_run2 checkpoint, samples a batch of SLOWED/m6A rows
from bc2082 (rich in both m5C and m6A), and reports:

  1. Whether the biology_mask compat matrix is correct (4 bases x 4 meth_ids).
  2. Whether biology_mask zeros out m6A flags on real samples.
  3. What the model actually predicts (mu_IPD) for those m6A rows vs truth.
  4. What it would predict WITHOUT biology_mask (same model weights,
     mask toggled off at inference).

Run via sbatch — see scripts/debug_m6a_regression.slurm.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "/data/users/ndutilleux/KinSim"
SHARD = (
    "/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega/refined/"
    "strepto_bc2082_shard_clean.pkl"
)
CKPT = (
    "/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega/checkpoints/"
    "v12_run2/lightning_ckpts/last.ckpt"
)

sys.path.insert(0, REPO)

from kinsim.data.dataset import _flatten_data_dict, log_transform
from kinsim.models.predictor import (
    ConvPredictor,
    _build_meth_compat_buffer,
    create_from_config,
)
from kinsim.utils.config import get_extraction_params
from kinsim.utils.sample_layout import get_sample_layout


# =========================================================================
# Step 1 — biology compat matrix
# =========================================================================
print("=" * 70)
print("STEP 1 — biology_mask compat matrix")
print("=" * 70)
compat = _build_meth_compat_buffer(num_meth_types=4)
print("                       none    m6A    m4C    m5C")
for i, b in enumerate("ACGT"):
    row = compat[i].tolist()
    print(f"  base {b} (id={i}):   {row[0]:6.1f} {row[1]:6.1f} {row[2]:6.1f} {row[3]:6.1f}")
print()
print("EXPECTED:")
print("  base A: 1.0 1.0 0.0 0.0   (none + m6A only)")
print("  base C: 1.0 0.0 1.0 1.0   (none + m4C + m5C)")
print("  base G: 1.0 0.0 0.0 0.0   (none only)")
print("  base T: 1.0 0.0 0.0 0.0   (none only)")

expected = torch.tensor([
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
])
if torch.allclose(compat, expected):
    print("\n[OK] compat matrix matches expectation")
else:
    print("\n[FAIL] compat matrix DIFFERS from expectation — diff:")
    print((compat - expected).tolist())


# =========================================================================
# Step 2 — load shard, find SLOWED/m6A rows
# =========================================================================
print()
print("=" * 70)
print("STEP 2 — load bc2082 shard, sample SLOWED/m6A rows")
print("=" * 70)
with open(SHARD, "rb") as f:
    d = pickle.load(f)

params = get_extraction_params()
layout = get_sample_layout(params)
flat = _flatten_data_dict(d, layout, num_meth_types=4)
del d
print(f"shard rows: {len(flat['kmer_ids']):,}")

mask = (flat["categories"] == 1) & (flat["parent_meths"] == 1)
print(f"SLOWED/m6A rows: {int(mask.sum()):,}")

idx_pool = np.where(mask)[0]
# Sample 64 rows (deterministic)
rng = np.random.default_rng(42)
idx = rng.choice(idx_pool, size=min(64, len(idx_pool)), replace=False)

kmer_ids = torch.from_numpy(flat["kmer_ids"][idx].astype(np.int64))
meth_full = torch.from_numpy(flat["meth_full"][idx]).float()  # (B, 14, 4)
truth = flat["signals_log"][idx].float()  # (B, 2)  log1p space
truth_ipd_log = truth[:, 0]
truth_pw_log = truth[:, 1]


# =========================================================================
# Step 3 — verify biology_mask doesn't zero m6A flags
# =========================================================================
print()
print("=" * 70)
print("STEP 3 — biology_mask effect on m6A flags in 64 SLOWED/m6A rows")
print("=" * 70)

# Decode kmer_ids to bases (B, 11)
B = kmer_ids.shape[0]
bases = torch.zeros(B, params.kmer_size, dtype=torch.long)
for i in range(B):
    val = int(kmer_ids[i].item())
    for j in range(params.kmer_size):
        bases[i, params.kmer_size - 1 - j] = val & 3
        val >>= 2

# Pre-mask m6A flags at kmer positions only (first 11 of 14)
m6a_pre = meth_full[:, :params.kmer_size, 1].clone()  # (B, 11)

# Apply mask
compat_at_pos = compat[bases]  # (B, 11, 4)
meth_full_masked = meth_full.clone()
meth_full_masked[:, :params.kmer_size, :] = (
    meth_full_masked[:, :params.kmer_size, :] * compat_at_pos
)
m6a_post = meth_full_masked[:, :params.kmer_size, 1]  # (B, 11)

# Compare
diff = (m6a_pre - m6a_post).abs().sum(dim=1)  # per-sample total change
n_changed = (diff > 1e-6).sum().item()
total_m6a_flags_pre = (m6a_pre > 0).sum().item()
total_m6a_flags_post = (m6a_post > 0).sum().item()
print(f"Out of 64 sampled SLOWED/m6A rows:")
print(f"  rows where biology_mask CHANGED m6A flags: {n_changed}")
print(f"  total m6A flag positions PRE-mask:  {total_m6a_flags_pre}")
print(f"  total m6A flag positions POST-mask: {total_m6a_flags_post}")
print(f"  m6A flags LOST by biology_mask:     {total_m6a_flags_pre - total_m6a_flags_post}")
if total_m6a_flags_post < total_m6a_flags_pre:
    print()
    print("[WARN] biology_mask zeros some m6A flags. Showing 3 first cases:")
    bad_idx = (diff > 1e-6).nonzero().flatten()[:3]
    for bi in bad_idx:
        bi = int(bi.item())
        kmer_str = "".join("ACGT"[int(bases[bi, j].item())] for j in range(params.kmer_size))
        # positions where m6A flag was lost
        lost_pos = ((m6a_pre[bi] > 0) & (m6a_post[bi] == 0)).nonzero().flatten().tolist()
        print(f"  kmer={kmer_str}  base@center={kmer_str[params.active_site_index]}  lost_at={lost_pos}")
elif total_m6a_flags_pre == total_m6a_flags_post:
    print("\n[OK] biology_mask preserved all m6A flags on these samples")


# =========================================================================
# Step 4 — load v12_run2 checkpoint, predict, compare with truth
# =========================================================================
print()
print("=" * 70)
print("STEP 4 — predict with v12_run2 model on the 64 m6A samples")
print("=" * 70)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
# Lightning ckpt has 'state_dict' with 'model.' prefix
model_sd = {
    k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")
}

# Need model_config — try to read from checkpoint dir
import json
cfg_path = Path(CKPT).parent.parent / "model_config.json"
if cfg_path.exists():
    with open(cfg_path) as f:
        cfg = json.load(f)
else:
    # Reasonable defaults matching the v12_run2 setup
    cfg = {
        "architecture": "conv",
        "base_embed_dim": 16,
        "conv_dim": 128,
        "n_conv_layers": 3,
        "kernel_size": 3,
        "head_dim": 128,
        "num_meth_types": 4,
        "meth_proj_dim": 8,
        "dropout": 0.1,
        "kmer_aware_film": True,
        "biology_mask": True,
        "log_sigma_clamp_max": 1.5,
    }

model = create_from_config(cfg)
model.load_state_dict(model_sd)
model.eval()

# Predict WITH biology_mask (as trained)
with torch.no_grad():
    out_with = model(kmer_ids, meth_full)
mu_ipd_with = out_with[:, 0]

# Predict WITHOUT biology_mask (toggle flag temporarily)
model.biology_mask = False
with torch.no_grad():
    out_without = model(kmer_ids, meth_full)
mu_ipd_without = out_without[:, 0]
model.biology_mask = True

# Compare
print(f"Truth μ_IPD (log1p):   {truth_ipd_log.mean().item():+.4f} ± {truth_ipd_log.std().item():.4f}")
print(f"Pred μ_IPD WITH mask:  {mu_ipd_with.mean().item():+.4f} ± {mu_ipd_with.std().item():.4f}")
print(f"Pred μ_IPD WITHOUT:    {mu_ipd_without.mean().item():+.4f} ± {mu_ipd_without.std().item():.4f}")
print()

# Pearson r on these 64 samples
def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.pow(2).sum() * b.pow(2).sum()).sqrt()
    return (a * b).sum() / denom if denom > 1e-9 else torch.tensor(0.0)

r_with = pearson(mu_ipd_with, truth_ipd_log)
r_without = pearson(mu_ipd_without, truth_ipd_log)
print(f"Pearson r (μ_IPD vs truth) WITH biology_mask:    {r_with.item():+.3f}")
print(f"Pearson r (μ_IPD vs truth) WITHOUT biology_mask: {r_without.item():+.3f}")
print()
if r_with < 0.1 < r_without:
    print("[FINDING] biology_mask is breaking m6A IPD prediction.")
    print("           Disabling it recovers correlation.")
elif r_with > 0.4:
    print("[FINDING] biology_mask is innocent on these samples — model fits m6A.")
elif r_without < 0.1:
    print("[FINDING] Model fails on m6A regardless of biology_mask.")
    print("           Problem is elsewhere (Beta-NLL or balance_kmers / augment).")
else:
    print("[FINDING] Mixed result, check the raw numbers above.")
