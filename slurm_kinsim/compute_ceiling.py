"""Compute theoretical ceiling scores for a KinSim .pkl shard or master.

Given an empirical distribution at each (kmer, meth) key, answers:
    "What score would a model get if it perfectly matched the true
     per-key N(μ, σ²) distribution?"

Ceilings are data-intrinsic — they depend only on within-key noise vs
between-key signal spread.  Model Pearson can never exceed the oracle;
can only approach it.

Usage:
    python compute_ceiling.py <pkl_path>
    python compute_ceiling.py <pkl_path> --per-meth
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

METH_NAMES = {0: "none", 1: "m6A", 2: "m4C", 3: "m5C"}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_ceilings(data: dict, meth_filter: int | None = None, space: str = "log1p") -> dict:
    """Compute ceilings from a raw .pkl dict.

    Args:
        data:         dict[(kmer_id, meth_id)] -> ndarray(N, 2|3|14) raw float.
        meth_filter:  If given, only keep keys with this meth_id.
        space:        "log1p" (training space) or "raw" (uint8 space).

    Returns:
        metrics dict.
    """
    all_true = []   # per-sample actual values
    all_mu   = []   # per-sample per-key mean (what a "perfect mean" model outputs)
    all_z    = []   # per-sample random draw ~ N(μ_k, σ_k²)
    key_mu   = []   # per-key mean (weighted by n)
    key_sig  = []   # per-key std
    key_n    = []
    rng = np.random.default_rng(42)

    for key, arr in data.items():
        if not isinstance(key, tuple):
            continue
        if not isinstance(arr, np.ndarray) or len(arr) < 2:
            continue
        _, meth_id = key
        if meth_filter is not None and meth_id != meth_filter:
            continue

        xy = arr[:, :2].astype(np.float32)  # (N, 2) IPD, PW
        if space == "log1p":
            xy = np.log1p(xy)

        mu_k  = xy.mean(axis=0)          # (2,)
        sig_k = xy.std(axis=0, ddof=0)   # (2,)  within-key noise
        n_k   = len(xy)

        all_true.append(xy)
        all_mu.append(np.broadcast_to(mu_k, xy.shape).copy())
        all_z.append(mu_k + sig_k * rng.standard_normal(xy.shape).astype(np.float32))
        key_mu.append(mu_k)
        key_sig.append(sig_k)
        key_n.append(n_k)

    if not all_true:
        return {}

    all_true = np.concatenate(all_true, axis=0)
    all_mu   = np.concatenate(all_mu,   axis=0)
    all_z    = np.concatenate(all_z,    axis=0)
    key_mu   = np.array(key_mu,  dtype=np.float32)  # (K, 2)
    key_sig  = np.array(key_sig, dtype=np.float32)  # (K, 2)
    key_n    = np.array(key_n,   dtype=np.int64)

    # ── Oracle Pearson (predict the per-key mean; maximum achievable) ──
    # r_oracle = Var(μ_between) / (Var(μ_between) + E[σ²_within])
    # Closed form AND empirical form — they agree.
    var_mu_between = np.average((key_mu - key_mu.mean(axis=0, keepdims=True))**2,
                                 axis=0, weights=key_n)
    e_sig2_within  = np.average(key_sig**2, axis=0, weights=key_n)
    r_oracle_formula = var_mu_between / (var_mu_between + e_sig2_within + 1e-12)
    r_oracle_emp = np.array([_pearson(all_mu[:, i], all_true[:, i]) for i in range(2)])

    # ── Random-from-distribution Pearson (noise floor for distributional model) ──
    # r_rand ≈ Var(μ) / (Var(μ) + 2·E[σ²])  (because z adds independent σ² noise)
    r_rand_emp = np.array([_pearson(all_z[:, i], all_true[:, i]) for i in range(2)])

    # ── MAE ceiling (predict the per-key mean) ──
    # For Gaussian per-key: E[|Y - μ|] = σ · √(2/π) ≈ 0.7979 σ
    mae_oracle = np.abs(all_mu - all_true).mean(axis=0)

    return {
        "space":        space,
        "n_keys":       int(len(key_mu)),
        "n_samples":    int(len(all_true)),
        "pearson_oracle_ipd":      float(r_oracle_emp[0]),
        "pearson_oracle_pw":       float(r_oracle_emp[1]),
        "pearson_oracle_formula":  (float(r_oracle_formula[0]), float(r_oracle_formula[1])),
        "pearson_random_ipd":      float(r_rand_emp[0]),
        "pearson_random_pw":       float(r_rand_emp[1]),
        "mae_oracle_ipd":          float(mae_oracle[0]),
        "mae_oracle_pw":           float(mae_oracle[1]),
        "e_sigma_within_ipd":      float(np.sqrt(e_sig2_within[0])),
        "e_sigma_within_pw":       float(np.sqrt(e_sig2_within[1])),
        "sigma_mu_between_ipd":    float(np.sqrt(var_mu_between[0])),
        "sigma_mu_between_pw":     float(np.sqrt(var_mu_between[1])),
    }


def _fmt(m: dict) -> str:
    if not m:
        return "(no data)"
    return (
        f"  Keys: {m['n_keys']:,}    Samples: {m['n_samples']:,}    Space: {m['space']}\n"
        f"  ── Pearson ────────────────────────────────────────────\n"
        f"   Oracle (ceiling) IPD = {m['pearson_oracle_ipd']:+.4f}    PW = {m['pearson_oracle_pw']:+.4f}\n"
        f"   Random draw      IPD = {m['pearson_random_ipd']:+.4f}    PW = {m['pearson_random_pw']:+.4f}\n"
        f"   Formula check    IPD = {m['pearson_oracle_formula'][0]:+.4f}    PW = {m['pearson_oracle_formula'][1]:+.4f}\n"
        f"  ── MAE (log1p space, oracle = predict per-key mean) ───\n"
        f"   MAE oracle       IPD = {m['mae_oracle_ipd']:.4f}      PW = {m['mae_oracle_pw']:.4f}\n"
        f"  ── Variance structure ─────────────────────────────────\n"
        f"   Between-key σ(μ) IPD = {m['sigma_mu_between_ipd']:.4f}     PW = {m['sigma_mu_between_pw']:.4f}  (signal)\n"
        f"   Within-key  E[σ] IPD = {m['e_sigma_within_ipd']:.4f}     PW = {m['e_sigma_within_pw']:.4f}  (noise)\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", help="Shard or master .pkl file")
    ap.add_argument("--per-meth", action="store_true", help="Break down by methylation type")
    ap.add_argument("--space", choices=["log1p", "raw"], default="log1p",
                    help="Signal space for metrics (default: log1p = training space)")
    args = ap.parse_args()

    path = Path(args.pkl)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {path}  ({path.stat().st_size / 1e9:.2f} GB)")
    with open(path, "rb") as f:
        data = pickle.load(f)
    n_tuple_keys = sum(1 for k in data if isinstance(k, tuple))
    print(f"  Tuple keys: {n_tuple_keys:,}\n")

    print("=" * 70)
    print(f"  OVERALL  ({args.space} space)")
    print("=" * 70)
    print(_fmt(compute_ceilings(data, meth_filter=None, space=args.space)))

    if args.per_meth:
        for mid in (0, 1, 2, 3):
            print("=" * 70)
            print(f"  meth = {METH_NAMES[mid]}")
            print("=" * 70)
            print(_fmt(compute_ceilings(data, meth_filter=mid, space=args.space)))

    print("─" * 70)
    print("  Interpretation")
    print("─" * 70)
    print("  Pearson oracle  = r² = Var(μ_between) / (Var(μ_between) + E[σ²_within])")
    print("    Ceiling for ANY model predicting a per-key mean.  A model that")
    print("    outputs the exact per-key distribution cannot exceed this.")
    print()
    print("  Pearson random  = what you get drawing z ~ N(μ_k, σ_k²) per sample.")
    print("    The floor for a correct distributional model (not point estimate).")
    print("    A well-trained model should sit BETWEEN random and oracle; the")
    print("    gap closed defines its 'efficiency'.")
    print()
    print("  MAE oracle      = mean |sample - μ_k|.  For Gaussian keys this is")
    print("    ~0.7979 × E[σ_within].  No model can beat it on held-out samples.")


if __name__ == "__main__":
    main()
