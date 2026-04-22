"""Refine a KinSim .pkl by removing contamination from methylated buckets.

For each (kmer, mod_type>0) bucket, the samples are a mixture of:
  - Real methylated signal
  - Unmethylated contamination (positions labelled by motif scan but not
    actually modified at that genomic site — partial methylation, hemi-sites,
    false-positive motifs, etc.)

Given the abundant (kmer, 0) bucket as a known reference for unmethylated
signal, fit a 2-component Gaussian mixture where the None component is
FIXED to (μ_N, Σ_N) and the methylated component (μ_m, Σ_m) + mixing
weight π are free.  EM recovers the true methylation fraction π from data,
without relying on motifs.csv's sometimes-noisy global fraction.

After fitting, the methylated bucket is replaced with samples drawn from
N(μ_m, Σ_m) at count round(π · N_original), giving a clean dictionary.

Fallback (too few samples / EM failure):
    5% Mahalanobis threshold against None (χ²_{2, 0.95} = 5.99).  Samples
    beyond threshold are treated as real methylated; μ_m, Σ_m estimated
    from those.

Usage:
    kinsim refine in.pkl out.pkl
    kinsim refine in.pkl out.pkl --report report.tsv --min-pi 0.1
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CHI2_95_2DOF = 5.991                              # χ² 0.95 quantile, 2 dof
MOD_NAMES = {0: "none", 1: "m6A", 2: "m4C", 3: "m5C"}


# ---------------------------------------------------------------------------
# Core: 2-component GMM with fixed None component
# ---------------------------------------------------------------------------

def em_fixed_none(
    samples_m: np.ndarray,      # (N, 2) mixture, log1p space
    mu_n:      np.ndarray,      # (2,)   None mean
    sigma_n:   np.ndarray,      # (2, 2) None covariance
    max_iter:  int = 30,
    tol:       float = 1e-4,
    ridge:     float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fit N(μ_m, Σ_m) + π against fixed N(μ_n, Σ_n).

    Returns:  (μ_m, Σ_m, π, converged).
    """
    n = len(samples_m)
    if n < 2:
        return mu_n.copy(), sigma_n.copy(), 0.0, False

    sigma_n_reg = sigma_n + ridge * np.eye(2)
    try:
        inv_n = np.linalg.inv(sigma_n_reg)
    except np.linalg.LinAlgError:
        return mu_n.copy(), sigma_n.copy(), 0.0, False
    det_n = np.linalg.det(sigma_n_reg)
    if det_n <= 0:
        return mu_n.copy(), sigma_n.copy(), 0.0, False

    std_n = np.sqrt(np.clip(np.diag(sigma_n_reg), 0, None))
    mu_m = mu_n + 1.0 * std_n
    sigma_m = sigma_n_reg.copy()
    pi = 0.5
    prev_ll = -np.inf

    for _ in range(max_iter):
        sigma_m_reg = sigma_m + ridge * np.eye(2)
        try:
            inv_m = np.linalg.inv(sigma_m_reg)
        except np.linalg.LinAlgError:
            return mu_n.copy(), sigma_n.copy(), 0.0, False
        det_m = np.linalg.det(sigma_m_reg)
        if det_m <= 0:
            return mu_n.copy(), sigma_n.copy(), 0.0, False

        diff_m = samples_m - mu_m
        diff_n = samples_m - mu_n
        log_p_m = -0.5 * np.einsum("ij,jk,ik->i", diff_m, inv_m, diff_m) - 0.5 * np.log(det_m)
        log_p_n = -0.5 * np.einsum("ij,jk,ik->i", diff_n, inv_n, diff_n) - 0.5 * np.log(det_n)

        log_pi_m = np.log(max(pi, 1e-12))       + log_p_m
        log_pi_n = np.log(max(1.0 - pi, 1e-12)) + log_p_n
        log_max  = np.maximum(log_pi_m, log_pi_n)
        log_sum  = log_max + np.log(np.exp(log_pi_m - log_max) + np.exp(log_pi_n - log_max))
        gamma    = np.exp(log_pi_m - log_sum)

        w = gamma.sum()
        if w < 1e-6:
            return mu_n.copy(), sigma_n.copy(), 0.0, False
        pi_new   = float(w / n)
        mu_m_new = (gamma[:, None] * samples_m).sum(axis=0) / w

        # Methylation can only SLOW the polymerase → μ_m[IPD] ≥ μ_n[IPD].
        if mu_m_new[0] < mu_n[0]:
            mu_m_new[0] = mu_n[0] + 0.05

        diff = samples_m - mu_m_new
        sigma_m_new = (gamma[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / w

        ll = float(log_sum.sum())
        converged = abs(ll - prev_ll) < tol

        mu_m, sigma_m, pi = mu_m_new, sigma_m_new, pi_new
        prev_ll = ll
        if converged:
            return mu_m, sigma_m, pi, True

    return mu_m, sigma_m, pi, False


def mahalanobis_fallback(
    samples_m: np.ndarray,
    mu_n:      np.ndarray,
    sigma_n:   np.ndarray,
    ridge:     float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep samples with d²(·, None) > χ²_{2, 0.95}; estimate μ_m, Σ_m from them."""
    sigma_n_reg = sigma_n + ridge * np.eye(2)
    try:
        inv_n = np.linalg.inv(sigma_n_reg)
    except np.linalg.LinAlgError:
        return mu_n.copy(), sigma_n.copy(), 0.0

    diff = samples_m - mu_n
    d2   = np.einsum("ij,jk,ik->i", diff, inv_n, diff)
    mask = d2 > CHI2_95_2DOF
    kept = samples_m[mask]
    if len(kept) < 3:
        return mu_n.copy(), sigma_n.copy(), 0.0

    mu_m    = kept.mean(axis=0)
    sigma_m = np.cov(kept, rowvar=False) + ridge * np.eye(2)
    if mu_m[0] < mu_n[0]:
        mu_m[0] = mu_n[0] + 0.05
    pi = float(mask.mean())
    return mu_m, sigma_m, pi


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def refine_pkl(
    in_path:     Path,
    out_path:    Path,
    report_path: Path | None = None,
    em_max_iter: int   = 30,
    em_tol:      float = 1e-4,
    min_samples: int   = 30,
    min_pi:      float = 0.05,
    min_sep:     float = 0.3,
    seed:        int   = 42,
) -> dict:
    log.info("Loading: %s  (%.2f GB)", in_path, in_path.stat().st_size / 1e9)
    with open(in_path, "rb") as f:
        data = pickle.load(f)

    orig_meta = data.pop("__meta__", None)
    rng = np.random.default_rng(seed)

    by_kmer: dict[int, dict[int, np.ndarray]] = {}
    for key, arr in data.items():
        if not isinstance(key, tuple) or not isinstance(arr, np.ndarray):
            continue
        kmer_id, meth_id = key
        by_kmer.setdefault(int(kmer_id), {})[int(meth_id)] = arr

    log.info("Input: %d unique kmers", len(by_kmer))

    out: dict = {}
    rows: list = []
    status_counter: Counter = Counter()
    n_samples_in  = 0
    n_samples_out = 0

    for kmer_id, buckets in by_kmer.items():
        none_arr = buckets.get(0)
        if none_arr is not None:
            out[(kmer_id, 0)] = none_arr
            n_samples_in  += len(none_arr)
            n_samples_out += len(none_arr)

        for meth_id, arr in buckets.items():
            if meth_id == 0:
                continue
            n_orig        = len(arr)
            n_samples_in += n_orig
            mod_name      = MOD_NAMES.get(meth_id, f"mod{meth_id}")

            if none_arr is None or len(none_arr) < 10:
                status = "no_none_pair"
                status_counter[status] += 1
                rows.append((kmer_id, mod_name, n_orig, 0, 0.0, 0.0, status))
                continue

            xy_none = np.log1p(none_arr[:, :2].astype(np.float32))
            xy_meth = np.log1p(arr[:, :2].astype(np.float32))

            mu_n = xy_none.mean(axis=0)
            if len(xy_none) >= 2:
                sigma_n = np.cov(xy_none, rowvar=False)
                if sigma_n.ndim < 2:
                    sigma_n = np.eye(2) * float(sigma_n)
            else:
                sigma_n = np.eye(2) * 0.01

            if n_orig < min_samples:
                mu_m, sigma_m, pi = mahalanobis_fallback(xy_meth, mu_n, sigma_n)
                status = "fallback_few_samples" if pi > 0 else "skip_fallback_empty"
            else:
                mu_m, sigma_m, pi, converged = em_fixed_none(
                    xy_meth, mu_n, sigma_n, max_iter=em_max_iter, tol=em_tol,
                )
                if not converged:
                    mu_m, sigma_m, pi = mahalanobis_fallback(xy_meth, mu_n, sigma_n)
                    status = "fallback_em_nonconverged" if pi > 0 else "skip_em_failed"
                elif pi < min_pi:
                    status = "skip_low_pi"
                else:
                    status = "em_ok"

            sep = float(
                np.linalg.norm(mu_m - mu_n) / np.sqrt(max(float(np.trace(sigma_n)), 1e-9))
            )
            if status == "em_ok" and sep < min_sep:
                status = "skip_low_sep"

            if status.startswith("skip") or pi <= 0.0:
                status_counter[status] += 1
                rows.append((kmer_id, mod_name, n_orig, 0, pi, sep, status))
                continue

            n_new = max(3, int(round(pi * n_orig)))
            try:
                L = np.linalg.cholesky(sigma_m + 1e-4 * np.eye(2))
            except np.linalg.LinAlgError:
                std = np.sqrt(np.clip(np.diag(sigma_m), 1e-4, None))
                L = np.diag(std)
            z        = rng.standard_normal((n_new, 2)).astype(np.float32)
            xy_new_l = mu_m + z @ L.T
            xy_new   = np.expm1(xy_new_l).clip(0, 255).astype(np.float32)
            frac_col = np.full((n_new, 1), float(pi), dtype=np.float32)
            new_arr  = np.concatenate([xy_new, frac_col], axis=1)

            out[(kmer_id, meth_id)] = new_arr
            n_samples_out += n_new
            status_counter[status] += 1
            rows.append((kmer_id, mod_name, n_orig, n_new, pi, sep, status))

    out["__meta__"] = {
        "refined_from":     str(in_path),
        "method":           "em_fixed_none + mahalanobis_fallback",
        "space_for_fit":    "log1p",
        "space_for_store":  "raw",
        "em_max_iter":      em_max_iter,
        "em_tol":           em_tol,
        "min_samples":      min_samples,
        "min_pi":           min_pi,
        "min_sep":          min_sep,
        "seed":             seed,
        "n_kmers":          len(by_kmer),
        "n_samples_in":     n_samples_in,
        "n_samples_out":    n_samples_out,
        "status_counts":    dict(status_counter),
        "original_meta":    orig_meta,
    }

    log.info("Writing: %s", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("Status summary (meth buckets only):")
    for s, c in status_counter.most_common():
        log.info("  %-28s %d", s, c)
    log.info("Samples in:  %d", n_samples_in)
    log.info("Samples out: %d  (Δ = %+d)", n_samples_out, n_samples_out - n_samples_in)

    if report_path is not None:
        log.info("Report: %s", report_path)
        with open(report_path, "w") as f:
            f.write("kmer_id\tmeth\tn_original\tn_kept\tpi\tseparation\tstatus\n")
            for r in rows:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]:.4f}\t{r[5]:.4f}\t{r[6]}\n")

    return dict(status_counter)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    from .utils.config import setup_logging

    ap = argparse.ArgumentParser(
        prog="kinsim refine",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_pkl", help="Input .pkl (typically a merged master)")
    ap.add_argument("output_pkl", help="Output refined .pkl")
    ap.add_argument("--report", help="Optional per-key TSV report")
    ap.add_argument("--em-max-iter", type=int,   default=30)
    ap.add_argument("--em-tol",      type=float, default=1e-4)
    ap.add_argument("--min-samples", type=int,   default=30,
                    help="Meth buckets with fewer samples skip EM and go to Mahalanobis fallback")
    ap.add_argument("--min-pi",      type=float, default=0.05,
                    help="Drop keys where fitted π < this (no real meth signal)")
    ap.add_argument("--min-sep",     type=float, default=0.3,
                    help="Drop keys where ||μ_m - μ_n||/√tr(Σ_n) < this (signal too weak)")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(verbose=args.verbose)

    in_p  = Path(args.input_pkl)
    out_p = Path(args.output_pkl)
    rep_p = Path(args.report) if args.report else None
    if not in_p.exists():
        print(f"ERROR: {in_p} not found", file=sys.stderr)
        sys.exit(1)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if rep_p is not None:
        rep_p.parent.mkdir(parents=True, exist_ok=True)

    refine_pkl(
        in_p, out_p, report_path=rep_p,
        em_max_iter=args.em_max_iter, em_tol=args.em_tol,
        min_samples=args.min_samples, min_pi=args.min_pi, min_sep=args.min_sep,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
