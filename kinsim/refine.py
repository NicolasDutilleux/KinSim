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
    Mahalanobis threshold against None: samples beyond the χ² threshold
    are kept as real methylated; μ_m, Σ_m estimated from those.
    - Default :  χ²_{2, 0.95} = 5.99  (5 %% chance of being None)
    - Strict  :  χ²_{2, 0.99} = 9.21  (1 %% chance of being None) —
      auto-enabled when the key has < `strict_n` samples (default 50)
      to avoid letting noise leak into the methylated bucket.

Usage:
    kinsim refine in.pkl out.pkl
    kinsim refine in.pkl out.pkl --report report.tsv --min-pi 0.1
    kinsim refine in.pkl out.pkl --strict-n 100          # stricter: strict below 100
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

CHI2_95_2DOF = 5.991        # χ² 0.95 quantile, 2 dof — 5% rejection (default)
CHI2_99_2DOF = 9.210        # χ² 0.99 quantile, 2 dof — 1% rejection (strict, low-count)
MOD_NAMES = {0: "none", 1: "m6A", 2: "m4C", 3: "m5C"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def refine_pkl(
    in_path:     Path,
    out_path:    Path,
    report_path: Path | None = None,
    seed:        int           = 42,
) -> dict:
    """Refine a KinSim master.pkl with a GLOBAL per-meth-type GMM filter.

    For each methylation type (m6A, m4C, m5C):
      1. Pool ALL samples across all (kmer, meth_id) buckets.
      2. Build a global None reference by sub-sampling all (kmer, none).
      3. Fit a 2-D GMM (BIC selects K in {2, 3}) on the pooled meth samples.
      4. Identify the real-meth cluster: highest mean IPD (methylation
         only RAISES IPD, never lowers it).
      5. Reject samples assigned to non-real clusters - they are false-
         positive motif matches. Buckets that shrink below 3 samples are
         dropped entirely.

    The output keeps all (kmer, none) buckets unchanged plus the surviving
    (kmer, meth) buckets that contain only the real-meth-cluster samples.
    """
    log.info("Loading: %s  (%.2f GB)", in_path, in_path.stat().st_size / 1e9)
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    orig_meta = data.pop("__meta__", None)
    rng = np.random.default_rng(seed)

    # 1. Group buckets by meth_id
    none_buckets = {}
    meth_buckets = {}
    for key, arr in data.items():
        if not isinstance(key, tuple) or not isinstance(arr, np.ndarray):
            continue
        kmer_id, meth_id = int(key[0]), int(key[1])
        if meth_id == 0:
            none_buckets[kmer_id] = arr
        else:
            meth_buckets.setdefault(meth_id, {})[kmer_id] = arr

    n_unique_kmers = len(set(none_buckets) |
                         {k for d in meth_buckets.values() for k in d})
    log.info("Input: %d unique kmers, %d (kmer, none) buckets",
             n_unique_kmers, len(none_buckets))
    for mid in sorted(meth_buckets):
        log.info("  %s: %d buckets, %d samples",
                 MOD_NAMES.get(mid, "mod%d" % mid),
                 len(meth_buckets[mid]),
                 sum(len(a) for a in meth_buckets[mid].values()))

    # 2. Global None reference
    if not none_buckets:
        log.error("No (kmer, none) buckets - refine cannot proceed")
        return {}
    none_pool_log = np.concatenate([
        np.log1p(arr[:, :2].astype(np.float32))
        for arr in none_buckets.values()
    ])
    if len(none_pool_log) > 200_000:
        idx = rng.choice(len(none_pool_log), 200_000, replace=False)
        none_pool_log = none_pool_log[idx]
    mu_n_global = none_pool_log.mean(axis=0).astype(np.float32)
    log.info("Global None reference (log1p): mu=[%.3f, %.3f]",
             mu_n_global[0], mu_n_global[1])

    # 3. Output: all None buckets unchanged
    out = {}
    n_samples_in = 0
    n_samples_out = 0
    for kmer_id, arr in none_buckets.items():
        out[(kmer_id, 0)] = arr.copy()
        n_samples_in  += len(arr)
        n_samples_out += len(arr)

    rows = []
    status_counter = Counter()
    n_meth_in = n_meth_kept = n_meth_dropped = 0
    n_buckets_dropped = 0

    from sklearn.mixture import GaussianMixture
    from collections import defaultdict as _dd

    # Resolve signature offsets per meth type from config
    from .utils.config import load_kinsim_config
    from .extract import METH_CTX_LEN, PROFILE_LEN
    cfg = load_kinsim_config()
    profile_start_col = 3 + METH_CTX_LEN   # = 14: cols 14..14+PROFILE_LEN-1 are IPD profile

    # 4. Process each meth type GLOBALLY
    for meth_id in sorted(meth_buckets):
        buckets = meth_buckets[meth_id]
        if not buckets:
            continue
        mod_name = MOD_NAMES.get(meth_id, "mod%d" % meth_id)

        # Pick the IPD columns to use as GMM features, based on the signature
        # offsets configured for this meth type. For m6A this is the IPD at
        # offsets [0, 5]; for m4C [0]; for m5C [2, 6] (the C itself has no
        # signal, so we MUST use profile aval to find the real-meth cluster).
        sig_cfg = cfg.get("kinetic_signatures", {}).get(mod_name, {})
        sig_offsets = list(sig_cfg.get("signal_offsets", [0]))
        sig_offsets = [o for o in sig_offsets if 0 <= o < PROFILE_LEN]
        if not sig_offsets:
            sig_offsets = [0]
        feature_cols = [profile_start_col + o for o in sig_offsets]
        log.info("[%s] signature offsets %s -> GMM features at cols %s",
                 mod_name, sig_offsets, feature_cols)

        all_arrays = []
        sample_kmer = []
        for kmer_id, arr in buckets.items():
            all_arrays.append(arr)
            sample_kmer.extend([kmer_id] * len(arr))
        meth_pool = np.concatenate(all_arrays).astype(np.float32)
        # Check storage layout supports the profile columns
        has_profile = meth_pool.shape[1] >= max(feature_cols) + 1
        if not has_profile:
            log.warning("[%s] pkl lacks profile cols (cols=%d) - falling back to "
                        "(IPD center, PW center) for GMM features",
                        mod_name, meth_pool.shape[1])
            meth_pool_log = np.log1p(meth_pool[:, :2])
        else:
            meth_pool_log = np.log1p(meth_pool[:, feature_cols])
        n_pool = len(meth_pool)
        n_samples_in += n_pool
        n_meth_in    += n_pool
        log.info("[%s] global pool: %d samples across %d buckets, "
                 "feature dim=%d", mod_name, n_pool, len(buckets),
                 meth_pool_log.shape[1])

        # Build a BALANCED None pool of equal size for the GMM fit. Including
        # the None reference in the fit gives the GMM a clear anchor for the
        # contamination cluster (otherwise BIC may pick a sub-structure of
        # the meth pool itself rather than separate meth from None).
        none_arrays = list(none_buckets.values())
        if none_arrays:
            none_full = np.concatenate(none_arrays).astype(np.float32)
            if has_profile and none_full.shape[1] >= max(feature_cols) + 1:
                none_full_log = np.log1p(none_full[:, feature_cols])
            else:
                none_full_log = np.log1p(none_full[:, :2])
            n_none_avail = len(none_full_log)
            n_balance    = min(n_pool, n_none_avail)
            if n_balance < n_none_avail:
                idx = rng.choice(n_none_avail, n_balance, replace=False)
                none_pool_log = none_full_log[idx]
            else:
                none_pool_log = none_full_log
            log.info("[%s] balanced None pool: %d samples", mod_name, len(none_pool_log))
        else:
            none_pool_log = np.empty((0, meth_pool_log.shape[1]), dtype=np.float32)

        # Combined pool: meth + balanced None
        combined_log = np.concatenate([meth_pool_log, none_pool_log], axis=0)
        n_combined   = len(combined_log)

        best_bic = np.inf
        best_gmm = None
        for k in (2, 3):
            if k * 5 > n_combined:
                continue
            try:
                gmm = GaussianMixture(
                    n_components=k, covariance_type="full",
                    reg_covar=1e-4, max_iter=100, n_init=2,
                    random_state=seed,
                )
                gmm.fit(combined_log)
                bic = float(gmm.bic(combined_log))
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except (ValueError, np.linalg.LinAlgError):
                continue

        if best_gmm is None:
            log.warning("[%s] GMM fit failed - keeping all samples", mod_name)
            for kmer_id, arr in buckets.items():
                out[(kmer_id, meth_id)] = arr.copy()
                n_samples_out += len(arr)
                n_meth_kept   += len(arr)
            continue

        K = best_gmm.n_components
        # Real-meth cluster: highest MEAN across all signature features
        # (methylation only RAISES IPD at signature positions). For m6A this
        # averages IPD@+0 and IPD@+5; for m5C IPD@+2 and IPD@+6.
        cluster_score = best_gmm.means_.mean(axis=1)            # (K,)
        real_cluster = int(np.argmax(cluster_score))
        labels = best_gmm.predict(meth_pool_log)
        keep_mask = (labels == real_cluster)
        n_kept_pool = int(keep_mask.sum())

        log.info("[%s] BIC K=%d. cluster_means_log_IPD_per_feature=%s. "
                 "real_cluster=%d. Kept %d/%d (%.1f%%); dropped %d as false positives.",
                 mod_name, K,
                 best_gmm.means_.round(3).tolist(),
                 real_cluster, n_kept_pool, n_pool,
                 100.0 * n_kept_pool / max(n_pool, 1),
                 n_pool - n_kept_pool)

        # Redistribute kept samples back to their original kmers
        kept_per_kmer = _dd(list)
        for i, kmer_id in enumerate(sample_kmer):
            if keep_mask[i]:
                kept_per_kmer[kmer_id].append(meth_pool[i])

        for kmer_id, arr_orig in buckets.items():
            samples = kept_per_kmer.get(kmer_id, [])
            if len(samples) < 3:
                n_meth_dropped     += len(arr_orig)
                n_buckets_dropped  += 1
                rows.append((kmer_id, mod_name, len(arr_orig), 0,
                             0.0, 0.0, "global_gmm_bucket_dropped"))
                continue
            arr_new = np.array(samples, dtype=np.float32)
            out[(kmer_id, meth_id)] = arr_new
            n_samples_out += len(arr_new)
            n_meth_kept   += len(arr_new)
            n_meth_dropped += (len(arr_orig) - len(arr_new))
            rows.append((kmer_id, mod_name, len(arr_orig), len(arr_new),
                         len(arr_new) / max(len(arr_orig), 1), 0.0,
                         "global_gmm_kept"))

        status_counter["%s_kept" % mod_name] = n_meth_kept
        status_counter["%s_dropped" % mod_name] = n_meth_dropped
        status_counter["%s_buckets_dropped" % mod_name] = n_buckets_dropped

    log.info("Total meth samples in:    %d", n_meth_in)
    log.info("Kept as meth:             %d", n_meth_kept)
    log.info("Dropped (false-positive): %d", n_meth_dropped)
    log.info("Buckets fully dropped:    %d", n_buckets_dropped)

    # --- 5. Write output ---
    out["__meta__"] = {
        "refined_from":     str(in_path),
        "method":           "global_gmm",
        "space_for_fit":    "log1p",
        "space_for_store":  "raw",
        "seed":             seed,
        "n_unique_kmers":   n_unique_kmers,
        "n_samples_in":     n_samples_in,
        "n_samples_out":    n_samples_out,
        "n_meth_in":        n_meth_in,
        "n_meth_kept":      n_meth_kept,
        "n_meth_dropped":   n_meth_dropped,
        "n_buckets_dropped": n_buckets_dropped,
        "status_counts":    dict(status_counter),
        "original_meta":    orig_meta,
    }

    log.info("Writing: %s", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("Status summary:")
    for s, c in status_counter.most_common():
        log.info("  %-32s %d", s, c)
    log.info("Samples in:  %d", n_samples_in)
    log.info("Samples out: %d  (Δ = %+d)", n_samples_out, n_samples_out - n_samples_in)

    if report_path is not None:
        log.info("Report: %s", report_path)
        with open(report_path, "w") as f:
            f.write("kmer_id\tmeth\tn_original\tn_kept\tpi\tseparation\tstatus\n")
            for r in rows:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]:.4f}\t{r[5]:.4f}\t{r[6]}\n")

    return dict(status_counter)


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
    ap.add_argument("--seed",    type=int,   default=42,
                    help="Random seed for None-pool subsampling and GMM init")
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

    refine_pkl(in_p, out_p, report_path=rep_p, seed=args.seed)


if __name__ == "__main__":
    main()
