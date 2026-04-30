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


def gmm_signature_validate(
    arr:        np.ndarray,         # (N, SAMPLE_NCOLS) — meth bucket samples
    mu_n:       np.ndarray,         # (2,)   None mean (log1p space)
    sigma_n:    np.ndarray,         # (2, 2) None covariance (log1p space)
    sig_offsets: list[int],          # kept for compat
    profile_start_col: int,          # kept for compat
    profile_len:       int,          # kept for compat
    k_max:      int   = 3,
    chi2_t:     float = 9.21,        # kept for compat
    min_sig_ratio: float = 1.3,      # kept for compat
    min_pi:     float = 0.05,
    ridge:      float = 1e-4,
    none_arr:   np.ndarray | None = None,   # (M, SAMPLE_NCOLS) — None bucket samples
) -> tuple[np.ndarray, np.ndarray, float, str] | None:
    """Joint-GMM clustering on combined (None + Meth) samples per kmer.

    Algorithm:
      1. Stack the (kmer, none) and (kmer, meth) samples into one combined
         dataset (log1p IPD, log1p PW).
      2. Fit a GMM with K=2 (or K=3 if BIC prefers, e.g. partial methylation
         as a third middle cluster).
      3. Identify the "real meth" cluster as the one with the highest mean
         IPD. The other cluster(s) are None / contamination.
      4. For the meth samples only, keep those assigned to the highest-IPD
         cluster. These are the truly methylated samples; the rest are
         contamination that snuck through upstream filtering.
      5. Re-estimate (μ_m, Σ_m) from the kept meth samples.

    Why combined GMM (vs Mahalanobis-only): the GMM learns the cluster
    boundary from the data (using both None and Meth populations) rather
    than assuming a fixed distance threshold. This is more robust when the
    None distribution is non-Gaussian or when the meth signal is partial.
    """
    from sklearn.mixture import GaussianMixture

    n = len(arr)
    if n < 2:
        return None
    if none_arr is None or len(none_arr) < 2:
        return None  # Need a None reference to do joint clustering

    # --- Per-offset clustering: one GMM per signature offset ---
    # For multi-offset signatures (m6A: [0, 5]; m5C: [2, 6]), each offset
    # gets its own joint-GMM (None + Meth on that single offset's IPD).
    # The final keep_mask is the INTERSECTION across all offsets — a sample
    # must look "real meth" at EVERY signature position to survive.
    # This is stricter than a single multi-dim GMM and treats each
    # signature event as an independent confirmation.
    sig_idx = [i for i in sig_offsets if 0 <= i < profile_len]
    if not sig_idx:
        sig_idx = [0]   # fallback to center

    # Original (IPD, PW) center values — used to compute the final (mu, Sigma)
    # to store for training. Clustering happens on signature features.
    xy_meth = np.log1p(arr[:, :2].astype(np.float32))

    # --- GMM with K = 1..k_max (BIC selection) ---
    # --- Per-offset joint-GMM, intersect the keep masks ---
    # For each signature offset (e.g. +0 and +5 for m6A), fit a 1D
    # joint-GMM on the combined (None + Meth) IPDs at that offset, identify
    # the "real meth" cluster (highest mean), classify the meth samples.
    # A sample must pass at EVERY offset to be kept.
    keep_masks_per_offset: list[np.ndarray] = []
    K_per_offset: list[int] = []
    K_chosen_any_above_1 = False

    for off in sig_idx:
        col = profile_start_col + off
        feats_meth_1d = np.log1p(arr[:, col].astype(np.float32)).reshape(-1, 1)
        feats_none_1d = np.log1p(none_arr[:, col].astype(np.float32)).reshape(-1, 1)
        feats_combined = np.concatenate([feats_none_1d, feats_meth_1d], axis=0)

        best_bic = np.inf
        best_gmm = None
        n_combined = len(feats_combined)
        for k in range(1, k_max + 1):
            if k > n_combined // 3 and k > 1:
                break
            try:
                gmm = GaussianMixture(
                    n_components=k, covariance_type="full",
                    reg_covar=ridge, max_iter=100, n_init=2, random_state=42,
                )
                gmm.fit(feats_combined)
                bic = float(gmm.bic(feats_combined))
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except (ValueError, np.linalg.LinAlgError):
                continue

        if best_gmm is None:
            return None

        K_off = best_gmm.n_components
        K_per_offset.append(K_off)

        if K_off == 1:
            # No separation found at this offset — accept all meth samples
            # (we can't filter, fall back on upstream confidence).
            keep_masks_per_offset.append(np.ones(n, dtype=bool))
            continue

        K_chosen_any_above_1 = True
        # Real-meth cluster: highest mean (methylation only raises IPD)
        mean_per_cluster = best_gmm.means_[:, 0]                # (K,)
        real_meth_id = int(np.argmax(mean_per_cluster))
        meth_labels = best_gmm.predict(feats_meth_1d)
        keep_masks_per_offset.append(meth_labels == real_meth_id)

    # Intersection: a sample passes only if "real meth" at every signature offset
    keep_mask = np.logical_and.reduce(keep_masks_per_offset)
    n_kept = int(keep_mask.sum())
    if n_kept < 3:
        return None

    if not K_chosen_any_above_1:
        # All offsets ended up K=1 (no separation possible) — keep all
        # meth samples. The caller's upstream jasmine/motifmaker
        # confidence is the safety net here.
        xy_kept = xy_meth
    else:
        xy_kept = xy_meth[keep_mask]

    mu_m    = xy_kept.mean(axis=0).astype(np.float32)
    sigma_m = (np.cov(xy_kept, rowvar=False) + ridge * np.eye(2)).astype(np.float32)
    if sigma_m.ndim < 2:
        sigma_m = np.eye(2, dtype=np.float32) * float(sigma_m)
    if mu_m[0] < mu_n[0]:
        mu_m[0] = mu_n[0] + 0.05
    pi = float(n_kept / n)
    if pi < min_pi:
        return None

    K_str = "_".join(str(k) for k in K_per_offset)              # e.g. "2_3" for m6A
    status = f"gmm_{K_str}_real_kept" if K_chosen_any_above_1 else "gmm_all1_kept"
    return mu_m, sigma_m, pi, status


def cluster_pick_farthest(
    samples_m: np.ndarray,        # (N, 2) meth samples, log1p space
    mu_n:      np.ndarray,        # (2,)   None mean
    sigma_n:   np.ndarray,        # (2, 2) None covariance
    k_max:     int   = 3,
    ridge:     float = 1e-4,
    min_d2:    float = CHI2_99_2DOF,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Cluster the meth bucket; keep the component farthest from None.

    Steps:
      1. Fit GMM with K=1..k_max components on the meth samples (BIC selects K).
      2. Compute Mahalanobis distance from each component centroid to None.
      3. Return (μ_m, Σ_m) of the component with the largest distance, with π
         set to that component's mixing weight (fraction of samples assigned).
      4. If the farthest centroid is still within χ²_{0.99} of None, return
         None — there is no real meth signal in this bucket.

    Robust to bimodal contamination: when meth samples are a mix of real
    methylated reads and near-None partial-methylation/false-positive reads,
    BIC favours K=2 and we keep only the far component.

    Returns:
        (μ_m, Σ_m, π) on success, or None if no separable cluster found.
    """
    from sklearn.mixture import GaussianMixture

    n = len(samples_m)
    if n < 5:
        return None

    sigma_n_reg = sigma_n + ridge * np.eye(2)
    try:
        inv_n = np.linalg.inv(sigma_n_reg)
    except np.linalg.LinAlgError:
        return None

    best_bic = np.inf
    best_gmm = None
    for k in range(1, k_max + 1):
        if k > n // 3:
            break
        try:
            gmm = GaussianMixture(
                n_components=k, covariance_type="full",
                reg_covar=ridge, max_iter=100, n_init=2, random_state=42,
            )
            gmm.fit(samples_m)
            bic = float(gmm.bic(samples_m))
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except (ValueError, np.linalg.LinAlgError):
            continue

    if best_gmm is None:
        return None

    distances = []
    for kk in range(best_gmm.n_components):
        diff = best_gmm.means_[kk] - mu_n
        d2 = float(diff @ inv_n @ diff)
        distances.append(d2)

    best_k = int(np.argmax(distances))
    if distances[best_k] < min_d2:
        return None  # closest component still too close to None

    mu_m    = best_gmm.means_[best_k].astype(np.float32).copy()
    sigma_m = best_gmm.covariances_[best_k].astype(np.float32)
    if mu_m[0] < mu_n[0]:
        mu_m[0] = mu_n[0] + 0.05

    pi = float(best_gmm.weights_[best_k])
    return mu_m, sigma_m, pi


def mahalanobis_fallback(
    samples_m: np.ndarray,
    mu_n:      np.ndarray,
    sigma_n:   np.ndarray,
    ridge:     float = 1e-4,
    chi2_threshold: float = CHI2_95_2DOF,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep samples with d²(·, None) > threshold; estimate μ_m, Σ_m from them.

    Threshold defaults to χ²_{2, 0.95} (5% rejection). When called with the
    stricter χ²_{2, 0.99} = 9.21, only keeps samples that have ≤1% chance
    of belonging to the None distribution. Use the strict variant for
    low-count keys where EM is unreliable and loose thresholds let noise
    leak into the methylated bucket.
    """
    sigma_n_reg = sigma_n + ridge * np.eye(2)
    try:
        inv_n = np.linalg.inv(sigma_n_reg)
    except np.linalg.LinAlgError:
        return mu_n.copy(), sigma_n.copy(), 0.0

    diff = samples_m - mu_n
    d2   = np.einsum("ij,jk,ik->i", diff, inv_n, diff)
    mask = d2 > chi2_threshold
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
    strict_n:    int   = 50,
    min_pi:      float = 0.05,
    min_sep:     float = 0.3,
    seed:        int   = 42,
    method:      str   = "global_gmm",
    chi2_99:     bool  = True,
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
    if method != "global_gmm":
        log.warning("Legacy method %r requested - using global_gmm.", method)

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
    ap.add_argument("--strict-n",    type=int,   default=50,
                    help="For keys with <N samples use the stricter χ²_{0.99} threshold "
                         "(1%% chance of being None) in the Mahalanobis fallback, instead "
                         "of the default χ²_{0.95} (5%%). Reduces noise leakage when EM "
                         "fit is unreliable or the fallback is triggered by low count.")
    ap.add_argument("--min-pi",      type=float, default=0.05,
                    help="Drop keys where fitted π < this (no real meth signal)")
    ap.add_argument("--min-sep",     type=float, default=0.3,
                    help="Drop keys where ||μ_m - μ_n||/√tr(Σ_n) < this (signal too weak)")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--method",
                    choices=["global_gmm", "gmm_signature", "clustered",
                             "mahalanobis", "em"],
                    default="global_gmm",
                    help="Refinement method (default: global_gmm). "
                         "'global_gmm' = pool ALL samples per meth_id, fit one "
                         "GMM globally, reject samples not in the highest-IPD "
                         "cluster (false-positive motif matches). Buckets that "
                         "shrink below 3 samples are dropped entirely. "
                         "'gmm_signature' = legacy per-bucket GMM. "
                         "'clustered' = legacy GMM picking the centroid farthest "
                         "from None (no profile validation). "
                         "'mahalanobis' = hard χ² filter on each sample's distance "
                         "from None. "
                         "'em' = legacy EM fixed-None mixture (can over-fit).")
    ap.add_argument("--no-chi2-99",  dest="chi2_99", action="store_false",
                    help="Use χ²_{0.95} (5%% rejection) instead of the default "
                         "χ²_{0.99} (1%% rejection). Keeps more samples but lets more "
                         "near-None contamination leak into the meth bucket.")
    ap.set_defaults(chi2_99=True)
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
        min_samples=args.min_samples, strict_n=args.strict_n,
        min_pi=args.min_pi, min_sep=args.min_sep,
        seed=args.seed,
        method=args.method,
        chi2_99=args.chi2_99,
    )


if __name__ == "__main__":
    main()
