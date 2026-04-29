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
    method:      str   = "gmm_signature",
    chi2_99:     bool  = True,
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

            # Threshold: stricter χ²_{0.99} (1% rejection) by default, or χ²_{0.95}
            # (5%) when chi2_99=False.  Low-count keys ALWAYS use the strict version
            # regardless, to avoid noise leakage.
            base_chi2 = CHI2_99_2DOF if chi2_99 else CHI2_95_2DOF
            chi2_t = CHI2_99_2DOF if n_orig < strict_n else base_chi2

            if method == "gmm_signature":
                # Load config (cached) and resolve signature offsets.
                from .utils.config import load_kinsim_config
                cfg = load_kinsim_config()
                sig_cfg  = cfg.get("kinetic_signatures", {}).get(mod_name, {})
                signal_offsets = list(sig_cfg.get("signal_offsets", [0]))
                gmm_cfg  = cfg.get("refine", {}).get("gmm_signature", {})
                k_max          = int(gmm_cfg.get("k_max", 3))
                cfg_chi2       = float(gmm_cfg.get("chi2_threshold", CHI2_99_2DOF))
                min_sig_ratio  = float(gmm_cfg.get("min_signature_ratio", 1.3))
                cfg_min_pi     = float(gmm_cfg.get("min_pi", 0.05))
                min_for_gmm    = int(gmm_cfg.get("min_samples_for_gmm", 5))

                # Resolve where the kinetic profile starts in `arr`.
                # See extract.py SAMPLE_NCOLS layout: 3 + 11 + 9 + 9 = 32.
                # The IPD profile starts at column 3 + METH_CTX_LEN.
                from .extract import METH_CTX_LEN, PROFILE_LEN
                profile_start_col = 3 + METH_CTX_LEN

                if arr.shape[1] < profile_start_col + PROFILE_LEN:
                    # Old-format pkl: no profile stored. Fall back to centroid-only.
                    log.warning("kmer=%d %s: pkl lacks kinetic profile (cols=%d) — "
                                "using centroid-only Mahalanobis filter",
                                kmer_id, mod_name, arr.shape[1])
                    mu_m, sigma_m, pi = mahalanobis_fallback(
                        xy_meth, mu_n, sigma_n, chi2_threshold=chi2_t,
                    )
                    status = "fallback_no_profile" if pi > 0 else "skip_fallback_empty"
                elif n_orig < min_for_gmm:
                    # Too few samples for GMM: Mahalanobis hard filter from None.
                    mu_m, sigma_m, pi = mahalanobis_fallback(
                        xy_meth, mu_n, sigma_n, chi2_threshold=chi2_t,
                    )
                    status = "lowN_mahal_kept" if pi > 0 else "lowN_mahal_empty"
                else:
                    result = gmm_signature_validate(
                        arr, mu_n, sigma_n,
                        sig_offsets=signal_offsets,
                        profile_start_col=profile_start_col,
                        profile_len=PROFILE_LEN,
                        k_max=k_max,
                        chi2_t=cfg_chi2,
                        min_sig_ratio=min_sig_ratio,
                        min_pi=cfg_min_pi,
                        none_arr=none_arr,            # joint clustering needs None reference
                    )
                    if result is None:
                        mu_m, sigma_m, pi = mu_n.copy(), sigma_n.copy(), 0.0
                        status = "skip_gmm_no_valid"
                    else:
                        mu_m, sigma_m, pi, status = result
            elif method == "clustered":
                # Cluster the meth bucket, pick the component farthest from None.
                # Handles bimodal contamination correctly: BIC selects K=2 when
                # the bucket is a mix of real meth + near-None contamination,
                # and we keep only the far cluster.
                result = cluster_pick_farthest(
                    xy_meth, mu_n, sigma_n,
                    k_max=3, min_d2=chi2_t,
                )
                if result is None:
                    mu_m, sigma_m, pi = mu_n.copy(), sigma_n.copy(), 0.0
                    status = "skip_no_far_cluster"
                else:
                    mu_m, sigma_m, pi = result
                    if pi < min_pi:
                        status = "skip_low_pi"
                    else:
                        status = "clustered_ok"
            elif method == "mahalanobis":
                # Hard cutoff: keep only samples beyond the χ² boundary from None.
                # This sidesteps EM's tendency to fit a wide Gaussian that swallows
                # both the real meth peak and near-None contamination.
                mu_m, sigma_m, pi = mahalanobis_fallback(
                    xy_meth, mu_n, sigma_n, chi2_threshold=chi2_t,
                )
                if pi <= 0:
                    status = "skip_mahal_empty"
                elif pi < min_pi:
                    status = "skip_low_pi"
                else:
                    status = "mahal_strict" if chi2_t > CHI2_95_2DOF else "mahal_ok"
            elif n_orig < min_samples:
                mu_m, sigma_m, pi = mahalanobis_fallback(
                    xy_meth, mu_n, sigma_n, chi2_threshold=chi2_t,
                )
                status = "fallback_few_samples_strict" if chi2_t > CHI2_95_2DOF and pi > 0 \
                         else "fallback_few_samples" if pi > 0 \
                         else "skip_fallback_empty"
            else:
                mu_m, sigma_m, pi, converged = em_fixed_none(
                    xy_meth, mu_n, sigma_n, max_iter=em_max_iter, tol=em_tol,
                )
                if not converged:
                    mu_m, sigma_m, pi = mahalanobis_fallback(
                        xy_meth, mu_n, sigma_n, chi2_threshold=chi2_t,
                    )
                    status = "fallback_em_nonconverged_strict" if chi2_t > CHI2_95_2DOF and pi > 0 \
                             else "fallback_em_nonconverged" if pi > 0 \
                             else "skip_em_failed"
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
            # Preserve meth context (and profile if present) by sampling from
            # the original array's rows. Each refined sample inherits the
            # context of a random original sample.
            n_input_cols = arr.shape[1]
            if n_input_cols > 3:
                # Sample row indices from the original bucket, preserving cols 3..end.
                src_idx = rng.integers(0, n_orig, size=n_new)
                tail    = arr[src_idx, 3:].astype(np.float32)
                new_arr = np.concatenate([xy_new, frac_col, tail], axis=1)
            else:
                new_arr = np.concatenate([xy_new, frac_col], axis=1)

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
        "strict_n":         strict_n,
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
                    choices=["gmm_signature", "clustered", "mahalanobis", "em"],
                    default="gmm_signature",
                    help="Refinement method (default: gmm_signature). "
                         "'gmm_signature' = GMM (BIC selects K) + per-component "
                         "validation using kinetic profile signature offsets from "
                         "kinsim_config.yaml. Handles m6A/m4C/m5C uniformly. "
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
