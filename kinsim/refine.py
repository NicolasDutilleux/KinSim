"""Refine a KinSim master .pkl by filtering motif false-positive ``slowed`` samples.

Input format: ``dict[kmer_id (int)] -> ndarray(N, 38)`` produced by
``kinsim extract`` + ``kinsim merge``. Cols 35/36/37 carry CATEGORY,
PARENT_METH and PARENT_OFFSET (see ``kinsim.utils.sample_layout``).

Two methods are supported:

* ``--method gmm`` (default) — for each meth type T present in the
  ``slowed`` rows, fit a 2-component Gaussian Mixture on the combined
  ``baseline + slowed_by_T`` IPD pool (baseline subsampled to match
  ``slowed_by_T`` count). Validate that ≥ ``baseline_validation_min``
  fraction of the baseline subsample lands in the lower-mean component;
  if so, drop ``slowed_by_T`` rows whose posterior in that component
  exceeds ``posterior_threshold``. If validation fails or the type has
  too few samples, keep all rows of that type. ``BASELINE`` and
  ``NEAR_METH`` always pass through unchanged.

* ``--method p95`` (legacy) — single global threshold = the
  ``secondary_percentile``-th percentile of the per-kmer baseline-mean
  distribution. Drop ``slowed`` rows below the threshold. Same threshold
  for every meth type, no per-kmer adaptation.

Why GMM is the default
----------------------
The p95 method applies one absolute IPD threshold to every meth type and
every kmer. That is unfair to (a) low-baseline kmers whose real meth
slowing produces IPDs below the global threshold, and (b) weak-signal
meth types (m4C, m5C) whose slowing factor is smaller than m6A's. The
GMM method fits a per-type model, lets the data decide where the
boundary between "baseline-like" and "real meth" sits, and self-validates
by checking that the baseline subsample really does cluster in the
lower-mean component before applying the cut.

Usage:
    kinsim refine in.pkl out.pkl                            # default GMM
    kinsim refine in.pkl out.pkl --method p95               # legacy
    kinsim refine in.pkl out.pkl --posterior-threshold 0.7  # stricter cut
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# p95 method (legacy) — single global threshold
# ---------------------------------------------------------------------------

# Min baseline samples per kmer required for that kmer to contribute a
# mean to the p95 threshold computation. Below this the per-kmer mean is
# too noisy to trust.
MIN_BASELINE_PER_KMER_FOR_THRESHOLD = 5


def slowed_split(
    data: dict,
    secondary_pct: float,
) -> tuple[dict, dict]:
    """Drop CATEGORY_SLOWED samples whose IPD falls below the
    ``secondary_pct`` percentile of the per-kmer baseline mean
    distribution.

    Args:
        data: dict[kmer_id -> ndarray(N, 38)]. Col 35 carries the
            category enum (0=baseline, 1=slowed, 2=near_meth).
        secondary_pct: percentile of the per-kmer baseline mean used as
            the lower threshold for slowed samples (typically 95).

    Returns:
        (new_data, stats) where new_data is a fresh dict with the same
        kmer keys and surviving rows, and stats is a small dict with
        in/out counts per category, the threshold used, and the
        baseline-distribution summary statistics.
    """
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
    )

    # 1. Threshold = percentile of per-kmer baseline means.
    kmer_baseline_means: list = []
    pooled_for_stats: list = []  # for diagnostics only
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_CATEGORY:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        m = cats == CATEGORY_BASELINE
        n_b = int(m.sum())
        if n_b == 0:
            continue
        ipds = arr[m, COL_IPD]
        pooled_for_stats.append(ipds)
        if n_b >= MIN_BASELINE_PER_KMER_FOR_THRESHOLD:
            kmer_baseline_means.append(float(ipds.mean()))

    if pooled_for_stats and kmer_baseline_means:
        pooled = np.concatenate(pooled_for_stats)
        kmer_means = np.array(kmer_baseline_means, dtype=np.float32)
        threshold = float(np.percentile(kmer_means, secondary_pct))

        log.info(
            "[refine.p95] baseline IPD per-sample:  n=%d  mean=%.2f  std=%.2f",
            len(pooled),
            float(pooled.mean()),
            float(pooled.std()),
        )
        log.info(
            "[refine.p95]   per-sample quantiles:    "
            "p5=%.0f  p50=%.0f  p75=%.0f  p90=%.0f  p95=%.0f  p99=%.0f  max=%.0f",
            float(np.percentile(pooled, 5)),
            float(np.percentile(pooled, 50)),
            float(np.percentile(pooled, 75)),
            float(np.percentile(pooled, 90)),
            float(np.percentile(pooled, 95)),
            float(np.percentile(pooled, 99)),
            float(pooled.max()),
        )
        log.info(
            "[refine.p95] baseline PER-KMER MEAN: n_kmers=%d (>= %d samples each)  "
            "mean=%.2f  std=%.2f",
            len(kmer_means),
            MIN_BASELINE_PER_KMER_FOR_THRESHOLD,
            float(kmer_means.mean()),
            float(kmer_means.std()),
        )
        log.info(
            "[refine.p95] threshold = p%g(baseline PER-KMER MEAN) = %.2f",
            secondary_pct,
            threshold,
        )
    else:
        threshold = 0.0
        log.warning("[refine.p95] no baseline samples — threshold=0 (no FP filter)")

    # 2. Filter slowed by IPD >= threshold; baseline + near_meth pass through.
    new_data: dict = {}
    n_baseline_in = n_baseline_out = 0
    n_near_in = n_near_out = 0
    n_slowed_in = n_slowed_kept = n_slowed_dropped = 0
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_CATEGORY:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        base_m = cats == CATEGORY_BASELINE
        slow_m = cats == CATEGORY_SLOWED
        near_m = cats == CATEGORY_NEAR_METH
        n_baseline_in += int(base_m.sum())
        n_near_in += int(near_m.sum())
        n_slowed_in += int(slow_m.sum())
        slow_keep_mask = slow_m & (arr[:, COL_IPD] >= threshold)
        n_slowed_kept += int(slow_keep_mask.sum())
        n_slowed_dropped += int(slow_m.sum() - slow_keep_mask.sum())
        keep_rows = base_m | near_m | slow_keep_mask
        if keep_rows.any():
            new_data[int(kid)] = arr[keep_rows].copy()
            n_baseline_out += int(base_m.sum())
            n_near_out += int(near_m.sum())

    log.info(
        "[refine.p95] baseline:  %d in -> %d kept (pass-through)",
        n_baseline_in,
        n_baseline_out,
    )
    log.info(
        "[refine.p95] near_meth: %d in -> %d kept (pass-through)",
        n_near_in,
        n_near_out,
    )
    log.info(
        "[refine.p95] slowed:    %d in -> %d kept, %d dropped (IPD < %.2f)  survival = %.2f%%",
        n_slowed_in,
        n_slowed_kept,
        n_slowed_dropped,
        threshold,
        100.0 * n_slowed_kept / max(n_slowed_in, 1),
    )

    stats = {
        "method": "p95_per_kmer_baseline_mean",
        "secondary_percentile": secondary_pct,
        "threshold": threshold,
        "n_baseline_in": n_baseline_in,
        "n_baseline_out": n_baseline_out,
        "n_near_in": n_near_in,
        "n_near_out": n_near_out,
        "n_slowed_in": n_slowed_in,
        "n_slowed_kept": n_slowed_kept,
        "n_slowed_dropped": n_slowed_dropped,
    }
    return new_data, stats


# ---------------------------------------------------------------------------
# GMM method (default) — per-meth-type 2-component mixture
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    """Stable logit. Used to translate posterior threshold into a log-pdf-diff cut."""
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"logit() requires 0 < p < 1, got {p}")
    return float(np.log(p / (1.0 - p)))


def _gmm_log_pdf_diff(
    x: np.ndarray,
    mu_low: float,
    sig_low: float,
    w_low: float,
    mu_hi: float,
    sig_hi: float,
    w_hi: float,
) -> np.ndarray:
    """Vectorised ``log p_low(x) − log p_hi(x)`` for a 2-component univariate GMM.

    Constants (the ``-0.5 log(2π)`` from each component) cancel and are
    omitted. Posterior(low | x) = sigmoid(diff), so a posterior cut at
    ``θ`` is equivalent to ``diff > logit(θ)``.
    """
    log_p_low = np.log(w_low) - np.log(sig_low) - 0.5 * ((x - mu_low) / sig_low) ** 2
    log_p_hi = np.log(w_hi) - np.log(sig_hi) - 0.5 * ((x - mu_hi) / sig_hi) ** 2
    return log_p_low - log_p_hi


def slowed_split_gmm(
    data: dict,
    posterior_threshold: float = 0.5,
    baseline_validation_min: float = 0.85,
    min_samples_for_gmm: int = 100,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Per-meth-type 2-component GMM filter on ``CATEGORY_SLOWED`` rows.

    For each meth type T present in the slowed rows:

    1. Pool ``slowed_by_T`` IPDs across all kmers; subsample baseline IPDs
       to the same count (or all baseline if fewer).
    2. Fit ``GaussianMixture(K=2)`` on the combined IPD pool.
    3. Validate that ≥ ``baseline_validation_min`` fraction of the baseline
       subsample is assigned to the lower-mean component. If not, the fit
       cannot be trusted for T — keep all ``slowed_by_T`` rows, log warn.
    4. Otherwise, drop ``slowed_by_T`` rows whose posterior in the
       lower-mean component exceeds ``posterior_threshold``.

    ``CATEGORY_BASELINE`` and ``CATEGORY_NEAR_METH`` pass through unchanged
    regardless of method. The fits are independent across meth types and
    could be parallelised; for the typical ≤ 4 types this is run
    sequentially (overhead < 5 s per type for ~1M slowed rows).

    Returns ``(new_data, stats)`` with the per-type GMM parameters
    (means, sigmas, weights) recorded in ``stats["per_type"]`` so the
    analyze report can overlay the fit on the per-type IPD distribution.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError(
            "sklearn is required for the GMM refine method. Install with: pip install scikit-learn"
        ) from exc

    from .utils.encoding import get_meth_ids
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PARENT_METH,
    )

    rng = np.random.default_rng(seed)
    name_by_mid = {v: k for k, v in get_meth_ids().items()}
    drop_logit = _logit(posterior_threshold)

    # ── Pass 1: harvest IPDs per (cat, parent_meth) — float32 to halve mem ──
    log.info("[refine.gmm] pass 1/2: harvesting IPDs ...")
    baseline_chunks: list = []
    slowed_chunks_by_T: dict[int, list] = {}

    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_PARENT_METH:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        parent = arr[:, COL_PARENT_METH].astype(np.int8)
        ipd = arr[:, COL_IPD]

        base_m = cats == CATEGORY_BASELINE
        if base_m.any():
            baseline_chunks.append(ipd[base_m].astype(np.float32))

        slow_m = cats == CATEGORY_SLOWED
        if slow_m.any():
            for T_id in np.unique(parent[slow_m]):
                T_id_int = int(T_id)
                if T_id_int == 0:
                    continue
                mask = slow_m & (parent == T_id)
                slowed_chunks_by_T.setdefault(T_id_int, []).append(ipd[mask].astype(np.float32))

    if not baseline_chunks:
        log.warning("[refine.gmm] no baseline samples — keeping all slowed (no filter)")
        return _passthrough(data, method="gmm_no_baseline")

    baseline_pool = np.concatenate(baseline_chunks)
    log.info(
        "[refine.gmm] baseline pool: n=%d  mean=%.2f  std=%.2f",
        len(baseline_pool),
        float(baseline_pool.mean()),
        float(baseline_pool.std()),
    )

    # ── Per-type GMM fits ──
    fit_params_by_T: dict[int, dict] = {}
    per_type_stats: dict = {}

    for T_id in sorted(slowed_chunks_by_T.keys()):
        T_name = name_by_mid.get(T_id, f"meth{T_id}")
        slowed_T = np.concatenate(slowed_chunks_by_T[T_id])
        n_T = len(slowed_T)

        if n_T < min_samples_for_gmm:
            log.warning(
                "[refine.gmm] %s: only %d slowed samples (< %d) — keeping all (no fit attempted)",
                T_name,
                n_T,
                min_samples_for_gmm,
            )
            per_type_stats[T_name] = {
                "n_in": n_T,
                "n_kept": n_T,
                "n_dropped": 0,
                "skipped": True,
                "reason": "too_few_samples",
            }
            continue

        # Equal-count baseline subsample (or all baseline if fewer).
        n_match = min(n_T, len(baseline_pool))
        if n_match == len(baseline_pool):
            base_sample = baseline_pool
        else:
            idx = rng.choice(len(baseline_pool), n_match, replace=False)
            base_sample = baseline_pool[idx]

        # Fit GMM(K=2) on the combined pool — single column, ~2N points.
        pool = np.concatenate([base_sample, slowed_T]).reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=2, random_state=seed, n_init=3, covariance_type="full"
        ).fit(pool)

        means = gmm.means_.ravel()
        sigmas = np.sqrt(gmm.covariances_.ravel())
        weights = gmm.weights_.ravel()
        lower_idx = int(np.argmin(means))
        higher_idx = 1 - lower_idx

        # Validation: predict on the baseline subsample, demand most of it
        # lands in the lower-mean component. If not, the fit is suspect
        # (baseline is split across components, or weights are flipped) —
        # don't filter for this type.
        base_assignments = gmm.predict(base_sample.reshape(-1, 1))
        pct_in_lower = float((base_assignments == lower_idx).mean())

        if pct_in_lower < baseline_validation_min:
            log.warning(
                "[refine.gmm] %s: VALIDATION FAILED — only %.1f%% of baseline in "
                "lower comp (need ≥ %.0f%%). GMM means=[%.2f, %.2f] "
                "sigmas=[%.2f, %.2f] weights=[%.2f, %.2f]. Keeping all %d slowed.",
                T_name,
                pct_in_lower * 100,
                baseline_validation_min * 100,
                means[lower_idx],
                means[higher_idx],
                sigmas[lower_idx],
                sigmas[higher_idx],
                weights[lower_idx],
                weights[higher_idx],
                n_T,
            )
            per_type_stats[T_name] = {
                "n_in": n_T,
                "n_kept": n_T,
                "n_dropped": 0,
                "skipped": True,
                "reason": "baseline_validation_failed",
                "gmm_means": means.tolist(),
                "gmm_sigmas": sigmas.tolist(),
                "gmm_weights": weights.tolist(),
                "baseline_in_lower_pct": pct_in_lower,
            }
            continue

        # Apply the cut: posterior_low > θ ⇔ log_pdf_diff > logit(θ).
        log_diff = _gmm_log_pdf_diff(
            slowed_T,
            means[lower_idx],
            sigmas[lower_idx],
            weights[lower_idx],
            means[higher_idx],
            sigmas[higher_idx],
            weights[higher_idx],
        )
        drop_count = int((log_diff > drop_logit).sum())
        n_kept = n_T - drop_count

        log.info(
            "[refine.gmm] %s: GMM low=N(%.2f, %.2f²)·%.3f  hi=N(%.2f, %.2f²)·%.3f  "
            "baseline_in_lower=%.1f%%. Kept %d/%d (%.1f%%), dropped %d.",
            T_name,
            means[lower_idx],
            sigmas[lower_idx],
            weights[lower_idx],
            means[higher_idx],
            sigmas[higher_idx],
            weights[higher_idx],
            pct_in_lower * 100,
            n_kept,
            n_T,
            100.0 * n_kept / n_T,
            drop_count,
        )

        fit_params_by_T[T_id] = {
            "mu_low": float(means[lower_idx]),
            "sig_low": float(sigmas[lower_idx]),
            "w_low": float(weights[lower_idx]),
            "mu_hi": float(means[higher_idx]),
            "sig_hi": float(sigmas[higher_idx]),
            "w_hi": float(weights[higher_idx]),
        }
        per_type_stats[T_name] = {
            "n_in": n_T,
            "n_kept": n_kept,
            "n_dropped": drop_count,
            "skipped": False,
            "gmm_means": means.tolist(),
            "gmm_sigmas": sigmas.tolist(),
            "gmm_weights": weights.tolist(),
            "baseline_in_lower_pct": pct_in_lower,
            "lower_idx": lower_idx,
        }

    # ── Pass 2: rebuild the data dict, applying per-type filter to slowed ──
    log.info("[refine.gmm] pass 2/2: filtering slowed rows ...")
    new_data: dict = {}
    n_baseline_in = n_baseline_out = 0
    n_near_in = n_near_out = 0
    n_slowed_in = n_slowed_kept = n_slowed_dropped = 0

    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_PARENT_METH:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        parent = arr[:, COL_PARENT_METH].astype(np.int8)
        ipd = arr[:, COL_IPD]

        base_m = cats == CATEGORY_BASELINE
        near_m = cats == CATEGORY_NEAR_METH
        slow_m = cats == CATEGORY_SLOWED
        n_baseline_in += int(base_m.sum())
        n_near_in += int(near_m.sum())
        n_slowed_in += int(slow_m.sum())

        # Per-type slowed keep mask — vectorised log-pdf-diff via stored params.
        slow_keep = np.zeros_like(slow_m)
        if slow_m.any():
            for T_id in np.unique(parent[slow_m]):
                T_id_int = int(T_id)
                if T_id_int == 0:
                    continue
                mask_T = slow_m & (parent == T_id)
                if not mask_T.any():
                    continue
                params = fit_params_by_T.get(T_id_int)
                if params is None:
                    # Type was skipped (validation failed or too few) —
                    # keep all slowed of this type.
                    slow_keep |= mask_T
                    continue
                ipd_T = ipd[mask_T]
                log_diff = _gmm_log_pdf_diff(
                    ipd_T,
                    params["mu_low"],
                    params["sig_low"],
                    params["w_low"],
                    params["mu_hi"],
                    params["sig_hi"],
                    params["w_hi"],
                )
                keep_local = log_diff <= drop_logit
                # Place keep_local back into the full-row mask.
                full_idx = np.where(mask_T)[0]
                tmp = np.zeros_like(mask_T)
                tmp[full_idx[keep_local]] = True
                slow_keep |= tmp

        n_slowed_kept += int(slow_keep.sum())
        n_slowed_dropped += int(slow_m.sum() - slow_keep.sum())

        keep_rows = base_m | near_m | slow_keep
        if keep_rows.any():
            new_data[int(kid)] = arr[keep_rows].copy()
            n_baseline_out += int(base_m.sum())
            n_near_out += int(near_m.sum())

    log.info(
        "[refine.gmm] baseline:  %d in -> %d kept (pass-through)",
        n_baseline_in,
        n_baseline_out,
    )
    log.info(
        "[refine.gmm] near_meth: %d in -> %d kept (pass-through)",
        n_near_in,
        n_near_out,
    )
    log.info(
        "[refine.gmm] slowed:    %d in -> %d kept, %d dropped  overall survival = %.2f%%",
        n_slowed_in,
        n_slowed_kept,
        n_slowed_dropped,
        100.0 * n_slowed_kept / max(n_slowed_in, 1),
    )

    stats = {
        "method": "gmm_per_meth_type",
        "posterior_threshold": posterior_threshold,
        "baseline_validation_min": baseline_validation_min,
        "min_samples_for_gmm": min_samples_for_gmm,
        "seed": seed,
        "n_baseline_pool": len(baseline_pool),
        "n_baseline_in": n_baseline_in,
        "n_baseline_out": n_baseline_out,
        "n_near_in": n_near_in,
        "n_near_out": n_near_out,
        "n_slowed_in": n_slowed_in,
        "n_slowed_kept": n_slowed_kept,
        "n_slowed_dropped": n_slowed_dropped,
        "per_type": per_type_stats,
    }
    return new_data, stats


def _passthrough(data: dict, method: str) -> tuple[dict, dict]:
    """Return ``data`` unchanged (deep-copied at row level) with a stats stub.

    Used when GMM cannot be fit at all (e.g. no baseline samples). The
    pkl is still rewritten so downstream tools find a valid file.
    """
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
    )

    new_data: dict = {}
    n_b = n_n = n_s = 0
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_CATEGORY:
            continue
        new_data[int(kid)] = arr.copy()
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        n_b += int((cats == CATEGORY_BASELINE).sum())
        n_n += int((cats == CATEGORY_NEAR_METH).sum())
        n_s += int((cats == CATEGORY_SLOWED).sum())

    stats = {
        "method": method,
        "skipped": True,
        "n_baseline_in": n_b,
        "n_baseline_out": n_b,
        "n_near_in": n_n,
        "n_near_out": n_n,
        "n_slowed_in": n_s,
        "n_slowed_kept": n_s,
        "n_slowed_dropped": 0,
    }
    return new_data, stats


# ---------------------------------------------------------------------------
# Orchestrator + CLI
# ---------------------------------------------------------------------------


def refine_pkl(
    in_path: Path,
    out_path: Path,
    method: str = "gmm",
    secondary_percentile: float | None = None,
    posterior_threshold: float = 0.5,
    baseline_validation_min: float = 0.85,
    min_samples_for_gmm: int = 100,
    seed: int = 42,
) -> dict:
    """Load a master .pkl, run the chosen refine method, write the refined pkl.

    Args:
        method: ``"gmm"`` (default, per-meth-type 2-comp Gaussian Mixture)
            or ``"p95"`` (legacy global per-kmer baseline-mean p95).
        secondary_percentile: percentile for the ``p95`` method. Defaults
            to ``refine.slowed_split.secondary_percentile`` in
            ``kinsim_config.yaml`` (typically 95).
        posterior_threshold / baseline_validation_min /
            min_samples_for_gmm: knobs for the ``gmm`` method.
    """
    log.info("Loading: %s  (%.2f GB)", in_path, in_path.stat().st_size / 1e9)
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    orig_meta = data.pop("__meta__", None)

    int_keyed = {k: v for k, v in data.items() if isinstance(k, (int, np.integer))}
    if not int_keyed:
        log.error("No int-keyed data found — input is not a kinsim extract pkl.")
        sys.exit(1)

    # Fail fast on the old layout (no PARENT_METH at col 36). Re-extract
    # is the right answer; both refine methods need the parent column for
    # per-meth-type behaviour (gmm) or for analyze diagnostics (p95).
    from .utils.sample_layout import SAMPLE_NCOLS

    sample_arr = next(iter(int_keyed.values()))
    if sample_arr.shape[1] < SAMPLE_NCOLS:
        log.error(
            "Input pkl uses an obsolete %d-col layout; current layout is %d cols "
            "(adds PARENT_METH at col 36, PARENT_OFFSET at col 37). "
            "Re-run `kinsim extract` to regenerate.",
            sample_arr.shape[1],
            SAMPLE_NCOLS,
        )
        sys.exit(1)

    if method == "gmm":
        log.info(
            "Refine: method=gmm  posterior_threshold=%g  baseline_validation_min=%g  "
            "min_samples_for_gmm=%d",
            posterior_threshold,
            baseline_validation_min,
            min_samples_for_gmm,
        )
        new_data, stats = slowed_split_gmm(
            int_keyed,
            posterior_threshold=posterior_threshold,
            baseline_validation_min=baseline_validation_min,
            min_samples_for_gmm=min_samples_for_gmm,
            seed=seed,
        )
    elif method == "p95":
        if secondary_percentile is None:
            from .utils.config import load_kinsim_config

            cfg = load_kinsim_config()
            secondary_percentile = float(
                ((cfg.get("refine") or {}).get("slowed_split") or {}).get(
                    "secondary_percentile", 95.0
                )
            )
        log.info("Refine: method=p95  secondary_percentile=%g", secondary_percentile)
        new_data, stats = slowed_split(int_keyed, secondary_percentile)
    else:
        raise ValueError(f"Unknown refine method: {method!r} (use 'gmm' or 'p95')")

    new_data["__meta__"] = {
        "refined_from": str(in_path),
        "method": stats["method"],
        "stats": stats,
        "original_meta": orig_meta,
    }
    log.info("Writing: %s", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return stats


def main(argv=None):
    from .utils.config import setup_logging

    ap = argparse.ArgumentParser(
        prog="kinsim refine",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_pkl", help="Input master .pkl from `kinsim merge`")
    ap.add_argument("output_pkl", help="Output refined .pkl")
    ap.add_argument(
        "--method",
        choices=("gmm", "p95"),
        default="gmm",
        help="Filter method (default: gmm — per-meth-type 2-component "
        "Gaussian Mixture; p95 — legacy global per-kmer-mean p95).",
    )
    # GMM knobs
    ap.add_argument(
        "--posterior-threshold",
        type=float,
        default=0.5,
        help="GMM only. Drop slowed_by_T rows whose posterior in the "
        "lower-mean component exceeds this (default: 0.5).",
    )
    ap.add_argument(
        "--baseline-validation-min",
        type=float,
        default=0.85,
        help="GMM only. Require this fraction of the baseline subsample "
        "to land in the lower-mean component before applying the cut "
        "(default: 0.85).",
    )
    ap.add_argument(
        "--min-samples-for-gmm",
        type=int,
        default=100,
        help="GMM only. Skip the fit (keep all slowed) when a meth type "
        "has fewer slowed rows than this (default: 100).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for GMM init and baseline subsampling.",
    )
    # p95 knob (legacy)
    ap.add_argument(
        "--secondary-percentile",
        type=float,
        default=None,
        help="p95 method only. Percentile of per-kmer baseline mean used "
        "as the lower IPD threshold. Overrides "
        "kinsim_config.yaml refine.slowed_split.secondary_percentile.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(verbose=args.verbose)

    in_p = Path(args.input_pkl)
    out_p = Path(args.output_pkl)
    if not in_p.exists():
        print(f"ERROR: {in_p} not found", file=sys.stderr)
        sys.exit(1)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    refine_pkl(
        in_p,
        out_p,
        method=args.method,
        secondary_percentile=args.secondary_percentile,
        posterior_threshold=args.posterior_threshold,
        baseline_validation_min=args.baseline_validation_min,
        min_samples_for_gmm=args.min_samples_for_gmm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
