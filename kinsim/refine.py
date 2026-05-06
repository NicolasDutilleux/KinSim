"""Refine a KinSim shard or directory by filtering motif false-positive ``slowed`` samples.

Input format: ``dict[kmer_id (int)] -> ndarray(N, 20)`` produced by
``kinsim extract``. Cols 17/18/19 carry CATEGORY, PARENT_METH and
PARENT_OFFSET (see ``kinsim.utils.sample_layout``).

Default method — ``--method gmm`` (baseline-anchored)
-----------------------------------------------------
Single rule. For each (meth_type T, parent_offset off) bucket:

  1. Fit a 2-component Gaussian mixture on (IPD, PW) of the joint
     baseline_subsample + slowed_by_T@+off pool, with **component 0
     initialised at the global baseline pool's (mean, cov)** so EM
     keeps it pinned at baseline kinetics. The free component fits
     the meth signal.
  2. Default ``--n-components 2,3``: BIC picks between K=2 (anchor +
     one meth lobe — the simple case) and K=3 (anchor + two free
     components, e.g. when the meth signal is bimodal). **Strict
     biological veto on K>2**: a K>2 fit is rejected if EM places
     any non-anchor component at or below the anchor's IPD.
     Methylation never produces sub-baseline kinetics, so a
     low-IPD free component is by construction overfitting noise
     and gets thrown out — no matter how good its BIC looks. With
     huge N, BIC alone is too lenient; the veto enforces the
     biological prior. Pass ``--n-components 2`` to disable K=3
     entirely.
  3. Keep only slowed rows whose argmax-posterior component has
     mean IPD **strictly above the global baseline pool mean
     (``base_mu``)**. Drop everything else — including the
     initialised anchor if it stayed near baseline AND any free
     component that landed at or below baseline (motif-FPs or
     unmethylated reads). The reference is ``base_mu``, not the
     post-EM mean of component 0; the "anchor" is only an init,
     sklearn lets all components drift during EM, and ``base_mu``
     is the invariant biological reference.

     With K=3 this typically drops two components (a low-IPD
     motif-FP cluster + the ex-anchor sitting near baseline) and
     keeps the high-IPD meth lobe; if the meth signal is bimodal
     (e.g. partial + full occupancy) BOTH above-baseline
     components survive.

No validation gate. The baseline anchor guarantees component 0 is
identifiable, so the drop rule is always defined. For weak-signal
types (m4C, m5C) where a free GMM might collapse to a single Gaussian
and miss the methylation entirely, the anchor forces the unconstrained
components to capture whatever variance differs from baseline.

Why raw IPD/PW (not log)? PacBio uint8 [0, 255] encoding gives roughly
Gaussian per-kmer distributions by design (that's what ipdSummary
assumes). The pooled aggregate across kmers is a mixture either way,
and raw space gives cleaner cluster separation than log: the meth
component (raw IPD ~90 for m6A) sits well above baseline (~26),
whereas log1p compresses them to log 4.5 vs 3.3 and harder-to-separate
intermediate components appear.

Legacy method — ``--method p95``
--------------------------------
Single global threshold = ``secondary_percentile`` of the per-kmer
baseline-mean distribution. Drops ``slowed`` rows below the threshold.
Kept for comparison; not recommended.

Usage::

    kinsim refine in.pkl out.pkl                  # default anchored GMM
    kinsim refine shards/ refined/                # sharded directory mode
    kinsim refine in.pkl out.pkl --method p95     # legacy fallback
    kinsim refine in.pkl out.pkl --n-components 3 # force K=3
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


def _gmm_posterior(
    X: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Vectorised posterior P(component k | x) for a multivariate GMM.

    Args:
        X:           (n, D) sample matrix.
        means:       (K, D) per-component means.
        covariances: (K, D, D) per-component covariance matrices ("full").
        weights:     (K,) per-component mixing weights.

    Returns:
        (n, K) posterior probabilities. Stable via the standard
        log-sum-exp trick (subtract per-row max before exponentiating).

    Drops the constant ``-½D log(2π)`` term that is identical across
    components and cancels in the softmax.
    """
    X = np.asarray(X)
    means = np.asarray(means)
    covariances = np.asarray(covariances)
    weights = np.asarray(weights)
    K = means.shape[0]
    log_p = np.empty((X.shape[0], K), dtype=np.float64)
    for k in range(K):
        cov_inv = np.linalg.inv(covariances[k])
        _sign, logdet = np.linalg.slogdet(covariances[k])
        delta = X - means[k][None, :]  # (n, D)
        # Mahalanobis squared distance, vectorised: (delta @ Σ⁻¹) ⊙ delta sum on axis=1
        mahal2 = np.einsum("ni,ij,nj->n", delta, cov_inv, delta)
        log_p[:, k] = np.log(weights[k]) - 0.5 * logdet - 0.5 * mahal2
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    return p / p.sum(axis=1, keepdims=True)


def _harvest_into(
    data: dict,
    baseline_chunks: list,
    slowed_chunks_by_TO: dict,
    slowed_frac_by_TO: dict,
) -> None:
    """Append (IPD, PW) and frac chunks from ``data`` into the containers.

    Mutates:
      ``baseline_chunks``      list of (n, 2) arrays of (IPD, PW)
      ``slowed_chunks_by_TO``  dict[(T_id, offset)] → list of (n, 2) (IPD, PW) chunks
      ``slowed_frac_by_TO``    dict[(T_id, offset)] → list of (n,) frac chunks

    The frac chunks let downstream code average per-bucket motif occupancy
    so generate.py can decompose ``p_fire = mean_occupancy × p_efficiency``
    and apply per-site target occupancy at inference time.
    """
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_FRACTION,
        COL_IPD,
        COL_PARENT_METH,
        COL_PARENT_OFFSET,
        COL_PW,
    )

    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_PARENT_OFFSET:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        parent = arr[:, COL_PARENT_METH].astype(np.int8)
        offset = arr[:, COL_PARENT_OFFSET].astype(np.int8)
        ipd_pw = arr[:, [COL_IPD, COL_PW]]
        frac = arr[:, COL_FRACTION]

        base_m = cats == CATEGORY_BASELINE
        if base_m.any():
            baseline_chunks.append(ipd_pw[base_m].astype(np.float32))

        slow_m = cats == CATEGORY_SLOWED
        if slow_m.any():
            for T_id in np.unique(parent[slow_m]):
                T_id_int = int(T_id)
                if T_id_int == 0:
                    continue
                m_T = slow_m & (parent == T_id)
                if not m_T.any():
                    continue
                for off in np.unique(offset[m_T]):
                    O_int = int(off)
                    mask = m_T & (offset == O_int)
                    slowed_chunks_by_TO.setdefault((T_id_int, O_int), []).append(
                        ipd_pw[mask].astype(np.float32)
                    )
                    slowed_frac_by_TO.setdefault((T_id_int, O_int), []).append(
                        frac[mask].astype(np.float32)
                    )


def _bucket_label(T_name: str, offset: int) -> str:
    """Canonical per-(meth, offset) bucket label, e.g. ``m6A@+0`` / ``m5C@+6``."""
    return f"{T_name}@{offset:+d}"


def _fit_gmms_per_bucket(
    baseline_pool: np.ndarray,
    slowed_chunks_by_TO: dict,
    slowed_frac_by_TO: dict,
    min_samples_for_gmm: int,
    candidate_ks: tuple,
    strict_bic_nats_per_sample: float,
    seed: int,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """Fit a baseline-anchored GMM per (meth, offset) bucket and return drop rule.

    Single rule:
      1. Component 0 is **initialised at baseline stats** (mean, cov of
         ``baseline_pool``) so EM keeps it pinned to baseline kinetics.
      2. The remaining K−1 components fit freely on
         (baseline_subsample + slowed_TO).
      3. Each slowed row is assigned (argmax posterior) to one component.
         Rows assigned to component 0 are dropped (they look baseline);
         the rest are kept.
      4. BIC picks K∈``candidate_ks`` (default {2, 3}).

    No validation gate: the baseline anchor guarantees component 0 is
    always interpretable as "baseline-like", so we always have a
    well-defined drop rule. For very weak signals (m4C, m5C@+6), this
    forces the unconstrained components to capture whatever variance
    differs from baseline — a free GMM might collapse to K=1 and miss
    them entirely.

    Returns ``(fit_params_by_TO, per_bucket_stats)``. Buckets with too
    few rows (< ``min_samples_for_gmm``) are skipped — all rows kept.
    """
    from sklearn.mixture import GaussianMixture

    from .utils.encoding import get_meth_ids

    name_by_mid = {v: k for k, v in get_meth_ids().items()}
    fit_params_by_TO: dict[tuple[int, int], dict] = {}
    per_bucket_stats: dict = {}

    # Fit in raw IPD/PW space. PacBio encodes uint8 [0, 255] so per-kmer
    # baseline distributions are roughly Gaussian (that's what ipdSummary
    # assumes for its null model). Pooled across kmers it's a mixture of
    # Gaussians; raw space gives cleaner cluster separation than log1p
    # for that mixture (meth components don't get squashed toward baseline).
    base_mu = baseline_pool.mean(axis=0).astype(np.float64)
    base_cov = (np.cov(baseline_pool.T) + 1e-3 * np.eye(2)).astype(np.float64)
    base_precision = np.linalg.inv(base_cov)
    log.info(
        "[refine.gmm] baseline anchor: IPD %.2f±%.2f  PW %.2f±%.2f",
        base_mu[0], np.sqrt(base_cov[0, 0]), base_mu[1], np.sqrt(base_cov[1, 1]),
    )

    bucket_keys = sorted(slowed_chunks_by_TO.keys(), key=lambda k: (name_by_mid.get(k[0], "z"), k[1]))
    for T_id, off in bucket_keys:
        T_name = name_by_mid.get(T_id, f"meth{T_id}")
        label = _bucket_label(T_name, off)
        slowed_TO = np.concatenate(slowed_chunks_by_TO[(T_id, off)])
        n_TO = len(slowed_TO)
        frac_chunks = slowed_frac_by_TO.get((T_id, off), [])
        mean_occ = float(np.concatenate(frac_chunks).mean()) if frac_chunks else 1.0

        if n_TO < min_samples_for_gmm:
            log.warning(
                "[refine.gmm] %s: only %d slowed samples (< %d) — keeping all (no fit)",
                label, n_TO, min_samples_for_gmm,
            )
            per_bucket_stats[label] = {
                "meth_type": T_name, "offset": off,
                "n_in": n_TO, "n_kept": n_TO, "n_dropped": 0,
                "p_fire": 1.0, "mean_occupancy": mean_occ,
                "skipped": True, "reason": "too_few_samples",
            }
            continue

        # Subsample baseline to roughly match slowed bucket size for the
        # joint fit (so the meth components don't get drowned by baseline mass).
        n_match = min(n_TO, len(baseline_pool))
        if n_match == len(baseline_pool):
            base_sample = baseline_pool
        else:
            idx = rng.choice(len(baseline_pool), n_match, replace=False)
            base_sample = baseline_pool[idx]
        slowed_arr = slowed_TO.astype(np.float64)
        pool = np.concatenate([base_sample, slowed_arr])

        # BIC over candidate Ks. For each K: component 0 is initialised
        # at baseline stats; components 1..K-1 are initialised at
        # high-IPD quantiles of the slowed pool so EM lands them in
        # the meth zone.
        #
        # Important: with N ~ millions, the BIC complexity penalty
        # (k·ln(N)) is small relative to typical ΔlogL, so K>2 wins
        # easily even when the extra component is fitting noise. We
        # add a hard biological veto on top of BIC: K>2 is rejected
        # if EM places ANY component at or below the global baseline
        # pool mean (base_mu). Methylation never produces
        # sub-baseline kinetics — a low-IPD component is
        # overfitting motif-kmer-context heterogeneity, not real
        # signal. Pass --n-components 2,3,4,5 to give EM more
        # components to work with; the veto still rejects fits with
        # any below-baseline mass.
        bic_per_k: dict[int, float] = {}
        vetoed_ks: dict[int, str] = {}
        # Cache surviving fits so the strict-K=2 selection below can
        # pick among them after seeing all BICs.
        fits_by_k: dict[int, "GaussianMixture"] = {}
        for k in candidate_ks:
            # Biology-aligned init: free components placed at fixed
            # multiples of ``base_mu`` (above baseline by construction).
            #   comp 0 (anchor):  base_mu × 1.0   = baseline
            #   comp 1 (free):    base_mu × 1.5   (m5C-ish IPD zone)
            #   comp 2 (free):    base_mu × 2.0   (m6A-ish IPD zone)
            #   comp 3 (free):    base_mu × 2.5   (very strong meth)
            #   ... etc, +0.5 × base_mu per added component.
            #
            # Why fixed multiples of base_mu and not slowed-pool
            # percentiles: percentiles can sit below baseline when the
            # slowed pool is trimodal (motif-FPs at low IPD pull the
            # 50th percentile down). That made the previous K=2 fit
            # converge with comp 1 below comp 0 — the failure mode
            # we just saw on bc2034. Multiples of base_mu guarantee
            # all free comps START above baseline; EM may still drift
            # them, but the basin of attraction is the meth zone.
            means_init = np.zeros((k, 2), dtype=np.float64)
            means_init[0] = base_mu
            for j in range(1, k):
                means_init[j] = base_mu * (1.0 + 0.5 * j)
            precisions_init = np.zeros((k, 2, 2), dtype=np.float64)
            precisions_init[0] = base_precision
            # Free components: large variance init so EM has room to converge.
            free_precision = np.linalg.inv(np.eye(2) * 100.0 + 1e-3 * np.eye(2))
            for j in range(1, k):
                precisions_init[j] = free_precision
            try:
                g = GaussianMixture(
                    n_components=k,
                    means_init=means_init,
                    precisions_init=precisions_init,
                    n_init=1,                  # single init = anchored
                    covariance_type="full",
                    max_iter=100,
                    random_state=seed,
                ).fit(pool)
            except (ValueError, np.linalg.LinAlgError) as exc:
                log.warning("[refine.gmm] %s: K=%d fit failed (%s) — skipping", label, k, exc)
                continue
            bic = float(g.bic(pool))
            bic_per_k[k] = bic

            # Biological veto on K>2 only. The reference is the
            # **global baseline pool mean** (``base_mu``), not the
            # post-EM anchor — EM can drift the anchor away from
            # baseline, so checking against where comp 0 ended up
            # would miss exactly the failure mode we want to catch.
            #
            # K=2 doesn't get a veto (it has no fallback K to fall to);
            # the downstream keep-rule still drops below-baseline
            # components from the K=2 fit. For K>2 we reject the
            # whole fit if any component lands at/below baseline —
            # the user's domain claim is that methylation never
            # produces sub-baseline kinetics, even cross-kmer.
            if k > 2:
                base_ipd = float(base_mu[0])
                # Strict below-baseline: a component sitting exactly at
                # baseline is the anchor doing its job; we only veto if a
                # component drifted strictly below baseline ("under" the
                # baseline distribution, per the user's prior).
                below = [
                    (j, float(g.means_[j, 0]))
                    for j in range(k)
                    if float(g.means_[j, 0]) < base_ipd - 1e-6
                ]
                if below:
                    parts = ", ".join(f"comp{j} IPD={mu:.1f}" for j, mu in below)
                    msg = (
                        f"component(s) STRICTLY below baseline IPD={base_ipd:.1f}: "
                        f"{parts}"
                    )
                    vetoed_ks[k] = msg
                    log.info(
                        "[refine.gmm] %s: K=%d VETOED (%s, BIC=%.0f) — "
                        "excluded from selection",
                        label, k, msg, bic,
                    )
                    continue

            fits_by_k[k] = g

        # Selection with strict K=2 preference. Standard BIC's complexity
        # penalty is k·ln(N) — peanuts when N ~ millions. To make K=2
        # the default unless K>2 is overwhelmingly better, we require
        # ΔBIC > strict_bic_nats_per_sample · N. Default 1.0 means K>2
        # has to improve average log-likelihood by ≥ 1 nat per sample
        # (i.e. "every sample is meaningfully better explained" on
        # average). On bc2034 m6A@+0, ΔBIC=1M, N=2.44M → 1M < 2.44M →
        # K=2 wins on strictness alone, before the veto runs.
        n_pool = len(pool)
        strictness_threshold = strict_bic_nats_per_sample * n_pool
        if not fits_by_k:
            best_gmm = None
        elif 2 in fits_by_k:
            best_k = 2
            best_gmm = fits_by_k[2]
            best_bic = bic_per_k[2]
            for k in sorted(fits_by_k.keys()):
                if k <= 2:
                    continue
                delta = best_bic - bic_per_k[k]   # positive means K beats K=2
                if delta > strictness_threshold:
                    log.info(
                        "[refine.gmm] %s: K=%d wins (ΔBIC=%.0f > %.0f = %.2f·N) — "
                        "switching from K=2", label, k, delta, strictness_threshold,
                        strict_bic_nats_per_sample,
                    )
                    best_k = k
                    best_gmm = fits_by_k[k]
                    best_bic = bic_per_k[k]
                else:
                    log.info(
                        "[refine.gmm] %s: K=%d not strict enough (ΔBIC=%.0f ≤ %.0f) — "
                        "keeping K=2", label, k, delta, strictness_threshold,
                    )
        else:
            # K=2 itself was unavailable (failed fit, etc.) — fall back to
            # whatever K survived. This branch is unusual.
            best_k = min(fits_by_k.keys())
            best_gmm = fits_by_k[best_k]
            best_bic = bic_per_k[best_k]
            log.info(
                "[refine.gmm] %s: K=2 unavailable — falling back to K=%d",
                label, best_k,
            )

        if best_gmm is None:
            log.warning("[refine.gmm] %s: all candidate Ks failed — keeping all", label)
            per_bucket_stats[label] = {
                "meth_type": T_name, "offset": off,
                "n_in": n_TO, "n_kept": n_TO, "n_dropped": 0,
                "p_fire": 1.0, "mean_occupancy": mean_occ,
                "skipped": True, "reason": "all_fits_failed",
            }
            continue

        gmm = best_gmm
        means = gmm.means_
        covariances = gmm.covariances_
        weights = gmm.weights_
        # Component 0 is the baseline anchor by construction; verify it
        # didn't drift far during EM (warn if it did, but keep going).
        anchor_drift_ipd = abs(float(means[0, 0]) - float(base_mu[0]))
        anchor_drift_pw = abs(float(means[0, 1]) - float(base_mu[1]))

        # Apply the rule: keep rows whose argmax-posterior component
        # has mean IPD strictly **above** the global baseline pool
        # mean (``base_mu``). Drop everything else — including the
        # initialised anchor if it stayed near baseline AND any free
        # component that landed at or below baseline (motif-FPs /
        # unmethylated reads masquerading as slowed).
        #
        # We compare against base_mu, NOT the post-EM mean of
        # component 0. The "anchor" is only an initialiser; sklearn
        # lets all components drift during EM. base_mu is the
        # invariant biological reference (where unmethylated kinetics
        # actually live across the whole genome).
        base_ipd = float(base_mu[0])
        keep_idxs = [
            k for k in range(len(means))
            if float(means[k, 0]) > base_ipd
        ]

        # Pretty-print all components in IPD-ascending order; tag the
        # baseline-anchored component (original index 0) and mark
        # which components survive the drop rule.
        sort_order = np.argsort(means[:, 0])
        comp_summary_parts = []
        for i in sort_order:
            mu_i, mu_p = float(means[i, 0]), float(means[i, 1])
            sig_i = float(np.sqrt(covariances[i, 0, 0]))
            sig_p = float(np.sqrt(covariances[i, 1, 1]))
            rho = float(covariances[i, 0, 1] / max(sig_i * sig_p, 1e-9))
            tags = []
            if i == 0:
                tags.append("ANCHOR")
            if i in keep_idxs:
                tags.append("KEEP")
            else:
                tags.append("DROP")
            comp_summary_parts.append(
                f"N(IPD {mu_i:.1f}±{sig_i:.1f}, PW {mu_p:.1f}±{sig_p:.1f}, "
                f"ρ={rho:+.2f})·{float(weights[i]):.3f} [{','.join(tags)}]"
            )
        comp_summary = "  ".join(comp_summary_parts)

        if not keep_idxs:
            log.warning(
                "[refine.gmm] %s: no component sits above baseline "
                "(IPD %.2f) — no detectable meth signal. Keeping all rows. "
                "Components: %s",
                label, base_ipd, comp_summary,
            )
            per_bucket_stats[label] = {
                "meth_type": T_name, "offset": off,
                "n_in": n_TO, "n_kept": n_TO, "n_dropped": 0,
                "p_fire": 1.0, "mean_occupancy": mean_occ,
                "skipped": True, "reason": "no_component_above_anchor",
                "n_components_used": best_k,
                "bic_per_k": bic_per_k,
                "gmm_means": means.tolist(),
            }
            continue
        post = _gmm_posterior(slowed_arr, means, covariances, weights)
        assigned = post.argmax(axis=1)
        keep_mask = np.isin(assigned, keep_idxs)
        n_kept = int(keep_mask.sum())
        drop_count = n_TO - n_kept

        log.info(
            "[refine.gmm] %s: K=%d  BIC=%.0f  drift(IPD,PW)=(%.1f, %.1f)  %s.  "
            "Kept %d/%d (%.1f%%), dropped %d.",
            label, best_k, best_bic, anchor_drift_ipd, anchor_drift_pw, comp_summary,
            n_kept, n_TO, 100.0 * n_kept / max(n_TO, 1), drop_count,
        )
        if anchor_drift_ipd > 5 * float(np.sqrt(base_cov[0, 0])):
            log.warning(
                "[refine.gmm] %s: baseline anchor drifted %.1f IPD units (>5σ_baseline) — "
                "fit may be unreliable for this bucket.", label, anchor_drift_ipd,
            )

        fit_params_by_TO[(T_id, off)] = {
            "means": means.tolist(),
            "covariances": covariances.tolist(),
            "weights": weights.tolist(),
            "baseline_idx": 0,
            "keep_idxs": list(keep_idxs),
        }
        per_bucket_stats[label] = {
            "meth_type": T_name, "offset": off,
            "n_in": n_TO, "n_kept": n_kept, "n_dropped": drop_count,
            "p_fire": n_kept / n_TO, "mean_occupancy": mean_occ,
            "skipped": False,
            "n_components_used": best_k,
            "n_components_candidates": list(candidate_ks),
            "bic_per_k": bic_per_k,
            "vetoed_ks": vetoed_ks,
            "gmm_means": means.tolist(),
            "gmm_covariances": covariances.tolist(),
            "gmm_weights": weights.tolist(),
            "anchor_drift_ipd": anchor_drift_ipd,
            "anchor_drift_pw": anchor_drift_pw,
        }

    return fit_params_by_TO, per_bucket_stats


def _apply_gmm_filter_to_data(
    data: dict,
    fit_params_by_TO: dict,
) -> tuple[dict, dict]:
    """Apply the per-(meth, offset) GMM drop rule to one in-memory data dict.

    Drops slowed rows whose argmax posterior is the baseline-anchored
    component (idx 0); keeps everything else. BASELINE and NEAR_METH
    rows pass through untouched. Buckets with no fit (too few samples
    or all candidate Ks failed) keep all rows.
    """
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PARENT_METH,
        COL_PARENT_OFFSET,
        COL_PW,
    )

    new_data: dict = {}
    n_baseline_in = n_baseline_out = 0
    n_near_in = n_near_out = 0
    n_slowed_in = n_slowed_kept = 0

    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_PARENT_OFFSET:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        parent = arr[:, COL_PARENT_METH].astype(np.int8)
        offset = arr[:, COL_PARENT_OFFSET].astype(np.int8)
        ipd_pw = arr[:, [COL_IPD, COL_PW]]

        base_m = cats == CATEGORY_BASELINE
        near_m = cats == CATEGORY_NEAR_METH
        slow_m = cats == CATEGORY_SLOWED
        n_baseline_in += int(base_m.sum())
        n_near_in += int(near_m.sum())
        n_slowed_in += int(slow_m.sum())

        slow_keep = np.zeros_like(slow_m)
        if slow_m.any():
            for T_id in np.unique(parent[slow_m]):
                T_id_int = int(T_id)
                if T_id_int == 0:
                    continue
                m_T = slow_m & (parent == T_id)
                if not m_T.any():
                    continue
                for off in np.unique(offset[m_T]):
                    O_int = int(off)
                    mask_TO = m_T & (offset == O_int)
                    if not mask_TO.any():
                        continue
                    params = fit_params_by_TO.get((T_id_int, O_int))
                    if params is None:
                        # Bucket was skipped (validation failed, too few, or
                        # not seen at fit time) — keep all rows in the bucket.
                        slow_keep |= mask_TO
                        continue
                    xs_TO = ipd_pw[mask_TO].astype(np.float64)
                    post = _gmm_posterior(
                        xs_TO,
                        np.asarray(params["means"]),
                        np.asarray(params["covariances"]),
                        np.asarray(params["weights"]),
                    )
                    # Keep only rows whose argmax-posterior component sits
                    # **above** the baseline anchor's mean IPD. Drops anchor
                    # rows + any below-anchor component (motif-FPs).
                    assigned = post.argmax(axis=1)
                    keep_idxs_TO = params.get("keep_idxs")
                    if keep_idxs_TO is None:
                        # Backwards-compat with older fit params.
                        keep_local = assigned != int(params["baseline_idx"])
                    else:
                        keep_local = np.isin(
                            assigned, np.asarray(keep_idxs_TO, dtype=int)
                        )
                    full_idx = np.where(mask_TO)[0]
                    tmp = np.zeros_like(mask_TO)
                    tmp[full_idx[keep_local]] = True
                    slow_keep |= tmp

        n_slowed_kept += int(slow_keep.sum())
        keep_rows = base_m | near_m | slow_keep
        if keep_rows.any():
            new_data[int(kid)] = arr[keep_rows].copy()
            n_baseline_out += int(base_m.sum())
            n_near_out += int(near_m.sum())

    return new_data, {
        "n_baseline_in": n_baseline_in,
        "n_baseline_out": n_baseline_out,
        "n_near_in": n_near_in,
        "n_near_out": n_near_out,
        "n_slowed_in": n_slowed_in,
        "n_slowed_kept": n_slowed_kept,
        "n_slowed_dropped": n_slowed_in - n_slowed_kept,
    }


def slowed_split_gmm(
    data: dict,
    min_samples_for_gmm: int = 100,
    n_components: int | tuple[int, ...] = (2, 3),
    strict_bic_nats_per_sample: float = 1.0,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Per-(meth, offset) 2D GMM filter with a baseline-anchored mixture.

    Single rule, per ``(meth_type T, parent_offset off)`` bucket:

    1. Pool joint ``(log1p(IPD), log1p(PW))`` of ``slowed_by_T@+off``
       across all kmers; subsample baseline likewise.
    2. Initialise component 0 at the baseline pool's ``(mean, cov)``
       so EM keeps it pinned at baseline kinetics. Initialise the
       remaining ``K-1`` components at high-IPD quantiles of the
       slowed pool. Fit on ``baseline_subsample + slowed`` jointly.
    3. Pick K∈``n_components`` (default ``(2, 3)``) by BIC, **with
       a biological veto on K>2**: any K>2 fit that places a
       non-anchor component at or below the anchor's IPD is
       rejected outright (overfit — methylation never sub-baseline).
    4. Drop rows whose ``argmax`` posterior is the anchor or any
       component below the anchor's IPD. Keep components strictly
       above (so a bimodal meth signal — partial + full occupancy
       — keeps both above-anchor lobes).

    No validation gate. The baseline anchor guarantees the drop rule
    is always defined; small drift is logged as a warning, not a
    failure. ``CATEGORY_BASELINE`` and ``CATEGORY_NEAR_METH`` rows pass
    through unchanged.

    Returns ``(new_data, stats)``; ``stats["per_bucket"]["<T>@+<off>"]``
    carries the BIC-selected K, the per-K BIC scores, the fitted GMM
    parameters (in log1p space), and the kept/dropped counts.
    """
    try:
        import sklearn.mixture  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "sklearn is required for the GMM refine method. Install with: pip install scikit-learn"
        ) from exc

    # Normalise n_components to a tuple of candidate Ks.
    if isinstance(n_components, int):
        candidate_ks: tuple[int, ...] = (n_components,)
    else:
        candidate_ks = tuple(int(k) for k in n_components)
    if not candidate_ks or any(k < 2 for k in candidate_ks):
        raise ValueError(f"n_components must be int ≥ 2 or tuple of ints ≥ 2, got {n_components}")

    rng = np.random.default_rng(seed)

    # ── Pass 1: harvest joint (IPD, PW) per (cat, parent_meth, offset) ──
    # 2D: methylation slows the polymerase on BOTH wait-time (IPD) and
    # incorporation duration (PW). Fitting jointly captures the (often
    # correlated) signature on both axes — strictly more discriminative
    # than 1D-IPD alone, especially for weak-signal types where one axis
    # dominates the other.
    log.info("[refine.gmm] pass 1/2: harvesting (IPD, PW) per (meth, offset) ...")
    baseline_chunks: list = []
    slowed_chunks_by_TO: dict[tuple[int, int], list] = {}
    slowed_frac_by_TO: dict[tuple[int, int], list] = {}
    _harvest_into(data, baseline_chunks, slowed_chunks_by_TO, slowed_frac_by_TO)

    if not baseline_chunks:
        log.warning("[refine.gmm] no baseline samples — keeping all slowed (no filter)")
        return _passthrough(data, method="gmm_no_baseline")

    baseline_pool = np.concatenate(baseline_chunks).astype(np.float64)  # (N, 2)
    log.info(
        "[refine.gmm] baseline pool: n=%d  IPD mean=%.2f std=%.2f  PW mean=%.2f std=%.2f",
        len(baseline_pool),
        float(baseline_pool[:, 0].mean()),
        float(baseline_pool[:, 0].std()),
        float(baseline_pool[:, 1].mean()),
        float(baseline_pool[:, 1].std()),
    )

    fit_params_by_TO, per_bucket_stats = _fit_gmms_per_bucket(
        baseline_pool,
        slowed_chunks_by_TO,
        slowed_frac_by_TO,
        min_samples_for_gmm=min_samples_for_gmm,
        candidate_ks=candidate_ks,
        strict_bic_nats_per_sample=strict_bic_nats_per_sample,
        seed=seed,
        rng=rng,
    )

    log.info("[refine.gmm] pass 2/2: filtering slowed rows ...")
    new_data, counts = _apply_gmm_filter_to_data(data, fit_params_by_TO)
    log.info(
        "[refine.gmm] baseline:  %d in -> %d kept (pass-through)",
        counts["n_baseline_in"],
        counts["n_baseline_out"],
    )
    log.info(
        "[refine.gmm] near_meth: %d in -> %d kept (pass-through)",
        counts["n_near_in"],
        counts["n_near_out"],
    )
    log.info(
        "[refine.gmm] slowed:    %d in -> %d kept, %d dropped  overall survival = %.2f%%",
        counts["n_slowed_in"],
        counts["n_slowed_kept"],
        counts["n_slowed_dropped"],
        100.0 * counts["n_slowed_kept"] / max(counts["n_slowed_in"], 1),
    )

    stats = {
        "method": "gmm_anchored_log1p",
        "min_samples_for_gmm": min_samples_for_gmm,
        "seed": seed,
        "n_baseline_pool": len(baseline_pool),
        **counts,
        "per_bucket": per_bucket_stats,
    }
    return new_data, stats


def slowed_split_gmm_shards(
    shards_dir: Path,
    output_dir: Path,
    min_samples_for_gmm: int = 100,
    n_components: int | tuple[int, ...] = (2, 3),
    strict_bic_nats_per_sample: float = 1.0,
    seed: int = 42,
    shard_glob: str = "*_shard.pkl",
) -> dict:
    """Sharded variant of :func:`slowed_split_gmm` — never holds the full corpus in RAM.

    Three-phase, bounded peak memory ≈ one shard's size:

    1. **Phase 1 (harvest pool)** — walk every shard in ``shards_dir``,
       load it, append its (IPD, PW) chunks into the global baseline /
       slowed-by-T pools, release the shard. Memory delta per shard is
       just the (IPD, PW) values, not the full pkl.
    2. **Phase 2 (fit GMMs globally)** — once, on the pooled data. Same
       per-meth-type 2D BIC fit + validation as the in-memory path.
    3. **Phase 3 (apply per-shard, write back)** — walk shards again,
       load each, score with the stored GMM params, atomically write
       ``<sample_id>_clean_shard.pkl`` into ``output_dir``. Aggregates
       per-shard counts into a global stats dict.

    Atomic writes guarantee any partial output from a crash is invisible:
    the next run resumes by re-reading whatever shards exist.

    Args mirror :func:`slowed_split_gmm` plus ``shard_glob`` for the pkl
    filename pattern. Returns the aggregate ``stats`` dict.
    """
    try:
        import sklearn.mixture  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "sklearn is required for the GMM refine method. Install with: pip install scikit-learn"
        ) from exc

    from .utils.io import atomic_write_pickle

    shards_dir = Path(shards_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(shards_dir.glob(shard_glob))
    if not shard_paths:
        log.error("No shards matching %s in %s", shard_glob, shards_dir)
        sys.exit(1)
    log.info("[refine.gmm.shards] %d shards under %s", len(shard_paths), shards_dir)

    if isinstance(n_components, int):
        candidate_ks: tuple[int, ...] = (n_components,)
    else:
        candidate_ks = tuple(int(k) for k in n_components)
    if not candidate_ks or any(k < 2 for k in candidate_ks):
        raise ValueError(f"n_components must be int ≥ 2 or tuple of ints ≥ 2, got {n_components}")
    rng = np.random.default_rng(seed)

    # ── Phase 1: harvest (IPD, PW) pool from every shard ──────────────
    log.info(
        "[refine.gmm.shards] phase 1/3: harvesting (IPD, PW) from %d shards ...", len(shard_paths)
    )
    baseline_chunks: list = []
    slowed_chunks_by_TO: dict[tuple[int, int], list] = {}
    slowed_frac_by_TO: dict[tuple[int, int], list] = {}
    for i, shard_path in enumerate(shard_paths, start=1):
        log.info("[refine.gmm.shards]   harvest %d/%d  %s", i, len(shard_paths), shard_path.name)
        with open(shard_path, "rb") as f:
            data = pickle.load(f)
        _harvest_into(data, baseline_chunks, slowed_chunks_by_TO, slowed_frac_by_TO)
        del data  # release the shard before loading the next

    if not baseline_chunks:
        log.warning("[refine.gmm.shards] no baseline samples — passing all shards through")
        # Copy each shard verbatim with passthrough meta.
        for shard_path in shard_paths:
            with open(shard_path, "rb") as f:
                data = pickle.load(f)
            data["__meta__"] = {
                "refined_from": str(shard_path),
                "method": "gmm_no_baseline",
                "stats": {"skipped": True},
                "original_meta": data.get("__meta__"),
            }
            out_path = output_dir / f"{shard_path.stem}_clean.pkl"
            atomic_write_pickle(data, out_path)
        return {"method": "gmm_no_baseline", "skipped": True, "n_shards": len(shard_paths)}

    # Cast to float64 — float32 sums of 56M values blow past float32
    # precision and the resulting mean is wrong by several IPD units
    # (the anchor logs would say IPD 26.4 when the real mean is 35.4).
    baseline_pool = np.concatenate(baseline_chunks).astype(np.float64)
    del baseline_chunks  # release the per-shard chunks now (~450 MB on bc2034)
    log.info(
        "[refine.gmm.shards] baseline pool: n=%d  IPD %.2f±%.2f  PW %.2f±%.2f",
        len(baseline_pool),
        float(baseline_pool[:, 0].mean()),
        float(baseline_pool[:, 0].std()),
        float(baseline_pool[:, 1].mean()),
        float(baseline_pool[:, 1].std()),
    )

    # ── Phase 2: fit GMMs globally (once, on the pooled data) ──────────
    log.info("[refine.gmm.shards] phase 2/3: fitting per-(meth, offset) GMMs ...")
    fit_params_by_TO, per_bucket_stats = _fit_gmms_per_bucket(
        baseline_pool,
        slowed_chunks_by_TO,
        slowed_frac_by_TO,
        min_samples_for_gmm=min_samples_for_gmm,
        candidate_ks=candidate_ks,
        strict_bic_nats_per_sample=strict_bic_nats_per_sample,
        seed=seed,
        rng=rng,
    )
    # Free the pool memory before phase 3.
    del slowed_chunks_by_TO, slowed_frac_by_TO, baseline_pool

    # ── Phase 3: apply per-shard, write atomically ─────────────────────
    log.info("[refine.gmm.shards] phase 3/3: filtering + writing %d shards ...", len(shard_paths))
    aggregate = {
        "n_baseline_in": 0,
        "n_baseline_out": 0,
        "n_near_in": 0,
        "n_near_out": 0,
        "n_slowed_in": 0,
        "n_slowed_kept": 0,
        "n_slowed_dropped": 0,
    }
    for i, shard_path in enumerate(shard_paths, start=1):
        log.info("[refine.gmm.shards]   apply %d/%d  %s", i, len(shard_paths), shard_path.name)
        with open(shard_path, "rb") as f:
            data = pickle.load(f)
        orig_meta = data.pop("__meta__", None)
        int_keyed = {k: v for k, v in data.items() if isinstance(k, (int, np.integer))}

        new_data, counts = _apply_gmm_filter_to_data(int_keyed, fit_params_by_TO)
        for k, v in counts.items():
            aggregate[k] += v

        new_data["__meta__"] = {
            "refined_from": str(shard_path),
            "method": "gmm_anchored_log1p",
            "stats": {**counts, "per_bucket": per_bucket_stats},
            "original_meta": orig_meta,
        }
        out_path = output_dir / f"{shard_path.stem}_clean.pkl"
        atomic_write_pickle(new_data, out_path)
        del data, new_data  # release before next shard

    log.info(
        "[refine.gmm.shards] DONE  baseline %d→%d  near %d→%d  "
        "slowed %d→%d kept (%d dropped, %.2f%% survival)",
        aggregate["n_baseline_in"],
        aggregate["n_baseline_out"],
        aggregate["n_near_in"],
        aggregate["n_near_out"],
        aggregate["n_slowed_in"],
        aggregate["n_slowed_kept"],
        aggregate["n_slowed_dropped"],
        100.0 * aggregate["n_slowed_kept"] / max(aggregate["n_slowed_in"], 1),
    )

    return {
        "method": "gmm_anchored_log1p",
        "min_samples_for_gmm": min_samples_for_gmm,
        "seed": seed,
        "n_shards": len(shard_paths),
        **aggregate,
        "per_bucket": per_bucket_stats,
    }


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
    min_samples_for_gmm: int = 100,
    n_components: int | tuple[int, ...] = (2, 3),
    strict_bic_nats_per_sample: float = 1.0,
    seed: int = 42,
) -> dict:
    """Refine one master .pkl OR a directory of shards (auto-detected).

    If ``in_path`` is a directory, the **sharded** refine path is used:
    pool harvest across shards → fit anchored GMM once globally → apply
    per-shard and write to ``out_path/`` (which must be a directory).
    Peak RAM is bounded by one shard, not the full corpus.

    If ``in_path`` is a file, the in-memory ``slowed_split_gmm`` /
    ``slowed_split`` path is used: load, fit, filter, atomic write.

    Args:
        method: ``"gmm"`` (default; baseline-anchored, log1p) or
            ``"p95"`` (legacy single global threshold).
        secondary_percentile: percentile for the ``p95`` method.
        min_samples_for_gmm / n_components / seed: knobs for ``gmm``.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    # ── Sharded mode: in_path is a directory of *_shard.pkl files. ──────
    if in_path.is_dir():
        if method != "gmm":
            log.error("Sharded refine only supports method=gmm (got %r)", method)
            sys.exit(1)
        from .utils.sample_layout import SAMPLE_NCOLS

        # Quick layout sanity check on the first shard.
        first = next(iter(sorted(in_path.glob("*_shard.pkl"))), None)
        if first is None:
            log.error("No *_shard.pkl files in %s", in_path)
            sys.exit(1)
        with open(first, "rb") as f:
            probe = pickle.load(f)
        for k, v in probe.items():
            if isinstance(k, (int, np.integer)) and isinstance(v, np.ndarray):
                if v.shape[1] < SAMPLE_NCOLS:
                    log.error(
                        "Shard %s uses an obsolete %d-col layout; "
                        "re-run `kinsim extract` (need %d cols).",
                        first.name,
                        v.shape[1],
                        SAMPLE_NCOLS,
                    )
                    sys.exit(1)
                break
        del probe

        log.info(
            "Refine (sharded, anchored): in=%s  out=%s  n_components=%s  "
            "strict_bic=%.2f nats/sample  min_samples_for_gmm=%d",
            in_path,
            out_path,
            n_components if isinstance(n_components, int) else ",".join(map(str, n_components)),
            strict_bic_nats_per_sample,
            min_samples_for_gmm,
        )
        return slowed_split_gmm_shards(
            in_path,
            out_path,
            min_samples_for_gmm=min_samples_for_gmm,
            n_components=n_components,
            strict_bic_nats_per_sample=strict_bic_nats_per_sample,
            seed=seed,
        )

    # ── Single-pkl mode (legacy / small datasets) ──
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
        n_comp_summary = (
            f"{n_components}"
            if isinstance(n_components, int)
            else "auto-BIC over " + ",".join(str(k) for k in n_components)
        )
        log.info(
            "Refine: method=gmm (anchored)  n_components=%s  "
            "strict_bic=%.2f nats/sample  min_samples_for_gmm=%d",
            n_comp_summary,
            strict_bic_nats_per_sample,
            min_samples_for_gmm,
        )
        new_data, stats = slowed_split_gmm(
            int_keyed,
            min_samples_for_gmm=min_samples_for_gmm,
            n_components=n_components,
            strict_bic_nats_per_sample=strict_bic_nats_per_sample,
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
    log.info("Writing (atomic): %s", out_path)
    from .utils.io import atomic_write_pickle

    atomic_write_pickle(new_data, out_path)
    return stats


def main(argv=None):
    from .utils.config import setup_logging

    ap = argparse.ArgumentParser(
        prog="kinsim refine",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_pkl", help="Shard .pkl from `kinsim extract`, OR a directory of *_shard.pkl (sharded mode)")
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
        "--min-samples-for-gmm",
        type=int,
        default=100,
        help="GMM only. Skip the fit (keep all slowed) when a (meth, offset) "
        "bucket has fewer rows than this (default: 100).",
    )
    ap.add_argument(
        "--n-components",
        type=str,
        default="2,3",
        help="GMM only. Either a single integer K (forced K) or a "
        "comma-separated list of candidate Ks for BIC-based selection. "
        "Default '2,3' — BIC picks between anchor+1-free (K=2, simple "
        "case) and anchor+2-free (K=3, when slowed pool is bimodal "
        "with motif-FP low-IPD mass + real-meth high-IPD mass). Pass "
        "'2' to force the simpler model. Combined with --strict-bic, "
        "K=2 is the default unless K>2 beats it by a large margin.",
    )
    ap.add_argument(
        "--strict-bic",
        type=float,
        default=1.0,
        help="GMM only. Strictness margin for selecting K>2 over K=2: "
        "K>2 wins only if its BIC beats K=2's by at least "
        "(strict_bic × N) nats, where N is the joint-pool size. "
        "Default 1.0 means 'K>2 must improve average per-sample "
        "log-likelihood by ≥1 nat' — strict, K=2 is sticky. "
        "Pass 0.0 for standard BIC (any improvement wins). "
        "Pass 2.0+ for even stricter K=2 preference.",
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

    # Parse --n-components: single int → forced K; "k1,k2,..." → BIC over candidates.
    raw = args.n_components.strip()
    parsed = tuple(int(p.strip()) for p in raw.split(",") if p.strip())
    if not parsed:
        print("ERROR: --n-components must contain at least one integer", file=sys.stderr)
        sys.exit(1)
    n_components_arg = parsed[0] if len(parsed) == 1 else parsed

    refine_pkl(
        in_p,
        out_p,
        method=args.method,
        secondary_percentile=args.secondary_percentile,
        min_samples_for_gmm=args.min_samples_for_gmm,
        n_components=n_components_arg,
        strict_bic_nats_per_sample=args.strict_bic,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
