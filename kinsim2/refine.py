"""Refine a KinSim bilateral shard by filtering motif false-positive ``slowed`` samples.

Input format: ``dict[kmer_id (int)] -> ndarray(N, 11+2*K)`` produced by
``kinsim extract`` (bilateral v2). Per-row fields include both strands'
kinetics, both strands' meth contexts, and PER-STRAND
``CATEGORY / PARENT_METH / PARENT_OFFSET`` (see :mod:`kinsim.utils.sample_layout`).

Bilateral refine: each strand is refined independently against its own
SLOWED pool. A row survives only if BOTH strands' SLOWED rows fall in
keep_idxs (or the strand is BASELINE/NEAR_METH which always pass).

Method — baseline-anchored GMM (only method; ``--method`` flag retired)
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

Usage::

    kinsim refine in.pkl out.pkl                  # default anchored GMM
    kinsim refine shards/ refined/                # sharded directory mode
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
# GMM method — per-meth-type 2-component mixture (default and only method)
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


def _maybe_cap_chunks(
    chunks: list,
    cap: int,
    paired_chunks: list | None = None,
    rng: np.random.Generator | None = None,
) -> bool:
    """Reservoir-cap ``chunks`` at ``cap`` rows in place.

    Concatenates + random-subsamples only when accumulated rows exceed
    ``1.5 × cap`` (avoids thrashing on small overshoots). If
    ``paired_chunks`` is provided (e.g. matching frac arrays), the same
    sample indices are applied so the pair stays row-aligned.

    Returns True if a cap was applied. ``cap <= 0`` means "no cap" and
    is a no-op.
    """
    if cap <= 0:
        return False
    total = sum(int(c.shape[0]) for c in chunks)
    if total <= int(cap * 1.5):
        return False
    pool = np.concatenate(chunks)
    if rng is None:
        rng = np.random.default_rng(42)
    idx = rng.choice(total, cap, replace=False)
    chunks[:] = [pool[idx].copy()]
    if paired_chunks is not None and paired_chunks:
        ppool = np.concatenate(paired_chunks)
        paired_chunks[:] = [ppool[idx].copy()]
    return True


def _harvest_into(
    data: dict,
    baseline_chunks: list,
    slowed_chunks_by_TO: dict,
    slowed_frac_by_TO: dict,
    *,
    layout,
    max_baseline_pool: int = 0,
    max_slowed_per_bucket: int = 0,
    rng: np.random.Generator | None = None,
) -> None:
    """Append (IPD, PW) and frac chunks from ``data`` into the containers.

    Bilateral v2: harvest from BOTH strands. A BASELINE on either strand
    contributes its strand's (IPD, PW) to the global baseline pool; a
    SLOWED on either strand contributes its strand's (IPD, PW) to that
    strand's (T, offset) slowed pool. The GMM fits a single per-(T, off)
    set of params used to filter both strands at apply time.
    """
    from .utils.sample_layout import CATEGORY_BASELINE, CATEGORY_SLOWED

    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] != layout.n_cols:
            continue
        cat_fwd = arr[:, layout.col_category_fwd].astype(np.int8)
        cat_rev = arr[:, layout.col_category_rev].astype(np.int8)
        pm_fwd = arr[:, layout.col_parent_meth_fwd].astype(np.int8)
        pm_rev = arr[:, layout.col_parent_meth_rev].astype(np.int8)
        po_fwd = arr[:, layout.col_parent_offset_fwd].astype(np.int8)
        po_rev = arr[:, layout.col_parent_offset_rev].astype(np.int8)
        ipd_pw_fwd = arr[:, [layout.col_ipd_fwd, layout.col_pw_fwd]]
        ipd_pw_rev = arr[:, [layout.col_ipd_rev, layout.col_pw_rev]]
        frac = arr[:, layout.col_fraction]

        for cat, ipd_pw, parent, offset in (
            (cat_fwd, ipd_pw_fwd, pm_fwd, po_fwd),
            (cat_rev, ipd_pw_rev, pm_rev, po_rev),
        ):
            base_m = cat == CATEGORY_BASELINE
            if base_m.any():
                baseline_chunks.append(ipd_pw[base_m].astype(np.float32))
            slow_m = cat == CATEGORY_SLOWED
            if not slow_m.any():
                continue
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

    _maybe_cap_chunks(baseline_chunks, max_baseline_pool, rng=rng)
    for key in list(slowed_chunks_by_TO.keys()):
        _maybe_cap_chunks(
            slowed_chunks_by_TO[key],
            max_slowed_per_bucket,
            paired_chunks=slowed_frac_by_TO.get(key),
            rng=rng,
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
    similarity_sigma_margin: float,
    seed: int,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    """Fit a baseline-anchored 1D-IPD GMM per (meth, offset) bucket.

    The fit and the drop rule operate on **IPD only**. PW means and
    standard deviations are reported per component for logging but do
    not influence the decision — empirical PW null is too noisy to
    serve as a separation axis.

    Per-bucket procedure:
      1. Component 0 is **initialised at the baseline IPD mean** so EM
         keeps it pinned to baseline kinetics.
      2. The remaining K−1 components fit freely on
         (baseline_subsample + slowed_TO), restricted to the IPD axis.
      3. K=1, K=2, K=3, ... compete by BIC. K=1 wins when the bucket
         has no separable structure (parsimony principle, plain BIC).
         K=2 is the default. K>2 must beat K=2 by a strict margin
         (``strict_bic_nats_per_sample × N``) AND have all components
         strictly above baseline.
      4. Each slowed row is assigned (argmax posterior) to one
         component. Components whose IPD mean is "similar to baseline"
         (within ``similarity_sigma_margin × σ_baseline_IPD``) are
         dropped. The rest are kept.

    Special-case K=1: refine cannot separate within a single Gaussian,
    so all rows are kept. Metadata flags whether the single component
    is above baseline (real but unimodal signal) or at/below baseline
    (null signal) so downstream consumers can choose to skip the
    offset.

    Returns ``(fit_params_by_TO, per_bucket_stats)``. Buckets with too
    few rows (< ``min_samples_for_gmm``) are skipped — all rows kept.
    """
    from sklearn.mixture import GaussianMixture

    from .utils.encoding import get_meth_ids

    name_by_mid = {v: k for k, v in get_meth_ids().items()}
    fit_params_by_TO: dict[tuple[int, int], dict] = {}
    per_bucket_stats: dict = {}

    # 1D fit on IPD. baseline_pool is (N, 2)=[IPD,PW]; we slice column 0
    # for the fit and keep PW stats for descriptive reporting only.
    base_mu_full = baseline_pool.mean(axis=0).astype(np.float64)  # (2,)  [IPD, PW]
    base_sigma_full = baseline_pool.std(axis=0).astype(np.float64)  # (2,)
    base_mu_ipd = float(base_mu_full[0])
    base_var_ipd = float(baseline_pool[:, 0].var()) + 1e-3
    base_sigma_ipd = float(np.sqrt(base_var_ipd))
    base_precision_ipd = 1.0 / base_var_ipd
    similarity_threshold = base_mu_ipd + similarity_sigma_margin * base_sigma_ipd
    log.info(
        "[refine.gmm] baseline anchor: IPD %.2f±%.2f  PW %.2f±%.2f  "
        "(similarity threshold = baseline_IPD + %.2f·σ = %.2f)",
        base_mu_full[0],
        base_sigma_full[0],
        base_mu_full[1],
        base_sigma_full[1],
        similarity_sigma_margin,
        similarity_threshold,
    )

    bucket_keys = sorted(
        slowed_chunks_by_TO.keys(), key=lambda k: (name_by_mid.get(k[0], "z"), k[1])
    )
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
                label,
                n_TO,
                min_samples_for_gmm,
            )
            per_bucket_stats[label] = {
                "meth_type": T_name,
                "offset": off,
                "n_in": n_TO,
                "n_kept": n_TO,
                "n_dropped": 0,
                "p_fire": 1.0,
                "mean_occupancy": mean_occ,
                "skipped": True,
                "reason": "too_few_samples",
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
        slowed_arr = slowed_TO.astype(np.float64)  # (n_TO, 2) for descriptive PW
        slowed_ipd = slowed_arr[:, 0:1]  # (n_TO, 1) for the fit
        pool_ipd = np.concatenate([base_sample[:, 0:1], slowed_ipd])  # (n_pool, 1)

        # BIC over candidate Ks. For each K: component 0 is initialised
        # at baseline IPD; components 1..K-1 are initialised at fixed
        # multiples of base_mu_ipd above baseline, so they start in
        # the meth zone.
        #
        # Important: with N ~ millions, the BIC complexity penalty
        # (k·ln(N)) is small relative to typical ΔlogL, so K>2 wins
        # easily even when the extra component is fitting noise.
        # Selection rule below is asymmetric:
        #   - K=1 vs K=2: plain BIC (parsimony — K=1 wins when there's
        #     no separable structure, K=2 over-fits noise).
        #   - K>2 vs K=2: must beat K=2 by ``strict_bic × N`` margin
        #     AND have no component below baseline IPD.
        bic_per_k: dict[int, float] = {}
        vetoed_ks: dict[int, str] = {}
        # Cache surviving fits so the strict selection below can
        # pick among them after seeing all BICs.
        fits_by_k: dict[int, GaussianMixture] = {}
        for k in candidate_ks:
            # Biology-aligned init on the IPD axis only.
            #   K=1: single component at baseline IPD (lets EM drift
            #        if there's signal; baseline is the obvious starting
            #        point for "no separable structure" data).
            #   K≥2: comp 0 at baseline_IPD; comps 1..K-1 at
            #        base_mu_ipd × (1.0 + 0.5·j), placing free
            #        components in the meth zone (1.5×, 2.0×, 2.5×, ...).
            means_init = np.zeros((k, 1), dtype=np.float64)
            means_init[0, 0] = base_mu_ipd
            for j in range(1, k):
                means_init[j, 0] = base_mu_ipd * (1.0 + 0.5 * j)
            precisions_init = np.zeros((k, 1, 1), dtype=np.float64)
            precisions_init[0, 0, 0] = base_precision_ipd
            # Free components: large variance init so EM has room to converge.
            free_precision = 1.0 / 100.0
            for j in range(1, k):
                precisions_init[j, 0, 0] = free_precision
            try:
                g = GaussianMixture(
                    n_components=k,
                    means_init=means_init,
                    precisions_init=precisions_init,
                    n_init=1,  # single init = anchored
                    covariance_type="full",
                    max_iter=100,
                    random_state=seed,
                ).fit(pool_ipd)
            except (ValueError, np.linalg.LinAlgError) as exc:
                log.warning("[refine.gmm] %s: K=%d fit failed (%s) — skipping", label, k, exc)
                continue
            bic = float(g.bic(pool_ipd))
            bic_per_k[k] = bic

            # Biological veto on K>2 only. The reference is the
            # **global baseline pool mean** (``base_mu_ipd``), not
            # the post-EM anchor — EM can drift the anchor away
            # from baseline, so checking against where comp 0
            # ended up would miss exactly the failure mode we
            # want to catch.
            #
            # K=1 and K=2 don't get a veto (K=1 has no contamination
            # cluster; K=2's drop rule already removes near-baseline
            # components). For K>2 we reject the whole fit if any
            # component lands at/below baseline — the domain claim
            # is that methylation never produces sub-baseline
            # kinetics, even cross-kmer.
            if k > 2:
                # Strict below-baseline: a component sitting exactly at
                # baseline is the anchor doing its job; we only veto if a
                # component drifted strictly below baseline.
                below = [
                    (j, float(g.means_[j, 0]))
                    for j in range(k)
                    if float(g.means_[j, 0]) < base_mu_ipd - 1e-6
                ]
                if below:
                    parts = ", ".join(f"comp{j} IPD={mu:.1f}" for j, mu in below)
                    msg = f"component(s) STRICTLY below baseline IPD={base_mu_ipd:.1f}: {parts}"
                    vetoed_ks[k] = msg
                    log.info(
                        "[refine.gmm] %s: K=%d VETOED (%s, BIC=%.0f) — excluded from selection",
                        label,
                        k,
                        msg,
                        bic,
                    )
                    continue

            fits_by_k[k] = g

        # Asymmetric selection:
        #   - K=1 wins on plain BIC if it beats K=2 (parsimony principle —
        #     K=2 over-fits noise when no separable structure exists).
        #   - K>2 must beat K=2 by ``strict_bic × N`` margin (sticky K=2).
        n_pool = len(pool_ipd)
        strictness_threshold = strict_bic_nats_per_sample * n_pool
        if not fits_by_k:
            best_gmm = None
            best_k = -1
            best_bic = float("inf")
        elif 2 in fits_by_k:
            best_k = 2
            best_gmm = fits_by_k[2]
            best_bic = bic_per_k[2]
            # K=1: lenient — wins if its BIC is lower (parsimony)
            if 1 in fits_by_k and bic_per_k[1] < best_bic:
                log.info(
                    "[refine.gmm] %s: K=1 wins on BIC (%.0f < K=2's %.0f) — "
                    "no separable structure, keeping all rows",
                    label,
                    bic_per_k[1],
                    best_bic,
                )
                best_k = 1
                best_gmm = fits_by_k[1]
                best_bic = bic_per_k[1]
            # K>2: strict — must beat current best by strict margin
            for k in sorted(fits_by_k.keys()):
                if k <= 2:
                    continue
                delta = best_bic - bic_per_k[k]
                if delta > strictness_threshold:
                    log.info(
                        "[refine.gmm] %s: K=%d wins (ΔBIC=%.0f > %.0f = %.2f·N) — "
                        "switching from K=%d",
                        label,
                        k,
                        delta,
                        strictness_threshold,
                        strict_bic_nats_per_sample,
                        best_k,
                    )
                    best_k = k
                    best_gmm = fits_by_k[k]
                    best_bic = bic_per_k[k]
                else:
                    log.info(
                        "[refine.gmm] %s: K=%d not strict enough (ΔBIC=%.0f ≤ %.0f) — keeping K=%d",
                        label,
                        k,
                        delta,
                        strictness_threshold,
                        best_k,
                    )
        else:
            # K=2 unavailable — fall back to lowest-BIC survivor.
            best_k = min(fits_by_k.keys(), key=lambda k: bic_per_k[k])
            best_gmm = fits_by_k[best_k]
            best_bic = bic_per_k[best_k]
            log.info(
                "[refine.gmm] %s: K=2 unavailable — falling back to K=%d (BIC=%.0f)",
                label,
                best_k,
                best_bic,
            )

        if best_gmm is None:
            log.warning("[refine.gmm] %s: all candidate Ks failed — keeping all", label)
            per_bucket_stats[label] = {
                "meth_type": T_name,
                "offset": off,
                "n_in": n_TO,
                "n_kept": n_TO,
                "n_dropped": 0,
                "p_fire": 1.0,
                "mean_occupancy": mean_occ,
                "skipped": True,
                "reason": "all_fits_failed",
            }
            continue

        gmm = best_gmm
        means_1d = gmm.means_  # (best_k, 1)
        covariances_1d = gmm.covariances_  # (best_k, 1, 1)
        weights = gmm.weights_  # (best_k,)
        # Component 0 is the baseline anchor by construction (only when K≥2);
        # verify it didn't drift far during EM (warn if it did).
        anchor_drift_ipd = abs(float(means_1d[0, 0]) - base_mu_ipd)

        # Compute argmax posterior for descriptive PW stats per component.
        post = _gmm_posterior(slowed_ipd, means_1d, covariances_1d, weights)
        assigned = post.argmax(axis=1)

        # Per-component descriptive PW stats (from the data assigned to
        # each component). Reported in the log but not used for the
        # drop decision — the fit is IPD-only.
        pw_per_comp = []
        for j in range(best_k):
            mask_j = assigned == j
            if mask_j.any():
                pw_mean = float(slowed_arr[mask_j, 1].mean())
                pw_std = float(slowed_arr[mask_j, 1].std())
            else:
                pw_mean = pw_std = 0.0
            pw_per_comp.append((pw_mean, pw_std))

        # Drop rule: drop components whose IPD mean is "similar to baseline"
        # (within ``similarity_sigma_margin × σ_baseline_IPD`` of base_mu_ipd).
        # Strict version (margin=0): drop only those at/below baseline.
        # Lenient version (margin=0.5): drop those within half a sigma of
        # baseline (treats them as essentially baseline-shaped).
        keep_idxs = [j for j in range(best_k) if float(means_1d[j, 0]) > similarity_threshold]

        # Pretty-print components, IPD-ascending. Tags: ANCHOR (j=0 for K≥2),
        # KEEP / DROP per the similarity rule, NULL for K=1 below baseline.
        sort_order = np.argsort(means_1d[:, 0])
        comp_summary_parts = []
        for i in sort_order:
            mu_i = float(means_1d[i, 0])
            sig_i = float(np.sqrt(covariances_1d[i, 0, 0]))
            mu_p, sig_p = pw_per_comp[i]
            tags = []
            if best_k >= 2 and i == 0:
                tags.append("ANCHOR")
            if i in keep_idxs:
                tags.append("KEEP")
            else:
                tags.append("DROP")
            comp_summary_parts.append(
                f"N(IPD {mu_i:.1f}±{sig_i:.1f}, PW {mu_p:.1f}±{sig_p:.1f})"
                f"·{float(weights[i]):.3f} [{','.join(tags)}]"
            )
        comp_summary = "  ".join(comp_summary_parts)

        # K=1 special case: refine cannot separate within a single Gaussian.
        # Keep all rows; flag null_signal if the component sits at/below
        # the similarity threshold (no detectable methylation signal).
        if best_k == 1:
            comp_mean = float(means_1d[0, 0])
            null_signal = comp_mean <= similarity_threshold
            log.info(
                "[refine.gmm] %s: K=1 selected — %s. %s. Kept all %d rows.",
                label,
                "null signal (mean ≤ similarity threshold)"
                if null_signal
                else "unimodal signal (no separable contamination cluster)",
                comp_summary,
                n_TO,
            )
            fit_params_by_TO[(T_id, off)] = {
                "means": means_1d.tolist(),
                "covariances": covariances_1d.tolist(),
                "weights": weights.tolist(),
                "baseline_idx": -1,
                "keep_idxs": [0],  # keep the single component → all rows
                "k1_null_signal": null_signal,
                "fit_dim": 1,
            }
            per_bucket_stats[label] = {
                "meth_type": T_name,
                "offset": off,
                "n_in": n_TO,
                "n_kept": n_TO,
                "n_dropped": 0,
                "p_fire": 1.0,
                "mean_occupancy": mean_occ,
                "skipped": False,
                "n_components_used": 1,
                "n_components_candidates": list(candidate_ks),
                "bic_per_k": bic_per_k,
                "vetoed_ks": vetoed_ks,
                "gmm_means": means_1d.tolist(),
                "gmm_covariances": covariances_1d.tolist(),
                "gmm_weights": weights.tolist(),
                "k1_null_signal": null_signal,
                "anchor_drift_ipd": 0.0,
                "anchor_drift_pw": 0.0,
            }
            continue

        if not keep_idxs:
            log.warning(
                "[refine.gmm] %s: no component sits above similarity threshold "
                "(IPD %.2f + %.2f·σ = %.2f) — no detectable meth signal. "
                "Keeping all rows. Components: %s",
                label,
                base_mu_ipd,
                similarity_sigma_margin,
                similarity_threshold,
                comp_summary,
            )
            per_bucket_stats[label] = {
                "meth_type": T_name,
                "offset": off,
                "n_in": n_TO,
                "n_kept": n_TO,
                "n_dropped": 0,
                "p_fire": 1.0,
                "mean_occupancy": mean_occ,
                "skipped": True,
                "reason": "no_component_above_threshold",
                "n_components_used": best_k,
                "bic_per_k": bic_per_k,
                "gmm_means": means_1d.tolist(),
            }
            continue

        keep_mask = np.isin(assigned, keep_idxs)
        n_kept = int(keep_mask.sum())
        drop_count = n_TO - n_kept

        log.info(
            "[refine.gmm] %s: K=%d  BIC=%.0f  drift(IPD)=%.1f  %s.  "
            "Kept %d/%d (%.1f%%), dropped %d.",
            label,
            best_k,
            best_bic,
            anchor_drift_ipd,
            comp_summary,
            n_kept,
            n_TO,
            100.0 * n_kept / max(n_TO, 1),
            drop_count,
        )
        if anchor_drift_ipd > 5 * base_sigma_ipd:
            log.warning(
                "[refine.gmm] %s: baseline anchor drifted %.1f IPD units (>5σ_baseline) — "
                "fit may be unreliable for this bucket.",
                label,
                anchor_drift_ipd,
            )

        fit_params_by_TO[(T_id, off)] = {
            "means": means_1d.tolist(),
            "covariances": covariances_1d.tolist(),
            "weights": weights.tolist(),
            "baseline_idx": 0,
            "keep_idxs": list(keep_idxs),
            "fit_dim": 1,
        }
        per_bucket_stats[label] = {
            "meth_type": T_name,
            "offset": off,
            "n_in": n_TO,
            "n_kept": n_kept,
            "n_dropped": drop_count,
            "p_fire": n_kept / n_TO,
            "mean_occupancy": mean_occ,
            "skipped": False,
            "n_components_used": best_k,
            "n_components_candidates": list(candidate_ks),
            "bic_per_k": bic_per_k,
            "vetoed_ks": vetoed_ks,
            "gmm_means": means_1d.tolist(),
            "gmm_covariances": covariances_1d.tolist(),
            "gmm_weights": weights.tolist(),
            "anchor_drift_ipd": anchor_drift_ipd,
            "anchor_drift_pw": 0.0,
        }

    # ── Sanity check: DROP component consistency across offsets ────────
    # The DROP component represents unmethylated reads at motif sites
    # (per-read partial methylation contamination). Different offsets of
    # the same meth type sample the SAME population of reads at the
    # SAME motif sites, so the DROP component's IPD mean should be
    # statistically indistinguishable across offsets. A large divergence
    # is diagnostic of either:
    #   (a) GMM mis-clustering — real-meth rows leaked into DROP at one
    #       offset but not another, or
    #   (b) motif-kmer-context bias — different offsets land on
    #       different kmer families, each with its own null kinetics.
    # Threshold: > 2σ_baseline_IPD between any pair of DROPs = warning.
    drop_means_by_T: dict[str, list[tuple[int, float]]] = {}
    for label, st in per_bucket_stats.items():
        if st.get("skipped"):
            continue
        if int(st.get("n_components_used", 0)) < 2:
            continue
        means = st.get("gmm_means")
        if not means:
            continue
        # Component 0 is the baseline anchor (initialised at base_mu_ipd).
        # The "DROP" components are those NOT in keep_idxs — typically
        # comp 0, but EM can drift it. Find DROP comps by exclusion.
        T_name = st["meth_type"]
        off = st["offset"]
        # Pick the DROP component with the largest weight (representative
        # of the unmethylated population in this bucket).
        weights = st.get("gmm_weights", [])
        params = fit_params_by_TO.get(
            (next((tid for tid, name in name_by_mid.items() if name == T_name), -1), off)
        )
        if params is None:
            continue
        keep_idxs_set = set(int(k) for k in params.get("keep_idxs", []))
        drop_idxs = [j for j in range(len(means)) if j not in keep_idxs_set]
        if not drop_idxs:
            continue
        # Largest-weight DROP component as the representative.
        rep_idx = max(drop_idxs, key=lambda j: float(weights[j]) if j < len(weights) else 0.0)
        rep_ipd = float(means[rep_idx][0])
        drop_means_by_T.setdefault(T_name, []).append((off, rep_ipd))

    warn_threshold = 2.0 * base_sigma_ipd
    for T_name, lst in drop_means_by_T.items():
        if len(lst) < 2:
            continue
        offs = [o for o, _ in lst]
        ipds = [m for _, m in lst]
        spread = max(ipds) - min(ipds)
        if spread > warn_threshold:
            pairs = ", ".join(f"@+{o}={m:.1f}" for o, m in sorted(lst))
            log.warning(
                "[refine.gmm] %s: DROP component IPD spreads %.1f units "
                "across offsets (> %.1f = 2·σ_baseline). Possible mis-clustering "
                "or motif-kmer-context bias. DROPs: %s",
                T_name,
                spread,
                warn_threshold,
                pairs,
            )

    return fit_params_by_TO, per_bucket_stats


def _strand_keep_mask(
    arr: np.ndarray,
    *,
    layout,
    col_category: int,
    col_parent_meth: int,
    col_parent_offset: int,
    col_ipd: int,
    fit_params_by_TO: dict,
) -> tuple[np.ndarray, int, int]:
    """For one strand, return (per-row keep mask, n_slow_in, n_slow_kept).

    BASELINE / NEAR_METH rows always pass. SLOWED rows are scored by the
    fitted GMM for their (T, off) bucket and kept iff their argmax-
    posterior component is in ``keep_idxs``. Missing buckets keep all.
    """
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
    )
    cats = arr[:, col_category].astype(np.int8)
    parent = arr[:, col_parent_meth].astype(np.int8)
    offset = arr[:, col_parent_offset].astype(np.int8)
    ipd_col = arr[:, [col_ipd]]

    keep = (cats == CATEGORY_BASELINE) | (cats == CATEGORY_NEAR_METH)
    slow_m = cats == CATEGORY_SLOWED
    n_slow_in = int(slow_m.sum())
    n_slow_kept = 0
    if not slow_m.any():
        return keep, n_slow_in, n_slow_kept

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
                keep |= mask_TO
                continue
            if int(params.get("fit_dim", 2)) == 1 and len(params["means"]) == 1:
                keep |= mask_TO
                n_slow_kept += int(mask_TO.sum())
                continue
            xs_TO = ipd_col[mask_TO].astype(np.float64)
            post = _gmm_posterior(
                xs_TO,
                np.asarray(params["means"]),
                np.asarray(params["covariances"]),
                np.asarray(params["weights"]),
            )
            assigned = post.argmax(axis=1)
            keep_idxs_TO = params.get("keep_idxs")
            if keep_idxs_TO is None:
                keep_local = assigned != int(params["baseline_idx"])
            else:
                keep_local = np.isin(assigned, np.asarray(keep_idxs_TO, dtype=int))
            full_idx = np.where(mask_TO)[0]
            tmp = np.zeros_like(mask_TO)
            tmp[full_idx[keep_local]] = True
            keep |= tmp
            n_slow_kept += int(keep_local.sum())
    return keep, n_slow_in, n_slow_kept


def _apply_gmm_filter_to_data(
    data: dict,
    fit_params_by_TO: dict,
    *,
    layout,
) -> tuple[dict, dict]:
    """Apply the GMM drop rule per-strand to a bilateral shard.

    A row survives iff BOTH strands keep it (independent per-strand
    score; OR-keep would let SLOWED noise leak from one strand into the
    training set via the other).
    """
    new_data: dict = {}
    counts = {
        "n_rows_in": 0,
        "n_rows_kept": 0,
        "n_slowed_in_fwd": 0,
        "n_slowed_kept_fwd": 0,
        "n_slowed_in_rev": 0,
        "n_slowed_kept_rev": 0,
    }

    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] != layout.n_cols:
            continue
        counts["n_rows_in"] += len(arr)

        keep_fwd, sf_in, sf_kept = _strand_keep_mask(
            arr,
            layout=layout,
            col_category=layout.col_category_fwd,
            col_parent_meth=layout.col_parent_meth_fwd,
            col_parent_offset=layout.col_parent_offset_fwd,
            col_ipd=layout.col_ipd_fwd,
            fit_params_by_TO=fit_params_by_TO,
        )
        keep_rev, sr_in, sr_kept = _strand_keep_mask(
            arr,
            layout=layout,
            col_category=layout.col_category_rev,
            col_parent_meth=layout.col_parent_meth_rev,
            col_parent_offset=layout.col_parent_offset_rev,
            col_ipd=layout.col_ipd_rev,
            fit_params_by_TO=fit_params_by_TO,
        )
        counts["n_slowed_in_fwd"] += sf_in
        counts["n_slowed_kept_fwd"] += sf_kept
        counts["n_slowed_in_rev"] += sr_in
        counts["n_slowed_kept_rev"] += sr_kept

        keep_rows = keep_fwd & keep_rev
        if keep_rows.any():
            new_data[int(kid)] = arr[keep_rows].copy()
            counts["n_rows_kept"] += int(keep_rows.sum())

    counts["n_slowed_dropped_fwd"] = counts["n_slowed_in_fwd"] - counts["n_slowed_kept_fwd"]
    counts["n_slowed_dropped_rev"] = counts["n_slowed_in_rev"] - counts["n_slowed_kept_rev"]
    return new_data, counts


def slowed_split_gmm(
    data: dict,
    *,
    layout,
    min_samples_for_gmm: int = 100,
    n_components: int | tuple[int, ...] = (1, 2, 3),
    strict_bic_nats_per_sample: float = 1.0,
    similarity_sigma_margin: float = 0.0,
    max_baseline_pool: int = 50_000_000,
    max_slowed_per_bucket: int = 10_000_000,
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
    3. Pick K∈``n_components`` (default ``(1, 2, 3)``) by BIC, **with
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

    # Normalise n_components to a tuple of candidate Ks. K=1 is allowed
    # (parsimony fallback when the bucket has no separable structure).
    if isinstance(n_components, int):
        candidate_ks: tuple[int, ...] = (n_components,)
    else:
        candidate_ks = tuple(int(k) for k in n_components)
    if not candidate_ks or any(k < 1 for k in candidate_ks):
        raise ValueError(f"n_components must be int ≥ 1 or tuple of ints ≥ 1, got {n_components}")

    rng = np.random.default_rng(seed)

    # ── Pass 1: harvest joint (IPD, PW) per (cat, parent_meth, offset) ──
    # IPD drives the fit and the drop rule (1D GMM). PW is harvested
    # alongside for descriptive logging — per-component PW means/stds
    # are reported but do not affect the decision.
    log.info("[refine.gmm] pass 1/2: harvesting (IPD, PW) per (meth, offset) ...")
    baseline_chunks: list = []
    slowed_chunks_by_TO: dict[tuple[int, int], list] = {}
    slowed_frac_by_TO: dict[tuple[int, int], list] = {}
    _harvest_into(
        data,
        baseline_chunks,
        slowed_chunks_by_TO,
        slowed_frac_by_TO,
        layout=layout,
        max_baseline_pool=max_baseline_pool,
        max_slowed_per_bucket=max_slowed_per_bucket,
        rng=rng,
    )

    if not baseline_chunks:
        log.warning("[refine.gmm] no baseline samples — keeping all slowed (no filter)")
        return _passthrough(data, method="gmm_no_baseline", layout=layout)

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
        similarity_sigma_margin=similarity_sigma_margin,
        seed=seed,
        rng=rng,
    )

    log.info("[refine.gmm] pass 2/2: filtering slowed rows (per-strand) ...")
    new_data, counts = _apply_gmm_filter_to_data(data, fit_params_by_TO, layout=layout)
    log.info(
        "[refine.gmm] rows: %d in -> %d kept (%.2f%%)",
        counts["n_rows_in"], counts["n_rows_kept"],
        100.0 * counts["n_rows_kept"] / max(counts["n_rows_in"], 1),
    )
    log.info(
        "[refine.gmm] slowed fwd: %d in -> %d kept (%d dropped)",
        counts["n_slowed_in_fwd"], counts["n_slowed_kept_fwd"], counts["n_slowed_dropped_fwd"],
    )
    log.info(
        "[refine.gmm] slowed rev: %d in -> %d kept (%d dropped)",
        counts["n_slowed_in_rev"], counts["n_slowed_kept_rev"], counts["n_slowed_dropped_rev"],
    )

    stats = {
        "method": "gmm_anchored_raw_ipd",
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
    *,
    layout,
    min_samples_for_gmm: int = 100,
    n_components: int | tuple[int, ...] = (1, 2, 3),
    strict_bic_nats_per_sample: float = 1.0,
    similarity_sigma_margin: float = 0.0,
    max_baseline_pool: int = 50_000_000,
    max_slowed_per_bucket: int = 10_000_000,
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
    if not candidate_ks or any(k < 1 for k in candidate_ks):
        raise ValueError(f"n_components must be int ≥ 1 or tuple of ints ≥ 1, got {n_components}")
    rng = np.random.default_rng(seed)

    # ── Phase 1: harvest (IPD, PW) pool from every shard ──────────────
    log.info(
        "[refine.gmm.shards] phase 1/3: harvesting (IPD, PW) from %d shards "
        "(caps: baseline=%s, slowed/bucket=%s) ...",
        len(shard_paths),
        f"{max_baseline_pool:,}" if max_baseline_pool > 0 else "off",
        f"{max_slowed_per_bucket:,}" if max_slowed_per_bucket > 0 else "off",
    )
    baseline_chunks: list = []
    slowed_chunks_by_TO: dict[tuple[int, int], list] = {}
    slowed_frac_by_TO: dict[tuple[int, int], list] = {}
    for i, shard_path in enumerate(shard_paths, start=1):
        log.info("[refine.gmm.shards]   harvest %d/%d  %s", i, len(shard_paths), shard_path.name)
        with open(shard_path, "rb") as f:
            data = pickle.load(f)
        # Reservoir caps are applied at the END of each _harvest_into call,
        # so peak memory is bounded by ~1.5 × cap regardless of corpus size.
        _harvest_into(
            data,
            baseline_chunks,
            slowed_chunks_by_TO,
            slowed_frac_by_TO,
            layout=layout,
            max_baseline_pool=max_baseline_pool,
            max_slowed_per_bucket=max_slowed_per_bucket,
            rng=rng,
        )
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
        similarity_sigma_margin=similarity_sigma_margin,
        seed=seed,
        rng=rng,
    )
    # Free the pool memory before phase 3.
    del slowed_chunks_by_TO, slowed_frac_by_TO, baseline_pool

    # ── Phase 3: apply per-shard, write atomically ─────────────────────
    log.info("[refine.gmm.shards] phase 3/3: filtering + writing %d shards ...", len(shard_paths))
    aggregate = {
        "n_rows_in": 0,
        "n_rows_kept": 0,
        "n_slowed_in_fwd": 0,
        "n_slowed_kept_fwd": 0,
        "n_slowed_in_rev": 0,
        "n_slowed_kept_rev": 0,
        "n_slowed_dropped_fwd": 0,
        "n_slowed_dropped_rev": 0,
    }
    for i, shard_path in enumerate(shard_paths, start=1):
        log.info("[refine.gmm.shards]   apply %d/%d  %s", i, len(shard_paths), shard_path.name)
        with open(shard_path, "rb") as f:
            data = pickle.load(f)
        orig_meta = data.pop("__meta__", None)
        int_keyed = {k: v for k, v in data.items() if isinstance(k, (int, np.integer))}

        new_data, counts = _apply_gmm_filter_to_data(int_keyed, fit_params_by_TO, layout=layout)
        for k, v in counts.items():
            aggregate[k] += v

        new_data["__meta__"] = {
            "refined_from": str(shard_path),
            "method": "gmm_anchored_raw_ipd",
            "stats": {**counts, "per_bucket": per_bucket_stats},
            "original_meta": orig_meta,
        }
        out_path = output_dir / f"{shard_path.stem}_clean.pkl"
        atomic_write_pickle(new_data, out_path)
        del data, new_data  # release before next shard

    log.info(
        "[refine.gmm.shards] DONE  rows %d -> %d (%.2f%% survival)  "
        "slowed_fwd %d -> %d (%d dropped)  slowed_rev %d -> %d (%d dropped)",
        aggregate["n_rows_in"], aggregate["n_rows_kept"],
        100.0 * aggregate["n_rows_kept"] / max(aggregate["n_rows_in"], 1),
        aggregate["n_slowed_in_fwd"], aggregate["n_slowed_kept_fwd"],
        aggregate["n_slowed_dropped_fwd"],
        aggregate["n_slowed_in_rev"], aggregate["n_slowed_kept_rev"],
        aggregate["n_slowed_dropped_rev"],
    )

    return {
        "method": "gmm_anchored_raw_ipd",
        "min_samples_for_gmm": min_samples_for_gmm,
        "seed": seed,
        "n_shards": len(shard_paths),
        **aggregate,
        "per_bucket": per_bucket_stats,
    }


def _passthrough(data: dict, method: str, *, layout) -> tuple[dict, dict]:
    """Return ``data`` unchanged (deep-copied at row level) with a stats stub.

    Used when GMM cannot be fit at all (e.g. no baseline samples). The
    pkl is still rewritten so downstream tools find a valid file.
    """
    from .utils.sample_layout import CATEGORY_SLOWED
    new_data: dict = {}
    n_rows = 0
    n_s_fwd = n_s_rev = 0
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] != layout.n_cols:
            continue
        new_data[int(kid)] = arr.copy()
        n_rows += len(arr)
        n_s_fwd += int((arr[:, layout.col_category_fwd] == CATEGORY_SLOWED).sum())
        n_s_rev += int((arr[:, layout.col_category_rev] == CATEGORY_SLOWED).sum())

    stats = {
        "method": method,
        "skipped": True,
        "n_rows_in": n_rows,
        "n_rows_kept": n_rows,
        "n_slowed_in_fwd": n_s_fwd,
        "n_slowed_kept_fwd": n_s_fwd,
        "n_slowed_dropped_fwd": 0,
        "n_slowed_in_rev": n_s_rev,
        "n_slowed_kept_rev": n_s_rev,
        "n_slowed_dropped_rev": 0,
    }
    return new_data, stats


# ---------------------------------------------------------------------------
# Orchestrator + CLI
# ---------------------------------------------------------------------------


def refine_pkl(
    in_path: Path,
    out_path: Path,
    min_samples_for_gmm: int = 100,
    n_components: int | tuple[int, ...] = (1, 2, 3),
    strict_bic_nats_per_sample: float = 1.0,
    similarity_sigma_margin: float = 0.0,
    max_baseline_pool: int = 50_000_000,
    max_slowed_per_bucket: int = 10_000_000,
    seed: int = 42,
) -> dict:
    """Refine one master .pkl OR a directory of shards (auto-detected).

    If ``in_path`` is a directory, the **sharded** path is used: pool
    harvest across shards → fit anchored GMM globally → apply per-shard
    and write to ``out_path/``. Peak RAM is bounded by one shard.

    If ``in_path`` is a file, the in-memory path is used: load, fit,
    filter, atomic write.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    from .data.dataset import read_shard_extraction_params
    from .utils.sample_layout import get_sample_layout

    def _layout_from_pkl(pkl_dict, src_name: str):
        """Resolve the SampleLayout for one shard. Fails fast if the shard's
        column count doesn't match its (declared or YAML-default) geometry."""
        params = read_shard_extraction_params(pkl_dict)
        layout = get_sample_layout(params)  # falls back to YAML if params None
        for k, v in pkl_dict.items():
            if isinstance(k, (int, np.integer)) and isinstance(v, np.ndarray):
                if v.shape[1] != layout.n_cols:
                    log.error(
                        "%s: shard has %d cols but layout (K=%d) expects %d. "
                        "Re-run `kinsim extract` under the current YAML, or "
                        "use a YAML / __meta__ matching the shard.",
                        src_name, v.shape[1], layout.kmer_size, layout.n_cols,
                    )
                    sys.exit(1)
                break
        return layout

    # ── Sharded mode: in_path is a directory of *_shard.pkl files. ──────
    if in_path.is_dir():
        first = next(iter(sorted(in_path.glob("*_shard.pkl"))), None)
        if first is None:
            log.error("No *_shard.pkl files in %s", in_path)
            sys.exit(1)
        with open(first, "rb") as f:
            probe = pickle.load(f)
        layout = _layout_from_pkl(probe, first.name)
        del probe

        log.info(
            "Refine (sharded, anchored, K=%d, %d cols): in=%s  out=%s  "
            "n_components=%s  strict_bic=%.2f nats/sample  min_samples_for_gmm=%d",
            layout.kmer_size, layout.n_cols, in_path, out_path,
            n_components if isinstance(n_components, int) else ",".join(map(str, n_components)),
            strict_bic_nats_per_sample, min_samples_for_gmm,
        )
        return slowed_split_gmm_shards(
            in_path,
            out_path,
            layout=layout,
            min_samples_for_gmm=min_samples_for_gmm,
            n_components=n_components,
            strict_bic_nats_per_sample=strict_bic_nats_per_sample,
            similarity_sigma_margin=similarity_sigma_margin,
            max_baseline_pool=max_baseline_pool,
            max_slowed_per_bucket=max_slowed_per_bucket,
            seed=seed,
        )

    # ── Single-pkl mode (legacy / small datasets) ──
    log.info("Loading: %s  (%.2f GB)", in_path, in_path.stat().st_size / 1e9)
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    layout = _layout_from_pkl(data, in_path.name)
    orig_meta = data.pop("__meta__", None)

    int_keyed = {k: v for k, v in data.items() if isinstance(k, (int, np.integer))}
    if not int_keyed:
        log.error("No int-keyed data found — input is not a kinsim extract pkl.")
        sys.exit(1)

    n_comp_summary = (
        f"{n_components}"
        if isinstance(n_components, int)
        else "auto-BIC over " + ",".join(str(k) for k in n_components)
    )
    log.info(
        "Refine: method=gmm (anchored, K=%d, %d cols)  n_components=%s  "
        "strict_bic=%.2f nats/sample  min_samples_for_gmm=%d",
        layout.kmer_size, layout.n_cols, n_comp_summary,
        strict_bic_nats_per_sample, min_samples_for_gmm,
    )
    new_data, stats = slowed_split_gmm(
        int_keyed,
        layout=layout,
        min_samples_for_gmm=min_samples_for_gmm,
        n_components=n_components,
        strict_bic_nats_per_sample=strict_bic_nats_per_sample,
        similarity_sigma_margin=similarity_sigma_margin,
        max_baseline_pool=max_baseline_pool,
        max_slowed_per_bucket=max_slowed_per_bucket,
        seed=seed,
    )

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
    ap.add_argument(
        "input_pkl",
        help="Shard .pkl from `kinsim extract`, OR a directory of *_shard.pkl (sharded mode)",
    )
    ap.add_argument("output_pkl", help="Output refined .pkl")
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
        default="1,2,3",
        help="GMM only. Either a single integer K (forced K) or a "
        "comma-separated list of candidate Ks for BIC-based selection. "
        "Default '1,2,3' — K=1 wins on plain BIC when the bucket has no "
        "separable structure (parsimony); K=2 is the workhorse for "
        "real meth + per-read-contamination cases; K=3 captures "
        "trimodal slowed pools (motif-FPs + partial + full occupancy). "
        "K=1 has lenient selection (plain BIC), K>2 must beat K=2 by "
        "the --strict-bic margin.",
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
        "Pass 2.0+ for even stricter K=2 preference. "
        "K=1 always uses plain BIC (parsimony), this margin doesn't apply.",
    )
    ap.add_argument(
        "--similarity-margin",
        type=float,
        default=0.0,
        help="GMM only. Drop components whose IPD mean is within "
        "(similarity_margin × σ_baseline_IPD) of the baseline IPD mean. "
        "0.0 (default) = strict above-baseline rule (current behaviour); "
        "0.5 = drop components within half a baseline-σ ('essentially "
        "baseline-shaped'); 1.0 = drop components within one full σ. "
        "Useful when weak-signal types (m4C, m5C distal offsets) "
        "produce components only marginally above baseline.",
    )
    ap.add_argument(
        "--max-baseline-pool",
        type=int,
        default=50_000_000,
        help="GMM only. Reservoir-cap the harvested baseline pool at this many "
        "rows. 0 = no cap. Default 50,000,000 — way more than needed for a "
        "2-component GMM (10K rows would converge identically), bounds peak "
        "memory at ~800 MB regardless of corpus size.",
    )
    ap.add_argument(
        "--max-slowed-per-bucket",
        type=int,
        default=10_000_000,
        help="GMM only. Reservoir-cap each (meth, offset) slowed pool at this "
        "many rows. 0 = no cap. Default 10,000,000 — generous for the meth "
        "lobe fit, bounds peak memory at ~80 MB per bucket.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for GMM init and baseline subsampling.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(verbose=args.verbose)

    in_p = Path(args.input_pkl)
    out_p = Path(args.output_pkl)
    if not in_p.exists():
        log.error("input not found: %s", in_p)
        sys.exit(1)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Parse --n-components: single int → forced K; "k1,k2,..." → BIC over candidates.
    raw = args.n_components.strip()
    parsed = tuple(int(p.strip()) for p in raw.split(",") if p.strip())
    if not parsed:
        log.error("--n-components must contain at least one integer")
        sys.exit(1)
    n_components_arg = parsed[0] if len(parsed) == 1 else parsed

    refine_pkl(
        in_p,
        out_p,
        min_samples_for_gmm=args.min_samples_for_gmm,
        n_components=n_components_arg,
        strict_bic_nats_per_sample=args.strict_bic,
        similarity_sigma_margin=args.similarity_margin,
        max_baseline_pool=args.max_baseline_pool,
        max_slowed_per_bucket=args.max_slowed_per_bucket,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
