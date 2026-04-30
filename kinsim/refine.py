"""Refine a KinSim .pkl in two passes:

Pass 1 — confirm methylated buckets (global per-meth-type GMM)
   For each meth type T (m6A, m4C, m5C, ...) all (kmer, T) buckets are
   pooled, a balanced None reference is sub-sampled, and a 2-D GMM is fit on
   the kinetic profile at the configured signature offsets. The cluster with
   the highest mean signal is treated as "real meth"; samples assigned to
   any other cluster are dropped as motif false-positives. Buckets that
   shrink below 3 surviving samples are dropped entirely.

Pass 2 — split the (kmer, 0) bucket into baseline vs slowed (v4 addition)
   The polymerase IPD/PW at position c can be elevated even when c itself
   is unmodified, IF an upstream confirmed methylation sits at a position
   c-k where k is one of the signature offsets for that mod type
   (e.g. m6A at c-5 elevates IPD at c). These "slowed" positions look like
   noise to a model trained only on (kmer, 0) → unmodified label, but the
   meth_context columns 3..13 actually encode the upstream context FiLM
   needs to disambiguate them.

   Pass 2 walks every (kmer, 0) sample, inspects its meth_context for an
   upstream methylation at a signature offset, and:
     - flags the sample as 'slowed_T' if a signature offset matches,
     - else keeps it as a true baseline.
   Baseline samples are capped at `n_baseline_per_kmer` per kmer (YAML
   parameter) to prevent class imbalance. The 95-th percentile of the
   baseline IPD distribution is used as a quality threshold: slowed samples
   whose center IPD falls below it are dropped (the expected slowing did
   not actually occur — likely an upstream motif false-call, edge effect,
   or low-coverage artefact).

   Both surviving baseline and slowed samples remain stored as (kmer, 0)
   in the output: the meth_context already in cols 3..13 is what FiLM
   uses to distinguish them at training time. No new dict key is needed.

Usage:
    kinsim refine in.pkl out.pkl
    kinsim refine in.pkl out.pkl --report report.tsv
    kinsim refine in.pkl out.pkl --no-slowed-split    # skip pass 2
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

CHI2_95_2DOF = 5.991        # χ² 0.95 quantile, 2 dof — 5% rejection (default)
CHI2_99_2DOF = 9.210        # χ² 0.99 quantile, 2 dof — 1% rejection (strict, low-count)
MOD_NAMES = {0: "none", 1: "m6A", 2: "m4C", 3: "m5C"}


# ---------------------------------------------------------------------------
# Pass 2: slowed-position vs baseline split (v4)
# ---------------------------------------------------------------------------


def _build_upstream_signature_targets(cfg: dict) -> list[tuple[int, int, str, int]]:
    """Resolve which (mc_index, expected_meth_id, type_name, offset) tuples
    flag a (kmer, 0) sample as a slowed position.

    The meth_context column ``mc[idx]`` corresponds to the meth_id at read
    position ``center - KMER_LEFT_PAD + idx`` (see kinsim.extract). For a
    sample whose center is at p+k (where p has a confirmed methylation of
    type T with signature offset k > 0), the upstream methylation at p
    appears at mc-index ``KMER_PRED_IDX - k``.

    We skip k = 0 because for a (kmer, 0) sample the center has meth_id = 0
    by construction — k = 0 can never produce a slowed signature here.

    Returns list of (mc_index, expected_meth_id, mod_name, signature_offset).
    """
    from .utils.encoding import KMER_PRED_IDX, get_meth_ids
    meth_ids = get_meth_ids()
    sigs = cfg.get("kinetic_signatures", {}) or {}

    targets: list[tuple[int, int, str, int]] = []
    for mod_name, sig_cfg in sigs.items():
        mid = meth_ids.get(mod_name)
        if not mid:                          # 0 (none) or missing
            continue
        for k in sig_cfg.get("signal_offsets", []):
            try:
                k = int(k)
            except (TypeError, ValueError):
                continue
            if k <= 0:                       # k = 0 invisible in (kmer, 0)
                continue
            mc_idx = KMER_PRED_IDX - k
            if mc_idx < 0:                   # offset too large for window
                log.warning("[%s] signature offset +%d exceeds meth_context "
                            "left pad (KMER_PRED_IDX=%d) — slowed split will "
                            "miss this offset.", mod_name, k, KMER_PRED_IDX)
                continue
            targets.append((mc_idx, mid, mod_name, k))
    return targets


def slowed_split(
    none_buckets:        dict,
    cfg:                 dict,
    n_baseline_per_kmer: int,
    secondary_pct:       float,
    rng:                 np.random.Generator,
) -> tuple[dict, dict]:
    """Pass 2 — split (kmer, 0) buckets into baseline + slowed, apply QC.

    Args:
        none_buckets:        dict[int kmer_id -> ndarray(N, 35)] of unmethylated
                             samples passed through from Pass 1.
        cfg:                 parsed kinsim_config.yaml.
        n_baseline_per_kmer: cap on baseline samples retained per kmer.
        secondary_pct:       percentile of baseline IPD used as a lower
                             threshold for slowed samples (e.g. 95).
        rng:                 numpy Generator for reproducible subsampling.

    Returns:
        new_none_buckets: dict[int kmer_id -> ndarray] with capped baseline
                          + QC-filtered slowed samples concatenated. Same
                          column layout as input.
        stats:            summary counters.
    """
    from .utils.encoding import KMER_PRED_IDX

    # Reverse map int meth_id -> name for offset distribution logging
    from .utils.encoding import get_meth_ids
    name_by_mid = {v: k for k, v in get_meth_ids().items()}

    targets = _build_upstream_signature_targets(cfg)
    if not targets:
        log.info("[slowed-split] no upstream signature offsets configured "
                 "(all signal_offsets are 0 or missing) — skipping pass 2.")
        return {kid: arr.copy() for kid, arr in none_buckets.items()}, {
            "n_baseline_in":    sum(len(a) for a in none_buckets.values()),
            "n_slowed_in":      0,
            "n_baseline_kept":  sum(len(a) for a in none_buckets.values()),
            "n_slowed_kept":    0,
            "n_slowed_dropped": 0,
            "threshold":        None,
        }

    log.info("[slowed-split] checking %d upstream signature targets:", len(targets))
    for mc_idx, mid, mname, k in targets:
        log.info("  %s offset +%d → meth_context[%d] == meth_id %d",
                 mname, k, mc_idx, mid)

    # Step 1: classify each sample of each (kmer, 0) bucket
    classified: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    n_baseline_in = 0
    n_slowed_in   = 0
    offset_distribution: Counter = Counter()

    METH_CTX_COL_START = 3                        # cols 3..13 hold meth_context
    METH_CTX_COL_END   = METH_CTX_COL_START + 11  # exclusive

    for kmer_id, arr in none_buckets.items():
        if arr.size == 0:
            classified[kmer_id] = (arr, arr)
            continue
        if arr.shape[1] < METH_CTX_COL_END:
            log.warning("[slowed-split] (kmer=%d, none) has only %d cols — "
                        "no meth_context, treating all as baseline.",
                        kmer_id, arr.shape[1])
            classified[kmer_id] = (arr, np.empty((0, arr.shape[1]), dtype=arr.dtype))
            n_baseline_in += len(arr)
            continue

        mc = arr[:, METH_CTX_COL_START:METH_CTX_COL_END].astype(np.int32)
        is_slowed = np.zeros(len(arr), dtype=bool)
        for mc_idx, mid, mname, k in targets:
            mask = (mc[:, mc_idx] == mid)
            if mask.any():
                offset_distribution[(mname, k)] += int(mask.sum())
            is_slowed |= mask

        baseline_arr = arr[~is_slowed]
        slowed_arr   = arr[is_slowed]
        n_baseline_in += int((~is_slowed).sum())
        n_slowed_in   += int(is_slowed.sum())

        # Cap baseline samples per kmer
        if len(baseline_arr) > n_baseline_per_kmer:
            idx = rng.choice(len(baseline_arr), n_baseline_per_kmer, replace=False)
            baseline_arr = baseline_arr[idx]

        classified[kmer_id] = (baseline_arr, slowed_arr)

    # Step 2: global baseline IPD percentile threshold
    baseline_ipds = []
    for baseline_arr, _ in classified.values():
        if len(baseline_arr):
            baseline_ipds.append(baseline_arr[:, 0])
    if baseline_ipds:
        pooled = np.concatenate(baseline_ipds)
        threshold = float(np.percentile(pooled, secondary_pct))
        log.info("[slowed-split] baseline IPD pool: n=%d, mean=%.2f, "
                 "median=%.2f, p%g=%.2f (threshold for slowed samples)",
                 len(pooled), float(pooled.mean()), float(np.median(pooled)),
                 secondary_pct, threshold)
    else:
        threshold = 0.0
        log.warning("[slowed-split] no baseline samples — threshold defaults "
                    "to 0 (no slowed samples will be filtered).")

    # Step 3: apply threshold to slowed, merge baseline + filtered slowed
    new_none_buckets: dict = {}
    n_baseline_kept   = 0
    n_slowed_kept     = 0
    n_slowed_dropped  = 0
    for kmer_id, (baseline_arr, slowed_arr) in classified.items():
        if len(slowed_arr):
            keep_mask = slowed_arr[:, 0] >= threshold
            slowed_kept = slowed_arr[keep_mask]
            n_slowed_dropped += int((~keep_mask).sum())
        else:
            slowed_kept = slowed_arr

        n_baseline_kept += len(baseline_arr)
        n_slowed_kept   += len(slowed_kept)

        if len(baseline_arr) == 0 and len(slowed_kept) == 0:
            continue
        if len(baseline_arr) == 0:
            new_none_buckets[kmer_id] = slowed_kept
        elif len(slowed_kept) == 0:
            new_none_buckets[kmer_id] = baseline_arr
        else:
            new_none_buckets[kmer_id] = np.concatenate([baseline_arr, slowed_kept], axis=0)

    log.info("[slowed-split] baseline: %d in → %d kept (cap=%d/kmer)",
             n_baseline_in, n_baseline_kept, n_baseline_per_kmer)
    log.info("[slowed-split] slowed:   %d in → %d kept, %d dropped "
             "(IPD < p%g = %.2f)",
             n_slowed_in, n_slowed_kept, n_slowed_dropped,
             secondary_pct, threshold)
    if offset_distribution:
        log.info("[slowed-split] slowed-offset distribution (before secondary filter):")
        for (mname, k), n in sorted(offset_distribution.items(),
                                    key=lambda x: (-x[1], x[0])):
            log.info("  %s @ +%d: %d samples", mname, k, n)

    stats = {
        "n_baseline_in":         n_baseline_in,
        "n_slowed_in":           n_slowed_in,
        "n_baseline_kept":       n_baseline_kept,
        "n_slowed_kept":         n_slowed_kept,
        "n_slowed_dropped":      n_slowed_dropped,
        "threshold":             threshold,
        "secondary_percentile":  secondary_pct,
        "n_baseline_per_kmer":   n_baseline_per_kmer,
        "offset_distribution":   {f"{m}+{k}": n for (m, k), n in offset_distribution.items()},
    }
    return new_none_buckets, stats


# ---------------------------------------------------------------------------
# v4 pass-2: secondary p95 filter on category-typed v4 master.pkl
# ---------------------------------------------------------------------------


def slowed_split_v4(
    data:                 dict,
    secondary_pct:        float,
    rng:                  np.random.Generator,
) -> tuple[dict, dict]:
    """Pass-2 secondary refine for the v4 36-col format.

    The v4 master.pkl is dict[int kmer_id -> ndarray(N, 36)] with col 35
    encoding the category (0=baseline, 1=meth, 2=slowed). This pass:

      1. Pools the IPD (col 0) of all baseline samples.
      2. Computes the secondary_pct percentile of that distribution as the
         lower threshold for slowed samples.
      3. Drops any slowed sample whose IPD is below the threshold (the
         expected slowing did not occur — likely an upstream FP that the
         pass-1 GMM bucket-level confirmation could not catch at the per-
         sample level).

    Meth and baseline samples are passed through unchanged.

    Returns (new_data dict, stats dict).
    """
    from .utils.sample_layout import (
        COL_CATEGORY, COL_IPD,
        CATEGORY_BASELINE, CATEGORY_METH, CATEGORY_SLOWED,
    )

    # 1. Pool baseline IPDs to compute threshold
    baseline_ipds: list = []
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_CATEGORY:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        m = (cats == CATEGORY_BASELINE)
        if m.any():
            baseline_ipds.append(arr[m, COL_IPD])
    if baseline_ipds:
        pooled = np.concatenate(baseline_ipds)
        threshold = float(np.percentile(pooled, secondary_pct))
        log.info("[slowed-split-v4] baseline IPD pool: n=%d, mean=%.2f, "
                 "median=%.2f, p%g=%.2f (threshold for slowed)",
                 len(pooled), float(pooled.mean()), float(np.median(pooled)),
                 secondary_pct, threshold)
    else:
        threshold = 0.0
        log.warning("[slowed-split-v4] no baseline samples — threshold=0")

    # 2. Filter slowed samples by IPD >= threshold
    new_data: dict = {}
    n_meth_in = n_meth_out = 0
    n_baseline_in = n_baseline_out = 0
    n_slowed_in = n_slowed_kept = n_slowed_dropped = 0
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_CATEGORY:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        meth_m = (cats == CATEGORY_METH)
        base_m = (cats == CATEGORY_BASELINE)
        slow_m = (cats == CATEGORY_SLOWED)
        n_meth_in     += int(meth_m.sum())
        n_baseline_in += int(base_m.sum())
        n_slowed_in   += int(slow_m.sum())
        # Slowed survivors: IPD >= threshold
        slow_keep_mask = slow_m & (arr[:, COL_IPD] >= threshold)
        slow_drop_mask = slow_m & ~slow_keep_mask
        n_slowed_kept    += int(slow_keep_mask.sum())
        n_slowed_dropped += int(slow_drop_mask.sum())
        # Reassemble surviving rows (meth + baseline + filtered slowed).
        keep_rows = meth_m | base_m | slow_keep_mask
        if keep_rows.any():
            new_data[int(kid)] = arr[keep_rows].copy()
            n_meth_out     += int(meth_m.sum())  # all meth pass through
            n_baseline_out += int(base_m.sum())  # all baseline pass through

    log.info("[slowed-split-v4] meth:     %d in -> %d kept",
             n_meth_in, n_meth_out)
    log.info("[slowed-split-v4] baseline: %d in -> %d kept",
             n_baseline_in, n_baseline_out)
    log.info("[slowed-split-v4] slowed:   %d in -> %d kept, %d dropped (IPD < p%g = %.2f)",
             n_slowed_in, n_slowed_kept, n_slowed_dropped, secondary_pct, threshold)

    stats = {
        "format":               "v4",
        "secondary_percentile": secondary_pct,
        "threshold":            threshold,
        "n_meth_in":            n_meth_in,
        "n_baseline_in":        n_baseline_in,
        "n_slowed_in":          n_slowed_in,
        "n_meth_out":           n_meth_out,
        "n_baseline_out":       n_baseline_out,
        "n_slowed_kept":        n_slowed_kept,
        "n_slowed_dropped":     n_slowed_dropped,
    }
    return new_data, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _detect_format(data: dict) -> str:
    """Return 'v3' or 'v4' based on the first non-meta key type."""
    for k in data:
        if k == "__meta__":
            continue
        if isinstance(k, tuple):
            return "v3"
        if isinstance(k, (int, np.integer)):
            return "v4"
    return "unknown"


def refine_pkl(
    in_path:               Path,
    out_path:              Path,
    report_path:           Optional[Path] = None,
    seed:                  int           = 42,
    enable_slowed_split:   bool          = True,
    n_baseline_per_kmer:   Optional[int]   = None,
    secondary_percentile:  Optional[float] = None,
) -> dict:
    """Refine a KinSim master.pkl. Auto-detects v3 vs v4 format.

    v3 input (tuple keys, 35-col arrays):
      Pass-1 GMM filter on (kmer, T) buckets — keeps real-meth-cluster
      samples, drops false-positive motif matches. Followed by an optional
      pass-2 slowed-split (heuristic, motif-match-based).

    v4 input (int kmer keys, 36-col arrays with CATEGORY column):
      Pass-2 only — secondary p95 filter on slowed samples. Pass-1 GMM
      has already happened upstream when the v4 extract used a refined
      master_clean.pkl as confirmed-meth source.
    """
    log.info("Loading: %s  (%.2f GB)", in_path, in_path.stat().st_size / 1e9)
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    orig_meta = data.pop("__meta__", None)
    rng = np.random.default_rng(seed)

    fmt = _detect_format(data)
    log.info("Refine: detected format = %s", fmt)

    # ---- v4 path: secondary p95 filter only ----
    if fmt == "v4":
        if not enable_slowed_split:
            log.warning("v4 input but --no-slowed-split given — output is "
                        "the input verbatim minus __meta__.")
            new_data = {k: v.copy() for k, v in data.items()
                        if isinstance(k, (int, np.integer))}
            stats = {"format": "v4", "skipped": True}
        else:
            sec_pct = (secondary_percentile if secondary_percentile is not None
                       else 95.0)
            new_data, stats = slowed_split_v4(data, sec_pct, rng)
        new_data["__meta__"] = {
            "refined_from":  str(in_path),
            "method":        "slowed_split_v4",
            "format":        "v4",
            "seed":          seed,
            "stats":         stats,
            "original_meta": orig_meta,
        }
        log.info("Writing: %s", out_path)
        with open(out_path, "wb") as f:
            pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return stats

    # ---- v3 path: full pass-1 GMM + optional pass-2 (existing behaviour) ----

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
    from .utils.sample_layout import METH_CTX_LEN, PROFILE_LEN
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

    # --- 5. Pass 2: slowed-vs-baseline split on (kmer, 0) buckets ---
    slowed_stats: Optional[dict] = None
    if enable_slowed_split:
        refine_cfg = (cfg.get("refine") or {}).get("slowed_split") or {}
        n_baseline = (n_baseline_per_kmer
                      if n_baseline_per_kmer is not None
                      else int(refine_cfg.get("n_baseline_per_kmer", 50)))
        sec_pct    = (secondary_percentile
                      if secondary_percentile is not None
                      else float(refine_cfg.get("secondary_percentile", 95)))
        log.info("Pass 2 (slowed-split): n_baseline_per_kmer=%d, "
                 "secondary_percentile=%.1f", n_baseline, sec_pct)

        # Strip current (kmer, 0) buckets from `out` before re-adding the
        # split versions. Pass 1 wrote them through unchanged (line ~339).
        none_in_out = {k[0]: v for k, v in out.items()
                       if isinstance(k, tuple) and k[1] == 0}
        for kid in list(none_in_out):
            out.pop((kid, 0), None)

        new_none, slowed_stats = slowed_split(
            none_in_out, cfg, n_baseline, sec_pct, rng,
        )
        for kmer_id, arr in new_none.items():
            out[(kmer_id, 0)] = arr

        # Adjust running sample-out counter: pass 1 had counted all none
        # samples; we now replace that contribution with the post-split count.
        n_samples_out -= sum(len(a) for a in none_in_out.values())
        n_samples_out += sum(len(a) for a in new_none.values())
    else:
        log.info("Pass 2 (slowed-split): SKIPPED (--no-slowed-split)")

    # --- 6. Write output ---
    out["__meta__"] = {
        "refined_from":     str(in_path),
        "method":           "global_gmm" + ("+slowed_split" if enable_slowed_split else ""),
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
        "slowed_split":     slowed_stats,
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
    ap.add_argument("--no-slowed-split", action="store_true",
                    help="Skip pass 2 (slowed-vs-baseline split on (kmer, 0) "
                         "buckets). Output keeps the v3 behaviour: all "
                         "(kmer, none) samples passed through unchanged.")
    ap.add_argument("--n-baseline-per-kmer", type=int, default=None,
                    help="Cap on baseline (kmer, none) samples kept per kmer "
                         "in pass 2. Overrides kinsim_config.yaml "
                         "refine.slowed_split.n_baseline_per_kmer.")
    ap.add_argument("--secondary-percentile", type=float, default=None,
                    help="Percentile of baseline IPD used as the lower "
                         "threshold for slowed samples in pass 2. Overrides "
                         "kinsim_config.yaml refine.slowed_split.secondary_percentile.")
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
        in_p, out_p, report_path=rep_p, seed=args.seed,
        enable_slowed_split   = not args.no_slowed_split,
        n_baseline_per_kmer   = args.n_baseline_per_kmer,
        secondary_percentile  = args.secondary_percentile,
    )


if __name__ == "__main__":
    main()
