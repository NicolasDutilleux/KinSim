"""Standalone GMM binarization of raw .pkl dictionaries.

Applies per-kmer 2-component GMM on (IPD, PW) to separate truly methylated
reads from unmethylated reads at motif sites.  Operates on merged master
.pkl files produced by ``kinsim merge``.

Pipeline:
    kinsim extract --no-binarize ...   → raw shards
    kinsim merge shards/ master_raw.pkl
    kinsim binarize master_raw.pkl master_clean.pkl
    kinsim-prep balance master_clean.pkl master_balanced.pkl  (optional)

The binarization logic is the same as ``_binarize_by_ipd`` in extract.py
but applied as a post-hoc step on the full merged dataset, giving the GMM
more samples per key for better separation.
"""

import argparse
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .utils.config import setup_logging
from .utils.encoding import K, METH_IDS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binarization
# ---------------------------------------------------------------------------

def binarize_pkl(
    data: dict,
    min_samples_split: int = 20,
    min_split_survivors: int = 3,
    gmm_ratio_threshold: float = 1.3,
    m6a_ipd_threshold: float = 1.5,
    m4c_ipd_threshold: float = 1.3,
    m5c_ipd_threshold: float = 1.2,
) -> dict:
    """Apply 2-stage GMM binarization to a raw .pkl dictionary.

    Stage 1: Key-level filtering by IPD/PW ratio vs unmethylated baseline.
    Stage 2: Per-kmer read-level GMM split for stoichiometric methylation.

    Args:
        data:                 Raw pkl dict with (kmer_id, meth_id) keys.
        min_samples_split:    Min samples to attempt per-kmer GMM.
        min_split_survivors:  Min methylated reads after split.
        gmm_ratio_threshold:  Min centroid ratio for accepting GMM split.
        m6a_ipd_threshold:    Key-level IPD ratio threshold for m6A.
        m4c_ipd_threshold:    Key-level IPD ratio threshold for m4C.
        m5c_ipd_threshold:    Key-level IPD ratio threshold for m5C.

    Returns:
        New dict with binarized methylation labels.
    """
    from sklearn.mixture import GaussianMixture

    ACCEPT_RULES = {
        1: lambda ir, pr: ir >= m6a_ipd_threshold,                                # m6A
        2: lambda ir, pr: ir >= m4c_ipd_threshold or (ir >= 1.1 and pr >= 1.15),   # m4C
        3: lambda ir, pr: ir >= m5c_ipd_threshold,                                 # m5C
    }
    METH_NAMES = {1: 'm6A', 2: 'm4C', 3: 'm5C'}

    # ── Step 1: per-key mean IPD / PW ─────────────────────────────────────
    key_means: dict[tuple, tuple[float, float]] = {}
    for key, arr in data.items():
        if not isinstance(key, tuple):
            continue
        key_means[key] = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))

    # ── Step 2: build none reference per kmer ─────────────────────────────
    none_ref: dict[int, tuple[float, float]] = {}
    for key, (m_ipd, m_pw) in key_means.items():
        if key[1] == 0:
            none_ref[key[0]] = (m_ipd, m_pw)

    log.info("Binarization: %d none reference kmers", len(none_ref))

    # ── Step 3: classify each meth key by ratio thresholds ────────────────
    keys_keep: set = set()
    keys_reject: set = set()
    type_stats: dict[int, dict] = defaultdict(lambda: {
        'kept': 0, 'rejected': 0, 'no_ref': 0,
        'kept_ipd_ratios': [], 'kept_pw_ratios': [],
    })

    for key, (m_ipd, m_pw) in key_means.items():
        kmer_id, meth_id = key
        if meth_id == 0:
            continue

        ts = type_stats[meth_id]

        if kmer_id not in none_ref:
            keys_reject.add(key)
            ts['no_ref'] += 1
            continue

        ref_ipd, ref_pw = none_ref[kmer_id]
        ipd_ratio = m_ipd / max(ref_ipd, 1e-9)
        pw_ratio = m_pw / max(ref_pw, 1e-9)

        accept_fn = ACCEPT_RULES.get(meth_id, lambda ir, pr: ir >= 1.5)
        if accept_fn(ipd_ratio, pw_ratio):
            keys_keep.add(key)
            ts['kept'] += 1
            ts['kept_ipd_ratios'].append(ipd_ratio)
            ts['kept_pw_ratios'].append(pw_ratio)
        else:
            keys_reject.add(key)
            ts['rejected'] += 1

    # Log per-type summary
    for meth_id in sorted(type_stats):
        ts = type_stats[meth_id]
        mtype = METH_NAMES.get(meth_id, f'type{meth_id}')
        n_tot = ts['kept'] + ts['rejected'] + ts['no_ref']
        if ts['kept_ipd_ratios']:
            kr = np.array(ts['kept_ipd_ratios'])
            pr = np.array(ts['kept_pw_ratios'])
            log.info(
                "  %s: %d/%d kept, %d rejected, %d no-ref  |  "
                "kept IPD_ratio: median=%.2f [min=%.2f max=%.2f]  "
                "PW_ratio: median=%.2f",
                mtype, ts['kept'], n_tot, ts['rejected'], ts['no_ref'],
                float(np.median(kr)), float(np.min(kr)), float(np.max(kr)),
                float(np.median(pr)),
            )
        else:
            log.info(
                "  %s: 0/%d kept, %d rejected, %d no-ref",
                mtype, n_tot, ts['rejected'], ts['no_ref'],
            )

    # ── Step 4: build new result + per-kmer sample split ──────────────────
    new_result: dict = {}
    none_extras: dict = {}
    n_kept_keys = 0
    n_rejected_keys = 0
    n_sample_split = 0
    n_kept_whole = 0

    for key, arr in data.items():
        if not isinstance(key, tuple):
            new_result[key] = arr
            continue
        kmer_id, meth_id = key
        if meth_id == 0:
            new_result[key] = arr
            continue

        # Rejected → all samples to none
        if key in keys_reject:
            low_arr = arr.copy()
            low_arr[:, 2] = 0.0
            if low_arr.shape[1] >= 14:
                low_arr[:, 3 + K // 2] = 0
            none_extras.setdefault((kmer_id, 0), []).append(low_arr)
            n_rejected_keys += 1
            continue

        # Accepted — try per-kmer read-level split for stoichiometry
        n_samples = len(arr)
        if n_samples < min_samples_split:
            kept = arr.copy()
            kept[:, 2] = 1.0
            new_result[key] = kept
            n_kept_keys += 1
            n_kept_whole += 1
            continue

        # Per-kmer GMM on raw (IPD, PW) to split meth vs unmeth reads
        ipd_pw = arr[:, :2].astype(np.float64)
        try:
            gmm = GaussianMixture(
                n_components=2, covariance_type='full',
                n_init=3, random_state=42, max_iter=100,
            )
            gmm.fit(ipd_pw)
        except Exception:
            kept = arr.copy()
            kept[:, 2] = 1.0
            new_result[key] = kept
            n_kept_keys += 1
            n_kept_whole += 1
            continue

        centroids_ipd = gmm.means_[:, 0]
        hi = int(np.argmax(centroids_ipd))
        lo = 1 - hi
        ratio = centroids_ipd[hi] / max(centroids_ipd[lo], 1e-9)

        # No clear read-level separation → keep all as meth
        if ratio < gmm_ratio_threshold:
            kept = arr.copy()
            kept[:, 2] = 1.0
            new_result[key] = kept
            n_kept_keys += 1
            n_kept_whole += 1
            continue

        labels = gmm.predict(ipd_pw)
        high_mask = labels == hi
        n_high = int(high_mask.sum())

        if n_high >= min_split_survivors:
            high_arr = arr[high_mask].copy()
            high_arr[:, 2] = 1.0
            new_result[key] = high_arr
            n_kept_keys += 1
            n_sample_split += 1

            low_arr = arr[~high_mask].copy()
            low_arr[:, 2] = 0.0
            if low_arr.shape[1] >= 14:
                low_arr[:, 3 + K // 2] = 0
            none_extras.setdefault((kmer_id, 0), []).append(low_arr)
        else:
            kept = arr.copy()
            kept[:, 2] = 1.0
            new_result[key] = kept
            n_kept_keys += 1
            n_kept_whole += 1

    # Merge reclassified samples into none keys
    for none_key, arrays in none_extras.items():
        extra = np.concatenate(arrays, axis=0)
        if none_key in new_result:
            new_result[none_key] = np.concatenate(
                [new_result[none_key], extra], axis=0,
            )
        else:
            new_result[none_key] = extra

    log.info(
        "Binarization complete: %d meth keys kept (%d sample-split, %d kept-whole), "
        "%d rejected as false positive",
        n_kept_keys, n_sample_split, n_kept_whole, n_rejected_keys,
    )
    return new_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="kinsim binarize",
        description=(
            "Apply GMM binarization to a raw .pkl dictionary.\n\n"
            "Separates truly methylated reads from unmethylated reads\n"
            "at motif sites using per-kmer 2-component GMM on (IPD, PW).\n\n"
            "Pipeline:\n"
            "  kinsim extract --no-binarize ...  → raw shards\n"
            "  kinsim merge shards/ master_raw.pkl\n"
            "  kinsim binarize master_raw.pkl master_clean.pkl\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input raw .pkl file")
    parser.add_argument("output", help="Output binarized .pkl file")
    parser.add_argument("--min-samples-split", type=int, default=20,
                        help="Min samples per key for GMM split (default: 20)")
    parser.add_argument("--min-split-survivors", type=int, default=3,
                        help="Min methylated reads after split (default: 3)")
    parser.add_argument("--gmm-ratio", type=float, default=1.3,
                        help="Min centroid IPD ratio for accepting split (default: 1.3)")
    parser.add_argument("--m6a-threshold", type=float, default=1.5,
                        help="Key-level IPD ratio threshold for m6A (default: 1.5)")
    parser.add_argument("--m4c-threshold", type=float, default=1.3,
                        help="Key-level IPD ratio threshold for m4C (default: 1.3)")
    parser.add_argument("--m5c-threshold", type=float, default=1.2,
                        help="Key-level IPD ratio threshold for m5C (default: 1.2)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    log.info("Loading raw pkl: %s", args.input)
    with open(args.input, "rb") as f:
        data = pickle.load(f)

    # Stats before
    n_keys = sum(1 for k in data if isinstance(k, tuple))
    n_samples = sum(len(v) for k, v in data.items() if isinstance(k, tuple))
    meth_counts = {}
    for k, v in data.items():
        if not isinstance(k, tuple):
            continue
        meth_id = k[1]
        meth_counts[meth_id] = meth_counts.get(meth_id, 0) + len(v)
    log.info("Before: %d keys, %d samples | per meth_id: %s", n_keys, n_samples, meth_counts)

    result = binarize_pkl(
        data,
        min_samples_split=args.min_samples_split,
        min_split_survivors=args.min_split_survivors,
        gmm_ratio_threshold=args.gmm_ratio,
        m6a_ipd_threshold=args.m6a_threshold,
        m4c_ipd_threshold=args.m4c_threshold,
        m5c_ipd_threshold=args.m5c_threshold,
    )

    # Stats after
    n_keys_after = sum(1 for k in result if isinstance(k, tuple))
    n_samples_after = sum(len(v) for k, v in result.items() if isinstance(k, tuple))
    meth_after = {}
    for k, v in result.items():
        if not isinstance(k, tuple):
            continue
        meth_id = k[1]
        meth_after[meth_id] = meth_after.get(meth_id, 0) + len(v)
    log.info("After:  %d keys, %d samples | per meth_id: %s", n_keys_after, n_samples_after, meth_after)

    # Update metadata
    meta = result.get("__meta__", {})
    if isinstance(meta, dict):
        meta["binarized"] = True
        meta["binarize_params"] = {
            "min_samples_split": args.min_samples_split,
            "min_split_survivors": args.min_split_survivors,
            "gmm_ratio_threshold": args.gmm_ratio,
            "m6a_ipd_threshold": args.m6a_threshold,
            "m4c_ipd_threshold": args.m4c_threshold,
            "m5c_ipd_threshold": args.m5c_threshold,
        }
        result["__meta__"] = meta

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(result, f)
    log.info("Saved: %s", args.output)
