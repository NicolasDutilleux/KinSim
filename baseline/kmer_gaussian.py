"""Baseline 2: Per-kmer Gaussian with IPD ratio shift for methylation.

Fits one N(mu, sigma^2) per observed kmer (ignoring methylation type).
For prediction of methylated positions, applies a multiplicative IPD ratio
shift based on the global methylation effect.

This is essentially a dictionary approach: one Gaussian per kmer for the
baseline (unmethylated) signal, plus a learned scalar multiplier per
methylation type.

Fit:
    1. Reads training BAMs from a manifest (same as main pipeline).
    2. Applies fraction-guided binarization (same as extract.py).
    3. For each kmer_id: collects all UNMETHYLATED IPD/PW → fits Gaussian.
    4. For each meth type: computes the global IPD ratio
       = mean(methylated IPD) / mean(unmethylated IPD).

Predict:
    1. Look up base Gaussian for the kmer: N(mu_ipd, sigma_ipd^2).
    2. Sample from it.
    3. If methylated: multiply IPD by the global IPD ratio for that type.
    4. Bernoulli coin-flip same as ConvPredictor.
"""

import json
import logging
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinsim.utils.config import load_manifest
from kinsim.utils.encoding import METH_IDS, K, KMER_MASK, BASE_MAP, decode_kmer
from kinsim.utils.motifs import load_motif_string, parse_motifs, scan_sequence

log = logging.getLogger(__name__)

METH_NAMES = {v: k for k, v in METH_IDS.items()}


# ---------------------------------------------------------------------------
# Fraction lookup (same as extract.py / global_gaussian.py)
# ---------------------------------------------------------------------------

def _build_fraction_lookup(motif_string: str) -> dict:
    fracs = {0: 0.0}
    if not motif_string:
        return fracs
    for entry in motif_string.split(';'):
        if not entry or ',' not in entry:
            continue
        parts = entry.split(',')
        if len(parts) < 3:
            continue
        m_id = METH_IDS.get(parts[0], 0)
        frac = float(parts[4]) if len(parts) >= 5 else 1.0
        fracs[m_id] = frac
    return fracs


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_from_manifest(manifest_path: str, max_reads_per_bam: int = 5000) -> dict:
    """Fit per-kmer Gaussians + global IPD ratios from manifest BAMs.

    Returns:
        dict with:
          'kmer_stats': {kmer_id: {'mu_ipd', 'sigma_ipd', 'mu_pw', 'sigma_pw', 'n'}}
          'ipd_ratios': {meth_id: float}  (global IPD ratio per meth type)
          'pw_ratios':  {meth_id: float}  (global PW ratio per meth type)
          'global_unmeth': {'mu_ipd', 'sigma_ipd', 'mu_pw', 'sigma_pw'}
              (fallback for unseen kmers)
    """
    import pysam

    entries = load_manifest(manifest_path)
    log.info("Manifest: %d entries", len(entries))

    # Collect per-kmer unmethylated samples + per-meth-type all samples
    kmer_unmeth = {}  # kmer_id → list of [ipd, pw]
    meth_ipds = {0: [], 1: [], 2: [], 3: []}  # meth_id → list of ipd values
    meth_pws = {0: [], 1: [], 2: [], 3: []}

    for entry in entries:
        bam_path = entry.bam_path
        if not os.path.isfile(bam_path):
            log.warning("BAM not found, skipping: %s", bam_path)
            continue

        motif_string = load_motif_string(entry.motifs)
        frac_lookup = _build_fraction_lookup(motif_string)
        motifs = parse_motifs(motif_string, revcomp=True)

        log.info("Processing %s", entry.sample_id)

        # Collect per (kmer, meth) for binarization
        per_key = {}
        n_reads = 0

        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            for read in bam:
                if max_reads_per_bam > 0 and n_reads >= max_reads_per_bam:
                    break
                seq = read.query_sequence
                if not (seq and len(seq) >= K and read.has_tag("fi")):
                    continue

                ipds = read.get_tag("fi")
                pws = read.get_tag("fp")
                min_len = min(len(seq), len(ipds), len(pws))
                meth_status = scan_sequence(seq[:min_len], motifs)

                mid = K // 2
                current_kmer = 0
                for i in range(min_len):
                    base_val = BASE_MAP.get(seq[i], 0)
                    current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK
                    if i >= K - 1:
                        center = i - mid
                        meth_id = int(meth_status[center])
                        if meth_id == 0 and np.random.random() >= 0.05:
                            continue
                        key = (current_kmer, meth_id)
                        frac = frac_lookup.get(meth_id, 0.0)
                        per_key.setdefault(key, []).append(
                            [float(ipds[center]), float(pws[center]), frac]
                        )

                n_reads += 1

        # Binarize and collect
        for (kmer_id, meth_id), samples in per_key.items():
            arr = np.array(samples, dtype=np.float32)

            if meth_id == 0:
                # Unmethylated → add to kmer dict and global unmeth pool
                kmer_unmeth.setdefault(kmer_id, []).extend(
                    arr[:, :2].tolist()
                )
                meth_ipds[0].extend(arr[:, 0].tolist())
                meth_pws[0].extend(arr[:, 1].tolist())
                continue

            n_samples = len(arr)
            mean_frac = float(arr[:, 2].mean())

            if mean_frac >= 0.99:
                meth_ipds[meth_id].extend(arr[:, 0].tolist())
                meth_pws[meth_id].extend(arr[:, 1].tolist())
            elif mean_frac <= 0.01:
                kmer_unmeth.setdefault(kmer_id, []).extend(
                    arr[:, :2].tolist()
                )
                meth_ipds[0].extend(arr[:, 0].tolist())
                meth_pws[0].extend(arr[:, 1].tolist())
            else:
                order = np.argsort(arr[:, 0])[::-1]
                arr_sorted = arr[order]
                n_high = max(1, int(np.ceil(mean_frac * n_samples)))

                # Top → methylated
                meth_ipds[meth_id].extend(arr_sorted[:n_high, 0].tolist())
                meth_pws[meth_id].extend(arr_sorted[:n_high, 1].tolist())
                # Bottom → unmethylated
                if n_high < n_samples:
                    kmer_unmeth.setdefault(kmer_id, []).extend(
                        arr_sorted[n_high:, :2].tolist()
                    )
                    meth_ipds[0].extend(arr_sorted[n_high:, 0].tolist())
                    meth_pws[0].extend(arr_sorted[n_high:, 1].tolist())

        log.info("  %s: %d reads", entry.sample_id, n_reads)

    # --- Fit per-kmer Gaussians (unmethylated only) ---
    kmer_stats = {}
    for kmer_id, samples in kmer_unmeth.items():
        arr = np.array(samples, dtype=np.float32)
        if len(arr) < 2:
            continue
        kmer_stats[kmer_id] = {
            'mu_ipd': float(np.mean(arr[:, 0])),
            'sigma_ipd': float(np.std(arr[:, 0])),
            'mu_pw': float(np.mean(arr[:, 1])),
            'sigma_pw': float(np.std(arr[:, 1])),
            'n': len(arr),
        }

    # --- Global unmethylated stats (fallback for unseen kmers) ---
    all_unmeth_ipd = np.array(meth_ipds[0], dtype=np.float32)
    all_unmeth_pw = np.array(meth_pws[0], dtype=np.float32)
    global_unmeth = {
        'mu_ipd': float(np.mean(all_unmeth_ipd)) if len(all_unmeth_ipd) > 0 else 10.0,
        'sigma_ipd': float(np.std(all_unmeth_ipd)) if len(all_unmeth_ipd) > 0 else 5.0,
        'mu_pw': float(np.mean(all_unmeth_pw)) if len(all_unmeth_pw) > 0 else 8.0,
        'sigma_pw': float(np.std(all_unmeth_pw)) if len(all_unmeth_pw) > 0 else 4.0,
    }

    # --- IPD/PW ratios per meth type ---
    mean_unmeth_ipd = global_unmeth['mu_ipd']
    mean_unmeth_pw = global_unmeth['mu_pw']

    ipd_ratios = {0: 1.0}
    pw_ratios = {0: 1.0}
    for meth_id in [1, 2, 3]:
        if len(meth_ipds[meth_id]) > 0:
            ipd_ratios[meth_id] = float(np.mean(meth_ipds[meth_id])) / max(mean_unmeth_ipd, 0.1)
            pw_ratios[meth_id] = float(np.mean(meth_pws[meth_id])) / max(mean_unmeth_pw, 0.1)
        else:
            ipd_ratios[meth_id] = 1.0
            pw_ratios[meth_id] = 1.0

    return {
        'kmer_stats': kmer_stats,
        'ipd_ratios': ipd_ratios,
        'pw_ratios': pw_ratios,
        'global_unmeth': global_unmeth,
    }


def print_model(model: dict):
    """Pretty-print the model summary."""
    n_kmers = len(model['kmer_stats'])
    total_samples = sum(v['n'] for v in model['kmer_stats'].values())

    print(f"\n=== Per-kmer Gaussian Baseline ===")
    print(f"Unique kmers observed: {n_kmers:,}")
    print(f"Total unmethylated samples: {total_samples:,}")
    print(f"Coverage: {n_kmers / (4**K) * 100:.1f}% of possible {4**K:,} 11-mers")
    print()

    print(f"Global unmethylated: mu_IPD={model['global_unmeth']['mu_ipd']:.2f}, "
          f"sigma_IPD={model['global_unmeth']['sigma_ipd']:.2f}, "
          f"mu_PW={model['global_unmeth']['mu_pw']:.2f}, "
          f"sigma_PW={model['global_unmeth']['sigma_pw']:.2f}")
    print()

    print(f"{'Meth type':<10} {'IPD ratio':>10} {'PW ratio':>10}")
    print("-" * 35)
    for meth_id in range(4):
        name = METH_NAMES.get(meth_id, f"id={meth_id}")
        print(f"{name:<10} {model['ipd_ratios'][meth_id]:>10.3f} "
              f"{model['pw_ratios'][meth_id]:>10.3f}")
    print()

    # Show top-5 and bottom-5 kmers by IPD
    sorted_kmers = sorted(model['kmer_stats'].items(), key=lambda x: x[1]['mu_ipd'])
    print("Lowest IPD kmers:")
    for kmer_id, stats in sorted_kmers[:5]:
        print(f"  {decode_kmer(kmer_id)}: mu_IPD={stats['mu_ipd']:.1f} (n={stats['n']})")
    print("Highest IPD kmers:")
    for kmer_id, stats in sorted_kmers[-5:]:
        print(f"  {decode_kmer(kmer_id)}: mu_IPD={stats['mu_ipd']:.1f} (n={stats['n']})")
    print()


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(model: dict, kmer_id: int, meth_id: int,
            fraction: float = 1.0, n: int = 1) -> np.ndarray:
    """Generate n (IPD, PW) samples for a given kmer + meth context.

    Steps:
    1. Look up base Gaussian for kmer (or global fallback if unseen).
    2. Sample from it.
    3. Bernoulli coin-flip for methylation.
    4. If methylated: multiply by the global IPD/PW ratio.

    Returns:
        np.ndarray of shape (n, 2) with [IPD, PW] clipped to [0, 255].
    """
    kmer_stats = model['kmer_stats']

    # Get base distribution (unmethylated)
    if kmer_id in kmer_stats:
        stats = kmer_stats[kmer_id]
    else:
        stats = model['global_unmeth']

    results = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        # Bernoulli coin-flip
        if meth_id > 0 and fraction < 1.0:
            if np.random.random() >= fraction:
                effective_meth = 0
            else:
                effective_meth = meth_id
        else:
            effective_meth = meth_id

        # Sample from base kmer Gaussian
        ipd = np.random.normal(stats['mu_ipd'], max(stats['sigma_ipd'], 0.1))
        pw = np.random.normal(stats['mu_pw'], max(stats['sigma_pw'], 0.1))

        # Apply IPD ratio shift for methylation
        if effective_meth > 0:
            ipd *= model['ipd_ratios'][effective_meth]
            pw *= model['pw_ratios'][effective_meth]

        results[i] = [np.clip(ipd, 0, 255), np.clip(pw, 0, 255)]

    return results


def save_model(model: dict, output_dir: str):
    """Save per-kmer model to output_dir (JSON meta + pickle kmer stats)."""
    os.makedirs(output_dir, exist_ok=True)

    meta = {
        'ipd_ratios': {str(k): v for k, v in model['ipd_ratios'].items()},
        'pw_ratios': {str(k): v for k, v in model['pw_ratios'].items()},
        'global_unmeth': model['global_unmeth'],
        'n_kmers': len(model['kmer_stats']),
    }
    meta_path = os.path.join(output_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    kmer_path = os.path.join(output_dir, "kmer_stats.pkl")
    with open(kmer_path, "wb") as f:
        pickle.dump(model['kmer_stats'], f)

    log.info("Saved: %s + %s", meta_path, kmer_path)


def load_model(model_dir: str) -> dict:
    """Load per-kmer model from output_dir."""
    meta_path = os.path.join(model_dir, "model_meta.json")
    kmer_path = os.path.join(model_dir, "kmer_stats.pkl")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(kmer_path, "rb") as f:
        kmer_stats = pickle.load(f)

    return {
        'kmer_stats': kmer_stats,
        'ipd_ratios': {int(k): v for k, v in meta['ipd_ratios'].items()},
        'pw_ratios': {int(k): v for k, v in meta['pw_ratios'].items()},
        'global_unmeth': meta['global_unmeth'],
    }
