"""Baseline 1: Global Gaussian — one N(mu, sigma^2) per methylation type.

No k-mer context at all. The simplest possible model: 4 Gaussians for IPD,
4 Gaussians for PW (one per meth state: none, m6A, m4C, m5C).

Fit:
    Reads training BAMs from a manifest (same format as kinsim extract output).
    For each (kmer, meth_id) key, applies the same fraction-guided binarization
    as the main pipeline: top F*N by IPD → methylated, rest → unmethylated.
    Aggregates all IPD/PW values per meth type (ignoring kmer), fits Gaussian.

Predict:
    Given a meth_id + fraction:
    - Bernoulli coin-flip (same as ConvPredictor pipeline)
    - Sample from the appropriate Gaussian
"""

import json
import logging
import os
import sys

import numpy as np

# Add parent to path for kinsim imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinsim.utils.config import load_manifest
from kinsim.utils.encoding import METH_IDS
from kinsim.utils.motifs import load_motif_string, parse_motifs, scan_sequence

log = logging.getLogger(__name__)

METH_NAMES = {v: k for k, v in METH_IDS.items()}


# ---------------------------------------------------------------------------
# Fraction lookup (same logic as extract.py)
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
# Fit: collect samples from manifest BAMs, aggregate per meth type
# ---------------------------------------------------------------------------

def fit_from_manifest(manifest_path: str, max_reads_per_bam: int = 5000) -> dict:
    """Read BAMs from manifest, collect IPD/PW per meth type.

    Uses the same binarization logic as the main pipeline:
    fraction-guided IPD splitting per (kmer, meth_id) key.

    Args:
        manifest_path: Path to manifest CSV.
        max_reads_per_bam: Max reads to process per BAM (for speed).

    Returns:
        dict with keys 0-3 (meth_id) → dict with mu_ipd, sigma_ipd,
        mu_pw, sigma_pw, n_samples.
    """
    import pysam
    from kinsim.utils.encoding import BASE_MAP, K, KMER_MASK

    entries = load_manifest(manifest_path)
    log.info("Manifest: %d entries", len(entries))

    # Collect per meth type: {meth_id: {'ipd': [...], 'pw': [...]}}
    collectors = {i: {'ipd': [], 'pw': []} for i in range(4)}

    for entry in entries:
        bam_path = entry.bam_path
        if not os.path.isfile(bam_path):
            log.warning("BAM not found, skipping: %s", bam_path)
            continue

        motif_string = load_motif_string(entry.motifs)
        frac_lookup = _build_fraction_lookup(motif_string)
        motifs = parse_motifs(motif_string, revcomp=True)

        log.info("Processing %s (%s)", entry.sample_id, bam_path)

        # First pass: collect per (kmer, meth_id) key for binarization
        per_key = {}  # (kmer, meth) → list of [ipd, pw, frac]
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
                        # Subsample unmethylated (same as main pipeline)
                        if meth_id == 0 and np.random.random() >= 0.05:
                            continue
                        key = (current_kmer, meth_id)
                        frac = frac_lookup.get(meth_id, 0.0)
                        per_key.setdefault(key, []).append(
                            [float(ipds[center]), float(pws[center]), frac]
                        )

                n_reads += 1

        # Binarize: same logic as _binarize_by_ipd in extract.py
        for (kmer_id, meth_id), samples in per_key.items():
            arr = np.array(samples, dtype=np.float32)
            if meth_id == 0:
                collectors[0]['ipd'].extend(arr[:, 0].tolist())
                collectors[0]['pw'].extend(arr[:, 1].tolist())
                continue

            n_samples = len(arr)
            mean_frac = float(arr[:, 2].mean())

            if mean_frac >= 0.99:
                collectors[meth_id]['ipd'].extend(arr[:, 0].tolist())
                collectors[meth_id]['pw'].extend(arr[:, 1].tolist())
            elif mean_frac <= 0.01:
                collectors[0]['ipd'].extend(arr[:, 0].tolist())
                collectors[0]['pw'].extend(arr[:, 1].tolist())
            else:
                order = np.argsort(arr[:, 0])[::-1]
                arr_sorted = arr[order]
                n_high = max(1, int(np.ceil(mean_frac * n_samples)))

                # Top n_high → methylated
                collectors[meth_id]['ipd'].extend(arr_sorted[:n_high, 0].tolist())
                collectors[meth_id]['pw'].extend(arr_sorted[:n_high, 1].tolist())
                # Rest → unmethylated
                if n_high < n_samples:
                    collectors[0]['ipd'].extend(arr_sorted[n_high:, 0].tolist())
                    collectors[0]['pw'].extend(arr_sorted[n_high:, 1].tolist())

        log.info("  %s: %d reads processed", entry.sample_id, n_reads)

    # Fit Gaussians
    model = {}
    for meth_id in range(4):
        ipd_arr = np.array(collectors[meth_id]['ipd'], dtype=np.float32)
        pw_arr = np.array(collectors[meth_id]['pw'], dtype=np.float32)
        n = len(ipd_arr)
        if n == 0:
            model[meth_id] = {
                'mu_ipd': 0.0, 'sigma_ipd': 1.0,
                'mu_pw': 0.0, 'sigma_pw': 1.0,
                'n_samples': 0,
            }
            continue
        model[meth_id] = {
            'mu_ipd': float(np.mean(ipd_arr)),
            'sigma_ipd': float(np.std(ipd_arr)),
            'mu_pw': float(np.mean(pw_arr)),
            'sigma_pw': float(np.std(pw_arr)),
            'n_samples': n,
        }

    return model


def print_model(model: dict):
    """Pretty-print the 4 Gaussians."""
    print("\n=== Global Gaussian Baseline ===")
    print(f"{'Type':<10} {'N':>10} {'mu_IPD':>8} {'sigma_IPD':>10} {'mu_PW':>8} {'sigma_PW':>10}")
    print("-" * 62)
    for meth_id in range(4):
        name = METH_NAMES.get(meth_id, f"id={meth_id}")
        m = model[meth_id]
        print(f"{name:<10} {m['n_samples']:>10} {m['mu_ipd']:>8.2f} {m['sigma_ipd']:>10.2f} "
              f"{m['mu_pw']:>8.2f} {m['sigma_pw']:>10.2f}")
    print()


# ---------------------------------------------------------------------------
# Predict: sample from the appropriate Gaussian
# ---------------------------------------------------------------------------

def predict(model: dict, meth_id: int, fraction: float = 1.0, n: int = 1) -> np.ndarray:
    """Generate n (IPD, PW) samples using Bernoulli coin-flip + Gaussian.

    Same logic as ConvPredictor generation: for each sample, coin-flip
    decides if methylated or not based on fraction.

    Returns:
        np.ndarray of shape (n, 2) with [IPD, PW] clipped to [0, 255].
    """
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

        m = model[effective_meth]
        ipd = np.random.normal(m['mu_ipd'], m['sigma_ipd'])
        pw = np.random.normal(m['mu_pw'], m['sigma_pw'])
        results[i] = [np.clip(ipd, 0, 255), np.clip(pw, 0, 255)]

    return results


def save_model(model: dict, output_path: str):
    """Save fitted Gaussians as JSON."""
    json_model = {str(k): v for k, v in model.items()}
    with open(output_path, "w") as f:
        json.dump(json_model, f, indent=2)
    log.info("Saved model to %s", output_path)


def load_model(model_path: str) -> dict:
    """Load fitted Gaussians from JSON."""
    with open(model_path) as f:
        json_model = json.load(f)
    return {int(k): v for k, v in json_model.items()}
