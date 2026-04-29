"""Extract raw IPD/PW training samples from BAM files.

This is the data-preparation pipeline for KinSim MLP training.
It has no dependency on the model itself.

Data format
-----------
Each shard is a pickle file containing:

    dict[(kmer_id: int, meth_id: int)] -> np.ndarray(N, 14)

where columns are [IPD, PW, fraction, mc_0, mc_1, ..., mc_10] as raw float32.
IPD and PW are read from the fi/fp BAM tags (uint8 [0, 255]).
Column 2 is the stoichiometric methylation fraction.
Columns 3–13 are the methylation IDs (0=none,1=m6A,2=m4C,3=m5C) for each
of the 11 positions in the k-mer window, with the active site at mc_5
(index K//2 = 5).

Backward compatibility: older shards may have only 2 columns [IPD, PW] or
3 columns [IPD, PW, fraction].  Dataset classes handle all three formats
by zero-padding the missing meth-context columns.

A special metadata key ``"__meta__"`` (string, not a tuple) may be present
in any shard or master .pkl.  It holds provenance information (version,
motifs, timestamp) and is automatically skipped by dataset classes.

Why raw (not log-transformed)?
    The extract/merge pipeline stores raw values so that:
      - Shards can be inspected and plotted without model knowledge
      - Different models can apply their own transforms at load time
      - MLPSignalDataset (data/dataset.py) applies log_transform once

CLI — single-BAM mode (original interface, unchanged):
    kinsim extract reads.bam "m6A,GATC,1" shard.pkl
    kinsim merge   shards/   master_data.pkl

CLI — manifest mode (new, recommended for SLURM array jobs):
    kinsim extract --manifest manifest.csv --task 3 --output-dir shards/
    kinsim merge   shards/  master_data.pkl

Manifest CSV format (see kinsim/config.py):
    sample_id,bam_path,motifs
    strain1,/data/bam1.bam,"m6A,GATC,1"
    strain2,/data/bam2.bam,/data/motifs/strain2.csv
"""

from __future__ import annotations

import datetime
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysam

from .utils.encoding import BASE_MAP, K, KMER_MASK, METH_IDS, kmer_mask
from .utils.motifs import (filter_motif_string_by_types, load_motif_string,
                           parse_meth_types_arg, parse_motifs,
                           reverse_complement, scan_sequence)

try:
    from . import __version__ as _KINSIM_VERSION
except (ImportError, AttributeError):
    try:
        from .__main__ import __version__ as _KINSIM_VERSION
    except ImportError:
        _KINSIM_VERSION = "unknown"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fail-fast BAM validation
# ---------------------------------------------------------------------------

def validate_bam_kinetics(bam_path: str, n_check: int = 10) -> str:
    """Raise ValueError if the BAM has no kinetic tags, return which pair it has.

    Accepts either ``fi``/``fp`` (unaligned BAM convention) or ``ip``/``pw``
    (aligned BAM convention after pbmm2).  Returns ``"fi"`` or ``"ip"`` to
    tell the caller which tag pair to read.

    Args:
        bam_path: Path to the BAM file.
        n_check:  Maximum reads to scan before giving up.

    Returns:
        "fi" if the BAM carries fi/fp; "ip" if it carries ip/pw.

    Raises:
        FileNotFoundError: If the BAM does not exist.
        ValueError: If no reads with a supported kinetic tag pair are found.
    """
    if not os.path.exists(bam_path):
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    log.debug("Validating kinetic tags in: %s", bam_path)
    reads_seen = 0
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if not read.query_sequence:
                continue
            if read.has_tag("fi"):
                log.debug("fi/fp tags confirmed in read %s", read.query_name)
                return "fi"
            if read.has_tag("ip"):
                log.debug("ip/pw tags confirmed in read %s", read.query_name)
                return "ip"
            reads_seen += 1
            if reads_seen >= n_check:
                break

    raise ValueError(
        f"BAM file has no kinetic tags (checked {reads_seen} reads): {bam_path}\n"
        "Looked for fi/fp (unaligned) or ip/pw (aligned after pbmm2).\n"
        "Kinetic tags are written by the PacBio instrument during primary analysis.\n"
        "Ensure the BAM was produced with --emit-kinetics (or equivalent)."
    )


# ---------------------------------------------------------------------------
# Fraction lookup from motif string
# ---------------------------------------------------------------------------

def _build_fraction_lookup(motif_string: str) -> dict[int, float]:
    """Parse the motif string to build a meth_id → fraction lookup.

    The motif string format is "m6A,GATC,1[,nDetected[,fraction]];..."
    When PacBio motifs.csv is the source, parse_motifs_csv preserves the
    fraction as the 5th field.  For plain motif strings without a fraction
    field, defaults to 1.0 (fully methylated).

    If multiple motifs share the same meth_id (e.g., two m6A motifs with
    different fractions), the last one wins.  This is acceptable because
    the kmer embedding already distinguishes different motif contexts.

    Returns:
        dict mapping meth_id → float fraction.  Always includes {0: 0.0}.
    """
    from .utils.encoding import METH_IDS

    fracs: dict[int, float] = {0: 0.0}
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
# Methylation-context window (asymmetric, upstream-biased)
# ---------------------------------------------------------------------------

# Asymmetric window around the prediction position. Both the kmer (sequence
# context) and the methylation context use the same window: [-7, +3] = 11
# positions, prediction position at index 7. The polymerase has more upstream
# context already incorporated than downstream prevued bases. See
# `kinsim/utils/encoding.py` for the central definitions.
from .utils.encoding import KMER_LEFT_PAD, KMER_RIGHT_PAD, KMER_PRED_IDX, K
METH_CTX_LEFT  = KMER_LEFT_PAD     # = 7
METH_CTX_RIGHT = KMER_RIGHT_PAD    # = 3
METH_CTX_LEN   = K                 # = 11

# Kinetic profile window: IPD/PW values at offsets [0, +PROFILE_LEN-1] from
# the prediction position. Used by refine to validate the signature pattern
# (e.g. for m5C the signal is at +2 and +6, not at the position itself).
PROFILE_START = 0
PROFILE_END   = 8
PROFILE_LEN   = PROFILE_END - PROFILE_START + 1   # = 9

# Total per-sample column count:
#   0..1   : IPD center, PW center
#   2      : fraction
#   3..13  : mc_0..mc_10  (11 meth context values, [-8, +2])
#   14..22 : profile_IPD_0..+8  (9 values)
#   23..31 : profile_PW_0..+8   (9 values)
SAMPLE_NCOLS = 3 + METH_CTX_LEN + 2 * PROFILE_LEN     # = 32


def _slice_meth_context(meth_status, center):
    """Return an 11-element list covering [-8, +2] around `center`.

    Out-of-range positions (start of read or end of read) are padded with 0
    (unmethylated) so every sample has the same fixed-length context.
    """
    n = len(meth_status)
    out = [0] * METH_CTX_LEN
    for k in range(METH_CTX_LEN):
        pos = center - METH_CTX_LEFT + k
        if 0 <= pos < n:
            out[k] = int(meth_status[pos])
    return out


def _slice_kinetic_profile(ipds, pws, center):
    """Return a list of 18 values: [profile_IPD_0..+8, profile_PW_0..+8].

    Out-of-range positions (past the end of the read) are padded with 0.
    """
    n_ipd = len(ipds)
    n_pw  = len(pws)
    ipd_prof = [0.0] * PROFILE_LEN
    pw_prof  = [0.0] * PROFILE_LEN
    for k in range(PROFILE_LEN):
        pos = center + PROFILE_START + k
        if 0 <= pos < n_ipd:
            ipd_prof[k] = float(ipds[pos])
        if 0 <= pos < n_pw:
            pw_prof[k]  = float(pws[pos])
    return ipd_prof + pw_prof


# ---------------------------------------------------------------------------
# IPD-based binarization helpers
# ---------------------------------------------------------------------------


def _binarize_by_ipd(result: dict) -> dict:
    """Binarize methylated keys using per-type IPD/PW ratio thresholds.

    For each (kmer, meth_id > 0) key, compute the IPD ratio and PW ratio
    relative to the same kmer's unmethylated baseline::

        IPD_ratio = mean_IPD(kmer, meth) / mean_IPD(kmer, none)
        PW_ratio  = mean_PW(kmer, meth)  / mean_PW(kmer, none)

    Then apply biologically-grounded acceptance rules per methylation type:

    - **m6A**: IPD_ratio >= 1.5
      Strong IPD elevation (2-6x on CLR, attenuated to ~2-3x on HiFi).
      PW typically *decreases* for m6A, so it is not used as positive signal.

    - **m4C**: IPD_ratio >= 1.3  OR  (IPD_ratio >= 1.1 AND PW_ratio >= 1.15)
      Moderate IPD elevation + distinctive PW elevation. A key with
      borderline IPD but clearly elevated PW is still accepted.

    - **m5C**: IPD_ratio >= 1.2
      Very weak kinetic effect on HiFi; m5C is at the detection limit
      after CCS averaging.

    Keys without a kmer-specific none counterpart are rejected (no reliable
    baseline to compute a ratio against).

    Within accepted keys that have >= 20 samples, a per-kmer 2-component GMM
    on raw (IPD, PW) separates methylated reads from unmethylated reads to
    handle stoichiometric / partial methylation.
    """
    from sklearn.mixture import GaussianMixture

    MIN_SAMPLES_SPLIT   = 20   # min samples for per-kmer read-level GMM
    MIN_SPLIT_SURVIVORS = 3    # min meth reads after per-kmer split

    # Per-type acceptance rules: (meth_id) → function(ipd_ratio, pw_ratio) → bool
    ACCEPT_RULES = {
        1: lambda ir, pr: ir >= 1.5,                                    # m6A
        2: lambda ir, pr: ir >= 1.3 or (ir >= 1.1 and pr >= 1.15),     # m4C
        3: lambda ir, pr: ir >= 1.2,                                    # m5C
    }
    METH_NAMES = {1: 'm6A', 2: 'm4C', 3: 'm5C'}

    # ── Step 1: per-key mean IPD / PW ──────────────────────────────────────
    key_means: dict[tuple, tuple[float, float]] = {}
    for key, arr in result.items():
        if not isinstance(key, tuple):
            continue
        key_means[key] = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))

    # ── Step 2: build none reference per kmer ──────────────────────────────
    none_ref: dict[int, tuple[float, float]] = {}
    for key, (m_ipd, m_pw) in key_means.items():
        if key[1] == 0:
            none_ref[key[0]] = (m_ipd, m_pw)

    log.info("Binarization: %d none reference kmers", len(none_ref))

    # ── Step 3: classify each meth key by ratio thresholds ─────────────────
    keys_keep:   set = set()
    keys_reject: set = set()
    # Per-type counters: meth_id → {kept, rejected, no_ref}
    type_stats: dict[int, dict] = defaultdict(lambda: {
        'kept': 0, 'rejected': 0, 'no_ref': 0,
        'kept_ipd_ratios': [], 'kept_pw_ratios': [],
    })

    for key, (m_ipd, m_pw) in key_means.items():
        kmer_id, meth_id = key
        if meth_id == 0:
            continue

        ts = type_stats[meth_id]

        # No none reference → reject
        if kmer_id not in none_ref:
            keys_reject.add(key)
            ts['no_ref'] += 1
            continue

        ref_ipd, ref_pw = none_ref[kmer_id]
        ipd_ratio = m_ipd / max(ref_ipd, 1e-9)
        pw_ratio  = m_pw  / max(ref_pw,  1e-9)

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
        ts    = type_stats[meth_id]
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

    # ── Step 4: build new result + per-kmer sample split ───────────────────
    new_result: dict  = {}
    none_extras: dict = {}
    n_kept_keys     = 0
    n_rejected_keys = 0
    n_sample_split  = 0
    n_kept_whole    = 0

    for key, arr in result.items():
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
                low_arr[:, 3 + KMER_PRED_IDX] = 0
            none_extras.setdefault((kmer_id, 0), []).append(low_arr)
            n_rejected_keys += 1
            continue

        # Accepted — try per-kmer read-level split for stoichiometry
        n_samples = len(arr)
        if n_samples < MIN_SAMPLES_SPLIT:
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
        if ratio < 1.3:
            kept = arr.copy()
            kept[:, 2] = 1.0
            new_result[key] = kept
            n_kept_keys += 1
            n_kept_whole += 1
            continue

        labels    = gmm.predict(ipd_pw)
        high_mask = labels == hi
        n_high    = int(high_mask.sum())

        if n_high >= MIN_SPLIT_SURVIVORS:
            high_arr = arr[high_mask].copy()
            high_arr[:, 2] = 1.0
            new_result[key] = high_arr
            n_kept_keys += 1
            n_sample_split += 1

            low_arr = arr[~high_mask].copy()
            low_arr[:, 2] = 0.0
            if low_arr.shape[1] >= 14:
                low_arr[:, 3 + KMER_PRED_IDX] = 0
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
            new_result[none_key] = np.concatenate([new_result[none_key], extra], axis=0)
        else:
            new_result[none_key] = extra

    log.info(
        "Binarization complete: %d meth keys kept (%d sample-split, %d kept-whole), "
        "%d rejected as false positive",
        n_kept_keys, n_sample_split, n_kept_whole, n_rejected_keys,
    )
    return new_result


# ---------------------------------------------------------------------------
# Extract: raw samples from one BAM file
# ---------------------------------------------------------------------------

def extract_samples_from_bam(
    bam_path: str,
    motif_string: str,
    max_samples_per_key: int = 10_000,
    revcomp: bool = True,
    use_reverse_strand: bool = True,
    max_reads: int = 0,
    kmer_size: int = K,
    unmeth_subsample_rate: float = 0.05,
    binarize: bool = True,
    meth_types: set[str] | None = None,
) -> dict:
    """Extract raw (IPD, PW) pairs from a BAM file for each k-mer context.

    For each read: extract sequence + fi/fp kinetic tags, scan methylation
    motifs, then slide a kmer_size-mer window collecting raw signal values.

    When ``use_reverse_strand=True`` (default) and the BAM contains ``ri``/``rp``
    complementary-strand kinetic tags, a second extraction pass processes the
    reverse strand.  For position *i* in the read, the reverse-strand kinetic
    signal ``ri[i]`` was measured as the polymerase traversed the complementary
    strand in its 5'→3' direction.  The correct sequence context for that signal
    is ``RC(seq[i-mid:i+mid+1])`` — not the forward k-mer.  Using RC kmers with
    the complementary-strand IPD/PW values effectively doubles the training set
    and makes the model strand-invariant.

    Reservoir sampling keeps memory bounded: once a (kmer, meth_id) key
    reaches max_samples_per_key, new samples randomly replace existing ones
    with probability max_samples_per_key / n_seen, giving an unbiased sample.

    Args:
        bam_path:             Path to BAM file with fi/fp kinetic tags.
        motif_string:         Semicolon-delimited motif string (e.g. "m6A,GATC,1").
        max_samples_per_key:  Maximum samples stored per (kmer, meth_id) key.
        revcomp:              Include reverse complement motif patterns (default True).
        use_reverse_strand:   Also extract ri/rp complementary-strand kinetics
                              using RC(k-mer) as the key.  Silently skipped for
                              reads or BAMs that lack ri/rp tags.
        max_reads:            Stop after this many reads (0 = no limit).
                              For smoke-testing only — reservoir sampling is biased
                              when the BAM is not fully read.
        kmer_size:            K-mer window size (default K=11). Must be odd.
        unmeth_subsample_rate: Fraction of unmethylated (meth_id=0) positions to
                              keep (default 0.05 = 5%). Methylated positions are
                              always kept. Controls the unmeth/meth balance in the
                              raw shard and prevents the dict from being dominated
                              by the ~4M possible unmethylated k-mers.

    Returns:
        dict with:
          - tuple keys ``(kmer_id, meth_id)`` → ``np.ndarray(N, 14)``
            columns: [IPD, PW, fraction, mc_0..mc_10]
          - ``"__meta__"``                     → dict with provenance metadata
    """
    kinetic_tag = validate_bam_kinetics(bam_path)
    ipd_tag = kinetic_tag                              # "fi" or "ip"
    pw_tag  = "fp" if kinetic_tag == "fi" else "pw"
    log.info("Using kinetic tags: %s/%s", ipd_tag, pw_tag)

    # Apply mod-type filter upstream: excluded motifs will never match,
    # so positions carrying those mod types stay unlabelled (= meth_id 0).
    # This matches the GFF-path behaviour where excluded positions are skipped.
    if meth_types is not None:
        motif_string = filter_motif_string_by_types(motif_string, meth_types)

    _mask  = kmer_mask(kmer_size)
    # Asymmetric kmer: prediction position is at index KMER_PRED_IDX (= 7)
    # within the 11-mer. The right edge of the kmer is at offset KMER_RIGHT_PAD
    # (= 3) from the prediction position. So when sliding through the read
    # with `i` as the right edge of the current kmer:
    #   center = i - KMER_RIGHT_PAD   (= i - 3)
    pred_off = KMER_RIGHT_PAD
    motifs = parse_motifs(motif_string, revcomp=revcomp)
    frac_lookup = _build_fraction_lookup(motif_string)

    samples: dict = defaultdict(list)
    counts:  dict = defaultdict(int)   # total observations seen per key
    n_reads_processed    = 0
    n_reads_with_reverse = 0

    log.info("Extracting from: %s", bam_path)
    log.info("Motifs: %s  |  reverse_strand=%s", motif_string, use_reverse_strand)

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if max_reads > 0 and n_reads_processed >= max_reads:
                log.info("--max-reads %d reached — stopping early (smoke test only)", max_reads)
                break
            seq = read.query_sequence
            if not (seq and len(seq) >= kmer_size and read.has_tag(ipd_tag)):
                continue

            ipds    = read.get_tag(ipd_tag)
            pws     = read.get_tag(pw_tag)
            min_len = min(len(seq), len(ipds), len(pws))

            # Per-read regex scan for methylation positions (forward strand).
            meth_status = scan_sequence(seq[:min_len], motifs)

            # --- Forward strand: slide kmer_size window, collect fi/fp ---
            current_kmer = 0
            for i in range(min_len):
                base_val     = BASE_MAP.get(seq[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & _mask

                if i >= kmer_size - 1:
                    center  = i - pred_off
                    meth_id = int(meth_status[center])
                    # Subsample unmethylated positions to control dict size
                    if meth_id == 0 and np.random.random() >= unmeth_subsample_rate:
                        continue
                    key     = (current_kmer, meth_id)
                    ipd_val = float(ipds[center])
                    pw_val  = float(pws[center])
                    frac    = frac_lookup.get(meth_id, 0.0)

                    # 11-position asymmetric methylation context [-8, +2] around
                    # `center`. All kinetic signatures (m6A +5, m5C +2 +6) are
                    # downstream of the modified base, so to predict the IPD at
                    # `center` the model needs to see modifications UPSTREAM of it.
                    mc = _slice_meth_context(meth_status, center)
                    # Kinetic profile aval [0, +8] from center: used by refine
                    # to validate that the signature pattern (e.g. m5C at +2/+6)
                    # is actually present in the sample's kinetic neighbourhood.
                    profile = _slice_kinetic_profile(ipds, pws, center)
                    row = [ipd_val, pw_val, frac] + mc + profile

                    counts[key] += 1
                    n = counts[key]
                    if n <= max_samples_per_key:
                        samples[key].append(row)
                    else:
                        # Reservoir sampling: replace a random existing entry
                        j = np.random.randint(0, n)
                        if j < max_samples_per_key:
                            samples[key][j] = row

            # --- Reverse strand: slide RC 11-mer window, collect ri/rp ---
            #
            # ri[i] is the IPD of the polymerase reading the complementary
            # strand at the position paired with seq[i], moving 5'→3' on the
            # complement.  Each strand is read independently so we apply the
            # same asymmetric kmer convention (-7, +3 from prediction pos)
            # to rc_seq — the polymerase's own reading direction.
            #
            # Implementation: slide an 11-mer window through rc_seq (the reverse
            # complement of the read).  At window position j in rc_seq, the
            # rc_center is at j - pred_off in rc_seq coords, which corresponds
            # to forward position fwd_center = min_rev_len - 1 - rc_center.
            # ri_tags[fwd_center] gives the complementary-strand IPD at that site.
            if use_reverse_strand and read.has_tag("ri") and read.has_tag("rp"):
                ri_tags = read.get_tag("ri")
                rp_tags = read.get_tag("rp")
                min_rev_len = min(min_len, len(ri_tags), len(rp_tags))

                if min_rev_len >= kmer_size:
                    n_reads_with_reverse += 1
                    rc_seq          = reverse_complement(seq[:min_rev_len])
                    rev_meth_status = scan_sequence(rc_seq, motifs)

                    rc_kmer = 0
                    for j in range(min_rev_len):
                        rc_base = BASE_MAP.get(rc_seq[j], 0)
                        rc_kmer = ((rc_kmer << 2) | rc_base) & _mask

                        if j >= kmer_size - 1:
                            rc_center  = j - pred_off
                            fwd_center = min_rev_len - 1 - rc_center

                            rc_meth_id = int(rev_meth_status[rc_center])
                            if rc_meth_id == 0 and np.random.random() >= unmeth_subsample_rate:
                                continue
                            rc_key = (rc_kmer, rc_meth_id)
                            ri_val = float(ri_tags[fwd_center])
                            rp_val = float(rp_tags[fwd_center])
                            frac   = frac_lookup.get(rc_meth_id, 0.0)

                            # 11-position asymmetric meth context [-8, +2] for RC window
                            rev_mc = _slice_meth_context(rev_meth_status, rc_center)
                            # Kinetic profile aval [0, +8] on the complementary strand
                            profile = _slice_kinetic_profile(ri_tags, rp_tags, fwd_center)
                            row = [ri_val, rp_val, frac] + rev_mc + profile

                            counts[rc_key] += 1
                            n = counts[rc_key]
                            if n <= max_samples_per_key:
                                samples[rc_key].append(row)
                            else:
                                j2 = np.random.randint(0, n)
                                if j2 < max_samples_per_key:
                                    samples[rc_key][j2] = row

            n_reads_processed += 1

    if use_reverse_strand and n_reads_with_reverse == 0:
        # If the BAM uses ip/pw tags (post-bystrandify or post-pbmm2 alignment),
        # ri/rp do not exist as separate tags — the complementary strand is in
        # a separate read of its own.  Don't alarm the user; just note the path.
        if kinetic_tag == "ip":
            log.info(
                "No ri/rp in %s (expected — ip/pw BAM, complementary "
                "strand is in its own read).", bam_path,
            )
        else:
            log.warning(
                "No ri/rp tags found in %s — reverse strand extraction skipped. "
                "BAM uses fi/fp but lacks ri/rp; complementary-strand kinetics "
                "may be missing.",
                bam_path,
            )

    n_keys    = len(samples)
    n_samples = sum(len(v) for v in samples.values())
    log.info(
        "Done: %d reads (%d with reverse strand) → %d unique (kmer, meth) keys, "
        "%d total samples",
        n_reads_processed, n_reads_with_reverse, n_keys, n_samples,
    )

    result = {key: np.array(vals, dtype=np.float32) for key, vals in samples.items()}

    # Binarize methylated keys: 2-component GMM on (IPD, PW) separates
    # truly methylated reads from unmethylated reads at motif sites.
    if binarize:
        result = _binarize_by_ipd(result)
        b_keys    = sum(1 for k in result if isinstance(k, tuple))
        b_samples = sum(len(v) for k, v in result.items() if isinstance(k, tuple))
        log.info("After IPD binarization: %d unique keys, %d total samples", b_keys, b_samples)
    else:
        log.info("Binarization skipped (--no-binarize): keeping raw methylation labels")

    # Attach provenance metadata so shards can be inspected and traced back.
    result["__meta__"] = {
        "kinsim_version":          _KINSIM_VERSION,
        "extraction_mode":         "motif",
        "source_bam":              str(bam_path),
        "motifs":                  motif_string,
        "meth_types":              sorted(meth_types) if meth_types else None,
        "kmer_size":               kmer_size,
        "unmeth_subsample_rate":   unmeth_subsample_rate,
        "use_reverse_strand":      use_reverse_strand,
        "max_samples_per_key":     max_samples_per_key,
        "n_reads_processed":       n_reads_processed,
        "n_reads_with_reverse":    n_reads_with_reverse,
        "n_unique_keys":           n_keys,
        "n_total_samples":         n_samples,
        "created":                 datetime.datetime.now().isoformat(timespec="seconds"),
    }

    return result


# ---------------------------------------------------------------------------
# GFF-based extraction: aligned BAM + ipdSummary GFF annotations
# ---------------------------------------------------------------------------

def extract_from_aligned_bam(
    bam_path: str,
    gff_path: str,
    max_samples_per_key: int = 10_000,
    max_reads: int = 0,
    kmer_size: int = K,
    unmeth_subsample_rate: float = 0.05,
    min_score: float = 20.0,
    min_ipd_ratio: float = 0.0,
    meth_types: set[str] | None = None,
) -> dict:
    """Extract raw (IPD, PW) pairs using GFF annotations for methylation labels.

    Instead of scanning motif patterns in the sequence (which is deterministic
    and produces the same label for the same kmer), this uses ipdSummary GFF
    output to label each genomic position based on kinetic signal analysis.

    This approach is:
    - **More accurate**: labels come from the SP3-C3 chemistry model, not
      sequence pattern matching
    - **Generalizable**: any modification type in the GFF is supported,
      including future types beyond m6A/m4C/m5C
    - **Read-level**: each read position is labelled independently based
      on its alignment to the reference

    Requires an aligned BAM (reads mapped to a reference). Each read's
    CIGAR is used to map read positions to reference coordinates, which
    are then looked up in the GFF annotation map.

    No binarization step is needed — the GFF already provides the ground
    truth labels from ipdSummary's statistical analysis.

    Args:
        bam_path:              Path to aligned BAM with fi/fp kinetic tags.
        gff_path:              Path to ipdSummary GFF3 output.
        max_samples_per_key:   Reservoir sampling cap per (kmer, meth_id).
        max_reads:             Stop after N reads (0 = no limit).
        kmer_size:             K-mer window size (default 11).
        unmeth_subsample_rate: Fraction of unmethylated positions to keep.
        min_score:             Minimum GFF score for a position to be
                               considered methylated (default 20).
        min_ipd_ratio:         Minimum IPD ratio in GFF (0 = no filter).

    Returns:
        dict with (kmer_id, meth_id) → np.ndarray(N, 3) [IPD, PW, fraction]
        and "__meta__" provenance key.
    """
    from .utils.io import load_gff_annotations

    kinetic_tag = validate_bam_kinetics(bam_path)
    ipd_tag = kinetic_tag                              # "fi" or "ip"
    pw_tag  = "fp" if kinetic_tag == "fi" else "pw"
    log.info("Using kinetic tags: %s/%s", ipd_tag, pw_tag)

    _mask = kmer_mask(kmer_size)
    pred_off = KMER_RIGHT_PAD     # asymmetric kmer: prediction at i - 3

    # Load GFF annotations (with optional mod-type filter).  Excluded types
    # are SKIPPED during load, so those genomic positions look like "no
    # annotation" downstream and end up labelled meth_id=0 (unmeth) in the
    # per-read map.  This keeps the unmeth class consistent with the filter.
    annotations = load_gff_annotations(
        gff_path, min_score=min_score, min_ipd_ratio=min_ipd_ratio,
        allowed_mods=meth_types,
    )
    if not annotations:
        log.error("No methylation annotations found in GFF: %s", gff_path)
        sys.exit(1)

    samples: dict = defaultdict(list)
    counts:  dict = defaultdict(int)
    n_reads_processed = 0
    n_mapped = 0
    n_meth_hits = 0

    log.info("Extracting (GFF mode) from: %s", bam_path)
    log.info("GFF annotations: %d positions", len(annotations))

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam:
            if max_reads > 0 and n_reads_processed >= max_reads:
                break

            # Skip unmapped, secondary, supplementary
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            seq = read.query_sequence
            if not (seq and len(seq) >= kmer_size and read.has_tag(ipd_tag)):
                continue

            ipds = read.get_tag(ipd_tag)
            pws  = read.get_tag(pw_tag)
            min_len = min(len(seq), len(ipds), len(pws))

            # Build per-base meth_id array from alignment + GFF
            contig = read.reference_name
            strand = '-' if read.is_reverse else '+'

            # get_aligned_pairs gives (query_pos, ref_pos) for each alignment column
            aligned_pairs = read.get_aligned_pairs(matches_only=True)

            # Build a query_pos → meth_id map from the alignment
            pos_meth: dict[int, int] = {}
            for query_pos, ref_pos in aligned_pairs:
                if query_pos is None or ref_pos is None:
                    continue
                if query_pos >= min_len:
                    continue
                key = (contig, ref_pos, strand)
                if key in annotations:
                    pos_meth[query_pos] = annotations[key]
                    n_meth_hits += 1

            n_mapped += 1

            # Slide kmer window
            current_kmer = 0
            for i in range(min_len):
                base_val = BASE_MAP.get(seq[i], -1)
                if base_val < 0:
                    current_kmer = 0
                    continue
                current_kmer = ((current_kmer << 2) | base_val) & _mask

                if i >= kmer_size - 1:
                    center = i - pred_off
                    meth_id = pos_meth.get(center, 0)

                    # Subsample unmethylated
                    if meth_id == 0 and np.random.random() >= unmeth_subsample_rate:
                        continue

                    key = (current_kmer, meth_id)
                    ipd_val = float(ipds[center])
                    pw_val  = float(pws[center])
                    # fraction = 1.0 for GFF-labelled positions (ipdSummary
                    # already decided this position is methylated)
                    frac = 1.0 if meth_id > 0 else 0.0

                    # Build the per-sample meth context from pos_meth (asymmetric).
                    mc = [0] * METH_CTX_LEN
                    for k in range(METH_CTX_LEN):
                        mc_pos = center - METH_CTX_LEFT + k
                        if 0 <= mc_pos < min_len:
                            mc[k] = pos_meth.get(mc_pos, 0)
                    # Kinetic profile aval
                    profile = _slice_kinetic_profile(ipds, pws, center)
                    row = [ipd_val, pw_val, frac] + mc + profile

                    counts[key] += 1
                    n = counts[key]
                    if n <= max_samples_per_key:
                        samples[key].append(row)
                    else:
                        j = np.random.randint(0, n)
                        if j < max_samples_per_key:
                            samples[key][j] = row

            n_reads_processed += 1
            if n_reads_processed % 5000 == 0:
                log.info("  %d reads processed (%d meth hits so far)...",
                         n_reads_processed, n_meth_hits)

    n_keys    = len(samples)
    n_samples = sum(len(v) for v in samples.values())
    log.info(
        "Done: %d reads mapped → %d unique (kmer, meth) keys, %d total samples, "
        "%d methylation hits",
        n_mapped, n_keys, n_samples, n_meth_hits,
    )

    # Count per meth type
    meth_names = {v: k for k, v in METH_IDS.items()}
    type_counts: dict[int, int] = defaultdict(int)
    for (_, m_id), vals in samples.items():
        type_counts[m_id] += len(vals)
    for m_id in sorted(type_counts):
        name = meth_names.get(m_id, f"type{m_id}")
        log.info("  %s: %d samples across %d keys",
                 name, type_counts[m_id],
                 sum(1 for k in samples if k[1] == m_id))

    result = {key: np.array(vals, dtype=np.float32) for key, vals in samples.items()}

    # No binarization needed — GFF labels are the ground truth from ipdSummary.

    result["__meta__"] = {
        "kinsim_version":         _KINSIM_VERSION,
        "extraction_mode":        "gff",
        "source_bam":             str(bam_path),
        "gff_path":               str(gff_path),
        "meth_types":             sorted(meth_types) if meth_types else None,
        "min_score":              min_score,
        "min_ipd_ratio":          min_ipd_ratio,
        "kmer_size":              kmer_size,
        "unmeth_subsample_rate":  unmeth_subsample_rate,
        "max_samples_per_key":    max_samples_per_key,
        "n_reads_processed":      n_reads_processed,
        "n_reads_mapped":         n_mapped,
        "n_meth_hits":            n_meth_hits,
        "n_unique_keys":          n_keys,
        "n_total_samples":        n_samples,
        "created":                datetime.datetime.now().isoformat(timespec="seconds"),
    }

    return result


# ---------------------------------------------------------------------------
# Merge: combine shards from multiple BAMs
# ---------------------------------------------------------------------------

def merge_shards(
    input_dir: str,
    output_file: str,
    max_samples_per_key: int = 50_000,
    glob_pattern: str = "auto",
) -> None:
    """Merge multiple shard pickle files into one master training set.

    Looks for shard files in input_dir using the following precedence:
      1. ``*_shard.pkl`` (produced by ``kinsim extract --manifest``)
      2. ``*_cgan.pkl``  (legacy naming, kept for backward compat)

    Override with ``glob_pattern`` to use a custom pattern.

    After concatenation, keys exceeding max_samples_per_key are randomly
    subsampled to keep the master file manageable.

    The ``"__meta__"`` key (provenance) is merged across all shards and stored
    in the output.

    Args:
        input_dir:           Directory containing shard .pkl files.
        output_file:         Path for the merged output .pkl file.
        max_samples_per_key: Maximum samples to keep per (kmer, meth_id).
        glob_pattern:        Glob pattern for shard files; "auto" tries
                             ``*_shard.pkl`` then ``*_cgan.pkl``.
    """
    import glob as _glob

    if glob_pattern == "auto":
        files = _glob.glob(os.path.join(input_dir, "*_shard.pkl"))
        if not files:
            files = _glob.glob(os.path.join(input_dir, "*_cgan.pkl"))
        if not files:
            files = _glob.glob(os.path.join(input_dir, "*.pkl"))
            # Exclude the output file itself to avoid self-merging
            files = [f for f in files if os.path.abspath(f) != os.path.abspath(output_file)]
    else:
        files = _glob.glob(os.path.join(input_dir, glob_pattern))

    if not files:
        log.error("No shard .pkl files found in %s", input_dir)
        sys.exit(1)

    files = sorted(files)
    log.info("Merging %d shards from: %s", len(files), input_dir)

    master: dict = defaultdict(list)
    shard_metas: list = []

    for f_path in files:
        log.info("  Loading shard: %s", os.path.basename(f_path))
        with open(f_path, "rb") as f:
            shard = pickle.load(f)

        # Collect and skip the metadata key
        if "__meta__" in shard:
            shard_metas.append(shard.pop("__meta__"))

        for key, arr in shard.items():
            if not isinstance(key, tuple):
                continue   # skip any other non-data keys
            master[key].append(arr)

    result = {}
    n_subsampled = 0
    for key, arrays in master.items():
        combined = np.concatenate(arrays, axis=0)
        if len(combined) > max_samples_per_key:
            idx      = np.random.choice(len(combined), max_samples_per_key, replace=False)
            combined = combined[idx]
            n_subsampled += 1
        result[key] = combined

    # Consolidate meth_types across shards — all shards must agree on the
    # active alphabet, otherwise training labels are inconsistent.
    shard_meth_types = [
        tuple(m["meth_types"]) if m.get("meth_types") else None
        for m in shard_metas
    ]
    unique_meth_types = set(shard_meth_types)
    if len(unique_meth_types) > 1:
        log.warning("Shards were extracted with different --meth-types: %s. "
                    "Training on this merged file will mix alphabets.",
                    unique_meth_types)
    merged_meth_types = (sorted(shard_meth_types[0]) if shard_meth_types and
                         shard_meth_types[0] is not None else None)

    # Merged metadata
    result["__meta__"] = {
        "kinsim_version":     _KINSIM_VERSION,
        "merged_from":        [m.get("source_bam", "?") for m in shard_metas],
        "meth_types":         merged_meth_types,
        "n_shards":           len(files),
        "max_samples_per_key": max_samples_per_key,
        "created":            datetime.datetime.now().isoformat(timespec="seconds"),
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(result, f)

    total_keys    = len(result) - 1   # exclude __meta__
    total_samples = sum(len(v) for k, v in result.items() if isinstance(k, tuple))
    log.info("Master dataset saved: %s", output_file)
    log.info(
        "  %d unique contexts, %d total samples (%d keys subsampled to cap=%d)",
        total_keys, total_samples, n_subsampled, max_samples_per_key,
    )


# ---------------------------------------------------------------------------
# Manifest-mode extraction helper
# ---------------------------------------------------------------------------

def extract_from_manifest_task(
    manifest_path: str,
    task_index: int,
    output_dir: str,
    max_samples_per_key: int = 10_000,
    revcomp: bool = True,
    use_reverse_strand: bool = True,
    max_reads: int = 0,
    kmer_size: int = K,
    unmeth_subsample_rate: float = 0.05,
    min_score: float = 20.0,
    min_ipd_ratio: float = 0.0,
    binarize: bool = True,
    meth_types: set[str] | None = None,
) -> None:
    """Extract one BAM from a manifest CSV (for SLURM array jobs).

    Reads the manifest at ``manifest_path``, picks the row at ``task_index``
    (1-based, matching SLURM_ARRAY_TASK_ID), runs extraction, and writes the
    shard to ``output_dir/<sample_id>_shard.pkl``.

    When the manifest row has a non-empty ``gff`` column, GFF-based extraction
    is used (``extract_from_aligned_bam``).  Otherwise, motif-based extraction
    is used (``extract_samples_from_bam``).

    Args:
        manifest_path:        Path to the manifest CSV.
        task_index:           1-based row index (SLURM_ARRAY_TASK_ID).
        output_dir:           Directory for the output shard .pkl.
        max_samples_per_key:  Reservoir cap per (kmer, meth_id) key.
        revcomp:              Scan reverse complement strand for motifs.
        use_reverse_strand:   Extract ri/rp complementary-strand kinetics.
        max_reads:            Stop after N reads (0 = no limit, smoke test only).
        min_score:            Minimum GFF score (only used in GFF mode).
        min_ipd_ratio:        Minimum IPD ratio filter (only used in GFF mode).
    """
    from .utils.config import load_manifest
    from .utils.motifs import load_motif_string as _load_motif_string

    entries = load_manifest(manifest_path)

    if task_index < 1 or task_index > len(entries):
        log.error(
            "Task index %d is out of range (manifest has %d entries, 1-indexed).",
            task_index, len(entries),
        )
        sys.exit(1)

    entry = entries[task_index - 1]
    log.info("Task %d/%d: %s", task_index, len(entries), entry.sample_id)
    log.info("  BAM:    %s", entry.bam_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_pkl = os.path.join(output_dir, f"{entry.sample_id}_shard.pkl")
    log.info("  Output: %s", output_pkl)

    if meth_types is not None:
        log.info("  Meth types filter: %s", sorted(meth_types))

    if entry.gff:
        # ---- GFF-based extraction ----
        log.info("  GFF:    %s (GFF mode)", entry.gff)
        result = extract_from_aligned_bam(
            entry.bam_path, entry.gff,
            max_samples_per_key=max_samples_per_key,
            max_reads=max_reads,
            kmer_size=kmer_size,
            unmeth_subsample_rate=unmeth_subsample_rate,
            min_score=min_score,
            min_ipd_ratio=min_ipd_ratio,
            meth_types=meth_types,
        )
    else:
        # ---- Motif-based extraction ----
        log.info("  Motifs: %s", entry.motifs)
        motif_string = _load_motif_string(entry.motifs)
        if not motif_string:
            log.warning("No motifs resolved for sample '%s' -- SKIPPING.", entry.sample_id)
            return

        result = extract_samples_from_bam(
            entry.bam_path, motif_string,
            max_samples_per_key=max_samples_per_key,
            revcomp=revcomp,
            use_reverse_strand=use_reverse_strand,
            max_reads=max_reads,
            kmer_size=kmer_size,
            unmeth_subsample_rate=unmeth_subsample_rate,
            binarize=binarize,
            meth_types=meth_types,
        )

    with open(output_pkl, "wb") as f:
        pickle.dump(result, f)

    meta = result.get("__meta__", {})
    log.info(
        "Shard saved: %s (%d keys, %d samples)",
        output_pkl,
        meta.get("n_unique_keys", "?"),
        meta.get("n_total_samples", "?"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse
    from .utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="kinsim data",
        description=(
            "Extract raw (IPD, PW) training samples from BAM files, or merge\n"
            "multiple shards into a master training set.\n\n"
            "The output is consumed by:\n"
            "  kinsim train --model mlp  master_data.pkl  checkpoints_mlp/\n\n"
            "Single-BAM extract (simple/testing):\n"
            "  kinsim extract reads.bam \"m6A,GATC,1\" shard.pkl\n\n"
            "Manifest-based extract (recommended for SLURM array jobs):\n"
            "  kinsim extract --manifest manifest.csv --task $SLURM_ARRAY_TASK_ID \\\n"
            "                 --output-dir shards/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- extract subcommand --
    p_extract = sub.add_parser(
        "extract",
        help="Extract raw (IPD, PW) samples from a BAM file (or manifest)",
        description=(
            "Collect individual IPD/PW observations per (11-mer, methylation_state).\n\n"
            "Single-BAM mode:  kinsim extract reads.bam MOTIFS shard.pkl\n"
            "Manifest mode:    kinsim extract --manifest manifest.csv --task N "
            "--output-dir shards/"
        ),
    )
    # Single-BAM positional args (optional — not required when --manifest is used)
    p_extract.add_argument("bam",    nargs="?", default=None,
                           help="Input BAM file with fi/fp kinetic tags")
    p_extract.add_argument("motifs", nargs="?", default=None,
                           help="Motif source: KinSim string ('m6A,GATC,1'), "
                                "PacBio motifs.csv, or REBASE file (auto-detected)")
    p_extract.add_argument("output", nargs="?", default=None,
                           help="Output .pkl shard file (single-BAM mode)")

    # Manifest mode args
    p_extract.add_argument("--manifest",   default=None,
                           help="Manifest CSV with columns: sample_id, bam_path, motifs")
    p_extract.add_argument("--task",       type=int, default=None,
                           help="1-based row index from the manifest (= SLURM_ARRAY_TASK_ID)")
    p_extract.add_argument("--output-dir", default=None,
                           help="Output directory for shard .pkl files (manifest mode)")

    # GFF-based extraction (aligned BAM + ipdSummary GFF)
    # Usage: kinsim extract <bam> --gff <gff> -o <output.pkl>
    p_extract.add_argument("--gff", default=None,
                           help="Path to ipdSummary GFF3 file. Enables GFF-based "
                                "extraction: methylation labels come from GFF annotations "
                                "instead of motif sequence scanning. Requires an aligned "
                                "BAM. Use -o/--output for the output path.")
    p_extract.add_argument("-o", "--output", dest="output_file", default=None,
                           help="Output .pkl shard file (used with --gff mode). "
                                "In motif mode, use the positional 'output' arg instead.")
    p_extract.add_argument("--min-score", type=float, default=20.0,
                           help="Minimum GFF score for methylation calls "
                                "(default: 20, i.e. p < 0.01). Only used with --gff.")
    p_extract.add_argument("--min-ipd-ratio", type=float, default=0.0,
                           help="Minimum IPD ratio filter for GFF records "
                                "(default: 0 = no filter). Only used with --gff.")

    # Common options
    p_extract.add_argument("--max-samples", type=int, default=20_000,
                           help="Max samples per (kmer, meth_id) via reservoir "
                                "sampling (default: 20000)")
    p_extract.add_argument("--no-revcomp", action="store_true",
                           help="Do not scan reverse complement strand for motifs")
    p_extract.add_argument("--no-reverse-strand", action="store_true",
                           help="Do not extract ri/rp complementary-strand kinetics. "
                                "By default, both fi/fp (forward) and ri/rp (reverse) "
                                "are extracted; the reverse-strand samples use "
                                "RC(11-mer) as the key, doubling training data and "
                                "making the model strand-invariant.")
    p_extract.add_argument("--min-fraction", type=float, default=0.40,
                           help="Minimum fraction threshold for PacBio CSV (default: 0.40)")
    p_extract.add_argument("--min-detected", type=int, default=20,
                           help="Minimum nDetected threshold for PacBio CSV (default: 20)")
    p_extract.add_argument("--kmer-size", type=int, default=None,
                           help="K-mer window size (default: from encoding.py K=11). "
                                "Must match the value used during training.")
    p_extract.add_argument("--unmeth-subsample-rate", type=float, default=0.05,
                           help="Fraction of unmethylated positions to keep (default: 0.05). "
                                "Methylated positions are always kept. Prevents the dict "
                                "from being dominated by ~4M unmethylated k-mers.")
    p_extract.add_argument("--max-reads", type=int, default=0,
                           help="Stop after N reads (0 = no limit). "
                                "Smoke-test only — biases reservoir sampling.")
    p_extract.add_argument("--no-binarize", action="store_true",
                           help="Skip GMM binarization of methylated keys. "
                                "Keeps raw motif-based labels so distributions "
                                "can be inspected before filtering.")
    p_extract.add_argument("--meth-types", default=None,
                           help="Comma-separated list of methylation types to "
                                "include (e.g. 'm6A,m4C'). Other types are "
                                "SKIPPED: excluded positions do not appear in "
                                "the training data at all (never relabelled as "
                                "unmeth). Default: accept all types recognised "
                                "by ipdSummary / PacBio motif files. 'all' is "
                                "an explicit synonym for no filter.")
    p_extract.add_argument("--verbose", "-v", action="store_true",
                           help="Enable DEBUG-level logging")

    # -- merge subcommand --
    p_merge = sub.add_parser(
        "merge",
        help="Merge multiple *_shard.pkl shards into one master",
        description=(
            "Concatenate raw sample arrays from all shards in a directory.\n"
            "Detects *_shard.pkl files (also supports legacy *_cgan.pkl naming).\n"
            "Subsamples per key if total exceeds --max-samples."
        ),
    )
    p_merge.add_argument("input_dir",
                         help="Directory containing shard .pkl files")
    p_merge.add_argument("output",
                         help="Output master training set .pkl file")
    p_merge.add_argument("--max-samples", type=int, default=50_000,
                         help="Max samples per (kmer, meth_id) after merging "
                              "(default: 50000)")
    p_merge.add_argument("--glob", default="auto",
                         dest="glob_pattern",
                         help="Glob pattern for shard files (default: auto-detect)")
    p_merge.add_argument("--verbose", "-v", action="store_true",
                         help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False))

    if args.command == "merge":
        merge_shards(
            args.input_dir, args.output,
            max_samples_per_key=args.max_samples,
            glob_pattern=args.glob_pattern,
        )

    else:   # extract
        meth_types = parse_meth_types_arg(args.meth_types)

        if args.manifest:
            # ---- Manifest mode ----
            if args.task is None:
                log.error("--task is required when using --manifest")
                sys.exit(1)
            if args.output_dir is None:
                log.error("--output-dir is required when using --manifest")
                sys.exit(1)
            extract_from_manifest_task(
                args.manifest,
                task_index           = args.task,
                output_dir           = args.output_dir,
                max_samples_per_key  = args.max_samples,
                revcomp              = not args.no_revcomp,
                use_reverse_strand   = not args.no_reverse_strand,
                max_reads            = args.max_reads,
                kmer_size            = args.kmer_size or K,
                unmeth_subsample_rate= args.unmeth_subsample_rate,
                min_score            = args.min_score,
                min_ipd_ratio        = args.min_ipd_ratio,
                binarize             = not args.no_binarize,
                meth_types           = meth_types,
            )

        elif args.gff:
            # ---- GFF-based extraction (aligned BAM + ipdSummary GFF) ----
            if not args.bam:
                log.error(
                    "GFF mode requires: kinsim extract <bam> --gff <gff> -o <output.pkl>\n"
                    "The BAM must be aligned (mapped to the reference used by ipdSummary)."
                )
                sys.exit(1)
            gff_output = args.output_file or args.output
            if not gff_output:
                log.error(
                    "GFF mode requires an output path:\n"
                    "  kinsim extract <bam> --gff <gff> -o <output.pkl>"
                )
                sys.exit(1)

            log.info("GFF-based extraction from: %s", os.path.basename(args.bam))
            if meth_types is not None:
                log.info("Meth types filter: %s", sorted(meth_types))
            result = extract_from_aligned_bam(
                args.bam, args.gff,
                max_samples_per_key=args.max_samples,
                max_reads=args.max_reads,
                kmer_size=args.kmer_size or K,
                unmeth_subsample_rate=args.unmeth_subsample_rate,
                min_score=args.min_score,
                min_ipd_ratio=args.min_ipd_ratio,
                meth_types=meth_types,
            )

            Path(gff_output).parent.mkdir(parents=True, exist_ok=True)
            with open(gff_output, "wb") as f:
                pickle.dump(result, f)

            meta = result.get("__meta__", {})
            log.info(
                "Shard saved: %s (%d contexts, %d samples)",
                gff_output,
                meta.get("n_unique_keys", "?"),
                meta.get("n_total_samples", "?"),
            )

        else:
            # ---- Single-BAM mode (motif-based) ----
            if not args.bam or not args.motifs or not args.output:
                log.error(
                    "Single-BAM mode requires: kinsim extract <bam> <motifs> <output>\n"
                    "Or use --gff mode: kinsim extract <bam> --gff <gff> <output>\n"
                    "Or use manifest mode: kinsim extract --manifest CSV --task N "
                    "--output-dir DIR"
                )
                sys.exit(1)

            motif_string = load_motif_string(
                args.motifs,
                min_fraction=args.min_fraction,
                min_detected=args.min_detected,
            )
            if not motif_string:
                log.error("No motifs found from the provided source.")
                sys.exit(1)

            log.info("Extracting samples from: %s", os.path.basename(args.bam))
            if meth_types is not None:
                log.info("Meth types filter: %s", sorted(meth_types))
            result = extract_samples_from_bam(
                args.bam, motif_string,
                max_samples_per_key=args.max_samples,
                revcomp=not args.no_revcomp,
                use_reverse_strand=not args.no_reverse_strand,
                max_reads=args.max_reads,
                kmer_size=args.kmer_size or K,
                unmeth_subsample_rate=args.unmeth_subsample_rate,
                binarize=not args.no_binarize,
                meth_types=meth_types,
            )

            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "wb") as f:
                pickle.dump(result, f)

            meta = result.get("__meta__", {})
            log.info(
                "Shard saved: %s (%d contexts, %d samples)",
                args.output,
                meta.get("n_unique_keys", "?"),
                meta.get("n_total_samples", "?"),
            )


if __name__ == "__main__":
    main()
