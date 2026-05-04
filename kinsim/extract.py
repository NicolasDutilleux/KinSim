"""Extract raw IPD/PW training samples from BAM files.

Sample-level data preparation for KinSim training. Two output formats
coexist; both store raw uint8/float32 values (the model applies log1p
once at training time):

  v3 (legacy, 35 cols):
      dict[(kmer_id, meth_id)] -> np.ndarray(N, 35)
      Cols: [IPD, PW, fraction, mc_0..mc_10, profile_IPD_0..+8,
             profile_PW_0..+8, rev_meth_-1, rev_meth_0, rev_meth_+1]

  v4 (default, 36 cols):
      dict[kmer_id]            -> np.ndarray(N, 36)
      Cols 0..34 unchanged; col 35 carries CATEGORY (baseline / slowed /
      near_meth — see kinsim.utils.sample_layout).

Both layouts can carry a ``"__meta__"`` string key with provenance.
Dataset loaders auto-detect the format and route accordingly.

CLI — single-BAM mode:
    kinsim extract             reads.bam motifs shard.pkl    # v3 (legacy)
    kinsim extract --v4        reads.bam motifs shard.pkl    # v4 (recommended)

CLI — manifest mode (recommended for SLURM array jobs):
    kinsim extract --v4 --manifest manifest.csv --task 3 --output-dir shards/

Manifest CSV format:
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

from .utils.encoding import BASE_MAP, K, KMER_MASK, METH_IDS, get_meth_ids, kmer_mask
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
        m_id = get_meth_ids().get(parts[0], 0)
        frac = float(parts[4]) if len(parts) >= 5 else 1.0
        fracs[m_id] = frac
    return fracs


# ---------------------------------------------------------------------------
# Methylation-context window (asymmetric, upstream-biased)
# ---------------------------------------------------------------------------

# Per-sample column layout (constants + pure-Python slicing helpers) lives in
# kinsim/utils/sample_layout.py so it is importable without pysam (refine and
# tests rely on it). Re-export the symbols here so the existing names used
# elsewhere in extract.py still resolve.
from .utils.encoding import KMER_LEFT_PAD, KMER_RIGHT_PAD, KMER_PRED_IDX, K
from .utils.sample_layout import (
    METH_CTX_LEFT, METH_CTX_RIGHT, METH_CTX_LEN,
    PROFILE_START, PROFILE_END, PROFILE_LEN,
    REV_METH_OFFSETS, REV_METH_LEN,
    SAMPLE_NCOLS,
    slice_meth_context as _slice_meth_context,
    slice_rev_meth     as _slice_rev_meth,
    slice_kinetic_profile as _slice_kinetic_profile,
)


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

    Supported BAM formats
    ---------------------
    * **Raw HiFi (recommended, fastest)** — single read per molecule with
      ``fi``/``fp`` tags (forward strand) and ``ri``/``rp`` (reverse strand).
      Both forward and reverse paths are used → full kinetic data captured
      in a single pass.
    * **Bystrandified (recommended, modern)** — two reads per molecule, each
      with ``ip``/``pw`` tags (one per strand, in polymerase 5'→3' order).
      Only the forward path is used per read; the complementary strand is
      already a separate read of its own.  Equivalent data volume to raw HiFi.
    * **Aligned post-pbmm2 (NOT supported)** — single read with ``ip``/``pw``
      after alignment.  ``ri``/``rp`` are dropped, and only forward-strand
      kinetics survive — half the training data.  Pass an unaligned (raw HiFi)
      or bystrandified BAM instead.

    For each read: extract sequence + kinetic tags, scan methylation
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

            # Per-read regex scan for methylation positions (this read's strand).
            meth_status = scan_sequence(seq[:min_len], motifs)

            # Complementary-strand methylation in this read's coords:
            # scan rc_seq, then reverse the array so position k matches
            # forward read position k (the partner base of read[k]).
            rc_seq_full = reverse_complement(seq[:min_len])
            meth_status_rc = scan_sequence(rc_seq_full, motifs)
            meth_complement = meth_status_rc[::-1]   # in this read's coords

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
                    # Complementary-strand methylation at -1, 0, +1 from center
                    # (active-site footprint). Captures bilateral methylation
                    # patterns (palindromic R-M Type II sites).
                    rev_meth = _slice_rev_meth(meth_complement, center)
                    row = [ipd_val, pw_val, frac] + mc + profile + rev_meth

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

                            # 11-position asymmetric meth context [-7, +3] for RC window
                            rev_mc = _slice_meth_context(rev_meth_status, rc_center)
                            # Kinetic profile aval [0, +8] on the complementary strand
                            profile = _slice_kinetic_profile(ri_tags, rp_tags, fwd_center)
                            # Complementary-strand methylation in rc_seq coords:
                            # the complementary of rc_seq is the original seq, so
                            # at rc_center we look at meth_status[fwd_center]; the
                            # offsets -1, 0, +1 in rc_seq map to fwd_center +1, 0, -1
                            # in original coords (because rc reverses direction).
                            rev_complement_in_rc = meth_status[::-1]
                            rc_rev_meth = _slice_rev_meth(rev_complement_in_rc, rc_center)
                            row = [ri_val, rp_val, frac] + rev_mc + profile + rc_rev_meth

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
# v4 extract: emits CATEGORY_BASELINE / CATEGORY_SLOWED / CATEGORY_NEAR_METH
# samples in a single pass over each BAM, in the 36-col layout keyed by
# kmer_id only (see kinsim/utils/sample_layout.py).
# ---------------------------------------------------------------------------


def _load_confirmed_set(refined_pkl_path: str) -> tuple[set, dict]:
    """Read an existing master_clean.pkl as a confirmation oracle for
    motif-match positions.

    Returns:
        confirmed_kmer_meth: set of (kmer_id, meth_id) tuples that survived
            the GMM step of a previous v3 refine. Used to gate which motif
            matches in the current extract are treated as candidate methylations.
        sig_offsets_by_name: dict[mod_name -> list[int]] of signature offsets
            taken from kinsim_config.yaml.

    Optional path: most users call extract without --refined-pkl and the
    refine GMM downstream filters false positives.
    """
    with open(refined_pkl_path, "rb") as f:
        data = pickle.load(f)
    confirmed: set = set()
    for k in data:
        if isinstance(k, tuple) and len(k) == 2:
            kid, mid = k
            if int(mid) > 0:
                confirmed.add((int(kid), int(mid)))

    from .utils.config import load_kinsim_config
    cfg = load_kinsim_config()
    sig_offsets_by_name: dict = {}
    for mname, scfg in (cfg.get("kinetic_signatures") or {}).items():
        offs = []
        for k in scfg.get("signal_offsets", []):
            try:
                offs.append(int(k))
            except (TypeError, ValueError):
                continue
        sig_offsets_by_name[mname] = offs
    return confirmed, sig_offsets_by_name


def _slide_kmers(seq: str, kmer_size: int) -> list:
    """Per-position kmer encoding (rolling bit-pack).

    Returns a list of length len(seq); element c is (kmer_id, valid) where
    kmer_id is the 2k-bit encoding of seq[c - KMER_LEFT_PAD : c + KMER_RIGHT_PAD + 1].
    Positions whose window falls outside the read are (0, False).

    Kept for tests / single-position lookups; the production v4 extract uses
    the vectorised _vec_kmers nested in extract_samples_v4_from_bam instead.
    """
    n = len(seq)
    out = [(0, False)] * n
    mask = kmer_mask(kmer_size)
    cur = 0
    init_count = 0
    for i in range(n):
        cur = ((cur << 2) | BASE_MAP.get(seq[i], 0)) & mask
        init_count += 1
        if init_count >= kmer_size:
            center = i - KMER_RIGHT_PAD
            if 0 <= center < n:
                out[center] = (cur, True)
    return out


def extract_samples_v4_from_bam(
    bam_path:                  str,
    motif_string:              str,
    refined_pkl_path:          str | None = None,
    n_baseline_per_kmer:       int  = 50,
    baseline_min_dist_to_meth: int  = K,
    near_meth_max_dist:        int  = 7,
    baseline_sample_rate:      float = 0.10,
    revcomp:                   bool = True,
    use_reverse_strand:        bool = True,
    max_reads:                 int  = 0,
    kmer_size:                 int  = K,
    meth_types:                set | None = None,
    seed:                      int  = 42,
) -> dict:
    """Extract v4 training samples from a BAM in a single pass.

    For each motif-match position p of type T identified by the regex scan:
      - p + k for k in signature_offsets[T] (incl. 0) → CATEGORY_SLOWED
        (position where the polymerase slowing is biophysically expected)
      - p + k for k in [0, near_meth_max_dist] not in sig_offsets[T] →
        CATEGORY_NEAR_METH (close to the meth but not at a signature
        offset; serves as negative control teaching the model that a
        methylation in the meth_context window alone is not enough to
        predict elevated IPD — only the offset matters)
    Positions at distance ≥ baseline_min_dist_to_meth from any meth or
    flag → CATEGORY_BASELINE candidates, capped per kmer at
    n_baseline_per_kmer via streaming reservoir sampling.

    A front-end Bernoulli sample at rate `baseline_sample_rate` skips most
    baseline candidates before reservoir work — drops Python overhead by
    ~1/rate while preserving uniform per-kmer sampling thanks to the
    reservoir downstream.

    `refined_pkl_path` is optional. When None (default), every motif match
    is a candidate; `kinsim refine` filters false positives downstream
    via the p95 IPD threshold. When given, only motif matches whose
    (kmer_at_p, T) appears in the previous master_clean.pkl are kept.

    Args:
        bam_path:                  Raw HiFi or bystrandified BAM (must
                                   carry fi/fp or ip/pw kinetic tags).
        motif_string:              Motif spec; one regex scan per read.
        refined_pkl_path:          Optional confirmation oracle.
        n_baseline_per_kmer:       Reservoir cap per kmer.
        baseline_min_dist_to_meth: Min bases between a baseline center and
                                   any methylation. Default K guarantees
                                   the meth_context window stays empty.
        near_meth_max_dist:        Proximity window for NEAR_METH (default 7,
                                   matches KMER_LEFT_PAD).
        baseline_sample_rate:      Front-end Bernoulli skip rate (0..1).
        revcomp:                   Scan reverse-complement motifs too.
        use_reverse_strand:        Process ri/rp tags as a second strand.
        max_reads:                 Smoke-test cap (0 = no limit).
        kmer_size:                 Window size (default K).
        meth_types:                Optional restriction to a subset of types.
        seed:                      RNG seed for reservoir + subsampling.

    Returns:
        dict with the v4 layout:
            int kmer_id  →  np.ndarray(N, 36)   sample rows
            "__meta__"   →  dict                provenance
        See kinsim.utils.sample_layout for column semantics.
    """
    from .utils.sample_layout import (
        SAMPLE_NCOLS, COL_CATEGORY,
        CATEGORY_BASELINE, CATEGORY_SLOWED, CATEGORY_NEAR_METH,
    )
    rng = np.random.default_rng(seed)
    kinetic_tag = validate_bam_kinetics(bam_path)
    ipd_tag = kinetic_tag
    pw_tag  = "fp" if kinetic_tag == "fi" else "pw"
    log.info("v4 extract — kinetic tags: %s/%s", ipd_tag, pw_tag)

    if meth_types is not None:
        motif_string = filter_motif_string_by_types(motif_string, meth_types)

    # Either load a confirmation set (optional) or treat every motif match
    # as a candidate (default — refine GMM does the filtering downstream).
    if refined_pkl_path is not None:
        confirmed, sig_offsets_by_name = _load_confirmed_set(refined_pkl_path)
        confirmation_mode = "pre-confirmed (refined-pkl)"
        log.info("v4 extract — confirmed buckets: %d  (from %s)",
                 len(confirmed), refined_pkl_path)
    else:
        confirmed = None
        from .utils.config import load_kinsim_config
        cfg = load_kinsim_config()
        sig_offsets_by_name = {}
        for mname, scfg in (cfg.get("kinetic_signatures") or {}).items():
            offs = []
            for k in scfg.get("signal_offsets", []):
                try:
                    offs.append(int(k))
                except (TypeError, ValueError):
                    continue
            sig_offsets_by_name[mname] = offs
        confirmation_mode = "standalone (motif-match)"
        log.info("v4 extract — standalone mode: every motif match treated "
                 "as candidate; refine GMM will drop FP downstream.")
    log.info("v4 extract — confirmation mode: %s", confirmation_mode)
    log.info("v4 extract — signature offsets: %s", sig_offsets_by_name)

    motifs       = parse_motifs(motif_string, revcomp=revcomp)
    frac_lookup  = _build_fraction_lookup(motif_string)
    meth_ids     = get_meth_ids()
    name_by_mid  = {v: k for k, v in meth_ids.items()}

    # samples: slowed + near_meth rows go here, no per-kmer cap (the
    # genomic distribution is what we want to see).
    # baseline_buffer + baseline_seen_per_kmer implement Vitter's
    # streaming reservoir sampling per kmer so memory stays bounded at
    # n_baseline_per_kmer * (# kmers seen) regardless of how many
    # baseline candidates pass through.
    samples:         dict = defaultdict(list)
    baseline_buffer: dict = defaultdict(list)
    baseline_seen_per_kmer: dict = defaultdict(int)
    n_slowed = n_near = n_baseline_seen = 0
    slowed_offset_dist: dict = defaultdict(int)
    near_offset_dist:   dict = defaultdict(int)
    n_reads_processed    = 0
    n_reads_with_reverse = 0
    PROGRESS_EVERY       = 10_000

    log.info("v4 extract from: %s", bam_path)
    log.info("Motifs: %s  |  reverse_strand=%s", motif_string, use_reverse_strand)

    def _build_row(ipd_v, pw_v, frac_v, mc, profile, rev_meth, category):
        row = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
        row[0] = ipd_v
        row[1] = pw_v
        row[2] = frac_v
        row[3:14]  = mc
        row[14:32] = profile
        row[32:35] = rev_meth
        row[COL_CATEGORY] = category
        return row

    # Per-type signature- and near-offset arrays — pre-computed once so the
    # per-read tagging loop below stays a few cheap numpy ops per type.
    sig_offsets_arr_by_mid: dict[int, np.ndarray] = {}
    near_offsets_arr_by_mid: dict[int, np.ndarray] = {}
    for mname, offs in sig_offsets_by_name.items():
        T_id = meth_ids.get(mname)
        if not T_id:
            continue
        sig_set = {int(o) for o in offs}
        sig_offsets_arr_by_mid[int(T_id)] = np.array(sorted(sig_set), dtype=np.int32)
        near = np.array(sorted(k for k in range(0, near_meth_max_dist + 1)
                               if k not in sig_set), dtype=np.int32)
        near_offsets_arr_by_mid[int(T_id)] = near

    def _vec_kmers(seq_str: str) -> tuple:
        """Vectorised per-position kmer encoding.

        Returns (kmer_ids: uint32[n], valid: bool[n]) for the read seq_str.
        kmer at center c is the encoding of seq[c-KMER_LEFT_PAD : c+KMER_RIGHT_PAD+1]
        (length kmer_size). Edge positions whose window falls outside the
        read are kmer_id=0, valid=False.
        """
        n = len(seq_str)
        kmer_ids = np.zeros(n, dtype=np.uint32)
        valid    = np.zeros(n, dtype=bool)
        if n < kmer_size:
            return kmer_ids, valid
        # Encode bases via ord & 6 trick: A=65 -> 0, C=67 -> 1, G=71 -> 3, T=84 -> 2 (incorrect)
        # Use explicit table to be safe.
        seq_bytes = np.frombuffer(seq_str.encode("ascii", errors="replace"), dtype=np.uint8)
        base = np.zeros(n, dtype=np.uint32)
        base[seq_bytes == ord('A')] = 0
        base[seq_bytes == ord('C')] = 1
        base[seq_bytes == ord('G')] = 2
        base[seq_bytes == ord('T')] = 3
        # Sliding rolling hash via stride tricks
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(base, kmer_size)        # (n-K+1, K)
        weights = (np.uint32(4) ** np.arange(kmer_size - 1, -1, -1).astype(np.uint32))
        kmers_at_start = windows.astype(np.uint32) @ weights  # (n-K+1,)
        # start s -> center s + KMER_LEFT_PAD; valid centers [KMER_LEFT_PAD, n-K+KMER_LEFT_PAD]
        n_starts = n - kmer_size + 1
        kmer_ids[KMER_LEFT_PAD:KMER_LEFT_PAD + n_starts] = kmers_at_start
        valid[KMER_LEFT_PAD:KMER_LEFT_PAD + n_starts] = True
        return kmer_ids, valid

    def _process_strand(seq_str, ipd_arr, pw_arr,
                        meth_status_arr, meth_status_complement_arr):
        """Extract all samples from one strand of one read into the global
        accumulators (samples / baseline_buffer).

        Caller must pre-orient the four arrays so that index i refers to
        the same physical position on the strand being processed.

        Implementation: vectorised end-to-end. The per-position Python
        loop and per-position bisect of the previous version are replaced
        by a handful of numpy ops over arrays of length n:
          - kmer encoding via sliding_window_view + dot product
          - signature/near tagging via fancy-index assignment
          - baseline_eligible mask via cumsum-based dilation
          - row construction via batched fancy indexing
        Only the per-emitted-sample kmer-grouping loop (slowed + near +
        kept-baseline ≈ a few thousand per read) remains in Python.
        """
        nonlocal n_slowed, n_near, n_baseline_seen
        n = min(len(seq_str), len(ipd_arr), len(pw_arr))
        if n < kmer_size:
            return
        # Force numpy arrays — pysam tags may be array.array
        meth_status_arr = np.asarray(meth_status_arr[:n], dtype=np.int32)
        meth_status_complement_arr = np.asarray(meth_status_complement_arr[:n], dtype=np.int32)
        ipd_arr = np.asarray(ipd_arr[:n], dtype=np.float32)
        pw_arr  = np.asarray(pw_arr[:n],  dtype=np.float32)

        # ---- 1. Kmer encoding (vectorised) ----
        kmer_ids, kmer_valid = _vec_kmers(seq_str[:n])

        # ---- 2. Phase 1: tag slowed and near positions (vectorised) ----
        # slowed[c] = parent meth_id if c is at signature offset of some meth, else 0
        # near[c]   = parent meth_id if c is in proximity window (non-sig) and not slowed
        slowed = np.zeros(n, dtype=np.int8)
        near   = np.zeros(n, dtype=np.int8)
        # Find motif positions (sparse, ~tens to hundreds per read)
        motif_mask = (meth_status_arr > 0) & kmer_valid
        motif_centers = np.where(motif_mask)[0]
        # Optional refined-pkl gate: keep only motif positions whose
        # (kmer_at_p, T) is in `confirmed`. Sparse loop in Python — OK.
        if confirmed is not None and len(motif_centers) > 0:
            keep_mask = np.zeros(len(motif_centers), dtype=bool)
            for i, p in enumerate(motif_centers):
                T = int(meth_status_arr[p])
                if (int(kmer_ids[p]), T) in confirmed:
                    keep_mask[i] = True
            motif_centers = motif_centers[keep_mask]

        # Tag slowed/near via vectorised offset broadcast per type.
        for T_id, sig_off in sig_offsets_arr_by_mid.items():
            mask_T = meth_status_arr[motif_centers] == T_id
            centers_T = motif_centers[mask_T]
            if len(centers_T) == 0:
                continue
            mname = name_by_mid.get(T_id, f"meth{T_id}")
            # SLOWED: positions p+k for each k in sig_off (k incl. 0)
            for k in sig_off.tolist():
                tgt = centers_T + k
                in_range = (tgt >= 0) & (tgt < n)
                tgt_in = tgt[in_range]
                slowed[tgt_in] = T_id
                slowed_offset_dist[(mname, int(k))] += int(in_range.sum())
            # NEAR: positions p+k for k in [0, near_max] not in sig, only
            # if not already slowed and meth_status[c]==0 for that target.
            for k in near_offsets_arr_by_mid[T_id].tolist():
                tgt = centers_T + k
                in_range = (tgt >= 0) & (tgt < n)
                tgt_in = tgt[in_range]
                # Only assign where slowed is still 0 AND near is still 0
                # (first writer wins for near).
                writeable = (slowed[tgt_in] == 0) & (near[tgt_in] == 0)
                writeable_idx = tgt_in[writeable]
                near[writeable_idx] = T_id
                near_offset_dist[(mname, int(k))] += int(writeable.sum())

        # ---- 3. baseline_eligible mask ----
        # A position is in the "flag zone" iff any motif / slowed / near
        # sits within ±baseline_min_dist_to_meth. Implemented as a window
        # count using cumsum — O(n) vectorised, no per-position bisect.
        flagged = (meth_status_arr > 0) | (slowed > 0) | (near > 0)
        if flagged.any():
            f_int = flagged.astype(np.int32)
            cs = np.concatenate(([0], np.cumsum(f_int)))   # length n+1
            d = baseline_min_dist_to_meth
            lo = np.clip(np.arange(n) - d,         0, n)
            hi = np.clip(np.arange(n) + d + 1,     0, n)
            in_zone = (cs[hi] - cs[lo]) > 0
        else:
            in_zone = np.zeros(n, dtype=bool)

        # ---- 4. Materialise index arrays per category ----
        slowed_mask = (slowed > 0) & kmer_valid
        near_mask   = (near   > 0) & kmer_valid & ~slowed_mask
        baseline_mask = ~in_zone & kmer_valid & ~slowed_mask & ~near_mask
        # Front-end subsample for baseline:
        if baseline_sample_rate < 1.0 and baseline_mask.any():
            r = rng.random(n).astype(np.float32)
            baseline_mask &= (r < baseline_sample_rate)

        slowed_idx   = np.where(slowed_mask)[0]
        near_idx     = np.where(near_mask)[0]
        baseline_idx = np.where(baseline_mask)[0]

        n_slowed += len(slowed_idx)
        n_near   += len(near_idx)
        n_baseline_seen += len(baseline_idx)

        # ---- 5. Build rows in batch via numpy fancy indexing ----
        def _batch_rows(idx, cat, frac_arr=None):
            if len(idx) == 0:
                return np.empty((0, SAMPLE_NCOLS), dtype=np.float32)
            m = len(idx)
            rows = np.zeros((m, SAMPLE_NCOLS), dtype=np.float32)
            rows[:, 0] = ipd_arr[idx]
            rows[:, 1] = pw_arr[idx]
            if frac_arr is not None:
                rows[:, 2] = frac_arr
            # meth_context cols 3..13: mc[k] for sample i = meth_status[idx[i] - 7 + k]
            for k in range(METH_CTX_LEN):
                tgt = idx + (k - METH_CTX_LEFT)
                in_r = (tgt >= 0) & (tgt < n)
                rows[in_r, 3 + k] = meth_status_arr[tgt[in_r]]
            # profile_IPD cols 14..22, profile_PW cols 23..31
            for k in range(PROFILE_LEN):
                tgt = idx + (PROFILE_START + k)
                in_r = (tgt >= 0) & (tgt < n)
                rows[in_r, 14 + k] = ipd_arr[tgt[in_r]]
                rows[in_r, 23 + k] = pw_arr [tgt[in_r]]
            # rev_meth cols 32..34
            for k, off in enumerate(REV_METH_OFFSETS):
                tgt = idx + off
                in_r = (tgt >= 0) & (tgt < n)
                rows[in_r, 32 + k] = meth_status_complement_arr[tgt[in_r]]
            rows[:, COL_CATEGORY] = cat
            return rows

        # Slowed rows: frac_v from frac_lookup of meth at center
        if len(slowed_idx) > 0:
            slowed_meth_at_center = meth_status_arr[slowed_idx]
            slowed_fracs = np.array(
                [frac_lookup.get(int(m), 0.0) for m in slowed_meth_at_center],
                dtype=np.float32,
            )
            slowed_rows = _batch_rows(slowed_idx, CATEGORY_SLOWED, slowed_fracs)
            slowed_kmers = kmer_ids[slowed_idx]
            for i in range(len(slowed_idx)):
                samples[int(slowed_kmers[i])].append(slowed_rows[i])

        if len(near_idx) > 0:
            near_meth_at_center = meth_status_arr[near_idx]
            near_fracs = np.array(
                [frac_lookup.get(int(m), 0.0) for m in near_meth_at_center],
                dtype=np.float32,
            )
            near_rows = _batch_rows(near_idx, CATEGORY_NEAR_METH, near_fracs)
            near_kmers = kmer_ids[near_idx]
            for i in range(len(near_idx)):
                samples[int(near_kmers[i])].append(near_rows[i])

        # ---- 6. Baseline reservoir: per-kmer cap via streaming reservoir ----
        if len(baseline_idx) > 0:
            baseline_rows = _batch_rows(baseline_idx, CATEGORY_BASELINE, None)
            baseline_kmers = kmer_ids[baseline_idx]
            for i in range(len(baseline_idx)):
                kid = int(baseline_kmers[i])
                baseline_seen_per_kmer[kid] += 1
                seen = baseline_seen_per_kmer[kid]
                if seen <= n_baseline_per_kmer:
                    baseline_buffer[kid].append(baseline_rows[i])
                else:
                    j = int(rng.integers(0, seen))
                    if j < n_baseline_per_kmer:
                        baseline_buffer[kid][j] = baseline_rows[i]

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if max_reads > 0 and n_reads_processed >= max_reads:
                log.info("--max-reads %d reached — stopping early", max_reads)
                break
            seq = read.query_sequence
            if not (seq and len(seq) >= kmer_size and read.has_tag(ipd_tag)):
                continue

            ipds = read.get_tag(ipd_tag)
            pws  = read.get_tag(pw_tag)
            min_len = min(len(seq), len(ipds), len(pws))

            meth_status = scan_sequence(seq[:min_len], motifs)
            rc_seq_full = reverse_complement(seq[:min_len])
            meth_status_rc = scan_sequence(rc_seq_full, motifs)
            meth_complement = meth_status_rc[::-1]   # this read's coords

            _process_strand(seq[:min_len], ipds[:min_len], pws[:min_len],
                            meth_status, meth_complement)

            if use_reverse_strand and read.has_tag("ri") and read.has_tag("rp"):
                ri = read.get_tag("ri")
                rp = read.get_tag("rp")
                min_rev_len = min(min_len, len(ri), len(rp))
                if min_rev_len >= kmer_size:
                    n_reads_with_reverse += 1
                    rc_seq = reverse_complement(seq[:min_rev_len])
                    rc_meth_status = scan_sequence(rc_seq, motifs)
                    # ri/rp arrays are stored in forward-read coords; we need
                    # them in rc_seq coords for symmetric processing.
                    ri_rc = list(reversed(list(ri[:min_rev_len])))
                    rp_rc = list(reversed(list(rp[:min_rev_len])))
                    # Complement-of-complement = forward strand meth, in rc coords
                    fwd_in_rc = list(reversed(list(meth_status[:min_rev_len])))
                    _process_strand(rc_seq, ri_rc, rp_rc,
                                    rc_meth_status, fwd_in_rc)

            n_reads_processed += 1
            if n_reads_processed % PROGRESS_EVERY == 0:
                log.info(
                    "  progress: %d reads | slowed=%d near=%d baseline_seen=%d "
                    "(buf_kmers=%d, mem~%dMB)",
                    n_reads_processed, n_slowed, n_near, n_baseline_seen,
                    len(baseline_buffer),
                    # crude estimate: each row ~144 bytes, plus samples dict
                    int((sum(len(v) for v in baseline_buffer.values())
                         + sum(len(v) for v in samples.values())) * 144 / 1e6),
                )

    # Baseline buffer is already capped at n_baseline_per_kmer per kmer
    # (streaming reservoir during the read loop). Just merge into samples.
    n_baseline_kept = 0
    for kmer_id, rows in baseline_buffer.items():
        if rows:
            samples[kmer_id].extend(rows)
            n_baseline_kept += len(rows)

    # Pack samples to ndarrays
    result: dict = {}
    for kmer_id, rows in samples.items():
        result[int(kmer_id)] = np.array(rows, dtype=np.float32)

    n_total = n_slowed + n_near + n_baseline_kept
    log.info("v4 extract done: reads=%d (rev=%d)  slowed=%d  near_meth=%d  "
             "baseline_seen=%d  baseline_kept=%d  total=%d",
             n_reads_processed, n_reads_with_reverse,
             n_slowed, n_near, n_baseline_seen, n_baseline_kept, n_total)
    log.info("v4 extract: %d unique kmers in output", len(result))
    if slowed_offset_dist:
        log.info("Slowed-offset distribution:")
        for (mname, k), n in sorted(slowed_offset_dist.items(),
                                    key=lambda x: (-x[1], x[0])):
            log.info("  %s @ +%d: %d", mname, k, n)
    if near_offset_dist:
        log.info("Near-meth-offset distribution:")
        for (mname, k), n in sorted(near_offset_dist.items(),
                                    key=lambda x: (-x[1], x[0])):
            log.info("  %s @ +%d: %d", mname, k, n)

    result["__meta__"] = {
        "kinsim_version":            _KINSIM_VERSION,
        "extraction_mode":           "v4",
        "confirmation_mode":         confirmation_mode,
        "source_bam":                str(bam_path),
        "refined_pkl":               str(refined_pkl_path) if refined_pkl_path else None,
        "motifs":                    motif_string,
        "meth_types":                sorted(meth_types) if meth_types else None,
        "kmer_size":                 kmer_size,
        "n_baseline_per_kmer":       n_baseline_per_kmer,
        "baseline_min_dist_to_meth": baseline_min_dist_to_meth,
        "near_meth_max_dist":        near_meth_max_dist,
        "use_reverse_strand":        use_reverse_strand,
        "n_reads_processed":         n_reads_processed,
        "n_reads_with_reverse":      n_reads_with_reverse,
        "n_slowed":                  n_slowed,
        "n_near_meth":               n_near,
        "n_baseline_seen":           n_baseline_seen,
        "n_baseline_kept":           n_baseline_kept,
        "slowed_offset_distribution":
            {f"{m}+{k}": n for (m, k), n in slowed_offset_dist.items()},
        "near_offset_distribution":
            {f"{m}+{k}": n for (m, k), n in near_offset_dist.items()},
        "n_unique_kmers":            len(result),
        "n_total_samples":           n_total,
        "created":                   datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return result


def extract_v4_from_manifest_task(
    manifest_path:             str,
    task_index:                int,
    output_dir:                str,
    refined_pkl_path:          str | None = None,
    n_baseline_per_kmer:       int  = 50,
    baseline_min_dist_to_meth: int  = K,
    near_meth_max_dist:        int  = 7,
    baseline_sample_rate:      float = 0.10,
    revcomp:                   bool = True,
    use_reverse_strand:        bool = True,
    max_reads:                 int  = 0,
    kmer_size:                 int  = K,
    meth_types:                set | None = None,
) -> None:
    """Manifest-mode wrapper for the v4 extract path.

    If `refined_pkl_path` is None (default), runs in standalone mode:
    every motif match is treated as a candidate and the FP filtering
    happens in the subsequent `kinsim refine` step.
    """
    from .utils.config import load_manifest
    from .utils.motifs import load_motif_string as _load_motif_string

    entries = load_manifest(manifest_path)
    if task_index < 1 or task_index > len(entries):
        log.error("Task index %d out of range (manifest has %d entries).",
                  task_index, len(entries))
        sys.exit(1)
    entry = entries[task_index - 1]
    log.info("v4 task %d/%d: %s", task_index, len(entries), entry.sample_id)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_pkl = os.path.join(output_dir, f"{entry.sample_id}_shard_v4.pkl")
    log.info("  Output: %s", output_pkl)

    motif_string = _load_motif_string(entry.motifs)
    if not motif_string:
        log.warning("No motifs resolved for '%s' — SKIPPING.", entry.sample_id)
        return

    result = extract_samples_v4_from_bam(
        entry.bam_path, motif_string, refined_pkl_path,
        n_baseline_per_kmer=n_baseline_per_kmer,
        baseline_min_dist_to_meth=baseline_min_dist_to_meth,
        near_meth_max_dist=near_meth_max_dist,
        baseline_sample_rate=baseline_sample_rate,
        revcomp=revcomp,
        use_reverse_strand=use_reverse_strand,
        max_reads=max_reads,
        kmer_size=kmer_size,
        meth_types=meth_types,
    )

    with open(output_pkl, "wb") as f:
        pickle.dump(result, f)
    meta = result.get("__meta__", {})
    log.info("v4 shard saved: %s  slowed=%d near_meth=%d baseline=%d",
             output_pkl,
             meta.get("n_slowed", 0),
             meta.get("n_near_meth", 0),
             meta.get("n_baseline_kept", 0))


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

    Auto-detects shard format by inspecting the first non-meta key:
      - tuple key  -> v3 format (kmer_id, meth_id) -> ndarray(N, 35)
      - int key    -> v4 format kmer_id            -> ndarray(N, 36)

    Mixing v3 and v4 shards in the same merge is rejected (the model loaders
    expect a single layout).

    Looks for shard files in input_dir using the following precedence:
      1. ``*_shard_v4.pkl``  (produced by v4 ``kinsim extract --refined-pkl``)
      2. ``*_shard.pkl``     (produced by ``kinsim extract`` bootstrap)
      3. ``*_cgan.pkl``      (legacy naming, kept for backward compat)

    Override with ``glob_pattern`` to use a custom pattern.

    After concatenation, keys exceeding max_samples_per_key are randomly
    subsampled to keep the master file manageable. For v4 (integer kmer
    keys), this caps total samples PER KMER across all categories
    (meth + slowed + baseline). To preserve the per-category balance, set
    a generous cap (e.g. 500-1000 per kmer for v4).

    The ``"__meta__"`` key (provenance) is merged across all shards and
    stored in the output, with a ``"format"`` field set to "v3" or "v4".
    """
    import glob as _glob

    if glob_pattern == "auto":
        files = _glob.glob(os.path.join(input_dir, "*_shard_v4.pkl"))
        if not files:
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
    detected_format: str | None = None    # "v3" or "v4"

    def _detect_format(shard: dict) -> str | None:
        for k in shard:
            if k == "__meta__":
                continue
            if isinstance(k, tuple):
                return "v3"
            if isinstance(k, (int, np.integer)):
                return "v4"
        return None

    for f_path in files:
        log.info("  Loading shard: %s", os.path.basename(f_path))
        with open(f_path, "rb") as f:
            shard = pickle.load(f)

        # Collect and skip the metadata key
        if "__meta__" in shard:
            shard_metas.append(shard.pop("__meta__"))

        f_format = _detect_format(shard)
        if f_format is None:
            log.warning("    skipping (no data keys): %s", f_path)
            continue
        if detected_format is None:
            detected_format = f_format
            log.info("  Format detected: %s", detected_format)
        elif f_format != detected_format:
            log.error("Cannot mix v3 (tuple keys) and v4 (int keys) shards in "
                      "one merge. Found %s in %s but earlier shards were %s.",
                      f_format, f_path, detected_format)
            sys.exit(1)

        for key, arr in shard.items():
            if detected_format == "v3" and not isinstance(key, tuple):
                continue
            if detected_format == "v4" and not isinstance(key, (int, np.integer)):
                continue
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
        "format":             detected_format or "unknown",
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
    total_samples = sum(len(v) for k, v in result.items() if k != "__meta__")
    log.info("Master dataset saved: %s  (format=%s)", output_file, detected_format)
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
    binarize: bool = True,
    meth_types: set[str] | None = None,
) -> None:
    """Extract one BAM from a manifest CSV (for SLURM array jobs).

    Reads the manifest at ``manifest_path``, picks the row at ``task_index``
    (1-based, matching SLURM_ARRAY_TASK_ID), runs motif-based extraction, and
    writes the shard to ``output_dir/<sample_id>_shard.pkl``.

    Args:
        manifest_path:        Path to the manifest CSV.
        task_index:           1-based row index (SLURM_ARRAY_TASK_ID).
        output_dir:           Directory for the output shard .pkl.
        max_samples_per_key:  Reservoir cap per (kmer, meth_id) key.
        revcomp:              Scan reverse complement strand for motifs.
        use_reverse_strand:   Extract ri/rp complementary-strand kinetics.
        max_reads:            Stop after N reads (0 = no limit, smoke test only).
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

    # Motif-based extraction (single supported path; GFF mode removed in v3)
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

    # ---- v4 mode flags ----
    p_extract.add_argument("--v4", action="store_true",
                           help="Run the v4 extraction (recommended): "
                                "single-pass emission of meth + slowed + "
                                "baseline samples in the 36-col CATEGORY "
                                "format keyed by kmer_id. False-positive "
                                "motifs are dropped by the subsequent "
                                "`kinsim refine` (GMM on meth + p95 on "
                                "slowed). Without --v4 (and without "
                                "--refined-pkl), kinsim extract emits the "
                                "legacy v3 35-col (kmer, meth_id) format.")
    p_extract.add_argument("--refined-pkl", default=None,
                           help="(v4 only, OPTIONAL) Path to a previously-"
                                "produced master_clean.pkl. When given, "
                                "implies --v4 and uses the bucket set as "
                                "an oracle for motif confirmation, skipping "
                                "the GMM step. Almost never needed: just use "
                                "--v4 alone and let refine do the GMM.")
    p_extract.add_argument("--n-baseline-per-kmer", type=int, default=None,
                           help="(v4 only) Cap on baseline samples kept per "
                                "kmer. Overrides kinsim_config.yaml "
                                "extract.n_baseline_per_kmer.")
    p_extract.add_argument("--baseline-min-dist", type=int, default=None,
                           help="(v4 only) Minimum distance (bases) a baseline "
                                "candidate must keep from any meth or slowed "
                                "position. Overrides "
                                "extract.baseline_min_dist_to_meth in YAML.")
    p_extract.add_argument("--near-meth-max-dist", type=int, default=None,
                           help="(v4 only) Max downstream distance for "
                                "CATEGORY_NEAR_METH samples (proximity window "
                                "around each methylation). Defaults to YAML "
                                "extract.near_meth_max_dist (typically 7).")
    p_extract.add_argument("--baseline-sample-rate", type=float, default=None,
                           help="(v4 only) Front-end subsample rate for "
                                "baseline candidates (0..1). Defaults to "
                                "YAML extract.baseline_sample_rate (typically "
                                "0.10). Lower = faster + less memory but "
                                "fewer baselines for rare kmers.")

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

        # Resolve v4 params (CLI overrides YAML). v4 mode is triggered by
        # either --v4 or --refined-pkl (the latter implies --v4).
        v4_mode = args.v4 or (args.refined_pkl is not None)
        v4_baseline_n = args.n_baseline_per_kmer
        v4_baseline_d = args.baseline_min_dist
        v4_near_d     = args.near_meth_max_dist
        v4_sample_r   = args.baseline_sample_rate
        if v4_mode:
            from .utils.config import load_kinsim_config
            cfg = load_kinsim_config()
            ext = (cfg.get("extract") or {})
            if v4_baseline_n is None:
                v4_baseline_n = int(ext.get("n_baseline_per_kmer", 50))
            if v4_baseline_d is None:
                v4_baseline_d = int(ext.get("baseline_min_dist_to_meth", K))
            if v4_near_d is None:
                v4_near_d = int(ext.get("near_meth_max_dist", 7))
            if v4_sample_r is None:
                v4_sample_r = float(ext.get("baseline_sample_rate", 0.10))

        if args.manifest:
            # ---- Manifest mode ----
            if args.task is None:
                log.error("--task is required when using --manifest")
                sys.exit(1)
            if args.output_dir is None:
                log.error("--output-dir is required when using --manifest")
                sys.exit(1)
            if v4_mode:
                extract_v4_from_manifest_task(
                    args.manifest,
                    task_index               = args.task,
                    output_dir               = args.output_dir,
                    refined_pkl_path         = args.refined_pkl,
                    n_baseline_per_kmer      = v4_baseline_n,
                    baseline_min_dist_to_meth= v4_baseline_d,
                    near_meth_max_dist       = v4_near_d,
                    baseline_sample_rate     = v4_sample_r,
                    revcomp                  = not args.no_revcomp,
                    use_reverse_strand       = not args.no_reverse_strand,
                    max_reads                = args.max_reads,
                    kmer_size                = args.kmer_size or K,
                    meth_types               = meth_types,
                )
            else:
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
                    binarize             = not args.no_binarize,
                    meth_types           = meth_types,
                )

        else:
            # ---- Single-BAM mode (motif-based) ----
            if not args.bam or not args.motifs or not args.output:
                log.error(
                    "Single-BAM mode requires: kinsim extract <bam> <motifs> <output>\n"
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
            if v4_mode:
                log.info("v4 mode (--refined-pkl=%s)", args.refined_pkl)
                result = extract_samples_v4_from_bam(
                    args.bam, motif_string, args.refined_pkl,
                    n_baseline_per_kmer       = v4_baseline_n,
                    baseline_min_dist_to_meth = v4_baseline_d,
                    near_meth_max_dist        = v4_near_d,
                    baseline_sample_rate      = v4_sample_r,
                    revcomp                   = not args.no_revcomp,
                    use_reverse_strand        = not args.no_reverse_strand,
                    max_reads                 = args.max_reads,
                    kmer_size                 = args.kmer_size or K,
                    meth_types                = meth_types,
                )
            else:
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
            if v4_mode:
                log.info(
                    "v4 shard saved: %s (kmers=%d, samples=%d, "
                    "slowed=%d near_meth=%d baseline=%d)",
                    args.output,
                    meta.get("n_unique_kmers", "?"),
                    meta.get("n_total_samples", "?"),
                    meta.get("n_slowed", "?"),
                    meta.get("n_near_meth", "?"),
                    meta.get("n_baseline_kept", "?"),
                )
            else:
                log.info(
                    "Shard saved: %s (%d contexts, %d samples)",
                    args.output,
                    meta.get("n_unique_keys", "?"),
                    meta.get("n_total_samples", "?"),
                )


if __name__ == "__main__":
    main()
