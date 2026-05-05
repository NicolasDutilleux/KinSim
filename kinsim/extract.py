"""Extract IPD/PW training samples from BAM files.

Storage format: ``dict[kmer_id (int)] -> np.ndarray(N, 36)`` plus an
optional ``"__meta__"`` provenance key. Col 35 carries the CATEGORY
enum (0=baseline, 1=slowed, 2=near_meth — see kinsim.utils.sample_layout).

CLI:

    # Single-BAM mode
    kinsim extract reads.bam motifs.csv shard.pkl

    # Manifest mode (recommended for SLURM array jobs)
    kinsim extract --manifest manifest.csv --task $SLURM_ARRAY_TASK_ID \\
                   --output-dir shards/

Manifest CSV format (3 columns, with header):
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

from .utils.encoding import KMER_LEFT_PAD, K, get_meth_ids
from .utils.motifs import (
    filter_motif_string_by_types,
    load_motif_string,
    parse_meth_types_arg,
    parse_motifs,
    reverse_complement,
    scan_sequence,
)
from .utils.sample_layout import (
    METH_CTX_LEFT,
    METH_CTX_LEN,
    PROFILE_LEN,
    PROFILE_START,
    REV_METH_OFFSETS,
    SAMPLE_NCOLS,
)

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
    fracs: dict[int, float] = {0: 0.0}
    if not motif_string:
        return fracs
    for entry in motif_string.split(";"):
        if not entry or "," not in entry:
            continue
        parts = entry.split(",")
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
# tests rely on it). The names used elsewhere in this module are re-exported
# at the top of the file.


def extract_samples_from_bam(
    bam_path: str,
    motif_string: str,
    n_baseline_per_kmer: int = 50,
    baseline_min_dist_to_meth: int = K,
    near_meth_max_dist: int = 7,
    baseline_sample_rate: float = 0.10,
    revcomp: bool = True,
    use_reverse_strand: bool = True,
    max_reads: int = 0,
    kmer_size: int = K,
    meth_types: set | None = None,
    seed: int = 42,
) -> dict:
    """Extract training samples from a BAM in a single pass.

    For each motif-match position p of type T identified by the regex
    scan:
      - p + k for k in signature_offsets[T] (incl. 0) → CATEGORY_SLOWED
        (position where the polymerase slowing is biophysically expected)
      - p + k for k in [0, near_meth_max_dist] not in sig_offsets[T] →
        CATEGORY_NEAR_METH (close to the meth but not at a signature
        offset; negative control teaching the model that a methylation
        in the meth_context window alone does not imply elevated IPD —
        only the offset matters)
    Positions at distance ≥ baseline_min_dist_to_meth from any meth or
    flag → CATEGORY_BASELINE candidates, capped per kmer at
    n_baseline_per_kmer via streaming reservoir sampling.

    A front-end Bernoulli sample at rate ``baseline_sample_rate`` skips
    most baseline candidates before reservoir work — drops Python
    overhead by ~1/rate while preserving uniform per-kmer sampling
    thanks to the reservoir downstream.

    False-positive motif matches survive this pass; ``kinsim refine``
    drops their CATEGORY_SLOWED rows downstream via a per-kmer-mean
    baseline IPD threshold.

    Returns:
        dict with int kmer_id keys and ndarray(N, 36) values, plus a
        ``"__meta__"`` provenance dict. See kinsim.utils.sample_layout
        for column semantics.
    """
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_PARENT_METH,
        COL_PARENT_OFFSET,
    )

    rng = np.random.default_rng(seed)
    kinetic_tag = validate_bam_kinetics(bam_path)
    ipd_tag = kinetic_tag
    pw_tag = "fp" if kinetic_tag == "fi" else "pw"
    log.info("extract — kinetic tags: %s/%s", ipd_tag, pw_tag)

    if meth_types is not None:
        motif_string = filter_motif_string_by_types(motif_string, meth_types)

    # Resolve signature offsets from kinsim_config.yaml. Every motif
    # match is a candidate methylation; the refine step downstream
    # drops false positives via a baseline IPD threshold.
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
    log.info("extract — signature offsets: %s", sig_offsets_by_name)

    motifs = parse_motifs(motif_string, revcomp=revcomp)
    frac_lookup = _build_fraction_lookup(motif_string)
    meth_ids = get_meth_ids()
    name_by_mid = {v: k for k, v in meth_ids.items()}

    # Log v4 knobs + motif breakdown so the run is fully traceable.
    log.info(
        "extract — knobs: n_baseline_per_kmer=%d  baseline_min_dist=%d  "
        "near_meth_max_dist=%d  baseline_sample_rate=%.2f  kmer_size=%d  seed=%d",
        n_baseline_per_kmer,
        baseline_min_dist_to_meth,
        near_meth_max_dist,
        baseline_sample_rate,
        kmer_size,
        seed,
    )
    motifs_by_type: dict = defaultdict(int)
    for m in motifs:
        motifs_by_type[name_by_mid.get(int(m["id"]), f"meth{m['id']}")] += 1
    log.info(
        "extract — motifs parsed: %d total (%s)",
        len(motifs),
        ", ".join(f"{name}={cnt}" for name, cnt in sorted(motifs_by_type.items())),
    )
    log.info(
        "extract — fraction_lookup: %s",
        {
            name_by_mid.get(mid, f"meth{mid}"): f"{frac:.3f}"
            for mid, frac in frac_lookup.items()
            if mid > 0
        },
    )

    # samples: slowed + near_meth rows go here, no per-kmer cap (the
    # genomic distribution is what we want to see).
    # baseline_buffer + baseline_seen_per_kmer implement Vitter's
    # streaming reservoir sampling per kmer so memory stays bounded at
    # n_baseline_per_kmer * (# kmers seen) regardless of how many
    # baseline candidates pass through.
    samples: dict = defaultdict(list)
    baseline_buffer: dict = defaultdict(list)
    baseline_seen_per_kmer: dict = defaultdict(int)
    n_slowed = n_near = n_baseline_seen = 0
    slowed_offset_dist: dict = defaultdict(int)
    near_offset_dist: dict = defaultdict(int)
    n_reads_processed = 0
    n_reads_with_reverse = 0
    PROGRESS_EVERY = 10_000

    log.info("extract from: %s", bam_path)
    log.info("Motifs: %s  |  reverse_strand=%s", motif_string, use_reverse_strand)

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
        near = np.array(
            sorted(k for k in range(0, near_meth_max_dist + 1) if k not in sig_set), dtype=np.int32
        )
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
        valid = np.zeros(n, dtype=bool)
        if n < kmer_size:
            return kmer_ids, valid
        # Encode bases via ord & 6 trick: A=65 -> 0, C=67 -> 1, G=71 -> 3, T=84 -> 2 (incorrect)
        # Use explicit table to be safe.
        seq_bytes = np.frombuffer(seq_str.encode("ascii", errors="replace"), dtype=np.uint8)
        base = np.zeros(n, dtype=np.uint32)
        base[seq_bytes == ord("A")] = 0
        base[seq_bytes == ord("C")] = 1
        base[seq_bytes == ord("G")] = 2
        base[seq_bytes == ord("T")] = 3
        # Sliding rolling hash via stride tricks
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(base, kmer_size)  # (n-K+1, K)
        weights = np.uint32(4) ** np.arange(kmer_size - 1, -1, -1).astype(np.uint32)
        kmers_at_start = windows.astype(np.uint32) @ weights  # (n-K+1,)
        # start s -> center s + KMER_LEFT_PAD; valid centers [KMER_LEFT_PAD, n-K+KMER_LEFT_PAD]
        n_starts = n - kmer_size + 1
        kmer_ids[KMER_LEFT_PAD : KMER_LEFT_PAD + n_starts] = kmers_at_start
        valid[KMER_LEFT_PAD : KMER_LEFT_PAD + n_starts] = True
        return kmer_ids, valid

    def _process_strand(seq_str, ipd_arr, pw_arr, meth_status_arr, meth_status_complement_arr):
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
        pw_arr = np.asarray(pw_arr[:n], dtype=np.float32)

        # ---- 1. Kmer encoding (vectorised) ----
        kmer_ids, kmer_valid = _vec_kmers(seq_str[:n])

        # ---- 2. Phase 1: tag slowed and near positions (vectorised) ----
        # slowed[c]            = parent meth_id if c is at a signature offset of some meth, else 0
        # near[c]              = parent meth_id if c is in proximity window (non-sig) and not slowed
        # slowed_parent_off[c] = the offset k that c sits at (last writer wins, matches `slowed`)
        # near_parent_off[c]   = same, for near (first writer wins)
        # The parent meth's genomic position is (c − parent_off); ``parent_off`` lets
        # analyze split slowed_by_T into sig-offset sub-buckets without re-inferring
        # from meth_context. int8 is plenty: |k| ≤ near_meth_max_dist ≤ 7.
        slowed = np.zeros(n, dtype=np.int8)
        near = np.zeros(n, dtype=np.int8)
        slowed_parent_off = np.zeros(n, dtype=np.int8)
        near_parent_off = np.zeros(n, dtype=np.int8)
        # Find motif positions (sparse, ~tens to hundreds per read)
        motif_mask = (meth_status_arr > 0) & kmer_valid
        motif_centers = np.where(motif_mask)[0]

        # Tag slowed/near via vectorised offset broadcast per type.
        for T_id, sig_off in sig_offsets_arr_by_mid.items():
            mask_T = meth_status_arr[motif_centers] == T_id
            centers_T = motif_centers[mask_T]
            if len(centers_T) == 0:
                continue
            mname = name_by_mid.get(T_id, f"meth{T_id}")
            # SLOWED: positions p+k for each k in sig_off (k incl. 0).
            # Last writer wins on conflict (matches the existing behaviour).
            for k in sig_off.tolist():
                tgt = centers_T + k
                in_range = (tgt >= 0) & (tgt < n)
                tgt_in = tgt[in_range]
                slowed[tgt_in] = T_id
                slowed_parent_off[tgt_in] = int(k)
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
                near_parent_off[writeable_idx] = int(k)
                near_offset_dist[(mname, int(k))] += int(writeable.sum())

        # ---- 3. baseline_eligible mask ----
        # A position is in the "flag zone" iff any motif / slowed / near
        # sits within ±baseline_min_dist_to_meth. Implemented as a window
        # count using cumsum — O(n) vectorised, no per-position bisect.
        flagged = (meth_status_arr > 0) | (slowed > 0) | (near > 0)
        if flagged.any():
            f_int = flagged.astype(np.int32)
            cs = np.concatenate(([0], np.cumsum(f_int)))  # length n+1
            d = baseline_min_dist_to_meth
            lo = np.clip(np.arange(n) - d, 0, n)
            hi = np.clip(np.arange(n) + d + 1, 0, n)
            in_zone = (cs[hi] - cs[lo]) > 0
        else:
            in_zone = np.zeros(n, dtype=bool)

        # ---- 4. Materialise index arrays per category ----
        slowed_mask = (slowed > 0) & kmer_valid
        near_mask = (near > 0) & kmer_valid & ~slowed_mask
        baseline_mask = ~in_zone & kmer_valid & ~slowed_mask & ~near_mask
        # Front-end subsample for baseline:
        if baseline_sample_rate < 1.0 and baseline_mask.any():
            r = rng.random(n).astype(np.float32)
            baseline_mask &= r < baseline_sample_rate

        slowed_idx = np.where(slowed_mask)[0]
        near_idx = np.where(near_mask)[0]
        baseline_idx = np.where(baseline_mask)[0]

        n_slowed += len(slowed_idx)
        n_near += len(near_idx)
        n_baseline_seen += len(baseline_idx)

        # ---- 5. Build rows in batch via numpy fancy indexing ----
        def _batch_rows(idx, cat, frac_arr=None, parent_meth_arr=None, parent_off_arr=None):
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
                rows[in_r, 23 + k] = pw_arr[tgt[in_r]]
            # rev_meth cols 32..34
            for k, off in enumerate(REV_METH_OFFSETS):
                tgt = idx + off
                in_r = (tgt >= 0) & (tgt < n)
                rows[in_r, 32 + k] = meth_status_complement_arr[tgt[in_r]]
            rows[:, COL_CATEGORY] = cat
            # cols 36/37 — parent meth attribution. Default 0 (baseline).
            if parent_meth_arr is not None:
                rows[:, COL_PARENT_METH] = parent_meth_arr
            if parent_off_arr is not None:
                rows[:, COL_PARENT_OFFSET] = parent_off_arr
            return rows

        # Slowed rows: frac_v from frac_lookup of meth at center
        if len(slowed_idx) > 0:
            slowed_meth_at_center = meth_status_arr[slowed_idx]
            slowed_fracs = np.array(
                [frac_lookup.get(int(m), 0.0) for m in slowed_meth_at_center],
                dtype=np.float32,
            )
            slowed_rows = _batch_rows(
                slowed_idx,
                CATEGORY_SLOWED,
                slowed_fracs,
                parent_meth_arr=slowed[slowed_idx],
                parent_off_arr=slowed_parent_off[slowed_idx],
            )
            slowed_kmers = kmer_ids[slowed_idx]
            for i in range(len(slowed_idx)):
                samples[int(slowed_kmers[i])].append(slowed_rows[i])

        if len(near_idx) > 0:
            near_meth_at_center = meth_status_arr[near_idx]
            near_fracs = np.array(
                [frac_lookup.get(int(m), 0.0) for m in near_meth_at_center],
                dtype=np.float32,
            )
            near_rows = _batch_rows(
                near_idx,
                CATEGORY_NEAR_METH,
                near_fracs,
                parent_meth_arr=near[near_idx],
                parent_off_arr=near_parent_off[near_idx],
            )
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
            pws = read.get_tag(pw_tag)
            min_len = min(len(seq), len(ipds), len(pws))

            meth_status = scan_sequence(seq[:min_len], motifs)
            rc_seq_full = reverse_complement(seq[:min_len])
            meth_status_rc = scan_sequence(rc_seq_full, motifs)
            meth_complement = meth_status_rc[::-1]  # this read's coords

            _process_strand(
                seq[:min_len], ipds[:min_len], pws[:min_len], meth_status, meth_complement
            )

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
                    _process_strand(rc_seq, ri_rc, rp_rc, rc_meth_status, fwd_in_rc)

            n_reads_processed += 1
            if n_reads_processed % PROGRESS_EVERY == 0:
                # Mem estimate (rows ~144B + python overhead ~100B per ndarray
                # in a list -> ~250B per stored sample).
                n_buf = sum(len(v) for v in baseline_buffer.values())
                n_smp = sum(len(v) for v in samples.values())
                mem_mb = int((n_buf + n_smp) * 250 / 1e6)

                # Running IPD stats per category from a sample of the buffer.
                # Sampling: peek at the first 5 rows of the first 200 kmer
                # buffers to keep this O(1) regardless of buffer size.
                def _peek_ipd(buf, max_kmers=200, max_per=5):
                    vals = []
                    for i, rows in enumerate(buf.values()):
                        if i >= max_kmers:
                            break
                        for r in rows[:max_per]:
                            vals.append(float(r[0]))
                    return vals

                slow_pw = _peek_ipd({k: v for k, v in samples.items()})
                base_pw = _peek_ipd(baseline_buffer)
                slow_med = float(np.median(slow_pw)) if slow_pw else 0.0
                base_med = float(np.median(base_pw)) if base_pw else 0.0
                log.info(
                    "  progress: %d reads | slowed=%d near=%d baseline_seen=%d "
                    "kept=%d | buf_kmers=%d mem~%dMB | "
                    "running IPD median: samples=%.1f baseline_buf=%.1f",
                    n_reads_processed,
                    n_slowed,
                    n_near,
                    n_baseline_seen,
                    n_buf,
                    len(baseline_buffer),
                    mem_mb,
                    slow_med,
                    base_med,
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
    log.info(
        "extract done: reads=%d (rev=%d)  slowed=%d  near_meth=%d  "
        "baseline_seen=%d  baseline_kept=%d  total=%d",
        n_reads_processed,
        n_reads_with_reverse,
        n_slowed,
        n_near,
        n_baseline_seen,
        n_baseline_kept,
        n_total,
    )
    log.info("extract: %d unique kmers in output", len(result))
    if slowed_offset_dist:
        log.info("Slowed-offset distribution:")
        for (mname, k), n in sorted(slowed_offset_dist.items(), key=lambda x: (-x[1], x[0])):
            log.info("  %s @ +%d: %d", mname, k, n)
    if near_offset_dist:
        log.info("Near-meth-offset distribution:")
        for (mname, k), n in sorted(near_offset_dist.items(), key=lambda x: (-x[1], x[0])):
            log.info("  %s @ +%d: %d", mname, k, n)

    result["__meta__"] = {
        "kinsim_version": _KINSIM_VERSION,
        "source_bam": str(bam_path),
        "motifs": motif_string,
        "meth_types": sorted(meth_types) if meth_types else None,
        "kmer_size": kmer_size,
        "n_baseline_per_kmer": n_baseline_per_kmer,
        "baseline_min_dist_to_meth": baseline_min_dist_to_meth,
        "near_meth_max_dist": near_meth_max_dist,
        "use_reverse_strand": use_reverse_strand,
        "n_reads_processed": n_reads_processed,
        "n_reads_with_reverse": n_reads_with_reverse,
        "n_slowed": n_slowed,
        "n_near_meth": n_near,
        "n_baseline_seen": n_baseline_seen,
        "n_baseline_kept": n_baseline_kept,
        "slowed_offset_distribution": {f"{m}+{k}": n for (m, k), n in slowed_offset_dist.items()},
        "near_offset_distribution": {f"{m}+{k}": n for (m, k), n in near_offset_dist.items()},
        "n_unique_kmers": len(result),
        "n_total_samples": n_total,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return result


def extract_from_manifest_task(
    manifest_path: str,
    task_index: int,
    output_dir: str,
    n_baseline_per_kmer: int = 50,
    baseline_min_dist_to_meth: int = K,
    near_meth_max_dist: int = 7,
    baseline_sample_rate: float = 0.10,
    revcomp: bool = True,
    use_reverse_strand: bool = True,
    max_reads: int = 0,
    kmer_size: int = K,
    meth_types: set | None = None,
) -> None:
    """Manifest-mode wrapper for ``extract_samples_from_bam``.

    Picks row ``task_index`` (1-based, matches SLURM_ARRAY_TASK_ID) from
    the manifest CSV and writes the shard to
    ``output_dir/<sample_id>_shard.pkl``.
    """
    from .utils.config import load_manifest
    from .utils.motifs import load_motif_string as _load_motif_string

    entries = load_manifest(manifest_path)
    if task_index < 1 or task_index > len(entries):
        log.error("Task index %d out of range (manifest has %d entries).", task_index, len(entries))
        sys.exit(1)
    entry = entries[task_index - 1]
    log.info("task %d/%d: %s", task_index, len(entries), entry.sample_id)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_pkl = os.path.join(output_dir, f"{entry.sample_id}_shard.pkl")
    log.info("  Output: %s", output_pkl)

    motif_string = _load_motif_string(entry.motifs)
    if not motif_string:
        log.warning("No motifs resolved for '%s' — SKIPPING.", entry.sample_id)
        return

    result = extract_samples_from_bam(
        entry.bam_path,
        motif_string,
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

    from .utils.io import atomic_write_pickle

    atomic_write_pickle(result, output_pkl)
    meta = result.get("__meta__", {})
    log.info(
        "shard saved: %s  slowed=%d near_meth=%d baseline=%d",
        output_pkl,
        meta.get("n_slowed", 0),
        meta.get("n_near_meth", 0),
        meta.get("n_baseline_kept", 0),
    )


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

    Concatenates per-kmer rows from each shard. If a kmer ends up with
    more than ``max_samples_per_key`` rows after concatenation, randomly
    subsamples down to that cap.
    """
    import glob as _glob

    if glob_pattern == "auto":
        files = _glob.glob(os.path.join(input_dir, "*_shard.pkl"))
        if not files:
            files = _glob.glob(os.path.join(input_dir, "*.pkl"))
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
        if "__meta__" in shard:
            shard_metas.append(shard.pop("__meta__"))
        for key, arr in shard.items():
            if not isinstance(key, (int, np.integer)):
                continue
            master[int(key)].append(arr)

    result: dict = {}
    n_subsampled = 0
    for key, arrays in master.items():
        combined = np.concatenate(arrays, axis=0)
        if len(combined) > max_samples_per_key:
            idx = np.random.choice(len(combined), max_samples_per_key, replace=False)
            combined = combined[idx]
            n_subsampled += 1
        result[key] = combined

    # Consolidate meth_types across shards — all shards must agree on the
    # active alphabet otherwise training labels are inconsistent.
    shard_meth_types = [
        tuple(m["meth_types"]) if m.get("meth_types") else None for m in shard_metas
    ]
    if len(set(shard_meth_types)) > 1:
        log.warning(
            "Shards were extracted with different --meth-types — merged dataset will mix alphabets."
        )
    merged_meth_types = (
        sorted(shard_meth_types[0])
        if shard_meth_types and shard_meth_types[0] is not None
        else None
    )

    result["__meta__"] = {
        "kinsim_version": _KINSIM_VERSION,
        "merged_from": [m.get("source_bam", "?") for m in shard_metas],
        "meth_types": merged_meth_types,
        "n_shards": len(files),
        "max_samples_per_key": max_samples_per_key,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    from .utils.io import atomic_write_pickle

    atomic_write_pickle(result, output_file)

    total_keys = len(result) - 1
    total_samples = sum(len(v) for k, v in result.items() if k != "__meta__")
    log.info(
        "Master saved: %s  %d kmers  %d samples  (%d kmers capped at %d)",
        output_file,
        total_keys,
        total_samples,
        n_subsampled,
        max_samples_per_key,
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
            '  kinsim extract reads.bam "m6A,GATC,1" shard.pkl\n\n'
            "Manifest-based extract (recommended for SLURM array jobs):\n"
            "  kinsim extract --manifest manifest.csv --task $SLURM_ARRAY_TASK_ID \\\n"
            "                 --output-dir shards/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")
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
    p_extract.add_argument(
        "bam", nargs="?", default=None, help="Input BAM file with fi/fp kinetic tags"
    )
    p_extract.add_argument(
        "motifs",
        nargs="?",
        default=None,
        help="Motif source: KinSim string ('m6A,GATC,1'), "
        "PacBio motifs.csv, or REBASE file (auto-detected)",
    )
    p_extract.add_argument(
        "output", nargs="?", default=None, help="Output .pkl shard file (single-BAM mode)"
    )

    # Manifest mode args
    p_extract.add_argument(
        "--manifest", default=None, help="Manifest CSV with columns: sample_id, bam_path, motifs"
    )
    p_extract.add_argument(
        "--task",
        type=int,
        default=None,
        help="1-based row index from the manifest (= SLURM_ARRAY_TASK_ID)",
    )
    p_extract.add_argument(
        "--output-dir", default=None, help="Output directory for shard .pkl files (manifest mode)"
    )

    p_extract.add_argument(
        "--no-revcomp", action="store_true", help="Do not scan reverse complement strand for motifs"
    )
    p_extract.add_argument(
        "--no-reverse-strand",
        action="store_true",
        help="Do not extract ri/rp complementary-strand kinetics. "
        "By default both forward and reverse strands are "
        "extracted, doubling training data and making the "
        "model strand-invariant.",
    )
    p_extract.add_argument(
        "--min-fraction",
        type=float,
        default=0.40,
        help="Minimum fraction threshold for PacBio CSV (default: 0.40)",
    )
    p_extract.add_argument(
        "--min-detected",
        type=int,
        default=20,
        help="Minimum nDetected threshold for PacBio CSV (default: 20)",
    )
    p_extract.add_argument(
        "--kmer-size",
        type=int,
        default=None,
        help="K-mer window size (default: K=11 from encoding.py). "
        "Must match the value used during training.",
    )
    p_extract.add_argument(
        "--max-reads",
        type=int,
        default=0,
        help="Stop after N reads (0 = no limit). Smoke-test only.",
    )
    p_extract.add_argument(
        "--meth-types",
        default=None,
        help="Comma-separated list of methylation types to include "
        "(e.g. 'm6A,m4C'). 'all' (default) accepts every type.",
    )
    p_extract.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG-level logging"
    )

    p_extract.add_argument(
        "--n-baseline-per-kmer",
        type=int,
        default=None,
        help="Cap on baseline samples kept per kmer. "
        "Overrides extract.n_baseline_per_kmer in YAML.",
    )
    p_extract.add_argument(
        "--baseline-min-dist",
        type=int,
        default=None,
        help="Minimum distance (bases) a baseline candidate must "
        "keep from any meth or slowed position. Overrides "
        "extract.baseline_min_dist_to_meth in YAML.",
    )
    p_extract.add_argument(
        "--near-meth-max-dist",
        type=int,
        default=None,
        help="Max downstream distance for CATEGORY_NEAR_METH "
        "samples (proximity window around each methylation). "
        "Defaults to YAML extract.near_meth_max_dist (typically 7).",
    )
    p_extract.add_argument(
        "--baseline-sample-rate",
        type=float,
        default=None,
        help="Front-end subsample rate for baseline candidates "
        "(0..1). Defaults to YAML extract.baseline_sample_rate "
        "(typically 0.10). Lower = faster + less memory but "
        "fewer baselines for rare kmers.",
    )

    # -- merge subcommand --
    p_merge = sub.add_parser(
        "merge",
        help="Merge multiple *_shard.pkl shards into one master",
        description=(
            "Concatenate sample arrays from all shards in a directory.\n"
            "Subsamples per kmer if total exceeds --max-samples."
        ),
    )
    p_merge.add_argument("input_dir", help="Directory containing shard .pkl files")
    p_merge.add_argument("output", help="Output master training set .pkl file")
    p_merge.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        help="Max samples per kmer after merging (default: 50000)",
    )
    p_merge.add_argument(
        "--glob",
        default="auto",
        dest="glob_pattern",
        help="Glob pattern for shard files (default: auto-detect)",
    )
    p_merge.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False))

    if args.command == "merge":
        merge_shards(
            args.input_dir,
            args.output,
            max_samples_per_key=args.max_samples,
            glob_pattern=args.glob_pattern,
        )
        return

    # ---- extract subcommand ----
    meth_types = parse_meth_types_arg(args.meth_types)

    # Resolve v4 knobs (CLI overrides YAML).
    from .utils.config import load_kinsim_config

    cfg = load_kinsim_config()
    ext = cfg.get("extract") or {}
    n_baseline_per_kmer = (
        args.n_baseline_per_kmer
        if args.n_baseline_per_kmer is not None
        else int(ext.get("n_baseline_per_kmer", 50))
    )
    baseline_min_dist_to_meth = (
        args.baseline_min_dist
        if args.baseline_min_dist is not None
        else int(ext.get("baseline_min_dist_to_meth", K))
    )
    near_meth_max_dist = (
        args.near_meth_max_dist
        if args.near_meth_max_dist is not None
        else int(ext.get("near_meth_max_dist", 7))
    )
    baseline_sample_rate = (
        args.baseline_sample_rate
        if args.baseline_sample_rate is not None
        else float(ext.get("baseline_sample_rate", 0.10))
    )

    if args.manifest:
        if args.task is None:
            log.error("--task is required when using --manifest")
            sys.exit(1)
        if args.output_dir is None:
            log.error("--output-dir is required when using --manifest")
            sys.exit(1)
        extract_from_manifest_task(
            args.manifest,
            task_index=args.task,
            output_dir=args.output_dir,
            n_baseline_per_kmer=n_baseline_per_kmer,
            baseline_min_dist_to_meth=baseline_min_dist_to_meth,
            near_meth_max_dist=near_meth_max_dist,
            baseline_sample_rate=baseline_sample_rate,
            revcomp=not args.no_revcomp,
            use_reverse_strand=not args.no_reverse_strand,
            max_reads=args.max_reads,
            kmer_size=args.kmer_size or K,
            meth_types=meth_types,
        )
        return

    # ---- Single-BAM mode ----
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
    result = extract_samples_from_bam(
        args.bam,
        motif_string,
        n_baseline_per_kmer=n_baseline_per_kmer,
        baseline_min_dist_to_meth=baseline_min_dist_to_meth,
        near_meth_max_dist=near_meth_max_dist,
        baseline_sample_rate=baseline_sample_rate,
        revcomp=not args.no_revcomp,
        use_reverse_strand=not args.no_reverse_strand,
        max_reads=args.max_reads,
        kmer_size=args.kmer_size or K,
        meth_types=meth_types,
    )

    from .utils.io import atomic_write_pickle

    atomic_write_pickle(result, args.output)

    meta = result.get("__meta__", {})
    log.info(
        "shard saved: %s (kmers=%d, samples=%d, slowed=%d near_meth=%d baseline=%d)",
        args.output,
        meta.get("n_unique_kmers", "?"),
        meta.get("n_total_samples", "?"),
        meta.get("n_slowed", "?"),
        meta.get("n_near_meth", "?"),
        meta.get("n_baseline_kept", "?"),
    )


if __name__ == "__main__":
    main()
