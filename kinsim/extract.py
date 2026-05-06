"""``kinsim extract`` — extract per-row training samples from aligned bystrandified BAMs.

Why aligned-only
----------------
KinSim used to have a second path that scanned motifs in raw HiFi CCS
read sequences and read ``fi[A]`` directly. That path is dead because
``query_sequence`` orientation in raw HiFi is arbitrary per-read
(CCS+lima choose it from barcodes), so for ~50% of reads ``fi`` and
``ri`` are swapped relative to the reference strand of the
methylation. The per-row signal averaged away, and the model trained
on the resulting shard learned nothing.

After bystrandify+pbmm2 alignment we have:
- One read per polymerase-pass strand (a single ``ip``/``pw`` array)
- ``read.is_reverse`` tells us which reference strand each read aligns to

PacBio kinetics convention: ``ip[read_pos]`` is the IPD when the
polymerase synthesised position ``read_pos`` of the read sequence by
reading the OPPOSITE strand as template. Therefore:

- ``read.is_reverse=False``: read sequence == reference + strand
   ⇒ ``ip`` reads − strand template
   ⇒ captures methylation on − strand at the corresponding ref_pos
- ``read.is_reverse=True``: read sequence == revcomp of + strand
   ⇒ ``ip`` reads + strand template
   ⇒ captures methylation on + strand at the corresponding ref_pos

This module pre-builds two per-strand methylation maps from the
reference and routes each aligned read's ``ip[read_pos]`` lookups to
the right map based on ``read.is_reverse``. Per-row signal is
preserved instead of being washed out.

Required prerequisites
----------------------
- Aligned bystrandified BAM (``ip``/``pw`` tags + ``@HD SO:coordinate``).
  See ``slurm_kinsim/strepto/0[01]_*.slurm`` for the prep pipeline.
- Reference FASTA the BAM was aligned against (manifest column ``ref_path``).
- Per-meth-type ``signal_offsets`` declared in ``kinsim_config.yaml``.

CLI
---
::

    kinsim extract --manifest manifest.csv --task $SLURM_ARRAY_TASK_ID \\
                   --output-dir shards/

Manifest schema (columns; ``ref_path`` is REQUIRED for this path)::

    sample_id,bam_path,motifs,ref_path

Storage layout
--------------
``dict[kmer_id (int) → np.ndarray(N, 38)]`` plus ``"__meta__"``. The
38-column layout is defined in :mod:`kinsim.utils.sample_layout`. Refine,
train, analyze operate row-by-row on this layout regardless of which
extract version produced the shard.
"""

from __future__ import annotations

import datetime
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysam

from .utils.config import load_kinsim_config
from .utils.encoding import KMER_LEFT_PAD, K, get_meth_ids
from .utils.io import atomic_write_pickle, load_reference
from .utils.motifs import (
    filter_motif_string_by_types,
    iupac_to_re,
    reverse_complement,
)
from .utils.sample_layout import (
    CATEGORY_BASELINE,
    CATEGORY_NEAR_METH,
    CATEGORY_SLOWED,
    COL_CATEGORY,
    COL_PARENT_METH,
    COL_PARENT_OFFSET,
    METH_CTX_LEFT,
    METH_CTX_LEN,
    REV_METH_OFFSETS,
    SAMPLE_NCOLS,
)

try:
    from . import __version__ as _KINSIM_VERSION
except (ImportError, AttributeError):
    _KINSIM_VERSION = "unknown"

log = logging.getLogger(__name__)

PROGRESS_EVERY = 10000


# ---------------------------------------------------------------------------
# Per-strand reference maps
# ---------------------------------------------------------------------------


def _parse_motif_string_no_rc(motif_string: str) -> list[dict]:
    """Parse a motif string into a list of dicts WITHOUT auto-revcomp.

    Each dict has ``pattern`` (compiled regex), ``id`` (int meth_id),
    ``mod_pos`` (0-based position of the modified base in the motif),
    and ``frac`` (per-position methylation fraction). We don't reuse
    ``parse_motifs(revcomp=True)`` because we want strict control over
    which strand each motif applies to: the user-provided sequences are
    the "forward strand" specification, and we scan the rc of the
    reference separately to find − strand methylations.
    """
    import re

    from .utils.motifs import _validate_mod_pos

    motifs = []
    if not motif_string:
        return motifs
    for entry in motif_string.split(";"):
        if not entry or "," not in entry:
            continue
        parts = entry.split(",")
        if len(parts) < 3:
            continue
        m_type, seq, pos = parts[0], parts[1], parts[2]
        try:
            mod_pos = int(pos) - 1
        except ValueError:
            log.warning("extract: invalid mod_pos for '%s' (%s) — skipped", seq, pos)
            continue
        try:
            _validate_mod_pos(seq, mod_pos, m_type)
        except ValueError as exc:
            log.warning("extract: motif '%s' (%s) failed validation: %s — skipped",
                        seq, m_type, exc)
            continue
        m_id = get_meth_ids().get(m_type, 0)
        if m_id == 0:
            continue
        frac = float(parts[4]) if len(parts) >= 5 else 1.0
        motifs.append({
            "pattern": re.compile(f"(?=({iupac_to_re(seq)}))"),
            "id": m_id,
            "mod_pos": mod_pos,
            "frac": frac,
            "seq": seq,
            "type": m_type,
        })
    return motifs


def _build_strand_maps(
    ref_seqs: dict[str, str],
    motifs: list[dict],
    sig_offsets_by_mid: dict[int, list[int]],
    near_max_dist: int,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Pre-scan the reference and build per-strand methylation maps.

    Returns six dicts:
        fwd_slowed, rev_slowed : (contig, ref_pos) → (meth_id, parent_offset, frac)
        fwd_near,   rev_near   : same shape (NEAR_METH within ±near_max_dist)
        fwd_meth,   rev_meth   : (contig, ref_pos) → meth_id (canonical centre only)

    Per-strand semantics: signature offsets propagate downstream in 5'→3'
    of the methylated strand. For + strand methylations that's + ref
    direction; for − strand it's − ref direction (since − strand 5'→3'
    runs from high to low + coords). Closest-offset-wins on overlapping
    motif claims: a position p+1 of one motif and p of another both
    qualify for SLOWED, but the one whose canonical site is at smaller
    |k| keeps the slot.
    """
    fwd_slowed: dict = {}
    rev_slowed: dict = {}
    fwd_meth: dict = {}
    rev_meth: dict = {}

    for contig, seq in ref_seqs.items():
        rc_seq = reverse_complement(seq)
        seq_len = len(seq)

        for motif in motifs:
            sig_off = sig_offsets_by_mid.get(motif["id"], [0])
            frac = float(motif.get("frac", 1.0))
            for match in motif["pattern"].finditer(seq):
                meth_pos = match.start() + motif["mod_pos"]
                if not (0 <= meth_pos < seq_len):
                    continue
                fwd_meth[(contig, meth_pos)] = motif["id"]
                for k in sig_off:
                    tgt = meth_pos + k
                    if 0 <= tgt < seq_len:
                        existing = fwd_slowed.get((contig, tgt))
                        if existing is None or abs(k) < abs(existing[1]):
                            fwd_slowed[(contig, tgt)] = (motif["id"], int(k), frac)

        for motif in motifs:
            sig_off = sig_offsets_by_mid.get(motif["id"], [0])
            frac = float(motif.get("frac", 1.0))
            for match in motif["pattern"].finditer(rc_seq):
                rc_meth_pos = match.start() + motif["mod_pos"]
                if not (0 <= rc_meth_pos < seq_len):
                    continue
                fwd_pos = seq_len - 1 - rc_meth_pos
                rev_meth[(contig, fwd_pos)] = motif["id"]
                for k in sig_off:
                    tgt = fwd_pos - k
                    if 0 <= tgt < seq_len:
                        existing = rev_slowed.get((contig, tgt))
                        if existing is None or abs(k) < abs(existing[1]):
                            rev_slowed[(contig, tgt)] = (motif["id"], int(k), frac)

    # NEAR_METH = within ±near_max_dist of a canonical meth on the same
    # strand AND not already a SLOWED position. Inherits the meth's frac.
    fwd_near: dict = {}
    rev_near: dict = {}
    fwd_meth_to_frac = {pos: fwd_slowed.get(pos, (0, 0, 1.0))[2]
                        for pos in fwd_meth.keys()}
    rev_meth_to_frac = {pos: rev_slowed.get(pos, (0, 0, 1.0))[2]
                        for pos in rev_meth.keys()}
    for (contig, p), m_id in fwd_meth.items():
        frac = fwd_meth_to_frac[(contig, p)]
        for k in range(1, near_max_dist + 1):
            tgt = p + k
            if (contig, tgt) in fwd_slowed:
                continue
            existing = fwd_near.get((contig, tgt))
            if existing is None or abs(k) < abs(existing[1]):
                fwd_near[(contig, tgt)] = (m_id, k, frac)
    for (contig, p), m_id in rev_meth.items():
        frac = rev_meth_to_frac[(contig, p)]
        for k in range(1, near_max_dist + 1):
            tgt = p - k
            if (contig, tgt) in rev_slowed:
                continue
            existing = rev_near.get((contig, tgt))
            if existing is None or abs(k) < abs(existing[1]):
                rev_near[(contig, tgt)] = (m_id, k, frac)

    return fwd_slowed, rev_slowed, fwd_near, rev_near, fwd_meth, rev_meth


# ---------------------------------------------------------------------------
# Per-contig precompute + per-read extraction
# ---------------------------------------------------------------------------


def _build_contig_arrays(
    contig: str,
    seq: str,
    fwd_slowed: dict,
    rev_slowed: dict,
    fwd_near: dict,
    rev_near: dict,
    fwd_meth: dict,
    rev_meth: dict,
    baseline_min_dist: int,
) -> dict:
    """Materialise per-position lookup arrays for one contig.

    Replaces the per-position dict lookups in the BAM inner loop with
    O(1) array indexing. Memory is ~14 bytes per ref base — negligible
    for bacterial genomes — and reads of millions of positions become
    measurably faster.
    """
    n = len(seq)
    base_int = np.full(n, -1, dtype=np.int8)  # -1 → non-ACGT
    base_int[np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("A")] = 0
    base_int[np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("C")] = 1
    base_int[np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("G")] = 2
    base_int[np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("T")] = 3
    valid_base = base_int >= 0

    # Sliding 11-mer encoding on the + strand. window[i] covers ref
    # positions i-LEFT .. i+(K-LEFT)-1; out-of-range → invalid.
    LEFT = KMER_LEFT_PAD
    RIGHT = K - LEFT
    kmer_fwd = np.full(n, -1, dtype=np.int64)
    kmer_window_valid = np.ones(n, dtype=bool)
    for j in range(K):
        # In + strand 5'→3' frame: the j-th base of the kmer at ref_pos i
        # sits at ref position (i - LEFT + j).
        offset = j - LEFT
        # roll() gives wrap-around; we mask out boundary positions below.
        rolled = np.roll(base_int, -offset)
        kmer_window_valid &= np.roll(valid_base, -offset)
        kmer_fwd = (kmer_fwd << 2) | (rolled.astype(np.int64) & 3)
    # Boundary mask: any position whose 11-mer reaches outside [0, n) is
    # invalid. The kmer at ref_pos i needs positions [i-LEFT, i+RIGHT-1].
    edge_invalid = np.zeros(n, dtype=bool)
    edge_invalid[:LEFT] = True
    if RIGHT > 0:
        edge_invalid[n - RIGHT + 1:] = True
    kmer_fwd[edge_invalid | ~kmer_window_valid] = -1

    # Reverse-complement kmer at the same ref position: the polymerase
    # reading the − strand sees a different 11-mer here. Bit-reverse pairs
    # of the + strand kmer and complement them (XOR with 0b1111... pattern).
    # XOR pattern over K bases: each 2-bit unit XORs with 3 (A↔T, C↔G).
    xor_mask = np.int64(0)
    for _ in range(K):
        xor_mask = (xor_mask << 2) | 3
    # Bit-reverse the K base-pairs:
    src = kmer_fwd.copy()
    kmer_rev = np.zeros_like(src)
    for _ in range(K):
        kmer_rev = (kmer_rev << 2) | (src & 3)
        src >>= 2
    kmer_rev ^= xor_mask
    kmer_rev[kmer_fwd < 0] = -1

    # Meth presence + categorisation arrays. int8 holds meth_id (1-3),
    # offset (-7..+7), and frac×100 quantised. We keep frac as float32.
    fwd_meth_arr = np.zeros(n, dtype=np.int8)
    rev_meth_arr = np.zeros(n, dtype=np.int8)
    for (c, p), m_id in fwd_meth.items():
        if c == contig and 0 <= p < n:
            fwd_meth_arr[p] = m_id
    for (c, p), m_id in rev_meth.items():
        if c == contig and 0 <= p < n:
            rev_meth_arr[p] = m_id

    def _from_dict(d, n_):
        T = np.zeros(n_, dtype=np.int8)
        off = np.zeros(n_, dtype=np.int8)
        frac = np.zeros(n_, dtype=np.float32)
        for (c, p), val in d.items():
            if c == contig and 0 <= p < n_:
                T[p] = val[0]
                off[p] = val[1]
                frac[p] = val[2]
        return T, off, frac

    fwd_slowed_T, fwd_slowed_off, fwd_slowed_frac = _from_dict(fwd_slowed, n)
    rev_slowed_T, rev_slowed_off, rev_slowed_frac = _from_dict(rev_slowed, n)
    fwd_near_T, fwd_near_off, fwd_near_frac = _from_dict(fwd_near, n)
    rev_near_T, rev_near_off, rev_near_frac = _from_dict(rev_near, n)

    # Baseline-exclusion mask: True at positions within ±baseline_min_dist
    # of ANY canonical meth (either strand). Single boolean array
    # replaces the per-position 22-dict-lookup distance check.
    meth_present = (fwd_meth_arr > 0) | (rev_meth_arr > 0)
    excluded = meth_present.copy()
    for d in range(1, baseline_min_dist + 1):
        excluded[:-d] |= meth_present[d:]
        excluded[d:] |= meth_present[:-d]

    return {
        "kmer_fwd": kmer_fwd,
        "kmer_rev": kmer_rev,
        "fwd_meth": fwd_meth_arr,
        "rev_meth": rev_meth_arr,
        "fwd_slowed_T": fwd_slowed_T,
        "fwd_slowed_off": fwd_slowed_off,
        "fwd_slowed_frac": fwd_slowed_frac,
        "rev_slowed_T": rev_slowed_T,
        "rev_slowed_off": rev_slowed_off,
        "rev_slowed_frac": rev_slowed_frac,
        "fwd_near_T": fwd_near_T,
        "fwd_near_off": fwd_near_off,
        "fwd_near_frac": fwd_near_frac,
        "rev_near_T": rev_near_T,
        "rev_near_off": rev_near_off,
        "rev_near_frac": rev_near_frac,
        "excluded": excluded,
    }


def _extract_one_bam(
    bam_path: str,
    motif_string: str,
    ref_seqs: dict[str, str],
    cfg: dict,
    n_baseline_per_kmer: int,
    baseline_min_dist_to_meth: int,
    baseline_sample_rate: float,
    near_max_dist: int,
    seed: int,
    max_reads: int,
) -> dict:
    """Run orientation-aware extraction. Returns dict[kmer_id] → ndarray(N, 20)."""

    rng = np.random.default_rng(seed)
    motifs = _parse_motif_string_no_rc(motif_string)
    if not motifs:
        raise ValueError("extract: no valid motifs after parsing.")

    sig_offsets_by_mid: dict = {}
    cfg_sigs = cfg.get("kinetic_signatures") or {}
    name_by_mid = {v: k for k, v in get_meth_ids().items()}
    for m_id, m_name in name_by_mid.items():
        if m_id == 0:
            continue
        if m_name in cfg_sigs:
            sig_offsets_by_mid[m_id] = [int(k) for k in cfg_sigs[m_name].get("signal_offsets", [0])]
    if not sig_offsets_by_mid:
        raise ValueError(
            "kinsim_config.yaml has no `kinetic_signatures` entries. "
            "Declare per-meth-type signal_offsets and re-run."
        )
    log.info("[extract] signal offsets: %s",
             {name_by_mid[mid]: offs for mid, offs in sig_offsets_by_mid.items()})

    log.info("[extract] pre-scanning reference for motif positions (per strand) ...")
    fwd_slowed, rev_slowed, fwd_near, rev_near, fwd_meth, rev_meth = _build_strand_maps(
        ref_seqs, motifs, sig_offsets_by_mid, near_max_dist
    )
    log.info(
        "[extract] motif positions: fwd_slowed=%d  rev_slowed=%d  fwd_meth=%d  rev_meth=%d",
        len(fwd_slowed), len(rev_slowed), len(fwd_meth), len(rev_meth),
    )

    # Per-contig precompute: replaces all per-position dict lookups in the
    # BAM inner loop with O(1) array indexing. Memory cost ~14 bytes per
    # ref base — trivial for bacterial genomes.
    log.info("[extract] precomputing per-contig lookup arrays ...")
    contig_arrs = {
        contig: _build_contig_arrays(
            contig, seq, fwd_slowed, rev_slowed, fwd_near, rev_near,
            fwd_meth, rev_meth, baseline_min_dist_to_meth,
        )
        for contig, seq in ref_seqs.items()
    }

    samples: dict[int, list] = defaultdict(list)
    baseline_buffer: dict[int, list] = defaultdict(list)
    baseline_seen_per_kmer: dict[int, int] = defaultdict(int)
    n_slowed = n_near = n_baseline_seen = n_baseline_kept = 0
    n_reads_processed = n_reads_used = 0

    # Polymerase-frame meth_context offsets: the 11 positions [-LEFT..+RIGHT].
    # Built once; used to vectorise the per-row context slice.
    mc_offsets = np.arange(METH_CTX_LEN, dtype=np.int64) - METH_CTX_LEFT
    rev_offsets = np.array(REV_METH_OFFSETS, dtype=np.int64)

    with pysam.AlignmentFile(bam_path, "rb", check_sq=True) as bam:
        bam_contigs = set(bam.references)
        ref_contigs = set(ref_seqs.keys())
        missing = ref_contigs - bam_contigs
        extra = bam_contigs - ref_contigs
        if missing:
            log.warning("[extract] reference has contigs missing from BAM: %s", sorted(missing))
        if extra:
            log.warning("[extract] BAM has contigs missing from reference: %s", sorted(extra))

        for read in bam:
            n_reads_processed += 1
            if max_reads > 0 and n_reads_processed > max_reads:
                log.info("--max-reads %d reached — stopping early", max_reads)
                break
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            arrs = contig_arrs.get(read.reference_name)
            if arrs is None:
                continue
            try:
                ipd = read.get_tag("ip")
            except KeyError:
                try:
                    ipd = read.get_tag("fi")
                except KeyError:
                    continue
            try:
                pw = read.get_tag("pw")
            except KeyError:
                try:
                    pw = read.get_tag("fp")
                except KeyError:
                    continue
            ipd = np.asarray(ipd, dtype=np.float32)
            pw = np.asarray(pw, dtype=np.float32)
            n_reads_used += 1

            # Strand-route the per-position lookups: ``ip`` carries the
            # polymerase-template-strand methylations. is_reverse=False
            # means polymerase read − strand → use rev_* arrays.
            if read.is_reverse:
                slowed_T = arrs["fwd_slowed_T"]
                slowed_off = arrs["fwd_slowed_off"]
                slowed_frac = arrs["fwd_slowed_frac"]
                near_T = arrs["fwd_near_T"]
                near_off = arrs["fwd_near_off"]
                near_frac = arrs["fwd_near_frac"]
                pol_meth_arr = arrs["fwd_meth"]
                opp_meth_arr = arrs["rev_meth"]
                kmer_arr = arrs["kmer_rev"]   # polymerase 5'→3' frame is the rc
                mc_dir = +1                    # downstream pol = + ref direction
                rev_dir = +1
            else:
                slowed_T = arrs["rev_slowed_T"]
                slowed_off = arrs["rev_slowed_off"]
                slowed_frac = arrs["rev_slowed_frac"]
                near_T = arrs["rev_near_T"]
                near_off = arrs["rev_near_off"]
                near_frac = arrs["rev_near_frac"]
                pol_meth_arr = arrs["rev_meth"]
                opp_meth_arr = arrs["fwd_meth"]
                kmer_arr = arrs["kmer_fwd"]
                mc_dir = -1
                rev_dir = -1
            excluded = arrs["excluded"]
            n_ref = pol_meth_arr.shape[0]

            for read_pos, ref_pos in read.get_aligned_pairs(matches_only=True):
                if ref_pos < 0 or ref_pos >= n_ref:
                    continue
                kmer_id = int(kmer_arr[ref_pos])
                if kmer_id < 0:
                    continue
                s_T = int(slowed_T[ref_pos])
                if s_T:
                    cat = CATEGORY_SLOWED
                    parent_meth = s_T
                    parent_off = int(slowed_off[ref_pos])
                    frac = float(slowed_frac[ref_pos])
                    n_slowed += 1
                else:
                    nm_T = int(near_T[ref_pos])
                    if nm_T:
                        cat = CATEGORY_NEAR_METH
                        parent_meth = nm_T
                        parent_off = int(near_off[ref_pos])
                        frac = float(near_frac[ref_pos])
                        n_near += 1
                    else:
                        if excluded[ref_pos]:
                            continue
                        if baseline_sample_rate < 1.0 and rng.random() >= baseline_sample_rate:
                            continue
                        n_baseline_seen += 1
                        cat = CATEGORY_BASELINE
                        parent_meth = 0
                        parent_off = 0
                        frac = 0.0

                # Vectorised meth_context: 11 lookups on a padded array
                # would be cleaner, but bounds-mask + take is just as fast
                # and avoids a copy.
                ctx_idx = ref_pos + mc_dir * mc_offsets
                in_bounds = (ctx_idx >= 0) & (ctx_idx < n_ref)
                mc = np.where(in_bounds, pol_meth_arr[np.clip(ctx_idx, 0, n_ref - 1)], 0)

                rev_idx = ref_pos + rev_dir * rev_offsets
                in_bounds = (rev_idx >= 0) & (rev_idx < n_ref)
                rev_vals = np.where(in_bounds, opp_meth_arr[np.clip(rev_idx, 0, n_ref - 1)], 0)

                row = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
                row[0] = ipd[read_pos]
                row[1] = pw[read_pos]
                row[2] = frac
                row[3:14] = mc
                row[14:17] = rev_vals
                row[COL_CATEGORY] = cat
                row[COL_PARENT_METH] = parent_meth
                row[COL_PARENT_OFFSET] = parent_off

                if cat == CATEGORY_BASELINE:
                    baseline_seen_per_kmer[kmer_id] += 1
                    seen = baseline_seen_per_kmer[kmer_id]
                    if seen <= n_baseline_per_kmer:
                        baseline_buffer[kmer_id].append(row)
                        n_baseline_kept += 1
                    else:
                        j = int(rng.integers(0, seen))
                        if j < n_baseline_per_kmer:
                            baseline_buffer[kmer_id][j] = row
                else:
                    samples[kmer_id].append(row)

            if n_reads_used % PROGRESS_EVERY == 0:
                log.info(
                    "[extract] progress: %d reads used | slowed=%d near=%d baseline_seen=%d kept=%d",
                    n_reads_used, n_slowed, n_near, n_baseline_seen, n_baseline_kept,
                )

    for kid, rows in baseline_buffer.items():
        if rows:
            samples[kid].extend(rows)

    result: dict = {}
    for kid, rows in samples.items():
        if rows:
            result[int(kid)] = np.array(rows, dtype=np.float32)

    log.info(
        "[extract] DONE  reads_used=%d  slowed=%d  near=%d  baseline_seen=%d  baseline_kept=%d",
        n_reads_used, n_slowed, n_near, n_baseline_seen, n_baseline_kept,
    )
    log.info("[extract] kmers in output: %d", len(result))
    return result


# ---------------------------------------------------------------------------
# Public entry — called from kinsim/extract.py when manifest provides ref_path
# ---------------------------------------------------------------------------


def extract_to_shard(
    bam_path: str,
    ref_path: str,
    motif_string: str,
    output_path: str,
    *,
    meth_types: set[str] | None = None,
    n_baseline_per_kmer: int = 50,
    baseline_min_dist_to_meth: int | None = None,
    baseline_sample_rate: float | None = None,
    near_max_dist: int | None = None,
    seed: int = 42,
    max_reads: int = -1,
) -> None:
    """Extract a shard from an aligned bystrandified BAM with strand-aware kinetics.

    The output pkl is a dict[kmer_id (int) -> ndarray(N, 38)] following
    the same layout as ``kinsim/extract.py``. Refine, train, analyze
    operate identically on the resulting shards.
    """
    cfg = load_kinsim_config()
    extract_cfg = cfg.get("extract") or {}
    if baseline_min_dist_to_meth is None:
        baseline_min_dist_to_meth = int(extract_cfg.get("baseline_min_dist_to_meth", K))
    if baseline_sample_rate is None:
        baseline_sample_rate = float(extract_cfg.get("baseline_sample_rate", 0.10))
    if near_max_dist is None:
        near_max_dist = int(extract_cfg.get("near_meth_max_dist", 7))

    log.info("[aligned] BAM: %s", bam_path)
    log.info("[aligned] REF: %s", ref_path)
    log.info("[aligned] motifs: %s", motif_string)
    log.info(
        "[aligned] knobs: n_baseline_per_kmer=%d  baseline_min_dist=%d  "
        "near_max_dist=%d  baseline_sample_rate=%.2f  seed=%d",
        n_baseline_per_kmer, baseline_min_dist_to_meth,
        near_max_dist, baseline_sample_rate, seed,
    )

    if meth_types is not None:
        motif_string = filter_motif_string_by_types(motif_string, meth_types)
        log.info("[aligned] filtered motifs to types %s: %s", sorted(meth_types), motif_string)

    ref_seqs = load_reference(ref_path)
    log.info("[aligned] loaded reference: %d contigs, total %d bp",
             len(ref_seqs), sum(len(s) for s in ref_seqs.values()))

    samples = _extract_one_bam(
        bam_path=bam_path,
        motif_string=motif_string,
        ref_seqs=ref_seqs,
        cfg=cfg,
        n_baseline_per_kmer=n_baseline_per_kmer,
        baseline_min_dist_to_meth=baseline_min_dist_to_meth,
        baseline_sample_rate=baseline_sample_rate,
        near_max_dist=near_max_dist,
        seed=seed,
        max_reads=max_reads,
    )

    samples["__meta__"] = {
        "extract_path": "aligned",
        "kinsim_version": _KINSIM_VERSION,
        "created": datetime.datetime.utcnow().isoformat(),
        "source_bam": bam_path,
        "source_ref": ref_path,
        "motif_string": motif_string,
        "n_baseline_per_kmer": n_baseline_per_kmer,
        "baseline_min_dist_to_meth": baseline_min_dist_to_meth,
        "baseline_sample_rate": baseline_sample_rate,
        "near_max_dist": near_max_dist,
        "seed": seed,
    }
    log.info("[aligned] writing shard (atomic): %s", output_path)
    atomic_write_pickle(samples, Path(output_path))
    n_kmers = sum(1 for k in samples if isinstance(k, (int, np.integer)))
    log.info("[aligned] shard saved: %s  kmers=%d", output_path, n_kmers)


# ---------------------------------------------------------------------------
# Manifest-mode driver — invoked by ``kinsim extract --manifest ...``
# ---------------------------------------------------------------------------


def extract_from_manifest_task(
    manifest_path: str,
    task_index: int,
    output_dir: str,
    *,
    n_baseline_per_kmer: int = 50,
    baseline_min_dist_to_meth: int | None = None,
    baseline_sample_rate: float | None = None,
    near_max_dist: int | None = None,
    seed: int = 42,
    max_reads: int = -1,
    meth_types: set[str] | None = None,
) -> None:
    """Extract one manifest row to ``<output_dir>/<sample_id>_shard.pkl``.

    Picks row ``task_index`` (1-based, matches ``$SLURM_ARRAY_TASK_ID``)
    and runs orientation-aware extraction. ``ref_path`` is REQUIRED in
    the manifest — without alignment, raw HiFi BAMs cannot be processed
    correctly (per-read strand orientation is ambiguous and the kinetic
    signal averages away).
    """
    import os as _os

    from .utils.config import load_manifest
    from .utils.motifs import load_motif_string as _load_motif_string

    entries = load_manifest(manifest_path)
    if task_index < 1 or task_index > len(entries):
        log.error("Task index %d out of range (manifest has %d entries).",
                  task_index, len(entries))
        sys.exit(1)
    entry = entries[task_index - 1]
    log.info("task %d/%d: %s", task_index, len(entries), entry.sample_id)

    if not entry.ref_path:
        log.error(
            "Manifest entry '%s' is missing 'ref_path'. KinSim extract "
            "requires aligned bystrandified BAMs (with ip/pw tags + a "
            "reference FASTA). Pre-process raw HiFi BAMs with "
            "ccs-kinetics-bystrandify + pbmm2 first; see "
            "slurm_kinsim/strepto/0[01]_*.slurm for the prep pipeline.",
            entry.sample_id,
        )
        sys.exit(1)
    if not Path(entry.ref_path).exists():
        log.error("ref_path does not exist for %s: %s",
                  entry.sample_id, entry.ref_path)
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_pkl = _os.path.join(output_dir, f"{entry.sample_id}_shard.pkl")
    log.info("  Output: %s", output_pkl)

    motif_string = _load_motif_string(entry.motifs)
    if not motif_string:
        log.warning("No motifs resolved for '%s' — SKIPPING.", entry.sample_id)
        return

    extract_to_shard(
        bam_path=entry.bam_path,
        ref_path=entry.ref_path,
        motif_string=motif_string,
        output_path=output_pkl,
        meth_types=meth_types,
        n_baseline_per_kmer=n_baseline_per_kmer,
        baseline_min_dist_to_meth=baseline_min_dist_to_meth,
        baseline_sample_rate=baseline_sample_rate,
        near_max_dist=near_max_dist,
        seed=seed,
        max_reads=max_reads,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    """``kinsim extract`` CLI — aligned bystrandified BAMs only.

    Manifest mode (recommended for SLURM array jobs)::

        kinsim extract --manifest manifest.csv --task $SLURM_ARRAY_TASK_ID \\
                       --output-dir shards/

    Single-BAM mode (testing / one-off)::

        kinsim extract <aligned_bam> <ref_fasta> <motifs> <output.pkl>
    """
    import argparse
    import sys as _sys

    from .utils.config import setup_logging

    if argv is None:
        argv = _sys.argv[1:]
    # Tolerate ``run(["extract", *rest])`` from __main__.py:
    if argv and argv[0] == "extract":
        argv = argv[1:]

    p = argparse.ArgumentParser(
        prog="kinsim extract",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", help="Manifest CSV with ref_path column")
    p.add_argument("--task", type=int, help="1-based row index from manifest")
    p.add_argument("--output-dir", help="Output directory for shards (manifest mode)")
    p.add_argument("--n-baseline-per-kmer", type=int, default=50)
    p.add_argument("--max-reads", type=int, default=-1)
    p.add_argument("--meth-types", default=None,
                   help="Comma-separated subset (e.g. 'm6A,m4C'); default = all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("positional", nargs="*", help="<aligned_bam> <ref> <motifs> <output> (single-BAM mode)")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    from .utils.motifs import parse_meth_types_arg
    meth_types = parse_meth_types_arg(args.meth_types)

    if args.manifest:
        if args.task is None or not args.output_dir:
            p.error("--manifest requires --task and --output-dir")
        extract_from_manifest_task(
            manifest_path=args.manifest,
            task_index=args.task,
            output_dir=args.output_dir,
            n_baseline_per_kmer=args.n_baseline_per_kmer,
            seed=args.seed,
            max_reads=args.max_reads,
            meth_types=meth_types,
        )
        return

    if len(args.positional) != 4:
        p.error("single-BAM mode needs exactly 4 positional args: "
                "<aligned_bam> <ref_fasta> <motifs> <output.pkl>")
    bam, ref, motifs_arg, out_pkl = args.positional
    from .utils.motifs import load_motif_string

    motif_string = load_motif_string(motifs_arg)
    if not motif_string:
        log.error("No motifs resolved from '%s'", motifs_arg)
        _sys.exit(1)

    extract_to_shard(
        bam_path=bam,
        ref_path=ref,
        motif_string=motif_string,
        output_path=out_pkl,
        meth_types=meth_types,
        n_baseline_per_kmer=args.n_baseline_per_kmer,
        seed=args.seed,
        max_reads=args.max_reads,
    )


if __name__ == "__main__":
    main()
