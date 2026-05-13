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
``dict[kmer_id (int) → np.ndarray(N, 20)]`` plus ``"__meta__"``. The
20-column layout is defined in :mod:`kinsim.utils.sample_layout`. Refine,
train, analyze operate row-by-row on this layout.
"""

from __future__ import annotations

import datetime
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysam

from .utils.config import (
    ExtractionParams,
    get_extraction_params,
    load_kinsim_config,
)
from .utils.encoding import get_meth_ids
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
    SampleLayout,
    get_sample_layout,
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
    params: ExtractionParams,
) -> dict:
    """Materialise per-position lookup arrays for one contig.

    Replaces the per-position dict lookups in the BAM inner loop with
    O(1) array indexing. Memory is ~14 bytes per ref base — negligible
    for bacterial genomes — and reads of millions of positions become
    measurably faster.

    Args:
        contig:             Contig name (used to filter the per-strand meth dicts).
        seq:                The contig sequence as a plain string.
        fwd_slowed/...:     Pre-built meth-state dicts (see :func:`_build_strand_maps`).
        baseline_min_dist:  Minimum distance (bases) from any meth to flag a
                            position as a baseline candidate.
        params:             Window geometry. Drives the kmer-encoding shift loop,
                            the boundary mask, and the inner-slice padding so the
                            same code handles arbitrary ``kmer_size``.
    """
    kmer_size = params.kmer_size
    upstream = params.upstream
    downstream = params.downstream

    n = len(seq)
    base_int = np.full(n, -1, dtype=np.int8)  # -1 → non-ACGT
    base_int[np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("A")] = 0
    base_int[np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("C")] = 1
    base_int[np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("G")] = 2
    base_int[np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord("T")] = 3
    valid_base = base_int >= 0

    # Sliding kmer encoding on the + strand. window[i] covers ref positions
    # i-upstream .. i+downstream; out-of-range → invalid.
    # Start the accumulator at 0 (not -1) — left-shifting a negative number
    # leaves the sign bit set forever, so every kmer would come out negative
    # and the inner loop would skip every position. int64 accommodates K up
    # to 31 (2K bits), enforced by ExtractionParams.
    kmer_fwd = np.zeros(n, dtype=np.int64)
    kmer_window_valid = np.ones(n, dtype=bool)
    for j in range(kmer_size):
        # In + strand 5'→3' frame: the j-th base of the kmer at ref_pos i
        # sits at ref position (i - upstream + j).
        offset = j - upstream
        # roll() gives wrap-around; we mask out boundary positions below.
        rolled = np.roll(base_int, -offset)
        kmer_window_valid &= np.roll(valid_base, -offset)
        kmer_fwd = (kmer_fwd << 2) | (rolled.astype(np.int64) & 3)
    # Boundary mask: any position whose kmer reaches outside [0, n) is invalid.
    # The kmer at ref_pos i needs positions [i-upstream, i+downstream].
    edge_invalid = np.zeros(n, dtype=bool)
    edge_invalid[:upstream] = True
    if downstream > 0:
        edge_invalid[n - downstream:] = True
    kmer_fwd[edge_invalid | ~kmer_window_valid] = -1

    # Reverse-complement kmer at the same ref position: the polymerase reading
    # the − strand sees a different kmer here. Bit-reverse pairs of the +
    # strand kmer and complement them (XOR with 0b1111... pattern over kmer
    # bases). Each 2-bit unit XORs with 3 (A↔T, C↔G).
    xor_mask = np.int64(0)
    for _ in range(kmer_size):
        xor_mask = (xor_mask << 2) | 3
    # Bit-reverse the kmer base-pairs:
    src = kmer_fwd.copy()
    kmer_rev = np.zeros_like(src)
    for _ in range(kmer_size):
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

    # Pre-pad meth arrays so the inner-loop meth_context slice never needs
    # bounds-checking. PAD must cover the widest offset the hot path will
    # ever touch — the forward meth-context window reaches ±max(upstream,
    # downstream), and the rev_meth window reaches max(|off|) for off in
    # rev_meth_offsets. Taking the max of all three keeps the slice safe
    # under arbitrary geometries.
    rev_max_abs = max((abs(int(o)) for o in params.rev_meth_offsets), default=0)
    pad = max(upstream, downstream, rev_max_abs)
    fwd_meth_padded = np.zeros(n + 2 * pad, dtype=np.int8)
    fwd_meth_padded[pad:pad + n] = fwd_meth_arr
    rev_meth_padded = np.zeros(n + 2 * pad, dtype=np.int8)
    rev_meth_padded[pad:pad + n] = rev_meth_arr

    return {
        "kmer_fwd": kmer_fwd,
        "kmer_rev": kmer_rev,
        "fwd_meth_padded": fwd_meth_padded,
        "rev_meth_padded": rev_meth_padded,
        "pad": pad,
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


def _check_bystrandified(bam_path: str, n_peek: int = 50) -> None:
    """Fail fast if the BAM looks like raw HiFi (not bystrandified).

    Raw HiFi reads carry both forward and reverse pass kinetic tags
    (``fi``, ``fp``, ``ri``, ``rp``). Bystrandified BAMs split each CCS
    into two reads (one per polymerase pass) and store a single
    ``ip``/``pw`` per read. Extract's strand routing assumes the latter:
    one read = one pass, ``read.is_reverse`` disambiguates which strand
    the polymerase templated.

    Running on an aligned-but-not-bystrandified BAM would silently fall
    back to ``fi``/``fp`` (forward pass only), losing half the data and
    misattributing which strand each read's IPDs reflect. This sniff
    peeks at the first ``n_peek`` primary mapped reads and bails with a
    clear error if it finds the ``ri`` tag — the unambiguous marker of
    raw HiFi.
    """
    log.info("[extract] sniffing %s for bystrandified format ...", bam_path)
    saw_ip = saw_ri = saw_fi = 0
    n_seen = 0
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            n_seen += 1
            if read.has_tag("ip"):
                saw_ip += 1
            if read.has_tag("ri"):
                saw_ri += 1
            if read.has_tag("fi"):
                saw_fi += 1
            if n_seen >= n_peek:
                break
    log.info(
        "[extract] sniff (%d primary reads peeked): ip=%d  fi=%d  ri=%d",
        n_seen, saw_ip, saw_fi, saw_ri,
    )
    if n_seen == 0:
        log.warning("[extract] BAM has no primary mapped reads — sniff inconclusive.")
        return
    if saw_ri > 0:
        raise RuntimeError(
            f"BAM '{bam_path}' looks like raw HiFi (found 'ri' tag — both "
            "fwd and rev kinetic passes in the same read). KinSim extract "
            "requires a *bystrandified* BAM where each polymerase pass is its "
            "own read with a single ip/pw array. Run "
            "ccs-kinetics-bystrandify before pbmm2 alignment, or use the "
            "prep pipeline at slurm_kinsim/<dataset>/00_bystrandify.slurm."
        )
    if saw_ip == 0 and saw_fi > 0:
        raise RuntimeError(
            f"BAM '{bam_path}' has 'fi' tags but no 'ip' tags — looks like "
            "raw HiFi (kinetics tagged but not bystrandified). KinSim extract "
            "will not silently mix forward and reverse passes. Run "
            "ccs-kinetics-bystrandify before pbmm2 alignment to produce "
            "one read per polymerase pass with a single ip/pw array."
        )


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
    params: ExtractionParams,
) -> dict:
    """Run orientation-aware extraction.

    Args:
        params: Window-geometry record. The forward meth-context slice has
            length ``params.kmer_size`` and runs from ``-upstream`` to
            ``+downstream`` around each aligned position; ``rev_meth`` is
            gathered at the offsets in ``params.rev_meth_offsets``. All
            inner-loop slice indices are precomputed from ``params`` once,
            outside the read loop, so the hot path stays a fixed-width
            numpy slice regardless of the configured geometry.

    Returns:
        ``dict[kmer_id (int)] -> np.ndarray(N, params.sample_ncols)``.
    """

    _check_bystrandified(bam_path)

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
            fwd_meth, rev_meth, baseline_min_dist_to_meth, params,
        )
        for contig, seq in ref_seqs.items()
    }

    samples: dict[int, list] = defaultdict(list)
    baseline_buffer: dict[int, list] = defaultdict(list)
    baseline_seen_per_kmer: dict[int, int] = defaultdict(int)
    n_slowed = n_near = n_baseline_seen = n_baseline_kept = 0
    n_reads_processed = n_reads_used = 0

    # Pad on each meth_padded array (matches the value computed in
    # _build_contig_arrays). All inner-loop slices index into the padded
    # arrays so no per-row bounds-checking is needed.
    rev_max_abs = max((abs(int(o)) for o in params.rev_meth_offsets), default=0)
    pad = max(params.upstream, params.downstream, rev_max_abs)
    upstream = params.upstream
    downstream = params.downstream
    # rev_meth offsets in the polymerase 5'→3' frame. For + strand reads
    # (pol_reverse_slice=True), the polymerase reads the reference backward,
    # so we gather at ``ppos - off``; for − strand reads we gather at
    # ``ppos + off``. Both arrays are int64 numpy so the fancy index is fast.
    rev_offsets_fwd = np.array(params.rev_meth_offsets, dtype=np.int64)   # − strand pol
    rev_offsets_rev = -rev_offsets_fwd                                    # + strand pol

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
            # ip/pw are the bystrandified per-pass kinetic tags. Hard-fail
            # if absent — falling back to fi/fp would silently mix forward
            # and reverse passes (the orientation ambiguity bug). The
            # bystrandify sniff at the top of this function should have
            # caught this, so reaching here means the BAM is malformed.
            try:
                ipd = read.get_tag("ip")
                pw = read.get_tag("pw")
            except KeyError as exc:
                raise RuntimeError(
                    f"Read {read.query_name!r}: missing ip/pw tag ({exc}). "
                    "BAM is not bystrandified — _check_bystrandified should "
                    "have failed earlier. This is a bug or a malformed BAM."
                ) from exc
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
                pol_padded = arrs["fwd_meth_padded"]
                opp_padded = arrs["rev_meth_padded"]
                kmer_arr = arrs["kmer_rev"]   # polymerase 5'→3' frame is the rc
                pol_reverse_slice = False     # mc/rev read forward in pol frame
            else:
                slowed_T = arrs["rev_slowed_T"]
                slowed_off = arrs["rev_slowed_off"]
                slowed_frac = arrs["rev_slowed_frac"]
                near_T = arrs["rev_near_T"]
                near_off = arrs["rev_near_off"]
                near_frac = arrs["rev_near_frac"]
                pol_padded = arrs["rev_meth_padded"]
                opp_padded = arrs["fwd_meth_padded"]
                kmer_arr = arrs["kmer_fwd"]
                pol_reverse_slice = True      # − strand polymerase reads ref backward
            excluded = arrs["excluded"]
            n_ref = excluded.shape[0]

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

                # Padded slices — no np.where, no np.clip per row.
                # Forward meth-context in polymerase frame:
                #   * + strand polymerase (pol_reverse_slice=True): the
                #     polymerase reads the reference 3'→5', so context
                #     [-upstream, +downstream] in the polymerase frame
                #     maps to ref positions [ppos - downstream, ppos +
                #     upstream] read in reverse.
                #   * − strand polymerase (pol_reverse_slice=False):
                #     context is read forward, [ppos - upstream, ppos +
                #     downstream].
                # Length of both slices is always ``params.kmer_size``.
                ppos = ref_pos + pad
                if pol_reverse_slice:
                    mc = pol_padded[ppos - downstream:ppos + upstream + 1][::-1]
                    rev = opp_padded[ppos + rev_offsets_rev]
                else:
                    mc = pol_padded[ppos - upstream:ppos + downstream + 1]
                    rev = opp_padded[ppos + rev_offsets_fwd]

                row = (
                    float(ipd[read_pos]), float(pw[read_pos]), frac,
                    *mc.tolist(),
                    *rev.tolist(),
                    cat, parent_meth, parent_off,
                )

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
    extraction_params: ExtractionParams | None = None,
) -> None:
    """Extract a shard from an aligned bystrandified BAM with strand-aware kinetics.

    The output pkl is a dict ``{kmer_id (int) -> ndarray(N, sample_ncols)}``
    following the layout in :mod:`kinsim.utils.sample_layout`. Refine, train
    and analyze operate identically on the resulting shards.

    Args:
        bam_path:           Path to a bystrandified, aligned BAM with ip/pw tags.
        ref_path:           Reference FASTA the BAM was aligned against.
        motif_string:       KinSim motif string (e.g. ``"m6A,GATC,1;..."``).
        output_path:        Destination shard pkl (written atomically).
        meth_types:         Optional whitelist of meth-type names to keep.
        n_baseline_per_kmer: Reservoir cap on baseline rows per kmer.
        baseline_min_dist_to_meth: Required ≥ ``kmer_size`` so the
                            meth-context window of a baseline never overlaps
                            a methylation. If ``None``, read from the YAML.
        baseline_sample_rate: Front-end skip probability for baseline candidates.
        near_max_dist:      Range scanned for NEAR_METH labelling.
        seed:               PRNG seed.
        max_reads:          Stop after this many reads (debug; -1 = all).
        extraction_params:  Window geometry to extract with. If ``None``, read
                            from ``kinsim_config.yaml`` via
                            :func:`~kinsim.utils.config.get_extraction_params`.
                            The resolved params are written into the shard's
                            ``__meta__["extraction_params"]`` block so
                            downstream consumers can validate compatibility.

    Raises:
        ValueError: If ``baseline_min_dist_to_meth < params.kmer_size`` (the
            baseline window would overlap motif positions), or if any other
            invariant is violated.
    """
    cfg = load_kinsim_config()
    extract_cfg = cfg.get("extract") or {}
    params = extraction_params or get_extraction_params()

    if baseline_min_dist_to_meth is None:
        baseline_min_dist_to_meth = int(extract_cfg.get("baseline_min_dist_to_meth", params.kmer_size))
    if baseline_sample_rate is None:
        baseline_sample_rate = float(extract_cfg.get("baseline_sample_rate", 0.10))
    if near_max_dist is None:
        near_max_dist = int(extract_cfg.get("near_meth_max_dist", 7))

    # Hard invariant — the meth_context window of a baseline candidate must
    # never overlap a methylation; otherwise BASELINE rows contain hidden
    # meth context, training learns a confounded signal, downstream metrics
    # become meaningless. Catching it here costs nothing and the alternative
    # is a silently corrupt corpus.
    if baseline_min_dist_to_meth < params.kmer_size:
        raise ValueError(
            f"extract: baseline_min_dist_to_meth ({baseline_min_dist_to_meth}) "
            f"is smaller than kmer_size ({params.kmer_size}). A baseline "
            f"position must be at least kmer_size bases away from any "
            f"methylation, otherwise its meth_context window would carry "
            f"hidden meth signal. Raise the value in `extract:` of "
            f"kinsim_config.yaml (typical: == kmer_size) and re-run."
        )

    log.info("[extract] BAM: %s", bam_path)
    log.info("[extract] REF: %s", ref_path)
    log.info("[extract] motifs: %s", motif_string)
    log.info(
        "[extract] window: kmer_size=%d  upstream=%d  downstream=%d  "
        "rev_meth_offsets=%s  sample_ncols=%d",
        params.kmer_size, params.upstream, params.downstream,
        list(params.rev_meth_offsets), params.sample_ncols,
    )
    log.info(
        "[extract] knobs: n_baseline_per_kmer=%d  baseline_min_dist=%d  "
        "near_max_dist=%d  baseline_sample_rate=%.2f  seed=%d",
        n_baseline_per_kmer, baseline_min_dist_to_meth,
        near_max_dist, baseline_sample_rate, seed,
    )

    if meth_types is not None:
        motif_string = filter_motif_string_by_types(motif_string, meth_types)
        log.info("[extract] filtered motifs to types %s: %s", sorted(meth_types), motif_string)

    ref_seqs = load_reference(ref_path)
    log.info("[extract] loaded reference: %d contigs, total %d bp",
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
        params=params,
    )

    samples["__meta__"] = {
        "extract_path": "aligned",
        "kinsim_version": _KINSIM_VERSION,
        "created": datetime.datetime.utcnow().isoformat(),
        "source_bam": bam_path,
        "source_ref": ref_path,
        "motif_string": motif_string,
        # Freeze the meth_id mapping at extract time. Train reads this from
        # the first shard's __meta__ to set num_meth_types — independent of
        # whatever kinsim_config.yaml looks like at train time, which may
        # have been edited or be missing.
        "meth_id_map": get_meth_ids(),
        # Window geometry — pinned at extraction time. The dataset compares
        # this against the active YAML on load and refuses to mix layouts.
        # Single source of truth for every consumer (refine, train, analyze).
        "extraction_params": params.to_dict(),
        "n_baseline_per_kmer": n_baseline_per_kmer,
        "baseline_min_dist_to_meth": baseline_min_dist_to_meth,
        "baseline_sample_rate": baseline_sample_rate,
        "near_max_dist": near_max_dist,
        "seed": seed,
    }
    log.info("[extract] writing shard (atomic): %s", output_path)
    atomic_write_pickle(samples, Path(output_path))
    n_kmers = sum(1 for k in samples if isinstance(k, (int, np.integer)))
    log.info("[extract] shard saved: %s  kmers=%d", output_path, n_kmers)


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
