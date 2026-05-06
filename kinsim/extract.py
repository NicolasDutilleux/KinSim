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
from .utils.encoding import KMER_LEFT_PAD, K, encode_kmer, get_meth_ids
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
    PROFILE_LEN,
    PROFILE_START,
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
            log.warning("extract_aligned: invalid mod_pos for '%s' (%s) — skipped", seq, pos)
            continue
        try:
            _validate_mod_pos(seq, mod_pos, m_type)
        except ValueError as exc:
            log.warning("extract_aligned: motif '%s' (%s) failed validation: %s — skipped",
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
) -> tuple[dict, dict, dict, dict]:
    """Pre-scan the reference and build per-strand methylation maps.

    Returns
    -------
    fwd_slowed_map, rev_slowed_map : dict[(contig, ref_pos)] → (meth_id, parent_offset)
        Positions where a methylation's signature offset lands. The
        ``fwd_*`` map covers methylations on the + reference strand
        (signature at ``mod_position + k`` in + strand direction); the
        ``rev_*`` map covers methylations on the − reference strand
        (signature at ``mod_position - k`` in + ref coords because
        − strand 5'→3' = + strand 3'→5').
    fwd_meth_map, rev_meth_map : dict[(contig, ref_pos)] → meth_id
        All methylated base positions per strand (the "centre" — only
        the modified base itself, not signature offsets). Used for
        meth_context filling and for the near_meth proximity check.
    """
    fwd_slowed: dict = {}
    rev_slowed: dict = {}
    fwd_meth: dict = {}
    rev_meth: dict = {}

    for contig, seq in ref_seqs.items():
        rc_seq = reverse_complement(seq)
        seq_len = len(seq)

        # Forward-strand motif occurrences: scan + strand reference.
        # Methylated A position = match.start() + mod_pos (in + ref coords).
        # Signature offsets propagate downstream on + strand → ref_pos + k.
        for motif in motifs:
            sig_off = sig_offsets_by_mid.get(motif["id"], [0])
            for match in motif["pattern"].finditer(seq):
                meth_pos = match.start() + motif["mod_pos"]
                if not (0 <= meth_pos < seq_len):
                    continue
                fwd_meth[(contig, meth_pos)] = motif["id"]
                for k in sig_off:
                    tgt = meth_pos + k
                    if 0 <= tgt < seq_len:
                        # Last-writer-wins on overlap, but prefer smaller |k|
                        # (closest to the modification — biophysically the
                        # primary signal).
                        existing = fwd_slowed.get((contig, tgt))
                        if existing is None or abs(k) < abs(existing[1]):
                            fwd_slowed[(contig, tgt)] = (motif["id"], int(k))

        # Reverse-strand motif occurrences: scan − strand (= rc of + strand).
        # rc_match.start() + motif["mod_pos"] = methylated A position in rc coords.
        # Convert to + ref coords: ref_pos = seq_len - 1 - rc_pos.
        # Signature offsets propagate downstream on − strand → ref_pos - k
        # (since − strand 5'→3' = + strand 3'→5', so "downstream" on − is
        # numerically smaller in + ref coords).
        for motif in motifs:
            sig_off = sig_offsets_by_mid.get(motif["id"], [0])
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
                            rev_slowed[(contig, tgt)] = (motif["id"], int(k))

    # Add near_meth window: positions within near_max_dist of a meth on the
    # same strand that are NOT already in slowed. We mark these so the
    # extraction loop can route them to CATEGORY_NEAR_METH (keeping refine's
    # negative-control samples).
    fwd_near: dict = {}
    rev_near: dict = {}
    for (contig, p), m_id in fwd_meth.items():
        for k in range(1, near_max_dist + 1):
            tgt = p + k
            if (contig, tgt) in fwd_slowed:
                continue
            existing = fwd_near.get((contig, tgt))
            if existing is None or abs(k) < abs(existing[1]):
                fwd_near[(contig, tgt)] = (m_id, k)
    for (contig, p), m_id in rev_meth.items():
        for k in range(1, near_max_dist + 1):
            tgt = p - k
            if (contig, tgt) in rev_slowed:
                continue
            existing = rev_near.get((contig, tgt))
            if existing is None or abs(k) < abs(existing[1]):
                rev_near[(contig, tgt)] = (m_id, k)

    return fwd_slowed, rev_slowed, fwd_near, rev_near, fwd_meth, rev_meth


# ---------------------------------------------------------------------------
# Per-position kmer + meth_context extraction from REFERENCE
# ---------------------------------------------------------------------------


def _kmer_at_ref(ref_seq: str, ref_pos: int, is_reverse: bool) -> tuple[int, bool]:
    """Encode the 11-base kmer centred at ``ref_pos`` on the strand the polymerase READ.

    For ``is_reverse=False`` (ip reads − strand), we encode the − strand
    kmer at ref_pos. The kmer is read 5'→3' on the − strand, which is
    the reverse-complement of the + strand kmer at ref_pos.

    For ``is_reverse=True`` (ip reads + strand), we encode the + strand
    kmer at ref_pos directly.

    Returns (kmer_id, valid). valid=False if the window is out of range
    or contains non-ACGT bases.
    """
    n = len(ref_seq)
    lo = ref_pos - KMER_LEFT_PAD
    hi = ref_pos + (K - KMER_LEFT_PAD)  # exclusive
    if lo < 0 or hi > n:
        return 0, False
    window = ref_seq[lo:hi]
    if not is_reverse:
        # Polymerase read − strand template; the kmer in the polymerase's
        # frame is the rev-comp of the + strand kmer at this window.
        window = reverse_complement(window)
    if any(b not in "ACGT" for b in window):
        return 0, False
    return encode_kmer(window), True


def _meth_context_at(
    contig: str,
    ref_pos: int,
    is_reverse: bool,
    fwd_meth_map: dict,
    rev_meth_map: dict,
) -> np.ndarray:
    """Build the meth_context array for a sample at ``ref_pos`` on the polymerase strand.

    The polymerase-strand meth at offset k (in polymerase 5'→3' frame)
    corresponds to:
       is_reverse=False (polymerase frame == − strand 5'→3'): ref_pos − k
       is_reverse=True  (polymerase frame == + strand 5'→3'): ref_pos + k

    We use the polymerase-strand methylation map for each direction
    (rev_meth_map for is_reverse=False, fwd_meth_map for is_reverse=True),
    so meth_context flags reflect the strand whose kinetics we're storing.
    """
    mc = np.zeros(METH_CTX_LEN, dtype=np.int8)
    pol_map = fwd_meth_map if is_reverse else rev_meth_map
    for k in range(METH_CTX_LEN):
        offset = k - METH_CTX_LEFT  # in polymerase 5'→3' frame
        if is_reverse:
            ref_off = ref_pos + offset
        else:
            ref_off = ref_pos - offset
        m_id = pol_map.get((contig, ref_off))
        if m_id is not None:
            mc[k] = m_id
    return mc


# ---------------------------------------------------------------------------
# Per-read extraction
# ---------------------------------------------------------------------------


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
    """Run orientation-aware extraction on one aligned BAM. Returns dict[kmer_id] → ndarray(N, 38)."""

    rng = np.random.default_rng(seed)
    motifs = _parse_motif_string_no_rc(motif_string)
    if not motifs:
        raise ValueError("extract_aligned: no valid motifs after parsing.")

    # Resolve signal offsets per meth type from kinsim_config.yaml.
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
    log.info("[aligned] signal offsets: %s",
             {name_by_mid[mid]: offs for mid, offs in sig_offsets_by_mid.items()})

    log.info("[aligned] pre-scanning reference for motif positions (per strand) ...")
    fwd_slowed, rev_slowed, fwd_near, rev_near, fwd_meth, rev_meth = _build_strand_maps(
        ref_seqs, motifs, sig_offsets_by_mid, near_max_dist
    )
    log.info(
        "[aligned] motif positions: fwd_slowed=%d  rev_slowed=%d  fwd_meth=%d  rev_meth=%d",
        len(fwd_slowed), len(rev_slowed), len(fwd_meth), len(rev_meth),
    )

    # --------- Iterate aligned reads ---------------------------------
    samples: dict[int, list] = defaultdict(list)
    baseline_buffer: dict[int, list] = defaultdict(list)
    baseline_seen_per_kmer: dict[int, int] = defaultdict(int)
    n_slowed = n_near = n_baseline_seen = n_baseline_kept = 0
    n_reads_processed = n_reads_used = 0

    # Pre-compute fast lookups by contig for the per-read inner loop.
    # We'll filter with `(contig, pos) in dict` directly — Python dict
    # lookups are O(1).
    with pysam.AlignmentFile(bam_path, "rb", check_sq=True) as bam:
        # Sanity check: BAM must be aligned to the reference whose contigs
        # match what we loaded.
        bam_contigs = set(bam.references)
        ref_contigs = set(ref_seqs.keys())
        missing = ref_contigs - bam_contigs
        extra = bam_contigs - ref_contigs
        if missing:
            log.warning("[aligned] reference has contigs missing from BAM: %s", sorted(missing))
        if extra:
            log.warning("[aligned] BAM has contigs missing from reference: %s", sorted(extra))

        for read in bam:
            n_reads_processed += 1
            if max_reads > 0 and n_reads_processed > max_reads:
                log.info("--max-reads %d reached — stopping early", max_reads)
                break
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            contig = read.reference_name
            ref_seq = ref_seqs.get(contig)
            if ref_seq is None:
                continue
            try:
                ipd = read.get_tag("ip")
            except KeyError:
                try:
                    ipd = read.get_tag("fi")  # fallback
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

            # Strand-route the methylation map: ``ip`` represents the IPD
            # of the polymerase synthesising this read. The polymerase
            # was reading the OPPOSITE strand of the read sequence as
            # template. So ``ip`` carries methylations on the strand
            # opposite to the read's aligned orientation.
            if read.is_reverse:
                slowed_map = fwd_slowed
                near_map = fwd_near
                opp_meth_map = rev_meth
            else:
                slowed_map = rev_slowed
                near_map = rev_near
                opp_meth_map = fwd_meth

            # Walk aligned positions and emit rows per category.
            # ``aligned_pairs`` returns (read_pos, ref_pos) only for
            # matched (no soft-clip, no indel) positions.
            for read_pos, ref_pos in read.get_aligned_pairs(matches_only=True):
                # SLOWED: this ref_pos hits a signature offset on the
                # polymerase-template strand for this read.
                slowed_hit = slowed_map.get((contig, ref_pos))
                near_hit = None if slowed_hit else near_map.get((contig, ref_pos))

                if slowed_hit is None and near_hit is None:
                    # Baseline candidate. Filter by distance to any meth
                    # position on EITHER strand to avoid contaminating
                    # baseline with positions inside the polymerase
                    # footprint of a methylation.
                    is_base = True
                    for d in range(1, baseline_min_dist_to_meth + 1):
                        if (
                            (contig, ref_pos + d) in fwd_meth
                            or (contig, ref_pos - d) in fwd_meth
                            or (contig, ref_pos + d) in rev_meth
                            or (contig, ref_pos - d) in rev_meth
                        ):
                            is_base = False
                            break
                    if not is_base:
                        continue
                    if baseline_sample_rate < 1.0 and rng.random() >= baseline_sample_rate:
                        continue
                    n_baseline_seen += 1
                    cat = CATEGORY_BASELINE
                    parent_meth = 0
                    parent_off = 0
                else:
                    if slowed_hit is not None:
                        cat = CATEGORY_SLOWED
                        parent_meth, parent_off = slowed_hit
                        n_slowed += 1
                    else:
                        cat = CATEGORY_NEAR_METH
                        parent_meth, parent_off = near_hit
                        n_near += 1

                # Encode kmer in polymerase-strand frame (so the same kmer
                # context is consistent across reads regardless of alignment
                # direction).
                kmer_id, kmer_valid = _kmer_at_ref(ref_seq, ref_pos, read.is_reverse)
                if not kmer_valid:
                    continue

                row = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
                row[0] = ipd[read_pos]
                row[1] = pw[read_pos]
                row[2] = 1.0  # frac is per-position; could be filled from motif map later
                # meth_context [-7..+3] in polymerase-strand frame
                row[3:3 + METH_CTX_LEN] = _meth_context_at(
                    contig, ref_pos, read.is_reverse, fwd_meth, rev_meth
                )
                # kinetic profile downstream — the next 9 polymerase positions.
                # In polymerase frame, "downstream" means later in the read,
                # so read_pos + 0 .. read_pos + 8.
                for k in range(PROFILE_LEN):
                    rp = read_pos + PROFILE_START + k
                    if 0 <= rp < len(ipd):
                        row[14 + k] = ipd[rp]
                        row[23 + k] = pw[rp]
                # rev_meth: methylations on the OPPOSITE strand at active-site
                # neighbours. In polymerase frame, the opposite strand is the
                # read sequence's strand (which the polymerase did NOT read as
                # template). So we look up ``opp_meth_map`` at the corresponding
                # ref_pos.
                for k, off in enumerate(REV_METH_OFFSETS):
                    if read.is_reverse:
                        ref_off = ref_pos + off
                    else:
                        ref_off = ref_pos - off
                    rev_id = opp_meth_map.get((contig, ref_off))
                    if rev_id is not None:
                        row[32 + k] = rev_id
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
                        # Vitter's reservoir
                        j = int(rng.integers(0, seen))
                        if j < n_baseline_per_kmer:
                            baseline_buffer[kmer_id][j] = row
                else:
                    samples[kmer_id].append(row)

            if n_reads_used % PROGRESS_EVERY == 0:
                log.info(
                    "[aligned] progress: %d reads used | slowed=%d near=%d baseline_seen=%d kept=%d",
                    n_reads_used, n_slowed, n_near, n_baseline_seen, n_baseline_kept,
                )

    # Merge baseline buffer into samples
    for kid, rows in baseline_buffer.items():
        if rows:
            samples[kid].extend(rows)

    # Pack to ndarrays
    result: dict = {}
    for kid, rows in samples.items():
        if rows:
            result[int(kid)] = np.array(rows, dtype=np.float32)

    log.info(
        "[aligned] DONE  reads_used=%d  slowed=%d  near=%d  baseline_seen=%d  baseline_kept=%d",
        n_reads_used, n_slowed, n_near, n_baseline_seen, n_baseline_kept,
    )
    log.info("[aligned] kmers in output: %d", len(result))
    return result


# ---------------------------------------------------------------------------
# Public entry — called from kinsim/extract.py when manifest provides ref_path
# ---------------------------------------------------------------------------


def extract_aligned_to_shard(
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

    extract_aligned_to_shard(
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

    extract_aligned_to_shard(
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
