"""``kinsim extract`` — bilateral v2 extract from raw HiFi aligned BAMs.

Reads ONE record per ZMW (raw HiFi format, ``fi``/``fp``/``ri``/``rp`` tags)
and produces per-position bilateral training samples. At each ref_pos
covered by an aligned read, one row stores:

* ``ipd_fwd``/``pw_fwd``: kinetics observed when the polymerase read the
  + ref strand as template → predicts + strand methylation.
* ``ipd_rev``/``pw_rev``: kinetics observed when the polymerase read the
  - ref strand as template → predicts - strand methylation.
* ``mc_fwd[K]``: full forward-strand meth context window.
* ``mc_rev[K]``: full reverse-strand meth context window.
* Two categories (one per strand): each position is independently
  baseline / slowed / near_meth on each strand.

Strand routing — PacBio raw HiFi convention::

    Pass 1: polymerase synthesises one strand by reading the OTHER as template.
    Pass 2: polymerase synthesises the complement.
    fi[i] = IPD at base i during pass 1.    ri[i] = IPD at base i during pass 2.

    is_reverse=False (read SEQ = + ref strand):
        fi reads -- strand template ⇒ - strand methylation kinetics
        ri reads + strand template ⇒ + strand methylation kinetics

    is_reverse=True (read SEQ = - ref strand):
        fi reads + strand template ⇒ + strand methylation kinetics
        ri reads - strand template ⇒ - strand methylation kinetics

After normalisation: ``ipd_fwd[ref_pos]`` is ALWAYS the IPD for + strand
methylation at that ref_pos, regardless of ``read.is_reverse``.

CLI::

    kinsim extract --manifest manifest.csv --task $SLURM_ARRAY_TASK_ID \\
                   --output-dir shards/
    kinsim extract <raw_hifi_aligned_bam> <ref_fasta> <motifs> <output.pkl>

Manifest schema::

    sample_id,bam_path,motifs,ref_path
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
)

try:
    from . import __version__ as _KINSIM_VERSION
except (ImportError, AttributeError):
    _KINSIM_VERSION = "unknown"

log = logging.getLogger(__name__)
PROGRESS_EVERY = 10000


def _parse_motif_string_no_rc(motif_string: str) -> list[dict]:
    """Parse a motif string into a list of dicts WITHOUT auto-revcomp."""
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
            log.warning("extract: motif '%s' (%s) failed validation: %s — skipped", seq, m_type, exc)
            continue
        m_id = get_meth_ids().get(m_type, 0)
        if m_id == 0:
            continue
        frac = float(parts[4]) if len(parts) >= 5 else 1.0
        motifs.append(
            {
                "pattern": re.compile(f"(?=({iupac_to_re(seq)}))"),
                "id": m_id,
                "mod_pos": mod_pos,
                "frac": frac,
                "seq": seq,
                "type": m_type,
            }
        )
    return motifs


def _build_strand_maps(
    ref_seqs: dict[str, str],
    motifs: list[dict],
    sig_offsets_by_mid: dict[int, list[int]],
    near_max_dist: int,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Pre-scan the reference and build per-strand methylation maps.

    Returns six dicts keyed by ``(contig, ref_pos)``:
      ``fwd_slowed`` / ``rev_slowed`` : (meth_id, parent_offset, frac)
      ``fwd_near``   / ``rev_near``   : same shape (NEAR_METH)
      ``fwd_meth``   / ``rev_meth``   : meth_id (canonical centre only)
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

    fwd_near: dict = {}
    rev_near: dict = {}
    fwd_meth_to_frac = {pos: fwd_slowed.get(pos, (0, 0, 1.0))[2] for pos in fwd_meth.keys()}
    rev_meth_to_frac = {pos: rev_slowed.get(pos, (0, 0, 1.0))[2] for pos in rev_meth.keys()}
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
    """Materialise per-position lookup arrays for one contig (bilateral v2).

    Per ref_pos, both strands' state arrays are kept in parallel. The
    forward-strand kmer (kmer_fwd_array) is what the bilateral storage
    uses; the model derives the reverse kmer via revcomp internally.
    """
    kmer_size = params.kmer_size
    upstream = params.upstream
    downstream = params.downstream
    n = len(seq)

    base_int = np.full(n, -1, dtype=np.int8)
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    base_int[seq_bytes == ord("A")] = 0
    base_int[seq_bytes == ord("C")] = 1
    base_int[seq_bytes == ord("G")] = 2
    base_int[seq_bytes == ord("T")] = 3
    valid_base = base_int >= 0

    # Forward-strand kmer encoding (the bilateral storage key).
    kmer_fwd = np.zeros(n, dtype=np.int64)
    kmer_window_valid = np.ones(n, dtype=bool)
    for j in range(kmer_size):
        offset = j - upstream
        rolled = np.roll(base_int, -offset)
        kmer_window_valid &= np.roll(valid_base, -offset)
        kmer_fwd = (kmer_fwd << 2) | (rolled.astype(np.int64) & 3)
    edge_invalid = np.zeros(n, dtype=bool)
    edge_invalid[:upstream] = True
    if downstream > 0:
        edge_invalid[n - downstream :] = True
    kmer_fwd[edge_invalid | ~kmer_window_valid] = -1

    # Methylation presence arrays.
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

    # Baseline exclusion = ±baseline_min_dist of any meth, either strand.
    meth_present = (fwd_meth_arr > 0) | (rev_meth_arr > 0)
    excluded = meth_present.copy()
    for d in range(1, baseline_min_dist + 1):
        excluded[:-d] |= meth_present[d:]
        excluded[d:] |= meth_present[:-d]

    # Pad meth arrays for the inner-loop K-window slices.
    pad = max(upstream, downstream)
    fwd_meth_padded = np.zeros(n + 2 * pad, dtype=np.int8)
    fwd_meth_padded[pad : pad + n] = fwd_meth_arr
    rev_meth_padded = np.zeros(n + 2 * pad, dtype=np.int8)
    rev_meth_padded[pad : pad + n] = rev_meth_arr

    return {
        "kmer_fwd": kmer_fwd,
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


def _check_raw_hifi(bam_path: str, n_peek: int = 50) -> None:
    """Fail fast if the BAM looks bystrandified (we now want raw HiFi)."""
    log.info("[extract] sniffing %s for raw HiFi format ...", bam_path)
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
    log.info("[extract] sniff (%d primary): ip=%d fi=%d ri=%d", n_seen, saw_ip, saw_fi, saw_ri)
    if n_seen == 0:
        log.warning("[extract] BAM has no primary mapped reads — sniff inconclusive.")
        return
    if saw_fi == 0 or saw_ri == 0:
        raise RuntimeError(
            f"BAM '{bam_path}' lacks fi or ri tag — bilateral extract needs raw "
            "HiFi aligned BAMs (1 rec/ZMW, both fi+ri tags present). "
            "If you have a bystrandified BAM, re-align the raw HiFi instead."
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
    """Bilateral extract from a raw HiFi aligned BAM.

    Returns: ``dict[kmer_id_fwd (int)] -> ndarray(N, params.sample_ncols)``.
    """
    _check_raw_hifi(bam_path)
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
    log.info(
        "[extract] signal offsets: %s",
        {name_by_mid[mid]: offs for mid, offs in sig_offsets_by_mid.items()},
    )

    log.info("[extract] pre-scanning reference for motif positions (per strand) ...")
    fwd_slowed, rev_slowed, fwd_near, rev_near, fwd_meth, rev_meth = _build_strand_maps(
        ref_seqs, motifs, sig_offsets_by_mid, near_max_dist
    )
    log.info(
        "[extract] motif positions: fwd_slowed=%d rev_slowed=%d fwd_meth=%d rev_meth=%d",
        len(fwd_slowed), len(rev_slowed), len(fwd_meth), len(rev_meth),
    )

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
    n_slowed_fwd = n_slowed_rev = n_near_fwd = n_near_rev = 0
    n_baseline_seen = n_baseline_kept = 0
    n_reads_processed = n_reads_used = 0

    upstream = params.upstream
    downstream = params.downstream
    pad_default = max(upstream, downstream)

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
                fi = read.get_tag("fi")
                fp = read.get_tag("fp")
                ri = read.get_tag("ri")
                rp = read.get_tag("rp")
            except KeyError as exc:
                raise RuntimeError(
                    f"Read {read.query_name!r}: missing fi/fp/ri/rp tag ({exc}). "
                    "Bilateral extract requires raw HiFi with all four kinetic tags."
                ) from exc
            fi = np.asarray(fi, dtype=np.float32)
            fp = np.asarray(fp, dtype=np.float32)
            ri = np.asarray(ri, dtype=np.float32)
            rp = np.asarray(rp, dtype=np.float32)
            n_reads_used += 1

            # Strand routing (PacBio raw HiFi convention).
            # ipd_fwd = + strand methylation kinetics; ipd_rev = - strand.
            if read.is_reverse:
                ipd_fwd_arr, pw_fwd_arr = fi, fp
                ipd_rev_arr, pw_rev_arr = ri, rp
            else:
                ipd_fwd_arr, pw_fwd_arr = ri, rp
                ipd_rev_arr, pw_rev_arr = fi, fp

            kmer_arr = arrs["kmer_fwd"]
            fwd_meth_padded = arrs["fwd_meth_padded"]
            rev_meth_padded = arrs["rev_meth_padded"]
            pad = arrs["pad"]
            fwd_slowed_T = arrs["fwd_slowed_T"]
            fwd_slowed_off = arrs["fwd_slowed_off"]
            fwd_slowed_frac = arrs["fwd_slowed_frac"]
            rev_slowed_T = arrs["rev_slowed_T"]
            rev_slowed_off = arrs["rev_slowed_off"]
            rev_slowed_frac = arrs["rev_slowed_frac"]
            fwd_near_T = arrs["fwd_near_T"]
            fwd_near_off = arrs["fwd_near_off"]
            rev_near_T = arrs["rev_near_T"]
            rev_near_off = arrs["rev_near_off"]
            excluded = arrs["excluded"]
            n_ref = excluded.shape[0]

            for read_pos, ref_pos in read.get_aligned_pairs(matches_only=True):
                if ref_pos < 0 or ref_pos >= n_ref:
                    continue
                kmer_id = int(kmer_arr[ref_pos])
                if kmer_id < 0:
                    continue

                # Per-strand category resolution.
                fwd_s = int(fwd_slowed_T[ref_pos])
                if fwd_s:
                    cat_fwd = CATEGORY_SLOWED
                    pm_fwd = fwd_s
                    po_fwd = int(fwd_slowed_off[ref_pos])
                    frac_fwd = float(fwd_slowed_frac[ref_pos])
                    n_slowed_fwd += 1
                else:
                    fwd_n = int(fwd_near_T[ref_pos])
                    if fwd_n:
                        cat_fwd = CATEGORY_NEAR_METH
                        pm_fwd = fwd_n
                        po_fwd = int(fwd_near_off[ref_pos])
                        frac_fwd = 0.0
                        n_near_fwd += 1
                    else:
                        cat_fwd = CATEGORY_BASELINE
                        pm_fwd = 0
                        po_fwd = 0
                        frac_fwd = 0.0

                rev_s = int(rev_slowed_T[ref_pos])
                if rev_s:
                    cat_rev = CATEGORY_SLOWED
                    pm_rev = rev_s
                    po_rev = int(rev_slowed_off[ref_pos])
                    frac_rev = float(rev_slowed_frac[ref_pos])
                    n_slowed_rev += 1
                else:
                    rev_n = int(rev_near_T[ref_pos])
                    if rev_n:
                        cat_rev = CATEGORY_NEAR_METH
                        pm_rev = rev_n
                        po_rev = int(rev_near_off[ref_pos])
                        frac_rev = 0.0
                        n_near_rev += 1
                    else:
                        cat_rev = CATEGORY_BASELINE
                        pm_rev = 0
                        po_rev = 0
                        frac_rev = 0.0

                # If BOTH strands are BASELINE, gate by excluded + sample rate.
                both_baseline = (cat_fwd == CATEGORY_BASELINE and cat_rev == CATEGORY_BASELINE)
                if both_baseline:
                    if excluded[ref_pos]:
                        continue
                    if baseline_sample_rate < 1.0 and rng.random() >= baseline_sample_rate:
                        continue
                    n_baseline_seen += 1

                # Bilateral meth-context windows (+ ref strand 5'→3' frame).
                ppos = ref_pos + pad
                mc_fwd = fwd_meth_padded[ppos - upstream : ppos + downstream + 1]
                mc_rev = rev_meth_padded[ppos - upstream : ppos + downstream + 1]

                # Fraction: take max across the two strands' active-site frac.
                # If only one strand has a meth, the other is 0 and max is the
                # meaningful one. If both are methylated, we keep the bigger.
                frac = max(frac_fwd, frac_rev)

                row = (
                    float(ipd_fwd_arr[read_pos]),
                    float(pw_fwd_arr[read_pos]),
                    float(ipd_rev_arr[read_pos]),
                    float(pw_rev_arr[read_pos]),
                    frac,
                    *mc_fwd.tolist(),
                    *mc_rev.tolist(),
                    cat_fwd, pm_fwd, po_fwd,
                    cat_rev, pm_rev, po_rev,
                )

                if both_baseline:
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
                    "[extract] progress: %d reads used | "
                    "slowed fwd=%d rev=%d  near fwd=%d rev=%d  baseline seen=%d kept=%d",
                    n_reads_used, n_slowed_fwd, n_slowed_rev,
                    n_near_fwd, n_near_rev, n_baseline_seen, n_baseline_kept,
                )

    for kid, rows in baseline_buffer.items():
        if rows:
            samples[kid].extend(rows)

    result: dict = {}
    for kid, rows in samples.items():
        if rows:
            result[int(kid)] = np.array(rows, dtype=np.float32)

    log.info(
        "[extract] DONE  reads_used=%d  slowed fwd=%d rev=%d  near fwd=%d rev=%d  "
        "baseline seen=%d kept=%d",
        n_reads_used, n_slowed_fwd, n_slowed_rev,
        n_near_fwd, n_near_rev, n_baseline_seen, n_baseline_kept,
    )
    log.info("[extract] kmers in output: %d", len(result))
    return result


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
    """Extract a bilateral v2 shard from a raw HiFi aligned BAM."""
    cfg = load_kinsim_config()
    extract_cfg = cfg.get("extract") or {}
    params = extraction_params or get_extraction_params()

    if baseline_min_dist_to_meth is None:
        baseline_min_dist_to_meth = int(
            extract_cfg.get("baseline_min_dist_to_meth", params.kmer_size)
        )
    if baseline_sample_rate is None:
        baseline_sample_rate = float(extract_cfg.get("baseline_sample_rate", 0.10))
    if near_max_dist is None:
        near_max_dist = int(extract_cfg.get("near_meth_max_dist", 7))

    if baseline_min_dist_to_meth < params.kmer_size:
        raise ValueError(
            f"extract: baseline_min_dist_to_meth ({baseline_min_dist_to_meth}) "
            f"< kmer_size ({params.kmer_size}). Raise the value in "
            f"`extract:` of kinsim_config.yaml (typical: == kmer_size)."
        )

    log.info("[extract] BAM: %s", bam_path)
    log.info("[extract] REF: %s", ref_path)
    log.info("[extract] motifs: %s", motif_string)
    log.info(
        "[extract] bilateral v2: kmer_size=%d upstream=%d downstream=%d sample_ncols=%d",
        params.kmer_size, params.upstream, params.downstream, params.sample_ncols,
    )

    if meth_types is not None:
        motif_string = filter_motif_string_by_types(motif_string, meth_types)
        log.info("[extract] filtered motifs to types %s: %s", sorted(meth_types), motif_string)

    ref_seqs = load_reference(ref_path)
    log.info(
        "[extract] loaded reference: %d contigs, total %d bp",
        len(ref_seqs), sum(len(s) for s in ref_seqs.values()),
    )

    samples = _extract_one_bam(
        bam_path=bam_path, motif_string=motif_string, ref_seqs=ref_seqs, cfg=cfg,
        n_baseline_per_kmer=n_baseline_per_kmer,
        baseline_min_dist_to_meth=baseline_min_dist_to_meth,
        baseline_sample_rate=baseline_sample_rate, near_max_dist=near_max_dist,
        seed=seed, max_reads=max_reads, params=params,
    )

    samples["__meta__"] = {
        "extract_path": "raw_hifi_aligned_bilateral_v2",
        "kinsim_version": _KINSIM_VERSION,
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_bam": bam_path,
        "source_ref": ref_path,
        "motif_string": motif_string,
        "meth_id_map": get_meth_ids(),
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
    """Extract one manifest row to ``<output_dir>/<sample_id>_shard.pkl``."""
    import os as _os
    from .utils.config import load_manifest
    from .utils.motifs import load_motif_string as _load_motif_string

    entries = load_manifest(manifest_path)
    if task_index < 1 or task_index > len(entries):
        log.error("Task index %d out of range (manifest has %d entries).", task_index, len(entries))
        sys.exit(1)
    entry = entries[task_index - 1]
    log.info("task %d/%d: %s", task_index, len(entries), entry.sample_id)

    if not entry.ref_path:
        log.error(
            "Manifest entry '%s' is missing 'ref_path'. Bilateral extract "
            "requires raw HiFi aligned BAMs + a reference FASTA.",
            entry.sample_id,
        )
        sys.exit(1)
    if not Path(entry.ref_path).exists():
        log.error("ref_path does not exist for %s: %s", entry.sample_id, entry.ref_path)
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_pkl = _os.path.join(output_dir, f"{entry.sample_id}_shard.pkl")
    log.info("  Output: %s", output_pkl)

    motif_string = _load_motif_string(entry.motifs)
    if not motif_string:
        log.warning("No motifs resolved for '%s' — SKIPPING.", entry.sample_id)
        return

    extract_to_shard(
        bam_path=entry.bam_path, ref_path=entry.ref_path, motif_string=motif_string,
        output_path=output_pkl, meth_types=meth_types,
        n_baseline_per_kmer=n_baseline_per_kmer,
        baseline_min_dist_to_meth=baseline_min_dist_to_meth,
        baseline_sample_rate=baseline_sample_rate, near_max_dist=near_max_dist,
        seed=seed, max_reads=max_reads,
    )


def main(argv=None) -> None:
    """``kinsim extract`` CLI — bilateral v2 from raw HiFi aligned BAMs."""
    import argparse
    import sys as _sys
    from .utils.config import setup_logging

    if argv is None:
        argv = _sys.argv[1:]
    if argv and argv[0] == "extract":
        argv = argv[1:]

    p = argparse.ArgumentParser(prog="kinsim extract", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", help="Manifest CSV with ref_path column")
    p.add_argument("--task", type=int, help="1-based row index from manifest")
    p.add_argument("--output-dir", help="Output directory for shards (manifest mode)")
    p.add_argument("--n-baseline-per-kmer", type=int, default=50)
    p.add_argument("--max-reads", type=int, default=-1)
    p.add_argument("--meth-types", default=None,
                   help="Comma-separated subset (e.g. 'm6A,m4C'); default = all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("positional", nargs="*",
                   help="<raw_hifi_aligned_bam> <ref> <motifs> <output> (single-BAM mode)")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    from .utils.motifs import parse_meth_types_arg
    meth_types = parse_meth_types_arg(args.meth_types)

    if args.manifest:
        if args.task is None or not args.output_dir:
            p.error("--manifest requires --task and --output-dir")
        extract_from_manifest_task(
            manifest_path=args.manifest, task_index=args.task,
            output_dir=args.output_dir, n_baseline_per_kmer=args.n_baseline_per_kmer,
            seed=args.seed, max_reads=args.max_reads, meth_types=meth_types,
        )
        return

    if len(args.positional) != 4:
        p.error("single-BAM mode needs 4 positional args: <raw_hifi_aligned_bam> <ref> <motifs> <output>")
    bam, ref, motifs_arg, out_pkl = args.positional
    from .utils.motifs import load_motif_string
    motif_string = load_motif_string(motifs_arg)
    if not motif_string:
        log.error("No motifs resolved from '%s'", motifs_arg)
        _sys.exit(1)

    extract_to_shard(
        bam_path=bam, ref_path=ref, motif_string=motif_string, output_path=out_pkl,
        meth_types=meth_types, n_baseline_per_kmer=args.n_baseline_per_kmer,
        seed=args.seed, max_reads=args.max_reads,
    )


if __name__ == "__main__":
    main()
