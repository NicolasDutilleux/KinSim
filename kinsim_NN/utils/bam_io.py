"""BAM I/O — extract bilateral (IPD_fwd, PW_fwd, IPD_rev, PW_rev) per ref position.

Handles BOTH input formats:

1. **Bystrandified aligned BAM** (default in current corpus): 2 records
   per ZMW with names ending in ``/ccs/fwd`` and ``/ccs/rev``. Each
   carries ``ip`` (IPD) and ``pw`` tags representing the kinetics on
   the synthesised strand. We pair records by ZMW name (everything
   before the trailing ``/fwd`` or ``/rev``) and combine to produce
   the 4 channels.

2. **Raw HiFi aligned BAM**: 1 record per ZMW with ``fi``/``fp``/``ri``/``rp``
   tags directly. No pairing needed.

The format is auto-detected from the tags present on the first aligned
record.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pysam


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BamFormat:
    """Lightweight descriptor of a BAM's kinetics format."""

    is_bystrandified: bool          # 2 records/ZMW with ip/pw vs 1/ZMW with fi/fp/ri/rp
    ipd_fwd_tag: str                # 'ip' (bystrandified fwd record) or 'fi'
    pw_fwd_tag: str                 # 'pw' or 'fp'
    ipd_rev_tag: str                # 'ip' (bystrandified rev record) or 'ri'
    pw_rev_tag: str                 # 'pw' or 'rp'


def detect_bam_format(bam_path: Path) -> BamFormat:
    """Inspect the first aligned read to decide the kinetics format."""
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            has_ip = r.has_tag("ip")
            has_fi = r.has_tag("fi")
            if has_fi:
                # Raw HiFi aligned format
                return BamFormat(
                    is_bystrandified=False,
                    ipd_fwd_tag="fi",
                    pw_fwd_tag="fp",
                    ipd_rev_tag="ri",
                    pw_rev_tag="rp",
                )
            if has_ip:
                return BamFormat(
                    is_bystrandified=True,
                    ipd_fwd_tag="ip",
                    pw_fwd_tag="pw",
                    ipd_rev_tag="ip",
                    pw_rev_tag="pw",
                )
            break
    raise RuntimeError(f"{bam_path}: no 'ip' or 'fi' tag on first aligned read")


# ---------------------------------------------------------------------------
# Read name helpers
# ---------------------------------------------------------------------------


def _zmw_key(read_name: str) -> tuple[str, str | None]:
    """Strip the bystrandify suffix and return (key, suffix).

    ``m84151_240303_022646_s4/106171215/ccs/fwd`` →
        (``m84151_240303_022646_s4/106171215/ccs``, ``fwd``)
    ``m84151_240303_022646_s4/106171215/ccs`` → (same, None)
    """
    if read_name.endswith("/fwd"):
        return read_name[:-4], "fwd"
    if read_name.endswith("/rev"):
        return read_name[:-4], "rev"
    return read_name, None


# ---------------------------------------------------------------------------
# Per-position kinetics extraction
# ---------------------------------------------------------------------------


def _aligned_pairs_array(read: pysam.AlignedSegment) -> np.ndarray:
    """Return aligned (query_pos, ref_pos) pairs as int64 (N, 2) array."""
    pairs = read.get_aligned_pairs(matches_only=True)
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def extract_window_from_read(
    read: pysam.AlignedSegment,
    ipd_tag: str,
    pw_tag: str,
    ref_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull IPD + PW values from one read at the requested reference positions.

    Returns:
        ipd: (K,) uint8, 0 where read doesn't cover.
        pw:  (K,) uint8, 0 where read doesn't cover.
        mask:(K,) bool, True where the read covers.
    """
    K = ref_positions.shape[0]
    ipd_out = np.zeros(K, dtype=np.uint8)
    pw_out = np.zeros(K, dtype=np.uint8)
    mask = np.zeros(K, dtype=bool)
    if not read.has_tag(ipd_tag) or not read.has_tag(pw_tag):
        return ipd_out, pw_out, mask

    pairs = _aligned_pairs_array(read)
    if pairs.size == 0:
        return ipd_out, pw_out, mask

    ip_arr = np.asarray(read.get_tag(ipd_tag), dtype=np.uint8)
    pw_arr = np.asarray(read.get_tag(pw_tag), dtype=np.uint8)

    ref_pairs = pairs[:, 1]
    qry_pairs = pairs[:, 0]
    # Vectorised: one C-level searchsorted instead of K Python iterations.
    # pairs are sorted by ref_pos in pysam (matches_only=True guarantees that).
    idxs = np.searchsorted(ref_pairs, ref_positions)
    clamped = np.minimum(idxs, ref_pairs.size - 1)
    found = (idxs < ref_pairs.size) & (ref_pairs[clamped] == ref_positions)
    q_pos = qry_pairs[clamped]
    in_bounds = found & (q_pos >= 0) & (q_pos < ip_arr.size)
    if in_bounds.any():
        # Fancy-index into the tag arrays where the read covers the ref position.
        valid_q = q_pos[in_bounds]
        ipd_out[in_bounds] = ip_arr[valid_q]
        pw_out[in_bounds] = pw_arr[valid_q]
        mask[in_bounds] = True
    return ipd_out, pw_out, mask


# ---------------------------------------------------------------------------
# Iterating reads covering a window — handles bystrandified pairing
# ---------------------------------------------------------------------------


@dataclass
class WindowSample:
    """One (ZMW, ref position) sample with 4 channels."""

    zmw_id: str
    ipd_fwd: np.ndarray   # (K,) uint8
    pw_fwd: np.ndarray
    ipd_rev: np.ndarray
    pw_rev: np.ndarray
    mask_fwd: np.ndarray  # (K,) bool — read covers this position
    mask_rev: np.ndarray


def iter_window_samples(
    bam: pysam.AlignmentFile,
    fmt: BamFormat,
    seqid: str,
    center_pos: int,
    half_width: int,
    min_mapq: int = 20,
) -> Iterator[WindowSample]:
    """Iterate per-ZMW bilateral samples covering the requested window.

    **Canonical strand convention** (matches ``kinsim2/extract.py``):

        ``ipd_fwd[ref_pos]`` = IPD measured during synthesis when the
        polymerase used the **+ reference strand** as template, i.e.
        + strand methylation kinetics.

        ``ipd_rev[ref_pos]`` = IPD on the − strand template, i.e. −
        strand methylation kinetics.

    PacBio raw HiFi tags map to this convention via ``read.is_reverse``:

        is_reverse=False (read SEQ = + ref strand):
            fi reads − strand template  →  fi[..]  is ipd_REV
            ri reads + strand template  →  ri[..]  is ipd_FWD
        is_reverse=True (read SEQ = − ref strand):
            fi reads + strand template  →  fi[..]  is ipd_FWD
            ri reads − strand template  →  ri[..]  is ipd_REV

    For bystrandified pairs, ``ccs/fwd`` record's ``ip`` ≡ ``fi`` (pass 1)
    and ``ccs/rev`` record's ``ip`` ≡ ``ri`` (pass 2), so the same routing
    rules apply to (fwd_record, rev_record) ↔ (fi-like, ri-like).
    """
    K = 2 * half_width + 1
    ref_positions = np.arange(center_pos - half_width, center_pos + half_width + 1,
                              dtype=np.int64)

    if fmt.is_bystrandified:
        # Collect both fwd and rev records per ZMW, then emit paired samples.
        zmw_pairs: dict[str, dict[str, pysam.AlignedSegment]] = defaultdict(dict)
        n_reads_seen = 0
        for r in bam.fetch(seqid, max(0, center_pos - half_width),
                           center_pos + half_width + 1):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            if r.mapping_quality < min_mapq:
                continue
            # Skip reads that don't actually cover the centre — pysam returns
            # any read overlapping the [center-half, center+half+1) window, but
            # we strictly require coverage of center_pos itself. Cheap reference-
            # bound check avoids the expensive get_aligned_pairs downstream.
            if r.reference_start > center_pos or r.reference_end <= center_pos:
                continue
            n_reads_seen += 1
            key, suffix = _zmw_key(r.query_name or "")
            if suffix not in ("fwd", "rev"):
                continue
            zmw_pairs[key][suffix] = r

        if n_reads_seen > 0 and not zmw_pairs:
            log.warning(
                "Bystrandified-mode iter at %s:%d saw %d reads but none had "
                "/fwd or /rev suffix in their name. Naming convention may "
                "have changed — check ZMW read naming in this BAM.",
                seqid, center_pos, n_reads_seen,
            )

        for zmw_id, pair in zmw_pairs.items():
            if "fwd" not in pair or "rev" not in pair:
                continue
            fwd_rec = pair["fwd"]
            rev_rec = pair["rev"]
            # Extract ip/pw from both records at the requested ref positions.
            # These are RAW per-record values (fi-like from /fwd, ri-like from /rev).
            ip_fi, pw_fi, mask_fi = extract_window_from_read(
                fwd_rec, fmt.ipd_fwd_tag, fmt.pw_fwd_tag, ref_positions,
            )
            ip_ri, pw_ri, mask_ri = extract_window_from_read(
                rev_rec, fmt.ipd_rev_tag, fmt.pw_rev_tag, ref_positions,
            )
            # Both records of a ZMW must agree on alignment direction (pbmm2
            # is deterministic per-ZMW). Use the /fwd record's direction.
            if fwd_rec.is_reverse != rev_rec.is_reverse:
                # Pathological mixed-orientation pair: skip rather than
                # silently route into the wrong channel.
                continue
            if fwd_rec.is_reverse:
                # reverse-mapped: fi-like (ip from /fwd) is + strand kinetics
                ipd_fwd_w, pw_fwd_w = ip_fi, pw_fi
                ipd_rev_w, pw_rev_w = ip_ri, pw_ri
                mask_fwd_w, mask_rev_w = mask_fi, mask_ri
            else:
                # forward-mapped: ri-like (ip from /rev) is + strand kinetics
                ipd_fwd_w, pw_fwd_w = ip_ri, pw_ri
                ipd_rev_w, pw_rev_w = ip_fi, pw_fi
                mask_fwd_w, mask_rev_w = mask_ri, mask_fi
            # Require both strands to cover the centre
            if not (mask_fwd_w[half_width] and mask_rev_w[half_width]):
                continue
            yield WindowSample(
                zmw_id=zmw_id,
                ipd_fwd=ipd_fwd_w, pw_fwd=pw_fwd_w,
                ipd_rev=ipd_rev_w, pw_rev=pw_rev_w,
                mask_fwd=mask_fwd_w, mask_rev=mask_rev_w,
            )
    else:
        # Raw HiFi: 1 record per ZMW with fi/fp/ri/rp tags
        for r in bam.fetch(seqid, max(0, center_pos - half_width),
                           center_pos + half_width + 1):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            if r.mapping_quality < min_mapq:
                continue
            # Skip reads that don't actually cover the centre — pysam returns
            # any read overlapping the [center-half, center+half+1) window, but
            # we strictly require coverage of center_pos itself. Cheap reference-
            # bound check avoids the expensive get_aligned_pairs downstream.
            if r.reference_start > center_pos or r.reference_end <= center_pos:
                continue
            ip_fi, pw_fi, mask_fi = extract_window_from_read(
                r, fmt.ipd_fwd_tag, fmt.pw_fwd_tag, ref_positions,  # 'fi', 'fp'
            )
            ip_ri, pw_ri, mask_ri = extract_window_from_read(
                r, fmt.ipd_rev_tag, fmt.pw_rev_tag, ref_positions,  # 'ri', 'rp'
            )
            if r.is_reverse:
                ipd_fwd_w, pw_fwd_w = ip_fi, pw_fi
                ipd_rev_w, pw_rev_w = ip_ri, pw_ri
                mask_fwd_w, mask_rev_w = mask_fi, mask_ri
            else:
                ipd_fwd_w, pw_fwd_w = ip_ri, pw_ri
                ipd_rev_w, pw_rev_w = ip_fi, pw_fi
                mask_fwd_w, mask_rev_w = mask_ri, mask_fi
            if not (mask_fwd_w[half_width] and mask_rev_w[half_width]):
                continue
            yield WindowSample(
                zmw_id=r.query_name or "",
                ipd_fwd=ipd_fwd_w, pw_fwd=pw_fwd_w,
                ipd_rev=ipd_rev_w, pw_rev=pw_rev_w,
                mask_fwd=mask_fwd_w, mask_rev=mask_rev_w,
            )


def _extract_window_vectorized(
    ref_pairs: np.ndarray,        # (N,) int64
    qry_pairs: np.ndarray,        # (N,) int64
    ip_arr: np.ndarray,           # (Lr,) uint8
    pw_arr: np.ndarray,           # (Lr,) uint8
    ref_positions: np.ndarray,    # (K,) int64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as :func:`extract_window_from_read` but takes already-decoded
    pair / tag arrays so we avoid recomputing them across multiple windows
    on the same read (the key win of the batched path)."""
    K = ref_positions.shape[0]
    ipd_out = np.zeros(K, dtype=np.uint8)
    pw_out = np.zeros(K, dtype=np.uint8)
    mask = np.zeros(K, dtype=bool)
    if ref_pairs.size == 0:
        return ipd_out, pw_out, mask
    idxs = np.searchsorted(ref_pairs, ref_positions)
    clamped = np.minimum(idxs, ref_pairs.size - 1)
    found = (idxs < ref_pairs.size) & (ref_pairs[clamped] == ref_positions)
    q_pos = qry_pairs[clamped]
    in_bounds = found & (q_pos >= 0) & (q_pos < ip_arr.size)
    if in_bounds.any():
        valid_q = q_pos[in_bounds]
        ipd_out[in_bounds] = ip_arr[valid_q]
        pw_out[in_bounds] = pw_arr[valid_q]
        mask[in_bounds] = True
    return ipd_out, pw_out, mask


def iter_chunk_samples(
    bam: pysam.AlignmentFile,
    fmt: BamFormat,
    seqid: str,
    sorted_ref_positions: np.ndarray,
    half_width: int,
    min_mapq: int = 20,
) -> Iterator[tuple[int, WindowSample]]:
    """Yield (center_pos, WindowSample) for MANY center positions in ONE fetch.

    Key wins vs :func:`iter_window_samples`:

    * **One** ``bam.fetch`` per chunk (not per position) → 200× fewer disk seeks
      when 200 positions are within a ~5 kb span.
    * **get_aligned_pairs cached** once per record → reused for all positions
      in the chunk. Each call is ~5-10 ms; for ~200 positions × ~80 ZMWs that
      saves ~80 seconds per chunk.
    * **Tag arrays decoded** once per record (was once per position).

    The chunk geometry is the caller's responsibility — group positions so they
    fit in a reasonable span (≤ ~10 kb). Within a chunk every ZMW pair is
    walked once and every position is checked against the cached pair index.
    """
    if sorted_ref_positions.size == 0:
        return
    span_start = max(0, int(sorted_ref_positions[0]) - half_width)
    span_end = int(sorted_ref_positions[-1]) + half_width + 1

    if fmt.is_bystrandified:
        zmw_pairs: dict[str, dict[str, pysam.AlignedSegment]] = defaultdict(dict)
        for r in bam.fetch(seqid, span_start, span_end):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            if r.mapping_quality < min_mapq:
                continue
            key, suffix = _zmw_key(r.query_name or "")
            if suffix in ("fwd", "rev"):
                zmw_pairs[key][suffix] = r

        for zmw_id, pair in zmw_pairs.items():
            if "fwd" not in pair or "rev" not in pair:
                continue
            fwd_rec = pair["fwd"]
            rev_rec = pair["rev"]
            if fwd_rec.is_reverse != rev_rec.is_reverse:
                continue
            is_rev = fwd_rec.is_reverse
            # Decode aligned pairs + tag arrays ONCE per record.
            if not (fwd_rec.has_tag(fmt.ipd_fwd_tag) and fwd_rec.has_tag(fmt.pw_fwd_tag)):
                continue
            if not (rev_rec.has_tag(fmt.ipd_rev_tag) and rev_rec.has_tag(fmt.pw_rev_tag)):
                continue
            fwd_pairs = _aligned_pairs_array(fwd_rec)
            rev_pairs = _aligned_pairs_array(rev_rec)
            if fwd_pairs.size == 0 or rev_pairs.size == 0:
                continue
            fwd_ref_pairs = fwd_pairs[:, 1]
            fwd_qry_pairs = fwd_pairs[:, 0]
            rev_ref_pairs = rev_pairs[:, 1]
            rev_qry_pairs = rev_pairs[:, 0]
            ip_fi_arr = np.asarray(fwd_rec.get_tag(fmt.ipd_fwd_tag), dtype=np.uint8)
            pw_fi_arr = np.asarray(fwd_rec.get_tag(fmt.pw_fwd_tag), dtype=np.uint8)
            ip_ri_arr = np.asarray(rev_rec.get_tag(fmt.ipd_rev_tag), dtype=np.uint8)
            pw_ri_arr = np.asarray(rev_rec.get_tag(fmt.pw_rev_tag), dtype=np.uint8)
            fwd_rs, fwd_re = fwd_rec.reference_start, fwd_rec.reference_end
            rev_rs, rev_re = rev_rec.reference_start, rev_rec.reference_end

            for center_pos in sorted_ref_positions:
                cpos = int(center_pos)
                # Quick coverage check (~50ns) before vectorized extraction
                if not (fwd_rs <= cpos < fwd_re and rev_rs <= cpos < rev_re):
                    continue
                ref_positions = np.arange(
                    cpos - half_width, cpos + half_width + 1, dtype=np.int64,
                )
                ip_fi, pw_fi, mask_fi = _extract_window_vectorized(
                    fwd_ref_pairs, fwd_qry_pairs, ip_fi_arr, pw_fi_arr, ref_positions,
                )
                ip_ri, pw_ri, mask_ri = _extract_window_vectorized(
                    rev_ref_pairs, rev_qry_pairs, ip_ri_arr, pw_ri_arr, ref_positions,
                )
                if is_rev:
                    ipd_fwd_w, pw_fwd_w = ip_fi, pw_fi
                    ipd_rev_w, pw_rev_w = ip_ri, pw_ri
                    mask_fwd_w, mask_rev_w = mask_fi, mask_ri
                else:
                    ipd_fwd_w, pw_fwd_w = ip_ri, pw_ri
                    ipd_rev_w, pw_rev_w = ip_fi, pw_fi
                    mask_fwd_w, mask_rev_w = mask_ri, mask_fi
                if mask_fwd_w[half_width] and mask_rev_w[half_width]:
                    yield cpos, WindowSample(
                        zmw_id=zmw_id,
                        ipd_fwd=ipd_fwd_w, pw_fwd=pw_fwd_w,
                        ipd_rev=ipd_rev_w, pw_rev=pw_rev_w,
                        mask_fwd=mask_fwd_w, mask_rev=mask_rev_w,
                    )
    else:
        # Raw HiFi — 1 record per ZMW with all four tags.
        for r in bam.fetch(seqid, span_start, span_end):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            if r.mapping_quality < min_mapq:
                continue
            if not (r.has_tag(fmt.ipd_fwd_tag) and r.has_tag(fmt.pw_fwd_tag)
                    and r.has_tag(fmt.ipd_rev_tag) and r.has_tag(fmt.pw_rev_tag)):
                continue
            pairs = _aligned_pairs_array(r)
            if pairs.size == 0:
                continue
            ref_pairs = pairs[:, 1]
            qry_pairs = pairs[:, 0]
            ip_fi_arr = np.asarray(r.get_tag(fmt.ipd_fwd_tag), dtype=np.uint8)
            pw_fi_arr = np.asarray(r.get_tag(fmt.pw_fwd_tag), dtype=np.uint8)
            ip_ri_arr = np.asarray(r.get_tag(fmt.ipd_rev_tag), dtype=np.uint8)
            pw_ri_arr = np.asarray(r.get_tag(fmt.pw_rev_tag), dtype=np.uint8)
            ref_rs, ref_re = r.reference_start, r.reference_end
            is_rev = r.is_reverse
            zmw_id = r.query_name or ""

            for center_pos in sorted_ref_positions:
                cpos = int(center_pos)
                if not (ref_rs <= cpos < ref_re):
                    continue
                ref_positions = np.arange(
                    cpos - half_width, cpos + half_width + 1, dtype=np.int64,
                )
                ip_fi, pw_fi, mask_fi = _extract_window_vectorized(
                    ref_pairs, qry_pairs, ip_fi_arr, pw_fi_arr, ref_positions,
                )
                ip_ri, pw_ri, mask_ri = _extract_window_vectorized(
                    ref_pairs, qry_pairs, ip_ri_arr, pw_ri_arr, ref_positions,
                )
                if is_rev:
                    ipd_fwd_w, pw_fwd_w = ip_fi, pw_fi
                    ipd_rev_w, pw_rev_w = ip_ri, pw_ri
                    mask_fwd_w, mask_rev_w = mask_fi, mask_ri
                else:
                    ipd_fwd_w, pw_fwd_w = ip_ri, pw_ri
                    ipd_rev_w, pw_rev_w = ip_fi, pw_fi
                    mask_fwd_w, mask_rev_w = mask_ri, mask_fi
                if mask_fwd_w[half_width] and mask_rev_w[half_width]:
                    yield cpos, WindowSample(
                        zmw_id=zmw_id,
                        ipd_fwd=ipd_fwd_w, pw_fwd=pw_fwd_w,
                        ipd_rev=ipd_rev_w, pw_rev=pw_rev_w,
                        mask_fwd=mask_fwd_w, mask_rev=mask_rev_w,
                    )


__all__ = [
    "BamFormat",
    "detect_bam_format",
    "WindowSample",
    "iter_window_samples",
    "iter_chunk_samples",
    "extract_window_from_read",
]
