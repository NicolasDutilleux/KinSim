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

    # Build a dict ref_pos → query_pos. For K small (~21) this loop is fine.
    ref_pairs = pairs[:, 1]
    qry_pairs = pairs[:, 0]
    # vectorise: searchsorted only if pairs is sorted by ref_pos (it is in pysam)
    for k, rp in enumerate(ref_positions):
        idx = np.searchsorted(ref_pairs, rp)
        if idx >= ref_pairs.size or ref_pairs[idx] != rp:
            continue
        q = int(qry_pairs[idx])
        if 0 <= q < ip_arr.size:
            ipd_out[k] = ip_arr[q]
            pw_out[k] = pw_arr[q]
            mask[k] = True
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


__all__ = [
    "BamFormat",
    "detect_bam_format",
    "WindowSample",
    "iter_window_samples",
    "extract_window_from_read",
]
