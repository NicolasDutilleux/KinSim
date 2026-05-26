"""``kinsim_nn generate`` — inject model-generated kinetics into an unmapped BAM.

Inputs:
    * input BAM (PBSIM3 sim or HiFi with fi/fp/ri/rp stripped)
    * reference FASTA (target genome)
    * trained G.pt + model_config.json
    * motifs.csv (PacBio format: motifString, centerPos, modificationType,
      fraction, ...)

Process:
    For each read of the input BAM:
      For each query position q (skipping first/last K/2 due to window):
        ref_pos = align_query_to_ref(read, q)  # or sequential if input is unaligned
        Build base_fwd[K], meth_fwd[K], meth_rev[K] for the window
          around ref_pos. Methylation comes from the motifs.csv scan
          over the reference: when the motif center lies inside the
          window AND a Bernoulli(fraction) draw fires, place meth_id at
          the configured signal offset.
        z = N(0, 1)
        signal[K, 4] = G(z, base_fwd, base_rev, meth_fwd, meth_rev)
        signal_center = signal[K // 2]
        fi[q] = uint8 from log1p(frames) channel 0
        fp[q] = uint8 from log1p(frames) channel 1
        ri[q] = uint8 from log1p(frames) channel 2
        rp[q] = uint8 from log1p(frames) channel 3
      Write read with fi/fp/ri/rp tags to output BAM.
"""
from __future__ import annotations

import argparse
import array
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pysam
import torch

from kinsim.utils.motifs import (
    load_motif_string,
    parse_motifs_per_strand,
    reverse_complement,
    scan_sequence,
)

from . import __version__
from .models.generator import TransformerGenerator
from .utils.config import load_config, setup_logging
from .utils.encoding import BASE_RC as _RC_TABLE
from .utils.encoding import encode_seq
from .utils.pacbio_codec import log1p_frames_to_uint8


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Motif scanning — REUSE kinsim's canonical parsers.
#
# This module previously hand-rolled centerPos/fraction/IUPAC handling and
# got the centerPos 1-based → 0-based conversion wrong (every meth was off
# by one base in generation). kinsim/utils/parsers/pacbio.py + motifs.py
# have been doing this correctly for years; we delegate to them.
# ---------------------------------------------------------------------------


def _build_strand_meth_maps(
    ref_seqs: dict[str, str],
    motif_string: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Return per-contig ``(fwd_map, rev_map, fwd_frac, rev_frac)``.

    * ``fwd_map[contig][p]`` = meth_id on the + strand at ref pos ``p`` (0 if none)
    * ``rev_map[contig][p]`` = meth_id on the − strand at ref pos ``p`` (0 if none),
      in **forward-ref coordinates** (so it can be indexed alongside fwd_map).
    * ``fwd_frac``, ``rev_frac`` = the per-motif fraction at the same positions.

    Uses :func:`kinsim.utils.motifs.parse_motifs_per_strand` for the canonical
    per-strand handling (palindromic motifs, IUPAC, centerPos→0-based).
    """
    fwd_motifs, rev_motifs = parse_motifs_per_strand(motif_string)
    fwd_map: dict[str, np.ndarray] = {}
    rev_map: dict[str, np.ndarray] = {}
    fwd_frac: dict[str, np.ndarray] = {}
    rev_frac: dict[str, np.ndarray] = {}
    for name, seq in ref_seqs.items():
        L = len(seq)
        fmap = scan_sequence(seq, fwd_motifs).astype(np.uint8)
        # Per-motif fraction overlay on the fwd map
        ffrac = np.zeros(L, dtype=np.float32)
        for motif in fwd_motifs:
            for match in motif["pattern"].finditer(seq):
                target = match.start() + motif["pos"]
                if 0 <= target < L:
                    ffrac[target] = motif["frac"]
        # Reverse strand: scan rc(seq), then flip to forward coords
        rc_seq = reverse_complement(seq)
        rc_hits = scan_sequence(rc_seq, rev_motifs).astype(np.uint8)
        rmap = rc_hits[::-1].copy()
        rfrac_rc = np.zeros(L, dtype=np.float32)
        for motif in rev_motifs:
            for match in motif["pattern"].finditer(rc_seq):
                target = match.start() + motif["pos"]
                if 0 <= target < L:
                    rfrac_rc[target] = motif["frac"]
        rfrac = rfrac_rc[::-1].copy()
        fwd_map[name] = fmap
        rev_map[name] = rmap
        fwd_frac[name] = ffrac
        rev_frac[name] = rfrac
    return fwd_map, rev_map, fwd_frac, rev_frac


# ---------------------------------------------------------------------------
# Generator loading
# ---------------------------------------------------------------------------


def _find_checkpoint(ckpt_dir: Path) -> Path:
    """Prefer best_G.pt > G.pt > most recent .pt by mtime."""
    best = ckpt_dir / "best_G.pt"
    if best.is_file():
        return best
    latest = ckpt_dir / "G.pt"
    if latest.is_file():
        return latest
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt in {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_generator(ckpt_dir: Path, device: torch.device) -> tuple[TransformerGenerator, dict]:
    config_path = ckpt_dir / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} missing")
    cfg = json.loads(config_path.read_text())
    g = TransformerGenerator(
        k=cfg["k"],
        n_meth_types=cfg["n_meth_types"],
        d_model=cfg["generator"]["d_model"],
        n_layers=cfg["generator"]["n_layers"],
        n_heads=cfg["generator"]["n_heads"],
        z_dim=cfg["generator"]["z_dim"],
        pos_embed_dim=cfg["generator"]["pos_embed_dim"],
        drop_rate=cfg["generator"].get("drop_rate", 0.0),
    ).to(device)
    ckpt_path = _find_checkpoint(ckpt_dir)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    g.load_state_dict(state["state_dict"])
    g.eval()
    log.info("Loaded G from %s (k=%d, M=%d)", ckpt_path, cfg["k"], cfg["n_meth_types"])
    return g, cfg


# ---------------------------------------------------------------------------
# Per-read generation
# ---------------------------------------------------------------------------


# Single source of truth: kinsim_NN.utils.encoding (delegates to kinsim's BASE_MAP).
_encode_seq = encode_seq


@torch.no_grad()
def _generate_signal_batched(
    g: TransformerGenerator,
    base_fwd_stack: np.ndarray,     # (B, K) uint8
    meth_fwd_stack: np.ndarray,     # (B, K) uint8
    meth_rev_stack: np.ndarray,     # (B, K) uint8
    n_meth_types: int,
    device: torch.device,
) -> np.ndarray:
    """Return the center (IPD_fwd, PW_fwd, IPD_rev, PW_rev) predictions for
    a batch of windows. Shape: (B, 4) float32 in log1p(frames) space."""
    out_np = _generate_signal_batched_full(
        g, base_fwd_stack, meth_fwd_stack, meth_rev_stack, n_meth_types, device,
    )
    K = base_fwd_stack.shape[1]
    return out_np[:, K // 2].astype(np.float32)           # (B, 4)


def _generate_signal_batched_full(
    g: TransformerGenerator,
    base_fwd_stack: np.ndarray,     # (B, K) uint8
    meth_fwd_stack: np.ndarray,     # (B, K) uint8
    meth_rev_stack: np.ndarray,     # (B, K) uint8
    n_meth_types: int,
    device: torch.device,
) -> np.ndarray:
    """Return the FULL K-position window prediction for a batch.
    Shape: (B, K, 4) float32 in log1p(frames) space.

    Used by the precompute-cache path: each inference's K outputs fill K
    consecutive reference positions instead of being discarded (the old
    per-position code threw away K-1 of K outputs)."""
    B, K = base_fwd_stack.shape
    base_rev = _RC_TABLE[base_fwd_stack]
    base_fwd_oh = np.zeros((B, K, 4), dtype=np.float32)
    base_rev_oh = np.zeros((B, K, 4), dtype=np.float32)
    np.put_along_axis(base_fwd_oh, base_fwd_stack.astype(np.int64)[..., None], 1.0, axis=-1)
    np.put_along_axis(base_rev_oh, base_rev.astype(np.int64)[..., None], 1.0, axis=-1)
    meth_fwd_oh = np.zeros((B, K, n_meth_types), dtype=np.float32)
    meth_rev_oh = np.zeros((B, K, n_meth_types), dtype=np.float32)
    np.put_along_axis(meth_fwd_oh, meth_fwd_stack.astype(np.int64)[..., None], 1.0, axis=-1)
    np.put_along_axis(meth_rev_oh, meth_rev_stack.astype(np.int64)[..., None], 1.0, axis=-1)

    z = g.sample_z(B, device=device)
    with torch.no_grad():
        out = g(
            z,
            torch.from_numpy(base_fwd_oh).to(device),
            torch.from_numpy(base_rev_oh).to(device),
            torch.from_numpy(meth_fwd_oh).to(device),
            torch.from_numpy(meth_rev_oh).to(device),
        )
    return out.detach().cpu().numpy().astype(np.float32)  # (B, K, 4)


def _draw_read_effective_meth(
    fwd_map: np.ndarray,                # (L_ref,) uint8
    rev_map: np.ndarray,                # (L_ref,) uint8
    fwd_frac: np.ndarray,               # (L_ref,) float32
    rev_frac: np.ndarray,               # (L_ref,) float32
    rng: random.Random,
    use_bernoulli: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ONE Bernoulli per labelled site for this read.

    Returns ``(eff_fwd, eff_rev)``: same shape as the input maps but with
    non-firing labelled sites zeroed out. The drawn maps are kept for
    the WHOLE read so adjacent sliding-window queries see consistent
    conditioning (the bug C1 from the audit: previously rng was rolled
    once per (window, site) which made the same site flip on/off as the
    window slid).

    For long contigs the copy is O(L) but L≪corpus size and we do this
    once per read, not per query position.
    """
    eff_fwd = fwd_map.copy()
    eff_rev = rev_map.copy()
    if not use_bernoulli:
        return eff_fwd, eff_rev
    fwd_idx = np.flatnonzero(eff_fwd)
    rev_idx = np.flatnonzero(eff_rev)
    for i in fwd_idx:
        if rng.random() > float(fwd_frac[i]):
            eff_fwd[i] = 0
    for i in rev_idx:
        if rng.random() > float(rev_frac[i]):
            eff_rev[i] = 0
    return eff_fwd, eff_rev


def _build_meth_window(
    eff_fwd: np.ndarray,                # (L_ref,) uint8 — per-read effective fwd meth
    eff_rev: np.ndarray,                # (L_ref,) uint8 — per-read effective rev meth
    ref_pos_window: np.ndarray,         # (K,) int64 — reference positions in window
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the per-read effective meth maps to the window. No Bernoulli
    here — :func:`_draw_read_effective_meth` already applied it."""
    K = ref_pos_window.shape[0]
    meth_fwd = np.zeros(K, dtype=np.uint8)
    meth_rev = np.zeros(K, dtype=np.uint8)
    L = eff_fwd.shape[0]
    valid = (ref_pos_window >= 0) & (ref_pos_window < L)
    if valid.any():
        valid_pos = ref_pos_window[valid]
        meth_fwd[valid] = eff_fwd[valid_pos]
        meth_rev[valid] = eff_rev[valid_pos]
    return meth_fwd, meth_rev


def _precompute_kinetics_map_for_contig(
    ref_seq: str,
    eff_fwd: np.ndarray,                # (L,) uint8 — GLOBAL effective meth + strand
    eff_rev: np.ndarray,                # (L,) uint8 — GLOBAL effective meth − strand
    g: TransformerGenerator,
    n_meth_types: int,
    K: int,
    half_width: int,
    n_z_samples: int,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Generate predicted kinetics for the WHOLE contig, once per z sample.

    Walks the genome at stride K so each inference's K outputs fill K
    consecutive reference positions (the old per-position code threw away
    K-1 of K outputs and re-fetched the same window for each base of each
    read).

    Per-read kinetic variance is preserved at lookup time by sampling a
    different ``z_idx`` per read (each shows kinetics from one of the N
    pre-drawn z latents). N=8 gives ~12 distinct per-position kinetic
    samples per read which is plenty to break determinism without bloating
    memory.

    Returns array ``(L, 4, n_z_samples)`` uint8 in PacBio codec
    (FRAMES_TABLE → byte). Positions outside ``[half_width, L-half_width)``
    stay at 0 (read filler fills them with ``default_value``).
    """
    L = len(ref_seq)
    kin_map = np.zeros((L, 4, n_z_samples), dtype=np.uint8)
    if L < 2 * half_width + 1:
        return kin_map  # contig too short for any window

    centers = list(range(half_width, L - half_width, K))
    log.info("  precompute contig L=%d → %d windows × %d z = %d inferences",
             L, len(centers), n_z_samples, len(centers) * n_z_samples)

    for z_idx in range(n_z_samples):
        for start in range(0, len(centers), batch_size):
            chunk = centers[start : start + batch_size]
            B = len(chunk)
            bf = np.zeros((B, K), dtype=np.uint8)
            mf = np.zeros((B, K), dtype=np.uint8)
            mr = np.zeros((B, K), dtype=np.uint8)
            for i, c in enumerate(chunk):
                bf[i] = _encode_seq(ref_seq[c - half_width : c + half_width + 1])
                mf[i] = eff_fwd[c - half_width : c + half_width + 1]
                mr[i] = eff_rev[c - half_width : c + half_width + 1]
            full = _generate_signal_batched_full(g, bf, mf, mr, n_meth_types, device)
            # full: (B, K, 4) log1p(frames) → uint8 PacBio codec
            u8 = log1p_frames_to_uint8(full)             # (B, K, 4)
            for i, c in enumerate(chunk):
                kin_map[c - half_width : c + half_width + 1, :, z_idx] = u8[i]
        if (z_idx + 1) % max(1, n_z_samples // 4) == 0:
            log.info("    z_sample %d/%d done", z_idx + 1, n_z_samples)

    return kin_map


def _draw_global_effective_meth(
    fwd_map: np.ndarray,                # (L,) uint8
    rev_map: np.ndarray,                # (L,) uint8
    fwd_frac: np.ndarray,               # (L,) float32
    rev_frac: np.ndarray,               # (L,) float32
    rng: random.Random,
    use_bernoulli: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """GLOBAL Bernoulli (one decision per genomic site, shared by all reads).

    Biologically this matches PacBio's ``fraction`` semantic: across a
    population, X% of all genomic sites of a motif are methylated. In any
    one cell a given site is either methylated or not (fully). The previous
    per-read draw over-modeled stochasticity by treating each (read, site)
    pair as an independent Bernoulli — different reads then disagreed on
    whether the SAME site was methylated, which doesn't match the biology.

    Per-read kinetic variance is now preserved by sampling a different
    ``z_idx`` into the kinetics map per read (see
    :func:`_precompute_kinetics_map_for_contig`).
    """
    eff_fwd = fwd_map.copy()
    eff_rev = rev_map.copy()
    if not use_bernoulli:
        return eff_fwd, eff_rev
    fwd_idx = np.flatnonzero(eff_fwd)
    rev_idx = np.flatnonzero(eff_rev)
    for i in fwd_idx:
        if rng.random() > float(fwd_frac[i]):
            eff_fwd[i] = 0
    for i in rev_idx:
        if rng.random() > float(rev_frac[i]):
            eff_rev[i] = 0
    return eff_fwd, eff_rev


def _cigar_to_match_pairs(
    cigartuples: list[tuple[int, int]],
    ref_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk CIGAR ops → arrays of matched (q_pos, ref_pos) pairs.

    Faster than ``pysam.AlignedSegment.get_aligned_pairs(matches_only=True)``
    for our needs because it stays in pure-numpy land (no Python tuple list).
    BAM op codes: 0=M, 1=I, 2=D, 3=N, 4=S, 5=H, 6=P, 7==, 8=X.
    M/=/X advance both query and ref. I/S advance query only. D/N advance
    ref only. H/P advance neither.
    """
    q_parts: list[np.ndarray] = []
    r_parts: list[np.ndarray] = []
    q_pos = 0
    r_pos = ref_start
    for op, length in cigartuples:
        if op == 0 or op == 7 or op == 8:    # M, =, X
            q_parts.append(np.arange(q_pos, q_pos + length, dtype=np.int64))
            r_parts.append(np.arange(r_pos, r_pos + length, dtype=np.int64))
            q_pos += length
            r_pos += length
        elif op == 1 or op == 4:              # I, S
            q_pos += length
        elif op == 2 or op == 3:              # D, N
            r_pos += length
        # H (5), P (6) advance neither
    if not q_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(q_parts), np.concatenate(r_parts)


def _process_read_from_cigar(
    qlen: int,
    ref_start: int,
    cigartuples: list[tuple[int, int]],
    is_rev: bool,
    kin_map: np.ndarray,                # (L, 4, n_z_samples) uint8
    z_seed: int,
    default_value: int,
    n_context_skip: int,
    per_read_z: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Worker-safe variant of :func:`_process_mapped_read_lookup`.

    Two z-sampling modes (controlled by ``per_read_z``):

    * **per_read_z=True (default, v2 fix)**: pick ONE z_idx for the whole
      read. All bases of this read share the same noise realisation.
      Models the biological reality that a single ZMW has one polymerase
      with correlated noise across its bases. This is what ipdSummary
      expects to find — CONSISTENT slowdown across many reads at a
      methylated site. Different reads at the same position pick different
      z_idx's so ensemble variance at the position level is preserved.
      Critical for downstream motif detection.

    * **per_read_z=False (legacy)**: per-(read, base) random z_idx.
      Maximum apparent variance but BREAKS per-read coherence and
      desensitises ipdSummary's per-position statistical tests. Earlier
      v1 default; kept for ablation but ipdSummary motif detection on
      v1 output found only weak generic motifs (not the real GATC/etc).

    ``z_seed`` is per-read so different reads pick different z's
    (reproducible given seed).

    Strand routing matches :func:`_process_mapped_read` exactly:
      is_rev=False → fi←ipd_rev(ch2), fp←pw_rev(ch3), ri←ipd_fwd(ch0), rp←pw_fwd(ch1)
      is_rev=True  → fi←ipd_fwd(ch0), fp←pw_fwd(ch1), ri←ipd_rev(ch2), rp←pw_rev(ch3)
    """
    fi = np.full(qlen, default_value, dtype=np.uint8)
    fp = np.full(qlen, default_value, dtype=np.uint8)
    ri = np.full(qlen, default_value, dtype=np.uint8)
    rp = np.full(qlen, default_value, dtype=np.uint8)
    q_arr, r_arr = _cigar_to_match_pairs(cigartuples, ref_start)
    if q_arr.size == 0:
        return fi, fp, ri, rp
    L = kin_map.shape[0]
    n_z = kin_map.shape[2]
    valid = (
        (q_arr >= n_context_skip)
        & (q_arr < qlen - n_context_skip)
        & (r_arr >= 0)
        & (r_arr < L)
    )
    if not valid.any():
        return fi, fp, ri, rp
    q_valid = q_arr[valid]
    r_valid = r_arr[valid]
    z_rng = np.random.default_rng(z_seed)
    if per_read_z:
        # Pick ONE z_idx for the whole read → coherent across bases
        z_idx = int(z_rng.integers(0, n_z))
        block = kin_map[r_valid, :, z_idx]    # (n_valid, 4) — all bases same z
    else:
        # Legacy: per-base random z_idx (high variance, breaks coherence)
        z_indices = z_rng.integers(0, n_z, size=r_valid.size)
        block = kin_map[r_valid[:, None], np.arange(4)[None, :], z_indices[:, None]]
    if is_rev:
        fi[q_valid] = block[:, 0]
        fp[q_valid] = block[:, 1]
        ri[q_valid] = block[:, 2]
        rp[q_valid] = block[:, 3]
    else:
        fi[q_valid] = block[:, 2]
        fp[q_valid] = block[:, 3]
        ri[q_valid] = block[:, 0]
        rp[q_valid] = block[:, 1]
    return fi, fp, ri, rp


def _unalign_read(read: pysam.AlignedSegment) -> None:
    """Convert an aligned read in-place into an unaligned record.

    Target flag = 4 EXACTLY (unmapped, not paired). PacBio raw HiFi BAMs
    have flag=4 on all CCS reads — anything else (e.g. flag=12 with
    mate_is_unmapped, or flag=20 with is_reverse) makes downstream tools
    like ``ccs-kinetics-bystrandify`` reject the read with misleading
    'has 0 PulseWidths' warnings. So we set the FLAG byte directly and
    leave the paired-end flags untouched (CCS reads aren't paired —
    is_paired is already False).

    Clears: ref_id, ref_start, mapq, cigar, mate ref_id/start, tlen.
    Preserves: query_name, query_sequence, query_qualities, all tags
    (including fi/fp/ri/rp we just wrote).
    """
    read.flag = 4
    read.reference_id = -1
    read.reference_start = -1
    read.mapping_quality = 0
    read.cigarstring = None
    read.next_reference_id = -1
    read.next_reference_start = -1
    read.template_length = 0


def _sanitize_header_for_unaligned(header_dict: dict) -> dict:
    """Sanitize a BAM header for unaligned output.

    Two transforms:

    1. Strip ``@SQ`` lines — an unaligned BAM must not reference contigs
       (pbmm2/samtools tolerate it but PacBio tools like
       ``ccs-kinetics-bystrandify`` reject SQ-referencing records).
    2. Force ``@HD SO:unknown``. The input was a coordinate-sorted aligned
       BAM (``SO:coordinate``); after unaligning every record there is no
       coordinate to sort by, so claiming SO:coordinate is a lie that
       bystrandify uses to pick its aligned-processing path, then
       silently rejects every record with the misleading "has 0
       PulseWidths" warning.
    """
    out = dict(header_dict)
    out["SQ"] = []
    if "HD" in out:
        out["HD"] = dict(out["HD"])
        out["HD"]["SO"] = "unknown"
    return out


# Module-global so forked workers inherit the kin_maps via copy-on-write
# (zero-copy on Linux; matters because the maps are ~256 MB per contig).
_WORKER_KIN_MAPS: dict[str, np.ndarray] = {}
_WORKER_CONFIG: dict = {}


def _worker_init(kin_maps: dict[str, np.ndarray], cfg_dict: dict) -> None:
    """Called once per worker at pool startup. Stores the kin_maps as a
    module-global so subsequent task calls can reach them without re-pickling
    (kin_maps are 256 MB each — never pickle them per task)."""
    global _WORKER_KIN_MAPS, _WORKER_CONFIG
    _WORKER_KIN_MAPS = kin_maps
    _WORKER_CONFIG = cfg_dict


def _worker_process_batch(
    batch: list[dict],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Run by each worker process. Takes a list of picklable read-descriptors,
    returns list of (fi, fp, ri, rp) tuples in the same order."""
    default_value = _WORKER_CONFIG["default_value"]
    n_context_skip = _WORKER_CONFIG["n_context_skip"]
    per_read_z = _WORKER_CONFIG.get("per_read_z", True)
    out = []
    for item in batch:
        kin_map = _WORKER_KIN_MAPS.get(item["ref_name"])
        if kin_map is None:
            # Unknown ref — should be filtered out by master, but be defensive
            empty = np.full(item["qlen"], default_value, dtype=np.uint8)
            out.append((empty, empty.copy(), empty.copy(), empty.copy()))
            continue
        out.append(_process_read_from_cigar(
            qlen=item["qlen"],
            ref_start=item["ref_start"],
            cigartuples=item["cigar"],
            is_rev=item["is_rev"],
            kin_map=kin_map,
            z_seed=item["z_seed"],
            default_value=default_value,
            n_context_skip=n_context_skip,
            per_read_z=per_read_z,
        ))
    return out


def _process_mapped_read_lookup(
    read: pysam.AlignedSegment,
    kin_map: np.ndarray,                # (L, 4, n_z_samples) uint8
    rng: random.Random,
    n_z_samples: int,
    qlen: int,
    default_value: int,
    n_context_skip: int,
    per_read_z: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill fi/fp/ri/rp by looking up the precomputed kinetics map.

    ``per_read_z`` controls per-read coherence (see
    :func:`_process_read_from_cigar` for the trade-off discussion).

    Strand routing matches :func:`_process_mapped_read` exactly:
      is_reverse=False → fi←ipd_rev(ch2), fp←pw_rev(ch3), ri←ipd_fwd(ch0), rp←pw_fwd(ch1)
      is_reverse=True  → fi←ipd_fwd(ch0), fp←pw_fwd(ch1), ri←ipd_rev(ch2), rp←pw_rev(ch3)
    """
    fi = np.full(qlen, default_value, dtype=np.uint8)
    fp = np.full(qlen, default_value, dtype=np.uint8)
    ri = np.full(qlen, default_value, dtype=np.uint8)
    rp = np.full(qlen, default_value, dtype=np.uint8)
    pairs = read.get_aligned_pairs(matches_only=True)
    if not pairs:
        return fi, fp, ri, rp

    is_rev = bool(read.is_reverse)
    L = kin_map.shape[0]
    n_z = kin_map.shape[2]
    # Vectorise the per-base lookup. pairs is list of (q, ref).
    pair_arr = np.asarray(pairs, dtype=np.int64)
    q_arr = pair_arr[:, 0]
    r_arr = pair_arr[:, 1]
    # Filter: skip context-edge positions AND off-contig refs (defensive)
    valid = (
        (q_arr >= n_context_skip)
        & (q_arr < qlen - n_context_skip)
        & (r_arr >= 0)
        & (r_arr < L)
    )
    if not valid.any():
        return fi, fp, ri, rp
    q_valid = q_arr[valid]
    r_valid = r_arr[valid]
    if per_read_z:
        z_idx = rng.randrange(n_z)
        block = kin_map[r_valid, :, z_idx]
    else:
        z_indices = np.asarray([rng.randrange(n_z) for _ in range(r_valid.size)], dtype=np.int64)
        block = kin_map[r_valid[:, None], np.arange(4)[None, :], z_indices[:, None]]
    if is_rev:
        fi[q_valid] = block[:, 0]
        fp[q_valid] = block[:, 1]
        ri[q_valid] = block[:, 2]
        rp[q_valid] = block[:, 3]
    else:
        fi[q_valid] = block[:, 2]
        fp[q_valid] = block[:, 3]
        ri[q_valid] = block[:, 0]
        rp[q_valid] = block[:, 1]
    return fi, fp, ri, rp


def _process_mapped_read(
    read: pysam.AlignedSegment,
    ref_seq: str,
    fwd_map: np.ndarray,
    rev_map: np.ndarray,
    fwd_frac: np.ndarray,
    rev_frac: np.ndarray,
    g: TransformerGenerator,
    n_meth_types: int,
    half_width: int,
    n_context_skip: int,
    default_value: int,
    use_bernoulli: bool,
    rng: random.Random,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate fi/fp/ri/rp for all query positions of an aligned read.

    Strand routing matches ``kinsim2.generate._route_strands``: the model
    emits (ipd_fwd, pw_fwd, ipd_rev, pw_rev) where ipd_fwd is + strand
    methylation kinetics. PacBio BAM convention:

      * is_reverse=False : fi←ipd_rev, fp←pw_rev, ri←ipd_fwd, rp←pw_fwd
      * is_reverse=True  : fi←ipd_fwd, fp←pw_fwd, ri←ipd_rev, rp←pw_rev

    No query-position flip is needed: pysam's get_aligned_pairs returns
    q_pos already in BAM-storage order, and we want to write tags in the
    same BAM-storage order.
    """
    qlen = read.query_length
    fi = np.full(qlen, default_value, dtype=np.uint8)
    fp = np.full(qlen, default_value, dtype=np.uint8)
    ri = np.full(qlen, default_value, dtype=np.uint8)
    rp = np.full(qlen, default_value, dtype=np.uint8)

    pairs = read.get_aligned_pairs(matches_only=True)
    if not pairs:
        return fi, fp, ri, rp
    pair_arr = np.asarray(pairs, dtype=np.int64)
    is_rev = bool(read.is_reverse)

    # ONE Bernoulli draw per (read, labelled site) — fixed for the duration
    # of this read so adjacent window queries see consistent conditioning.
    eff_fwd, eff_rev = _draw_read_effective_meth(
        fwd_map, rev_map, fwd_frac, rev_frac, rng, use_bernoulli,
    )

    # Gather all windows that pass filtering, then batch G calls
    windows = []
    for q_pos, r_pos in pair_arr:
        if q_pos < n_context_skip or q_pos >= qlen - n_context_skip:
            continue
        if r_pos - half_width < 0 or r_pos + half_width + 1 > len(ref_seq):
            continue
        base_fwd = _encode_seq(ref_seq[r_pos - half_width: r_pos + half_width + 1])
        ref_window = np.arange(r_pos - half_width, r_pos + half_width + 1, dtype=np.int64)
        meth_fwd, meth_rev = _build_meth_window(eff_fwd, eff_rev, ref_window)
        windows.append((int(q_pos), base_fwd, meth_fwd, meth_rev))
    if not windows:
        return fi, fp, ri, rp

    for start in range(0, len(windows), batch_size):
        chunk = windows[start: start + batch_size]
        bf = np.stack([w[1] for w in chunk], axis=0)        # (B, K) uint8
        mf = np.stack([w[2] for w in chunk], axis=0)
        mr = np.stack([w[3] for w in chunk], axis=0)
        centers = _generate_signal_batched(g, bf, mf, mr, n_meth_types, device)
        u8 = log1p_frames_to_uint8(centers)                  # (B, 4) uint8
        # Channel layout in u8: (IPD_fwd=+, PW_fwd=+, IPD_rev=−, PW_rev=−)
        for j, (q_pos, _, _, _) in enumerate(chunk):
            if is_rev:
                fi[q_pos] = u8[j, 0]   # ipd_fwd
                fp[q_pos] = u8[j, 1]   # pw_fwd
                ri[q_pos] = u8[j, 2]   # ipd_rev
                rp[q_pos] = u8[j, 3]   # pw_rev
            else:
                fi[q_pos] = u8[j, 2]   # ipd_rev (BAM fi = − strand kinetics)
                fp[q_pos] = u8[j, 3]   # pw_rev
                ri[q_pos] = u8[j, 0]   # ipd_fwd
                rp[q_pos] = u8[j, 1]   # pw_fwd
    return fi, fp, ri, rp


def _process_unmapped_read(
    read: pysam.AlignedSegment,
    motif_string: str,
    g: TransformerGenerator,
    n_meth_types: int,
    half_width: int,
    n_context_skip: int,
    default_value: int,
    use_bernoulli: bool,
    rng: random.Random,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Generate fi/fp/ri/rp for an UNMAPPED read (e.g. PBSIM3 sim output).

    Without a reference alignment, we scan motifs against the query
    sequence itself. The query is the synthesis-strand sequence (no
    reverse handling needed); we write channels in canonical order.
    """
    qseq = read.query_sequence
    if qseq is None:
        return None
    qlen = len(qseq)
    fi = np.full(qlen, default_value, dtype=np.uint8)
    fp = np.full(qlen, default_value, dtype=np.uint8)
    ri = np.full(qlen, default_value, dtype=np.uint8)
    rp = np.full(qlen, default_value, dtype=np.uint8)

    # Scan motifs against THIS read's sequence (it serves as the "reference"
    # for an unaligned read).
    fwd_map_q, rev_map_q, fwd_frac_q, rev_frac_q = _build_strand_meth_maps(
        {"_query": qseq}, motif_string,
    )
    fmap, rmap = fwd_map_q["_query"], rev_map_q["_query"]
    ffrac, rfrac = fwd_frac_q["_query"], rev_frac_q["_query"]

    # ONE Bernoulli draw per (read, labelled site) — fixed for this read.
    eff_fwd, eff_rev = _draw_read_effective_meth(
        fmap, rmap, ffrac, rfrac, rng, use_bernoulli,
    )

    windows = []
    for q_pos in range(max(half_width, n_context_skip),
                       qlen - max(half_width, n_context_skip)):
        base_fwd = _encode_seq(qseq[q_pos - half_width: q_pos + half_width + 1])
        ref_window = np.arange(q_pos - half_width, q_pos + half_width + 1, dtype=np.int64)
        meth_fwd, meth_rev = _build_meth_window(eff_fwd, eff_rev, ref_window)
        windows.append((q_pos, base_fwd, meth_fwd, meth_rev))
    if not windows:
        return fi, fp, ri, rp

    for start in range(0, len(windows), batch_size):
        chunk = windows[start: start + batch_size]
        bf = np.stack([w[1] for w in chunk], axis=0)
        mf = np.stack([w[2] for w in chunk], axis=0)
        mr = np.stack([w[3] for w in chunk], axis=0)
        centers = _generate_signal_batched(g, bf, mf, mr, n_meth_types, device)
        u8 = log1p_frames_to_uint8(centers)
        # Unmapped reads have no alignment direction. Treat the query
        # sequence as the molecule's natural synthesis-order strand, so
        # fi (pass 1) captures − strand kinetics and ri captures +. Same
        # as the forward-mapped branch.
        for j, (q_pos, _, _, _) in enumerate(chunk):
            fi[q_pos] = u8[j, 2]   # ipd_rev
            fp[q_pos] = u8[j, 3]   # pw_rev
            ri[q_pos] = u8[j, 0]   # ipd_fwd
            rp[q_pos] = u8[j, 1]   # pw_fwd
    return fi, fp, ri, rp


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _run_mapped_reads_multiprocess(
    in_bam: pysam.AlignmentFile,
    out_bam: pysam.AlignmentFile,
    kin_maps: dict[str, np.ndarray],
    n_z_samples: int,
    default_value: int,
    n_context_skip: int,
    rng: random.Random,
    n_workers: int,
    batch_size: int = 500,
    max_pending_batches: int | None = None,
    unmapped_passthrough: bool = True,
    per_read_z: bool = True,
    emit_unaligned: bool = True,
) -> tuple[int, int]:
    """Master loop: parallelise the per-read CIGAR + lookup + tag-build over
    a worker pool. Master keeps the sequential responsibilities (BAM read,
    BAM write, set_tag) and ships off the CPU-bound per-read math.

    Returns ``(n_processed, n_unmapped_or_skipped_for_main_path)``.

    Workers fork from master *after* kin_maps are precomputed → they inherit
    the maps via copy-on-write. Per-task IPC carries only the picklable
    read descriptor (qlen, ref_start, cigar, is_rev, ref_name, z_idx) and
    returns 4 uint8 arrays — minimal bandwidth.

    Read order is preserved by submitting batches into a FIFO deque of
    (reads_in_batch, future) and consuming results in submission order.
    """
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    if max_pending_batches is None:
        max_pending_batches = max(4, n_workers * 2)

    cfg_dict = {
        "n_z_samples": n_z_samples,
        "default_value": default_value,
        "n_context_skip": n_context_skip,
        "per_read_z": per_read_z,
    }

    n_processed = 0
    n_skipped_unmapped = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(kin_maps, cfg_dict),
    ) as executor:
        pending: deque = deque()
        read_iter = in_bam.fetch(until_eof=True)
        eof = False

        while True:
            # Fill the work queue
            while len(pending) < max_pending_batches and not eof:
                reads_batch: list[pysam.AlignedSegment] = []
                items: list[dict] = []
                while len(items) < batch_size:
                    try:
                        read = next(read_iter)
                    except StopIteration:
                        eof = True
                        break
                    if read.is_secondary or read.is_supplementary:
                        # Aligned-only artifacts; skip in unaligned mode (header
                        # has no @SQ to reference). Pass-through otherwise.
                        if not emit_unaligned:
                            out_bam.write(read)
                        continue
                    if read.is_unmapped:
                        if unmapped_passthrough:
                            out_bam.write(read)
                            n_skipped_unmapped += 1
                            continue
                        # else: caller handles unmapped separately
                        out_bam.write(read)
                        n_skipped_unmapped += 1
                        continue
                    ref_name = read.reference_name
                    if ref_name not in kin_maps:
                        log.warning("Skipping read on unknown ref: %s", ref_name)
                        continue
                    items.append({
                        "qlen": read.query_length,
                        "ref_start": read.reference_start,
                        "cigar": list(read.cigartuples) if read.cigartuples else [],
                        "is_rev": bool(read.is_reverse),
                        "ref_name": ref_name,
                        # Per-read seed → reproducible per-position random z
                        # inside the worker (NumPy default_rng with this seed).
                        "z_seed": rng.randrange(2**31),
                    })
                    reads_batch.append(read)
                if items:
                    future = executor.submit(_worker_process_batch, items)
                    pending.append((reads_batch, future))

            # Drain one completed batch (in submission order)
            if pending:
                reads_batch, future = pending.popleft()
                results = future.result()
                for read, (fi, fp, ri, rp) in zip(reads_batch, results):
                    read.set_tag("fi", array.array("B", fi.tobytes()))
                    read.set_tag("fp", array.array("B", fp.tobytes()))
                    read.set_tag("ri", array.array("B", ri.tobytes()))
                    read.set_tag("rp", array.array("B", rp.tobytes()))
                    if emit_unaligned:
                        _unalign_read(read)
                    out_bam.write(read)
                    n_processed += 1
                    if n_processed % 1000 == 0:
                        log.info("Processed %d mapped reads", n_processed)
            else:
                if eof:
                    break

    return n_processed, n_skipped_unmapped


def generate(
    input_bam: Path,
    ref_fasta: Path,
    ckpt_dir: Path,
    motifs_csv: Path,
    output_bam: Path,
    cfg_yaml: Path | None,
    seed: int = 42,
    use_bernoulli: bool = True,
    precompute_cache: bool = True,
    n_z_samples: int = 32,
    n_workers: int = 1,
    per_read_z: bool = True,
    emit_unaligned: bool = True,
) -> None:
    setup_logging()
    cfg = load_config(cfg_yaml)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    g, model_cfg = _load_generator(ckpt_dir, device)
    n_meth_types = int(model_cfg["n_meth_types"])
    K = int(model_cfg["k"])
    half_width = K // 2
    n_context_skip = max(cfg.generate.n_context_skip, half_width)

    # Load reference and build motif map per contig
    fa = pysam.FastaFile(str(ref_fasta))
    ref_seqs = {r: fa.fetch(r) for r in fa.references}
    fa.close()
    log.info("Reference contigs: %d", len(ref_seqs))

    # Use kinsim's canonical loader (handles PacBio CSV, REBASE, KinSim string;
    # converts centerPos 1-based → 0-based correctly; validates IUPAC base).
    motif_string = load_motif_string(
        str(motifs_csv), min_fraction=0.0, min_detected=0,
    )
    if not motif_string:
        log.warning("Empty motif string from %s — generate will produce baseline-only kinetics",
                    motifs_csv)
    # Count entries by splitting on ; and dropping empties (handles trailing ;).
    n_motifs = sum(1 for e in motif_string.split(";") if e.strip()) if motif_string else 0
    log.info("Loaded %d motifs from %s", n_motifs, motifs_csv)

    fwd_maps, rev_maps, fwd_fracs, rev_fracs = _build_strand_meth_maps(
        ref_seqs, motif_string,
    )
    total_sites = sum(
        int(np.count_nonzero(fwd_maps[r])) + int(np.count_nonzero(rev_maps[r]))
        for r in ref_seqs
    )
    log.info("Total methylation sites across reference: %d", total_sites)

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----------------------------------------------------------------------
    # FAST PATH: precompute a per-contig kinetics map (one inference per K
    # ref positions × n_z_samples), then each MAPPED read just looks up.
    # ~100-200× faster than the per-position fallback, and uses GLOBAL
    # Bernoulli (one decision per genomic site, matching PacBio fraction
    # semantics). Unmapped reads still go through the per-position path.
    # ----------------------------------------------------------------------
    kin_maps: dict[str, np.ndarray] = {}
    if precompute_cache:
        log.info("Precompute path: drawing GLOBAL Bernoulli + generating per-contig kinetics maps")
        eff_fwds: dict[str, np.ndarray] = {}
        eff_revs: dict[str, np.ndarray] = {}
        for rid in ref_seqs:
            ef, er = _draw_global_effective_meth(
                fwd_maps[rid], rev_maps[rid],
                fwd_fracs[rid], rev_fracs[rid],
                rng, use_bernoulli,
            )
            eff_fwds[rid] = ef
            eff_revs[rid] = er
        total_after_bernoulli = sum(
            int(np.count_nonzero(eff_fwds[r])) + int(np.count_nonzero(eff_revs[r]))
            for r in ref_seqs
        )
        log.info("  GLOBAL Bernoulli kept %d/%d meth sites", total_after_bernoulli, total_sites)

        for rid, seq in ref_seqs.items():
            log.info("Precomputing kinetics map: %s (len=%d)", rid, len(seq))
            kin_maps[rid] = _precompute_kinetics_map_for_contig(
                seq, eff_fwds[rid], eff_revs[rid],
                g, n_meth_types, K, half_width, n_z_samples, device,
            )
            mb = kin_maps[rid].nbytes / 1e6
            log.info("  %s kin_map: %.1f MB (L=%d × 4 ch × %d z)", rid, mb, len(seq), n_z_samples)
        log.info("Precompute done. Kinetics cached for %d contigs.", len(kin_maps))

    in_bam = pysam.AlignmentFile(str(input_bam), "rb", check_sq=False)
    header = in_bam.header.to_dict()
    pg = header.setdefault("PG", [])
    pg.append({
        "ID": f"kinsim_nn-{__version__}",
        "PN": "kinsim_nn",
        "VN": __version__,
        "CL": f"kinsim_nn generate (ckpt={ckpt_dir.name})",
    })
    if emit_unaligned:
        header = _sanitize_header_for_unaligned(header)
        log.info("emit_unaligned=True: stripped @SQ lines; reads will be written unmapped")
    out_bam = pysam.AlignmentFile(str(output_bam), "wb", header=header)

    n_reads = 0
    n_unmapped = 0

    # FAST MULTIPROCESS PATH — only applies when precompute_cache is on,
    # n_workers > 1, AND the input BAM has mapped reads. For unmapped reads
    # (PBSIM3 sim output), per-read GPU work is still needed → fallback to
    # the sequential path. We detect by peeking: if input is mapped-mostly,
    # use multiprocess; otherwise fall through.
    if precompute_cache and n_workers > 1:
        log.info("Multiprocess path enabled: %d workers, batch=500", n_workers)
        n_proc, n_skip = _run_mapped_reads_multiprocess(
            in_bam=in_bam,
            out_bam=out_bam,
            kin_maps=kin_maps,
            n_z_samples=n_z_samples,
            default_value=cfg.generate.default_fi_for_unknown,
            n_context_skip=n_context_skip,
            rng=rng,
            n_workers=n_workers,
            unmapped_passthrough=True,
            per_read_z=per_read_z,
            emit_unaligned=emit_unaligned,
        )
        log.info("Multiprocess done. mapped=%d  unmapped_passthrough=%d",
                 n_proc, n_skip)
        n_reads = n_proc
        n_unmapped = n_skip
        in_bam.close()
        out_bam.close()
        log.info("Done. Wrote %d reads (%d unmapped passthrough) -> %s",
                 n_reads, n_unmapped, output_bam)
        return

    for read in in_bam.fetch(until_eof=True):
        if read.is_secondary or read.is_supplementary:
            if not emit_unaligned:
                out_bam.write(read)
            continue
        if read.is_unmapped:
            # PBSIM3 sim output or any unaligned read: scan motifs against
            # the query sequence directly.
            result = _process_unmapped_read(
                read=read,
                motif_string=motif_string,
                g=g,
                n_meth_types=n_meth_types,
                half_width=half_width,
                n_context_skip=n_context_skip,
                default_value=cfg.generate.default_fi_for_unknown,
                use_bernoulli=use_bernoulli,
                rng=rng,
                device=device,
            )
            if result is None:
                out_bam.write(read)
                continue
            fi, fp, ri, rp = result
            n_unmapped += 1
        else:
            ref_id = read.reference_name
            if ref_id not in ref_seqs:
                log.warning("Skipping read on unknown ref: %s", ref_id)
                continue
            if precompute_cache:
                fi, fp, ri, rp = _process_mapped_read_lookup(
                    read=read,
                    kin_map=kin_maps[ref_id],
                    rng=rng,
                    n_z_samples=n_z_samples,
                    qlen=read.query_length,
                    default_value=cfg.generate.default_fi_for_unknown,
                    n_context_skip=n_context_skip,
                    per_read_z=per_read_z,
                )
            else:
                fi, fp, ri, rp = _process_mapped_read(
                    read=read,
                    ref_seq=ref_seqs[ref_id],
                    fwd_map=fwd_maps[ref_id],
                    rev_map=rev_maps[ref_id],
                    fwd_frac=fwd_fracs[ref_id],
                    rev_frac=rev_fracs[ref_id],
                    g=g,
                    n_meth_types=n_meth_types,
                    half_width=half_width,
                    n_context_skip=n_context_skip,
                    default_value=cfg.generate.default_fi_for_unknown,
                    use_bernoulli=use_bernoulli,
                    rng=rng,
                    device=device,
                )
        # ``array.array('B', bytes)`` is ~10× faster than ``.tolist()`` —
        # matters at 124k reads.
        read.set_tag("fi", array.array("B", fi.tobytes()))
        read.set_tag("fp", array.array("B", fp.tobytes()))
        read.set_tag("ri", array.array("B", ri.tobytes()))
        read.set_tag("rp", array.array("B", rp.tobytes()))
        if emit_unaligned:
            _unalign_read(read)
        out_bam.write(read)
        n_reads += 1
        if n_reads % 100 == 0:
            log.info("Processed %d reads", n_reads)

    in_bam.close()
    out_bam.close()
    log.info("Done. Wrote %d reads (%d unmapped) -> %s",
             n_reads, n_unmapped, output_bam)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kinsim_nn generate", description=__doc__)
    ap.add_argument("input_bam")
    ap.add_argument("ref_fasta")
    ap.add_argument("ckpt_dir")
    ap.add_argument("motifs_csv")
    ap.add_argument("output_bam")
    ap.add_argument("--config", default=None, help="kinsim_nn_config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-precompute", action="store_true",
                    help="Disable the precompute-cache fast path. With this set, "
                         "each (read, position) gets its own model inference "
                         "(~100-200× slower, but per-read per-site Bernoulli).")
    ap.add_argument("--n-z-samples", type=int, default=32,
                    help="Number of z noise realisations to pre-draw per genomic "
                         "position when --precompute is on. Each (read, base) "
                         "picks one of these N samples INDEPENDENTLY → much "
                         "more apparent variance than fixed-z-per-read. "
                         "Higher = more variance, more memory (L × 4 × N bytes). "
                         "Default: %(default)d (= 1 GB per 8 Mbp contig).")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="Number of worker processes for parallel read "
                         "processing (only used with precompute path). Workers "
                         "fork from master after kin_maps are built, inheriting "
                         "them via copy-on-write (no per-task pickling). Set to "
                         "1 to disable multiprocess (sequential mapped-read "
                         "processing). Typically set to --cpus-per-task value. "
                         "Default: %(default)d (sequential).")
    ap.add_argument("--no-bernoulli", action="store_true",
                    help="Override YAML and disable Bernoulli sampling on motif fraction "
                         "(always set meth at every motif site).")
    ap.add_argument("--force-bernoulli", action="store_true",
                    help="Override YAML and force Bernoulli sampling on motif fraction.")
    ap.add_argument("--no-per-read-z", action="store_true",
                    help="Use per-(read, base) random z_idx instead of one z per read. "
                         "Max apparent variance but breaks read-level coherence — "
                         "desensitises ipdSummary's per-position tests. Kept for "
                         "ablation; v2 default is per-read-z=True.")
    ap.add_argument("--no-emit-unaligned", action="store_true",
                    help="Keep alignment fields on output reads. Default (emit-"
                         "unaligned) writes flag=4 / ref_id=-1 / cigar=None so the "
                         "BAM is directly consumable by ccs-kinetics-bystrandify.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.verbose:
        setup_logging(verbose=True)

    # Resolve use_bernoulli: CLI flags override YAML; otherwise YAML default.
    cfg = load_config(Path(args.config) if args.config else None)
    use_bernoulli = cfg.generate.use_fraction_bernoulli
    if args.no_bernoulli:
        use_bernoulli = False
    if args.force_bernoulli:
        use_bernoulli = True

    generate(
        input_bam=Path(args.input_bam),
        ref_fasta=Path(args.ref_fasta),
        ckpt_dir=Path(args.ckpt_dir),
        motifs_csv=Path(args.motifs_csv),
        output_bam=Path(args.output_bam),
        cfg_yaml=Path(args.config) if args.config else None,
        seed=args.seed,
        use_bernoulli=use_bernoulli,
        precompute_cache=not args.no_precompute,
        n_z_samples=args.n_z_samples,
        n_workers=args.n_workers,
        per_read_z=not args.no_per_read_z,
        emit_unaligned=not args.no_emit_unaligned,
    )


if __name__ == "__main__":
    main()
