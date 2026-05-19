"""Generate kinetic signals for PBSIM3 reads using a trained MLP predictor.

The model predicts (μ, σ) per context,
then either samples from N(μ, σ²) (stochastic, default) or returns μ directly
(deterministic, via --deterministic flag).

Two calling modes (auto-detected):
  Directory mode:   kinsim mlp generate <pbsim3_dir> <checkpoint.pt> <motifs> <output_dir>
  Per-genome mode:  kinsim mlp generate <fq.gz> <maf.gz> <ref.fna> <ckpt.pt> <motifs> <out.bam>

Directory mode supports the same two layouts as dictionary inject (auto-detected):
  - Species subdirectories: pbsim3_dir/Ecoli/, pbsim3_dir/Salmonella/, ...
  - Flat layout: all files directly in pbsim3_dir, matched by basename.

Motif input (auto-detected):
  - KinSim motif string       — "m6A,GATC,1;m4C,CCWGG,1"  (applied to all species)
  - Per-species mapping file  — text file with "species_name|motif_string" per line
  - PacBio motifs.csv         — file path ending in .csv
  - REBASE file               — any other file path

The reference genome is pre-scanned once for methylation sites using EMBOSS
fuzznuc as the primary backend (falls back to Python regex automatically if
fuzznuc is not installed).  Results are cached in O(1)-lookup arrays,
avoiding repeated per-read scanning.

Output BAMs use the suffix _mlp.bam: unaligned BAM (flag=4) with all four
PacBio kinetic tags:
  fi:B:C  — forward strand IPD (polymerase on template strand)
  fp:B:C  — forward strand PW
  ri:B:C  — reverse strand IPD (polymerase on complementary strand)
  rp:B:C  — reverse strand PW

For ri/rp, the kmer context is RC(forward_kmer_at_position_i): the polymerase
reading the reverse strand encounters the reverse-complement sequence, so the
11-mer it "sees" at each position is the RC of the forward 11-mer.
The methylation state (meth_id) is shared: the meth_map (built with revcomp=True)
already encodes both-strand methylation at each reference position.
"""

from __future__ import annotations

import array
import gzip
import json
import logging
import os
import sys

import numpy as np
import pysam
import torch
import torch.nn as nn

from .models.predictor import (
    ConvPredictor,
    create_from_config,
    load_state_dict_from_ckpt,
)
from .utils.encoding import (
    BASE_MAP,
    KMER_MASK,
    KMER_PRED_IDX,
    K,
)
from .utils.io import (
    find_pbsim3_files,
    get_extended_context,
    load_reference,
    parse_maf,
    resolve_motifs_for_species,
)
from .utils.motifs import (
    build_reference_frac_map,
    build_reference_meth_map,
    filter_motif_string_by_types,
    load_motif_string,
    parse_meth_types_arg,
    parse_motifs,
    scan_sequence,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RC kmer helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lookup-table mode (model distillation)
# ---------------------------------------------------------------------------
#
# Instead of running the model.forward() per position, pre-compute (μ, σ)
# in log-space for every (kmer, scenario) and consume that table at
# inference time. The table is produced by ``kinsim predict-kmers`` —
# loaded once and shared across all reads. Pure-numpy, no GPU, trivially
# parallelisable.
#
# Scenario detection (per row):
#   For each k_pos in [0, K) where meth_ctx[k_pos] != 0:
#     offset = KMER_PRED_IDX - k_pos      # offset of the meth from pred pos
#     scenario = (meth_id, offset) IF that pair appears in signal_offsets,
#                else "none"
#   First match wins (signal_offsets[T] is small and ordered).
#
# The meth_ctx[k_pos] convention is the same as the asymmetric kmer window:
#   k_pos = 0  → 7 bases upstream of pred (offset +7)
#   k_pos = 7  → pred position (offset 0)
#   k_pos = 10 → 3 bases downstream of pred (offset -3)


def _load_lookup_table(npz_path: str) -> dict:
    """Load a ``predict-kmers``-produced LUT.

    Returns a dict with:
        ``lut_log``        (N_scenarios, N_kmers, 4) float32 — log-space
                           (mu_ipd, mu_pw, sigma_ipd, sigma_pw).
        ``scenario_matrix`` (N_meth, K) int64 — scenario_idx for each
                           (meth_id, k_pos); 0 = none/fallback.
        ``scenario_labels`` list[str] for logging.
    """
    from .utils.encoding import KMER_PRED_IDX as _PI
    from .utils.encoding import K as _K
    from .utils.encoding import get_meth_ids

    data = np.load(npz_path, allow_pickle=False)
    required = {"scenarios_label", "scenarios_meth_id", "scenarios_offset"}
    if not required.issubset(set(data.files)):
        raise ValueError(
            f"{npz_path} missing scenario metadata — re-run kinsim predict-kmers "
            f"(needs scenarios_{{label,meth_id,offset}} arrays)."
        )

    labels = [str(s) for s in data["scenarios_label"].tolist()]
    m_ids = data["scenarios_meth_id"].astype(np.int64)
    offsets = data["scenarios_offset"].astype(np.int64)

    # Find a "none" scenario (meth_id == 0). predict-kmers always emits it.
    none_idx = next(
        (i for i, mid in enumerate(m_ids) if int(mid) == 0),
        None,
    )
    if none_idx is None:
        raise ValueError(f"{npz_path} has no 'none' scenario — re-run predict-kmers.")

    # Build (N_scenarios, N_kmers, 4) array of log-space μ/σ.
    n_kmers = int(data["kmer_id"].shape[0])
    n_scenarios = len(labels)
    lut_log = np.empty((n_scenarios, n_kmers, 4), dtype=np.float32)
    for i, label in enumerate(labels):
        sk = label.replace("@", "_at_").replace("+", "p").replace("-", "m")
        for k_log_field, slot in (
            ("mu_ipd_log", 0),
            ("mu_pw_log", 1),
            ("sigma_ipd_log", 2),
            ("sigma_pw_log", 3),
        ):
            key = f"{sk}__{k_log_field}"
            if key not in data.files:
                raise KeyError(
                    f"{npz_path} missing '{key}' — re-run predict-kmers with "
                    "log-space output (recent change)."
                )
            lut_log[i, :, slot] = data[key].astype(np.float32)

    # Build scenario_matrix: scenario_matrix[meth_id, k_pos] -> scenario_idx
    # (0 = none-fallback). meth_id ranges over user-declared types.
    meth_ids_map = get_meth_ids()
    num_m = max(meth_ids_map.values()) + 1
    scenario_matrix = np.zeros((num_m, _K), dtype=np.int64)  # default → none_idx
    scenario_matrix.fill(int(none_idx))
    # For each non-none scenario: place into the (meth_id, k_pos) cell.
    for i, (mid, off) in enumerate(zip(m_ids, offsets)):
        if int(mid) == 0:
            continue
        k_pos = _PI - int(off)
        if 0 <= k_pos < _K and int(mid) < num_m:
            scenario_matrix[int(mid), k_pos] = i

    log.info(
        "LUT loaded from %s: %d scenarios × %d kmers × 4 outputs (%.0f MB log-space)",
        npz_path,
        n_scenarios,
        n_kmers,
        lut_log.nbytes / 1e6,
    )
    log.info("  scenarios: %s", labels)

    return {
        "lut_log": lut_log,
        "scenario_matrix": scenario_matrix,
        "scenario_labels": labels,
        "none_idx": int(none_idx),
        "pred_idx": _PI,
        "K": _K,
        "n_kmers": n_kmers,
    }


@torch.no_grad()
def generate_signals_lookup(
    lut: dict,
    kmer_ids,
    meth_contexts,
    deterministic: bool = False,
) -> np.ndarray:
    """Pure-numpy LUT-based signal generation.

    Replaces ``generate_signals_batch`` when the model has been distilled
    into a (kmer, scenario) lookup table. Sampling is done in **log1p
    space** (same as training) then ``inv_log_transform``-ed back to
    uint8-equivalent units — bit-equivalent semantics with the model
    path for any row whose meth_ctx matches a known scenario.

    Args
    ----
    lut             dict from :func:`_load_lookup_table`.
    kmer_ids        array-like (N,) of 22-bit kmer IDs.
    meth_contexts   array-like (N, K) int64 meth context windows.
    deterministic   if True, return the mean (no sampling).

    Returns
    -------
    np.ndarray (N, 2) — IPD and PW in [0, 255], float32.
    """
    K_ = lut["K"]
    pred_idx = lut["pred_idx"]
    scenario_matrix = lut["scenario_matrix"]
    lut_log = lut["lut_log"]
    none_idx = lut["none_idx"]

    kmer_arr = np.asarray(kmer_ids, dtype=np.int64)
    ctx = np.asarray(meth_contexts, dtype=np.int64)
    if ctx.ndim == 1:
        ctx = ctx.reshape(1, -1)
    N = kmer_arr.shape[0]

    # Identify scenario per row:
    #   scenarios_per_pos[i, k] = scenario_matrix[ctx[i, k], k]  (none_idx by default)
    # Then pick the FIRST non-none scenario across k_pos in [0, K).
    col_idx = np.broadcast_to(np.arange(K_)[None, :], ctx.shape)
    nonzero_mask = ctx > 0
    if nonzero_mask.any():
        # Default everyone to none, then overwrite for valid scenarios.
        scenarios_per_pos = np.full(ctx.shape, none_idx, dtype=np.int64)
        scenarios_per_pos[nonzero_mask] = scenario_matrix[ctx[nonzero_mask], col_idx[nonzero_mask]]
        # First non-none wins — pick the smallest k_pos where scenario != none_idx.
        is_real = scenarios_per_pos != none_idx  # (N, K)
        any_real = is_real.any(axis=1)  # (N,)
        first_real_kpos = np.argmax(is_real, axis=1)  # 0 if no real (we'll mask)
        scenario_idx = np.where(
            any_real,
            scenarios_per_pos[np.arange(N), first_real_kpos],
            none_idx,
        )
    else:
        scenario_idx = np.full(N, none_idx, dtype=np.int64)

    # LUT lookup: rows of (mu_ipd_log, mu_pw_log, sigma_ipd_log, sigma_pw_log).
    rows = lut_log[scenario_idx, kmer_arr]  # (N, 4)
    mu_ipd_log = rows[:, 0]
    mu_pw_log = rows[:, 1]
    sigma_ipd_log = rows[:, 2]
    sigma_pw_log = rows[:, 3]

    if deterministic:
        ipd_log = mu_ipd_log
        pw_log = mu_pw_log
    else:
        ipd_log = mu_ipd_log + sigma_ipd_log * np.random.randn(N).astype(np.float32)
        pw_log = mu_pw_log + sigma_pw_log * np.random.randn(N).astype(np.float32)

    # inv_log_transform: expm1 clamped to [0, 255]
    ipd = np.clip(np.expm1(ipd_log), 0.0, 255.0).astype(np.float32)
    pw = np.clip(np.expm1(pw_log), 0.0, 255.0).astype(np.float32)
    return np.stack([ipd, pw], axis=1)


def _rc_kmer_vec(kmer_ids: np.ndarray) -> np.ndarray:
    """Vectorised reverse complement of 11-mer IDs (matches scalar _rc_kmer)."""
    rc = np.zeros_like(kmer_ids, dtype=np.int64)
    k = kmer_ids.astype(np.int64).copy()
    for _ in range(K):
        base = k & 3
        rc = (rc << 2) | (base ^ 3)
        k >>= 2
    return rc


# Base lookup table — vectorised seq → int conversion. Non-ACGT → 0 (matches
# scalar ``BASE_MAP.get(c, 0)`` fallback). Lowercase handled the same way.
_BASE_LUT_GEN = np.zeros(256, dtype=np.int64)
for _ch, _v in zip(b"ACGT", (0, 1, 2, 3)):
    _BASE_LUT_GEN[_ch] = _v
for _ch, _v in zip(b"acgt", (0, 1, 2, 3)):
    _BASE_LUT_GEN[_ch] = _v

# Powers for kmer encoding: kmer_id = sum(base[j] * 4**(K-1-j)).
_KMER_POWERS = (4 ** np.arange(K - 1, -1, -1)).astype(np.int64)


def _process_read_unmapped_vec(
    seq: str,
    read_len: int,
    fallback_motifs,
    p_eff_lookup: dict,
    sig_offsets: dict,
    pred_idx: int,
    ctx_len: int,
):
    """Vectorised unmapped-path processing of a single read.

    Replaces the per-base Python loop with numpy whole-array ops. The
    semantics match the original ``_process_batch`` unmapped branch
    position-by-position; only the np.random call order is batched
    (still numpy's global RNG, still the same distribution).

    Returns
    -------
    kmer_ids       int64 (L,)     0 for the first ``K-1`` positions
    rc_kmer_ids    int64 (L,)
    meth_ids       int64 (L,)
    fractions      float32 (L,)
    meth_ctxs      int64 (L, ctx_len)
    is_n_context   bool   (L,)    True for the first ``K-1`` positions
    """
    L = read_len
    is_n_context = np.zeros(L, dtype=bool)
    is_n_context[: K - 1] = True

    # --- Sequence → base ints + rolling kmer IDs --------------------------
    seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    bases = _BASE_LUT_GEN[seq_arr] if seq_arr.size else np.empty(0, dtype=np.int64)
    kmer_ids = np.zeros(L, dtype=np.int64)
    if L >= K:
        windows = np.lib.stride_tricks.sliding_window_view(bases, K)
        kmer_ids[K - 1 :] = (windows.astype(np.int64) @ _KMER_POWERS) & KMER_MASK

    # --- Meth context window per position ---------------------------------
    # meth_ctxs[i, k_pos] = meth_status[i + k_pos - (K - 1)] for i >= K-1.
    meth_status = scan_sequence(seq, fallback_motifs)
    meth_status = np.asarray(meth_status, dtype=np.int64)
    meth_ctxs = np.zeros((L, ctx_len), dtype=np.int64)
    if L >= K and meth_status.size >= ctx_len:
        ms_windows = np.lib.stride_tricks.sliding_window_view(meth_status, ctx_len)
        target_n = L - K + 1
        n_copy = min(ms_windows.shape[0], target_n)
        meth_ctxs[K - 1 : K - 1 + n_copy] = ms_windows[:n_copy].astype(np.int64)

    # --- Vectorised _apply_p_fire_to_mc -----------------------------------
    if p_eff_lookup and meth_ctxs.size:
        max_T = max((T for T, _ in p_eff_lookup), default=0)
        if meth_ctxs.size:
            max_T = max(max_T, int(meth_ctxs.max()))
        rate_table = np.ones((max_T + 1, ctx_len), dtype=np.float32)
        for (T, k), p_eff in p_eff_lookup.items():
            k_pos = pred_idx - k
            if 0 <= k_pos < ctx_len and max_T >= T:
                if k in sig_offsets.get(T, set()):
                    rate_table[T, k_pos] = float(p_eff)
        # rates[i, k_pos] = rate_table[meth_ctxs[i, k_pos], k_pos]
        col_idx = np.broadcast_to(np.arange(ctx_len)[None, :], meth_ctxs.shape)
        rates = rate_table[meth_ctxs, col_idx]
        rolls = np.random.random(meth_ctxs.shape).astype(np.float32)
        no_fire = (meth_ctxs > 0) & (rates < 1.0) & (rolls >= rates)
        meth_ctxs[no_fire] = 0

    # --- Derived outputs --------------------------------------------------
    meth_ids = meth_ctxs[:, pred_idx].copy()
    fractions = (meth_ids > 0).astype(np.float32)
    rc_kmer_ids = _rc_kmer_vec(kmer_ids)

    # Zero everything for the first K-1 positions (matches original)
    if K - 1 > 0:
        kmer_ids[: K - 1] = 0
        rc_kmer_ids[: K - 1] = 0
        meth_ids[: K - 1] = 0
        fractions[: K - 1] = 0.0
        meth_ctxs[: K - 1] = 0

    return kmer_ids, rc_kmer_ids, meth_ids, fractions, meth_ctxs, is_n_context


def _rc_kmer(kmer_id: int) -> int:
    """Reverse complement a 22-bit encoded 11-mer.

    Encoding convention (big-endian): seq[0] occupies bits 21-20, seq[10]
    occupies bits 1-0.  Complement maps A↔T (0↔3) and C↔G (1↔2), i.e.
    XOR with 3 on each 2-bit unit.  Reversing bit-pair order yields the RC.

    Example: encode_kmer("ACGT...") → _rc_kmer() → encode_kmer(reverse_complement("ACGT..."))
    """
    rc = 0
    for _ in range(K):  # K = 11 iterations, one per base
        base = kmer_id & 3
        rc = (rc << 2) | (base ^ 3)  # complement: 0↔3 (A↔T), 1↔2 (C↔G)
        kmer_id >>= 2
    return rc


# ---------------------------------------------------------------------------
# Statistical firing — p_fire from GMM survival in refine
# ---------------------------------------------------------------------------


def _build_p_efficiency_lookup(
    p_fire_dict: dict | None,
    mean_occupancy_dict: dict | None,
) -> dict[tuple[int, int], float]:
    """Decompose training-corpus p_fire into the occupancy-independent
    ``p_efficiency = p_fire / mean_occupancy`` term, keyed by ``(meth_id, offset)``.

    Generate combines this with each target site's per-site frac:
        Bernoulli rate = target_frac × p_efficiency

    Returning ``{}`` (empty) makes generate fall back to "always fire" —
    the safe default for checkpoints predating this plumbing.
    """
    from .utils.encoding import get_meth_ids

    if not p_fire_dict:
        return {}
    occ_dict = mean_occupancy_dict or {}
    ids = get_meth_ids()
    out: dict[tuple[int, int], float] = {}
    for label, p in p_fire_dict.items():
        if "@" not in label:
            continue
        name, off_str = label.split("@", 1)
        T = ids.get(name)
        if T is None:
            continue
        try:
            offset = int(off_str)
        except ValueError:
            continue
        # Without mean_occupancy (older checkpoints), treat p_fire as the
        # composite rate and assume occupancy=1.0 so the decomposition is
        # an identity operation downstream.
        occ = float(occ_dict.get(label, 1.0))
        if occ <= 0.0:
            continue
        eff = float(p) / occ
        if eff > 1.0:
            eff = 1.0  # numerical safety: noise can push above 1
        out[(T, offset)] = eff
    return out


def _build_sig_offsets_by_meth_id() -> dict[int, set[int]]:
    """``{meth_id: {signature_offsets}}`` from kinsim_config.yaml.

    Used to recognise that a row at ``ref_pos`` with mc[i]=T is a
    *signature* offset of T — only those rows are subject to the firing
    Bernoulli; non-signature offsets pass through (the model trained on
    them as NEAR_METH-ish).
    """
    from .utils.config import get_signature_offsets, load_kinsim_config
    from .utils.encoding import get_meth_ids

    cfg = load_kinsim_config()
    ids = get_meth_ids()
    out: dict[int, set[int]] = {}
    for name in cfg.get("kinetic_signatures") or {}:
        T = ids.get(name)
        if T is not None:
            out[T] = set(get_signature_offsets(name))
    return out


def _apply_p_fire_to_mc(
    ctx: np.ndarray,
    p_eff_lookup: dict[tuple[int, int], float],
    sig_offsets: dict[int, set[int]],
    pred_idx: int,
    ref_pos: int,
    target_frac_arr: np.ndarray | None,
) -> None:
    """Roll an independent Bernoulli per non-zero mc entry; zero non-firing.

    For ``ctx[i] = T`` (a methylation in this row's context window) the
    row is at offset ``k = pred_idx - i`` of a canonical site at
    ``ref_pos - k`` (in this generate path mc is built in ref coords —
    canonical sits at ``ref_pos + (i - pred_idx)``).

    Bernoulli rate = ``target_site_frac × p_efficiency[(T, k)]``. The
    target frac comes from the per-position ``target_frac_arr`` (built
    from the destination genome's motifs.csv) so each ref site uses
    *its own* occupancy, not the training-corpus average.

    On no-fire, ``ctx[i]`` is zeroed: the model emits baseline-like
    signal at this row, which simultaneously kills the canonical centre
    AND its phantom +5 / +2 / +6 footprints.
    """
    if not p_eff_lookup:
        return
    n = target_frac_arr.shape[0] if target_frac_arr is not None else 0
    for i in range(len(ctx)):
        T = int(ctx[i])
        if T == 0:
            continue
        k = pred_idx - i
        offsets = sig_offsets.get(T)
        if not offsets or k not in offsets:
            continue
        p_eff = p_eff_lookup.get((T, k), 1.0)
        # Canonical ref_pos: row sits at offset (i - pred_idx) from canonical
        # in ref coords (mc was built from ref_meth, both strands collapsed).
        canonical_pos = ref_pos + (i - pred_idx)
        if target_frac_arr is not None and 0 <= canonical_pos < n:
            target_frac = float(target_frac_arr[canonical_pos])
        else:
            target_frac = 1.0  # fallback (unmapped path / out-of-bounds)
        rate = target_frac * p_eff
        if rate < 1.0 and np.random.random() >= rate:
            ctx[i] = 0


# ---------------------------------------------------------------------------
# Batched MLP inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_signals_batch(
    model: ConvPredictor,
    kmer_ids: list,
    meth_ids: list,
    fractions: list,
    meth_contexts: list,
    device: torch.device,
    deterministic: bool = False,
    rev_meth_contexts=None,
) -> np.ndarray:
    """Generate IPD/PW signals for a batch of contexts.

    Builds the per-position one-hot meth tensor from each row's mc context
    (the per-row Bernoulli has already happened upstream — fractions arrive
    as 0 or 1) and runs the model in chunks to bound GPU memory.

    Args:
        rev_meth_contexts: Optional (N, n_rev_meth) int64 array of
            complementary-strand meth_id at the active-site neighbours
            (offsets from ``ExtractionParams.rev_meth_offsets``). When
            ``None``, the rev_meth block stays zero — matches the legacy
            behaviour, but loses the partner-strand signal at palindromic
            motifs the model was trained on.

    Returns:
        np.ndarray of shape (N, 2) with raw [IPD, PW] values in [0, 255].
    """
    N = len(kmer_ids)
    CHUNK = 50_000  # positions per GPU forward pass — prevents OOM on long reads

    meth_ids_bin = np.asarray(meth_ids, dtype=np.int64)
    fractions_bin = np.asarray(fractions, dtype=np.float32)

    # Window geometry comes from the trained model's config — every shape
    # downstream (kmer_size, active-site index, rev_meth count, num meth
    # types) flows from there, so changing kinsim_config.yaml to K=21 and
    # retraining propagates automatically without touching this code.
    cfg = model.get_config()
    K_SIZE = int(cfg.get("kmer_size", 11))
    PRED_IDX = int(cfg.get("active_site_index", 7))
    N_REV = int(cfg.get("n_rev_meth", 3))
    TOTAL_POS = K_SIZE + N_REV
    NUM_M = int(cfg.get("num_meth_types", 4))
    ctx_np = np.asarray(meth_contexts, dtype=np.int64)
    # meth_full layout matches the trained dataset:
    #   positions [0, K_SIZE)          → forward meth context (offsets [-upstream..+downstream])
    #   positions [K_SIZE, TOTAL_POS)  → rev_meth at active-site neighbours
    # When ``rev_meth_contexts`` is provided, the rev_meth block is filled
    # from the complementary-strand meth map — matches training. When
    # ``None`` the block stays zero (legacy behaviour, loses palindromic
    # partner-strand signal).
    meth_full_np = np.zeros((N, TOTAL_POS, NUM_M), dtype=np.float32)

    # Forward block scatter.
    non_zero = ctx_np > 0
    if non_zero.any():
        rows, cols = np.where(non_zero)
        meth_full_np[rows, cols, ctx_np[rows, cols]] = 1.0

    # rev_meth block scatter (optional — current path is partial without it).
    if rev_meth_contexts is not None:
        rev_ctx_np = np.asarray(rev_meth_contexts, dtype=np.int64)
        if rev_ctx_np.shape != (N, N_REV):
            raise ValueError(
                f"rev_meth_contexts shape {rev_ctx_np.shape} != expected (N, n_rev_meth) "
                f"= ({N}, {N_REV}). Check rev_meth_offsets vs model.n_rev_meth."
            )
        rev_non_zero = rev_ctx_np > 0
        if rev_non_zero.any():
            r_rows, r_cols = np.where(rev_non_zero)
            meth_full_np[r_rows, K_SIZE + r_cols, rev_ctx_np[r_rows, r_cols]] = 1.0

    # Override the pred_idx column with stoichiometric fraction (0 or 1 per row).
    # The single-scatter above set meth_full_np[i, PRED_IDX, meth_id] = 1.0 for
    # non-zero meth_id rows; we clear that column then re-set with fractions so
    # zero-fraction (no-fire) rows correctly land at all-zeros.
    meth_full_np[:, PRED_IDX, :] = 0.0
    meth_full_np[np.arange(N), PRED_IDX, meth_ids_bin] = fractions_bin

    def _run_chunk(k_slice, mf_slice):
        k_t = torch.tensor(k_slice, dtype=torch.long, device=device)
        mf_t = torch.tensor(mf_slice, dtype=torch.float, device=device)
        if deterministic:
            return model.predict_mean(k_t, mf_t).cpu().numpy()
        else:
            return model.sample(k_t, mf_t).cpu().numpy()

    if N <= CHUNK:
        return _run_chunk(np.array(kmer_ids), meth_full_np)

    chunks = []
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        chunks.append(
            _run_chunk(
                np.array(kmer_ids[start:end]),
                meth_full_np[start:end],
            )
        )
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------


def _read_ckpt_meth_types(checkpoint_path: str) -> list[str] | None:
    """Return the ``meth_types`` list stored in the checkpoint's model_config.json.

    ``None`` means the checkpoint was trained on the full alphabet (no filter).
    Missing config file is treated as ``None`` — the caller reports any hard
    errors when actually loading the model.
    """
    config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path) as f:
            return json.load(f).get("meth_types")
    except (OSError, json.JSONDecodeError):
        return None


def _load_p_efficiency(checkpoint_path: str) -> dict[tuple[int, int], float]:
    """Read p_fire + mean_occupancy from model_config.json and decompose.

    Returns ``{(meth_id, offset): p_efficiency}``. Empty when the
    checkpoint predates this plumbing — generate falls back to "always
    fire" (no Bernoulli), which matches pre-decomposition behaviour.
    """
    config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.json")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return _build_p_efficiency_lookup(cfg.get("p_fire"), cfg.get("mean_occupancy"))


def _resolve_meth_types(
    cli_meth_types: set[str] | None,
    ckpt_meth_types: list[str] | None,
) -> set[str] | None:
    """Reconcile the CLI ``--meth-types`` override with the checkpoint's alphabet.

    Policy:
      - If the user passes ``--meth-types`` on the CLI, use it verbatim (and
        warn if it requests a type the model never saw during training).
      - Otherwise fall back to the checkpoint's recorded alphabet.
      - ``None`` at either level means "no filter" (accept all types).

    Warnings are important because silently generating signal for an unseen
    methylation type produces garbage — the model has no learned distribution
    for it and will emit whatever the unmethylated head defaults to.
    """
    ckpt_set = set(ckpt_meth_types) if ckpt_meth_types else None

    if cli_meth_types is None:
        if ckpt_set is not None:
            log.info("Using checkpoint alphabet: %s", sorted(ckpt_set))
        else:
            log.info("No --meth-types filter; using all types from the motif source")
        return ckpt_set

    if ckpt_set is not None:
        extra = cli_meth_types - ckpt_set
        if extra:
            log.warning(
                "--meth-types requests %s but checkpoint was trained on %s. "
                "Signal for the extra type(s) %s will be unreliable.",
                sorted(cli_meth_types),
                sorted(ckpt_set),
                sorted(extra),
            )
    log.info("Using CLI --meth-types override: %s", sorted(cli_meth_types))
    return cli_meth_types


def _apply_meth_types(
    motif_string: str,
    meth_types: set[str] | None,
) -> str:
    """Filter the motif string by allowed mod types and fail fast on empties."""
    if meth_types is None:
        return motif_string
    filtered = filter_motif_string_by_types(motif_string, meth_types)
    if not filtered:
        log.error(
            "After --meth-types filter (%s) no motifs remain. "
            "Check that your motif source contains the requested types.",
            sorted(meth_types),
        )
        sys.exit(1)
    return filtered


def _load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a trained ConvPredictor from a checkpoint file."""
    config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.json")
    if not os.path.exists(config_path):
        log.error(
            "model_config.json not found in %s. "
            "This file is written by 'kinsim train' at the start of training.",
            os.path.dirname(checkpoint_path),
        )
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # Geometry guard — the numpy hot path is K=11-specific.
    ckpt_kmer_size = int(config.get("kmer_size", K))
    ckpt_active_idx = int(config.get("active_site_index", config.get("KMER_PRED_IDX", 7)))
    if ckpt_kmer_size != K:
        log.error(
            "Checkpoint kmer_size=%d but generate.py only supports K=%d.",
            ckpt_kmer_size,
            K,
        )
        sys.exit(1)
    if ckpt_active_idx != KMER_PRED_IDX:
        log.error(
            "Checkpoint active_site_index=%d but generate.py expects %d.",
            ckpt_active_idx,
            KMER_PRED_IDX,
        )
        sys.exit(1)

    model = create_from_config(config).to(device)
    model.load_state_dict(load_state_dict_from_ckpt(checkpoint_path))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "Model loaded: K=%d  active_site_index=%d  params=%s  checkpoint=%s",
        ckpt_kmer_size,
        ckpt_active_idx,
        f"{n_params:,}",
        os.path.basename(checkpoint_path),
    )
    return model


# ---------------------------------------------------------------------------
# Main injection
# ---------------------------------------------------------------------------


def generate_signals(
    fastq_path: str,
    maf_path: str,
    ref_path: str,
    checkpoint_path: str,
    motif_string: str,
    output_bam: str,
    circular: bool = True,
    revcomp: bool = True,
    device: str = "cuda",
    batch_reads: int = 1000,
    no_fuzznuc: bool = False,
    deterministic: bool = False,
) -> None:
    """Inject MLP-predicted IPD/PW signals into PBSIM3 reads.

    Pipeline:
      1. Load reference genome
      2. Pre-scan reference for methylation sites (fuzznuc primary, regex fallback)
      3. Load trained ConvPredictor from checkpoint
      4. Parse .maf alignment mapping
      5. For batches of reads in .fq.gz:
         a. Collect all (kmer_id, meth_id) contexts using the pre-computed map
         b. Generate signals in one batched forward pass
         c. Write unaligned BAM with fi/fp/ri/rp tags

    Args:
        circular:      Treat genome as circular (default True for bacteria).
        revcomp:       Scan reverse complement strand for motifs (default True).
        no_fuzznuc:    Force Python regex for reference scanning; skip fuzznuc.
                       By default fuzznuc is tried first, falling back to regex
                       automatically if EMBOSS is not installed.
        batch_reads:   Number of reads to batch for GPU inference (default 1000).
        deterministic: If True, use predicted μ only (no stochastic sampling).
                       Default False matches natural PacBio signal variability.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    log.info("Loading reference: %s", ref_path)
    ref_seqs = load_reference(ref_path)

    backend = "regex (forced)" if no_fuzznuc else "fuzznuc (primary, regex fallback)"
    log.info("Pre-scanning reference for methylation sites (%s)...", backend)
    meth_map = build_reference_meth_map(
        ref_seqs, motif_string, revcomp=revcomp, no_fuzznuc=no_fuzznuc
    )
    # Separate forward/reverse strand maps — rev_meth_map fills the
    # complementary-strand block of meth_full at active-site neighbours,
    # matching how `kinsim extract` builds training data.
    from .utils.config import get_extraction_params
    from .utils.motifs import build_reference_meth_map_per_strand

    fwd_meth_map, rev_meth_map = build_reference_meth_map_per_strand(
        ref_seqs,
        motif_string,
    )
    rev_meth_offsets = tuple(int(o) for o in get_extraction_params().rev_meth_offsets)
    # Per-position fraction (target-genome occupancy) — pairs with p_efficiency
    # so the per-site Bernoulli rate at generate is target_frac × p_efficiency.
    frac_map = build_reference_frac_map(ref_seqs, motif_string, revcomp=revcomp)
    from .utils.config import load_kinsim_config
    use_motif_fraction = bool(
        (load_kinsim_config().get("generation") or {}).get("use_motif_fraction", False)
    )
    log.info("Per-site motif fraction at generate: %s (false => p_fire alone)",
             use_motif_fraction)

    # Keep regex motifs for the fallback path (unmapped reads)
    fallback_motifs = parse_motifs(motif_string, revcomp=revcomp)

    log.info("Loading checkpoint: %s", checkpoint_path)
    model = _load_model(checkpoint_path, device)
    p_eff_lookup = _load_p_efficiency(checkpoint_path)
    sig_offsets = _build_sig_offsets_by_meth_id()
    if p_eff_lookup:
        log.info(
            "Statistical firing enabled: %d (meth, offset) buckets with p_fire ∈ [%.2f, %.2f]",
            len(p_eff_lookup),
            min(p_eff_lookup.values()),
            max(p_eff_lookup.values()),
        )
    else:
        log.info("No p_fire in checkpoint — every motif site fires deterministically.")
    mode_label = "deterministic (mean)" if deterministic else "stochastic (sample)"
    log.info("Inference mode: %s", mode_label)

    log.info("Parsing MAF: %s", maf_path)
    maf_mapping = parse_maf(maf_path)

    log.info("Generating signals for reads from %s...", fastq_path)
    n_reads = 0
    n_mapped = 0
    n_unmapped = 0

    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "unknown"},
            "RG": [{"ID": "00000001", "PL": "PACBIO", "DS": "READTYPE=CCS"}],
        }
    )

    open_func = gzip.open if fastq_path.endswith(".gz") else open

    with (
        pysam.AlignmentFile(output_bam, "wb", header=header, threads=4) as bam_out,
        open_func(fastq_path, "rt") as fq,
    ):
        batch = []

        while True:
            hdr_line = fq.readline()
            if not hdr_line:
                break
            seq_line = fq.readline()
            fq.readline()  # '+'
            qual_line = fq.readline()

            read_name = hdr_line.strip()[1:].split()[0]
            seq = seq_line.strip()
            qual_str = qual_line.strip()
            read_len = len(seq)
            n_reads += 1

            batch.append({"name": read_name, "seq": seq, "qual": qual_str, "len": read_len})

            if len(batch) >= batch_reads:
                n_m, n_u = _process_batch(
                    batch,
                    ref_seqs,
                    maf_mapping,
                    meth_map,
                    frac_map,
                    fallback_motifs,
                    p_eff_lookup,
                    sig_offsets,
                    model,
                    device,
                    deterministic,
                    circular,
                    bam_out,
                    header,
                    rev_meth_map=rev_meth_map,
                    fwd_meth_map=fwd_meth_map,
                    rev_meth_offsets=rev_meth_offsets,
                    use_motif_fraction=use_motif_fraction,
                )
                n_mapped += n_m
                n_unmapped += n_u
                batch = []

        if batch:
            n_m, n_u = _process_batch(
                batch,
                ref_seqs,
                maf_mapping,
                meth_map,
                frac_map,
                fallback_motifs,
                p_eff_lookup,
                sig_offsets,
                model,
                device,
                deterministic,
                circular,
                bam_out,
                header,
                rev_meth_map=rev_meth_map,
                fwd_meth_map=fwd_meth_map,
                rev_meth_offsets=rev_meth_offsets,
                use_motif_fraction=use_motif_fraction,
            )
            n_mapped += n_m
            n_unmapped += n_u

    log.info(
        "Done. %d reads processed (%d with ref context, %d without).", n_reads, n_mapped, n_unmapped
    )
    log.info("Output: %s", output_bam)


def _process_batch(
    batch,
    ref_seqs,
    maf_mapping,
    meth_map,
    frac_map,
    fallback_motifs,
    p_eff_lookup,
    sig_offsets,
    model,
    device,
    deterministic,
    circular,
    bam_out,
    header,
    lookup_table=None,
    rev_meth_map=None,
    fwd_meth_map=None,
    rev_meth_offsets=(),
    use_motif_fraction: bool = False,
):
    """Process a batch of reads with batched MLP inference.

    Builds a flat list of (kmer_id, meth_id, fraction) triples for all
    positions across all reads in the batch, runs a single forward pass,
    then writes each read to the BAM with its slice of the generated signals.

    Per-row firing: for every non-zero entry in the meth context window,
    if it sits at a signature offset of its meth type, roll
    Bernoulli(p_fire[T,k]) and zero the entry on no-fire. Handles canonical
    centres and downstream footprints in one sweep — no phantom +5/+2/+6.

    Returns:
        Tuple (n_mapped, n_unmapped) — read counts for the batch.
    """
    # Accumulators are now lists of per-read numpy arrays; concatenated to a
    # single flat tensor before model inference. The previous per-position
    # Python list-append pattern was the dominant cost (~95 % of wall time)
    # — see _process_read_unmapped_vec for the unmapped path's vectorisation.
    # Per-pass meth context accumulators — fi-pass and ri-pass see
    # DIFFERENT templates (extract.py routes pol_padded/opp_padded by
    # read.is_reverse). For + strand reads (the implicit assumption in
    # the mapped path — strand routing for − strand reads is the next
    # bug to fix): fi reads the − strand template, ri reads the + strand
    # template. So:
    #     mc_fi    ← rev_meth_map   |   mc_ri    ← fwd_meth_map
    #     rev_fi   ← fwd_meth_map   |   rev_ri   ← rev_meth_map
    # The previous one-combined-map behaviour summed both strands into
    # mc at every position — for palindromic motifs (e.g. m6A on both
    # strands of GATC) that fed the model TWO m6A entries at neighbouring
    # positions when training only ever showed ONE.
    all_kmer_ids: list[np.ndarray] = []
    all_rc_kmer_ids: list[np.ndarray] = []
    all_meth_id_fi: list[np.ndarray] = []
    all_meth_id_ri: list[np.ndarray] = []
    all_frac_fi: list[np.ndarray] = []
    all_frac_ri: list[np.ndarray] = []
    all_meth_ctx_fi: list[np.ndarray] = []
    all_meth_ctx_ri: list[np.ndarray] = []
    all_rev_ctx_fi: list[np.ndarray] = []
    all_rev_ctx_ri: list[np.ndarray] = []
    is_n_context: list[np.ndarray] = []
    read_offsets = [0]
    n_positions = 0
    _N_REV = len(rev_meth_offsets)
    _REV_OFFSETS = np.asarray(rev_meth_offsets, dtype=np.int64)
    _ZERO_REV = np.zeros(_N_REV, dtype=np.int64)
    _DO_REV = rev_meth_map is not None and _N_REV > 0
    _STRAND_AWARE = rev_meth_map is not None and fwd_meth_map is not None and _N_REV > 0
    _K = K
    # Window geometry comes from the trained model's config. ``_load_model``
    # already refuses non-K=11 checkpoints (hot path is K=11-specific), but
    # reading from model_config keeps the indices self-consistent.
    _cfg = model.get_config()
    _CTX_LEN = int(_cfg.get("kmer_size", 11))
    _PRED_IDX = int(_cfg.get("active_site_index", 7))
    _LEFT = _PRED_IDX
    _RIGHT = _CTX_LEN - _PRED_IDX - 1
    _ZERO_CTX = np.zeros(_CTX_LEN, dtype=np.int64)  # placeholder for N/unmapped

    n_mapped = n_unmapped = 0

    for read_data in batch:
        read_name = read_data["name"]
        seq = read_data["seq"]
        read_len = read_data["len"]

        maf_info = maf_mapping.get(read_name)

        if maf_info and maf_info[0] in ref_seqs:
            # ---- Mapped path: use reference context for edge bases ----
            ref_name, ref_start, _ref_strand, _ref_src_size = maf_info
            ref_seq = ref_seqs[ref_name]
            ref_len = len(ref_seq)
            ref_meth = meth_map[ref_name]  # combined, used as legacy/non-strand-aware fallback
            ref_fwd_meth = fwd_meth_map[ref_name] if _STRAND_AWARE else None
            ref_rev_meth = rev_meth_map[ref_name] if _DO_REV else None
            # p_fire from refine already absorbs corpus occupancy + CCS-noise
            # non-firing into one rate. Re-applying the motif fraction on top
            # would double-count what p_fire learned. Default: fraction=1.0,
            # p_fire alone drives per-read firing. Set use_motif_fraction=True
            # to recover the legacy decomposition behaviour.
            ref_frac = frac_map.get(ref_name) if (use_motif_fraction and frac_map) else None

            ext_context = get_extended_context(ref_seq, ref_start, read_len, circular)
            current_kmer = 0

            # Per-read accumulators. fi-pass and ri-pass keep separate meth
            # contexts because each polymerase pass reads a different template.
            r_kmer: list = []
            r_rc_kmer: list = []
            r_meth_id_fi: list = []
            r_meth_id_ri: list = []
            r_frac_fi: list = []
            r_frac_ri: list = []
            r_mc_fi: list = []
            r_mc_ri: list = []
            r_rev_fi: list = []
            r_rev_ri: list = []
            r_is_n: list = []

            for i in range(len(ext_context)):
                base_val = BASE_MAP.get(ext_context[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                if i >= K - 1:
                    read_pos = i - (K - 1)
                    if 0 <= read_pos < read_len:
                        context_window = ext_context[i - (K - 1) : i + 1]
                        has_n = "N" in context_window
                        r_is_n.append(has_n)

                        if has_n:
                            r_kmer.append(0)
                            r_rc_kmer.append(0)
                            r_meth_id_fi.append(0)
                            r_meth_id_ri.append(0)
                            r_frac_fi.append(0.0)
                            r_frac_ri.append(0.0)
                            r_mc_fi.append(_ZERO_CTX)
                            r_mc_ri.append(_ZERO_CTX)
                            if _DO_REV:
                                r_rev_fi.append(_ZERO_REV)
                                r_rev_ri.append(_ZERO_REV)
                        else:
                            ref_pos = ref_start + read_pos
                            # Strand-aware path: fi sees rev-strand template,
                            # ri sees fwd-strand template (matches extract.py's
                            # pol_padded/opp_padded for + strand reads).
                            # Fallback: combined map for both passes when the
                            # caller doesn't supply per-strand maps.
                            mc_fi = np.zeros(_CTX_LEN, dtype=np.int64)
                            mc_ri = np.zeros(_CTX_LEN, dtype=np.int64)
                            fi_src = ref_rev_meth if _STRAND_AWARE else ref_meth
                            ri_src = ref_fwd_meth if _STRAND_AWARE else ref_meth
                            for k_pos in range(_CTX_LEN):
                                rp_k = ref_pos + k_pos - _LEFT
                                if circular:
                                    idx = rp_k % ref_len
                                    mc_fi[k_pos] = int(fi_src[idx])
                                    mc_ri[k_pos] = int(ri_src[idx])
                                elif 0 <= rp_k < ref_len:
                                    mc_fi[k_pos] = int(fi_src[rp_k])
                                    mc_ri[k_pos] = int(ri_src[rp_k])
                            # Apply per-site Bernoulli firing on each pass
                            # independently — each polymerase pass is a
                            # separate event in the BAM record.
                            _apply_p_fire_to_mc(
                                mc_fi,
                                p_eff_lookup,
                                sig_offsets,
                                _PRED_IDX,
                                ref_pos,
                                ref_frac,
                            )
                            _apply_p_fire_to_mc(
                                mc_ri,
                                p_eff_lookup,
                                sig_offsets,
                                _PRED_IDX,
                                ref_pos,
                                ref_frac,
                            )
                            mid_fi = int(mc_fi[_PRED_IDX])
                            mid_ri = int(mc_ri[_PRED_IDX])
                            r_kmer.append(current_kmer)
                            r_rc_kmer.append(_rc_kmer(current_kmer))
                            r_meth_id_fi.append(mid_fi)
                            r_meth_id_ri.append(mid_ri)
                            r_frac_fi.append(1.0 if mid_fi else 0.0)
                            r_frac_ri.append(1.0 if mid_ri else 0.0)
                            r_mc_fi.append(mc_fi)
                            r_mc_ri.append(mc_ri)
                            if _DO_REV:
                                # rev_meth = synthesized-strand neighbours
                                # (opposite of the polymerase's template).
                                # For + strand reads: fi template = rev, so
                                # rev_meth for fi = fwd; ri's template = fwd,
                                # so rev_meth for ri = rev.
                                rev_fi = np.zeros(_N_REV, dtype=np.int64)
                                rev_ri = np.zeros(_N_REV, dtype=np.int64)
                                fi_rev_src = ref_fwd_meth if _STRAND_AWARE else ref_rev_meth
                                ri_rev_src = ref_rev_meth if _STRAND_AWARE else ref_rev_meth
                                for j, off in enumerate(_REV_OFFSETS):
                                    tgt = ref_pos + int(off)
                                    if circular:
                                        idx = tgt % ref_len
                                        rev_fi[j] = int(fi_rev_src[idx])
                                        rev_ri[j] = int(ri_rev_src[idx])
                                    elif 0 <= tgt < ref_len:
                                        rev_fi[j] = int(fi_rev_src[tgt])
                                        rev_ri[j] = int(ri_rev_src[tgt])
                                r_rev_fi.append(rev_fi)
                                r_rev_ri.append(rev_ri)

            if r_kmer:
                all_kmer_ids.append(np.asarray(r_kmer, dtype=np.int64))
                all_rc_kmer_ids.append(np.asarray(r_rc_kmer, dtype=np.int64))
                all_meth_id_fi.append(np.asarray(r_meth_id_fi, dtype=np.int64))
                all_meth_id_ri.append(np.asarray(r_meth_id_ri, dtype=np.int64))
                all_frac_fi.append(np.asarray(r_frac_fi, dtype=np.float32))
                all_frac_ri.append(np.asarray(r_frac_ri, dtype=np.float32))
                all_meth_ctx_fi.append(np.stack(r_mc_fi, axis=0).astype(np.int64))
                all_meth_ctx_ri.append(np.stack(r_mc_ri, axis=0).astype(np.int64))
                if _DO_REV:
                    all_rev_ctx_fi.append(np.stack(r_rev_fi, axis=0).astype(np.int64))
                    all_rev_ctx_ri.append(np.stack(r_rev_ri, axis=0).astype(np.int64))
                is_n_context.append(np.asarray(r_is_n, dtype=bool))
                n_positions += len(r_kmer)

            n_mapped += 1

        else:
            # ---- Unmapped path: per-read scan, no ref context ----
            # Vectorised — see _process_read_unmapped_vec for the per-read
            # numpy implementation that replaces the old per-base Python loop
            # (~50–100× faster on long HiFi reads).
            kmer_ids_r, rc_kmer_ids_r, meth_ids_r, fractions_r, meth_ctxs_r, is_n_r = (
                _process_read_unmapped_vec(
                    seq,
                    read_len,
                    fallback_motifs,
                    p_eff_lookup,
                    sig_offsets,
                    _PRED_IDX,
                    _CTX_LEN,
                )
            )
            # Unmapped path: no reference info → strand-aware context not
            # available. Use the motif-scanned context (combined map) for
            # both passes — same row of data feeds fi and ri.
            all_kmer_ids.append(kmer_ids_r)
            all_rc_kmer_ids.append(rc_kmer_ids_r)
            all_meth_id_fi.append(meth_ids_r)
            all_meth_id_ri.append(meth_ids_r)
            all_frac_fi.append(fractions_r)
            all_frac_ri.append(fractions_r)
            all_meth_ctx_fi.append(meth_ctxs_r)
            all_meth_ctx_ri.append(meth_ctxs_r)
            if _DO_REV:
                # No ref → rev_meth stays zero on the unmapped path.
                zero_rev = np.zeros((kmer_ids_r.size, _N_REV), dtype=np.int64)
                all_rev_ctx_fi.append(zero_rev)
                all_rev_ctx_ri.append(zero_rev)
            is_n_context.append(is_n_r)
            n_positions += kmer_ids_r.size

            n_unmapped += 1

        read_offsets.append(n_positions)

    # Flatten the per-read accumulators to single tensors before inference.
    if all_kmer_ids:
        flat_kmer = np.concatenate(all_kmer_ids)
        flat_rc_kmer = np.concatenate(all_rc_kmer_ids)
        flat_mid_fi = np.concatenate(all_meth_id_fi)
        flat_mid_ri = np.concatenate(all_meth_id_ri)
        flat_frac_fi = np.concatenate(all_frac_fi)
        flat_frac_ri = np.concatenate(all_frac_ri)
        flat_mc_fi = np.concatenate(all_meth_ctx_fi, axis=0)
        flat_mc_ri = np.concatenate(all_meth_ctx_ri, axis=0)
        flat_rev_fi = np.concatenate(all_rev_ctx_fi, axis=0) if _DO_REV else None
        flat_rev_ri = np.concatenate(all_rev_ctx_ri, axis=0) if _DO_REV else None
        flat_is_n = np.concatenate(is_n_context)
    else:
        flat_kmer = flat_rc_kmer = flat_mid_fi = flat_mid_ri = np.empty(0, dtype=np.int64)
        flat_frac_fi = flat_frac_ri = np.empty(0, dtype=np.float32)
        flat_mc_fi = flat_mc_ri = np.empty((0, _CTX_LEN), dtype=np.int64)
        flat_rev_fi = np.empty((0, _N_REV), dtype=np.int64) if _DO_REV else None
        flat_rev_ri = np.empty((0, _N_REV), dtype=np.int64) if _DO_REV else None
        flat_is_n = np.empty(0, dtype=bool)

    # Inference — dispatch on whether a lookup table is provided.
    #   LUT mode (no GPU)  : ~1 μs/position via numpy fancy index + log-space sampling
    #   Model mode (GPU)   : ~1 ms/position amortised via batched model.forward
    # Both produce uint8-equivalent IPD/PW in physical units.
    if flat_kmer.size > 0:
        if lookup_table is not None:
            all_signals = generate_signals_lookup(
                lookup_table,
                flat_kmer,
                flat_mc_fi,
                deterministic,
            )
            all_rc_signals = generate_signals_lookup(
                lookup_table,
                flat_rc_kmer,
                flat_mc_ri,
                deterministic,
            )
        else:
            # fi pass: rev-strand template context.
            all_signals = generate_signals_batch(
                model,
                flat_kmer,
                flat_mid_fi,
                flat_frac_fi,
                flat_mc_fi,
                device,
                deterministic,
                rev_meth_contexts=flat_rev_fi,
            )
            # ri pass: fwd-strand template context.
            all_rc_signals = generate_signals_batch(
                model,
                flat_rc_kmer,
                flat_mid_ri,
                flat_frac_ri,
                flat_mc_ri,
                device,
                deterministic,
                rev_meth_contexts=flat_rev_ri,
            )
    else:
        all_signals = np.zeros((0, 2), dtype=np.float32)
        all_rc_signals = np.zeros((0, 2), dtype=np.float32)

    # Split signals back to individual reads and write BAM records
    for idx, read_data in enumerate(batch):
        start = read_offsets[idx]
        end = read_offsets[idx + 1]
        signals = all_signals[start:end]
        rc_signals = all_rc_signals[start:end]
        is_n = flat_is_n[start:end]

        ipd_vals = np.clip(signals[:, 0], 0, 255).astype(np.uint8)
        pw_vals = np.clip(signals[:, 1], 0, 255).astype(np.uint8)
        ri_vals = np.clip(rc_signals[:, 0], 0, 255).astype(np.uint8)
        rp_vals = np.clip(rc_signals[:, 1], 0, 255).astype(np.uint8)

        # N-context positions: replace with a safe default of 1 (not 0, which
        # could be mis-interpreted as a missing tag by downstream tools).
        # Vectorised — replaces the per-position Python loop.
        if is_n.any():
            ipd_vals[is_n] = 1
            pw_vals[is_n] = 1
            ri_vals[is_n] = 1
            rp_vals[is_n] = 1

        seg = pysam.AlignedSegment(header)
        seg.query_name = read_data["name"]
        seg.flag = 4  # unmapped
        seg.query_sequence = read_data["seq"]
        seg.query_qualities = pysam.qualitystring_to_array(read_data["qual"])
        rg_id = header.to_dict().get("RG", [{}])[0].get("ID", "00000001")
        seg.set_tag("RG", rg_id)
        # Use .tobytes() instead of .tolist() — saves the per-read allocation of
        # 4 × L Python int objects. For a 10 kb read × 1000 reads/batch this
        # removes ~40 s of pure Python overhead per batch.
        seg.set_tag("fi", array.array("B", ipd_vals.tobytes()))
        seg.set_tag("fp", array.array("B", pw_vals.tobytes()))
        seg.set_tag("ri", array.array("B", ri_vals.tobytes()))
        seg.set_tag("rp", array.array("B", rp_vals.tobytes()))
        bam_out.write(seg)

    return n_mapped, n_unmapped


# ---------------------------------------------------------------------------
# Directory mode
# ---------------------------------------------------------------------------


def generate_directory(
    pbsim3_dir: str,
    checkpoint_path: str,
    motif_source: str,
    output_dir: str,
    circular: bool = True,
    revcomp: bool = True,
    device: str = "cuda",
    batch_reads: int = 1000,
    no_fuzznuc: bool = False,
    deterministic: bool = False,
    min_fraction: float = 0.40,
    min_detected: int = 20,
    meth_types: set[str] | None = None,
) -> None:
    """Generate signals for all species found under pbsim3_dir.

    Supports the same two directory layouts as dictionary inject (auto-detected):
      - Species subdirectories: pbsim3_dir/Ecoli/, pbsim3_dir/Salmonella/, ...
      - Flat layout: all files directly in pbsim3_dir, matched by basename.

    Output BAMs are written to output_dir as <species_name>_mlp.bam.
    """
    genomes = find_pbsim3_files(pbsim3_dir)
    if not genomes:
        log.error("No genome sets found in %s", pbsim3_dir)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    log.info("Found %d species in %s", len(genomes), pbsim3_dir)

    for fq_path, maf_path, ref_path, species in genomes:
        motif_string = resolve_motifs_for_species(motif_source, species, min_fraction, min_detected)
        if not motif_string:
            log.error("No motifs found for species '%s'.", species)
            sys.exit(1)

        motif_string = _apply_meth_types(motif_string, meth_types)

        out_bam = os.path.join(output_dir, species + "_mlp.bam")
        log.info("--- %s ---", species)
        generate_signals(
            fq_path,
            maf_path,
            ref_path,
            checkpoint_path,
            motif_string,
            out_bam,
            circular=circular,
            revcomp=revcomp,
            device=device,
            batch_reads=batch_reads,
            no_fuzznuc=no_fuzznuc,
            deterministic=deterministic,
        )

    log.info("All done. %d BAM(s) written to: %s", len(genomes), output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BAM input mode
# ---------------------------------------------------------------------------


def generate_from_bam(
    input_bam: str,
    ref_path: str,
    checkpoint_path: str,
    motif_string: str,
    output_bam: str,
    circular: bool = True,
    revcomp: bool = True,
    device: str = "cuda",
    batch_reads: int = 1000,
    no_fuzznuc: bool = False,
    deterministic: bool = False,
    region: str | None = None,
    use_lookup: str | None = None,
) -> None:
    """Generate kinetic signals for reads from an existing (aligned) BAM file.

    Reads sequences and alignment coordinates directly from the BAM — no FASTQ
    or MAF file needed.  Alignment info (reference_name, reference_start) is
    used exactly as the MAF would be, giving accurate reference-context padding
    at read edges.  Unaligned reads fall back to read-only context.

    Use this with a BAM that has had fi/fp/ri/rp tags stripped.
    Output BAM has the same reads + sequences as input, with fresh fi/fp/ri/rp.
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device_obj)

    log.info("Loading reference: %s", ref_path)
    ref_seqs = load_reference(ref_path)

    backend = "regex (forced)" if no_fuzznuc else "fuzznuc (primary, regex fallback)"
    log.info("Pre-scanning reference for methylation sites (%s)...", backend)
    meth_map = build_reference_meth_map(
        ref_seqs, motif_string, revcomp=revcomp, no_fuzznuc=no_fuzznuc
    )
    from .utils.config import get_extraction_params
    from .utils.motifs import build_reference_meth_map_per_strand

    fwd_meth_map, rev_meth_map = build_reference_meth_map_per_strand(
        ref_seqs,
        motif_string,
    )
    rev_meth_offsets = tuple(int(o) for o in get_extraction_params().rev_meth_offsets)
    frac_map = build_reference_frac_map(ref_seqs, motif_string, revcomp=revcomp)
    from .utils.config import load_kinsim_config
    use_motif_fraction = bool(
        (load_kinsim_config().get("generation") or {}).get("use_motif_fraction", False)
    )
    log.info("Per-site motif fraction at generate: %s (false => p_fire alone)",
             use_motif_fraction)
    fallback_motifs = parse_motifs(motif_string, revcomp=revcomp)

    # Either load the trained model (default) or a pre-computed lookup table.
    lookup_table = None
    if use_lookup:
        log.info("Loading lookup table: %s", use_lookup)
        lookup_table = _load_lookup_table(use_lookup)
        model = None  # not needed in LUT mode
    else:
        log.info("Loading checkpoint: %s", checkpoint_path)
        model = _load_model(checkpoint_path, device_obj)
    p_eff_lookup = _load_p_efficiency(checkpoint_path)
    sig_offsets = _build_sig_offsets_by_meth_id()
    if p_eff_lookup:
        log.info(
            "Statistical firing enabled: %d (meth, offset) buckets with p_fire ∈ [%.2f, %.2f]",
            len(p_eff_lookup),
            min(p_eff_lookup.values()),
            max(p_eff_lookup.values()),
        )
    else:
        log.info("No p_fire in checkpoint — every motif site fires deterministically.")
    mode_label = "deterministic (mean)" if deterministic else "stochastic (sample)"
    log.info("Inference mode: %s", mode_label)

    n_reads = n_mapped = n_unmapped = 0
    batch: list = []
    batch_maf: dict = {}

    log.info("Reading reads from: %s", input_bam)

    # Build a clean unaligned header (no SQ entries) so pbmm2 treats
    # the output as unaligned and properly converts fi/fp/ri/rp → ip/pw.
    #
    # CRITICAL: copy the PacBio `pb:` tag from the source @HD line if it
    # exists, otherwise add a sane default. Downstream PacBio tools
    # (pbindex, ipdSummary) read this tag at startup and KeyError'd in a
    # previous incident when it was missing — the resulting silent
    # absence of a .pbi file then poisons the whole validation chain.
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as bam_in:
        in_dict = bam_in.header.to_dict()
        hd = {"VN": "1.6", "SO": "unknown"}
        in_hd = in_dict.get("HD", {})
        if "pb" in in_hd:
            hd["pb"] = in_hd["pb"]
        else:
            # SMRT-Tools 25.x default. Matches the version used by ipdSummary
            # in slurm_kinsim/callers/ipdsummary.slurm.
            hd["pb"] = "3.0.7"
        out_dict = {"HD": hd}
        # Single synthetic RG with a pbindex-friendly ID. Replaces the
        # bystrandify-mangled RG IDs (e.g. `f7c20c6c/33--33-744DFBEF`)
        # that pbindex rejects. Format must be ``<8-hex>/<int>--<int>``:
        # pbindex parses the prefix as a movie hash via stoul(), so the
        # 8 chars MUST be valid hex (0-9a-f). All-zero hash is safe and
        # recognisable; non-hex (e.g. "kinsim") triggers stoul FATAL.
        synthetic_rg = {"ID": "00000000/0--0", "PL": "PACBIO", "DS": "READTYPE=CCS"}
        if "RG" in in_dict and in_dict["RG"]:
            first = dict(in_dict["RG"][0])
            first["ID"] = synthetic_rg["ID"]
            first.setdefault("PL", synthetic_rg["PL"])
            first.setdefault("DS", synthetic_rg["DS"])
            out_dict["RG"] = [first]
        else:
            out_dict["RG"] = [synthetic_rg]
        header_out = pysam.AlignmentHeader.from_dict(out_dict)

    # Multi-threaded BGZF I/O — htslib uses these threads for parallel
    # decompression (input) and parallel compression (output). Single biggest
    # I/O win since pysam defaults to 1 thread. Use 4 to match the SLURM
    # --cpus-per-task=4 we typically request for generate.
    #
    # ``region`` (chr:start-end) routes through pysam.fetch() for parallel
    # array-job execution. Requires the input BAM to be indexed (.bai).
    with (
        pysam.AlignmentFile(input_bam, "rb", check_sq=False, threads=4) as bam_in,
        pysam.AlignmentFile(output_bam, "wb", header=header_out, threads=4) as bam_out,
    ):
        if region:
            log.info("Fetching reads from region: %s", region)
            try:
                reads_iter = bam_in.fetch(region=region)
            except (ValueError, OSError) as e:
                log.error(
                    "Cannot fetch region '%s' — input BAM must be indexed "
                    "(samtools index ...). Underlying error: %s",
                    region,
                    e,
                )
                sys.exit(1)
        else:
            reads_iter = bam_in
        for read in reads_iter:
            if read.query_sequence is None:
                continue

            seq = read.query_sequence
            qual = read.query_qualities
            qual_str = pysam.array_to_qualitystring(qual) if qual is not None else "I" * len(seq)

            batch.append(
                {
                    "name": read.query_name,
                    "seq": seq,
                    "qual": qual_str,
                    "len": len(seq),
                }
            )

            # Build maf_mapping entry from BAM alignment — same fields parse_maf returns.
            # Gated by KINSIM_USE_REF_CTX (default off): the mapped-path inner
            # loop is unvectorised and ~50× slower than the unmapped path. For
            # whole-genome motif validation the edge-accuracy gain from
            # reference-context padding is statistically negligible, so we
            # default to the fast unmapped path. Set KINSIM_USE_REF_CTX=1 to
            # re-enable the (slow) mapped path.
            if (
                os.environ.get("KINSIM_USE_REF_CTX") == "1"
                and not read.is_unmapped
                and read.reference_name is not None
                and read.reference_name in ref_seqs
            ):
                ref_len = len(ref_seqs[read.reference_name])
                batch_maf[read.query_name] = (
                    read.reference_name,
                    read.reference_start,
                    "+",
                    ref_len,
                )

            n_reads += 1

            if len(batch) >= batch_reads:
                n_m, n_u = _process_batch(
                    batch,
                    ref_seqs,
                    batch_maf,
                    meth_map,
                    frac_map,
                    fallback_motifs,
                    p_eff_lookup,
                    sig_offsets,
                    model,
                    device_obj,
                    deterministic,
                    circular,
                    bam_out,
                    header_out,
                    lookup_table=lookup_table,
                    rev_meth_map=rev_meth_map,
                    fwd_meth_map=fwd_meth_map,
                    rev_meth_offsets=rev_meth_offsets,
                    use_motif_fraction=use_motif_fraction,
                )
                n_mapped += n_m
                n_unmapped += n_u
                batch = []
                batch_maf = {}
                if n_reads % 1000 == 0:
                    log.info("Progress: %d reads processed...", n_reads)

        if batch:
            n_m, n_u = _process_batch(
                batch,
                ref_seqs,
                batch_maf,
                meth_map,
                frac_map,
                fallback_motifs,
                p_eff_lookup,
                sig_offsets,
                model,
                device_obj,
                deterministic,
                circular,
                bam_out,
                header_out,
                lookup_table=lookup_table,
                rev_meth_map=rev_meth_map,
                fwd_meth_map=fwd_meth_map,
                rev_meth_offsets=rev_meth_offsets,
                use_motif_fraction=use_motif_fraction,
            )
            n_mapped += n_m
            n_unmapped += n_u

    log.info(
        "Done. %d reads processed (%d with ref context, %d without).", n_reads, n_mapped, n_unmapped
    )
    log.info("Output: %s", output_bam)


def _main_from_bam(argv):
    """CLI for BAM input mode: stripped real BAM → BAM with synthetic fi/fp/ri/rp."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim generate",
        description=(
            "Generate synthetic kinetic signals for reads in an existing BAM.\n\n"
            "Input BAM must have fi/fp/ri/rp tags removed (use strip-kinetics first).\n"
            "Alignment coordinates are read from the BAM — no MAF file needed.\n\n"
            "Usage:\n"
            "  kinsim generate <input.bam> <ref.fna> <checkpoint.pt> <motifs> <output.bam>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_bam", help="Aligned BAM with fi/fp/ri/rp stripped")
    parser.add_argument("ref", help="Reference genome FASTA (.fna / .fa / .gz)")
    parser.add_argument("checkpoint", help="Trained checkpoint (.pt)")
    parser.add_argument("motifs", help="Motif string, PacBio motifs.csv, or REBASE file")
    parser.add_argument("output", help="Output BAM with synthetic fi/fp/ri/rp tags")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Treat genome as linear (default: circular for bacteria)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-reads", type=int, default=1000)
    parser.add_argument("--no-revcomp", action="store_true")
    parser.add_argument("--no-fuzznuc", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--min-fraction", type=float, default=0.40)
    parser.add_argument("--min-detected", type=int, default=20)
    parser.add_argument(
        "--meth-types",
        default=None,
        help="Comma-separated methylation types to simulate "
        "(e.g. 'm6A,m4C'). Filters the motif source "
        "before generation. Default: use the checkpoint's "
        "training alphabet. 'all' disables the filter.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Process only reads overlapping this region (e.g. 'chr1:1-460000'). "
        "Enables array-job parallelism: launch N tasks, each with a different "
        "region, then samtools merge the outputs. Input BAM must be indexed.",
    )
    parser.add_argument(
        "--use-lookup",
        default=None,
        metavar="PATH_TO_NPZ",
        help="Use a pre-computed lookup table (from `kinsim predict-kmers`) instead "
        "of running the model on the GPU. The LUT distils the trained model "
        "into a (kmer, scenario) → (μ, σ) table — pure-numpy inference, no GPU "
        "needed, ~1000× faster per-position. The checkpoint is still loaded "
        "(only for p_fire metadata); pass the same checkpoint that produced the "
        "LUT to keep firing semantics consistent.",
    )
    args = parser.parse_args(argv)

    motif_string = load_motif_string(
        args.motifs, min_fraction=args.min_fraction, min_detected=args.min_detected
    )
    if not motif_string:
        log.error("No motifs found from the provided source.")
        sys.exit(1)

    meth_types = _resolve_meth_types(
        parse_meth_types_arg(args.meth_types),
        _read_ckpt_meth_types(args.checkpoint),
    )
    motif_string = _apply_meth_types(motif_string, meth_types)

    generate_from_bam(
        input_bam=args.input_bam,
        ref_path=args.ref,
        checkpoint_path=args.checkpoint,
        motif_string=motif_string,
        output_bam=args.output,
        circular=not args.linear,
        revcomp=not args.no_revcomp,
        device=args.device,
        batch_reads=args.batch_reads,
        no_fuzznuc=args.no_fuzznuc,
        deterministic=args.deterministic,
        region=args.region,
        use_lookup=args.use_lookup,
    )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if argv and os.path.isdir(argv[0]):
        _main_directory(argv)
    elif argv and argv[0].endswith(".bam"):
        _main_from_bam(argv)
    else:
        _main_per_genome(argv)


def _main_directory(argv):
    """CLI for directory mode: processes all species in pbsim3_dir."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim mlp generate",
        description=(
            "Generate MLP kinetic signals for all PBSIM3 species in a directory.\n\n"
            "Supports two directory layouts (auto-detected):\n\n"
            "  Species subdirectories (recommended):\n"
            "    pbsim3_dir/\n"
            "      Ecoli/          <- species name = subdir name\n"
            "        reads.fq.gz\n"
            "        reads.maf.gz\n"
            "        Ecoli.fna\n"
            "      Salmonella/\n"
            "        ...\n\n"
            "  Flat layout (files matched by basename):\n"
            "    pbsim3_dir/\n"
            "      Ecoli.fq.gz   Ecoli.maf.gz   Ecoli.fna\n"
            "      Salmonella.fq.gz ...\n\n"
            "Per-genome mode (single genome):\n"
            "  kinsim mlp generate <fq.gz> <maf.gz> <ref.fna> <ckpt.pt> <motifs> <out.bam>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pbsim3_dir", help="Directory containing species subdirs or flat .fq.gz files"
    )
    parser.add_argument("checkpoint", help="Trained MLP checkpoint (.pt)")
    parser.add_argument(
        "motifs",
        help="Motifs: KinSim string (applied to all), PacBio .csv, "
        "REBASE file, or per-species file ('species|motif_string' per line)",
    )
    parser.add_argument("output_dir", help="Output directory for generated BAM files")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Treat genomes as linear (default: circular for bacteria)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--batch-reads",
        type=int,
        default=1000,
        help="Number of reads to batch for GPU inference (default: 1000)",
    )
    parser.add_argument(
        "--no-revcomp", action="store_true", help="Do not scan reverse complement strand for motifs"
    )
    parser.add_argument(
        "--no-fuzznuc",
        action="store_true",
        help="Force Python regex for reference methylation scanning",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use predicted mean (mu) only - no stochastic sampling. "
        "Produces identical signals for every read at the same "
        "context (useful for ablations). Default: stochastic.",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.40,
        help="Minimum fraction threshold (PacBio CSV only, default: 0.40)",
    )
    parser.add_argument(
        "--min-detected",
        type=int,
        default=20,
        help="Minimum nDetected threshold (PacBio CSV only, default: 20)",
    )
    parser.add_argument(
        "--meth-types",
        default=None,
        help="Comma-separated methylation types to simulate "
        "(e.g. 'm6A,m4C'). Filters each species' motif "
        "string before generation. Default: use the "
        "checkpoint's training alphabet. 'all' disables "
        "the filter.",
    )
    args = parser.parse_args(argv)

    cli_meth_types = parse_meth_types_arg(args.meth_types)
    ckpt_meth_types = _read_ckpt_meth_types(args.checkpoint)
    meth_types = _resolve_meth_types(cli_meth_types, ckpt_meth_types)

    generate_directory(
        pbsim3_dir=args.pbsim3_dir,
        checkpoint_path=args.checkpoint,
        motif_source=args.motifs,
        output_dir=args.output_dir,
        circular=not args.linear,
        revcomp=not args.no_revcomp,
        device=args.device,
        batch_reads=args.batch_reads,
        no_fuzznuc=args.no_fuzznuc,
        deterministic=args.deterministic,
        min_fraction=args.min_fraction,
        min_detected=args.min_detected,
        meth_types=meth_types,
    )


def _main_per_genome(argv):
    """CLI for per-genome mode: processes a single .fq.gz file."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim mlp generate",
        description=(
            "Generate kinetic signals for PBSIM3 reads using a trained MLP predictor.\n\n"
            "Uses the .maf alignment to resolve reference context for edge bases.\n"
            "The reference is pre-scanned once for methylation sites; per-read\n"
            "lookups are O(1). Outputs an unaligned BAM with fi (IPD) and fp (PW) tags.\n\n"
            "Data preparation:\n"
            "  kinsim extract --manifest manifest.csv --task N --output-dir shards/\n"
            "  kinsim refine  shards/   refined/\n"
            "  kinsim train   refined/  checkpoints/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fastq", help="PBSIM3 simulated reads (.fq or .fq.gz)")
    parser.add_argument("maf", help="PBSIM3 alignment file (.maf or .maf.gz)")
    parser.add_argument("ref", help="Reference genome FASTA (.fna, .fa, or .gz)")
    parser.add_argument("checkpoint", help="Trained MLP checkpoint (.pt)")
    parser.add_argument(
        "motifs",
        help="Motif source: KinSim string ('m6A,GATC,1'), "
        "PacBio motifs.csv, or REBASE file (auto-detected)",
    )
    parser.add_argument("output", help="Output unaligned BAM file")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Treat genome as linear (default: circular for bacteria)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--batch-reads",
        type=int,
        default=1000,
        help="Number of reads to batch for GPU inference (default: 1000)",
    )
    parser.add_argument(
        "--no-revcomp",
        action="store_true",
        help="Do not scan reverse complement strand for motifs "
        "(use when motif source already includes both orientations)",
    )
    parser.add_argument(
        "--no-fuzznuc",
        action="store_true",
        help="Force Python regex for reference methylation scanning. "
        "By default, EMBOSS fuzznuc is tried first as the primary "
        "backend and falls back to regex automatically if fuzznuc "
        "is not installed.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use predicted mean (mu) only - no stochastic sampling. "
        "Produces identical signals for every read at the same "
        "context (useful for ablations). Default: stochastic.",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.40,
        help="Minimum fraction threshold (PacBio CSV only, default: 0.40)",
    )
    parser.add_argument(
        "--min-detected",
        type=int,
        default=20,
        help="Minimum nDetected threshold (PacBio CSV only, default: 20)",
    )
    parser.add_argument(
        "--meth-types",
        default=None,
        help="Comma-separated methylation types to simulate "
        "(e.g. 'm6A,m4C'). Filters the motif source "
        "before generation. Default: use the checkpoint's "
        "training alphabet. 'all' disables the filter.",
    )
    args = parser.parse_args(argv)

    motif_string = load_motif_string(
        args.motifs, min_fraction=args.min_fraction, min_detected=args.min_detected
    )
    if not motif_string:
        log.error("No motifs found from the provided source.")
        sys.exit(1)

    meth_types = _resolve_meth_types(
        parse_meth_types_arg(args.meth_types),
        _read_ckpt_meth_types(args.checkpoint),
    )
    motif_string = _apply_meth_types(motif_string, meth_types)

    generate_signals(
        fastq_path=args.fastq,
        maf_path=args.maf,
        ref_path=args.ref,
        checkpoint_path=args.checkpoint,
        motif_string=motif_string,
        output_bam=args.output,
        circular=not args.linear,
        revcomp=not args.no_revcomp,
        device=args.device,
        batch_reads=args.batch_reads,
        no_fuzznuc=args.no_fuzznuc,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
