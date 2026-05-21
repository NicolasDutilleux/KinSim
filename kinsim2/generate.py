"""Generate bilateral kinetic signals for a raw HiFi aligned BAM.

Bilateral v2: the trained ConvPredictor predicts (ipd_fwd, pw_fwd,
ipd_rev, pw_rev) jointly in a single forward pass per kmer. Per-read
routing back to BAM tags follows ``extract.py`` (the inverse of how
training rows were sourced):

  - ``read.is_reverse == False``:  fi <- ipd_rev, fp <- pw_rev,
                                   ri <- ipd_fwd, rp <- pw_fwd
  - ``read.is_reverse == True``:   fi <- ipd_fwd, fp <- pw_fwd,
                                   ri <- ipd_rev, rp <- pw_rev

Input is a raw HiFi BAM (one record per ZMW with fi/fp/ri/rp tags, or
none — generate strips and injects). When the BAM is aligned, each
read's mapped region is scanned against per-strand methylation maps;
unmapped reads get baseline kinetics (no meth context).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pysam
import torch

from .models.predictor import create_from_config, load_state_dict_from_ckpt
from .utils._defaults import BAM_TAG_MAX
from .utils.config import get_extraction_params, setup_logging
from .utils.encoding import BASE_MAP, get_meth_ids
from .utils.motifs import build_reference_meth_map_per_strand, load_motif_string

log = logging.getLogger(__name__)


SIGNAL_BATCH_CHUNK: int = 50_000


def _load_model(checkpoint_path: str, device: torch.device):
    """Load ConvPredictor from a checkpoint + sibling model_config.json."""
    ckpt_path = Path(checkpoint_path)
    cfg_path = ckpt_path.parent / "model_config.json"
    if not cfg_path.exists():
        cfg_path = ckpt_path.with_suffix(".json")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found next to {ckpt_path}; "
            f"generate needs it to rebuild the architecture."
        )
    cfg = json.loads(cfg_path.read_text())
    model = create_from_config(cfg)
    state_dict = load_state_dict_from_ckpt(str(ckpt_path))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, cfg


def _encode_kmer_vec(seq_arr: np.ndarray, kmer_size: int, upstream: int) -> np.ndarray:
    """Vectorised kmer encoding at every position of ``seq_arr``.

    ``seq_arr`` is the per-base integer (0..3, with 4 reserved for N) for
    the reference sequence. Returns ``(L,)`` int64 of kmer IDs where the
    active site sits at index ``i + upstream``. Positions where the
    window would overrun the ends, or where any base is N, get kmer_id=0
    (the caller masks them via ``is_n_context``).
    """
    L = len(seq_arr)
    out = np.zeros(L, dtype=np.int64)
    is_n = np.zeros(L, dtype=bool)
    downstream = kmer_size - upstream - 1
    if L < kmer_size:
        is_n[:] = True
        return out, is_n
    for i in range(L):
        ref_start = i - upstream
        ref_end = i + downstream + 1
        if ref_start < 0 or ref_end > L:
            is_n[i] = True
            continue
        window = seq_arr[ref_start:ref_end]
        if (window == 4).any():
            is_n[i] = True
            continue
        kid = 0
        for b in window:
            kid = (kid << 2) | int(b)
        out[i] = kid
    return out, is_n


def _encode_kmer_vec_fast(seq_arr: np.ndarray, kmer_size: int, upstream: int):
    """Vectorised kmer ID per position using rolling shifts on int64 lanes."""
    L = len(seq_arr)
    out = np.zeros(L, dtype=np.int64)
    is_n = np.zeros(L, dtype=bool)
    downstream = kmer_size - upstream - 1
    if L < kmer_size:
        is_n[:] = True
        return out, is_n
    base_int = seq_arr.astype(np.int64)
    base_int_safe = np.where(base_int == 4, 0, base_int)
    n_centers = L - kmer_size + 1
    kids = np.zeros(n_centers, dtype=np.int64)
    for k in range(kmer_size):
        kids = (kids << 2) | base_int_safe[k : k + n_centers]
    centers = np.arange(n_centers) + upstream
    n_in_window = np.zeros(n_centers, dtype=bool)
    for k in range(kmer_size):
        n_in_window |= (base_int[k : k + n_centers] == 4)
    out[centers] = kids
    is_n[centers] = n_in_window
    is_n[: upstream] = True
    is_n[L - downstream :] = True
    return out, is_n


def _build_meth_one_hot(
    meth_ids_arr: np.ndarray,
    kmer_size: int,
    upstream: int,
    num_meth_types: int,
) -> np.ndarray:
    """Materialise (L, K, M) one-hot meth context for every reference position.

    ``meth_ids_arr[i]`` is the meth_id at ref position ``i``. The per-row
    window centred on ``i`` (active site) reads positions
    ``[i - upstream, i + downstream]``.
    """
    L = len(meth_ids_arr)
    downstream = kmer_size - upstream - 1
    out = np.zeros((L, kmer_size, num_meth_types), dtype=np.float32)
    padded = np.zeros(L + kmer_size, dtype=np.int64)
    padded[upstream : upstream + L] = meth_ids_arr.astype(np.int64)
    for k in range(kmer_size):
        col = padded[k : k + L]
        valid = (col > 0) & (col < num_meth_types)
        rows = np.where(valid)[0]
        if rows.size:
            out[rows, k, col[rows]] = 1.0
    return out


def _seq_to_int(seq: str) -> np.ndarray:
    """Map an ASCII sequence to 0..3 (A/C/G/T) or 4 (anything else)."""
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    out = np.full_like(arr, 4, dtype=np.int64)
    for ch, code in BASE_MAP.items():
        out[arr == ord(ch)] = code
    return out


def _revcomp_int(seq_int: np.ndarray) -> np.ndarray:
    """Reverse-complement an integer-encoded sequence (A<->T, C<->G, N pass-through)."""
    comp = seq_int.copy()
    mask = comp < 4
    comp[mask] = comp[mask] ^ 3
    return comp[::-1].copy()


def _route_strands(
    is_reverse: bool,
    ipd_fwd: np.ndarray,
    pw_fwd: np.ndarray,
    ipd_rev: np.ndarray,
    pw_rev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map (+, -) strand kinetics back to BAM (fi, fp, ri, rp) tags.

    Inverse of ``extract.py`` routing. The mapping depends on
    ``read.is_reverse`` because pbmm2 re-orients the per-base arrays to
    match the BAM SEQ; the trained model emits + and - strand kinetics
    in reference coordinates.
    """
    if is_reverse:
        return ipd_fwd, pw_fwd, ipd_rev, pw_rev
    return ipd_rev, pw_rev, ipd_fwd, pw_fwd


def _predict_for_positions(
    model,
    kmer_ids: np.ndarray,
    mc_fwd: np.ndarray,
    mc_rev: np.ndarray,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    """Run the bilateral model in chunks; return raw uint8 (N, 4)."""
    N = kmer_ids.shape[0]
    out = np.zeros((N, 4), dtype=np.uint8)
    if N == 0:
        return out
    with torch.no_grad():
        for start in range(0, N, SIGNAL_BATCH_CHUNK):
            stop = min(start + SIGNAL_BATCH_CHUNK, N)
            k_t = torch.from_numpy(kmer_ids[start:stop]).long().to(device)
            mf_t = torch.from_numpy(mc_fwd[start:stop]).float().to(device)
            mr_t = torch.from_numpy(mc_rev[start:stop]).float().to(device)
            if deterministic:
                preds = model.predict_mean(k_t, mf_t, mr_t)
            else:
                preds = model.sample(k_t, mf_t, mr_t)
            out[start:stop] = preds.clamp(0, BAM_TAG_MAX).round().byte().cpu().numpy()
    return out


def _resolve_p_fire_lookup(cfg: dict) -> dict[tuple[int, int], float]:
    """Pull ``{(meth_id, signal_offset_k): p_fire}`` from the model_config.

    The checkpoint's ``p_fire`` dict uses string keys
    ``"<meth_name>@<+offset>"`` (e.g. ``"m6A@+5"``) written by refine.
    Convert to ``(int meth_id, int k)`` via the checkpoint's frozen
    meth_id_map. Returns ``{}`` if the checkpoint predates refine
    (generate will then use the global default rate).
    """
    raw = cfg.get("p_fire") or {}
    meth_id_map = cfg.get("meth_id_map") or get_meth_ids()
    out: dict[tuple[int, int], float] = {}
    for label, p in raw.items():
        if "@" not in label:
            continue
        t_name, off_str = label.split("@", 1)
        m_id = meth_id_map.get(t_name)
        if m_id is None:
            continue
        try:
            k = int(off_str)
        except ValueError:
            continue
        out[(int(m_id), k)] = float(p)
    return out


def _build_sig_offsets_by_meth_id() -> dict[int, set[int]]:
    """``{meth_id: {signature_offsets}}`` from kinsim_config.yaml.

    Only signature offsets are subject to the firing Bernoulli — at
    non-signature offsets the model already learned to emit baseline-
    like kinetics, so they pass through.
    """
    from .utils.config import get_signature_offsets, load_kinsim_config
    cfg = load_kinsim_config()
    ids = get_meth_ids()
    out: dict[int, set[int]] = {}
    for name in cfg.get("kinetic_signatures") or {}:
        m_id = ids.get(name)
        if m_id is not None:
            out[m_id] = set(get_signature_offsets(name))
    return out


def _build_pfire_rate_table(
    p_fire_lookup: dict[tuple[int, int], float],
    sig_offsets_by_meth: dict[int, set[int]],
    kmer_size: int,
    active_site_index: int,
    num_meth_types: int,
    default_rate: float,
) -> np.ndarray:
    """``(K, M)`` Bernoulli rate per (k_pos, meth_id) for the firing roll.

    Non-signature positions get rate=1.0 (always fire — model emits its
    learned signature there unmodified). Signature positions
    ``(meth_id, signal_offset_k)`` get either:
      - the per-bucket survival rate from ``p_fire_lookup`` (refine meta),
      - or ``default_rate`` if that bucket isn't in the lookup.

    Applied in-place to the one-hot meth context: roll U ~ U(0,1) per
    entry, zero out the meth code where ``U >= rate``. The signal at
    that row drops to the baseline kinetics emitted by the model.
    """
    rate_table = np.ones((kmer_size, num_meth_types), dtype=np.float32)
    for m_id in range(1, num_meth_types):
        offsets = sig_offsets_by_meth.get(m_id, set())
        for k_off in offsets:
            k_pos = active_site_index - k_off
            if 0 <= k_pos < kmer_size:
                rate_table[k_pos, m_id] = float(
                    p_fire_lookup.get((m_id, k_off), default_rate)
                )
    return rate_table


def _apply_p_fire_inplace(
    meth_ctx: np.ndarray, rate_table: np.ndarray, rng: np.random.Generator,
) -> None:
    """Roll Bernoulli per (L, K, M) one-hot entry; zero the non-firing positions.

    Vectorised. Independent rolls per row × position × meth-channel —
    every read sees a fresh draw, so motif occupancy realisations differ
    read-to-read at the same site (matches PacBio biology).
    """
    if meth_ctx.size == 0:
        return
    rolls = rng.random(meth_ctx.shape, dtype=np.float32)
    keep = rolls < rate_table[None, :, :]
    meth_ctx *= keep


def _process_read(
    read: pysam.AlignedSegment,
    model,
    fwd_meth_map: dict,
    rev_meth_map: dict,
    ref_seqs: dict,
    kmer_size: int,
    upstream: int,
    num_meth_types: int,
    device: torch.device,
    deterministic: bool,
    rate_table: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate fi/fp/ri/rp arrays for one aligned read.

    For mapped reads, the active-site position runs along the reference
    region the read covers. For unmapped reads (and N-context positions),
    kinetics fall back to a baseline value of 1. ``rate_table`` drives
    the per-row Bernoulli firing on signature offsets — non-firing
    methylations are zeroed in the meth context so the model emits
    baseline kinetics at that row.
    """
    seq_len = read.query_length or 0
    fi = np.ones(seq_len, dtype=np.uint8)
    fp = np.ones(seq_len, dtype=np.uint8)
    ri = np.ones(seq_len, dtype=np.uint8)
    rp = np.ones(seq_len, dtype=np.uint8)
    if seq_len == 0 or read.is_unmapped or read.reference_name not in ref_seqs:
        return fi, fp, ri, rp

    ref_name = read.reference_name
    ref_seq = ref_seqs[ref_name]
    fwd_meth = fwd_meth_map.get(ref_name)
    rev_meth = rev_meth_map.get(ref_name)
    if fwd_meth is None or rev_meth is None:
        return fi, fp, ri, rp

    ref_start = int(read.reference_start)
    ref_end = int(read.reference_end)
    if ref_end <= ref_start:
        return fi, fp, ri, rp

    # Slice the reference + per-strand meth maps over the alignment span.
    seq_int = _seq_to_int(ref_seq[ref_start:ref_end])
    fwd_window = fwd_meth[ref_start:ref_end]
    rev_window = rev_meth[ref_start:ref_end]

    if read.is_reverse:
        seq_int = _revcomp_int(seq_int)
        fwd_window, rev_window = rev_window[::-1].copy(), fwd_window[::-1].copy()

    kmer_ids, is_n = _encode_kmer_vec_fast(seq_int, kmer_size, upstream)
    mc_fwd = _build_meth_one_hot(fwd_window, kmer_size, upstream, num_meth_types)
    mc_rev = _build_meth_one_hot(rev_window, kmer_size, upstream, num_meth_types)
    # Per-row Bernoulli firing on the one-hot meth tensors. Independent
    # draws on fwd vs rev strand so cross-strand non-coincident firing
    # is preserved.
    _apply_p_fire_inplace(mc_fwd, rate_table, rng)
    _apply_p_fire_inplace(mc_rev, rate_table, rng)

    preds = _predict_for_positions(
        model, kmer_ids, mc_fwd, mc_rev, device, deterministic,
    )
    preds[is_n] = 1  # N context falls back to baseline signal

    # Project predictions into read coordinates via the CIGAR / pairs.
    # Vectorised — fancy index instead of the per-position Python loop.
    fi_arr, fp_arr, ri_arr, rp_arr = _route_strands(
        bool(read.is_reverse),
        preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3],
    )

    pairs = np.array(read.get_aligned_pairs(matches_only=True), dtype=np.int64)
    if pairs.size:
        q_pos = pairs[:, 0]
        r_pos = pairs[:, 1]
        rel = r_pos - ref_start
        valid = (rel >= 0) & (rel < len(fi_arr))
        if valid.any():
            q_pos = q_pos[valid]
            rel = rel[valid]
            ref_idx = (len(fi_arr) - 1 - rel) if read.is_reverse else rel
            fi[q_pos] = fi_arr[ref_idx]
            fp[q_pos] = fp_arr[ref_idx]
            ri[q_pos] = ri_arr[ref_idx]
            rp[q_pos] = rp_arr[ref_idx]

    return fi, fp, ri, rp


def _set_kinetic_tags(read: pysam.AlignedSegment, fi: np.ndarray, fp: np.ndarray,
                     ri: np.ndarray, rp: np.ndarray) -> None:
    """Attach fi/fp/ri/rp tags as PacBio B:C arrays."""
    from array import array as pyarray

    def _b(name: str, arr: np.ndarray) -> None:
        read.set_tag(name, pyarray("B", arr.astype(np.uint8).tolist()), value_type="B")

    _b("fi", fi)
    _b("fp", fp)
    _b("ri", ri)
    _b("rp", rp)


def _load_reference(ref_path: str) -> dict:
    """Load reference FASTA into ``{contig: str}``."""
    seqs: dict = {}
    try:
        with pysam.FastaFile(ref_path) as fh:
            for name in fh.references:
                seqs[name] = fh.fetch(name)
    except OSError as exc:
        log.error("Could not open reference FASTA %s: %s", ref_path, exc)
        sys.exit(1)
    if not seqs:
        log.error("Reference FASTA %s contains no sequences", ref_path)
        sys.exit(1)
    return seqs


def _strip_existing_kinetics(read: pysam.AlignedSegment) -> None:
    """Remove any pre-existing kinetic tags so the fresh write is clean."""
    for tag in ("fi", "fp", "ri", "rp", "ip", "pw"):
        try:
            read.set_tag(tag, None)
        except (KeyError, ValueError):
            pass


def generate_from_bam(
    input_bam: str,
    ref_path: str,
    checkpoint_path: str,
    motif_source: str,
    output_bam: str,
    *,
    deterministic: bool = False,
    device_str: str = "cuda",
    region: str | None = None,
    p_fire_default: float = 0.5,
    seed: int = 42,
) -> None:
    """Generate bilateral kinetics for every read in ``input_bam``.

    The output BAM is unaligned-style raw HiFi (flag=4) with the four
    PacBio tags injected. Feed the output to ``ccs-kinetics-bystrandify``
    then ``pbmm2 align`` for ipdSummary downstream.

    Per-row Bernoulli firing on signature offsets uses the per-bucket
    ``p_fire`` table from the checkpoint's ``model_config.json`` when
    present (written by ``refine``); buckets not in the lookup fall back
    to ``p_fire_default``. Independent draws per read mean motif
    occupancy realisations vary across reads at the same site.
    """
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    log.info("Loading model: %s", checkpoint_path)
    model, cfg = _load_model(checkpoint_path, device)
    kmer_size = int(cfg.get("kmer_size", get_extraction_params().kmer_size))
    upstream = int(cfg.get("active_site_index", get_extraction_params().upstream))
    num_meth_types = int(cfg.get("num_meth_types", max(get_meth_ids().values()) + 1))
    log.info(
        "Model: K=%d upstream=%d num_meth_types=%d arch=%s",
        kmer_size, upstream, num_meth_types,
        cfg.get("architecture", "?"),
    )

    # Build the (K, M) firing-rate table. Pull per-bucket survival rates
    # from the checkpoint's p_fire dict (refine meta carried through
    # train); fall back to the default for unseen buckets.
    p_fire_lookup = _resolve_p_fire_lookup(cfg)
    sig_offsets_by_meth = _build_sig_offsets_by_meth_id()
    rate_table = _build_pfire_rate_table(
        p_fire_lookup, sig_offsets_by_meth,
        kmer_size=kmer_size, active_site_index=upstream,
        num_meth_types=num_meth_types, default_rate=p_fire_default,
    )
    log.info(
        "p_fire rate table: default=%.2f  per-bucket buckets=%d  signature buckets=%d",
        p_fire_default, len(p_fire_lookup),
        sum(len(v) for v in sig_offsets_by_meth.values()),
    )
    rng = np.random.default_rng(int(seed))

    log.info("Loading reference: %s", ref_path)
    ref_seqs = _load_reference(ref_path)

    log.info("Resolving motifs: %s", motif_source)
    motif_string = load_motif_string(motif_source)
    fwd_meth_map, rev_meth_map = build_reference_meth_map_per_strand(ref_seqs, motif_string)

    log.info("Opening input BAM: %s", input_bam)
    in_bam = pysam.AlignmentFile(input_bam, "rb", check_sq=False)
    out_path = Path(output_bam)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = in_bam.header.to_dict()
    # Stamp our @PG entry so the provenance is recoverable.
    pg = header.setdefault("PG", [])
    pg.append({
        "ID": "kinsim2",
        "PN": "kinsim2",
        "VN": "0.1.0",
        "CL": " ".join(sys.argv),
        "DS": (
            f"bilateral generate K={kmer_size} upstream={upstream} "
            f"num_meth_types={num_meth_types} arch={cfg.get('architecture', '?')}"
        ),
    })
    out_bam = pysam.AlignmentFile(str(out_path), "wb", header=header)

    n_read = n_mapped = n_unmapped = 0
    iter_reads = (
        in_bam.fetch(region=region, until_eof=True)
        if region
        else in_bam.fetch(until_eof=True)
    )
    for read in iter_reads:
        n_read += 1
        fi, fp, ri, rp = _process_read(
            read, model, fwd_meth_map, rev_meth_map, ref_seqs,
            kmer_size=kmer_size, upstream=upstream,
            num_meth_types=num_meth_types, device=device,
            deterministic=deterministic,
            rate_table=rate_table, rng=rng,
        )
        _strip_existing_kinetics(read)
        _set_kinetic_tags(read, fi, fp, ri, rp)
        # Output is "unaligned HiFi" by convention for the validate chain;
        # downstream tools re-align with pbmm2.
        read.flag = 4
        read.reference_id = -1
        read.reference_start = -1
        read.next_reference_id = -1
        read.next_reference_start = -1
        read.cigartuples = None
        read.mapping_quality = 0
        out_bam.write(read)
        if read.is_unmapped:
            n_unmapped += 1
        else:
            n_mapped += 1
        if n_read % 10_000 == 0:
            log.info("  processed %d reads (mapped=%d, unmapped=%d)", n_read, n_mapped, n_unmapped)

    in_bam.close()
    out_bam.close()
    log.info(
        "Done. %d reads written to %s  (mapped=%d, unmapped=%d)",
        n_read, output_bam, n_mapped, n_unmapped,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="kinsim generate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_bam", help="Raw HiFi aligned BAM (input)")
    ap.add_argument("ref_path", help="Reference FASTA the BAM was aligned against")
    ap.add_argument("checkpoint", help="Path to checkpoint .pt or .ckpt")
    ap.add_argument("motifs", help="Motif string or motifs.csv path")
    ap.add_argument("output_bam", help="Output BAM (unaligned HiFi with fi/fp/ri/rp)")
    ap.add_argument(
        "--deterministic", action="store_true",
        help="Emit mu directly instead of sampling from N(mu, sigma^2).",
    )
    ap.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Inference device (auto-falls-back to cpu).",
    )
    ap.add_argument(
        "--region", default=None,
        help="Restrict to a samtools region (e.g. 'chr1:1000-2000').",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for stochastic sampling AND Bernoulli firing "
        "(default: 42). Same seed + same input -> identical output BAM.",
    )
    ap.add_argument(
        "--p-fire", dest="p_fire", type=float, default=0.5,
        help="Bernoulli firing rate at signature offsets when the checkpoint's "
        "p_fire dict is empty or doesn't cover the (meth, offset) bucket "
        "(default: 0.5 — half of motif sites visibly fire per read, matching "
        "the typical range refine reports across the Strepto+Vega corpus). "
        "Set to 1.0 to disable firing (always emit the model's learned signal).",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(verbose=args.verbose)

    # Seed torch + numpy globally so model.sample() is reproducible across runs.
    import random
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    generate_from_bam(
        args.input_bam,
        args.ref_path,
        args.checkpoint,
        args.motifs,
        args.output_bam,
        deterministic=args.deterministic,
        device_str=args.device,
        region=args.region,
        p_fire_default=float(args.p_fire),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
