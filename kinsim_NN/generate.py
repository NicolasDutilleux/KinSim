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
import re
import sys
from pathlib import Path

import numpy as np
import pysam
import torch

from . import __version__
from .models.generator import TransformerGenerator
from .utils.config import load_config, setup_logging
from .utils.pacbio_codec import log1p_frames_to_uint8


log = logging.getLogger(__name__)


_BASE_TO_CODE = {b: i for i, b in enumerate("ACGT")}
_BASE_TO_CODE.update({b: i for i, b in enumerate("acgt")})
_IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
    "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}


# ---------------------------------------------------------------------------
# Motif scanning
# ---------------------------------------------------------------------------


def _iupac_to_regex(motif: str) -> str:
    return "".join(_IUPAC.get(b.upper(), b) for b in motif)


def _revcomp(seq: str) -> str:
    rc = str.maketrans("ACGTacgtRYSWKMBDHVN", "TGCAtgcaYRSWMKVHDBN")
    return seq.translate(rc)[::-1]


def _parse_motifs_csv(path: Path, meth_id_by_name: dict[str, int]) -> list[dict]:
    """Return list of motif dicts: {pattern_fwd, pattern_rev, center_offset,
    meth_id, fraction, name}. center_offset is the 0-based index inside the
    motif where the methylation lies."""
    import csv

    motifs: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r.get("motifString") or r.get("motif") or r.get("Motif")
            if not name:
                continue
            mod = r.get("modificationType") or r.get("mod_type") or r.get("type") or ""
            if mod not in meth_id_by_name:
                continue
            try:
                center = int(r.get("centerPos") or r.get("offset") or 0)
            except ValueError:
                center = 0
            try:
                fraction = float(r.get("fraction") or r.get("frac_mod") or 1.0)
            except ValueError:
                fraction = 1.0
            motifs.append({
                "name": name,
                "pattern_fwd": _iupac_to_regex(name),
                "pattern_rev": _iupac_to_regex(_revcomp(name)),
                "len": len(name),
                "center_offset": center,
                "meth_id": meth_id_by_name[mod],
                "fraction": fraction,
            })
    return motifs


def _build_ref_meth_map(
    ref_seq: str,
    motifs: list[dict],
) -> dict[tuple[str, int], tuple[int, float]]:
    """Scan ref for motif matches. Returns ``{(strand, pos): (meth_id, fraction)}``.

    The position is the absolute reference coordinate of the methylated base
    (motif start + center_offset).
    """
    out: dict[tuple[str, int], tuple[int, float]] = {}
    for m in motifs:
        # forward strand
        for match in re.finditer(m["pattern_fwd"], ref_seq, flags=re.IGNORECASE):
            pos = match.start() + m["center_offset"]
            key = ("+", pos)
            if key not in out:
                out[key] = (m["meth_id"], m["fraction"])
        # reverse strand: scan ref for revcomp pattern
        for match in re.finditer(m["pattern_rev"], ref_seq, flags=re.IGNORECASE):
            # Position of the methylated base on the rev-strand template, in
            # forward-ref coordinates: match.end() - 1 - center_offset
            pos = match.end() - 1 - m["center_offset"]
            key = ("-", pos)
            if key not in out:
                out[key] = (m["meth_id"], m["fraction"])
    return out


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


def _encode_seq(seq: str) -> np.ndarray:
    return np.fromiter(
        (_BASE_TO_CODE.get(b, 0) for b in seq),
        dtype=np.uint8, count=len(seq),
    )


_RC_TABLE = np.array([3, 2, 1, 0], dtype=np.uint8)


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
    B, K = base_fwd_stack.shape
    base_rev = _RC_TABLE[base_fwd_stack]
    # one-hot via fancy-indexing
    base_fwd_oh = np.zeros((B, K, 4), dtype=np.float32)
    base_rev_oh = np.zeros((B, K, 4), dtype=np.float32)
    np.put_along_axis(base_fwd_oh, base_fwd_stack.astype(np.int64)[..., None], 1.0, axis=-1)
    np.put_along_axis(base_rev_oh, base_rev.astype(np.int64)[..., None], 1.0, axis=-1)
    meth_fwd_oh = np.zeros((B, K, n_meth_types), dtype=np.float32)
    meth_rev_oh = np.zeros((B, K, n_meth_types), dtype=np.float32)
    np.put_along_axis(meth_fwd_oh, meth_fwd_stack.astype(np.int64)[..., None], 1.0, axis=-1)
    np.put_along_axis(meth_rev_oh, meth_rev_stack.astype(np.int64)[..., None], 1.0, axis=-1)

    z = g.sample_z(B, device=device)
    out = g(
        z,
        torch.from_numpy(base_fwd_oh).to(device),
        torch.from_numpy(base_rev_oh).to(device),
        torch.from_numpy(meth_fwd_oh).to(device),
        torch.from_numpy(meth_rev_oh).to(device),
    )
    out_np = out.cpu().numpy()                            # (B, K, 4)
    return out_np[:, K // 2].astype(np.float32)           # (B, 4)


def _build_meth_window(
    meth_map: dict[tuple[str, int], tuple[int, float]],
    ref_pos_window: np.ndarray,
    rng: random.Random,
    use_bernoulli: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (meth_fwd[K], meth_rev[K]) by overlaying motif map entries
    that fall inside the window. Bernoulli sampling per motif site."""
    K = ref_pos_window.shape[0]
    meth_fwd = np.zeros(K, dtype=np.uint8)
    meth_rev = np.zeros(K, dtype=np.uint8)
    for i, rp in enumerate(ref_pos_window):
        rp = int(rp)
        for strand, arr in [("+", meth_fwd), ("-", meth_rev)]:
            entry = meth_map.get((strand, rp))
            if entry is None:
                continue
            mid, frac = entry
            if use_bernoulli and rng.random() > frac:
                continue
            arr[i] = mid
    return meth_fwd, meth_rev


def _process_mapped_read(
    read: pysam.AlignedSegment,
    ref_seq: str,
    meth_map: dict[tuple[str, int], tuple[int, float]],
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

    Strand handling: for reverse-mapped reads, BAM stores B:C tag arrays
    in synthesis-strand orientation (which is the reverse complement of
    forward-ref direction). We therefore (a) flip the query index
    (synthesis_q = qlen - 1 - q) and (b) swap the channel pairs so the
    forward-ref signal channel 0/1 lands on what BAM calls ri/rp.
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

    # Gather all windows that pass filtering, then batch G calls
    windows = []
    for q_pos, r_pos in pair_arr:
        if q_pos < n_context_skip or q_pos >= qlen - n_context_skip:
            continue
        if r_pos - half_width < 0 or r_pos + half_width + 1 > len(ref_seq):
            continue
        base_fwd = _encode_seq(ref_seq[r_pos - half_width: r_pos + half_width + 1])
        ref_window = np.arange(r_pos - half_width, r_pos + half_width + 1, dtype=np.int64)
        meth_fwd, meth_rev = _build_meth_window(meth_map, ref_window, rng, use_bernoulli)
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
        for j, (q_pos, _, _, _) in enumerate(chunk):
            if is_rev:
                # BAM stores arrays in synthesis-strand orientation for rev reads
                idx = qlen - 1 - q_pos
                fi[idx] = u8[j, 2]   # ref's IPD_rev → synthesis-fwd's fi
                fp[idx] = u8[j, 3]
                ri[idx] = u8[j, 0]
                rp[idx] = u8[j, 1]
            else:
                fi[q_pos] = u8[j, 0]
                fp[q_pos] = u8[j, 1]
                ri[q_pos] = u8[j, 2]
                rp[q_pos] = u8[j, 3]
    return fi, fp, ri, rp


def _process_unmapped_read(
    read: pysam.AlignedSegment,
    motifs: list[dict],
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

    # Build per-(strand, query-pos) meth map by scanning the read sequence
    query_meth_map = _build_ref_meth_map(qseq, motifs)

    windows = []
    for q_pos in range(max(half_width, n_context_skip),
                       qlen - max(half_width, n_context_skip)):
        base_fwd = _encode_seq(qseq[q_pos - half_width: q_pos + half_width + 1])
        ref_window = np.arange(q_pos - half_width, q_pos + half_width + 1, dtype=np.int64)
        meth_fwd, meth_rev = _build_meth_window(query_meth_map, ref_window, rng, use_bernoulli)
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
        for j, (q_pos, _, _, _) in enumerate(chunk):
            fi[q_pos] = u8[j, 0]
            fp[q_pos] = u8[j, 1]
            ri[q_pos] = u8[j, 2]
            rp[q_pos] = u8[j, 3]
    return fi, fp, ri, rp


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def generate(
    input_bam: Path,
    ref_fasta: Path,
    ckpt_dir: Path,
    motifs_csv: Path,
    output_bam: Path,
    cfg_yaml: Path | None,
    seed: int = 42,
    use_bernoulli: bool = True,
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

    motifs = _parse_motifs_csv(motifs_csv, model_cfg["meth_id_by_name"])
    log.info("Loaded %d motifs", len(motifs))

    meth_maps = {r: _build_ref_meth_map(seq, motifs) for r, seq in ref_seqs.items()}
    log.info(
        "Total motif sites: %d",
        sum(len(m) for m in meth_maps.values()),
    )

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    in_bam = pysam.AlignmentFile(str(input_bam), "rb", check_sq=False)
    header = in_bam.header.to_dict()
    pg = header.setdefault("PG", [])
    pg.append({
        "ID": f"kinsim_nn-{__version__}",
        "PN": "kinsim_nn",
        "VN": __version__,
        "CL": f"kinsim_nn generate (ckpt={ckpt_dir.name})",
    })
    out_bam = pysam.AlignmentFile(str(output_bam), "wb", header=header)

    n_reads = 0
    n_unmapped = 0
    for read in in_bam.fetch(until_eof=True):
        if read.is_secondary or read.is_supplementary:
            out_bam.write(read)
            continue
        if read.is_unmapped:
            # PBSIM3 sim output or any unaligned read: scan motifs against
            # the query sequence directly.
            result = _process_unmapped_read(
                read=read,
                motifs=motifs,
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
            fi, fp, ri, rp = _process_mapped_read(
                read=read,
                ref_seq=ref_seqs[ref_id],
                meth_map=meth_maps[ref_id],
                g=g,
                n_meth_types=n_meth_types,
                half_width=half_width,
                n_context_skip=n_context_skip,
                default_value=cfg.generate.default_fi_for_unknown,
                use_bernoulli=use_bernoulli,
                rng=rng,
                device=device,
            )
        # Strip any existing kinetics tags before writing fresh ones
        for tag in ("fi", "fp", "ri", "rp", "ip", "pw"):
            try:
                read.set_tag(tag, None)
            except KeyError:
                pass
        # Explicit array.array("B", ...) so the subtype is unambiguously uint8
        # (avoids fragile inference from a list of ints).
        read.set_tag("fi", array.array("B", fi.tolist()))
        read.set_tag("fp", array.array("B", fp.tolist()))
        read.set_tag("ri", array.array("B", ri.tolist()))
        read.set_tag("rp", array.array("B", rp.tolist()))
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
    ap.add_argument("--no-bernoulli", action="store_true",
                    help="Override YAML and disable Bernoulli sampling on motif fraction "
                         "(always set meth at every motif site).")
    ap.add_argument("--force-bernoulli", action="store_true",
                    help="Override YAML and force Bernoulli sampling on motif fraction.")
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
    )


if __name__ == "__main__":
    main()
