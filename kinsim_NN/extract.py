"""``kinsim_nn extract`` — labeler-driven extraction → shards.

For each strain in the manifest, this:
  1. Loads the reference FASTA.
  2. Runs the chain of labelers (configured in YAML) to produce
     ``(ref_id, pos, meth_id, strand)`` records.
  3. Samples baseline positions ≥ ``baseline_min_dist`` bp from any
     labeled meth position.
  4. For each (position, strand) labelled, walks the aligned BAM and
     extracts the bilateral 4-channel signal for up to
     ``reads_cap_per_position`` reads.
  5. Writes a single ``shards/<sample_id>_shard.pkl`` file.

Designed for SLURM array execution: pass ``--task <i>`` to process
manifest row ``i`` only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pysam

from . import __version__
from .data.shard import (
    CATEGORY_BASELINE,
    CATEGORY_NEAR_METH,
    CATEGORY_SLOWED,
    SHARD_CONFIG_VERSION,
    empty_shard,
    finalize_shard,
    hash_zmw,
    write_shard,
)
from .labelers import create_labeler
from .utils.bam_io import (
    detect_bam_format,
    iter_chunk_samples,
    iter_window_samples,
)
from .utils.config import KinsimNNConfig, load_config, setup_logging
from .utils.encoding import N_BASE_COUNT, encode_seq


log = logging.getLogger(__name__)


# Base encoding lives in kinsim_NN.utils.encoding (single source of truth).
_encode_seq = encode_seq


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Return list of {sample_id, bam_path, ref_path, strain_dir, ...} dicts."""
    out = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("sample_id"):
                continue
            out.append(dict(row))
    return out


def _resolve_strain_dir(row: dict) -> Path:
    """Resolve the per-strain "home" directory used by labeler file_patterns.

    Prefers ``parent(ref_path)`` because per-strain artefacts (motifs.gff,
    motifs.csv, REBASE annotations, jasmine_5mC.bam) are typically placed
    next to the reference FASTA — not next to the pipeline-produced BAM
    (which lives under ``pipeline/<strain>/``).

    Falls back to ``parent(bam_path)`` if ``ref_path`` is missing.
    """
    ref = row.get("ref_path")
    if ref:
        return Path(ref).parent
    return Path(row["bam_path"]).parent


def _load_ref_fasta(ref_path: Path) -> dict[str, str]:
    """Load all contigs from a FASTA → {contig_id: seq}. Uses pysam.FastaFile
    for indexed random access; falls back to a full read otherwise."""
    try:
        fa = pysam.FastaFile(str(ref_path))
        out = {ref: fa.fetch(ref) for ref in fa.references}
        fa.close()
        return out
    except (OSError, ValueError) as e:
        log.warning("FastaFile failed (%s) — falling back to plain reader", e)
        out = {}
        with open(ref_path) as f:
            cur_id, parts = None, []
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if cur_id is not None:
                        out[cur_id] = "".join(parts)
                    cur_id = line[1:].split()[0]
                    parts = []
                else:
                    parts.append(line)
            if cur_id is not None:
                out[cur_id] = "".join(parts)
        return out


def _build_labelers(cfg: KinsimNNConfig) -> list:
    """Instantiate labelers from the YAML config in order."""
    labelers = []
    for entry in cfg.labelers:
        t = entry.get("type")
        kwargs = {k: v for k, v in entry.items() if k != "type"}
        labeler = create_labeler(t, **kwargs)
        labelers.append(labeler)
    return labelers


def _collect_labels(
    labelers: list,
    cfg: KinsimNNConfig,
    ref_seqs: dict[str, str],
    strain_dir: Path,
) -> dict[tuple[str, int, str], int]:
    """Run all labelers, merge labels.

    Returns ``{(ref_id, pos_0based, strand): meth_id}``. Earlier labelers
    win on conflicts. Labelers loop OUTER so each source file (GFF, BAM)
    is parsed only ONCE per strain — the labeler itself iterates the
    contigs internally.
    """
    result: dict[tuple[str, int, str], int] = {}
    meth_id_by_name = cfg.meth_id_by_name
    for labeler in labelers:
        for ref_id, ref_seq in ref_seqs.items():
            for rid, pos, mid, strand in labeler.label(
                ref_id, ref_seq, strain_dir,
                meth_id_by_name=meth_id_by_name,
                treat_modified_base_as=cfg.treat_modified_base_as,
            ):
                key = (rid, pos, strand)
                if key not in result:
                    result[key] = mid
    return result


def _sample_baselines(
    label_positions: set[tuple[str, int, str]],
    ref_seqs: dict[str, str],
    n_samples: int,
    min_dist: int,
    half_width: int,
    rng: random.Random,
) -> list[tuple[str, int, str]]:
    """Sample baseline positions ≥ ``min_dist`` bp from any labelled
    methylation position."""
    # Build per-contig labelled-set for fast lookups (kept compact via bitarrays
    # using numpy).
    forbidden_by_ref: dict[str, np.ndarray] = {}
    for ref_id, seq in ref_seqs.items():
        forbidden = np.zeros(len(seq), dtype=bool)
        forbidden_by_ref[ref_id] = forbidden

    for (rid, pos, _strand) in label_positions:
        if rid not in forbidden_by_ref:
            continue
        f = forbidden_by_ref[rid]
        lo = max(0, pos - min_dist)
        hi = min(f.size, pos + min_dist + 1)
        f[lo:hi] = True

    refs = list(ref_seqs.keys())
    out = []
    tries = 0
    max_tries = max(n_samples * 50, 1000)
    while len(out) < n_samples and tries < max_tries:
        tries += 1
        rid = rng.choice(refs)
        L = len(ref_seqs[rid])
        if L < 2 * half_width + 10:
            continue
        pos = rng.randint(half_width, L - half_width - 1)
        if forbidden_by_ref[rid][pos]:
            continue
        strand = rng.choice(["+", "-"])
        out.append((rid, pos, strand))
    if len(out) < n_samples:
        log.warning(
            "Baseline sampling reached only %d/%d after %d tries",
            len(out), n_samples, tries,
        )
    return out


def _flush_position_to_builder(
    pos: int,
    samples: list,
    base_fwd: np.ndarray,
    meth_fwd: np.ndarray,
    meth_rev: np.ndarray,
    category: int,
    parent_meth: int,
    parent_offset: int,
    strand_center: str,
    builder: dict,
    ref_id_idx: int,
) -> int:
    """Append all collected samples for one ref position into the shard builder.

    ``category`` is one of CATEGORY_{BASELINE,SLOWED,NEAR_METH}. ``parent_meth``
    is the meth id of the methylation that this position originated from
    (0 for baseline). ``parent_offset`` is the bp offset from that parent
    meth's centre (0 for baseline).
    """
    n = 0
    strand_int = 1 if strand_center == "+" else -1
    for sample in samples:
        signal = np.stack(
            [sample.ipd_fwd, sample.pw_fwd, sample.ipd_rev, sample.pw_rev],
            axis=-1,
        ).astype(np.uint8)
        builder["base_fwd"].append(base_fwd)
        builder["meth_fwd"].append(meth_fwd)
        builder["meth_rev"].append(meth_rev)
        builder["signal"].append(signal)
        builder["category"].append(category)
        builder["parent_meth"].append(parent_meth)
        builder["parent_offset"].append(parent_offset)
        builder["ref_id"].append(ref_id_idx)
        builder["ref_pos"].append(int(pos))
        builder["strand"].append(strand_int)
        builder["zmw"].append(hash_zmw(sample.zmw_id))
        n += 1
    return n


def _build_window_tensors(
    pos: int,
    seqid: str,
    ref_seq: str,
    cfg: KinsimNNConfig,
    labels: dict[tuple[str, int, str], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return ``(base_fwd, meth_fwd, meth_rev)`` for one window center, or
    None if the window goes off the contig edge."""
    half = cfg.window.half_width
    K = cfg.window.k
    if pos - half < 0 or pos + half + 1 > len(ref_seq):
        return None
    base_fwd = _encode_seq(ref_seq[pos - half: pos + half + 1])
    meth_fwd = np.zeros(K, dtype=np.uint8)
    meth_rev = np.zeros(K, dtype=np.uint8)
    for k in range(K):
        win_pos = int(pos) - half + k
        fwd_id = labels.get((seqid, win_pos, "+"))
        if fwd_id is not None:
            meth_fwd[k] = fwd_id
        rev_id = labels.get((seqid, win_pos, "-"))
        if rev_id is not None:
            meth_rev[k] = rev_id
    return base_fwd, meth_fwd, meth_rev


def _extract_chunk_batched(
    bam: pysam.AlignmentFile,
    bam_fmt,
    seqid: str,
    chunk_positions: list[tuple[int, str, int, int, int]],
    # [(ref_pos, strand, category, parent_meth, parent_offset), ...] sorted by ref_pos
    ref_seq: str,
    cfg: KinsimNNConfig,
    rng: random.Random,
    builder: dict,
    ref_id_idx: int,
    labels: dict[tuple[str, int, str], int],
) -> int:
    """Process a spatially-coherent batch of positions with one BAM fetch.

    Per-position state (window tensors + reservoir of samples) is kept in
    dicts keyed by ref_pos. ``iter_chunk_samples`` walks each ZMW pair ONCE
    and yields per-(center_pos, ZMW); we route into the right position's
    reservoir.
    """
    if not chunk_positions:
        return 0
    half = cfg.window.half_width
    cap = int(cfg.extract.reads_cap_per_position)

    # Pre-build window tensors per position (cheap CPU work, no I/O).
    pos_tensors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    pos_meta: dict[int, tuple[str, int, int, int]] = {}  # strand, category, parent_meth, parent_offset
    for p, strand, category, parent_meth, parent_offset in chunk_positions:
        t = _build_window_tensors(p, seqid, ref_seq, cfg, labels)
        if t is None:
            continue
        pos_tensors[p] = t
        pos_meta[p] = (strand, category, parent_meth, parent_offset)
    if not pos_tensors:
        return 0

    sorted_positions = np.array(sorted(pos_tensors.keys()), dtype=np.int64)

    # Reservoir accumulator per position
    pos_samples: dict[int, list] = {p: [] for p in pos_tensors}
    pos_n_seen: dict[int, int] = {p: 0 for p in pos_tensors}

    n_yields = 0
    for center_pos, sample in iter_chunk_samples(
        bam, bam_fmt, seqid, sorted_positions, half,
        min_mapq=cfg.extract.min_mapq,
    ):
        n_yields += 1
        if center_pos not in pos_samples:
            continue
        n = pos_n_seen[center_pos] + 1
        pos_n_seen[center_pos] = n
        samples = pos_samples[center_pos]
        if len(samples) < cap:
            samples.append(sample)
        else:
            j = rng.randrange(n)
            if j < cap:
                samples[j] = sample

    # Diagnostic: log a warning if a non-empty chunk got zero yields. Means
    # iter_chunk_samples filtered EVERY ZMW pair out — usually a min_mapq
    # threshold too high, or strand-mismatch dominating, or empty tag arrays.
    if n_yields == 0 and len(pos_tensors) > 0:
        log.warning(
            "[chunk %s:%d-%d] %d centers but iter_chunk_samples yielded 0 "
            "samples — check min_mapq, strand orientation, or tag presence.",
            seqid,
            int(sorted_positions[0]),
            int(sorted_positions[-1]),
            len(pos_tensors),
        )

    # Flush per-position
    n_added = 0
    for pos in sorted_positions:
        p = int(pos)
        if not pos_samples[p]:
            continue
        base_fwd, meth_fwd, meth_rev = pos_tensors[p]
        strand, category, parent_meth, parent_offset = pos_meta[p]
        n_added += _flush_position_to_builder(
            p, pos_samples[p], base_fwd, meth_fwd, meth_rev,
            category, parent_meth, parent_offset, strand,
            builder, ref_id_idx,
        )
    return n_added


def extract_strain(
    manifest_row: dict,
    output_dir: Path,
    cfg: KinsimNNConfig,
) -> None:
    """Extract a single strain → ``<output_dir>/<sample_id>_shard.pkl``."""
    sample_id = manifest_row["sample_id"]
    bam_path = Path(manifest_row["bam_path"])
    ref_path = Path(manifest_row.get("ref_path", ""))
    strain_dir = _resolve_strain_dir(manifest_row)

    if not bam_path.is_file():
        log.error("[%s] BAM missing: %s", sample_id, bam_path)
        return
    if not ref_path.is_file():
        log.error("[%s] Reference missing: %s", sample_id, ref_path)
        return

    log.info("[%s] BAM=%s  ref=%s  strain_dir=%s", sample_id, bam_path, ref_path, strain_dir)

    ref_seqs = _load_ref_fasta(ref_path)
    ref_names = list(ref_seqs.keys())
    ref_name_to_idx = {n: i for i, n in enumerate(ref_names)}
    log.info("[%s] %d contigs", sample_id, len(ref_seqs))

    # Labels
    labelers = _build_labelers(cfg)
    labels = _collect_labels(labelers, cfg, ref_seqs, strain_dir)
    log.info("[%s] %d labelled positions", sample_id, len(labels))

    # Baselines — use a stable hash so seeding is reproducible across processes
    # (CPython's builtin hash() of strings is randomized per process).
    stable_hash = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(cfg.train.seed + stable_hash)
    baseline_positions = _sample_baselines(
        set(labels.keys()),
        ref_seqs,
        cfg.extract.baseline_per_strain,
        cfg.extract.baseline_min_dist,
        cfg.window.half_width,
        rng,
    )
    log.info("[%s] %d baseline positions sampled", sample_id, len(baseline_positions))

    # BAM I/O
    bam_fmt = detect_bam_format(bam_path)
    log.info("[%s] BAM format: bystrandified=%s tags=(%s, %s, %s, %s)",
             sample_id, bam_fmt.is_bystrandified,
             bam_fmt.ipd_fwd_tag, bam_fmt.pw_fwd_tag,
             bam_fmt.ipd_rev_tag, bam_fmt.pw_rev_tag)

    builder = empty_shard(cfg.window.k)
    bam = pysam.AlignmentFile(str(bam_path), "rb")

    # Methylated positions — dedup by (rid, pos) so palindromic motifs that
    # contribute BOTH "+" and "-" labels at the same position only produce ONE
    # parent for the expansion.
    unique_meth_positions: dict[tuple[str, int], tuple[int, str]] = {}
    for (rid, pos, strand), mid in labels.items():
        key = (rid, pos)
        if key not in unique_meth_positions:
            unique_meth_positions[key] = (mid, strand)

    # Optional random subsample of meth POSITIONS per strain (before expansion).
    # Strepto strains can have 400k+ labelled positions; capping here bounds
    # BAM I/O without changing the SLOWED/NEAR_METH ratio.
    meth_items = list(unique_meth_positions.items())
    cap = int(getattr(cfg.extract, "meth_per_strain_cap", 0) or 0)
    if cap and len(meth_items) > cap:
        rng.shuffle(meth_items)
        meth_items = meth_items[:cap]
        log.info("[%s] Subsampled %d meth positions to %d (cap)",
                 sample_id, len(unique_meth_positions), cap)

    # ----------------------------------------------------------------------
    # v3-style emission expansion: each meth position p of type T spawns
    # emission candidates at p+k for k in [0, near_meth_max_dist]. Category
    # is SLOWED if k in T.signal_offsets, else NEAR_METH. Conflicts resolved
    # with SLOWED-beats-NEAR_METH precedence; within same category last
    # writer wins (deterministic by iteration order).
    # ----------------------------------------------------------------------
    signal_offsets_by_id: dict[int, frozenset[int]] = {
        t.id: frozenset(t.signal_offsets) for t in cfg.methylation_types
    }
    near_meth_max_dist = int(getattr(cfg.extract, "near_meth_max_dist", 10))

    # key=(rid, emit_pos, strand) → (category, parent_meth, parent_offset)
    emissions: dict[tuple[str, int, str], tuple[int, int, int]] = {}
    for (rid, p), (mid, strand) in meth_items:
        if rid not in ref_seqs:
            continue
        sig_offs = signal_offsets_by_id.get(mid, frozenset())
        for k in range(0, near_meth_max_dist + 1):
            emit_pos = p + k
            key = (rid, emit_pos, strand)
            if k in sig_offs:
                emissions[key] = (CATEGORY_SLOWED, mid, k)
            else:
                existing = emissions.get(key)
                if existing is not None and existing[0] == CATEGORY_SLOWED:
                    continue  # SLOWED stays; don't downgrade
                emissions[key] = (CATEGORY_NEAR_METH, mid, k)

    # Split by category for independent caps
    slowed_items: list[tuple[str, int, str, int, int]] = []     # (rid, pos, strand, parent_meth, parent_offset)
    near_meth_items: list[tuple[str, int, str, int, int]] = []
    for (rid, pos, strand), (cat, pmeth, poff) in emissions.items():
        if cat == CATEGORY_SLOWED:
            slowed_items.append((rid, pos, strand, pmeth, poff))
        else:
            near_meth_items.append((rid, pos, strand, pmeth, poff))

    n_slowed_total = len(slowed_items)
    n_near_meth_total = len(near_meth_items)
    slowed_cap = int(getattr(cfg.extract, "slowed_per_strain_cap", 0) or 0)
    near_meth_cap = int(getattr(cfg.extract, "near_meth_per_strain_cap", 0) or 0)
    if slowed_cap and len(slowed_items) > slowed_cap:
        rng.shuffle(slowed_items)
        slowed_items = slowed_items[:slowed_cap]
    if near_meth_cap and len(near_meth_items) > near_meth_cap:
        rng.shuffle(near_meth_items)
        near_meth_items = near_meth_items[:near_meth_cap]
    log.info(
        "[%s] Emission expansion: %d meth positions → %d SLOWED (cap→%d) + "
        "%d NEAR_METH (cap→%d); near_meth_max_dist=%d",
        sample_id, len(meth_items),
        n_slowed_total, len(slowed_items),
        n_near_meth_total, len(near_meth_items),
        near_meth_max_dist,
    )

    # Chunk positions by spatial locality (same contig, span ≤ CHUNK_SPAN_BP).
    # Each chunk → ONE bam.fetch, ONE get_aligned_pairs per record. Massive
    # speedup vs per-position fetch on bystrandified BAMs.
    CHUNK_SPAN_BP = 5000      # max ref-distance between first and last pos in a chunk
    CHUNK_MAX_POSITIONS = 256  # cap so memory stays bounded (~200kb of sample buffers)
    PROGRESS_EVERY_CHUNKS = 50

    import time as _time
    from collections import defaultdict as _dd

    def _run_chunked(
        items: list[tuple[str, int, str, int, int]],
        category: int,
        phase_label: str,
    ) -> int:
        """Chunk a list of (rid, pos, strand, parent_meth, parent_offset) by
        spatial locality and run them through _extract_chunk_batched."""
        # Sort by (rid, pos) for OS pagecache locality
        items_sorted = sorted(items, key=lambda x: (x[0], x[1]))
        by_ref: dict[str, list[tuple[int, str, int, int, int]]] = _dd(list)
        for rid, pos, strand, pmeth, poff in items_sorted:
            by_ref[rid].append((pos, strand, category, pmeth, poff))
        n_samples = 0
        t0 = _time.time()
        n_chunks_done = 0
        total = sum(len(v) for v in by_ref.values())
        done = 0
        for rid, ritems in by_ref.items():
            i = 0
            while i < len(ritems):
                chunk_start_pos = ritems[i][0]
                j = i + 1
                while (j < len(ritems)
                       and ritems[j][0] - chunk_start_pos < CHUNK_SPAN_BP
                       and (j - i) < CHUNK_MAX_POSITIONS):
                    j += 1
                chunk = ritems[i:j]
                n_samples += _extract_chunk_batched(
                    bam, bam_fmt, rid, chunk, ref_seqs[rid], cfg, rng, builder,
                    ref_name_to_idx[rid], labels,
                )
                done += len(chunk)
                i = j
                n_chunks_done += 1
                if n_chunks_done % PROGRESS_EVERY_CHUNKS == 0:
                    elapsed = _time.time() - t0
                    rate = done / max(elapsed, 1e-3)
                    eta = (total - done) / max(rate, 1e-3)
                    log.info(
                        "[%s] %s chunks=%d positions=%d/%d (%.0f pos/s, ETA %.0f min, samples=%d)",
                        sample_id, phase_label, n_chunks_done, done, total,
                        rate, eta / 60, n_samples,
                    )
        log.info(
            "[%s] %s phase done: %d chunks, %d samples in %.1f min",
            sample_id, phase_label, n_chunks_done, n_samples,
            (_time.time() - t0) / 60,
        )
        return n_samples

    n_slowed_samples = _run_chunked(slowed_items, CATEGORY_SLOWED, "SLOWED")
    n_near_meth_samples = _run_chunked(near_meth_items, CATEGORY_NEAR_METH, "NEAR_METH")

    # Baseline positions — same chunked path, category=BASELINE, no parent.
    baseline_emission_items = [
        (rid, pos, strand, 0, 0) for (rid, pos, strand) in baseline_positions
    ]
    n_baseline_samples = _run_chunked(
        baseline_emission_items, CATEGORY_BASELINE, "baseline",
    )

    n_meth_samples = n_slowed_samples + n_near_meth_samples

    bam.close()

    meta = {
        "config_version": SHARD_CONFIG_VERSION,
        "k": cfg.window.k,
        "half_width": cfg.window.half_width,
        "n_channels": cfg.window.n_channels,
        "n_meth_types": cfg.n_meth_types,
        "meth_id_by_name": cfg.meth_id_by_name,
        "ref_names": ref_names,
        "strain_id": sample_id,
        "git_sha": _git_sha(),
        "kinsim_nn_version": __version__,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "label_sources": [e.get("type") for e in cfg.labelers],
        "n_meth_samples": n_meth_samples,
        "n_baseline_samples": n_baseline_samples,
        "n_slowed_samples": n_slowed_samples,
        "n_near_meth_samples": n_near_meth_samples,
        "signal_offsets_by_id": {
            t.id: list(t.signal_offsets) for t in cfg.methylation_types
        },
        "near_meth_max_dist": int(getattr(cfg.extract, "near_meth_max_dist", 10)),
    }

    shard = finalize_shard(builder, meta, cfg.window.k)
    out_path = output_dir / f"{sample_id}_shard.pkl"
    write_shard(out_path, shard)
    log.info(
        "[%s] Done. SLOWED=%d  NEAR_METH=%d  BASELINE=%d  shard=%s  "
        "non_ACGT_bases_silently_encoded_as_A=%d",
        sample_id, n_slowed_samples, n_near_meth_samples, n_baseline_samples,
        out_path, N_BASE_COUNT[0],
    )
    if N_BASE_COUNT[0] > 1000:
        log.warning(
            "[%s] %d non-ACGT bases were silently encoded as A. "
            "Consider hard-masking ambiguity to N in the reference or adding "
            "a 5th class to the base alphabet.",
            sample_id, N_BASE_COUNT[0],
        )
    N_BASE_COUNT[0] = 0  # reset for next strain in batched runs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kinsim_nn extract", description=__doc__)
    ap.add_argument(
        "--manifest", required=True,
        help="Manifest CSV with sample_id + bam_path + ref_path columns. "
             "Methylation labels are located via labeler file_pattern (typically "
             "{strain_dir}/motifs.gff), not via a 'motifs' column.",
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write shards")
    ap.add_argument("--config", default=None, help="kinsim_nn_config.yaml path")
    ap.add_argument("--task", type=int, default=None,
                    help="If set, process ONLY manifest row at this index (0-based, for SLURM array).")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(verbose=args.verbose)

    cfg = load_config(args.config)
    manifest = _load_manifest(Path(args.manifest))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task is not None:
        if not (0 <= args.task < len(manifest)):
            sys.exit(f"--task {args.task} out of range (manifest has {len(manifest)} rows)")
        extract_strain(manifest[args.task], out_dir, cfg)
    else:
        for row in manifest:
            extract_strain(row, out_dir, cfg)


if __name__ == "__main__":
    main()
