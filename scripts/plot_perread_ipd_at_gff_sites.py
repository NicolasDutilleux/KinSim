"""Per-read IPD distribution at ipdSummary-called positions vs random control.

For each of the top-N GFF calls of a given ``--mod-type`` (sorted by QV), walk
the aligned bystrandified BAM, collect per-read ``ip`` (or ``fi``) values at the
exact reference position, and plot a 4×5 grid of histograms. A parallel grid of
random non-called positions provides the control.

Usage::

    python scripts/plot_perread_ipd_at_gff_sites.py \\
        --bam <aligned.bam> --gff <ipdSummary.gff> \\
        --output-prefix <out> [--mod-type m6A] [--n-sites 20]

Outputs::

    <prefix>_methylated.png    top-N called positions
    <prefix>_control.png       N random positions not in the GFF (or QV<10)

Strand handling: bystrandified BAMs split each ZMW into ``ccs/fwd`` and
``ccs/rev`` records. We accept reads whose orientation matches the GFF strand
(``+`` → ``not read.is_reverse``; ``-`` → ``read.is_reverse``). Tags read with
fallback ``ip`` → ``fi`` so the script works on either bystrandified (ip/pw) or
raw HiFi (fi/fp) BAMs.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pysam


def parse_gff(gff_path: Path, mod_type: str | None) -> list[dict]:
    """Return list of {seqid, pos (0-based), strand, qv, ipd_ratio, mod_type}."""
    rows: list[dict] = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, _src, mtype, start, _end, score, strand, _frame, attrs = parts[:9]
            if mod_type and mtype != mod_type:
                continue
            try:
                qv = float(score)
                start_i = int(start)
            except ValueError:
                continue
            ipd_ratio = None
            coverage = None
            for kv in attrs.split(";"):
                if kv.startswith("IPDRatio="):
                    try:
                        ipd_ratio = float(kv.split("=", 1)[1])
                    except ValueError:
                        pass
                elif kv.startswith("coverage="):
                    try:
                        coverage = int(kv.split("=", 1)[1])
                    except ValueError:
                        pass
            rows.append({
                "seqid": seqid,
                "pos": start_i - 1,
                "strand": strand,
                "qv": qv,
                "ipd_ratio": ipd_ratio,
                "coverage": coverage,
                "mod_type": mtype,
            })
    return rows


def detect_ipd_tag(bam_path: Path) -> str:
    """Return 'ip' or 'fi' depending on which tag the first aligned read has."""
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped:
                continue
            if r.has_tag("ip"):
                return "ip"
            if r.has_tag("fi"):
                return "fi"
            break
    raise RuntimeError(f"{bam_path}: no 'ip' or 'fi' tag on first aligned read")


def collect_perread_ipd(
    bam: pysam.AlignmentFile,
    seqid: str,
    pos: int,
    strand: str,
    ipd_tag: str,
) -> np.ndarray:
    """Return uint8 array of per-read IPD values at the given ref position."""
    vals: list[int] = []
    for read in bam.fetch(seqid, pos, pos + 1):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        rs = "-" if read.is_reverse else "+"
        if rs != strand:
            continue
        if not read.has_tag(ipd_tag):
            continue
        ip_arr = read.get_tag(ipd_tag)
        pairs = read.get_aligned_pairs(matches_only=True)
        if not pairs:
            continue
        pair_arr = np.asarray(pairs, dtype=np.int64)
        hit = pair_arr[pair_arr[:, 1] == pos]
        if hit.size == 0:
            continue
        q = int(hit[0, 0])
        if 0 <= q < len(ip_arr):
            vals.append(int(ip_arr[q]))
    return np.asarray(vals, dtype=np.int16)


def baseline_window_median(
    bam: pysam.AlignmentFile,
    seqid: str,
    pos: int,
    strand: str,
    ipd_tag: str,
    window: int = 500,
    exclude: int = 5,
) -> float:
    """Median of per-read IPD across all positions in [pos-window, pos+window]
    EXCLUDING the [pos-exclude, pos+exclude] central band — quick proxy for the
    unmethylated baseline at this locus."""
    vals: list[int] = []
    lo, hi = max(0, pos - window), pos + window
    for read in bam.fetch(seqid, lo, hi):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        rs = "-" if read.is_reverse else "+"
        if rs != strand:
            continue
        if not read.has_tag(ipd_tag):
            continue
        ip_arr = read.get_tag(ipd_tag)
        pairs = read.get_aligned_pairs(matches_only=True)
        if not pairs:
            continue
        pair_arr = np.asarray(pairs, dtype=np.int64)
        mask = (
            (pair_arr[:, 1] >= lo)
            & (pair_arr[:, 1] < hi)
            & (np.abs(pair_arr[:, 1] - pos) > exclude)
        )
        kept = pair_arr[mask]
        for q, _r in kept:
            if 0 <= q < len(ip_arr):
                vals.append(int(ip_arr[int(q)]))
    return float(np.median(vals)) if vals else float("nan")


def pick_control_sites(
    bam: pysam.AlignmentFile,
    methylated_lookup: set[tuple[str, int, str]],
    n: int,
    seqids: list[str],
    chrom_lengths: dict[str, int],
    rng: random.Random,
) -> list[dict]:
    """Sample N random (seqid, pos, strand) tuples not in `methylated_lookup`."""
    out: list[dict] = []
    tries = 0
    max_tries = n * 200
    while len(out) < n and tries < max_tries:
        tries += 1
        seqid = rng.choice(seqids)
        L = chrom_lengths[seqid]
        if L < 200:
            continue
        pos = rng.randint(100, L - 100)
        strand = rng.choice(["+", "-"])
        if (seqid, pos, strand) in methylated_lookup:
            continue
        out.append({
            "seqid": seqid, "pos": pos, "strand": strand,
            "qv": 0.0, "ipd_ratio": None, "mod_type": "control",
        })
    return out


def plot_grid(sites: list[dict], per_site_vals: list[np.ndarray],
              per_site_baseline: list[float], title: str, output_png: Path) -> None:
    """4×5 grid of per-site IPD histograms."""
    n = min(len(sites), 20)
    fig, axes = plt.subplots(4, 5, figsize=(20, 14), squeeze=False)
    for i in range(20):
        ax = axes[i // 5, i % 5]
        if i >= n:
            ax.axis("off")
            continue
        s = sites[i]
        vals = per_site_vals[i]
        base = per_site_baseline[i]
        if vals.size == 0:
            ax.text(0.5, 0.5, "no reads", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{s['seqid']}:{s['pos']} ({s['strand']})  no data", fontsize=9)
            continue
        ax.hist(vals, bins=np.arange(0, 256, 4), color="#888", edgecolor="black", linewidth=0.3)
        med = float(np.median(vals))
        ax.axvline(med, color="#D55E00", linestyle="--", linewidth=1.5, label=f"med={med:.0f}")
        if np.isfinite(base):
            ax.axvline(base, color="#0072B2", linestyle="--", linewidth=1.5, label=f"bg={base:.0f}")
        ax.legend(loc="upper right", fontsize=7)
        ipd_r = s.get("ipd_ratio")
        ipd_r_s = f" ipdR={ipd_r:.2f}" if ipd_r is not None else ""
        ax.set_title(
            f"{s['seqid']}:{s['pos']} ({s['strand']})\nQV={s['qv']:.0f}{ipd_r_s}  n={vals.size}",
            fontsize=9,
        )
        ax.set_xlim(0, 256)
        ax.set_xlabel("per-read IPD (uint8)")
        ax.set_ylabel("count")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(output_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--mod-type", default="m6A",
                    help="GFF type column filter (m6A / m4C / modified_base). Default m6A.")
    ap.add_argument("--n-sites", type=int, default=20)
    ap.add_argument("--min-qv-control", type=float, default=10.0,
                    help="GFF rows with QV ≥ this are excluded from control sampling")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    bam_path, gff_path = Path(args.bam), Path(args.gff)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Parsing GFF: {gff_path}")
    gff_rows = parse_gff(gff_path, args.mod_type)
    print(f"  {len(gff_rows)} rows of type={args.mod_type!r}")
    if not gff_rows:
        sys.exit(f"No GFF rows of type={args.mod_type!r}")

    gff_rows.sort(key=lambda r: r["qv"], reverse=True)
    top_sites = gff_rows[: args.n_sites]
    print(f"  top QV range: {top_sites[-1]['qv']:.1f} to {top_sites[0]['qv']:.1f}")

    print(f"Detecting IPD tag on {bam_path} ...")
    ipd_tag = detect_ipd_tag(bam_path)
    print(f"  tag = {ipd_tag!r}")

    bam = pysam.AlignmentFile(str(bam_path), "rb")
    chrom_lengths = dict(zip(bam.references, bam.lengths))

    print(f"Collecting per-read IPD at {len(top_sites)} called sites ...")
    meth_vals: list[np.ndarray] = []
    meth_base: list[float] = []
    for i, s in enumerate(top_sites):
        v = collect_perread_ipd(bam, s["seqid"], s["pos"], s["strand"], ipd_tag)
        b = baseline_window_median(bam, s["seqid"], s["pos"], s["strand"], ipd_tag)
        meth_vals.append(v)
        meth_base.append(b)
        print(f"  [{i+1:>2}/{len(top_sites)}] {s['seqid']}:{s['pos']} ({s['strand']})  "
              f"QV={s['qv']:.0f}  n_reads={v.size}  med={float(np.median(v)) if v.size else 0:.1f}  bg={b:.1f}")

    methylated_lookup = {
        (r["seqid"], r["pos"], r["strand"])
        for r in gff_rows if r["qv"] >= args.min_qv_control
    }
    seqids_avail = [s for s in bam.references if chrom_lengths[s] > 1000]
    if not seqids_avail:
        sys.exit("No chromosomes with length > 1000 in BAM")
    print(f"Sampling {args.n_sites} control sites (QV < {args.min_qv_control}) ...")
    control_sites = pick_control_sites(
        bam, methylated_lookup, args.n_sites, seqids_avail, chrom_lengths, rng,
    )
    ctrl_vals: list[np.ndarray] = []
    ctrl_base: list[float] = []
    for i, s in enumerate(control_sites):
        v = collect_perread_ipd(bam, s["seqid"], s["pos"], s["strand"], ipd_tag)
        b = baseline_window_median(bam, s["seqid"], s["pos"], s["strand"], ipd_tag)
        ctrl_vals.append(v)
        ctrl_base.append(b)
        print(f"  [{i+1:>2}/{len(control_sites)}] {s['seqid']}:{s['pos']} ({s['strand']})  "
              f"n_reads={v.size}  med={float(np.median(v)) if v.size else 0:.1f}  bg={b:.1f}")

    bam.close()

    plot_grid(
        top_sites, meth_vals, meth_base,
        title=f"Per-read IPD at top-{args.n_sites} ipdSummary {args.mod_type} sites  ({bam_path.name})",
        output_png=Path(f"{out_prefix}_methylated.png"),
    )
    plot_grid(
        control_sites, ctrl_vals, ctrl_base,
        title=f"Per-read IPD at {args.n_sites} random non-{args.mod_type} positions  ({bam_path.name})",
        output_png=Path(f"{out_prefix}_control.png"),
    )

    all_meth = np.concatenate([v for v in meth_vals if v.size]) if meth_vals else np.array([])
    all_ctrl = np.concatenate([v for v in ctrl_vals if v.size]) if ctrl_vals else np.array([])
    print()
    if all_meth.size:
        print(f"  pooled methylated  n={all_meth.size}  med={np.median(all_meth):.1f}  "
              f"mean={all_meth.mean():.1f}  σ={all_meth.std():.1f}")
    if all_ctrl.size:
        print(f"  pooled control     n={all_ctrl.size}  med={np.median(all_ctrl):.1f}  "
              f"mean={all_ctrl.mean():.1f}  σ={all_ctrl.std():.1f}")


if __name__ == "__main__":
    main()
