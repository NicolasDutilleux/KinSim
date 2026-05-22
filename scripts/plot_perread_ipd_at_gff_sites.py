"""Per-read IPD distribution at the top-N ipdSummary-called positions.

For each of the top-N GFF rows of a given ``--mod-type`` (sorted by QV), open
the aligned BAM, collect every per-read ``ip`` (or ``fi``) value at that exact
reference position, and plot a 4×5 grid of histograms. One PNG, plus a pooled
summary printed at the end.

Usage::

    python scripts/plot_perread_ipd_at_gff_sites.py \\
        --bam <aligned.bam> --gff <ipdSummary.gff> \\
        --output <out.png> [--mod-type m6A] [--n-sites 20]

Strand handling: bystrandified BAMs split each ZMW into ``ccs/fwd`` and
``ccs/rev`` records. We accept reads whose orientation matches the GFF strand
(``+`` → ``not read.is_reverse``; ``-`` → ``read.is_reverse``). Tag autodetect:
``ip`` first, ``fi`` fallback.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pysam


def parse_gff_top(gff_path: Path, mod_type: str, n: int) -> list[dict]:
    rows: list[dict] = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != mod_type:
                continue
            try:
                qv = float(parts[5])
                start = int(parts[3]) - 1
            except ValueError:
                continue
            ipd_ratio = None
            coverage = None
            for kv in parts[8].split(";"):
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
                "seqid": parts[0], "pos": start, "strand": parts[6],
                "qv": qv, "ipd_ratio": ipd_ratio, "coverage": coverage,
            })
    rows.sort(key=lambda r: r["qv"], reverse=True)
    return rows[:n]


def detect_ipd_tag(bam_path: Path) -> str:
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


def collect_perread(bam: pysam.AlignmentFile, seqid: str, pos: int,
                    strand: str, ipd_tag: str) -> np.ndarray:
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mod-type", default="m6A")
    ap.add_argument("--n-sites", type=int, default=20)
    args = ap.parse_args(argv)

    bam_path, gff_path = Path(args.bam), Path(args.gff)
    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    print(f"GFF: {gff_path}")
    top = parse_gff_top(gff_path, args.mod_type, args.n_sites)
    print(f"  type={args.mod_type!r}: kept top {len(top)} by QV "
          f"(range {top[-1]['qv']:.1f} → {top[0]['qv']:.1f})" if top else "  no rows match")
    if not top:
        sys.exit("no GFF rows after filter")

    ipd_tag = detect_ipd_tag(bam_path)
    print(f"BAM: {bam_path}\n  tag = {ipd_tag!r}")

    per_site: list[np.ndarray] = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for i, s in enumerate(top):
            v = collect_perread(bam, s["seqid"], s["pos"], s["strand"], ipd_tag)
            per_site.append(v)
            med = float(np.median(v)) if v.size else 0.0
            print(f"  [{i+1:>2}/{len(top)}] {s['seqid']}:{s['pos']} ({s['strand']})  "
                  f"QV={s['qv']:>5.0f}  ipdR={s.get('ipd_ratio') or 0:.2f}  "
                  f"n_reads={v.size:>4}  med={med:>5.1f}")

    n = len(top)
    fig, axes = plt.subplots(4, 5, figsize=(20, 14), squeeze=False)
    for i in range(20):
        ax = axes[i // 5, i % 5]
        if i >= n:
            ax.axis("off")
            continue
        s = top[i]
        vals = per_site[i]
        if vals.size == 0:
            ax.text(0.5, 0.5, "no reads", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{s['seqid']}:{s['pos']} ({s['strand']})", fontsize=9)
            continue
        ax.hist(vals, bins=np.arange(0, 256, 4),
                color="#888", edgecolor="black", linewidth=0.3)
        med = float(np.median(vals))
        ax.axvline(med, color="#D55E00", linestyle="--", linewidth=1.5,
                   label=f"med={med:.0f}")
        ax.legend(loc="upper right", fontsize=8)
        ipd_r = s.get("ipd_ratio")
        ipd_r_s = f"  ipdR={ipd_r:.2f}" if ipd_r is not None else ""
        ax.set_title(f"{s['seqid']}:{s['pos']} ({s['strand']})\n"
                     f"QV={s['qv']:.0f}{ipd_r_s}  n={vals.size}", fontsize=9)
        ax.set_xlim(0, 256)
        ax.set_xlabel("per-read IPD (uint8)")
        ax.set_ylabel("count")

    pooled = np.concatenate([v for v in per_site if v.size]) if per_site else np.array([])
    if pooled.size:
        footer = (
            f"Pooled across {sum(1 for v in per_site if v.size)} sites "
            f"({pooled.size} reads): "
            f"median={np.median(pooled):.1f}  mean={pooled.mean():.1f}  "
            f"σ={pooled.std():.1f}  "
            f"q25={np.percentile(pooled, 25):.1f}  q75={np.percentile(pooled, 75):.1f}  "
            f"q95={np.percentile(pooled, 95):.1f}"
        )
    else:
        footer = "Pooled: no reads"

    fig.suptitle(
        f"Per-read IPD at top-{n} ipdSummary {args.mod_type} sites — {bam_path.name}",
        fontsize=13,
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#eee", edgecolor="#888"))
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_png}")
    print(f"\n{footer}")


if __name__ == "__main__":
    main()
