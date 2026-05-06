"""Extract reads from a BAM, plot 3D density of IPD x PW per methylation type.

Reads directly from a BAM + motifs CSV. No .pkl needed.
Methylation labels come from motif scanning (sequence-based).
Optional IPD floor filter for m6A keys removes noisy low-signal keys.

Usage:
    python scripts/plot_3d_from_bam.py <bam> <motifs> [--max-reads 5000] [--min-ipd-m6a 70] [-o plot.html]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinsim.utils.encoding import BASE_MAP, KMER_MASK, METH_IDS, K
from kinsim.utils.motifs import load_motif_string, parse_motifs, scan_sequence

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')

METH_NAMES = {v: k for k, v in METH_IDS.items()}


# ── Extract from BAM ────────────────────────────────────────────────────────

def extract_from_bam(bam_path: str, motif_string: str, max_reads: int = 5000) -> dict:
    """Extract per-(kmer, meth_id) samples from a BAM.

    Returns: {(kmer_id, meth_id): np.ndarray(N, 2) with [IPD, PW]}
    """
    import pysam

    motifs = parse_motifs(motif_string, revcomp=True)
    result: dict[tuple, list] = {}
    n_reads = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if max_reads > 0 and n_reads >= max_reads:
                break
            seq = read.query_sequence
            if not (seq and len(seq) >= K and read.has_tag("fi")):
                continue

            ipds = read.get_tag("fi")
            pws = read.get_tag("fp")
            min_len = min(len(seq), len(ipds), len(pws))
            meth_status = scan_sequence(seq[:min_len], motifs)

            mid = K // 2
            current_kmer = 0
            for i in range(min_len):
                base_val = BASE_MAP.get(seq[i], -1)
                if base_val < 0:
                    current_kmer = 0
                    continue
                current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK
                if i >= K - 1:
                    center = i - mid
                    meth_id = int(meth_status[center])
                    key = (current_kmer, meth_id)
                    result.setdefault(key, []).append([float(ipds[center]), float(pws[center])])

            n_reads += 1
            if n_reads % 1000 == 0:
                log.info("  %d reads processed...", n_reads)

    log.info("Extracted %d reads, %d keys", n_reads, len(result))

    # Convert to arrays
    return {k: np.array(v, dtype=np.float32) for k, v in result.items()}


# ── Build per-type mean arrays ──────────────────────────────────────────────

def compute_group_means(data: dict, min_ipd_m6a: float = 0.0) -> dict[int, tuple]:
    """Returns {meth_id: (ipd_means, pw_means)} from per-key means."""
    collectors: dict[int, tuple[list, list]] = {}
    for (_kmer_id, meth_id), arr in data.items():
        mu_ipd = float(arr[:, 0].mean())
        mu_pw = float(arr[:, 1].mean())

        # Apply m6A IPD floor
        if meth_id == 1 and min_ipd_m6a > 0 and mu_ipd < min_ipd_m6a:
            continue

        collectors.setdefault(meth_id, ([], []))
        collectors[meth_id][0].append(mu_ipd)
        collectors[meth_id][1].append(mu_pw)

    return {
        mid: (np.array(ipds), np.array(pws))
        for mid, (ipds, pws) in collectors.items()
    }


# ── Plot 3D ─────────────────────────────────────────────────────────────────

COLORS = {
    0: "#636EFA",   # none — blue
    1: "#EF553B",   # m6A  — red
    2: "#00CC96",   # m4C  — green
    3: "#AB63FA",   # m5C  — purple
}


def _darken(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f'rgb({int(r*factor)},{int(g*factor)},{int(b*factor)})'


def build_figure(groups: dict, min_ipd_m6a: float, grid_n: int = 100):
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde

    all_ipd = np.concatenate([g[0] for g in groups.values()])
    all_pw = np.concatenate([g[1] for g in groups.values()])
    ipd_hi = max(float(np.percentile(all_ipd, 99.5)), 150.0)
    pw_hi = max(float(np.percentile(all_pw, 99.5)), 80.0)

    ipd_grid = np.linspace(0.0, ipd_hi, grid_n)
    pw_grid = np.linspace(0.0, pw_hi, grid_n)
    ipd_mesh, pw_mesh = np.meshgrid(ipd_grid, pw_grid)
    grid_pts = np.vstack([ipd_mesh.ravel(), pw_mesh.ravel()])

    fig = go.Figure()

    # Plot none first (bottom), methylated last (on top)
    meth_order = [0, 3, 2, 1]

    for mid in meth_order:
        if mid not in groups:
            continue
        ipd_vals, pw_vals = groups[mid]
        name = METH_NAMES.get(mid, f"meth{mid}")
        n = len(ipd_vals)

        if n < 30:
            log.info("  %s: only %d keys, skipping", name, n)
            continue

        max_kde = 50_000
        if n > max_kde:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, max_kde, replace=False)
            ipd_s, pw_s = ipd_vals[idx], pw_vals[idx]
        else:
            ipd_s, pw_s = ipd_vals, pw_vals

        log.info("  %s: %d keys, KDE on %d", name, n, len(ipd_s))

        try:
            kde = gaussian_kde(np.vstack([ipd_s, pw_s]), bw_method=0.15)
            z = kde(grid_pts).reshape(grid_n, grid_n)
        except np.linalg.LinAlgError:
            log.warning("  %s: KDE failed", name)
            continue

        z_max = z.max()
        if z_max > 0:
            z = z / z_max

        color = COLORS.get(mid, "#999999")
        fig.add_trace(go.Surface(
            x=ipd_grid, y=pw_grid, z=z,
            name=f'{name} (n={n:,})', showscale=False,
            opacity=1.0,
            colorscale=[
                [0.0, _darken(color, 0.3)],
                [0.3, _darken(color, 0.6)],
                [1.0, color],
            ],
            showlegend=True,
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=False,
                       highlightcolor='white', highlightwidth=1),
            ),
        ))

    title = '3D Density: IPD x PW per Methylation Type'
    if min_ipd_m6a > 0:
        title += f'<br><sub>m6A filtered: mean IPD >= {min_ipd_m6a:.0f}</sub>'

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='Mean IPD (raw)',
            yaxis_title='Mean PW (raw)',
            zaxis_title='Normalized Density',
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.0)),
            bgcolor='rgb(240, 240, 240)',
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1,
            font=dict(size=13),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        width=1000, height=750,
    )
    return fig


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract from BAM + 3D density plot per methylation type"
    )
    parser.add_argument("bam", help="PacBio HiFi BAM with fi/fp tags")
    parser.add_argument("motifs", help="Motif CSV path or motif string")
    parser.add_argument("--max-reads", type=int, default=5000,
                        help="Max reads to process (default: 5000)")
    parser.add_argument("--min-ipd-m6a", type=float, default=70.0,
                        help="Min mean IPD for m6A keys (default: 70)")
    parser.add_argument("--grid", type=int, default=100)
    parser.add_argument("-o", "--output", default="bam_3d_density.html",
                        help="Output file (.html or .png)")
    args = parser.parse_args()

    log.info("Resolving motifs: %s", args.motifs)
    motif_string = load_motif_string(args.motifs)
    log.info("Motif string: %s", motif_string[:200])

    log.info("Extracting from BAM: %s (max %d reads)", args.bam, args.max_reads)
    raw = extract_from_bam(args.bam, motif_string, max_reads=args.max_reads)

    for mid in sorted(set(k[1] for k in raw)):
        n_keys = sum(1 for k in raw if k[1] == mid)
        name = METH_NAMES.get(mid, f"meth{mid}")
        log.info("  Extracted: %s = %d keys", name, n_keys)

    log.info("Computing per-key means (m6A IPD >= %.0f)...", args.min_ipd_m6a)
    groups = compute_group_means(raw, min_ipd_m6a=args.min_ipd_m6a)

    for mid, (ipds, pws) in sorted(groups.items()):
        name = METH_NAMES.get(mid, f"meth{mid}")
        n_filtered = sum(1 for k in raw if k[1] == mid) - len(ipds) if mid == 1 else 0
        log.info("  %s: %d keys  IPD [%.1f - %.1f]  PW [%.1f - %.1f]%s",
                 name, len(ipds), ipds.min(), ipds.max(), pws.min(), pws.max(),
                 f"  ({n_filtered} filtered below IPD={args.min_ipd_m6a})" if n_filtered else "")

    log.info("Building 3D plot...")
    fig = build_figure(groups, min_ipd_m6a=args.min_ipd_m6a, grid_n=args.grid)

    ext = Path(args.output).suffix.lower()
    if ext in (".png", ".jpg", ".svg", ".pdf"):
        fig.write_image(args.output, scale=3)
    else:
        fig.write_html(args.output)
    log.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
