"""Quick 3D density plot for methylation type separation.

Loads a merged .pkl and plots 3D density surfaces (IPD x PW x density)
per methylation type, with a minimum IPD cutoff for m6A clarity.

Usage:
    python scripts/plot_3d_meth.py <master_data.pkl> [--output plot.html] [--min-ipd-m6a 70]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Add repo root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinsim.utils.encoding import METH_IDS

METH_NAMES = {v: k for k, v in METH_IDS.items()}

# Per-type colors (solid, not washed out)
COLORS = {
    0: "#3366CC",   # none  — blue
    1: "#DC3912",   # m6A   — red
    2: "#FF9900",   # m4C   — orange
    3: "#109618",   # m5C   — green
}


def load_per_key_means(pkl_path: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load pkl, compute per-key (kmer, meth) mean IPD/PW.

    Returns: {meth_id: (ipd_means, pw_means)} arrays.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    collectors: dict[int, tuple[list, list]] = {}
    for key, arr in data.items():
        if not isinstance(key, tuple):
            continue
        _kmer_id, meth_id = key
        if not isinstance(arr, np.ndarray) or len(arr) == 0:
            continue
        mu_ipd = float(arr[:, 0].mean())
        mu_pw = float(arr[:, 1].mean())
        collectors.setdefault(meth_id, ([], []))
        collectors[meth_id][0].append(mu_ipd)
        collectors[meth_id][1].append(mu_pw)

    return {
        mid: (np.array(ipds, dtype=np.float64), np.array(pws, dtype=np.float64))
        for mid, (ipds, pws) in collectors.items()
    }


def build_figure(
    groups: dict[int, tuple[np.ndarray, np.ndarray]],
    min_ipd_m6a: float = 70.0,
    grid_n: int = 100,
):
    """Build a plotly 3D figure with solid, distinguishable surfaces."""
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde

    # Determine grid bounds from all data
    all_ipd = np.concatenate([g[0] for g in groups.values()])
    all_pw = np.concatenate([g[1] for g in groups.values()])
    ipd_hi = max(float(np.percentile(all_ipd, 99.5)), 150.0)
    pw_hi = max(float(np.percentile(all_pw, 99.5)), 80.0)

    ipd_grid = np.linspace(0.0, ipd_hi, grid_n)
    pw_grid = np.linspace(0.0, pw_hi, grid_n)
    ipd_mesh, pw_mesh = np.meshgrid(ipd_grid, pw_grid)
    grid_pts = np.vstack([ipd_mesh.ravel(), pw_mesh.ravel()])

    fig = go.Figure()

    meth_order = [0, 3, 2, 1]  # Plot none first (bottom), m6A last (on top)

    for mid in meth_order:
        if mid not in groups:
            continue
        ipd_vals, pw_vals = groups[mid]
        name = METH_NAMES.get(mid, f"meth{mid}")

        # Apply m6A IPD cutoff — only keep keys with mean IPD >= threshold
        if mid == 1 and min_ipd_m6a > 0:
            mask = ipd_vals >= min_ipd_m6a
            ipd_vals = ipd_vals[mask]
            pw_vals = pw_vals[mask]
            n_filtered = int((~mask).sum())
            if n_filtered > 0:
                print(f"  m6A: filtered {n_filtered} keys below IPD={min_ipd_m6a} "
                      f"({len(ipd_vals)} remaining)")

        n = len(ipd_vals)
        if n < 50:
            print(f"  {name}: only {n} keys, skipping")
            continue

        # Subsample for KDE
        max_kde = 50_000
        if n > max_kde:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, max_kde, replace=False)
            ipd_s, pw_s = ipd_vals[idx], pw_vals[idx]
        else:
            ipd_s, pw_s = ipd_vals, pw_vals

        print(f"  {name}: {n:,} keys, KDE on {len(ipd_s):,}")

        try:
            kde = gaussian_kde(np.vstack([ipd_s, pw_s]), bw_method=0.15)
            z = kde(grid_pts).reshape(grid_n, grid_n)
        except np.linalg.LinAlgError:
            print(f"  {name}: KDE failed, skipping")
            continue

        # Normalize each surface to [0, 1] for consistent visual height
        z_max = z.max()
        if z_max > 0:
            z_norm = z / z_max
        else:
            z_norm = z

        color = COLORS.get(mid, "#999999")

        # Use a two-tone colorscale: dark base -> bright peak
        # This gives solid, readable surfaces without transparency
        fig.add_trace(go.Surface(
            x=ipd_grid,
            y=pw_grid,
            z=z_norm,
            name=f"{name} (n={n:,})",
            showscale=False,
            opacity=1.0,
            colorscale=[
                [0.0, _darken(color, 0.3)],
                [0.3, _darken(color, 0.6)],
                [1.0, color],
            ],
            showlegend=True,
            legendgroup=name,
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=False,
                       highlightcolor="white", highlightwidth=1),
            ),
        ))

    fig.update_layout(
        title=dict(
            text=(f"3D Density: IPD x PW per Methylation Type"
                  f"<br><sub>m6A filtered: IPD >= {min_ipd_m6a}</sub>"),
            x=0.5,
        ),
        scene=dict(
            xaxis_title="Mean IPD (raw)",
            yaxis_title="Mean PW (raw)",
            zaxis_title="Normalized Density",
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.0)),
            bgcolor="rgb(240, 240, 240)",
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            font=dict(size=13),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        width=1000,
        height=750,
    )

    return fig


def _darken(hex_color: str, factor: float) -> str:
    """Darken a hex color by factor (0=black, 1=original)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"rgb({r},{g},{b})"


def main():
    parser = argparse.ArgumentParser(
        description="3D density plot of methylation type separation"
    )
    parser.add_argument("pkl", help="Merged .pkl file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HTML file (default: <pkl_stem>_3d_meth.html)")
    parser.add_argument("--min-ipd-m6a", type=float, default=70.0,
                        help="Minimum mean IPD for m6A keys (default: 70)")
    parser.add_argument("--grid", type=int, default=100,
                        help="Grid resolution (default: 100)")
    args = parser.parse_args()

    print(f"Loading {args.pkl}...")
    groups = load_per_key_means(args.pkl)

    for mid, (ipds, pws) in sorted(groups.items()):
        name = METH_NAMES.get(mid, f"meth{mid}")
        print(f"  {name}: {len(ipds):,} keys  "
              f"IPD [{ipds.min():.1f} - {ipds.max():.1f}]  "
              f"PW [{pws.min():.1f} - {pws.max():.1f}]")

    print(f"\nBuilding 3D plot (grid={args.grid}, m6A cutoff={args.min_ipd_m6a})...")
    fig = build_figure(groups, min_ipd_m6a=args.min_ipd_m6a, grid_n=args.grid)

    output = args.output or str(Path(args.pkl).stem + "_3d_meth.html")
    fig.write_html(output)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
