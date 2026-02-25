"""Comprehensive analysis of a trained 11-mer kinetic dictionary (.pkl).

Generates:
  - TXT report (printed to stdout AND written to <basename>_report.txt)
  - Interactive HTML report with Plotly visualizations (requires: pip install -e .[analyze])

Analysis sections:
  1. Overview — file size, total entries / samples, overall coverage
  2. Per-methylation-state coverage — unique 11-mers, %, sample count distributions
  3. Signal statistics — IPD/PW mean and sigma distributions per meth type
  4. Low-coverage warnings — entries with n < 5 / 10 / 50
  5. 1-base neighbor sensitivity — how much does a single base change in the
     11-mer context affect IPD/PW?  Computed separately for methylated and
     unmethylated kmers; includes per-position breakdown (which of the 11
     positions in the context window matters most).

HTML visualizations:
  - Dashboard stat cards
  - Coverage bar chart
  - Sample count histogram (log scale)
  - IPD mean violin+box per meth type
  - PW mean violin+box per meth type
  - Signal sigma distributions (IPD + PW)
  - IPD vs PW scatter (WebGL, subsampled)
  - Neighbor sensitivity histograms  (|ΔIPD| and |ΔPW|, meth vs unmeth)
  - Per-position sensitivity profile (grouped bar)
  - Meth vs unmeth sensitivity box comparison
"""

import os
import sys
import pickle
import time
import dataclasses
import io
import numpy as np

from ..encoding import (get_ipd_stats, get_pw_stats,
                        TOTAL_POSSIBLE_KMERS, METH_IDS, K)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLORS = {0: '#636EFA', 1: '#EF553B', 2: '#00CC96', 3: '#AB63FA'}
_LABELS = {0: 'Unmethylated', 1: 'm6A', 2: 'm4C', 3: 'm5C'}
_ID_TO_NAME = {v: k for k, v in METH_IDS.items()}

# For violin subsampling — Plotly KDE becomes slow above ~100K points per trace
_VIOLIN_SUBSAMPLE = 100_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MethGroupStats:
    """Statistics collected for one methylation state."""
    meth_id: int
    name: str                    # "none", "m6A", "m4C", "m5C"
    n_entries: int
    sample_counts: np.ndarray    # (n_entries,)  — n per accumulator
    ipd_means: np.ndarray        # (n_entries,)  — mu_ipd per kmer
    ipd_sigmas: np.ndarray       # (n_entries,)  — sigma_ipd per kmer
    pw_means: np.ndarray         # (n_entries,)  — mu_pw per kmer
    pw_sigmas: np.ndarray        # (n_entries,)  — sigma_pw per kmer
    kmer_ids: np.ndarray         # (n_entries,) int32 — for neighbor lookup


@dataclasses.dataclass
class DictStats:
    """All statistics collected from one dictionary."""
    pkl_path: str
    total_entries: int
    total_samples: float
    groups: dict                 # meth_id -> MethGroupStats
    file_size_mb: float


@dataclasses.dataclass
class NeighborSensitivity:
    """1-base neighbor sensitivity results for one category (meth or unmeth)."""
    delta_ipd: np.ndarray        # |delta mean_ipd| for each neighbor pair found
    delta_pw: np.ndarray         # |delta mean_pw| for each neighbor pair found
    positions: np.ndarray        # which of the 11 positions was mutated (0-10)
    n_source_entries: int        # how many source kmers were analysed
    n_pairs_found: int           # how many neighbor pairs had both kmers in dict


# ---------------------------------------------------------------------------
# Step 1: collect_stats  (single pass over the dictionary)
# ---------------------------------------------------------------------------

def collect_stats(lookup: dict, pkl_path: str) -> DictStats:
    """One-pass collection of all per-kmer statistics needed for both reports."""
    # Group items by meth_id
    partitions: dict = {}
    for (kmer, meth_id), acc in lookup.items():
        if meth_id not in partitions:
            partitions[meth_id] = []
        partitions[meth_id].append((kmer, acc))

    groups = {}
    for meth_id, items in partitions.items():
        n = len(items)
        sample_counts = np.empty(n, dtype=np.float64)
        ipd_means     = np.empty(n, dtype=np.float64)
        ipd_sigmas    = np.empty(n, dtype=np.float64)
        pw_means      = np.empty(n, dtype=np.float64)
        pw_sigmas     = np.empty(n, dtype=np.float64)
        kmer_ids      = np.empty(n, dtype=np.int32)

        for i, (kmer, acc) in enumerate(items):
            sample_counts[i] = acc[0]
            mu_ipd, sig_ipd  = get_ipd_stats(acc)
            mu_pw,  sig_pw   = get_pw_stats(acc)
            ipd_means[i]     = mu_ipd
            ipd_sigmas[i]    = sig_ipd
            pw_means[i]      = mu_pw
            pw_sigmas[i]     = sig_pw
            kmer_ids[i]      = kmer

        name = _ID_TO_NAME.get(meth_id, f'id={meth_id}')
        groups[meth_id] = MethGroupStats(
            meth_id=meth_id, name=name, n_entries=n,
            sample_counts=sample_counts, ipd_means=ipd_means,
            ipd_sigmas=ipd_sigmas, pw_means=pw_means,
            pw_sigmas=pw_sigmas, kmer_ids=kmer_ids,
        )

    file_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    return DictStats(
        pkl_path=pkl_path,
        total_entries=len(lookup),
        total_samples=sum(acc[0] for acc in lookup.values()),
        groups=groups,
        file_size_mb=file_size_mb,
    )


# ---------------------------------------------------------------------------
# Step 2: compute_neighbor_sensitivity  O(n × 33)
# ---------------------------------------------------------------------------

def compute_neighbor_sensitivity(
    lookup: dict,
    stats: DictStats,
    max_entries: int = 500_000,
) -> dict:
    """Compute 1-base neighbor sensitivity for methylated and unmethylated kmers.

    For each source kmer, generate all 33 single-base substitutions (11 positions
    × 3 alternative bases) using bit manipulation.  For each neighbor found in
    the dictionary, record |delta_mean_ipd|, |delta_mean_pw|, and which position
    was mutated.

    Returns dict with keys 'methylated' and 'unmethylated'.
    """
    rng = np.random.default_rng(42)
    results = {}

    # category -> list of meth_ids to include
    categories = [
        ('unmethylated', [0]),
        ('methylated',   [mid for mid in stats.groups if mid != 0]),
    ]

    for category, meth_ids in categories:
        # Collect (kmer_int, meth_id, ipd_mean, pw_mean) for this category
        source = []
        for mid in meth_ids:
            if mid not in stats.groups:
                continue
            g = stats.groups[mid]
            for i in range(g.n_entries):
                source.append((int(g.kmer_ids[i]), mid,
                                float(g.ipd_means[i]), float(g.pw_means[i])))

        if not source:
            results[category] = NeighborSensitivity(
                delta_ipd=np.array([], dtype=np.float32),
                delta_pw=np.array([], dtype=np.float32),
                positions=np.array([], dtype=np.int8),
                n_source_entries=0, n_pairs_found=0)
            continue

        # Subsample if needed
        n_source = len(source)
        if max_entries > 0 and n_source > max_entries:
            idx = rng.choice(n_source, max_entries, replace=False)
            source = [source[i] for i in idx]

        delta_ipd_list = []
        delta_pw_list  = []
        pos_list       = []

        for kmer_int, meth_id, ipd_mean, pw_mean in source:
            for pos in range(K):
                bit_pos = (K - 1 - pos) * 2          # e.g. pos=0 → bit 20
                orig_base = (kmer_int >> bit_pos) & 0x3
                mask      = ~(0x3 << bit_pos)
                for new_base in range(4):
                    if new_base == orig_base:
                        continue
                    neighbor = (kmer_int & mask) | (new_base << bit_pos)
                    nb_acc = lookup.get((neighbor, meth_id))
                    if nb_acc is None:
                        continue
                    nb_ipd_mu, _ = get_ipd_stats(nb_acc)
                    nb_pw_mu,  _ = get_pw_stats(nb_acc)
                    delta_ipd_list.append(abs(ipd_mean - nb_ipd_mu))
                    delta_pw_list.append(abs(pw_mean  - nb_pw_mu))
                    pos_list.append(pos)

        results[category] = NeighborSensitivity(
            delta_ipd=np.array(delta_ipd_list, dtype=np.float32),
            delta_pw=np.array(delta_pw_list,   dtype=np.float32),
            positions=np.array(pos_list,        dtype=np.int8),
            n_source_entries=len(source),
            n_pairs_found=len(delta_ipd_list),
        )

    return results


# ---------------------------------------------------------------------------
# Step 3a: render_txt_report
# ---------------------------------------------------------------------------

def _pct(val, total):
    return f'{100.0 * val / total:.2f}%' if total else 'n/a'


def render_txt_report(stats: DictStats, sensitivity: dict, output_path: str):
    """Print TXT report to stdout and write to *output_path*."""
    buf = io.StringIO()

    def p(line=''):
        print(line)
        buf.write(line + '\n')

    W = 72  # report width

    # ── Section 1: Overview ──────────────────────────────────────────────
    p('=' * W)
    p('KinSim Dictionary Analysis Report')
    p('=' * W)
    p(f"Dictionary    : {stats.pkl_path}")
    p(f"File size     : {stats.file_size_mb:.1f} MB")
    p(f"Total entries : {stats.total_entries:,}")
    p(f"Total samples : {stats.total_samples:,.0f}")
    p(f"Possible kmers: {TOTAL_POSSIBLE_KMERS:,}  (4^{K})")
    p(f"Overall cov.  : {_pct(stats.total_entries, TOTAL_POSSIBLE_KMERS)}")
    p()

    # ── Section 2: Per-meth coverage table ───────────────────────────────
    p('-' * W)
    p('Per-methylation-state coverage')
    p('-' * W)
    for meth_id in sorted(stats.groups.keys()):
        g = stats.groups[meth_id]
        label = 'Unmethylated' if meth_id == 0 else f'Methylated ({g.name})'
        c = g.sample_counts
        p(f'\n  {label}:')
        p(f'    Unique 11-mers : {g.n_entries:,} / {TOTAL_POSSIBLE_KMERS:,}'
          f'  ({_pct(g.n_entries, TOTAL_POSSIBLE_KMERS)})')
        p(f'    Total samples  : {np.sum(c):,.0f}')
        p(f'    Sample count   : mean={np.mean(c):.1f}  median={np.median(c):.1f}')
        p(f'                     p5={np.percentile(c,5):.0f}'
          f'  p25={np.percentile(c,25):.0f}'
          f'  p75={np.percentile(c,75):.0f}'
          f'  p95={np.percentile(c,95):.0f}')
        p(f'                     min={np.min(c):.0f}  max={np.max(c):.0f}')
    p()

    # ── Section 3: Signal statistics ─────────────────────────────────────
    p('-' * W)
    p('Signal statistics  (distribution of per-kmer means and sigmas)')
    p('-' * W)
    for meth_id in sorted(stats.groups.keys()):
        g = stats.groups[meth_id]
        label = 'Unmethylated' if meth_id == 0 else f'Methylated ({g.name})'
        p(f'\n  {label}:')
        for sig_name, means, sigmas in [
            ('IPD', g.ipd_means, g.ipd_sigmas),
            ('PW',  g.pw_means,  g.pw_sigmas),
        ]:
            p(f'    {sig_name} mean   :'
              f' mean={np.mean(means):.3f}'
              f'  std={np.std(means):.3f}'
              f'  median={np.median(means):.3f}'
              f'  p5={np.percentile(means,5):.3f}'
              f'  p95={np.percentile(means,95):.3f}')
            p(f'    {sig_name} sigma  :'
              f' mean={np.mean(sigmas):.3f}'
              f'  std={np.std(sigmas):.3f}'
              f'  median={np.median(sigmas):.3f}'
              f'  p5={np.percentile(sigmas,5):.3f}'
              f'  p95={np.percentile(sigmas,95):.3f}')
    p()

    # ── Section 4: Low-coverage warnings ─────────────────────────────────
    p('-' * W)
    p('Low-coverage entries  (may produce unreliable signals during injection)')
    p('-' * W)
    for meth_id in sorted(stats.groups.keys()):
        g = stats.groups[meth_id]
        label = 'Unmethylated' if meth_id == 0 else f'Methylated ({g.name})'
        n5  = int(np.sum(g.sample_counts < 5))
        n10 = int(np.sum(g.sample_counts < 10))
        n50 = int(np.sum(g.sample_counts < 50))
        ne  = g.n_entries
        p(f'  {label:25s}  n<5: {n5:,} ({_pct(n5,ne)})'
          f'   n<10: {n10:,} ({_pct(n10,ne)})'
          f'   n<50: {n50:,} ({_pct(n50,ne)})')
    p()

    # ── Section 5: 1-base neighbor sensitivity ────────────────────────────
    p('-' * W)
    p('1-base neighbor sensitivity analysis')
    p('-' * W)
    p('  How much does a single base change in the 11-mer context affect')
    p('  the expected (mean) IPD / PW signal?')
    p()
    for category in ['unmethylated', 'methylated']:
        ns = sensitivity.get(category)
        if ns is None or ns.n_pairs_found == 0:
            p(f'  {category.capitalize()}: no neighbor pairs found (skipping)')
            continue
        p(f'  {category.capitalize()} '
          f'({ns.n_source_entries:,} source kmers analysed, '
          f'{ns.n_pairs_found:,} neighbor pairs):')
        p(f'    |Δ IPD mean|  : mean={np.mean(ns.delta_ipd):.4f}'
          f'  median={np.median(ns.delta_ipd):.4f}'
          f'  p75={np.percentile(ns.delta_ipd,75):.4f}'
          f'  p95={np.percentile(ns.delta_ipd,95):.4f}')
        p(f'    |Δ PW  mean|  : mean={np.mean(ns.delta_pw):.4f}'
          f'  median={np.median(ns.delta_pw):.4f}'
          f'  p75={np.percentile(ns.delta_pw,75):.4f}'
          f'  p95={np.percentile(ns.delta_pw,95):.4f}')

        # Per-position summary
        ipd_by_pos = []
        pw_by_pos  = []
        for pos in range(K):
            mask = ns.positions == pos
            if np.any(mask):
                ipd_by_pos.append(f'{pos}:{np.mean(ns.delta_ipd[mask]):.3f}')
                pw_by_pos.append( f'{pos}:{np.mean(ns.delta_pw[mask]):.3f}')
            else:
                ipd_by_pos.append(f'{pos}:n/a')
                pw_by_pos.append( f'{pos}:n/a')
        p(f'    Per-pos mean |Δ IPD|  '
          f'(pos 0=5\' flank, 5=center/modified base, 10=3\' flank):')
        p(f'      {", ".join(ipd_by_pos)}')
        p(f'    Per-pos mean |Δ PW|:')
        p(f'      {", ".join(pw_by_pos)}')
        p()

    p('=' * W)

    # Write file
    with open(output_path, 'w') as fh:
        fh.write(buf.getvalue())
    print(f'\nTXT report saved to: {output_path}')


# ---------------------------------------------------------------------------
# Step 3b: render_html_report  (requires plotly)
# ---------------------------------------------------------------------------

def render_html_report(
    stats: DictStats,
    sensitivity: dict,
    output_path: str,
    max_scatter: int = 10_000,
):
    """Generate a self-contained interactive HTML report using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print('WARNING: plotly not installed — skipping HTML report.')
        print('  Install with: pip install plotly  or  pip install -e .[analyze]')
        return

    rng = np.random.default_rng(0)
    figures = []   # list of (section_title, go.Figure)

    meth_ids_sorted = sorted(stats.groups.keys())

    # ── Figure 1: Coverage bar chart ─────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[_LABELS.get(m, m) for m in meth_ids_sorted],
        y=[stats.groups[m].n_entries for m in meth_ids_sorted],
        marker_color=[_COLORS.get(m, '#888') for m in meth_ids_sorted],
        text=[f'{_pct(stats.groups[m].n_entries, TOTAL_POSSIBLE_KMERS)}'
              for m in meth_ids_sorted],
        textposition='auto',
        hovertemplate='%{x}<br>Unique 11-mers: %{y:,}<extra></extra>',
    ))
    fig.update_layout(
        title=f'11-mer Coverage per Methylation State '
              f'(total possible: {TOTAL_POSSIBLE_KMERS:,})',
        yaxis_title='Unique 11-mers in dictionary',
        xaxis_title='Methylation state',
    )
    figures.append(('Coverage', fig))

    # ── Figure 2: Sample count distribution (log-scale histogram) ────────
    fig = go.Figure()
    for m in meth_ids_sorted:
        g = stats.groups[m]
        fig.add_trace(go.Histogram(
            x=g.sample_counts,
            name=_LABELS.get(m, m),
            marker_color=_COLORS.get(m, '#888'),
            opacity=0.70,
            nbinsx=80,
            hovertemplate='Count bucket: %{x}<br>Frequency: %{y}<extra></extra>',
        ))
    fig.update_layout(
        title='Sample Count Distribution per Kmer (log-scale x)',
        xaxis_title='Samples per kmer (n)',
        yaxis_title='Number of kmers',
        barmode='overlay',
        xaxis_type='log',
    )
    figures.append(('Sample count distribution', fig))

    # ── Figure 3: IPD mean violin+box ────────────────────────────────────
    fig = go.Figure()
    for m in meth_ids_sorted:
        g = stats.groups[m]
        # subsample for violin KDE performance
        vals = g.ipd_means
        if len(vals) > _VIOLIN_SUBSAMPLE:
            vals = rng.choice(vals, _VIOLIN_SUBSAMPLE, replace=False)
        fig.add_trace(go.Violin(
            y=vals,
            name=_LABELS.get(m, m),
            line_color=_COLORS.get(m, '#888'),
            box_visible=True,
            meanline_visible=True,
            points=False,
            hoverinfo='y+name',
        ))
    fig.update_layout(
        title='IPD Mean Distribution per Methylation State',
        yaxis_title='Mean IPD per 11-mer context',
        violinmode='overlay',
    )
    figures.append(('IPD mean distribution', fig))

    # ── Figure 4: PW mean violin+box ─────────────────────────────────────
    fig = go.Figure()
    for m in meth_ids_sorted:
        g = stats.groups[m]
        vals = g.pw_means
        if len(vals) > _VIOLIN_SUBSAMPLE:
            vals = rng.choice(vals, _VIOLIN_SUBSAMPLE, replace=False)
        fig.add_trace(go.Violin(
            y=vals,
            name=_LABELS.get(m, m),
            line_color=_COLORS.get(m, '#888'),
            box_visible=True,
            meanline_visible=True,
            points=False,
            hoverinfo='y+name',
        ))
    fig.update_layout(
        title='PW Mean Distribution per Methylation State',
        yaxis_title='Mean PW per 11-mer context',
        violinmode='overlay',
    )
    figures.append(('PW mean distribution', fig))

    # ── Figure 5: Signal sigma (variability) histograms side by side ─────
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['IPD sigma (within-context variability)',
                                        'PW sigma (within-context variability)'])
    for m in meth_ids_sorted:
        g = stats.groups[m]
        label = _LABELS.get(m, m)
        color = _COLORS.get(m, '#888')
        fig.add_trace(go.Histogram(
            x=g.ipd_sigmas, name=label, legendgroup=label,
            marker_color=color, opacity=0.65, nbinsx=60,
            hovertemplate='σ IPD: %{x:.2f}<br>Kmers: %{y}<extra></extra>',
        ), row=1, col=1)
        fig.add_trace(go.Histogram(
            x=g.pw_sigmas, name=label, legendgroup=label,
            showlegend=False,
            marker_color=color, opacity=0.65, nbinsx=60,
            hovertemplate='σ PW: %{x:.2f}<br>Kmers: %{y}<extra></extra>',
        ), row=1, col=2)
    fig.update_layout(title='Within-context Signal Variability (sigma distributions)',
                      barmode='overlay')
    fig.update_xaxes(title_text='σ IPD', row=1, col=1)
    fig.update_xaxes(title_text='σ PW',  row=1, col=2)
    figures.append(('Signal variability (sigma)', fig))

    # ── Figure 6: IPD vs PW scatter (WebGL, subsampled) ──────────────────
    fig = go.Figure()
    for m in meth_ids_sorted:
        g = stats.groups[m]
        n = g.n_entries
        if n > max_scatter:
            idx = rng.choice(n, max_scatter, replace=False)
            x_vals = g.ipd_means[idx]
            y_vals = g.pw_means[idx]
        else:
            x_vals = g.ipd_means
            y_vals = g.pw_means
        fig.add_trace(go.Scattergl(
            x=x_vals, y=y_vals,
            mode='markers',
            name=_LABELS.get(m, m),
            marker=dict(color=_COLORS.get(m, '#888'), size=3, opacity=0.4),
            hovertemplate='IPD mean: %{x:.2f}<br>PW mean: %{y:.2f}<extra></extra>',
        ))
    note = f' (subsampled to {max_scatter:,}/group)' if any(
        stats.groups[m].n_entries > max_scatter for m in meth_ids_sorted) else ''
    fig.update_layout(
        title=f'IPD Mean vs PW Mean per Kmer Context{note}',
        xaxis_title='Mean IPD',
        yaxis_title='Mean PW',
    )
    figures.append(('IPD vs PW correlation', fig))

    # ── Figures 7 & 8: Neighbor sensitivity histograms ───────────────────
    for sig_name, delta_attr, title_signal in [
        ('IPD', 'delta_ipd', '|Δ IPD mean|'),
        ('PW',  'delta_pw',  '|Δ PW mean|'),
    ]:
        unmeth_ns = sensitivity.get('unmethylated')
        meth_ns   = sensitivity.get('methylated')
        has_unmeth = unmeth_ns is not None and unmeth_ns.n_pairs_found > 0
        has_meth   = meth_ns   is not None and meth_ns.n_pairs_found > 0

        if not has_unmeth and not has_meth:
            continue

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'Unmethylated: {title_signal}',
                            f'Methylated: {title_signal}'],
        )
        if has_unmeth:
            d = getattr(unmeth_ns, delta_attr)
            fig.add_trace(go.Histogram(
                x=d, name='Unmethylated',
                marker_color=_COLORS[0], opacity=0.75, nbinsx=80,
            ), row=1, col=1)
            fig.add_vline(x=float(np.mean(d)), line_dash='dash',
                          line_color='black', row=1, col=1,
                          annotation_text=f'mean={np.mean(d):.3f}',
                          annotation_position='top right')
            fig.add_vline(x=float(np.median(d)), line_dash='dot',
                          line_color='grey', row=1, col=1,
                          annotation_text=f'median={np.median(d):.3f}',
                          annotation_position='top left')
        if has_meth:
            d = getattr(meth_ns, delta_attr)
            fig.add_trace(go.Histogram(
                x=d, name='Methylated (all)',
                marker_color='#EF553B', opacity=0.75, nbinsx=80,
            ), row=1, col=2)
            fig.add_vline(x=float(np.mean(d)), line_dash='dash',
                          line_color='black', row=1, col=2,
                          annotation_text=f'mean={np.mean(d):.3f}',
                          annotation_position='top right')
            fig.add_vline(x=float(np.median(d)), line_dash='dot',
                          line_color='grey', row=1, col=2,
                          annotation_text=f'median={np.median(d):.3f}',
                          annotation_position='top left')

        fig.update_layout(
            title=f'1-Base Neighbor Sensitivity: {title_signal}'
                  f'  (dashed = mean, dotted = median)',
        )
        fig.update_xaxes(title_text=title_signal, row=1, col=1)
        fig.update_xaxes(title_text=title_signal, row=1, col=2)
        fig.update_yaxes(title_text='Number of kmer pairs', row=1, col=1)
        figures.append((f'Neighbor sensitivity ({sig_name})', fig))

    # ── Figure 9: Per-position sensitivity profile ────────────────────────
    has_any_pos = any(
        sensitivity.get(cat) is not None
        and sensitivity[cat].n_pairs_found > 0
        for cat in ('unmethylated', 'methylated')
    )
    if has_any_pos:
        positions = list(range(K))
        unmeth_ns = sensitivity.get('unmethylated')
        meth_ns   = sensitivity.get('methylated')

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Mean |Δ IPD| per position',
                            'Mean |Δ PW| per position'],
        )
        for col_idx, (delta_attr, ylabel) in enumerate(
                [('delta_ipd', '|Δ IPD|'), ('delta_pw', '|Δ PW|')], start=1):
            for cat, ns, color, name in [
                ('unmethylated', unmeth_ns, _COLORS[0], 'Unmethylated'),
                ('methylated',   meth_ns,   '#EF553B',  'Methylated'),
            ]:
                if ns is None or ns.n_pairs_found == 0:
                    continue
                d     = getattr(ns, delta_attr)
                y_pos = []
                for pos in positions:
                    mask = ns.positions == pos
                    y_pos.append(float(np.mean(d[mask])) if np.any(mask) else 0.0)
                fig.add_trace(go.Bar(
                    x=[str(p) for p in positions],
                    y=y_pos,
                    name=name,
                    legendgroup=name,
                    showlegend=(col_idx == 1),
                    marker_color=color,
                    opacity=0.8,
                    hovertemplate=f'{name}<br>Position: %{{x}}<br>'
                                  f'Mean {ylabel}: %{{y:.4f}}<extra></extra>',
                ), row=1, col=col_idx)

        fig.update_layout(
            title='Per-Position Sensitivity Profile  '
                  '(pos 0 = 5\' flank, 5 = center/modified base, 10 = 3\' flank)',
            barmode='group',
        )
        for col_idx, ylabel in enumerate(['Mean |Δ IPD|', 'Mean |Δ PW|'], start=1):
            fig.update_xaxes(title_text='Position in 11-mer', row=1, col=col_idx)
            fig.update_yaxes(title_text=ylabel, row=1, col=col_idx)
        figures.append(('Per-position sensitivity profile', fig))

    # ── Figure 10: Meth vs unmeth sensitivity box comparison ─────────────
    has_both = (sensitivity.get('unmethylated') is not None
                and sensitivity['unmethylated'].n_pairs_found > 0
                and sensitivity.get('methylated') is not None
                and sensitivity['methylated'].n_pairs_found > 0)
    if has_both:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['|Δ IPD mean|  (meth vs unmeth)',
                            '|Δ PW mean|   (meth vs unmeth)'],
        )
        unmeth_ns = sensitivity['unmethylated']
        meth_ns   = sensitivity['methylated']
        for col_idx, delta_attr in enumerate(
                ['delta_ipd', 'delta_pw'], start=1):
            for ns, label, color in [
                (unmeth_ns, 'Unmethylated', _COLORS[0]),
                (meth_ns,   'Methylated',   '#EF553B'),
            ]:
                d = getattr(ns, delta_attr)
                fig.add_trace(go.Box(
                    y=d, name=label,
                    marker_color=color,
                    boxmean='sd',
                    legendgroup=label,
                    showlegend=(col_idx == 1),
                ), row=1, col=col_idx)
        fig.update_layout(
            title='Meth vs Unmeth: 1-base Neighbor Sensitivity Comparison',
            boxmode='group',
        )
        figures.append(('Sensitivity comparison (meth vs unmeth)', fig))

    # ── Assemble HTML ─────────────────────────────────────────────────────
    meth_count_str = ', '.join(
        f'{_LABELS.get(m, m)}: {stats.groups[m].n_entries:,}'
        for m in meth_ids_sorted
    )
    # Stat cards HTML
    cards = [
        ('Total entries',  f'{stats.total_entries:,}'),
        ('Total samples',  f'{stats.total_samples:,.0f}'),
        ('Overall cov.',   _pct(stats.total_entries, TOTAL_POSSIBLE_KMERS)),
        ('Meth states',    str(len(stats.groups))),
        ('File size',      f'{stats.file_size_mb:.1f} MB'),
    ]
    card_html = '\n'.join(
        f'<div class="stat-card">'
        f'<div class="val">{v}</div>'
        f'<div class="lbl">{k}</div>'
        f'</div>'
        for k, v in cards
    )

    # Navigation links
    nav_links = '\n'.join(
        f'<a href="#section-{i}">{title}</a>'
        for i, (title, _) in enumerate(figures)
    )

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KinSim Dictionary Report — {os.path.basename(stats.pkl_path)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: #f0f2f5; color: #333; }}
  .page {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
  .header {{ background: linear-gradient(135deg,#1a2a4a,#2c5282);
             color:#fff; padding:32px; border-radius:12px; margin-bottom:24px; }}
  .header h1 {{ font-size:1.9em; margin-bottom:6px; }}
  .header p  {{ opacity:.8; font-size:.95em; }}
  .stat-grid {{ display:grid;
                grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                gap:16px; margin-bottom:24px; }}
  .stat-card {{ background:#fff; padding:20px; border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,.08); text-align:center; }}
  .stat-card .val {{ font-size:1.9em; font-weight:700; color:#2c5282; }}
  .stat-card .lbl {{ color:#718096; margin-top:6px; font-size:.85em; }}
  .nav {{ background:#fff; padding:14px 20px; border-radius:10px;
          box-shadow:0 2px 8px rgba(0,0,0,.08); margin-bottom:24px;
          display:flex; flex-wrap:wrap; gap:12px; align-items:center; }}
  .nav span {{ font-weight:600; color:#4a5568; }}
  .nav a {{ color:#3182ce; text-decoration:none; font-size:.9em;
            padding:4px 10px; border:1px solid #bee3f8; border-radius:20px; }}
  .nav a:hover {{ background:#ebf8ff; }}
  .section {{ background:#fff; padding:24px; border-radius:10px;
              box-shadow:0 2px 8px rgba(0,0,0,.08); margin-bottom:24px; }}
  .section h2 {{ color:#2c5282; border-bottom:3px solid #3182ce;
                 padding-bottom:10px; margin-bottom:18px; font-size:1.25em; }}
  .plot-box {{ width:100%; min-height:480px; }}
  .footer {{ text-align:center; color:#a0aec0; font-size:.8em; padding:20px 0; }}
</style>
</head>
<body>
<div class="page">

<div class="header">
  <h1>KinSim Dictionary Analysis Report</h1>
  <p>{stats.pkl_path}</p>
  <p style="margin-top:8px;opacity:.7">{meth_count_str}</p>
</div>

<div class="stat-grid">
{card_html}
</div>

<div class="nav">
  <span>Jump to:</span>
  {nav_links}
</div>
"""]

    for i, (title, fig) in enumerate(figures):
        plot_json = pio.to_json(fig)
        html_parts.append(f"""
<div class="section" id="section-{i}">
  <h2>{title}</h2>
  <div id="plot_{i}" class="plot-box"></div>
  <script>
    (function(){{
      var spec = {plot_json};
      spec.layout = spec.layout || {{}};
      spec.layout.autosize = true;
      Plotly.newPlot('plot_{i}', spec.data, spec.layout,
                     {{responsive: true, displayModeBar: true}});
    }})();
  </script>
</div>
""")

    html_parts.append("""
<div class="footer">Generated by KinSim &mdash; kinsim dictionary analyze</div>
</div>
</body>
</html>
""")

    with open(output_path, 'w') as fh:
        fh.write('\n'.join(html_parts))
    print(f'HTML report saved to: {output_path}')


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def analyze_dict(
    pkl_path: str,
    output_dir: str = None,
    no_html: bool = False,
    max_scatter: int = 10_000,
    max_neighbor_entries: int = 500_000,
):
    """Load a dictionary, compute comprehensive statistics, render TXT + HTML reports."""
    t0 = time.time()

    # Resolve output paths
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pkl_path))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pkl_path))[0]
    txt_path  = os.path.join(output_dir, f'{base}_report.txt')
    html_path = os.path.join(output_dir, f'{base}_report.html')

    # Load
    print(f'Loading dictionary: {pkl_path}')
    with open(pkl_path, 'rb') as fh:
        lookup = pickle.load(fh)

    if not lookup:
        print('Dictionary is empty.')
        return

    # Collect per-kmer statistics
    print(f'Collecting statistics from {len(lookup):,} entries...')
    stats = collect_stats(lookup, pkl_path)

    # Neighbor sensitivity
    sensitivity = {}
    if max_neighbor_entries != 0:
        print(f'Computing 1-base neighbor sensitivity '
              f'(cap {max_neighbor_entries:,} entries per category)...')
        sensitivity = compute_neighbor_sensitivity(
            lookup, stats, max_entries=max_neighbor_entries)
        for cat, ns in sensitivity.items():
            print(f'  {cat}: {ns.n_source_entries:,} source kmers, '
                  f'{ns.n_pairs_found:,} neighbor pairs')
    else:
        print('Skipping neighbor sensitivity (--max-neighbor-entries 0)')

    # TXT report (always — stdout + file)
    print()
    render_txt_report(stats, sensitivity, txt_path)

    # HTML report (optional)
    if not no_html:
        print('Generating HTML report...')
        render_html_report(stats, sensitivity, html_path,
                           max_scatter=max_scatter)

    print(f'\nAnalysis completed in {time.time() - t0:.1f}s')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog='kinsim dictionary analyze',
        description=(
            'Comprehensive analysis of a trained 11-mer kinetic dictionary.\n\n'
            'Generates:\n'
            '  <basename>_report.txt  — printed to stdout AND saved to file\n'
            '  <basename>_report.html — interactive Plotly visualizations\n'
            '                          (requires: pip install plotly, or '
            'pip install -e .[analyze])\n\n'
            'Analysis includes:\n'
            '  · Coverage per methylation state\n'
            '  · Sample count distributions\n'
            '  · IPD/PW mean and sigma distributions\n'
            '  · Low-coverage entry counts\n'
            '  · 1-base neighbor sensitivity (methylated and unmethylated)\n'
            '  · Per-position sensitivity profile\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('pkl', help='Path to the trained dictionary (.pkl file)')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory for reports '
                             '(default: same directory as pkl)')
    parser.add_argument('--no-html', action='store_true',
                        help='Skip HTML report (TXT only, no plotly required)')
    parser.add_argument('--max-scatter', type=int, default=10_000,
                        metavar='N',
                        help='Max points per group in IPD-vs-PW scatter '
                             '(default: 10000)')
    parser.add_argument('--max-neighbor-entries', type=int, default=500_000,
                        metavar='N',
                        help='Max kmers per category for neighbor sensitivity '
                             '(default: 500000; 0 = skip analysis)')
    args = parser.parse_args(argv)
    analyze_dict(
        pkl_path=args.pkl,
        output_dir=args.output_dir,
        no_html=args.no_html,
        max_scatter=args.max_scatter,
        max_neighbor_entries=args.max_neighbor_entries,
    )


if __name__ == '__main__':
    main()
