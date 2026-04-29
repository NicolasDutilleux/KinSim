"""Comprehensive analysis of a KinSim training .pkl file.

Data format (produced by ``kinsim extract`` / ``kinsim merge``):

    {(kmer_id, meth_id): np.ndarray(N, 2/3/14)}
        columns: [IPD, PW]              (legacy 2-col)
                 [IPD, PW, fraction]    (legacy 3-col)
                 [IPD, PW, fraction, mc_0..mc_10]  (current 14-col, K=11)

Auto-detects:
  - K-mer size           (from __meta__["kmer_size"] or inferred from key range)
  - Methylation types    (whatever is present in the file — no hardcoded list)

Generates:
  - <basename>_report.txt   — text report printed to stdout AND written to file
  - <basename>_report.html  — interactive Plotly visualisations (requires plotly)

Analysis sections:
  1. Overview            — file size, format, kmer size, key/sample counts
  2. Per-type coverage   — unique kmers, %, sample count distributions,
                           methylation fraction distributions (incl. breakdown)
  3. Signal statistics   — IPD/PW mean and sigma per meth type
  4. Low-coverage keys   — keys with n < 5 / 10 / 50 samples
  5. Neighbor sensitivity — how a single base change in the k-mer context
                            affects expected IPD/PW; per-position breakdown

CLI usage:
    kinsim analyze bc2033_shard.pkl
    kinsim analyze master_data.pkl --output-dir reports/ --no-html
"""

from __future__ import annotations

import dataclasses
import io
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np

from .utils.encoding import METH_IDS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookups and helpers
# ---------------------------------------------------------------------------

_ID_TO_NAME = {v: k for k, v in METH_IDS.items()}

# Colour palette — cycles for any number of meth types
_COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA',
    '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
    '#FF97FF', '#FECB52',
]

_VIOLIN_SUBSAMPLE = 100_000  # Plotly KDE slows above ~100K points per trace


def _meth_name(meth_id: int) -> str:
    """Human-readable name for a methylation state id."""
    return _ID_TO_NAME.get(meth_id, f'meth_id={meth_id}')


def _meth_color(meth_id: int) -> str:
    return _COLORS[meth_id % len(_COLORS)]


def _darken_hex(hex_color: str, factor: float) -> str:
    """Darken a hex color by factor (0=black, 1=original)."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f'rgb({int(r*factor)},{int(g*factor)},{int(b*factor)})'


def _pct(val, total) -> str:
    return f'{100.0 * val / total:.2f}%' if total else 'n/a'


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MethGroupStats:
    """Statistics collected for one methylation state."""
    meth_id: int
    name: str                     # e.g. "none", "m6A", "m4C", "m5C", "meth_id=5"
    n_entries: int
    sample_counts: np.ndarray     # (n_entries,)  — samples per key
    ipd_means: np.ndarray         # (n_entries,)  — per-key IPD mean
    ipd_sigmas: np.ndarray        # (n_entries,)  — per-key IPD std
    pw_means: np.ndarray          # (n_entries,)  — per-key PW mean
    pw_sigmas: np.ndarray         # (n_entries,)  — per-key PW std
    fraction_means: np.ndarray    # (n_entries,)  — per-key mean frac (NaN if absent)
    kmer_ids: np.ndarray          # (n_entries,)  int64 — used for neighbor lookup


@dataclasses.dataclass
class DictStats:
    """All statistics collected from one .pkl file."""
    pkl_path: str
    kmer_size: int
    total_possible_kmers: int     # 4 ** kmer_size
    total_entries: int
    total_samples: int
    groups: dict                  # meth_id -> MethGroupStats
    file_size_mb: float
    meta: dict                    # __meta__ contents (empty dict if absent)


@dataclasses.dataclass
class NeighborSensitivity:
    """1-base substitution sensitivity results for one category."""
    delta_ipd: np.ndarray         # |Δ mean_ipd| per neighbor pair found
    delta_pw: np.ndarray          # |Δ mean_pw|  per neighbor pair found
    positions: np.ndarray         # int8 — which position (0…K-1) was mutated
    n_source_entries: int
    n_pairs_found: int


# ---------------------------------------------------------------------------
# K-mer size auto-detection
# ---------------------------------------------------------------------------

def _detect_kmer_size(data: dict, meta: dict) -> int:
    """Infer kmer size from __meta__ or from the range of kmer_ids in the data."""
    if meta and 'kmer_size' in meta:
        return int(meta['kmer_size'])
    # Infer from the largest kmer_id present:
    # a kmer_id < 4^k → needs 2k bits → k = ceil(bits / 2)
    max_kmer = 0
    for k in data:
        if isinstance(k, tuple) and len(k) >= 1:
            if k[0] > max_kmer:
                max_kmer = k[0]
    if max_kmer == 0:
        return 11  # safe default
    bits = max_kmer.bit_length()
    return max(1, (bits + 1) // 2)


# ---------------------------------------------------------------------------
# Per-key stat extractors (one per format)
# ---------------------------------------------------------------------------

def _key_stats_samples(arr: np.ndarray):
    """Extract (n, mu_ipd, sig_ipd, mu_pw, sig_pw, frac) from a raw-sample array."""
    n = len(arr)
    mu_ipd  = float(np.mean(arr[:, 0]))
    sig_ipd = float(np.std(arr[:, 0]))
    mu_pw   = float(np.mean(arr[:, 1]))
    sig_pw  = float(np.std(arr[:, 1]))
    frac    = float(np.mean(arr[:, 2])) if arr.shape[1] >= 3 else float('nan')
    return n, mu_ipd, sig_ipd, mu_pw, sig_pw, frac


def compute_signature_profiles(data: dict, kmer_size: int = 11) -> dict:
    """Aggregate the kinetic profile per meth type across all keys.

    Returns dict[meth_name] -> {'profile_ipd': np.ndarray(PROFILE_LEN,),
                                'profile_pw':  np.ndarray(PROFILE_LEN,),
                                'n_samples':   int,
                                'sig_offsets': list[int]}
    """
    from .extract import METH_CTX_LEN, PROFILE_LEN
    from .utils.config import load_kinsim_config

    cfg = load_kinsim_config()
    profile_start_col = 3 + METH_CTX_LEN
    pw_start_col      = profile_start_col + PROFILE_LEN
    needed_cols       = pw_start_col + PROFILE_LEN  # = 32

    out: dict = {}
    for meth_id_int in [0, 1, 2, 3]:
        name = _meth_name(meth_id_int)
        sums_ipd = np.zeros(PROFILE_LEN, dtype=np.float64)
        sums_pw  = np.zeros(PROFILE_LEN, dtype=np.float64)
        n_total  = 0
        for k, v in data.items():
            if not (isinstance(k, tuple) and len(k) == 2 and isinstance(v, np.ndarray)):
                continue
            kmer_id, meth_id = k
            if int(meth_id) != meth_id_int:
                continue
            if v.shape[1] < needed_cols:
                continue  # old-format pkl, no profile stored
            sums_ipd += v[:, profile_start_col:pw_start_col].sum(axis=0)
            sums_pw  += v[:, pw_start_col:needed_cols].sum(axis=0)
            n_total  += len(v)
        if n_total == 0:
            continue
        sig_offsets = list(cfg.get("kinetic_signatures", {})
                              .get(name, {}).get("signal_offsets", []))
        out[name] = {
            "profile_ipd": (sums_ipd / n_total).astype(np.float32),
            "profile_pw":  (sums_pw  / n_total).astype(np.float32),
            "n_samples":   n_total,
            "sig_offsets": sig_offsets,
        }
    return out


# ---------------------------------------------------------------------------
# Stats collection (single pass)
# ---------------------------------------------------------------------------

def collect_stats(data: dict, pkl_path: str) -> DictStats:
    """One-pass collection of all per-key statistics from a .pkl file."""
    meta = data.get('__meta__', {})
    if not isinstance(meta, dict):
        meta = {}

    kmer_size = _detect_kmer_size(data, meta)
    total_possible = 4 ** kmer_size

    # Group items by meth_id
    partitions: dict[int, list] = {}
    for k, v in data.items():
        if not (isinstance(k, tuple) and len(k) == 2 and isinstance(v, np.ndarray)):
            continue
        kmer_id, meth_id = k
        if meth_id not in partitions:
            partitions[meth_id] = []
        partitions[meth_id].append((kmer_id, v))

    groups: dict[int, MethGroupStats] = {}
    total_samples = 0

    for meth_id, items in partitions.items():
        n = len(items)
        sample_counts  = np.empty(n, dtype=np.float64)
        ipd_means      = np.empty(n, dtype=np.float64)
        ipd_sigmas     = np.empty(n, dtype=np.float64)
        pw_means       = np.empty(n, dtype=np.float64)
        pw_sigmas      = np.empty(n, dtype=np.float64)
        fraction_means = np.full(n, float('nan'), dtype=np.float64)
        kmer_ids       = np.empty(n, dtype=np.int64)

        for i, (kmer_id, arr) in enumerate(items):
            cnt, mu_ipd, sig_ipd, mu_pw, sig_pw, frac = _key_stats_samples(arr)
            sample_counts[i]  = cnt
            ipd_means[i]      = mu_ipd
            ipd_sigmas[i]     = sig_ipd
            pw_means[i]       = mu_pw
            pw_sigmas[i]      = sig_pw
            fraction_means[i] = frac
            kmer_ids[i]       = kmer_id
            total_samples    += cnt

        groups[meth_id] = MethGroupStats(
            meth_id=meth_id,
            name=_meth_name(meth_id),
            n_entries=n,
            sample_counts=sample_counts,
            ipd_means=ipd_means,
            ipd_sigmas=ipd_sigmas,
            pw_means=pw_means,
            pw_sigmas=pw_sigmas,
            fraction_means=fraction_means,
            kmer_ids=kmer_ids,
        )

    n_data_keys = sum(len(v) for v in partitions.values())
    file_size_mb = Path(pkl_path).stat().st_size / (1024 * 1024)

    return DictStats(
        pkl_path=pkl_path,
        kmer_size=kmer_size,
        total_possible_kmers=total_possible,
        total_entries=n_data_keys,
        total_samples=int(total_samples),
        groups=groups,
        file_size_mb=file_size_mb,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Neighbor sensitivity  O(n × K × 3)
# ---------------------------------------------------------------------------

def compute_neighbor_sensitivity(
    data: dict,
    stats: DictStats,
    max_entries: int = 200_000,
) -> dict:
    """Compute 1-base substitution sensitivity for methylated and unmethylated kmers.

    For each source kmer, generates all K×3 single-base substitutions and
    records |Δ mean_ipd|, |Δ mean_pw|, and the mutated position for every
    neighbor found in the dictionary.

    Returns dict with keys 'unmethylated' and 'methylated'.
    """
    rng = np.random.default_rng(42)
    K   = stats.kmer_size

    # Build fast mean lookup: (kmer_id, meth_id) -> (mu_ipd, mu_pw)
    mean_lookup: dict = {}
    for k, v in data.items():
        if not (isinstance(k, tuple) and len(k) == 2 and isinstance(v, np.ndarray)):
            continue
        mean_lookup[k] = (float(np.mean(v[:, 0])), float(np.mean(v[:, 1])))

    categories = [
        ('unmethylated', [0]),
        ('methylated',   [mid for mid in stats.groups if mid != 0]),
    ]

    results = {}
    for category, meth_ids in categories:
        # Collect source entries for this category
        source = []
        for mid in meth_ids:
            g = stats.groups.get(mid)
            if g is None:
                continue
            for i in range(g.n_entries):
                key = (int(g.kmer_ids[i]), mid)
                if key in mean_lookup:
                    source.append((int(g.kmer_ids[i]), mid,
                                   mean_lookup[key][0], mean_lookup[key][1]))

        if not source:
            results[category] = NeighborSensitivity(
                delta_ipd=np.array([], dtype=np.float32),
                delta_pw=np.array([], dtype=np.float32),
                positions=np.array([], dtype=np.int8),
                n_source_entries=0, n_pairs_found=0,
            )
            continue

        # Subsample large groups
        if max_entries > 0 and len(source) > max_entries:
            idx = rng.choice(len(source), max_entries, replace=False)
            source = [source[i] for i in idx]

        delta_ipd_list: list = []
        delta_pw_list:  list = []
        pos_list:       list = []

        for kmer_int, meth_id, ipd_mean, pw_mean in source:
            for pos in range(K):
                bit_pos   = (K - 1 - pos) * 2       # MSB-first encoding
                orig_base = (kmer_int >> bit_pos) & 0x3
                mask_bits = ~(0x3 << bit_pos)
                for new_base in range(4):
                    if new_base == orig_base:
                        continue
                    neighbor = (kmer_int & mask_bits) | (new_base << bit_pos)
                    nb = mean_lookup.get((neighbor, meth_id))
                    if nb is None:
                        continue
                    delta_ipd_list.append(abs(ipd_mean - nb[0]))
                    delta_pw_list.append(abs(pw_mean  - nb[1]))
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
# TXT report
# ---------------------------------------------------------------------------

def render_txt_report(stats: DictStats, sensitivity: dict, output_path: str,
                       signature_profiles: dict | None = None) -> None:
    """Print TXT report to stdout and write to *output_path*."""
    buf = io.StringIO()
    K = stats.kmer_size

    def p(line: str = '') -> None:
        print(line)
        buf.write(line + '\n')

    W = 72

    # ── 1. Overview ──────────────────────────────────────────────────────
    p('=' * W)
    p('KinSim  —  Training Data Analysis Report')
    p('=' * W)
    p(f"File          : {stats.pkl_path}")
    p(f"K-mer size    : {K}  (4^{K} = {stats.total_possible_kmers:,} possible)")
    p(f"File size     : {stats.file_size_mb:.1f} MB")
    p(f"Total keys    : {stats.total_entries:,}")
    p(f"Total samples : {stats.total_samples:,}")
    p(f"Possible kmers: {stats.total_possible_kmers:,}  (4^{K})")
    p(f"Overall cov.  : {_pct(stats.total_entries, stats.total_possible_kmers)}")
    if stats.meta:
        if 'created' in stats.meta:
            p(f"Created       : {stats.meta['created']}")
        src = stats.meta.get('source_bam', stats.meta.get('merged_from', ''))
        if src:
            s = src if isinstance(src, str) else ', '.join(src[:3]) + ('...' if len(src) > 3 else '')
            p(f"Source        : {s}")
    p()

    # ── 2. Per-meth coverage ─────────────────────────────────────────────
    p('-' * W)
    p('Per-methylation-state coverage')
    p('-' * W)
    for meth_id in sorted(stats.groups.keys()):
        g = stats.groups[meth_id]
        label = 'Unmethylated' if meth_id == 0 else f'Methylated ({g.name})'
        c = g.sample_counts
        p(f'\n  {label}:')
        p(f'    Unique {K}-mers : {g.n_entries:,} / {stats.total_possible_kmers:,}'
          f'  ({_pct(g.n_entries, stats.total_possible_kmers)})')
        p(f'    Total samples  : {np.sum(c):,.0f}')
        p(f'    Samples/key    :'
          f' mean={np.mean(c):.1f}  median={np.median(c):.1f}'
          f'  p5={np.percentile(c, 5):.0f}  p25={np.percentile(c, 25):.0f}'
          f'  p75={np.percentile(c, 75):.0f}  p95={np.percentile(c, 95):.0f}')
        p(f'                     min={np.min(c):.0f}  max={np.max(c):.0f}')
        # Methylation fraction (only for methylated, only if available)
        if meth_id != 0:
            valid = g.fraction_means[~np.isnan(g.fraction_means)]
            if len(valid) > 0:
                p(f'    Meth fraction  :'
                  f' mean={np.mean(valid):.3f}  median={np.median(valid):.3f}'
                  f'  p5={np.percentile(valid, 5):.3f}'
                  f'  p95={np.percentile(valid, 95):.3f}')
                # Fraction breakdown: how many keys at 100%, 0%, in-between
                n_full   = int(np.sum(valid >= 0.999))
                n_zero   = int(np.sum(valid <= 0.001))
                n_mixed  = len(valid) - n_full - n_zero
                p(f'    Fraction split  :'
                  f' frac=1.0: {n_full:,} ({_pct(n_full, len(valid))})'
                  f'  frac=0.0: {n_zero:,} ({_pct(n_zero, len(valid))})'
                  f'  mixed: {n_mixed:,} ({_pct(n_mixed, len(valid))})')
    p()

    # ── 3. Signal statistics ──────────────────────────────────────────────
    p('-' * W)
    p('Signal statistics  (distribution of per-key means and sigmas)')
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
              f' mean={np.mean(means):.3f}  std={np.std(means):.3f}'
              f'  median={np.median(means):.3f}'
              f'  p5={np.percentile(means, 5):.3f}'
              f'  p95={np.percentile(means, 95):.3f}')
            p(f'    {sig_name} sigma  :'
              f' mean={np.mean(sigmas):.3f}  std={np.std(sigmas):.3f}'
              f'  median={np.median(sigmas):.3f}'
              f'  p5={np.percentile(sigmas, 5):.3f}'
              f'  p95={np.percentile(sigmas, 95):.3f}')
    p()

    # ── 3.5 Kinetic signature profile per meth type ──────────────────────
    if signature_profiles:
        p('-' * W)
        p('Kinetic signature profiles  (mean IPD/PW at offsets 0..+8 from prediction pos)')
        p('-' * W)
        for name, sp in signature_profiles.items():
            if name == 'none':
                continue
            offsets_str = '  '.join(f'+{i}' for i in range(len(sp['profile_ipd'])))
            ipd_str = '  '.join(f'{v:5.1f}' for v in sp['profile_ipd'])
            pw_str  = '  '.join(f'{v:5.1f}' for v in sp['profile_pw'])
            sig = sp.get('sig_offsets', [])
            sig_marker = '  '.join('***' if i in sig else '   '
                                    for i in range(len(sp['profile_ipd'])))
            p(f"\n  {name}  (n={sp['n_samples']:,} samples, signature at {sig})")
            p(f"    Offset      :  {offsets_str}")
            p(f"    Signature   :  {sig_marker}")
            p(f"    IPD profile :  {ipd_str}")
            p(f"    PW  profile :  {pw_str}")
            # Compute the signature score: mean at sig offsets / mean at non-sig
            if sig:
                ipd_arr = sp['profile_ipd']
                sig_idx     = [i for i in sig if 0 <= i < len(ipd_arr)]
                nonsig_idx  = [i for i in range(len(ipd_arr)) if i not in sig_idx]
                if sig_idx and nonsig_idx:
                    score = float(ipd_arr[sig_idx].mean()
                                   / max(float(ipd_arr[nonsig_idx].mean()), 1.0))
                    p(f"    Sig/non-sig IPD ratio : {score:.3f}  "
                      f"(>1.3 expected for real {name})")
        p()

    # ── 4. Low-coverage warnings ──────────────────────────────────────────
    p('-' * W)
    p('Low-coverage keys  (keys with few samples — less reliable signals)')
    p('-' * W)
    for meth_id in sorted(stats.groups.keys()):
        g = stats.groups[meth_id]
        label = 'Unmethylated' if meth_id == 0 else f'Methylated ({g.name})'
        ne  = g.n_entries
        n5  = int(np.sum(g.sample_counts < 5))
        n10 = int(np.sum(g.sample_counts < 10))
        n50 = int(np.sum(g.sample_counts < 50))
        p(f'  {label:30s}'
          f'  n<5: {n5:,} ({_pct(n5, ne)})'
          f'   n<10: {n10:,} ({_pct(n10, ne)})'
          f'   n<50: {n50:,} ({_pct(n50, ne)})')
    p()

    # ── 5. Neighbor sensitivity ───────────────────────────────────────────
    p('-' * W)
    p('1-base neighbor sensitivity analysis')
    p('-' * W)
    center = K // 2
    p(f'  How much does a single base change in the {K}-mer context affect')
    p(f'  the expected (mean) IPD / PW?  '
      f'(pos {center}[C] = center / modified base)')
    p()
    for category in ['unmethylated', 'methylated']:
        ns = sensitivity.get(category)
        if ns is None or ns.n_pairs_found == 0:
            p(f'  {category.capitalize()}: no neighbor pairs found (skipping)')
            continue
        p(f'  {category.capitalize()} '
          f'({ns.n_source_entries:,} source kmers, '
          f'{ns.n_pairs_found:,} neighbor pairs):')
        p(f'    |Δ IPD mean|  :'
          f' mean={np.mean(ns.delta_ipd):.4f}'
          f'  median={np.median(ns.delta_ipd):.4f}'
          f'  p75={np.percentile(ns.delta_ipd, 75):.4f}'
          f'  p95={np.percentile(ns.delta_ipd, 95):.4f}')
        p(f'    |Δ PW  mean|  :'
          f' mean={np.mean(ns.delta_pw):.4f}'
          f'  median={np.median(ns.delta_pw):.4f}'
          f'  p75={np.percentile(ns.delta_pw, 75):.4f}'
          f'  p95={np.percentile(ns.delta_pw, 95):.4f}')
        ipd_by_pos, pw_by_pos = [], []
        for pos in range(K):
            mask = ns.positions == pos
            tag  = '[C]' if pos == center else ''
            ipd_by_pos.append(
                f'{pos}{tag}:{np.mean(ns.delta_ipd[mask]):.3f}' if np.any(mask) else f'{pos}:n/a')
            pw_by_pos.append(
                f'{pos}{tag}:{np.mean(ns.delta_pw[mask]):.3f}' if np.any(mask) else f'{pos}:n/a')
        p(f'    Per-pos |Δ IPD|:')
        p(f'      {", ".join(ipd_by_pos)}')
        p(f'    Per-pos |Δ PW|:')
        p(f'      {", ".join(pw_by_pos)}')
        p()

    p('=' * W)

    with open(output_path, 'w') as fh:
        fh.write(buf.getvalue())
    print(f'\nTXT report saved: {output_path}')


# ---------------------------------------------------------------------------
# HTML report (Plotly)
# ---------------------------------------------------------------------------

def render_html_report(
    stats: DictStats,
    sensitivity: dict,
    output_path: str,
    max_scatter: int = 10_000,
    min_ipd_m6a: float = 0.0,
) -> None:
    """Generate a self-contained interactive HTML report using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        log.warning(
            'plotly not installed — skipping HTML report.  '
            'Install with: pip install plotly'
        )
        return

    rng = np.random.default_rng(0)
    K   = stats.kmer_size
    figures: list[tuple[str, object]] = []
    meth_ids_sorted = sorted(stats.groups.keys())

    def lbl(mid: int) -> str:
        return 'Unmethylated' if mid == 0 else stats.groups[mid].name

    def clr(mid: int) -> str:
        return _meth_color(mid)

    # ── Fig 1: Coverage bar chart ─────────────────────────────────────────
    fig = go.Figure(go.Bar(
        x=[lbl(m) for m in meth_ids_sorted],
        y=[stats.groups[m].n_entries for m in meth_ids_sorted],
        marker_color=[clr(m) for m in meth_ids_sorted],
        text=[_pct(stats.groups[m].n_entries, stats.total_possible_kmers)
              for m in meth_ids_sorted],
        textposition='auto',
        hovertemplate='%{x}<br>Unique keys: %{y:,}<extra></extra>',
    ))
    fig.update_layout(
        title=f'{K}-mer Coverage per Methylation State  '
              f'(total possible: {stats.total_possible_kmers:,})',
        yaxis_title=f'Unique {K}-mers',
    )
    figures.append(('Coverage', fig))

    # ── Fig 2: Sample count distribution ─────────────────────────────────
    fig = go.Figure()
    for m in meth_ids_sorted:
        g = stats.groups[m]
        fig.add_trace(go.Histogram(
            x=g.sample_counts, name=lbl(m),
            marker_color=clr(m), opacity=0.70, nbinsx=80,
            hovertemplate='n: %{x}<br>Keys: %{y}<extra></extra>',
        ))
    fig.update_layout(
        title='Sample Count Distribution per Key  (log-scale x)',
        xaxis_title='Samples per key', yaxis_title='Number of keys',
        barmode='overlay', xaxis_type='log',
    )
    figures.append(('Sample count distribution', fig))

    # ── Fig 3: IPD mean violin ────────────────────────────────────────────
    fig = go.Figure()
    for m in meth_ids_sorted:
        g = stats.groups[m]
        vals = g.ipd_means
        if len(vals) > _VIOLIN_SUBSAMPLE:
            vals = rng.choice(vals, _VIOLIN_SUBSAMPLE, replace=False)
        fig.add_trace(go.Violin(
            y=vals, name=lbl(m), line_color=clr(m),
            box_visible=True, meanline_visible=True, points=False,
        ))
    fig.update_layout(
        title=f'IPD Mean Distribution per Methylation State',
        yaxis_title=f'Mean IPD per {K}-mer context', violinmode='overlay',
    )
    figures.append(('IPD mean distribution', fig))

    # ── Fig 4: PW mean violin ─────────────────────────────────────────────
    fig = go.Figure()
    for m in meth_ids_sorted:
        g = stats.groups[m]
        vals = g.pw_means
        if len(vals) > _VIOLIN_SUBSAMPLE:
            vals = rng.choice(vals, _VIOLIN_SUBSAMPLE, replace=False)
        fig.add_trace(go.Violin(
            y=vals, name=lbl(m), line_color=clr(m),
            box_visible=True, meanline_visible=True, points=False,
        ))
    fig.update_layout(
        title='PW Mean Distribution per Methylation State',
        yaxis_title=f'Mean PW per {K}-mer context', violinmode='overlay',
    )
    figures.append(('PW mean distribution', fig))

    # ── Fig 5: Sigma distributions ────────────────────────────────────────
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['IPD sigma  (within-context variability)',
                                        'PW sigma   (within-context variability)'])
    for m in meth_ids_sorted:
        g = stats.groups[m]
        l, c = lbl(m), clr(m)
        fig.add_trace(go.Histogram(
            x=g.ipd_sigmas, name=l, legendgroup=l,
            marker_color=c, opacity=0.65, nbinsx=60,
        ), row=1, col=1)
        fig.add_trace(go.Histogram(
            x=g.pw_sigmas, name=l, legendgroup=l, showlegend=False,
            marker_color=c, opacity=0.65, nbinsx=60,
        ), row=1, col=2)
    fig.update_layout(title='Within-context Signal Variability (sigma)', barmode='overlay')
    fig.update_xaxes(title_text='σ IPD', row=1, col=1)
    fig.update_xaxes(title_text='σ PW',  row=1, col=2)
    figures.append(('Signal variability (sigma)', fig))

    # ── Fig 6: IPD vs PW scatter ──────────────────────────────────────────
    fig = go.Figure()
    for m in meth_ids_sorted:
        g = stats.groups[m]
        n = g.n_entries
        if n > max_scatter:
            idx    = rng.choice(n, max_scatter, replace=False)
            x_vals = g.ipd_means[idx]
            y_vals = g.pw_means[idx]
        else:
            x_vals = g.ipd_means
            y_vals = g.pw_means
        fig.add_trace(go.Scattergl(
            x=x_vals, y=y_vals, mode='markers', name=lbl(m),
            marker=dict(color=clr(m), size=3, opacity=0.4),
            hovertemplate='IPD: %{x:.2f}<br>PW: %{y:.2f}<extra></extra>',
        ))
    suffix = (f' (subsampled to {max_scatter:,}/group)'
              if any(stats.groups[m].n_entries > max_scatter for m in meth_ids_sorted)
              else '')
    fig.update_layout(
        title=f'IPD Mean vs PW Mean per Key{suffix}',
        xaxis_title='Mean IPD', yaxis_title='Mean PW',
    )
    figures.append(('IPD vs PW correlation', fig))

    # ── Fig 7: Methylation fraction distribution ──────────────────────────
    any_frac = any(
        not np.all(np.isnan(stats.groups[m].fraction_means))
        for m in meth_ids_sorted if m != 0
    )
    if any_frac:
        fig = go.Figure()
        for m in meth_ids_sorted:
            if m == 0:
                continue
            g    = stats.groups[m]
            vals = g.fraction_means[~np.isnan(g.fraction_means)]
            if len(vals) == 0:
                continue
            fig.add_trace(go.Histogram(
                x=vals, name=lbl(m), marker_color=clr(m),
                opacity=0.70, nbinsx=60,
                hovertemplate='Fraction: %{x:.2f}<br>Keys: %{y}<extra></extra>',
            ))
        fig.update_layout(
            title='Methylation Fraction Distribution per Key  (methylated types only)',
            xaxis_title='Mean methylation fraction per key',
            yaxis_title='Number of keys',
            barmode='overlay',
        )
        figures.append(('Methylation fraction', fig))

    # ── Figs 8–9: Neighbor sensitivity histograms ─────────────────────────
    for sig_name, delta_attr, title_sig in [
        ('IPD', 'delta_ipd', '|Δ IPD mean|'),
        ('PW',  'delta_pw',  '|Δ PW mean|'),
    ]:
        unmeth_ns = sensitivity.get('unmethylated')
        meth_ns   = sensitivity.get('methylated')
        has_u = unmeth_ns is not None and unmeth_ns.n_pairs_found > 0
        has_m = meth_ns   is not None and meth_ns.n_pairs_found > 0
        if not has_u and not has_m:
            continue
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f'Unmethylated: {title_sig}',
                                            f'Methylated:   {title_sig}'])
        for ns, rc, name, color in [
            (unmeth_ns, (1, 1), 'Unmethylated', _meth_color(0)),
            (meth_ns,   (1, 2), 'Methylated',   '#EF553B'),
        ]:
            if ns is None or ns.n_pairs_found == 0:
                continue
            d = getattr(ns, delta_attr)
            fig.add_trace(go.Histogram(
                x=d, name=name, marker_color=color, opacity=0.75, nbinsx=80,
            ), row=rc[0], col=rc[1])
            fig.add_vline(
                x=float(np.mean(d)), line_dash='dash', line_color='black',
                row=rc[0], col=rc[1],
                annotation_text=f'mean={np.mean(d):.3f}',
                annotation_position='top right',
            )
        fig.update_layout(title=f'1-Base Neighbor Sensitivity: {title_sig}  (dashed = mean)')
        figures.append((f'Neighbor sensitivity ({sig_name})', fig))

    # ── Fig 10: Per-position sensitivity ─────────────────────────────────
    has_any_pos = any(
        sensitivity.get(c) is not None and sensitivity[c].n_pairs_found > 0
        for c in ('unmethylated', 'methylated')
    )
    if has_any_pos:
        center     = K // 2
        positions  = list(range(K))
        pos_labels = [f'{p}[C]' if p == center else str(p) for p in positions]
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Mean |Δ IPD| per position',
                                            'Mean |Δ PW|  per position'])
        for col_i, (delta_attr, ylabel) in enumerate(
                [('delta_ipd', '|Δ IPD|'), ('delta_pw', '|Δ PW|')], start=1):
            for cat, color, name in [
                ('unmethylated', _meth_color(0), 'Unmethylated'),
                ('methylated',   '#EF553B',       'Methylated'),
            ]:
                ns = sensitivity.get(cat)
                if ns is None or ns.n_pairs_found == 0:
                    continue
                d     = getattr(ns, delta_attr)
                y_pos = []
                for pos in positions:
                    mask = ns.positions == pos
                    y_pos.append(float(np.mean(d[mask])) if np.any(mask) else 0.0)
                fig.add_trace(go.Bar(
                    x=pos_labels, y=y_pos, name=name,
                    legendgroup=name, showlegend=(col_i == 1),
                    marker_color=color, opacity=0.8,
                ), row=1, col=col_i)
        fig.update_layout(
            title=f'Per-Position Sensitivity  '
                  f'(pos {center}[C] = center / modified base)',
            barmode='group',
        )
        figures.append(('Per-position sensitivity', fig))

    # ── Fig 11: Meth vs unmeth sensitivity box ────────────────────────────
    has_both = (
        sensitivity.get('unmethylated') is not None
        and sensitivity['unmethylated'].n_pairs_found > 0
        and sensitivity.get('methylated') is not None
        and sensitivity['methylated'].n_pairs_found > 0
    )
    if has_both:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['|Δ IPD|: meth vs unmeth',
                                            '|Δ PW|:  meth vs unmeth'])
        for col_i, delta_attr in enumerate(['delta_ipd', 'delta_pw'], start=1):
            for ns, name, color in [
                (sensitivity['unmethylated'], 'Unmethylated', _meth_color(0)),
                (sensitivity['methylated'],   'Methylated',   '#EF553B'),
            ]:
                fig.add_trace(go.Box(
                    y=getattr(ns, delta_attr), name=name,
                    marker_color=color, boxmean='sd',
                    legendgroup=name, showlegend=(col_i == 1),
                ), row=1, col=col_i)
        fig.update_layout(
            title='Meth vs Unmeth: Neighbor Sensitivity Comparison',
            boxmode='group',
        )
        figures.append(('Sensitivity comparison', fig))

    # ── Fig: 3D Density Surface (IPD × PW × density per meth type) ─────
    try:
        from scipy.stats import gaussian_kde
        _HAS_SCIPY = True
    except ImportError:
        _HAS_SCIPY = False
        log.warning("scipy not installed — skipping 3D density surface.")

    if _HAS_SCIPY:
        # Build a shared grid in raw signal space
        all_ipd_vals = np.concatenate([
            stats.groups[m].ipd_means for m in meth_ids_sorted
        ])
        all_pw_vals = np.concatenate([
            stats.groups[m].pw_means for m in meth_ids_sorted
        ])
        ipd_lo = 0.0
        ipd_hi = max(float(np.percentile(all_ipd_vals, 99.5)), 150.0)
        pw_lo  = 0.0
        pw_hi  = max(float(np.percentile(all_pw_vals, 99.5)), 80.0)
        grid_n = 80
        ipd_grid = np.linspace(ipd_lo, ipd_hi, grid_n)
        pw_grid  = np.linspace(pw_lo, pw_hi, grid_n)
        ipd_mesh, pw_mesh = np.meshgrid(ipd_grid, pw_grid)
        grid_pts = np.vstack([ipd_mesh.ravel(), pw_mesh.ravel()])

        fig = go.Figure()
        for m in meth_ids_sorted:
            g = stats.groups[m]
            ipd_vals = g.ipd_means
            pw_vals  = g.pw_means

            # Apply m6A IPD cutoff if requested
            if m == 1 and min_ipd_m6a > 0:
                mask = ipd_vals >= min_ipd_m6a
                ipd_vals = ipd_vals[mask]
                pw_vals  = pw_vals[mask]
                log.info("3D plot: m6A filtered to IPD >= %.0f (%d -> %d keys)",
                         min_ipd_m6a, g.n_entries, len(ipd_vals))

            n = len(ipd_vals)
            if n < 50:
                continue
            # Subsample for KDE performance
            max_kde = 50_000
            if n > max_kde:
                idx = rng.choice(n, max_kde, replace=False)
                ipd_s = ipd_vals[idx]
                pw_s  = pw_vals[idx]
            else:
                ipd_s = ipd_vals
                pw_s  = pw_vals
            try:
                kde = gaussian_kde(np.vstack([ipd_s, pw_s]), bw_method=0.15)
                z   = kde(grid_pts).reshape(grid_n, grid_n)
            except np.linalg.LinAlgError:
                continue
            # Normalize to [0, 1] for consistent visual height
            z_max = z.max()
            if z_max > 0:
                z = z / z_max
            color = clr(m)
            fig.add_trace(go.Surface(
                x=ipd_grid, y=pw_grid, z=z,
                name=f'{lbl(m)} (n={n:,})', showscale=False,
                opacity=1.0,
                colorscale=[
                    [0.0, _darken_hex(color, 0.3)],
                    [0.3, _darken_hex(color, 0.6)],
                    [1.0, color],
                ],
                showlegend=True,
                contours=dict(
                    z=dict(show=True, usecolormap=True, project_z=False,
                           highlightcolor='white', highlightwidth=1),
                ),
            ))
        title_text = '3D Density Surface: IPD × PW per Methylation Type'
        if min_ipd_m6a > 0:
            title_text += f'<br><sub>m6A filtered: IPD >= {min_ipd_m6a:.0f}</sub>'
        fig.update_layout(
            title=dict(text=title_text, x=0.5),
            scene=dict(
                xaxis_title='Mean IPD',
                yaxis_title='Mean PW',
                zaxis_title='Normalized Density',
                camera=dict(eye=dict(x=1.8, y=-1.5, z=1.0)),
                bgcolor='rgb(240, 240, 240)',
            ),
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='rgba(0,0,0,0.3)',
                borderwidth=1,
            ),
        )
        figures.append(('3D density surface (per type)', fig))

        # ── Combined surface with height-based colormap ───────────────────
        # KDE on ALL data points together, single surface colored by Z
        max_kde_all = 100_000
        total = len(all_ipd_vals)
        if total > max_kde_all:
            idx_all = rng.choice(total, max_kde_all, replace=False)
            ipd_all_s = all_ipd_vals[idx_all]
            pw_all_s  = all_pw_vals[idx_all]
        else:
            ipd_all_s = all_ipd_vals
            pw_all_s  = all_pw_vals
        try:
            kde_all = gaussian_kde(np.vstack([ipd_all_s, pw_all_s]), bw_method=0.15)
            z_all   = kde_all(grid_pts).reshape(grid_n, grid_n)

            fig2 = go.Figure(go.Surface(
                x=ipd_grid, y=pw_grid, z=z_all,
                colorscale='Jet',
                colorbar=dict(title='Density'),
                opacity=0.95,
            ))
            fig2.update_layout(
                title='3D Density Surface: Combined IPD × PW (all methylation types)',
                scene=dict(
                    xaxis_title='Mean IPD',
                    yaxis_title='Mean PW',
                    zaxis_title='Density',
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
                ),
            )
            figures.append(('3D density surface (combined)', fig2))
        except np.linalg.LinAlgError:
            log.warning("KDE failed for combined surface — skipping.")

    # ── Assemble HTML ─────────────────────────────────────────────────────
    meth_summary = ', '.join(
        f'{lbl(m)}: {stats.groups[m].n_entries:,}' for m in meth_ids_sorted
    )
    cards = [
        ('K-mer size',     f'K={K}'),
        ('Total keys',     f'{stats.total_entries:,}'),
        ('Total samples',  f'{stats.total_samples:,}'),
        ('Coverage',       _pct(stats.total_entries, stats.total_possible_kmers)),
        ('Meth types',     str(len(stats.groups))),
        ('File size',      f'{stats.file_size_mb:.1f} MB'),
    ]
    card_html = '\n'.join(
        f'<div class="stat-card">'
        f'<div class="val">{v}</div>'
        f'<div class="lbl">{k}</div>'
        f'</div>'
        for k, v in cards
    )
    nav_links = '\n'.join(
        f'<a href="#section-{i}">{title}</a>'
        for i, (title, _) in enumerate(figures)
    )

    html_parts = [f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KinSim Analysis — {Path(stats.pkl_path).name}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f0f2f5;color:#333}}
.page{{max-width:1600px;margin:0 auto;padding:24px}}
.header{{background:linear-gradient(135deg,#1a2a4a,#2c5282);color:#fff;padding:32px;border-radius:12px;margin-bottom:24px}}
.header h1{{font-size:1.9em;margin-bottom:6px}}
.header p{{opacity:.8;font-size:.9em}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:24px}}
.stat-card{{background:#fff;padding:20px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);text-align:center}}
.stat-card .val{{font-size:1.7em;font-weight:700;color:#2c5282}}
.stat-card .lbl{{color:#718096;margin-top:6px;font-size:.82em}}
.nav{{background:#fff;padding:14px 20px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:24px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}}
.nav span{{font-weight:600;color:#4a5568}}
.nav a{{color:#3182ce;text-decoration:none;font-size:.88em;padding:3px 10px;border:1px solid #bee3f8;border-radius:20px}}
.nav a:hover{{background:#ebf8ff}}
.section{{background:#fff;padding:24px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:24px}}
.section h2{{color:#2c5282;border-bottom:3px solid #3182ce;padding-bottom:10px;margin-bottom:18px}}
.plot-box{{width:100%;min-height:480px}}
.footer{{text-align:center;color:#a0aec0;font-size:.8em;padding:20px 0}}
</style></head><body>
<div class="page">
<div class="header">
  <h1>KinSim Training Data Analysis</h1>
  <p>{Path(stats.pkl_path).name}</p>
  <p style="margin-top:6px;opacity:.65">{meth_summary}</p>
</div>
<div class="stat-grid">{card_html}</div>
<div class="nav"><span>Jump to:</span>
{nav_links}
</div>
"""]

    for i, (title, fig) in enumerate(figures):
        plot_json = pio.to_json(fig)
        html_parts.append(f"""
<div class="section" id="section-{i}">
  <h2>{title}</h2>
  <div id="plot_{i}" class="plot-box"></div>
  <script>(function(){{
    var spec = {plot_json};
    spec.layout = spec.layout || {{}};
    spec.layout.autosize = true;
    Plotly.newPlot('plot_{i}', spec.data, spec.layout,
                   {{responsive:true, displayModeBar:true}});
  }})();</script>
</div>""")

    html_parts.append("""
<div class="footer">Generated by KinSim &mdash; kinsim analyze</div>
</div></body></html>""")

    with open(output_path, 'w') as fh:
        fh.write('\n'.join(html_parts))
    log.info("HTML report saved: %s", output_path)

    # ── Export individual figures as standalone HTML files ─────────────────
    export_dir = Path(output_path).parent / 'figures'
    export_dir.mkdir(parents=True, exist_ok=True)
    has_kaleido = True
    try:
        import kaleido  # noqa: F401
    except ImportError:
        has_kaleido = False

    for i, (title, fig) in enumerate(figures):
        slug = title.lower().replace(' ', '_').replace('/', '_')
        slug = ''.join(c for c in slug if c.isalnum() or c == '_')
        fig_path = export_dir / f'{i:02d}_{slug}.html'
        pio.write_html(fig, str(fig_path), include_plotlyjs='cdn')
        if has_kaleido:
            png_path = export_dir / f'{i:02d}_{slug}.png'
            fig.write_image(str(png_path), scale=3)
    if has_kaleido:
        log.info("Individual figures exported to: %s/  (%d HTML + %d PNG)", export_dir, len(figures), len(figures))
    else:
        log.info("Individual figures exported to: %s/  (%d HTML, install kaleido for PNG)", export_dir, len(figures))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def analyze_pkl(
    pkl_path: str,
    output_dir: str | None = None,
    no_html: bool = False,
    max_scatter: int = 10_000,
    max_neighbor_entries: int = 200_000,
    min_ipd_m6a: float = 0.0,
) -> None:
    """Load a .pkl file, compute statistics, write TXT + HTML reports."""
    t0 = time.time()
    pkl_path = str(Path(pkl_path).resolve())

    if output_dir is None:
        output_dir = str(Path(pkl_path).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    base      = Path(pkl_path).stem
    txt_path  = str(Path(output_dir) / f'{base}_report.txt')
    html_path = str(Path(output_dir) / f'{base}_report.html')

    log.info("Loading: %s", pkl_path)
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)

    if not data:
        log.error("File is empty: %s", pkl_path)
        return

    log.info("Collecting statistics...")
    stats = collect_stats(data, pkl_path)
    log.info(
        "kmer_size: %d  keys: %d  samples: %d  types: %s",
        stats.kmer_size,
        stats.total_entries, stats.total_samples,
        [stats.groups[m].name for m in sorted(stats.groups)],
    )

    sensitivity: dict = {}
    if max_neighbor_entries != 0:
        log.info(
            "Computing neighbor sensitivity (cap=%d per category)...",
            max_neighbor_entries,
        )
        sensitivity = compute_neighbor_sensitivity(
            data, stats, max_entries=max_neighbor_entries,
        )
        for cat, ns in sensitivity.items():
            log.info(
                "  %s: %d source kmers, %d neighbor pairs",
                cat, ns.n_source_entries, ns.n_pairs_found,
            )
    else:
        log.info("Skipping neighbor sensitivity (--max-neighbor-entries 0)")

    log.info("Computing kinetic signature profiles per meth type...")
    sig_profiles = compute_signature_profiles(data, kmer_size=stats.kmer_size)

    print()
    render_txt_report(stats, sensitivity, txt_path, signature_profiles=sig_profiles)

    if not no_html:
        log.info("Generating HTML report...")
        render_html_report(stats, sensitivity, html_path, max_scatter=max_scatter, min_ipd_m6a=min_ipd_m6a)

    log.info("Analysis complete in %.1fs", time.time() - t0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse
    from .utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog='kinsim analyze',
        description=(
            'Comprehensive analysis of a KinSim training .pkl file.\n\n'
            'Auto-detects kmer size and all methylation types present.\n'
            'Works on:\n'
            '  · kinsim extract shards  (*_shard.pkl)\n'
            '  · kinsim merge output    (master_data.pkl)\n'
            '  · kinsim-prep filter output  (training_data.pkl)\n\n'
            'Outputs:\n'
            '  <basename>_report.txt   — text report (stdout + file)\n'
            '  <basename>_report.html  — interactive Plotly charts\n'
            '                           (requires: pip install plotly)\n\n'
            'Analysis sections:\n'
            '  · Per-type coverage and sample count distributions\n'
            '  · IPD/PW signal mean and sigma distributions\n'
            '  · Methylation fraction distributions\n'
            '  · Low-coverage key counts (n < 5/10/50)\n'
            '  · 1-base neighbor sensitivity + per-position profile\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('pkl',
                        help='Path to .pkl shard, merged dict, or balanced dict')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory (default: same dir as pkl)')
    parser.add_argument('--no-html', action='store_true',
                        help='Skip HTML report (text only, no plotly needed)')
    parser.add_argument('--max-scatter', type=int, default=10_000, metavar='N',
                        help='Max points per group in IPD-vs-PW scatter (default: 10000)')
    parser.add_argument('--max-neighbor-entries', type=int, default=200_000, metavar='N',
                        help='Max kmers per category for neighbor sensitivity '
                             '(default: 200000; 0 = skip)')
    parser.add_argument('--min-ipd-m6a', type=float, default=0.0, metavar='X',
                        help='Min mean IPD for m6A keys in 3D plot (default: 0 = no filter)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable DEBUG-level logging')
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    analyze_pkl(
        pkl_path=args.pkl,
        output_dir=args.output_dir,
        no_html=args.no_html,
        max_scatter=args.max_scatter,
        max_neighbor_entries=args.max_neighbor_entries,
        min_ipd_m6a=args.min_ipd_m6a,
    )


if __name__ == '__main__':
    main()
