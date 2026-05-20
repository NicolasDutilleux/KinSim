"""Analyse a KinSim training shard or refined directory.

Storage format from ``kinsim.utils.sample_layout``:

    dict[kmer_id (int)] -> np.ndarray(N, 20)

Cols 17/18/19 carry CATEGORY (0=baseline, 1=slowed, 2=near_meth),
PARENT_METH (meth_id of the parent methylation), and PARENT_OFFSET
(this row's distance from the parent meth position). Buckets are
emitted **per (meth_type, offset)** — e.g. ``slowed_by_m6A_at_+0`` and
``slowed_by_m6A_at_+5`` are reported separately so a noisy offset of a
meth type cannot mask a clean offset of the same type in the plots.

Outputs (to ``--output-dir`` or the input's directory by default):

    <basename>_report.txt   text report (also printed to stdout)
    <basename>_report.html  Plotly figures + report header

The HTML report is a focused 5-figure verification dashboard:
    1. IPD distribution per category (with refine threshold)
    2. Per-kmer baseline-mean distribution (showing where the threshold sits)
    3. Kinetic signature profiles (offsets 0..+8) per (meth, signature offset) bucket
    4. Sample counts per (meth, offset) bucket
    5. 3D joint (IPD, PW) KDE per bucket

CLI:

    kinsim analyze master_clean.pkl
    kinsim analyze master_clean.pkl --output-dir reports/ --no-html
"""

from __future__ import annotations

import dataclasses
import io
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from .utils.encoding import METH_IDS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _id_to_name() -> dict[int, str]:
    """Dynamic id→name from YAML — picks up user-added meth types at call time.

    Called per-figure so a YAML edit between analyze invocations is reflected
    without re-importing the module.
    """
    from .utils.encoding import get_meth_ids
    return {v: k for k, v in get_meth_ids().items()}


_ID_TO_NAME = _id_to_name()  # back-compat snapshot for callers that read it directly

# Plotly palette — cycles for any number of meth types.
_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def _meth_name(meth_id: int) -> str:
    return _ID_TO_NAME.get(meth_id, f"meth_id={meth_id}")


def _meth_color(meth_id: int) -> str:
    return _COLORS[meth_id % len(_COLORS)]


def _darken_hex(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgb({int(r * factor)},{int(g * factor)},{int(b * factor)})"


def _pct(val, total) -> str:
    return f"{100.0 * val / total:.2f}%" if total else "n/a"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MethGroupStats:
    """Per-meth-type aggregate stats (one row per kmer)."""

    meth_id: int
    name: str
    n_entries: int
    sample_counts: np.ndarray
    ipd_means: np.ndarray
    ipd_sigmas: np.ndarray
    pw_means: np.ndarray
    pw_sigmas: np.ndarray
    fraction_means: np.ndarray
    kmer_ids: np.ndarray


@dataclasses.dataclass
class DictStats:
    """Top-level stats for one .pkl."""

    pkl_path: str
    kmer_size: int
    total_possible_kmers: int
    total_entries: int
    total_samples: int
    groups: dict
    file_size_mb: float
    meta: dict


# ---------------------------------------------------------------------------
# Format check
# ---------------------------------------------------------------------------


def _check_v4_input(data: dict) -> None:
    """Fail fast on inputs that are not in the current 20-col layout."""
    for k, v in data.items():
        if k == "__meta__":
            continue
        if not isinstance(k, (int, np.integer)):
            raise ValueError(
                f"Input is not in the current int-keyed layout: got key type "
                f"{type(k).__name__}, expected int kmer_id. Re-run "
                f"`kinsim extract`."
            )
        if not isinstance(v, np.ndarray) or v.ndim != 2:
            raise ValueError(f"Input value for kmer {k!r} is not a 2D ndarray.")
        return  # one probe is enough


def _detect_kmer_size(data: dict, meta: dict) -> int:
    """Infer kmer size, preferring the source of truth in ``__meta__``.

    Resolution order:
      1. ``meta["extraction_params"]["kmer_size"]`` (post-v0.5 shards).
      2. ``meta["kmer_size"]`` (legacy direct field, kept for back-compat).
      3. Bit-length of the largest int key (fallback for unannotated shards).
      4. ``11`` if the shard is otherwise empty.
    """
    if meta:
        ext = meta.get("extraction_params") or {}
        if "kmer_size" in ext:
            return int(ext["kmer_size"])
        if "kmer_size" in meta:
            return int(meta["kmer_size"])
    max_kmer = 0
    for k in data:
        if k == "__meta__":
            continue
        if isinstance(k, (int, np.integer)) and int(k) > max_kmer:
            max_kmer = int(k)
    if max_kmer == 0:
        return 11
    bits = int(max_kmer).bit_length()
    return max(1, (bits + 1) // 2)


# ---------------------------------------------------------------------------
# Per-category IPD/PW distribution
# ---------------------------------------------------------------------------


def compute_category_distributions(data: dict) -> dict:
    """Per-category IPD/PW distribution stats.

    Returns ``dict[category_name] -> {n, ipd_mean, ipd_std, ipd_quantiles,
    ipd_max, pw_mean, pw_std, pw_quantiles, ipd_hist, ipd_hist_edges}``.

    Resolves column indices from the shard's ``__meta__["extraction_params"]``
    when present; falls back to the legacy K=11 layout otherwise.
    """
    from .data.dataset import read_shard_extraction_params
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NAMES,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PW,
        get_sample_layout,
    )

    params = read_shard_extraction_params(data)
    layout = get_sample_layout(params) if params is not None else None
    col_cat = layout.col_category if layout else COL_CATEGORY
    col_ipd = layout.col_ipd if layout else COL_IPD
    col_pw = layout.col_pw if layout else COL_PW

    by_cat: dict[int, list] = {0: [], 1: [], 2: []}
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= col_cat:
            continue
        cats = arr[:, col_cat].astype(np.int8)
        for cat_id in (CATEGORY_BASELINE, CATEGORY_SLOWED, CATEGORY_NEAR_METH):
            mask = cats == cat_id
            if mask.any():
                by_cat[cat_id].append(arr[mask][:, [col_ipd, col_pw]])

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256], dtype=np.float32)
    qs = (5, 25, 50, 75, 90, 95, 99)
    out: dict = {}
    for cat_id, chunks in by_cat.items():
        if not chunks:
            continue
        pooled = np.concatenate(chunks, axis=0)
        ipd = pooled[:, 0].astype(np.float32)
        pw = pooled[:, 1].astype(np.float32)
        h, _ = np.histogram(ipd, bins=bins)
        out[CATEGORY_NAMES[cat_id]] = {
            "n": len(ipd),
            "ipd_mean": float(ipd.mean()),
            "ipd_std": float(ipd.std()),
            "ipd_quantiles": {p: float(np.percentile(ipd, p)) for p in qs},
            "ipd_max": float(ipd.max()),
            "pw_mean": float(pw.mean()),
            "pw_std": float(pw.std()),
            "pw_quantiles": {p: float(np.percentile(pw, p)) for p in qs},
            "ipd_hist": h.astype(np.int64),
            "ipd_hist_edges": bins,
        }
    return out


# ---------------------------------------------------------------------------
# Kinetic signature profiles (per bucket)
# ---------------------------------------------------------------------------


def compute_signature_profiles(data: dict, kmer_size: int = 11) -> dict:
    """Aggregate IPD/PW means per (category, parent_meth, parent_offset) bucket.

    Buckets: ``baseline``, ``slowed_by_<T>_at_+<off>``, ``near_meth_by_<T>_at_+<off>``.
    Splitting per (T, offset) keeps noisy offsets (e.g. m6A@+5 from a
    Type I R-M motif that doesn't actually carry +5) inspectable
    independently from clean ones (m6A@+0 from GATC).

    Returns ``dict[bucket] -> {mean_ipd, mean_pw, n_samples}`` — scalars,
    not per-offset arrays. The legacy 9-position downstream profile was
    dropped: with PARENT_OFFSET written at extract time, comparing the
    mean IPD across buckets is the cleaner diagnostic.
    """
    from .data.dataset import read_shard_extraction_params
    from .utils.encoding import get_meth_ids
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PARENT_METH,
        COL_PARENT_OFFSET,
        COL_PW,
        get_sample_layout,
    )

    # Resolve column layout from shard meta (new shards) or fall back to
    # the legacy K=11 module constants. Same pattern as verify_generate.py.
    params = read_shard_extraction_params(data)
    layout = get_sample_layout(params) if params is not None else None
    col_cat = layout.col_category if layout else COL_CATEGORY
    col_ipd = layout.col_ipd if layout else COL_IPD
    col_pw = layout.col_pw if layout else COL_PW
    col_parent_meth = layout.col_parent_meth if layout else COL_PARENT_METH
    col_parent_offset = layout.col_parent_offset if layout else COL_PARENT_OFFSET

    meth_ids = get_meth_ids()
    name_by_mid = {v: k for k, v in meth_ids.items()}

    baseline_acc = [0.0, 0.0, 0]  # sum_ipd, sum_pw, n
    slowed_acc: dict[tuple[int, int], list] = {}
    near_acc: dict[tuple[int, int], list] = {}

    for kid, v in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(v, np.ndarray):
            continue
        if v.shape[1] <= col_parent_offset:
            continue
        cats = v[:, col_cat].astype(np.int8)
        parent = v[:, col_parent_meth].astype(np.int8)
        offset = v[:, col_parent_offset].astype(np.int8)
        ipd = v[:, col_ipd]
        pw = v[:, col_pw]

        base_m = cats == CATEGORY_BASELINE
        if base_m.any():
            baseline_acc[0] += float(ipd[base_m].sum())
            baseline_acc[1] += float(pw[base_m].sum())
            baseline_acc[2] += int(base_m.sum())

        for cat_id, acc_dict in (
            (CATEGORY_SLOWED, slowed_acc),
            (CATEGORY_NEAR_METH, near_acc),
        ):
            m_cat = cats == cat_id
            if not m_cat.any():
                continue
            for T_id in np.unique(parent[m_cat]):
                T_int = int(T_id)
                if T_int == 0:
                    continue
                m_T = m_cat & (parent == T_int)
                if not m_T.any():
                    continue
                for off in np.unique(offset[m_T]):
                    O_int = int(off)
                    mask = m_T & (offset == O_int)
                    if not mask.any():
                        continue
                    acc = acc_dict.setdefault((T_int, O_int), [0.0, 0.0, 0])
                    acc[0] += float(ipd[mask].sum())
                    acc[1] += float(pw[mask].sum())
                    acc[2] += int(mask.sum())

    def _pack(s_i: float, s_p: float, n: int) -> dict:
        return {
            "mean_ipd": float(s_i / max(n, 1)),
            "mean_pw": float(s_p / max(n, 1)),
            "n_samples": n,
        }

    out: dict = {}
    if baseline_acc[2] > 0:
        out["baseline"] = _pack(baseline_acc[0], baseline_acc[1], baseline_acc[2])
    for (T_id, off), (s_i, s_p, n) in slowed_acc.items():
        mname = name_by_mid.get(T_id, f"meth{T_id}")
        out[_bucket_name("slowed_by_", mname, off)] = _pack(s_i, s_p, n)
    for (T_id, off), (s_i, s_p, n) in near_acc.items():
        mname = name_by_mid.get(T_id, f"meth{T_id}")
        out[_bucket_name("near_meth_by_", mname, off)] = _pack(s_i, s_p, n)
    return out


# ---------------------------------------------------------------------------
# Methylation-context distribution per bucket
# ---------------------------------------------------------------------------


def compute_meth_context_distribution(data: dict, kmer_size: int | None = None) -> dict:
    """For each (category, parent_meth, parent_offset) bucket, count
    meth_id occurrences at each meth_context offset.

    Parent attribution is read directly from ``COL_PARENT_METH`` /
    ``COL_PARENT_OFFSET`` and the inner accumulation is fully
    vectorised (one boolean mask per meth_id over all rows of the
    bucket). Splitting per-offset surfaces things like "the m6A flag at
    +5 sees meth_context A at the wrong column" — a clue that the
    expected signature offset for that meth type doesn't apply to the
    motif that produced this row.

    Returns ``dict[bucket] -> {counts, fractions, n_samples, meth_ids, meth_names}``.
    """
    from .utils.encoding import get_meth_ids
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_PARENT_METH,
        COL_PARENT_OFFSET,
        METH_CTX_LEN as _LEGACY_METH_CTX_LEN,
    )

    meth_ids = get_meth_ids()
    meth_id_list = sorted(set(meth_ids.values()))
    name_by_mid = {v: k for k, v in meth_ids.items()}
    nmid = len(meth_id_list)

    # K-aware: prefer caller-provided kmer_size, fall back to the legacy
    # K=11 constant. The mc slice is ``v[:, 3 : 3 + mc_len]``.
    mc_len = int(kmer_size) if kmer_size else _LEGACY_METH_CTX_LEN

    def _empty():
        return {
            "counts": np.zeros((mc_len, nmid), dtype=np.int64),
            "n_samples": 0,
        }

    def _accumulate(bkt: dict, mc_arr: np.ndarray) -> None:
        """Vectorised per-position meth_id counts.

        For a (n, mc_len) mc_arr, ``(mc_arr == mid).sum(axis=0)``
        is the per-position count of meth_id ``mid`` — one numpy reduction
        per meth_id, NMID total (~4). No Python row loop.
        """
        if len(mc_arr) == 0:
            return
        for i, mid in enumerate(meth_id_list):
            bkt["counts"][:, i] += (mc_arr == mid).sum(axis=0).astype(np.int64)
        bkt["n_samples"] += len(mc_arr)

    baseline_b: dict = _empty()
    slowed_buckets: dict[tuple[int, int], dict] = {}
    near_buckets: dict[tuple[int, int], dict] = {}

    for kid, v in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(v, np.ndarray):
            continue
        if v.shape[1] <= COL_PARENT_OFFSET:
            continue
        cats = v[:, COL_CATEGORY].astype(np.int8)
        parent = v[:, COL_PARENT_METH].astype(np.int8)
        offset = v[:, COL_PARENT_OFFSET].astype(np.int8)
        mc = v[:, 3 : 3 + mc_len].astype(np.int32)

        base_mask = cats == CATEGORY_BASELINE
        if base_mask.any():
            _accumulate(baseline_b, mc[base_mask])

        for cat_id, bucket_dict in (
            (CATEGORY_SLOWED, slowed_buckets),
            (CATEGORY_NEAR_METH, near_buckets),
        ):
            m_cat = cats == cat_id
            if not m_cat.any():
                continue
            for T_id in np.unique(parent[m_cat]):
                T_int = int(T_id)
                if T_int == 0:
                    continue
                m_T = m_cat & (parent == T_int)
                if not m_T.any():
                    continue
                for off in np.unique(offset[m_T]):
                    O_int = int(off)
                    mask = m_T & (offset == O_int)
                    if not mask.any():
                        continue
                    _accumulate(bucket_dict.setdefault((T_int, O_int), _empty()), mc[mask])

    final: dict = {}

    def _finalise(bkt: dict) -> dict:
        bkt["fractions"] = bkt["counts"] / max(bkt["n_samples"], 1)
        bkt["meth_ids"] = meth_id_list
        bkt["meth_names"] = [name_by_mid.get(m, f"meth{m}") for m in meth_id_list]
        return bkt

    if baseline_b["n_samples"] > 0:
        final["baseline"] = _finalise(baseline_b)
    for (T_id, off), bkt in slowed_buckets.items():
        if bkt["n_samples"] > 0:
            mname = name_by_mid.get(T_id, f"meth{T_id}")
            final[_bucket_name("slowed_by_", mname, off)] = _finalise(bkt)
    for (T_id, off), bkt in near_buckets.items():
        if bkt["n_samples"] > 0:
            mname = name_by_mid.get(T_id, f"meth{T_id}")
            final[_bucket_name("near_meth_by_", mname, off)] = _finalise(bkt)
    return final


# ---------------------------------------------------------------------------
# Top-level stats collection
# ---------------------------------------------------------------------------


def collect_stats(data: dict, pkl_path: str) -> DictStats:
    """One-pass per-key statistics, partitioned by meth_id at the centre.

    Memory-efficient: never materialises ``v[mask]`` sub-arrays. Per-key
    stats (mean, std, count, fraction) are computed inline via
    ``np.mean(arr, where=mask)`` / ``np.std(arr, where=mask)`` — these
    take a boolean ``where`` argument and skip masked elements without
    allocating an intermediate copy. Inner partition iterates the four
    known meth_ids ``{0, 1, 2, 3}`` instead of calling ``np.unique`` per
    kmer (saves the per-kmer allocation and a Python attribute lookup).

    Memory: ~1.3× pkl size (vs ~6× when partition copies were stored).
    """
    from .utils.encoding import KMER_PRED_IDX

    meta = data.get("__meta__", {})
    if not isinstance(meta, dict):
        meta = {}

    kmer_size = _detect_kmer_size(data, meta)
    total_possible = 4**kmer_size

    # Known meth_ids — dynamic from YAML so user-added types appear.
    from .utils.encoding import get_meth_ids as _gmi
    KNOWN_METH_IDS = tuple(sorted(_gmi().values()))

    # Pre-allocate per-meth scalar lists (cheap append, np.array() at end).
    # No ndarray slices are stored — only per-(kmer, meth) scalar stats.
    per_meth: dict[int, dict[str, list]] = {
        mid: {
            "kmer_ids": [],
            "n": [],
            "mu_ipd": [],
            "sig_ipd": [],
            "mu_pw": [],
            "sig_pw": [],
            "frac": [],
        }
        for mid in KNOWN_METH_IDS
    }

    total_samples = 0
    for kid, v in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(v, np.ndarray):
            continue
        # Need at least IPD/PW/frac + mc-up-to-prediction-index. 3 + KMER_PRED_IDX + 1
        # = 11 for K=11 (legacy: ``< 14`` was a magic constant from an older layout).
        if v.shape[1] < 3 + KMER_PRED_IDX + 1:
            continue
        # One column read; int8 enough for meth_ids in [0, 3].
        center = v[:, 3 + KMER_PRED_IDX].astype(np.int8)
        ipd_col = v[:, 0]
        pw_col = v[:, 1]
        frac_col = v[:, 2]
        for mid in KNOWN_METH_IDS:
            mask = center == mid
            n = int(mask.sum())
            if n == 0:
                continue
            s = per_meth[mid]
            s["kmer_ids"].append(int(kid))
            s["n"].append(n)
            s["mu_ipd"].append(float(np.mean(ipd_col, where=mask)))
            s["sig_ipd"].append(float(np.std(ipd_col, where=mask)))
            s["mu_pw"].append(float(np.mean(pw_col, where=mask)))
            s["sig_pw"].append(float(np.std(pw_col, where=mask)))
            s["frac"].append(float(np.mean(frac_col, where=mask)))
            total_samples += n

    groups: dict[int, MethGroupStats] = {}
    for mid, s in per_meth.items():
        if not s["n"]:
            continue
        groups[mid] = MethGroupStats(
            meth_id=mid,
            name=_meth_name(mid),
            n_entries=len(s["n"]),
            sample_counts=np.asarray(s["n"], dtype=np.float64),
            ipd_means=np.asarray(s["mu_ipd"], dtype=np.float64),
            ipd_sigmas=np.asarray(s["sig_ipd"], dtype=np.float64),
            pw_means=np.asarray(s["mu_pw"], dtype=np.float64),
            pw_sigmas=np.asarray(s["sig_pw"], dtype=np.float64),
            fraction_means=np.asarray(s["frac"], dtype=np.float64),
            kmer_ids=np.asarray(s["kmer_ids"], dtype=np.int64),
        )

    n_data_keys = sum(g.n_entries for g in groups.values())
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
# TXT report
# ---------------------------------------------------------------------------


def _bucket_order_key(name: str) -> tuple:
    """Display order: baseline, then slowed_by_<T>_at_+<off>, then near_meth_by_<T>_at_+<off>.

    Within each category, sort by meth-type then by offset (numerical,
    so ``+10`` doesn't sort before ``+2``).
    """
    if name == "baseline":
        return (0, "", 0)
    if name.startswith("slowed_by_"):
        prefix = 1
        body = name[len("slowed_by_") :]
    elif name.startswith("near_meth_by_"):
        prefix = 2
        body = name[len("near_meth_by_") :]
    else:
        return (3, name, 0)
    if "_at_" in body:
        meth_part, off_part = body.rsplit("_at_", 1)
        try:
            off_int = int(off_part)
        except ValueError:
            off_int = 0
        return (prefix, meth_part, off_int)
    return (prefix, body, 0)


def _bucket_name(prefix: str, meth_name: str, offset: int) -> str:
    """Canonical analyze bucket name: ``slowed_by_m6A_at_+0`` etc."""
    return f"{prefix}{meth_name}_at_{offset:+d}"


def render_txt_report(
    stats: DictStats,
    output_path: str,
    signature_profiles: dict | None = None,
    context_distribution: dict | None = None,
    category_distributions: dict | None = None,
    refine_meta: dict | None = None,
) -> None:
    """Write a plain-text analysis report to *output_path* and stdout."""
    buf = io.StringIO()
    K = stats.kmer_size

    def p(line: str = "") -> None:
        print(line)
        buf.write(line + "\n")

    W = 72

    # ── 1. Overview ─────────────────────────────────────────────────────
    p("=" * W)
    p("KinSim — Training Data Analysis Report")
    p("=" * W)
    p(f"File          : {stats.pkl_path}")
    p(f"K-mer size    : {K}  (4^{K} = {stats.total_possible_kmers:,} possible)")
    p(f"File size     : {stats.file_size_mb:.1f} MB")
    p(f"Total keys    : {stats.total_entries:,}")
    p(f"Total samples : {stats.total_samples:,}")
    p(f"Coverage      : {_pct(stats.total_entries, stats.total_possible_kmers)}")
    if stats.meta:
        if "created" in stats.meta:
            p(f"Created       : {stats.meta['created']}")
        src = stats.meta.get("source_bam", stats.meta.get("merged_from", ""))
        if src:
            s = (
                src
                if isinstance(src, str)
                else ", ".join(src[:3]) + ("..." if len(src) > 3 else "")
            )
            p(f"Source        : {s}")
    p()

    # ── 2. Per-meth coverage ────────────────────────────────────────────
    p("-" * W)
    p("Per-methylation-state coverage  (partitioned by meth_id at centre)")
    p("-" * W)
    for meth_id in sorted(stats.groups.keys()):
        g = stats.groups[meth_id]
        label = "Unmethylated" if meth_id == 0 else f"Methylated ({g.name})"
        c = g.sample_counts
        p(f"\n  {label}:")
        p(
            f"    Unique {K}-mers : {g.n_entries:,} / {stats.total_possible_kmers:,}"
            f"  ({_pct(g.n_entries, stats.total_possible_kmers)})"
        )
        p(f"    Total samples  : {np.sum(c):,.0f}")
        p(
            f"    Samples/key    :"
            f" mean={np.mean(c):.1f}  median={np.median(c):.1f}"
            f"  p5={np.percentile(c, 5):.0f}  p25={np.percentile(c, 25):.0f}"
            f"  p75={np.percentile(c, 75):.0f}  p95={np.percentile(c, 95):.0f}"
        )
        p(f"                     min={np.min(c):.0f}  max={np.max(c):.0f}")
        if meth_id != 0:
            valid = g.fraction_means[~np.isnan(g.fraction_means)]
            if len(valid) > 0:
                p(
                    f"    Meth fraction  :"
                    f" mean={np.mean(valid):.3f}  median={np.median(valid):.3f}"
                    f"  p5={np.percentile(valid, 5):.3f}"
                    f"  p95={np.percentile(valid, 95):.3f}"
                )
    p()

    # ── 3. Signal statistics ────────────────────────────────────────────
    p("-" * W)
    p("Signal statistics  (distribution of per-key means and sigmas)")
    p("-" * W)
    for meth_id in sorted(stats.groups.keys()):
        g = stats.groups[meth_id]
        label = "Unmethylated" if meth_id == 0 else f"Methylated ({g.name})"
        p(f"\n  {label}:")
        for sig_name, means, sigmas in [
            ("IPD", g.ipd_means, g.ipd_sigmas),
            ("PW", g.pw_means, g.pw_sigmas),
        ]:
            p(
                f"    {sig_name} mean   :"
                f" mean={np.mean(means):.3f}  std={np.std(means):.3f}"
                f"  median={np.median(means):.3f}"
                f"  p5={np.percentile(means, 5):.3f}"
                f"  p95={np.percentile(means, 95):.3f}"
            )
            p(
                f"    {sig_name} sigma  :"
                f" mean={np.mean(sigmas):.3f}  std={np.std(sigmas):.3f}"
                f"  median={np.median(sigmas):.3f}"
            )
    p()

    # ── 4. Per-bucket IPD/PW means ──────────────────────────────────────
    if signature_profiles:
        p("-" * W)
        p("Per-bucket IPD/PW means  (one value per (category, meth, offset) bucket)")
        p("-" * W)
        p("Each line is the mean of the centre row's IPD/PW across all samples in")
        p("that bucket. SLOWED buckets should have higher IPD than baseline;")
        p("NEAR_METH should look baseline-like (meth in mc context, but no slowing).")

        baseline_profile = signature_profiles.get("baseline")
        b_ipd = baseline_profile["mean_ipd"] if baseline_profile else None
        for name in sorted(signature_profiles.keys(), key=_bucket_order_key):
            sp = signature_profiles[name]
            tag = ""
            if name == "baseline":
                tag = "  [reference]"
            elif name.startswith("near_meth_by_"):
                tag = "  [should look baseline-like]"
            elif name.startswith("slowed_by_"):
                tag = "  [should be HIGHER than baseline]"
            line = (
                f"  {name}{tag}\n"
                f"    n={sp['n_samples']:,}   IPD={sp['mean_ipd']:6.2f}   "
                f"PW={sp['mean_pw']:6.2f}"
            )
            if b_ipd and name != "baseline":
                delta = sp["mean_ipd"] - b_ipd
                ratio = sp["mean_ipd"] / max(b_ipd, 1.0)
                line += f"   Δ_IPD={delta:+6.2f}   ratio={ratio:5.2f}x"
            p(line)
        p()

    # ── 5. IPD distribution per category + refine threshold check ──────
    if category_distributions:
        p("-" * W)
        p("IPD distribution per category  (raw uint8 from PacBio fi/fp tags)")
        p("-" * W)
        threshold = None
        if refine_meta:
            stats_ref = refine_meta.get("stats") or {}
            threshold = stats_ref.get("threshold")
            sec_pct = stats_ref.get("secondary_percentile")
            if threshold is not None:
                p(
                    f"Refine secondary filter: p{sec_pct:g}(per-kmer baseline mean) "
                    f"= {threshold:.2f}"
                )
                ni = stats_ref.get("n_slowed_in", 0)
                nk = stats_ref.get("n_slowed_kept", 0)
                nd = stats_ref.get("n_slowed_dropped", 0)
                if ni:
                    p(
                        f"  slowed survival: {nk:,}/{ni:,} = {100.0 * nk / ni:.1f}% "
                        f"(dropped {nd:,} below threshold)"
                    )
            p()

        p(
            f"  {'category':<12s} {'n':>12s} {'mean':>7s} {'std':>7s} "
            f"{'p5':>5s} {'p25':>5s} {'p50':>5s} {'p75':>5s} {'p90':>5s} {'p95':>5s} "
            f"{'p99':>5s} {'max':>5s}"
        )
        for cat_name in ("baseline", "slowed", "near_meth"):
            d = category_distributions.get(cat_name)
            if d is None:
                continue
            q = d["ipd_quantiles"]
            p(
                f"  {cat_name:<12s} {d['n']:>12,d} {d['ipd_mean']:>7.2f} "
                f"{d['ipd_std']:>7.2f} {q[5]:>5.0f} {q[25]:>5.0f} {q[50]:>5.0f} "
                f"{q[75]:>5.0f} {q[90]:>5.0f} {q[95]:>5.0f} {q[99]:>5.0f} "
                f"{d['ipd_max']:>5.0f}"
            )
        p()

        bd = category_distributions.get("baseline")
        if bd is not None and bd["n"] > 0:
            p("  Baseline IPD histogram  (where the threshold sits)")
            edges = bd["ipd_hist_edges"]
            counts = bd["ipd_hist"]
            total = float(counts.sum()) or 1.0
            max_bar = 50
            max_pct = float(counts.max() / total * 100.0)
            for i in range(len(counts)):
                lo, hi = float(edges[i]), float(edges[i + 1])
                pct = float(counts[i]) / total * 100.0
                bar_len = int((pct / max_pct) * max_bar)
                bar = "#" * bar_len
                marker = ""
                if threshold is not None and lo <= threshold < hi:
                    marker = f"   <-- p{sec_pct:g} threshold = {threshold:.0f}"
                p(
                    f"    [{lo:5.0f}-{hi:5.0f})  {bar:<{max_bar}s}  {pct:5.2f}% "
                    f"({int(counts[i]):>10,d}){marker}"
                )
            p()

        p("  PW distribution per category (mean / median / p95):")
        for cat_name in ("baseline", "slowed", "near_meth"):
            d = category_distributions.get(cat_name)
            if d is None:
                continue
            q = d["pw_quantiles"]
            p(
                f"    {cat_name:<12s} mean={d['pw_mean']:5.1f}  "
                f"median={q[50]:5.1f}  p95={q[95]:5.1f}  std={d['pw_std']:5.2f}"
            )
        p()

        if threshold is not None and "slowed" in category_distributions:
            s = category_distributions["slowed"]
            min_kept = s["ipd_quantiles"][5]
            if min_kept < threshold:
                p(
                    f"  WARNING: kept slowed has IPD < threshold "
                    f"(p5={min_kept:.1f} < {threshold:.1f})"
                )
            else:
                p(
                    f"  Sanity OK: all kept slowed have IPD >= threshold "
                    f"({threshold:.1f}); minimum kept p5 = {min_kept:.1f}"
                )
            if "near_meth" in category_distributions:
                n = category_distributions["near_meth"]
                p(f"  near_meth median = {n['ipd_quantiles'][50]:.1f} (should look like baseline)")
        p()

    # ── 6. Methylation-context distribution per bucket ──────────────────
    if context_distribution:
        from .utils.encoding import KMER_PRED_IDX
        from .utils.sample_layout import METH_CTX_LEFT, METH_CTX_LEN

        # K-aware label width — derive from a real bucket's shape so K=21
        # shards get the right offset labels.
        first_cd = next(iter(context_distribution.values()), None)
        if first_cd is not None and first_cd.get("fractions") is not None:
            mc_len = int(first_cd["fractions"].shape[0])
        else:
            mc_len = METH_CTX_LEN
        # When K!=11 the left/right padding shifts. With the default
        # (upstream=7, downstream=3) we keep METH_CTX_LEFT=7 — for other
        # geometries the active site stays at KMER_PRED_IDX, so the offset
        # is ``i - KMER_PRED_IDX``.
        mc_left = METH_CTX_LEFT if mc_len == METH_CTX_LEN else KMER_PRED_IDX

        p("-" * W)
        p("Methylation context distribution  (which meth_id sits at each offset")
        p("around the prediction position, per bucket)")
        p("-" * W)
        p(f"Centre [C] sits at meth_context index {KMER_PRED_IDX} (offset 0).")
        p()
        pos_labels = []
        for i in range(mc_len):
            off = i - mc_left
            tag = "[C]" if off == 0 else ""
            pos_labels.append(f"{off:+d}{tag}")

        for name in sorted(context_distribution.keys(), key=_bucket_order_key):
            cd = context_distribution[name]
            n = cd["n_samples"]
            fractions = cd["fractions"]
            mids = cd["meth_ids"]
            mnames = cd["meth_names"]
            p(f"  {name}  (n={n:,} samples)")

            hdr = "    {:8s}".format("mc_pos")
            for lbl in pos_labels:
                hdr += f"{lbl:>7s}"
            p(hdr)

            for col_idx, (mid, mn) in enumerate(zip(mids, mnames)):
                row = fractions[:, col_idx]
                if not np.any(row > 0.0005):
                    continue
                row_label = f"{mn} (id={mid})"
                line = f"    {row_label:8s}"
                for v in row:
                    pct = v * 100.0
                    if pct >= 99.95:
                        line += f"{'100':>7s}"
                    elif pct < 0.05:
                        line += f"{'.':>7s}"
                    else:
                        line += f"{pct:6.1f} "
                p(line)
            p()

    # ── 7. Low-coverage warnings ────────────────────────────────────────
    p("-" * W)
    p("Low-coverage keys  (keys with few samples — less reliable signals)")
    p("-" * W)
    for meth_id in sorted(stats.groups.keys()):
        g = stats.groups[meth_id]
        label = "Unmethylated" if meth_id == 0 else f"Methylated ({g.name})"
        ne = g.n_entries
        n5 = int(np.sum(g.sample_counts < 5))
        n10 = int(np.sum(g.sample_counts < 10))
        n50 = int(np.sum(g.sample_counts < 50))
        p(
            f"  {label:30s}"
            f"  n<5: {n5:,} ({_pct(n5, ne)})"
            f"   n<10: {n10:,} ({_pct(n10, ne)})"
            f"   n<50: {n50:,} ({_pct(n50, ne)})"
        )
    p()
    p("=" * W)

    from .utils.io import atomic_write_text

    atomic_write_text(buf.getvalue(), output_path)
    print(f"\nTXT report saved: {output_path}")


# ---------------------------------------------------------------------------
# HTML report (Plotly)
# ---------------------------------------------------------------------------


def _build_html_figures(
    data: dict,
    category_distributions: dict,
    signature_profiles: dict,
    refine_meta: dict | None,
) -> list[tuple[str, object]]:
    """Build the four verification figures as ``(title, plotly.Figure)``."""
    import plotly.graph_objects as go

    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
    )

    figures: list[tuple[str, object]] = []
    cat_palette = {
        "baseline": "#1f77b4",
        "slowed": "#d62728",
        "near_meth": "#2ca02c",
    }

    threshold = None
    if refine_meta:
        threshold = (refine_meta.get("stats") or {}).get("threshold")

    # ── Fig 1: IPD distribution per category ────────────────────────────
    fig1 = go.Figure()
    for cat_name in ("baseline", "slowed", "near_meth"):
        d = category_distributions.get(cat_name)
        if d is None:
            continue
        ipds = []
        for kid, arr in data.items():
            if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
                continue
            if arr.shape[1] <= COL_CATEGORY:
                continue
            cats = arr[:, COL_CATEGORY].astype(np.int8)
            target = {
                "baseline": CATEGORY_BASELINE,
                "slowed": CATEGORY_SLOWED,
                "near_meth": CATEGORY_NEAR_METH,
            }[cat_name]
            mask = cats == target
            if mask.any():
                ipds.append(arr[mask, COL_IPD])
        if not ipds:
            continue
        ipds_pool = np.concatenate(ipds)
        if len(ipds_pool) > 200_000:
            idx = np.random.default_rng(0).choice(len(ipds_pool), 200_000, replace=False)
            ipds_pool = ipds_pool[idx]
        fig1.add_trace(
            go.Histogram(
                x=ipds_pool,
                name=f"{cat_name} (n={d['n']:,})",
                marker_color=cat_palette[cat_name],
                opacity=0.55,
                xbins=dict(start=0, end=256, size=4),
                histnorm="probability density",
            )
        )
    if threshold is not None:
        fig1.add_vline(
            x=threshold,
            line_color="black",
            line_dash="dash",
            annotation_text=f"threshold = {threshold:.1f}",
            annotation_position="top right",
        )
    fig1.update_layout(
        title="IPD distribution per category  "
        "(slowed should be right-shifted; near_meth should overlap baseline)",
        xaxis_title="IPD (uint8 [0, 255])",
        yaxis_title="Probability density",
        barmode="overlay",
    )
    figures.append(("IPD distribution per category", fig1))

    # ── Fig 2: Per-kmer baseline mean distribution + threshold ──────────
    kmer_means = []
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_CATEGORY:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        m = cats == CATEGORY_BASELINE
        if int(m.sum()) >= 5:
            kmer_means.append(float(arr[m, COL_IPD].mean()))
    if kmer_means:
        fig2 = go.Figure()
        kmer_means_arr = np.array(kmer_means, dtype=np.float32)
        fig2.add_trace(
            go.Histogram(
                x=kmer_means_arr,
                name=f"per-kmer baseline mean (n={len(kmer_means_arr):,})",
                marker_color="#1f77b4",
                xbins=dict(start=0, end=256, size=2),
            )
        )
        if threshold is not None:
            fig2.add_vline(
                x=threshold,
                line_color="black",
                line_dash="dash",
                annotation_text=f"p95 threshold = {threshold:.1f}",
                annotation_position="top right",
            )
        fig2.update_layout(
            title="Per-kmer baseline mean IPD  (threshold for slowed survival is the p95 of this)",
            xaxis_title="Mean baseline IPD per kmer",
            yaxis_title="Number of kmers",
        )
        figures.append(("Per-kmer baseline mean distribution", fig2))

    # ── Fig 3: Per-bucket mean IPD ──────────────────────────────────────
    if signature_profiles:
        ordered = sorted(signature_profiles.keys(), key=_bucket_order_key)
        means = [signature_profiles[n]["mean_ipd"] for n in ordered]
        ns = [signature_profiles[n]["n_samples"] for n in ordered]
        colors = [
            "#1f77b4" if n == "baseline" else "#d62728" if n.startswith("slowed_by_") else "#2ca02c"
            for n in ordered
        ]
        fig3 = go.Figure(
            go.Bar(
                x=ordered,
                y=means,
                marker_color=colors,
                text=[f"{m:.2f}" for m in means],
                textposition="auto",
                customdata=ns,
                hovertemplate="%{x}<br>mean IPD: %{y:.2f}<br>n: %{customdata:,}<extra></extra>",
            )
        )
        fig3.update_layout(
            title="Per-bucket mean IPD  "
            "(SLOWED should sit higher than baseline; NEAR_METH ≈ baseline)",
            yaxis_title="Mean IPD",
            xaxis_tickangle=-45,
        )
        figures.append(("Per-bucket mean IPD", fig3))

    # ── Fig 4: Sample counts per bucket ─────────────────────────────────
    if signature_profiles:
        names = sorted(signature_profiles.keys(), key=_bucket_order_key)
        counts = [signature_profiles[n]["n_samples"] for n in names]
        colors = [
            "#1f77b4" if n == "baseline" else "#d62728" if n.startswith("slowed_by_") else "#2ca02c"
            for n in names
        ]
        fig4 = go.Figure(
            go.Bar(
                x=names,
                y=counts,
                marker_color=colors,
                text=[f"{c:,}" for c in counts],
                textposition="auto",
            )
        )
        fig4.update_layout(
            title="Sample counts per bucket",
            yaxis_title="# samples",
            yaxis_type="log",
        )
        figures.append(("Sample counts per bucket", fig4))

    # ── Fig 5: 3D density surfaces — per-bucket joint (IPD, PW) KDE ─────
    fig5 = _build_ipd_pw_density_figure(data)
    if fig5 is not None:
        figures.append(("3D density: joint (IPD, PW) per bucket", fig5))

    # ── Fig 6: per-kmer IPD distribution for top-N kmers with methylation ─
    fig6 = _build_kmer_trend_figure(data, top_n=12)
    if fig6 is not None:
        figures.append(
            (
                "Per-kmer IPD distributions — top 12 by SLOWED count",
                fig6,
            )
        )

    # ── Fig 7+: two sections per meth type ─
    #   • top 12 by slowed-meth count — highest-signal kmers
    #   • 12 random kmers (≥ 50 slowed rows)  — sanity check that bimodality
    #     isn't a top-n artefact. Each subplot overlays a Gaussian fit
    #     (μ, σ from data) on top of the histogram so the user can see
    #     whether the slowed distribution is single- or multi-modal.
    for meth_name, panel_label, fig_m in _build_per_meth_kmer_figures(data, top_n=12):
        figures.append(
            (
                f"Per-kmer baseline vs slowed/{meth_name} — {panel_label}",
                fig_m,
            )
        )

    return figures


def _build_kmer_trend_figure(data: dict, top_n: int = 12):
    """Per-kmer IPD distribution for the N kmers carrying the most methylation.

    For each of the top N kmers (ranked by CATEGORY_SLOWED row count), the
    subplot overlays:
      - baseline rows (unmodified IPD)
      - one trace per parent_meth (m6A/m4C/m5C) of the SLOWED rows

    Reading the figure:
      - Clear separation between baseline and SLOWED peaks ⇒ the model
        has a clean signal to fit.
      - TWO distinct peaks on the SLOWED side (bimodal) ⇒ the kmer
        carries mixed populations (e.g. partial occupancy). A single
        Gaussian will average them and predict a wrong μ for both
        modes — the model is structurally underfit on that kmer.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .utils.encoding import decode_kmer, get_meth_ids
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PARENT_METH,
    )

    counts: list[tuple[int, int]] = []
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_PARENT_METH:
            continue
        n_slow = int((arr[:, COL_CATEGORY].astype(np.int8) == CATEGORY_SLOWED).sum())
        if n_slow > 0:
            counts.append((int(kid), n_slow))
    if not counts:
        return None
    counts.sort(key=lambda x: x[1], reverse=True)
    top = counts[:top_n]

    name_by_id = {v: k for k, v in get_meth_ids().items()}
    color_by_id = {1: "#d62728", 2: "#9467bd", 3: "#2ca02c", 4: "#ff7f0e"}

    cols = 3
    rows = (len(top) + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{decode_kmer(kid)}  (n_slow={n:,})" for kid, n in top],
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )

    legend_seen: set[str] = set()
    for i, (kid, _) in enumerate(top):
        r, c = i // cols + 1, i % cols + 1
        arr = data[kid]
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        parents = arr[:, COL_PARENT_METH].astype(np.int8)
        ipds = arr[:, COL_IPD]

        m_base = cats == CATEGORY_BASELINE
        if m_base.any():
            show = "baseline" not in legend_seen
            legend_seen.add("baseline")
            fig.add_trace(
                go.Histogram(
                    x=ipds[m_base],
                    name="baseline",
                    marker_color="#1f77b4",
                    opacity=0.5,
                    xbins=dict(start=0, end=256, size=4),
                    histnorm="probability density",
                    showlegend=show,
                    legendgroup="baseline",
                ),
                row=r,
                col=c,
            )

        m_slow = cats == CATEGORY_SLOWED
        for mid in sorted({int(p) for p in parents[m_slow]}):
            if mid == 0:
                continue
            m = m_slow & (parents == mid)
            if not m.any():
                continue
            label = f"slowed/{name_by_id.get(mid, mid)}"
            show = label not in legend_seen
            legend_seen.add(label)
            fig.add_trace(
                go.Histogram(
                    x=ipds[m],
                    name=label,
                    marker_color=color_by_id.get(mid, "#888"),
                    opacity=0.55,
                    xbins=dict(start=0, end=256, size=4),
                    histnorm="probability density",
                    showlegend=show,
                    legendgroup=label,
                ),
                row=r,
                col=c,
            )
        fig.update_xaxes(range=[0, 200], row=r, col=c)

    fig.update_layout(
        title=f"Top {len(top)} kmers by SLOWED count — bimodal SLOWED peaks ⇒ mixed populations the model can't fit with one Gaussian.",
        barmode="overlay",
        height=260 * rows,
    )
    return fig


def _build_per_meth_kmer_figures(data: dict, top_n: int = 12):
    """Two figures per meth type: top-N by slowed count, and N random kmers.

    Each subplot overlays:
      • baseline histogram (blue) + fitted Gaussian curve
      • slowed/T histogram (meth-typed colour) + fitted Gaussian curve
      • annotation with μ, σ for both distributions

    Diagnostic for the "bimodal slowed peak" question:
      - Slowed Gaussian fits the histogram well → unimodal, refine done.
      - Histogram has 2 bumps but only 1 Gaussian → bimodal, refine didn't
        clean. The Gaussian curve will sit awkwardly between the bumps.
      - 2 bumps on EVERY kmer of a given meth type → systematic substoichio
        or extract misattribution.

    Yields ``(meth_name, panel_label, plotly_figure)`` triples.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .utils.encoding import decode_kmer, get_meth_ids
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PARENT_METH,
    )

    name_by_id = {v: k for k, v in get_meth_ids().items()}
    color_by_id = {1: "#d62728", 2: "#9467bd", 3: "#2ca02c", 4: "#ff7f0e"}
    MIN_RANDOM_N = 50           # min slowed rows for a kmer to be eligible for the random panel
    GAUSS_X = np.arange(0, 201, dtype=np.float32)

    def _gauss_curve(values: np.ndarray):
        """Return (mu, sigma, y) for a 1-D Gaussian fitted by moments to `values`.

        Returns ``(None, None, None)`` if values is empty or sigma is 0.
        """
        if values.size == 0:
            return None, None, None
        mu = float(values.mean())
        sigma = float(values.std(ddof=0))
        if sigma <= 0:
            return mu, sigma, None
        y = np.exp(-0.5 * ((GAUSS_X - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return mu, sigma, y

    # Discover meth types present in the data.
    present: set[int] = set()
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_PARENT_METH:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        parents = arr[:, COL_PARENT_METH].astype(np.int8)
        present.update({int(p) for p in parents[cats == CATEGORY_SLOWED] if p > 0})

    rng = np.random.default_rng(42)
    for mid in sorted(present):
        meth_name = name_by_id.get(mid, f"meth{mid}")
        # Rank kmers by slowed-rows of THIS meth type.
        counts: list[tuple[int, int]] = []
        for kid, arr in data.items():
            if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
                continue
            if arr.shape[1] <= COL_PARENT_METH:
                continue
            cats = arr[:, COL_CATEGORY].astype(np.int8)
            parents = arr[:, COL_PARENT_METH].astype(np.int8)
            n = int(((cats == CATEGORY_SLOWED) & (parents == mid)).sum())
            if n > 0:
                counts.append((int(kid), n))
        if not counts:
            continue
        counts.sort(key=lambda x: x[1], reverse=True)

        # Top panel: highest-signal kmers.
        top_panel = counts[:top_n]
        # Random panel: uniformly drawn from kmers with ≥ MIN_RANDOM_N slowed
        # rows. Avoids pathological low-n kmers in the random sample.
        eligible = [c for c in counts if c[1] >= MIN_RANDOM_N]
        if len(eligible) > top_n:
            idx = rng.choice(len(eligible), size=top_n, replace=False)
            random_panel = [eligible[int(i)] for i in idx]
        else:
            random_panel = eligible

        panels = [
            (f"top {len(top_panel)} by slowed-{meth_name} count", top_panel),
            (f"{len(random_panel)} random (≥ {MIN_RANDOM_N} slowed rows)", random_panel),
        ]

        for panel_label, kmer_list in panels:
            if not kmer_list:
                continue

            cols = 3
            rows = (len(kmer_list) + cols - 1) // cols
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[
                    f"{decode_kmer(kid)}  (n_{meth_name}={n:,})" for kid, n in kmer_list
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.06,
            )

            legend_seen: set[str] = set()
            for i, (kid, _) in enumerate(kmer_list):
                r, c = i // cols + 1, i % cols + 1
                arr = data[kid]
                cats = arr[:, COL_CATEGORY].astype(np.int8)
                parents = arr[:, COL_PARENT_METH].astype(np.int8)
                ipds = arr[:, COL_IPD]

                m_base = cats == CATEGORY_BASELINE
                m_slow = (cats == CATEGORY_SLOWED) & (parents == mid)
                base_ipds = ipds[m_base]
                slow_ipds = ipds[m_slow]

                # Baseline histogram + Gaussian.
                if base_ipds.size > 0:
                    show = "baseline" not in legend_seen
                    legend_seen.add("baseline")
                    fig.add_trace(
                        go.Histogram(
                            x=base_ipds, name="baseline", marker_color="#1f77b4",
                            opacity=0.55, xbins=dict(start=0, end=256, size=4),
                            histnorm="probability density",
                            showlegend=show, legendgroup="baseline",
                        ),
                        row=r, col=c,
                    )
                    mu_b, sg_b, y_b = _gauss_curve(base_ipds)
                    if y_b is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=GAUSS_X, y=y_b, mode="lines",
                                line=dict(color="#1f77b4", width=2),
                                name="baseline fit", legendgroup="baseline",
                                showlegend=False, hoverinfo="skip",
                            ),
                            row=r, col=c,
                        )

                # Slowed histogram + Gaussian.
                if slow_ipds.size > 0:
                    label = f"slowed/{meth_name}"
                    show = label not in legend_seen
                    legend_seen.add(label)
                    fig.add_trace(
                        go.Histogram(
                            x=slow_ipds, name=label,
                            marker_color=color_by_id.get(mid, "#888"),
                            opacity=0.6, xbins=dict(start=0, end=256, size=4),
                            histnorm="probability density",
                            showlegend=show, legendgroup=label,
                        ),
                        row=r, col=c,
                    )
                    mu_s, sg_s, y_s = _gauss_curve(slow_ipds)
                    if y_s is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=GAUSS_X, y=y_s, mode="lines",
                                line=dict(color=color_by_id.get(mid, "#888"), width=2),
                                name=f"{label} fit", legendgroup=label,
                                showlegend=False, hoverinfo="skip",
                            ),
                            row=r, col=c,
                        )

                # μ/σ annotation top-right of each subplot.
                mu_b_s = f"μ_b={mu_b:.1f}, σ_b={sg_b:.1f}" if base_ipds.size > 0 else "μ_b=—"
                mu_s_s = (
                    f"μ_s={mu_s:.1f}, σ_s={sg_s:.1f}" if slow_ipds.size > 0 else "μ_s=—"
                )
                fig.add_annotation(
                    row=r, col=c,
                    xref="x domain", yref="y domain",
                    x=0.98, y=0.98, xanchor="right", yanchor="top",
                    text=f"{mu_b_s}<br>{mu_s_s}",
                    showarrow=False,
                    font=dict(size=9, color="#222"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="rgba(0,0,0,0.2)", borderwidth=1, borderpad=2,
                )
                fig.update_xaxes(range=[0, 200], row=r, col=c)

            fig.update_layout(
                title=(
                    f"slowed/{meth_name} — {panel_label}. "
                    "Curve = Gaussian fit (μ, σ) on top-right per kmer."
                ),
                barmode="overlay",
                height=260 * rows,
            )
            yield meth_name, panel_label, fig


def _build_ipd_pw_density_figure(data: dict):
    """Layered Plotly Surface plot — one 2D KDE per bucket on a shared (IPD, PW) grid.

    Lets the user rotate the plot and visually verify that the slowed
    surfaces sit in a different region of (IPD, PW) than baseline /
    near_meth surfaces. Uses ``scipy.stats.gaussian_kde``.
    """
    try:
        import plotly.graph_objects as go
        from scipy.stats import gaussian_kde
    except ImportError:
        log.warning("scipy or plotly not installed — skipping 3D density figure")
        return None

    from .utils.encoding import get_meth_ids
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PARENT_METH,
        COL_PARENT_OFFSET,
        COL_PW,
    )

    name_by_mid = {v: k for k, v in get_meth_ids().items()}
    rng = np.random.default_rng(0)

    # Gather (IPD, PW) per (category, meth, offset) bucket. Cap at 50k
    # points per bucket — KDE is off(n²) on bandwidth selection and slows
    # hard above that, with negligible visual gain (the surface is
    # already smooth).
    BUCKET_CAP = 50_000
    buckets: dict[str, list] = {}
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_PARENT_OFFSET:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        parent = arr[:, COL_PARENT_METH].astype(np.int8)
        offset = arr[:, COL_PARENT_OFFSET].astype(np.int8)
        ipd_pw = arr[:, [COL_IPD, COL_PW]]

        m_base = cats == CATEGORY_BASELINE
        if m_base.any():
            buckets.setdefault("baseline", []).append(ipd_pw[m_base])

        for cat_id, prefix in (
            (CATEGORY_SLOWED, "slowed_by_"),
            (CATEGORY_NEAR_METH, "near_meth_by_"),
        ):
            m_cat = cats == cat_id
            if not m_cat.any():
                continue
            for T_id in np.unique(parent[m_cat]):
                T_id_int = int(T_id)
                if T_id_int == 0:
                    continue
                m_T = m_cat & (parent == T_id_int)
                if not m_T.any():
                    continue
                T_name = name_by_mid.get(T_id_int, f"meth{T_id_int}")
                for off in np.unique(offset[m_T]):
                    O_int = int(off)
                    mask = m_T & (offset == O_int)
                    if not mask.any():
                        continue
                    buckets.setdefault(_bucket_name(prefix, T_name, O_int), []).append(ipd_pw[mask])

    if not buckets:
        return None

    # Cap each bucket and concatenate.
    pooled: dict[str, np.ndarray] = {}
    for name, chunks in buckets.items():
        x = np.concatenate(chunks).astype(np.float32)
        if len(x) > BUCKET_CAP:
            idx = rng.choice(len(x), BUCKET_CAP, replace=False)
            x = x[idx]
        pooled[name] = x

    # Shared grid in raw signal space.
    all_pts = np.concatenate(list(pooled.values()))
    ipd_hi = float(np.percentile(all_pts[:, 0], 99.5))
    pw_hi = float(np.percentile(all_pts[:, 1], 99.5))
    grid_n = 60
    ipd_grid = np.linspace(0.0, max(ipd_hi, 80.0), grid_n)
    pw_grid = np.linspace(0.0, max(pw_hi, 60.0), grid_n)
    ipd_mesh, pw_mesh = np.meshgrid(ipd_grid, pw_grid)
    grid_pts = np.vstack([ipd_mesh.ravel(), pw_mesh.ravel()])

    palette = {
        "baseline": "#1f77b4",
    }
    slowed_palette = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    near_palette = ["#2ca02c", "#17becf", "#bcbd22", "#7f7f7f"]
    s_idx = n_idx = 0

    fig = go.Figure()
    for name in sorted(pooled.keys(), key=_bucket_order_key):
        x = pooled[name]
        if len(x) < 50:
            continue
        try:
            kde = gaussian_kde(x.T, bw_method=0.2)
            z = kde(grid_pts).reshape(grid_n, grid_n)
        except np.linalg.LinAlgError:
            continue
        z_max = z.max()
        if z_max <= 0:
            continue
        z = z / z_max  # normalise so each surface peaks at 1 (visual comparability)
        if name == "baseline":
            color = palette["baseline"]
        elif name.startswith("slowed_by_"):
            color = slowed_palette[s_idx % len(slowed_palette)]
            s_idx += 1
        else:
            color = near_palette[n_idx % len(near_palette)]
            n_idx += 1
        fig.add_trace(
            go.Surface(
                x=ipd_grid,
                y=pw_grid,
                z=z,
                name=f"{name} (n={len(x):,})",
                showscale=False,
                opacity=1.0,
                # Lighting gives the opaque surfaces some shape definition
                # so adjacent peaks are still visually distinguishable.
                lighting=dict(
                    ambient=0.55,
                    diffuse=0.55,
                    specular=0.4,
                    roughness=0.7,
                ),
                colorscale=[[0.0, _darken_hex(color, 0.4)], [1.0, color]],
                showlegend=True,
            )
        )

    fig.update_layout(
        title="Joint (IPD, PW) density per bucket  "
        "(opaque surfaces — click legend entries to isolate buckets, drag to rotate)",
        scene=dict(
            xaxis_title="IPD (uint8 [0, 255])",
            yaxis_title="PW  (uint8 [0, 255])",
            zaxis_title="Normalised density",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            bgcolor="rgb(240, 240, 240)",
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
        ),
    )
    return fig


def _write_html_report(figures, stats: DictStats, output_path: str, meth_summary: str) -> None:
    """Render figures into a self-contained HTML file with a small dashboard."""
    import plotly.io as pio

    K = stats.kmer_size
    cards = [
        ("K-mer size", f"K={K}"),
        ("Total keys", f"{stats.total_entries:,}"),
        ("Total samples", f"{stats.total_samples:,}"),
        ("Coverage", _pct(stats.total_entries, stats.total_possible_kmers)),
        ("Meth types", str(len(stats.groups))),
        ("File size", f"{stats.file_size_mb:.1f} MB"),
    ]
    card_html = "\n".join(
        f'<div class="stat-card"><div class="val">{v}</div><div class="lbl">{k}</div></div>'
        for k, v in cards
    )
    nav_links = "\n".join(
        f'<a href="#section-{i}">{title}</a>' for i, (title, _) in enumerate(figures)
    )
    html_parts = [
        f"""<!DOCTYPE html>
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
"""
    ]
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
    from .utils.io import atomic_write_text

    atomic_write_text("\n".join(html_parts), output_path)
    log.info("HTML report saved: %s", output_path)

    # Standalone per-figure exports.
    export_dir = Path(output_path).parent / "figures"
    export_dir.mkdir(parents=True, exist_ok=True)
    has_kaleido = True
    try:
        import kaleido  # noqa: F401
    except ImportError:
        has_kaleido = False
    for i, (title, fig) in enumerate(figures):
        slug = title.lower().replace(" ", "_").replace("/", "_")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        fig_path = export_dir / f"{i:02d}_{slug}.html"
        pio.write_html(fig, str(fig_path), include_plotlyjs="cdn")
        if has_kaleido:
            png_path = export_dir / f"{i:02d}_{slug}.png"
            try:
                fig.write_image(str(png_path), scale=3)
            except Exception as exc:
                # PNG export is best-effort (kaleido has flaky dependencies
                # on some container images), HTML is still valid. Surface
                # the failure so the user knows they only have HTMLs,
                # rather than silently shipping a partial report.
                log.warning("PNG export failed for %s: %s — HTML only", slug, exc)
    log.info("Individual figures exported to: %s/", export_dir)


def render_html_report(
    stats: DictStats,
    output_path: str,
    data: dict,
    signature_profiles: dict | None = None,
    category_distributions: dict | None = None,
    refine_meta: dict | None = None,
) -> None:
    """Generate the verification dashboard (4 focused figures)."""
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        log.warning("plotly not installed — skipping HTML report. Install with: pip install plotly")
        return

    figures = _build_html_figures(
        data,
        category_distributions or {},
        signature_profiles or {},
        refine_meta,
    )
    meth_summary = f"baseline / slowed / near_meth — {stats.total_samples:,} samples"
    _write_html_report(figures, stats, output_path, meth_summary)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _concat_shards(shards_dir, glob: str = "*_shard*.pkl") -> tuple[dict, dict | None]:
    """Walk a shards directory and concatenate per-kmer arrays into one dict.

    Memory peak ≈ corpus size at completion, but loads shards sequentially
    so the working set never exceeds (corpus + one shard) at any moment.
    Returns ``(merged_data, refine_meta)`` where ``refine_meta`` is taken
    from the first shard's ``__meta__`` (all shards share the same refine
    parameters by construction).
    """
    shard_paths = sorted(Path(shards_dir).glob(glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shards matching {glob} in {shards_dir}")

    log.info(
        "Sharded analyze: loading + concatenating %d shards from %s", len(shard_paths), shards_dir
    )
    merged: dict = {}
    refine_meta: dict | None = None

    for i, shard_path in enumerate(shard_paths, start=1):
        log.info("  load %d/%d  %s", i, len(shard_paths), shard_path.name)
        with open(shard_path, "rb") as f:
            data = pickle.load(f)
        meta = data.pop("__meta__", None)
        if refine_meta is None and isinstance(meta, dict):
            refine_meta = meta
        for kid, arr in data.items():
            if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
                continue
            kid_int = int(kid)
            if kid_int in merged:
                merged[kid_int] = np.concatenate([merged[kid_int], arr], axis=0)
            else:
                merged[kid_int] = arr
        del data
    return merged, refine_meta


def analyze_pkl(
    pkl_path: str,
    output_dir: str | None = None,
    no_html: bool = False,
) -> None:
    """Load a .pkl OR a shards directory, compute statistics, write TXT + HTML reports.

    If ``pkl_path`` is a directory, all ``*_shard*.pkl`` files are loaded
    and concatenated per-kmer before running the pipeline. Memory peak
    is the full corpus — for very large corpora consider splitting the
    analyze run by meth type or by sample subset.

    If ``pkl_path`` is a file, it's loaded as a single master_clean.pkl.
    """
    t0 = time.time()
    in_path = Path(pkl_path).resolve()

    if in_path.is_dir():
        # Sharded mode
        if output_dir is None:
            output_dir = str(in_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base = in_path.name + "_combined"
        pkl_path = str(in_path)
        data, sharded_meta = _concat_shards(in_path)
    else:
        pkl_path = str(in_path)
        if output_dir is None:
            output_dir = str(in_path.parent)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base = in_path.stem
        log.info("Loading: %s", pkl_path)
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        sharded_meta = None

    txt_path = str(Path(output_dir) / f"{base}_report.txt")
    html_path = str(Path(output_dir) / f"{base}_report.html")

    if not data:
        log.error("Input is empty: %s", pkl_path)
        return

    _check_v4_input(data)

    log.info("Collecting statistics ...")
    stats = collect_stats(data, pkl_path)
    log.info(
        "kmer_size: %d  keys: %d  samples: %d  types: %s",
        stats.kmer_size,
        stats.total_entries,
        stats.total_samples,
        [stats.groups[m].name for m in sorted(stats.groups)],
    )

    log.info("Computing kinetic signature profiles ...")
    sig_profiles = compute_signature_profiles(data, kmer_size=stats.kmer_size)

    log.info("Computing meth-context distribution per bucket ...")
    ctx_dist = compute_meth_context_distribution(data, kmer_size=stats.kmer_size)

    log.info("Computing per-category IPD distribution ...")
    cat_dist = compute_category_distributions(data)

    refine_meta = (
        sharded_meta
        if sharded_meta is not None
        else (data.get("__meta__") if isinstance(data.get("__meta__"), dict) else None)
    )

    print()
    render_txt_report(
        stats,
        txt_path,
        signature_profiles=sig_profiles,
        context_distribution=ctx_dist,
        category_distributions=cat_dist,
        refine_meta=refine_meta,
    )

    if not no_html:
        log.info("Generating HTML report ...")
        render_html_report(
            stats,
            html_path,
            data=data,
            signature_profiles=sig_profiles,
            category_distributions=cat_dist,
            refine_meta=refine_meta,
        )

    log.info("Analysis complete in %.1fs", time.time() - t0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    import argparse

    from .utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="kinsim analyze",
        description=(
            "Analyse a KinSim training .pkl (extract + merge + refine output).\n"
            "Writes a text report and an interactive Plotly HTML dashboard.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pkl",
        help="Path to a master_clean .pkl (single file) OR a directory of "
        "refined *_shard.pkl files (sharded mode — concatenates them in-memory).",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None, help="Output directory (default: same dir as pkl)"
    )
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report (text only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    analyze_pkl(
        pkl_path=args.pkl,
        output_dir=args.output_dir,
        no_html=args.no_html,
    )


if __name__ == "__main__":
    main()
