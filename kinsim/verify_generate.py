"""``kinsim verify-generate`` — compare per-(kmer, meth_id) kinetics between two shards.

Workflow
--------
1. Run ``kinsim extract`` on the REFERENCE aligned BAM → ref_shard.pkl
2. Run ``kinsim generate`` to produce a synthetic BAM, align it
   (bystrandify + pbmm2), then run ``kinsim extract`` on it →
   gen_shard.pkl
3. Run ``kinsim verify-generate ref_shard.pkl gen_shard.pkl report.tsv``

The output TSV has one row per (kmer, meth_id) key seen in either
shard, with both shards' n / mu_ipd / sd_ipd / mu_pw / sd_pw and the
deltas. Summary lines at the bottom give Pearson r and MAE on the
paired-key arrays — the headline sanity numbers for "did the model
reproduce the per-(kmer, meth) distribution we trained on".

Usage::

    kinsim verify-generate ref_shard.pkl gen_shard.pkl report.tsv [--min-samples N]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from .utils.config import setup_logging
from .utils.encoding import decode_kmer, get_meth_ids
from .utils.sample_layout import (
    COL_IPD,
    COL_PARENT_METH,
    COL_PW,
    get_sample_layout,
)

log = logging.getLogger(__name__)


def _summarize_shard(path: str | Path) -> dict[tuple[int, int], tuple]:
    """Load a shard and return ``{(kmer_id, parent_meth): (n, μ_ipd, σ_ipd, μ_pw, σ_pw)}``.

    Reads the shard's :class:`~kinsim.utils.config.ExtractionParams` from its
    ``__meta__`` block (when present) so the column indices adapt to whatever
    geometry the shard was extracted under. Legacy K=11 shards fall back to
    the module-level ``COL_*`` constants.
    """
    from .data.dataset import read_shard_extraction_params  # noqa: PLC0415

    with open(path, "rb") as f:
        data = pickle.load(f)
    # Resolve column layout from shard meta (new shards) or fall back to legacy.
    params = read_shard_extraction_params(data)
    layout = get_sample_layout(params) if params is not None else None
    col_ipd = layout.col_ipd if layout else COL_IPD
    col_pw = layout.col_pw if layout else COL_PW
    col_parent_meth = layout.col_parent_meth if layout else COL_PARENT_METH

    data.pop("__meta__", None)
    out: dict[tuple[int, int], tuple] = {}
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.ndim != 2 or arr.shape[1] <= col_parent_meth:
            continue
        parent = arr[:, col_parent_meth].astype(np.int8)
        for m_id in np.unique(parent):
            mask = parent == m_id
            ipd = arr[mask, col_ipd].astype(np.float32)
            pw = arr[mask, col_pw].astype(np.float32)
            n = len(ipd)
            if n == 0:
                continue
            out[(int(kid), int(m_id))] = (
                n,
                float(ipd.mean()),
                float(ipd.std(ddof=1)) if n > 1 else 0.0,
                float(pw.mean()),
                float(pw.std(ddof=1)) if n > 1 else 0.0,
            )
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def verify(
    ref_shard: str,
    gen_shard: str,
    output_tsv: Path,
    *,
    min_samples: int = 5,
) -> dict:
    log.info("Loading reference shard: %s", ref_shard)
    ref_stats = _summarize_shard(ref_shard)
    log.info("Loading generated shard: %s", gen_shard)
    gen_stats = _summarize_shard(gen_shard)
    all_keys = sorted(set(ref_stats) | set(gen_stats))
    log.info("Keys: ref=%d gen=%d union=%d", len(ref_stats), len(gen_stats), len(all_keys))

    name_by_mid = {v: k for k, v in get_meth_ids().items()}
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing TSV: %s", output_tsv)

    paired_ref_ipd, paired_gen_ipd = [], []
    paired_ref_pw, paired_gen_pw = [], []
    n_written = 0
    with open(output_tsv, "w") as f:
        f.write(
            "kmer_id\tkmer\tmeth\tn_ref\tmu_ipd_ref\tsd_ipd_ref\tmu_pw_ref\tsd_pw_ref\t"
            "n_gen\tmu_ipd_gen\tsd_ipd_gen\tmu_pw_gen\tsd_pw_gen\td_mu_ipd\td_mu_pw\n"
        )
        EMPTY = (0, 0.0, 0.0, 0.0, 0.0)
        for key in all_keys:
            kmer_id, meth_id = key
            r = ref_stats.get(key, EMPTY)
            g = gen_stats.get(key, EMPTY)
            if r[0] < min_samples and g[0] < min_samples:
                continue
            kmer_str = decode_kmer(kmer_id) if kmer_id >= 0 else "?"
            meth_str = name_by_mid.get(meth_id, f"mod{meth_id}")
            f.write(
                f"{kmer_id}\t{kmer_str}\t{meth_str}\t"
                f"{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.4f}\t{r[4]:.4f}\t"
                f"{g[0]}\t{g[1]:.4f}\t{g[2]:.4f}\t{g[3]:.4f}\t{g[4]:.4f}\t"
                f"{g[1] - r[1]:+.4f}\t{g[3] - r[3]:+.4f}\n"
            )
            n_written += 1
            if r[0] >= min_samples and g[0] >= min_samples:
                paired_ref_ipd.append(r[1]); paired_gen_ipd.append(g[1])
                paired_ref_pw.append(r[3]); paired_gen_pw.append(g[3])

    rri, rgi = np.asarray(paired_ref_ipd), np.asarray(paired_gen_ipd)
    rrp, rgp = np.asarray(paired_ref_pw), np.asarray(paired_gen_pw)
    r_ipd = _pearson(rri, rgi)
    r_pw = _pearson(rrp, rgp)
    mae_ipd = float(np.mean(np.abs(rgi - rri))) if len(rri) else float("nan")
    mae_pw = float(np.mean(np.abs(rgp - rrp))) if len(rrp) else float("nan")

    log.info("=" * 56)
    log.info("  VERIFY-GENERATE SUMMARY")
    log.info("=" * 56)
    log.info("  Rows written:          %d", n_written)
    log.info("  Paired keys (n >= %d): %d", min_samples, len(rri))
    log.info("  Pearson r (mu_ipd):    %.4f", r_ipd)
    log.info("  Pearson r (mu_pw):     %.4f", r_pw)
    log.info("  MAE (mu_ipd):          %.4f", mae_ipd)
    log.info("  MAE (mu_pw):           %.4f", mae_pw)
    log.info("=" * 56)
    return {
        "n_rows": n_written,
        "n_paired": len(rri),
        "pearson_mu_ipd": r_ipd,
        "pearson_mu_pw": r_pw,
        "mae_mu_ipd": mae_ipd,
        "mae_mu_pw": mae_pw,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="kinsim verify-generate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("ref_shard", help="Reference shard.pkl from `kinsim extract` on the real BAM")
    ap.add_argument("gen_shard", help="Generated shard.pkl from `kinsim extract` on the simulated BAM")
    ap.add_argument("output_tsv", help="Output TSV with per-(kmer, meth) comparison")
    ap.add_argument("--min-samples", type=int, default=5,
                    help="Drop rows where both shards have fewer than this many samples")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(verbose=args.verbose)

    for label, p in (("ref_shard", args.ref_shard), ("gen_shard", args.gen_shard)):
        if not Path(p).exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    verify(args.ref_shard, args.gen_shard, Path(args.output_tsv),
           min_samples=args.min_samples)


if __name__ == "__main__":
    main()
