"""Naive-Gaussian baseline generation.

Two output modes, controlled via the ``--format`` flag:

  --format bam        Build a kinsim-generate-compatible lookup NPZ from
                      ``baseline.json``. Then feed it to ``kinsim generate
                      --use-lookup`` to produce a BAM with naive kinetics
                      (every kmer at site (T, k) gets the SAME N(μ_T_k,
                      σ_T_k²) — no kmer context).

  --format pkl        Take an existing shard.pkl (refined or raw) and
                      replace its IPD/PW columns with samples from the
                      naive Gaussian for the corresponding (parent_meth,
                      parent_offset). Lets you compare distributions
                      via kinsim analyze without going through the full
                      BAM pipeline.

Both modes share the per-(T, k) Gaussian computed in log1p space from
the histograms in baseline.json (output of `kinsim_baseline compute`).

Example::

    # 1. Build the naive lookup NPZ
    python -m kinsim_baseline generate \\
        --format bam --baseline baseline.json --output naive_lookup.npz

    # 2. Run kinsim generate with it (any path works, checkpoint is
    #    ignored when --use-lookup is set):
    kinsim generate input.bam ref.fa /dev/null/dummy_ckpt motifs.csv \\
        output_naive.bam --use-lookup naive_lookup.npz

    # OR — shard mode
    python -m kinsim_baseline generate \\
        --format pkl --baseline baseline.json \\
        --input shards/strepto_bc2034_shard.pkl \\
        --output shards/strepto_bc2034_shard_naive.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian extraction from baseline.json histograms
# ---------------------------------------------------------------------------


def _hist_log_moments(h: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    """Mean and std of log1p(bin) weighted by histogram counts.

    Returns ``(None, None)`` for empty histograms.
    """
    n = int(h.sum())
    if n == 0:
        return None, None
    bins = np.arange(256, dtype=np.float64)
    log_bins = np.log1p(bins)
    mu = float((log_bins * h).sum() / n)
    var = float(((log_bins - mu) ** 2 * h).sum() / n)
    return mu, max(float(np.sqrt(var)), 1e-6)


def load_baseline_gaussians(baseline_json: str | Path) -> dict:
    """Parse baseline.json and return per-(T, k) log-space (μ, σ) for IPD and PW.

    Returns ``{(meth_type_str, offset_int): {mu_ipd, sigma_ipd, mu_pw, sigma_pw, n}}``.
    Values in log1p space — matches what kinsim generate consumes via
    ``--use-lookup``.
    """
    with open(baseline_json) as f:
        bundle = json.load(f)
    # The compute.py output stores 256-bin counts under "ipd" / "pw" with
    # composite keys "<meth_type>__<offset>". Robust parse — handle either
    # nested {ipd: {key: hist}} or flat {key: {ipd, pw}}.
    ipd_section = bundle.get("ipd", bundle.get("IPD", {}))
    pw_section = bundle.get("pw", bundle.get("PW", {}))

    gaussians: dict[tuple[str, int], dict] = {}
    for key, hist in ipd_section.items():
        # Key format: "<meth_type>__<offset>"
        if "__" not in key:
            continue
        T, off_str = key.rsplit("__", 1)
        try:
            off = int(off_str)
        except ValueError:
            continue
        h_ipd = np.asarray(hist, dtype=np.int64)
        h_pw = np.asarray(pw_section.get(key, []), dtype=np.int64)
        mu_i, sg_i = _hist_log_moments(h_ipd)
        mu_p, sg_p = _hist_log_moments(h_pw) if h_pw.size > 0 else (None, None)
        if mu_i is None:
            continue
        gaussians[(T, off)] = {
            "mu_ipd": mu_i,
            "sigma_ipd": sg_i,
            "mu_pw": mu_p if mu_p is not None else mu_i,  # fallback if PW missing
            "sigma_pw": sg_p if sg_p is not None else sg_i,
            "n": int(h_ipd.sum()),
        }
    log.info("Loaded %d (T, k) buckets from %s", len(gaussians), baseline_json)
    return gaussians


# ---------------------------------------------------------------------------
# BAM mode — write a kinsim-generate-compatible lookup NPZ
# ---------------------------------------------------------------------------


def make_naive_lookup(
    gaussians: dict,
    output_npz: str | Path,
    baseline_mu_log: float | None = None,
    baseline_sigma_log: float | None = None,
) -> None:
    """Write a flat NPZ where every kmer has the same (μ, σ) per scenario.

    Layout matches what ``kinsim generate --use-lookup`` consumes (see
    ``kinsim/generate.py::_load_lookup_table``). The ``none`` scenario
    (unmethylated baseline) uses ``baseline_mu/sigma_log`` if provided,
    otherwise defaults to log1p(15) / 0.5 — typical resting-state IPD.
    """
    from kinsim.utils.encoding import K, get_meth_ids

    if baseline_mu_log is None:
        baseline_mu_log = float(np.log1p(15.0))  # ~2.77
    if baseline_sigma_log is None:
        baseline_sigma_log = 0.5

    n_kmers = 4**K
    meth_id_map = get_meth_ids()
    name_to_id = {name: int(mid) for name, mid in meth_id_map.items()}

    bundle: dict[str, np.ndarray] = {"kmer_id": np.arange(n_kmers, dtype=np.int64)}
    labels = ["none"]
    m_ids = [0]
    offsets = [0]

    # "none" scenario — flat baseline for every kmer
    bundle["none__mu_ipd_log"] = np.full(n_kmers, baseline_mu_log, dtype=np.float32)
    bundle["none__mu_pw_log"] = np.full(n_kmers, baseline_mu_log, dtype=np.float32)
    bundle["none__sigma_ipd_log"] = np.full(n_kmers, baseline_sigma_log, dtype=np.float32)
    bundle["none__sigma_pw_log"] = np.full(n_kmers, baseline_sigma_log, dtype=np.float32)

    # One scenario per (T, k) from the baseline histograms.
    for (T, k), gauss in sorted(gaussians.items()):
        if T not in name_to_id:
            log.warning("Skipping (%s, %+d) — not declared in kinsim_config.yaml", T, k)
            continue
        sk = f"{T}_at_p{k}" if k >= 0 else f"{T}_at_m{-k}"
        bundle[f"{sk}__mu_ipd_log"] = np.full(n_kmers, gauss["mu_ipd"], dtype=np.float32)
        bundle[f"{sk}__mu_pw_log"] = np.full(n_kmers, gauss["mu_pw"], dtype=np.float32)
        bundle[f"{sk}__sigma_ipd_log"] = np.full(n_kmers, gauss["sigma_ipd"], dtype=np.float32)
        bundle[f"{sk}__sigma_pw_log"] = np.full(n_kmers, gauss["sigma_pw"], dtype=np.float32)
        labels.append(f"{T}@{'+' if k >= 0 else ''}{k}")
        m_ids.append(name_to_id[T])
        offsets.append(k)

    bundle["scenarios_label"] = np.asarray(labels)
    bundle["scenarios_meth_id"] = np.asarray(m_ids, dtype=np.int64)
    bundle["scenarios_offset"] = np.asarray(offsets, dtype=np.int64)

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **bundle)
    log.info(
        "Wrote %s (%d scenarios × %d kmers, %.1f MB on disk)",
        output_npz,
        len(labels),
        n_kmers,
        output_npz.stat().st_size / 1e6,
    )
    log.info("Scenarios: %s", labels)
    log.info(
        "\n  Next: kinsim generate <input.bam> <ref.fa> <DUMMY_CKPT> <motifs.csv> "
        "<output.bam> --use-lookup %s\n",
        output_npz,
    )


# ---------------------------------------------------------------------------
# PKL mode — naive shard generation
# ---------------------------------------------------------------------------


def make_naive_shard(
    input_shard: str | Path,
    gaussians: dict,
    output_shard: str | Path,
    seed: int = 42,
    baseline_mu_log: float | None = None,
    baseline_sigma_log: float | None = None,
) -> None:
    """Replace IPD/PW columns of an existing shard with naive Gaussian samples.

    Per row:
      - If ``CATEGORY == BASELINE``: sample from N(baseline_mu, baseline_sigma²)
      - If ``CATEGORY == SLOWED``: sample from N(μ_T_k, σ_T_k²) where (T, k)
        come from ``COL_PARENT_METH`` / ``COL_PARENT_OFFSET``.
      - If ``CATEGORY == NEAR_METH``: sample from baseline (same as BASELINE
        — near_meth is the negative control, expected to look like baseline).

    All other columns (kmer_id key, meth_context, rev_meth, category, parent_*)
    are passed through. Samples are clipped to [0, 255] and stored as float32
    to match the existing shard convention.

    Useful for `kinsim analyze` comparison: same row structure, same kmer/meth
    context, but kinetics from a context-free Gaussian — visualises whether
    the ML model's kmer-conditional predictions improve over a global average.
    """
    from kinsim.utils.encoding import get_meth_ids
    from kinsim.utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PARENT_METH,
        COL_PARENT_OFFSET,
        COL_PW,
    )

    if baseline_mu_log is None:
        baseline_mu_log = float(np.log1p(15.0))
    if baseline_sigma_log is None:
        baseline_sigma_log = 0.5

    name_by_id = {v: k for k, v in get_meth_ids().items()}
    # Pre-build log-space Gaussians keyed by (meth_id, offset).
    g_by_mid_off: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    meth_id_map = get_meth_ids()
    for (T, k), gauss in gaussians.items():
        if T not in meth_id_map:
            continue
        mid = int(meth_id_map[T])
        g_by_mid_off[(mid, int(k))] = (
            gauss["mu_ipd"],
            gauss["sigma_ipd"],
            gauss["mu_pw"],
            gauss["sigma_pw"],
        )

    rng = np.random.default_rng(seed)

    log.info("Reading %s", input_shard)
    with open(input_shard, "rb") as f:
        data = pickle.load(f)

    n_total = 0
    n_baseline = n_slowed = n_near = 0
    n_unmapped_bucket = 0  # rows where (parent_meth, parent_offset) wasn't in gaussians

    out: dict = {}
    for k, v in data.items():
        # Preserve __meta__ + any string-keyed entries.
        if not isinstance(k, (int, np.integer)) or not isinstance(v, np.ndarray):
            out[k] = v
            continue
        arr = v.copy()  # don't mutate the input
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        parent_meth = arr[:, COL_PARENT_METH].astype(np.int8)
        parent_off = arr[:, COL_PARENT_OFFSET].astype(np.int8)

        ipd_new = np.empty(len(arr), dtype=np.float32)
        pw_new = np.empty(len(arr), dtype=np.float32)

        # Baseline + near_meth → baseline Gaussian.
        m_base = (cats == CATEGORY_BASELINE) | (cats == CATEGORY_NEAR_METH)
        if m_base.any():
            n = int(m_base.sum())
            mu_log = baseline_mu_log
            sg_log = baseline_sigma_log
            ipd_new[m_base] = np.expm1(rng.normal(mu_log, sg_log, size=n))
            pw_new[m_base] = np.expm1(rng.normal(mu_log, sg_log, size=n))

        # Slowed rows — group by (parent_meth, parent_offset).
        m_slow = cats == CATEGORY_SLOWED
        if m_slow.any():
            slow_idx = np.where(m_slow)[0]
            for i in slow_idx:
                mid = int(parent_meth[i])
                off = int(parent_off[i])
                key = (mid, off)
                if key in g_by_mid_off:
                    mu_i, sg_i, mu_p, sg_p = g_by_mid_off[key]
                else:
                    mu_i = mu_p = baseline_mu_log
                    sg_i = sg_p = baseline_sigma_log
                    n_unmapped_bucket += 1
                ipd_new[i] = np.expm1(rng.normal(mu_i, sg_i))
                pw_new[i] = np.expm1(rng.normal(mu_p, sg_p))

        # Clip to uint8 range. Stored as float32 to match existing layout.
        arr[:, COL_IPD] = np.clip(ipd_new, 0.0, 255.0).astype(np.float32)
        arr[:, COL_PW] = np.clip(pw_new, 0.0, 255.0).astype(np.float32)
        out[k] = arr

        n_total += len(arr)
        n_baseline += int((cats == CATEGORY_BASELINE).sum())
        n_slowed += int(m_slow.sum())
        n_near += int((cats == CATEGORY_NEAR_METH).sum())

    # Bump the meta to record the naive baseline source.
    meta = out.get("__meta__", {})
    if isinstance(meta, dict):
        meta = dict(meta)  # mutable copy
        meta["naive_baseline_source"] = str(input_shard)
        meta["naive_baseline_buckets"] = sorted(
            f"{name_by_id.get(mid, '?')}@{off:+d}" for (mid, off) in g_by_mid_off
        )
        meta["naive_baseline_unmapped_slowed_rows"] = n_unmapped_bucket
        out["__meta__"] = meta

    output_shard = Path(output_shard)
    output_shard.parent.mkdir(parents=True, exist_ok=True)
    with open(output_shard, "wb") as f:
        pickle.dump(out, f)
    log.info(
        "Wrote %s — %d total rows (%d baseline, %d slowed, %d near_meth)",
        output_shard,
        n_total,
        n_baseline,
        n_slowed,
        n_near,
    )
    if n_unmapped_bucket > 0:
        log.warning(
            "%d slowed rows had a (parent_meth, parent_offset) not present in baseline.json "
            "— fell back to baseline Gaussian. Re-run kinsim_baseline compute with all "
            "relevant motifs to cover them.",
            n_unmapped_bucket,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python -m kinsim_baseline generate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--format",
        required=True,
        choices=("bam", "pkl"),
        help="bam: emit a kinsim-generate lookup NPZ. pkl: rewrite a shard's IPD/PW.",
    )
    p.add_argument("--baseline", required=True, help="Path to baseline.json (kinsim_baseline compute output).")
    p.add_argument("--output", required=True, help="Output path (.npz for bam, .pkl for pkl).")
    p.add_argument("--input", default=None, help="Input shard.pkl (required with --format pkl).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for pkl mode (default: 42).")
    p.add_argument(
        "--baseline-mu-log",
        type=float,
        default=None,
        help="log1p μ for the unmethylated baseline (default: log1p(15) ≈ 2.77).",
    )
    p.add_argument(
        "--baseline-sigma-log",
        type=float,
        default=None,
        help="log1p σ for the unmethylated baseline (default: 0.5).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    gaussians = load_baseline_gaussians(args.baseline)
    if not gaussians:
        log.error("No (T, k) buckets parsed from %s — aborting.", args.baseline)
        sys.exit(1)

    if args.format == "bam":
        make_naive_lookup(
            gaussians,
            args.output,
            baseline_mu_log=args.baseline_mu_log,
            baseline_sigma_log=args.baseline_sigma_log,
        )
    elif args.format == "pkl":
        if not args.input:
            log.error("--input is required with --format pkl")
            sys.exit(1)
        make_naive_shard(
            args.input,
            gaussians,
            args.output,
            seed=args.seed,
            baseline_mu_log=args.baseline_mu_log,
            baseline_sigma_log=args.baseline_sigma_log,
        )
    else:
        log.error("Unknown format: %s", args.format)
        sys.exit(1)


if __name__ == "__main__":
    main()
