"""Diagnose per-offset behaviour of CATEGORY_SLOWED samples.

For each (parent_meth, parent_offset) pair found in the data, prints:
  n samples, IPD mean/p50/p95, PW mean/p50/p95, and the IPD profile across
  positions 0..+8 (so the "should peak at +0" / "should peak at +5" check
  is visible at a glance).

Why: when ``slowed_by_m6A`` looks flat at baseline level, the question is
whether *both* offsets (0 and 5) are flat (extraction / flagging bug) or
only +5 is flat because the configured signature doesn't apply to N-spaced
Type I motifs.

Works on:
  - a single .pkl   (master, master_clean, or single shard)
  - a directory     (loops every ``*_shard*.pkl`` inside, accumulates totals)

CLI:
  python scripts/diagnose_offset_split.py <pkl_or_dir>
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(message)s")
log = logging.getLogger("diagnose_offset_split")


def _load_iter(path: Path):
    """Yield (shard_label, dict) tuples — one per shard or just one for a single .pkl."""
    if path.is_dir():
        shards = sorted(path.glob("*_shard*.pkl"))
        if not shards:
            shards = sorted(path.glob("*.pkl"))
        if not shards:
            raise FileNotFoundError(f"No .pkl files in {path}")
        for sp in shards:
            with open(sp, "rb") as f:
                yield sp.name, pickle.load(f)
    else:
        with open(path, "rb") as f:
            yield path.name, pickle.load(f)


def diagnose(path: Path) -> None:
    from kinsim.utils.encoding import get_meth_ids
    from kinsim.utils.sample_layout import (
        CATEGORY_NAMES,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PARENT_METH,
        COL_PARENT_OFFSET,
        COL_PW,
        METH_CTX_LEN,
        PROFILE_LEN,
    )

    name_by_mid = {v: k for k, v in get_meth_ids().items()}
    profile_start = 3 + METH_CTX_LEN
    pw_start = profile_start + PROFILE_LEN
    needed = pw_start + PROFILE_LEN

    # Per-(category, parent_meth, parent_offset) accumulators.
    # We keep raw arrays for IPD/PW (concatenated chunks per bucket) and
    # streaming sums for the per-offset profile.
    bucket_ipd: dict[tuple, list] = {}
    bucket_pw: dict[tuple, list] = {}
    bucket_prof_ipd: dict[tuple, np.ndarray] = {}
    bucket_prof_n: dict[tuple, int] = {}

    n_shards = 0
    for shard_name, data in _load_iter(path):
        n_shards += 1
        log.info("Loaded shard %s with %d kmer keys", shard_name, len(data) - ("__meta__" in data))
        for kid, arr in data.items():
            if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
                continue
            if arr.shape[1] <= COL_PARENT_OFFSET:
                log.warning(
                    "Shard %s appears to be older than v4-with-PARENT_OFFSET (ncols=%d)",
                    shard_name,
                    arr.shape[1],
                )
                return
            cats = arr[:, COL_CATEGORY].astype(np.int8)
            for cat_id in (CATEGORY_SLOWED, CATEGORY_NEAR_METH):
                m_cat = cats == cat_id
                if not m_cat.any():
                    continue
                parent = arr[:, COL_PARENT_METH].astype(np.int8)
                offset = arr[:, COL_PARENT_OFFSET].astype(np.int8)
                # Pre-extract IPD/PW/profile to avoid copies on each unique-pair mask.
                ipd = arr[:, COL_IPD]
                pw = arr[:, COL_PW]
                prof = arr[:, profile_start:pw_start]

                t_vals = np.unique(parent[m_cat])
                for t_id in t_vals:
                    if t_id == 0:
                        continue
                    off_vals = np.unique(offset[m_cat & (parent == t_id)])
                    for off in off_vals:
                        mask = m_cat & (parent == t_id) & (offset == off)
                        n = int(mask.sum())
                        if n == 0:
                            continue
                        key = (int(cat_id), int(t_id), int(off))
                        bucket_ipd.setdefault(key, []).append(ipd[mask])
                        bucket_pw.setdefault(key, []).append(pw[mask])
                        if key not in bucket_prof_ipd:
                            bucket_prof_ipd[key] = np.zeros(PROFILE_LEN, dtype=np.float64)
                            bucket_prof_n[key] = 0
                        bucket_prof_ipd[key] += prof[mask].sum(axis=0)
                        bucket_prof_n[key] += n

    if not bucket_ipd:
        log.error("No SLOWED or NEAR samples found in %s", path)
        return

    # Sort: SLOWED first then NEAR; within each, by meth_name then offset.
    keys = sorted(bucket_ipd.keys(), key=lambda k: (k[0], name_by_mid.get(k[1], "z"), k[2]))

    print()
    print("=" * 100)
    print(f"Per-offset diagnostic — {path}")
    print(f"shards loaded: {n_shards}, columns required: {needed}")
    print("=" * 100)
    hdr = (
        f"  {'category':<10s}{'parent':<6s}{'off':>5s}{'n':>12s}"
        f"{'IPD_mean':>10s}{'IPD_p50':>9s}{'IPD_p95':>9s}"
        f"{'PW_mean':>9s}{'PW_p50':>9s}{'PW_p95':>9s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in keys:
        cat_id, t_id, off = key
        cat_name = CATEGORY_NAMES.get(cat_id, str(cat_id))
        t_name = name_by_mid.get(t_id, f"meth{t_id}")
        ipd_pool = np.concatenate(bucket_ipd[key]).astype(np.float32)
        pw_pool = np.concatenate(bucket_pw[key]).astype(np.float32)
        n = len(ipd_pool)
        print(
            f"  {cat_name:<10s}{t_name:<6s}{off:>+5d}{n:>12,d}"
            f"{ipd_pool.mean():>10.2f}{np.percentile(ipd_pool, 50):>9.1f}"
            f"{np.percentile(ipd_pool, 95):>9.1f}"
            f"{pw_pool.mean():>9.2f}{np.percentile(pw_pool, 50):>9.1f}"
            f"{np.percentile(pw_pool, 95):>9.1f}"
        )

    # Per-bucket IPD profile across positions 0..+8.
    print()
    print("Per-bucket IPD profile (mean across offsets 0..+8 from prediction position)")
    print("If signature offset matches the column that peaks → flag is correct for that offset.")
    print()
    pos_hdr = "  " + "  ".join(f"+{i:1d}".rjust(5) for i in range(PROFILE_LEN))
    for key in keys:
        cat_id, t_id, off = key
        cat_name = CATEGORY_NAMES.get(cat_id, str(cat_id))
        t_name = name_by_mid.get(t_id, f"meth{t_id}")
        prof = bucket_prof_ipd[key] / max(bucket_prof_n[key], 1)
        if 0 <= off < PROFILE_LEN:
            marker_arr = ["   "] * PROFILE_LEN
            marker_arr[off] = "***"
            mark_line = "  " + "  ".join(f"{m:>5s}" for m in marker_arr)
        else:
            mark_line = "  (offset out of profile window)"
        title = f"{cat_name}/{t_name}/+{off}  (n={bucket_prof_n[key]:,})"
        print(f"  {title}")
        print(pos_hdr)
        print("  " + "  ".join(f"{v:5.1f}" for v in prof))
        print(f"  ^ flagged signature offset = +{off}")
        print(mark_line)
        print()


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/diagnose_offset_split.py <pkl_or_shards_dir>")
        sys.exit(2)
    path = Path(sys.argv[1]).resolve()
    if not path.exists():
        log.error("Not found: %s", path)
        sys.exit(1)
    diagnose(path)


if __name__ == "__main__":
    main()
