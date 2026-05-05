"""End-to-end unit tests for the storage / refine / analyze pipeline.

Three categories: baseline / slowed / near_meth. The methylation
centers themselves land in SLOWED or NEAR_METH depending on whether 0
is a signature offset for that type. Parent meth attribution is
written at extract time into ``COL_PARENT_METH`` (col 36).

Tests cover:
  1. Storage constants + get_categories on the 38-col layout.
  2. Refine: slowed_split filters CATEGORY_SLOWED below the
     per-kmer-mean baseline percentile; CATEGORY_BASELINE and
     CATEGORY_NEAR_METH pass through untouched.
  3. Analyze: compute_signature_profiles and
     compute_meth_context_distribution produce
     "baseline" / "slowed_by_<T>" / "near_meth_by_<T>" buckets
     using the COL_PARENT_METH column directly (no mc[] inference).
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np

from kinsim.refine import refine_pkl, slowed_split
from kinsim.utils.encoding import KMER_PRED_IDX, get_meth_ids
from kinsim.utils.sample_layout import (
    CATEGORY_BASELINE,
    CATEGORY_NAMES,
    CATEGORY_NEAR_METH,
    CATEGORY_SLOWED,
    COL_CATEGORY,
    COL_IPD,
    COL_PARENT_METH,
    COL_PARENT_OFFSET,
    SAMPLE_NCOLS,
    get_categories,
)

# ---------------------------------------------------------------------------
# Storage spec
# ---------------------------------------------------------------------------


def test_category_constants_are_three():
    assert CATEGORY_BASELINE == 0
    assert CATEGORY_SLOWED == 1
    assert CATEGORY_NEAR_METH == 2
    assert set(CATEGORY_NAMES.keys()) == {0, 1, 2}
    assert CATEGORY_NAMES[0] == "baseline"
    assert CATEGORY_NAMES[1] == "slowed"
    assert CATEGORY_NAMES[2] == "near_meth"


def test_get_categories_reads_col35():
    arr = np.zeros((3, SAMPLE_NCOLS), dtype=np.float32)
    arr[0, COL_CATEGORY] = CATEGORY_BASELINE
    arr[1, COL_CATEGORY] = CATEGORY_SLOWED
    arr[2, COL_CATEGORY] = CATEGORY_NEAR_METH
    cats = get_categories(arr)
    assert cats.tolist() == [CATEGORY_BASELINE, CATEGORY_SLOWED, CATEGORY_NEAR_METH]


def test_layout_column_contract():
    """Layout columns 36/37 are PARENT_METH and PARENT_OFFSET respectively."""
    assert SAMPLE_NCOLS == 38
    assert COL_CATEGORY == 35
    assert COL_PARENT_METH == 36
    assert COL_PARENT_OFFSET == 37


def test_analyze_uses_parent_meth_column_not_meth_context():
    """compute_signature_profiles attributes via COL_PARENT_METH, not mc[].

    Two slowed rows: one with PARENT_METH=m6A but NO m6A in mc, another with
    PARENT_METH=m5C and an m6A in mc (red herring). The new vectorised analyze
    must trust col 36, not the mc inference, so the buckets reflect the
    explicit parent attribution.
    """
    from kinsim.analyze import compute_signature_profiles

    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]
    m5c = meth_ids["m5C"]

    # 50 baselines so kid=0 has stats; 1 slowed-by-m6A (no mc trace);
    # 1 slowed-by-m5C (with an m6A red herring in mc).
    rows = []
    for _ in range(50):
        r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
        r[COL_IPD] = 30.0
        r[14:23] = 30.0
        rows.append(r)
    r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
    r[COL_IPD] = 200.0
    r[14:23] = 200.0
    r[COL_CATEGORY] = CATEGORY_SLOWED
    r[COL_PARENT_METH] = m6a
    # mc is all zeros for this row — no m6A trace
    rows.append(r)
    r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
    r[COL_IPD] = 150.0
    r[14:23] = 150.0
    r[COL_CATEGORY] = CATEGORY_SLOWED
    r[COL_PARENT_METH] = m5c
    r[3 + KMER_PRED_IDX] = m6a  # red herring — m6A in mc but parent says m5C
    rows.append(r)
    data = {0: np.stack(rows)}

    profiles = compute_signature_profiles(data)
    assert "slowed_by_m6A" in profiles
    assert "slowed_by_m5C" in profiles
    assert profiles["slowed_by_m6A"]["n_samples"] == 1
    assert profiles["slowed_by_m5C"]["n_samples"] == 1


def test_refine_fails_fast_on_obsolete_layout():
    """refine_pkl exits with code 1 when input lacks parent-meth columns."""
    import sys

    OLD_NCOLS = 36  # pre-PARENT_METH layout
    data = {0: np.zeros((10, OLD_NCOLS), dtype=np.float32)}
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "old.pkl"
        out = Path(td) / "out.pkl"
        with open(inp, "wb") as f:
            pickle.dump(data, f)
        try:
            refine_pkl(inp, out)
        except SystemExit as e:
            assert e.code == 1
        else:
            raise AssertionError("expected SystemExit on obsolete layout")
        assert not out.exists()
        # Silence the unused-import linter complaint on `sys`
        del sys


# ---------------------------------------------------------------------------
# Refine v4
# ---------------------------------------------------------------------------


def _build_v4_master(n_kmers: int = 10) -> dict:
    """Synthetic master with the 3-cat scheme:
    50 baseline (IPD=50), 30 slowed-high (IPD=180), 20 slowed-low (IPD=20),
    25 near_meth (IPD=55) per kmer.

    All slowed/near rows are tagged as parent m6A (id=1) at offset 0 so the
    parent-meth columns are populated consistently with what extract emits.
    """
    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]
    data = {}
    for kid in range(n_kmers):
        rows = []
        for _ in range(50):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 50.0
            r[COL_CATEGORY] = CATEGORY_BASELINE
            rows.append(r)
        for _ in range(30):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 180.0
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = m6a
            r[COL_PARENT_OFFSET] = 0
            rows.append(r)
        for _ in range(20):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 20.0
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = m6a
            r[COL_PARENT_OFFSET] = 0
            rows.append(r)
        for _ in range(25):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 55.0
            r[COL_CATEGORY] = CATEGORY_NEAR_METH
            r[COL_PARENT_METH] = m6a
            r[COL_PARENT_OFFSET] = 1
            rows.append(r)
        data[kid] = np.stack(rows)
    return data


def test_slowed_split_only_filters_slowed():
    """Only SLOWED below the per-kmer-mean baseline percentile is dropped.
    BASELINE and NEAR_METH pass through untouched."""
    data = _build_v4_master(5)
    _new_data, stats = slowed_split(data, secondary_pct=95.0)

    # All kmers have baseline mean = 50 (50 samples, all IPD=50) so
    # p95 of per-kmer means = 50.
    assert stats["threshold"] == 50.0
    assert stats["n_baseline_in"] == 5 * 50
    assert stats["n_baseline_out"] == 5 * 50
    assert stats["n_near_in"] == 5 * 25
    assert stats["n_near_out"] == 5 * 25
    assert stats["n_slowed_in"] == 5 * 50
    assert stats["n_slowed_kept"] == 5 * 30  # IPD=180 above threshold
    assert stats["n_slowed_dropped"] == 5 * 20  # IPD=20 below


def test_refine_pkl_writes_output():
    """refine_pkl loads, runs slowed_split, writes a clean pkl."""
    data = _build_v4_master(3)
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.pkl"
        out = Path(td) / "out.pkl"
        with open(inp, "wb") as f:
            pickle.dump(data, f)
        refine_pkl(inp, out)
        with open(out, "rb") as f:
            refined = pickle.load(f)
        meta = refined.pop("__meta__")
        assert meta["method"] == "p95_per_kmer_baseline_mean"
        # Per kmer: 50 baseline + 30 slowed kept + 25 near = 105
        for kid in range(3):
            assert kid in refined
            arr = refined[kid]
            assert arr.shape[1] == SAMPLE_NCOLS
            assert len(arr) == 105


# ---------------------------------------------------------------------------
# Analyze v4 dispatch
# ---------------------------------------------------------------------------


def _build_v4_with_signatures(n_kmers: int = 5) -> dict:
    """Plant m6A signatures (peaks at +0 and +5) and m5C near-meth at centre.

    Parent meth and offset are written into cols 36/37 — analyze should
    read them directly and produce per-meth-type buckets.
    """
    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]
    m5c = meth_ids["m5C"]
    data = {}
    for kid in range(n_kmers):
        rows = []
        # Baseline: flat profile
        for _ in range(40):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 30.0
            r[COL_CATEGORY] = CATEGORY_BASELINE
            r[14:23] = 30.0
            rows.append(r)
        # Slowed-by-m6A at p = m6A (offset 0): m6A in mc[7]
        for _ in range(20):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 200.0
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = m6a
            r[COL_PARENT_OFFSET] = 0
            r[3 + KMER_PRED_IDX] = m6a
            r[14:23] = 30.0
            r[14] = 200.0
            r[19] = 180.0
            rows.append(r)
        # Slowed-by-m6A at p = m6A+5 (offset 5): m6A at mc[2] (-5 from centre)
        for _ in range(15):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 150.0
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = m6a
            r[COL_PARENT_OFFSET] = 5
            r[3 + KMER_PRED_IDX - 5] = m6a
            r[14:23] = 30.0
            r[14] = 150.0
            r[19] = 130.0
            rows.append(r)
        # Near_meth-by-m6A at p = m6A+3 (non-sig): m6A at mc[4] (-3 from centre)
        for _ in range(10):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 32.0
            r[COL_CATEGORY] = CATEGORY_NEAR_METH
            r[COL_PARENT_METH] = m6a
            r[COL_PARENT_OFFSET] = 3
            r[3 + KMER_PRED_IDX - 3] = m6a
            r[14:23] = 32.0
            rows.append(r)
        # Near_meth-by-m5C at p = m5C (offset 0; 0 not in m5C sig=[2,6])
        for _ in range(8):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 35.0
            r[COL_CATEGORY] = CATEGORY_NEAR_METH
            r[COL_PARENT_METH] = m5c
            r[COL_PARENT_OFFSET] = 0
            r[3 + KMER_PRED_IDX] = m5c
            r[14:23] = 35.0
            rows.append(r)
        data[kid] = np.stack(rows)
    return data


def test_compute_signature_profiles():
    """Profiles aggregated by CATEGORY column with per-type attribution."""
    from kinsim.analyze import compute_signature_profiles

    data = _build_v4_with_signatures(5)
    profiles = compute_signature_profiles(data)

    # Expected buckets
    assert "baseline" in profiles
    assert "slowed_by_m6A" in profiles
    assert "near_meth_by_m6A" in profiles
    assert "near_meth_by_m5C" in profiles

    base = profiles["baseline"]
    assert base["n_samples"] == 5 * 40
    for v in base["profile_ipd"]:
        assert abs(v - 30.0) < 0.1

    slow = profiles["slowed_by_m6A"]
    # 20 (k=0, IPD@+0=200, IPD@+5=180) + 15 (k=5, IPD@+0=150, IPD@+5=130)
    assert slow["n_samples"] == 5 * 35
    # Average IPD at +0 = (20*200 + 15*150) / 35 = (4000+2250)/35 = 178.57
    assert abs(slow["profile_ipd"][0] - 178.57) < 1.0

    near_a = profiles["near_meth_by_m6A"]
    assert near_a["n_samples"] == 5 * 10
    assert abs(near_a["profile_ipd"][0] - 32.0) < 0.1

    near_c = profiles["near_meth_by_m5C"]
    assert near_c["n_samples"] == 5 * 8
    assert abs(near_c["profile_ipd"][0] - 35.0) < 0.1


def test_compute_meth_context_distribution():
    """v4 path: context buckets match the CATEGORY split."""
    from kinsim.analyze import compute_meth_context_distribution

    data = _build_v4_with_signatures(5)
    ctx = compute_meth_context_distribution(data)

    assert "baseline" in ctx
    assert "slowed_by_m6A" in ctx
    assert "near_meth_by_m6A" in ctx
    assert "near_meth_by_m5C" in ctx

    meth_ids = get_meth_ids()
    m5c = meth_ids["m5C"]

    # baseline: all-none
    base = ctx["baseline"]
    none_col = base["meth_ids"].index(0)
    for pos in range(11):
        assert abs(base["fractions"][pos, none_col] - 1.0) < 1e-6

    # near_meth_by_m5C: m5C at center
    near_c = ctx["near_meth_by_m5C"]
    m5c_col = near_c["meth_ids"].index(m5c)
    assert abs(near_c["fractions"][KMER_PRED_IDX, m5c_col] - 1.0) < 1e-6


if __name__ == "__main__":
    test_category_constants_are_three()
    print("[pass] category constants")
    test_get_categories_reads_col35()
    print("[pass] get_categories")
    test_layout_column_contract()
    print("[pass] layout column contract (38 cols, 35 cat, 36 parent_meth, 37 parent_off)")
    test_slowed_split_only_filters_slowed()
    print("[pass] slowed_split")
    test_refine_pkl_writes_output()
    print("[pass] refine_pkl")
    test_refine_fails_fast_on_obsolete_layout()
    print("[pass] refine fails fast on obsolete layout")
    test_compute_signature_profiles()
    print("[pass] compute_signature_profiles")
    test_analyze_uses_parent_meth_column_not_meth_context()
    print("[pass] analyze trusts COL_PARENT_METH over mc[]")
    test_compute_meth_context_distribution()
    print("[pass] compute_meth_context_distribution")
    print("\nAll tests passed.")
