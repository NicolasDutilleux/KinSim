"""End-to-end unit tests for the v4 storage / refine / analyze pipeline.

What these tests pin down:

1. **Storage spec** — `is_v4_format` and `get_categories` correctly route
   v3 (35-col) and v4 (36-col) arrays.
2. **Refine pass-2 (v4)** — `slowed_split_v4` only filters slowed samples
   below the configured percentile of the baseline IPD; meth and baseline
   pass through untouched.
3. **Analyze v4** — `compute_signature_profiles` and
   `compute_meth_context_distribution` dispatch on the CATEGORY column
   directly, producing the same bucket structure as the v3 inference path
   would have produced.

These tests are the v4 equivalent of `tests/test_refine_slowed_split.py`,
which still covers the v3 inference path. Both should keep passing.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np

from kinsim.refine import refine_pkl, slowed_split_v4, _detect_format
from kinsim.utils.encoding import KMER_PRED_IDX, get_meth_ids
from kinsim.utils.sample_layout import (
    SAMPLE_NCOLS, SAMPLE_NCOLS_V3, COL_CATEGORY, COL_IPD,
    CATEGORY_BASELINE, CATEGORY_METH, CATEGORY_SLOWED,
    is_v4_format, get_categories,
)


# ---------------------------------------------------------------------------
# Storage spec
# ---------------------------------------------------------------------------

def test_is_v4_format():
    """is_v4_format gates on the 36-col layout."""
    arr_v3 = np.zeros((4, SAMPLE_NCOLS_V3), dtype=np.float32)
    arr_v4 = np.zeros((4, SAMPLE_NCOLS), dtype=np.float32)
    assert not is_v4_format(arr_v3)
    assert is_v4_format(arr_v4)


def test_get_categories_v4_reads_col35():
    """v4 path: get_categories returns col 35 verbatim."""
    arr = np.zeros((3, SAMPLE_NCOLS), dtype=np.float32)
    arr[0, COL_CATEGORY] = CATEGORY_BASELINE
    arr[1, COL_CATEGORY] = CATEGORY_METH
    arr[2, COL_CATEGORY] = CATEGORY_SLOWED
    cats = get_categories(arr)
    assert cats.tolist() == [
        CATEGORY_BASELINE, CATEGORY_METH, CATEGORY_SLOWED,
    ]


def test_get_categories_v3_inferred():
    """v3 fallback: infers from meth_context using upstream signature offsets."""
    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]
    arr = np.zeros((3, SAMPLE_NCOLS_V3), dtype=np.float32)
    # Row 0: empty mc — should be baseline.
    # Row 1: m6A at center (mc[7]) — should be CATEGORY_METH.
    arr[1, 3 + KMER_PRED_IDX] = m6a
    # Row 2: m6A at offset -5 (mc[2]) — slowed by m6A.
    arr[2, 3 + KMER_PRED_IDX - 5] = m6a
    sig = {"m6A": [0, 5]}
    cats = get_categories(arr, signature_offsets_by_meth=sig)
    assert cats.tolist() == [
        CATEGORY_BASELINE, CATEGORY_METH, CATEGORY_SLOWED,
    ]


# ---------------------------------------------------------------------------
# Refine pass-2 (v4)
# ---------------------------------------------------------------------------

def _build_v4_master(n_kmers: int = 10) -> dict:
    """Synthetic v4 master: 50 baseline (IPD=50), 30 meth (IPD=180),
    20 slowed-high (IPD=140), 20 slowed-low (IPD=20) per kmer."""
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
            r[COL_CATEGORY] = CATEGORY_METH
            rows.append(r)
        for _ in range(20):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 140.0
            r[COL_CATEGORY] = CATEGORY_SLOWED
            rows.append(r)
        for _ in range(20):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 20.0
            r[COL_CATEGORY] = CATEGORY_SLOWED
            rows.append(r)
        data[kid] = np.stack(rows)
    return data


def test_detect_format_v4():
    data = _build_v4_master(2)
    assert _detect_format(data) == "v4"


def test_slowed_split_v4_filters_slowed_only():
    """Pass-2 v4: meth and baseline pass through, slowed below p95(baseline)
    are dropped."""
    data = _build_v4_master(5)
    rng = np.random.default_rng(0)
    new_data, stats = slowed_split_v4(data, secondary_pct=95.0, rng=rng)

    assert stats["format"] == "v4"
    # All baseline IPDs are 50 -> p95 = 50. Slowed IPD=140 above, IPD=20 below.
    assert stats["threshold"] == 50.0
    # 5 kmers * 50 baseline = 250
    assert stats["n_baseline_in"]  == 5 * 50
    assert stats["n_baseline_out"] == 5 * 50      # pass-through
    assert stats["n_meth_in"]      == 5 * 30
    assert stats["n_meth_out"]     == 5 * 30      # pass-through
    assert stats["n_slowed_in"]    == 5 * 40
    assert stats["n_slowed_kept"]  == 5 * 20      # high-IPD survive
    assert stats["n_slowed_dropped"] == 5 * 20    # low-IPD dropped


def test_refine_pkl_v4_dispatch_writes_output():
    """refine_pkl detects v4 input and calls slowed_split_v4, writing a valid
    v4 output pkl."""
    data = _build_v4_master(3)
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.pkl"
        out = Path(td) / "out.pkl"
        with open(inp, "wb") as f:
            pickle.dump(data, f)
        refine_pkl(inp, out, seed=42)
        with open(out, "rb") as f:
            refined = pickle.load(f)
        meta = refined.pop("__meta__")
        assert meta["format"] == "v4"
        assert meta["method"] == "slowed_split_v4"
        # All meth + baseline + half the slowed survive: 80 + 20 = 100 per kmer.
        for kid in range(3):
            assert kid in refined
            arr = refined[kid]
            assert arr.shape[1] == SAMPLE_NCOLS
            assert len(arr) == 100


# ---------------------------------------------------------------------------
# Analyze v4 dispatch
# ---------------------------------------------------------------------------

def _build_v4_with_signatures(n_kmers: int = 5) -> dict:
    """v4 master with planted m6A signatures so analyze can recover them.
    profile_IPD@+0 col = 14, profile_IPD@+5 col = 19."""
    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]
    data = {}
    for kid in range(n_kmers):
        rows = []
        # Baseline: flat profile
        for _ in range(40):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 30.0; r[COL_CATEGORY] = CATEGORY_BASELINE
            r[14:23] = 30.0
            rows.append(r)
        # m6A meth: peak at +0 and +5
        for _ in range(20):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 200.0; r[COL_CATEGORY] = CATEGORY_METH
            r[3 + KMER_PRED_IDX] = m6a
            r[14:23] = 30.0
            r[14] = 200.0
            r[19] = 180.0
            rows.append(r)
        # Slowed by m6A: m6A at offset -5 in mc, profile peaks at +5 too
        for _ in range(15):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 150.0; r[COL_CATEGORY] = CATEGORY_SLOWED
            r[3 + KMER_PRED_IDX - 5] = m6a
            r[14:23] = 30.0
            r[14] = 150.0
            r[19] = 130.0
            rows.append(r)
        data[kid] = np.stack(rows)
    return data


def test_compute_signature_profiles_v4():
    """v4 path: profiles are aggregated by CATEGORY column."""
    from kinsim.analyze import compute_signature_profiles

    data = _build_v4_with_signatures(5)
    profiles = compute_signature_profiles(data)

    assert "m6A" in profiles
    assert "none/baseline" in profiles
    assert "none/slowed_by_m6A" in profiles

    m6a = profiles["m6A"]
    assert m6a["n_samples"] == 5 * 20
    # m6A profile peaks at +0 and +5
    assert abs(m6a["profile_ipd"][0] - 200.0) < 0.1
    assert abs(m6a["profile_ipd"][5] - 180.0) < 0.1

    base = profiles["none/baseline"]
    assert base["n_samples"] == 5 * 40
    # Baseline profile is flat at 30.0
    for v in base["profile_ipd"]:
        assert abs(v - 30.0) < 0.1

    slow = profiles["none/slowed_by_m6A"]
    assert slow["n_samples"] == 5 * 15
    # Slowed profile peaks at +0 and +5 (weaker than meth)
    assert abs(slow["profile_ipd"][0] - 150.0) < 0.1
    assert abs(slow["profile_ipd"][5] - 130.0) < 0.1


def test_compute_meth_context_distribution_v4():
    """v4 path: context distribution buckets match the CATEGORY split."""
    from kinsim.analyze import compute_meth_context_distribution

    data = _build_v4_with_signatures(5)
    ctx = compute_meth_context_distribution(data)

    assert "m6A" in ctx
    assert "none/baseline" in ctx
    assert "none/slowed_by_m6A" in ctx

    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]

    # In the m6A bucket every sample has m6A at center.
    m6a_bkt = ctx["m6A"]
    m6a_col = m6a_bkt["meth_ids"].index(m6a)
    assert abs(m6a_bkt["fractions"][KMER_PRED_IDX, m6a_col] - 1.0) < 1e-6

    # In the slowed-by-m6A bucket every sample has m6A at offset -5 (mc[2]).
    slow_bkt = ctx["none/slowed_by_m6A"]
    assert abs(slow_bkt["fractions"][KMER_PRED_IDX - 5, m6a_col] - 1.0) < 1e-6

    # Baseline bucket has all-none everywhere.
    base_bkt = ctx["none/baseline"]
    none_col = base_bkt["meth_ids"].index(0)
    for pos in range(11):
        assert abs(base_bkt["fractions"][pos, none_col] - 1.0) < 1e-6


if __name__ == "__main__":
    test_is_v4_format();                           print("[pass] is_v4_format")
    test_get_categories_v4_reads_col35();          print("[pass] get_categories v4")
    test_get_categories_v3_inferred();             print("[pass] get_categories v3")
    test_detect_format_v4();                       print("[pass] _detect_format v4")
    test_slowed_split_v4_filters_slowed_only();    print("[pass] slowed_split_v4")
    test_refine_pkl_v4_dispatch_writes_output();   print("[pass] refine_pkl v4 dispatch")
    test_compute_signature_profiles_v4();          print("[pass] compute_signature_profiles v4")
    test_compute_meth_context_distribution_v4();   print("[pass] compute_meth_context_distribution v4")
    print("\nAll v4 tests passed.")
