"""End-to-end unit tests for the v4 storage / refine / analyze pipeline
(simplified 3-category scheme: baseline / slowed / near_meth — no
separate METH category since the methylation centers themselves land
in SLOWED or NEAR_METH depending on whether 0 ∈ signature_offsets[T]).

What these tests pin down:

1. **Storage spec** — `is_v4_format` and `get_categories` correctly route
   v3 (35-col) and v4 (36-col) arrays. v3 inference of NEAR_METH from
   meth_context is also covered.

2. **Refine v4** — `slowed_split_v4` only filters CATEGORY_SLOWED below
   the configured percentile of baseline IPD. CATEGORY_BASELINE and
   CATEGORY_NEAR_METH pass through untouched.

3. **Analyze v4** — `compute_signature_profiles` and
   `compute_meth_context_distribution` produce buckets keyed by
   "baseline", "slowed_by_<T>", "near_meth_by_<T>".

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
    CATEGORY_BASELINE, CATEGORY_SLOWED, CATEGORY_NEAR_METH, CATEGORY_NAMES,
    is_v4_format, get_categories,
)


# ---------------------------------------------------------------------------
# Storage spec
# ---------------------------------------------------------------------------

def test_category_constants_are_three():
    """The simplified v4 scheme has exactly three categories."""
    assert CATEGORY_BASELINE  == 0
    assert CATEGORY_SLOWED    == 1
    assert CATEGORY_NEAR_METH == 2
    assert set(CATEGORY_NAMES.keys()) == {0, 1, 2}
    assert CATEGORY_NAMES[0] == "baseline"
    assert CATEGORY_NAMES[1] == "slowed"
    assert CATEGORY_NAMES[2] == "near_meth"


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
    arr[1, COL_CATEGORY] = CATEGORY_SLOWED
    arr[2, COL_CATEGORY] = CATEGORY_NEAR_METH
    cats = get_categories(arr)
    assert cats.tolist() == [CATEGORY_BASELINE, CATEGORY_SLOWED, CATEGORY_NEAR_METH]


def test_get_categories_v3_inferred_3cats():
    """v3 fallback: infers all 3 categories from meth_context.

    Setup:
      Row 0: empty mc                    → BASELINE
      Row 1: m6A at center (mc[7])      → SLOWED   (0 ∈ m6A sig=[0,5])
      Row 2: m5C at center               → NEAR_METH (0 ∉ m5C sig=[2,6])
      Row 3: m6A at offset -5            → SLOWED   (5 ∈ m6A sig)
      Row 4: m6A at offset -3            → NEAR_METH (3 ∉ m6A sig, in window)
    """
    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]
    m5c = meth_ids["m5C"]
    arr = np.zeros((5, SAMPLE_NCOLS_V3), dtype=np.float32)
    arr[1, 3 + KMER_PRED_IDX]     = m6a
    arr[2, 3 + KMER_PRED_IDX]     = m5c
    arr[3, 3 + KMER_PRED_IDX - 5] = m6a
    arr[4, 3 + KMER_PRED_IDX - 3] = m6a

    sig = {"m6A": [0, 5], "m5C": [2, 6]}
    cats = get_categories(arr, signature_offsets_by_meth=sig, near_meth_max_dist=7)
    assert cats.tolist() == [
        CATEGORY_BASELINE, CATEGORY_SLOWED, CATEGORY_NEAR_METH,
        CATEGORY_SLOWED, CATEGORY_NEAR_METH,
    ]


# ---------------------------------------------------------------------------
# Refine v4
# ---------------------------------------------------------------------------

def _build_v4_master(n_kmers: int = 10) -> dict:
    """Synthetic v4 master with the 3-cat scheme:
       50 baseline (IPD=50), 30 slowed-high (IPD=180), 20 slowed-low (IPD=20),
       25 near_meth (IPD=55) per kmer."""
    data = {}
    for kid in range(n_kmers):
        rows = []
        for _ in range(50):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 50.0; r[COL_CATEGORY] = CATEGORY_BASELINE
            rows.append(r)
        for _ in range(30):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 180.0; r[COL_CATEGORY] = CATEGORY_SLOWED
            rows.append(r)
        for _ in range(20):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 20.0; r[COL_CATEGORY] = CATEGORY_SLOWED
            rows.append(r)
        for _ in range(25):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 55.0; r[COL_CATEGORY] = CATEGORY_NEAR_METH
            rows.append(r)
        data[kid] = np.stack(rows)
    return data


def test_detect_format_v4():
    data = _build_v4_master(2)
    assert _detect_format(data) == "v4"


def test_slowed_split_v4_only_filters_slowed():
    """Only SLOWED below p95(baseline) is dropped. BASELINE and NEAR_METH
    pass through untouched."""
    data = _build_v4_master(5)
    rng = np.random.default_rng(0)
    new_data, stats = slowed_split_v4(data, secondary_pct=95.0, rng=rng)

    assert stats["format"]    == "v4"
    assert stats["threshold"] == 50.0   # all baselines IPD=50 -> p95=50

    # baseline + near_meth pass through
    assert stats["n_baseline_in"] == 5 * 50
    assert stats["n_baseline_out"] == 5 * 50
    assert stats["n_near_in"]      == 5 * 25
    assert stats["n_near_out"]     == 5 * 25

    # slowed: high (IPD=180) above threshold survive; low (IPD=20) drop
    assert stats["n_slowed_in"]      == 5 * 50
    assert stats["n_slowed_kept"]    == 5 * 30
    assert stats["n_slowed_dropped"] == 5 * 20


def test_refine_pkl_v4_dispatch_writes_output():
    """refine_pkl detects v4 and runs slowed_split_v4."""
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
        assert meta["method"] == "v4_p95_slowed"
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
    """Plant m6A signatures (peaks at +0 and +5) so analyze can recover them.
    profile_IPD@+0 col = 14, profile_IPD@+5 col = 19."""
    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]
    m5c = meth_ids["m5C"]
    data = {}
    for kid in range(n_kmers):
        rows = []
        # Baseline: flat profile
        for _ in range(40):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 30.0; r[COL_CATEGORY] = CATEGORY_BASELINE
            r[14:23] = 30.0
            rows.append(r)
        # Slowed-by-m6A at center (k=0): m6A in mc[7], peak at +0 and +5
        for _ in range(20):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 200.0; r[COL_CATEGORY] = CATEGORY_SLOWED
            r[3 + KMER_PRED_IDX] = m6a
            r[14:23] = 30.0; r[14] = 200.0; r[19] = 180.0
            rows.append(r)
        # Slowed-by-m6A at p+5: m6A at offset -5 in mc, profile peak too
        for _ in range(15):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 150.0; r[COL_CATEGORY] = CATEGORY_SLOWED
            r[3 + KMER_PRED_IDX - 5] = m6a
            r[14:23] = 30.0; r[14] = 150.0; r[19] = 130.0
            rows.append(r)
        # Near_meth-by-m6A at p+3: m6A at offset -3 (not sig), baseline IPD
        for _ in range(10):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 32.0; r[COL_CATEGORY] = CATEGORY_NEAR_METH
            r[3 + KMER_PRED_IDX - 3] = m6a
            r[14:23] = 32.0
            rows.append(r)
        # Near_meth-by-m5C at center (k=0 not in m5C sig=[2,6])
        for _ in range(8):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = 35.0; r[COL_CATEGORY] = CATEGORY_NEAR_METH
            r[3 + KMER_PRED_IDX] = m5c
            r[14:23] = 35.0
            rows.append(r)
        data[kid] = np.stack(rows)
    return data


def test_compute_signature_profiles_v4():
    """v4 path: profiles aggregated by CATEGORY column."""
    from kinsim.analyze import compute_signature_profiles

    data = _build_v4_with_signatures(5)
    profiles = compute_signature_profiles(data)

    # Expected buckets
    assert "baseline"            in profiles
    assert "slowed_by_m6A"       in profiles
    assert "near_meth_by_m6A"    in profiles
    assert "near_meth_by_m5C"    in profiles

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


def test_compute_meth_context_distribution_v4():
    """v4 path: context buckets match the CATEGORY split."""
    from kinsim.analyze import compute_meth_context_distribution

    data = _build_v4_with_signatures(5)
    ctx = compute_meth_context_distribution(data)

    assert "baseline"          in ctx
    assert "slowed_by_m6A"     in ctx
    assert "near_meth_by_m6A"  in ctx
    assert "near_meth_by_m5C"  in ctx

    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]
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
    test_category_constants_are_three();           print("[pass] category constants")
    test_is_v4_format();                           print("[pass] is_v4_format")
    test_get_categories_v4_reads_col35();          print("[pass] get_categories v4")
    test_get_categories_v3_inferred_3cats();       print("[pass] get_categories v3 (3 cats)")
    test_detect_format_v4();                       print("[pass] _detect_format v4")
    test_slowed_split_v4_only_filters_slowed();    print("[pass] slowed_split_v4")
    test_refine_pkl_v4_dispatch_writes_output();   print("[pass] refine_pkl v4 dispatch")
    test_compute_signature_profiles_v4();          print("[pass] compute_signature_profiles v4")
    test_compute_meth_context_distribution_v4();   print("[pass] compute_meth_context_distribution v4")
    print("\nAll v4 tests passed.")
