"""Unit tests for kinsim.refine.slowed_split (v4 pass 2).

Covers the two contracts the v4 split relies on:

1. Meth-context column layout — for a sample whose center sits at p+k
   (where p had a methylation of type T), the meth_context column at index
   KMER_PRED_IDX - k must equal the meth_id for T. This is what allows
   slowed_split to detect upstream methylations purely from the stored
   columns 3..13 without re-reading the BAM.

2. Slowed-vs-baseline classification + percentile filter — given a curated
   none-bucket dict, slowed_split must (a) correctly tag samples whose
   meth_context flags an upstream signature offset, (b) cap baseline
   samples per kmer, and (c) drop slowed samples whose center IPD falls
   below the configured percentile of the baseline IPD distribution.
"""

from __future__ import annotations

import numpy as np

from kinsim.refine import _build_upstream_signature_targets, slowed_split
from kinsim.utils.encoding import KMER_PRED_IDX, get_meth_ids
from kinsim.utils.sample_layout import METH_CTX_LEN, slice_meth_context as _slice_meth_context


# ---------------------------------------------------------------------------
# Property: meth_context[KMER_PRED_IDX - k] reflects the meth at p when the
# sample's center is at p+k. This is the contract slowed_split depends on.
# ---------------------------------------------------------------------------

def test_meth_context_encodes_upstream_meth_at_correct_offset():
    """For every signature offset k > 0 of every meth type T:
        sample at center p+k must have mc[KMER_PRED_IDX - k] == meth_id[T].
    Verifies the offset arithmetic shared by extract._slice_meth_context
    and refine._build_upstream_signature_targets.
    """
    meth_ids = get_meth_ids()
    # Build a meth_status array of length 50 with a meth at p=20 of every
    # known type at distinct positions: m6A@20, m4C@30, m5C@10.
    meth_status = np.zeros(50, dtype=np.int8)
    plant = {meth_ids["m6A"]: 20,
             meth_ids["m4C"]: 30,
             meth_ids["m5C"]: 10}
    for mid, p in plant.items():
        meth_status[p] = mid

    for mid, p in plant.items():
        for k in (1, 2, 3, 5, 6, 7):    # range of signature offsets we care about
            center = p + k
            mc = _slice_meth_context(meth_status, center)
            mc_idx = KMER_PRED_IDX - k
            assert 0 <= mc_idx < METH_CTX_LEN, (
                f"k={k} maps to mc_idx={mc_idx} out of [0,{METH_CTX_LEN})"
            )
            assert mc[mc_idx] == mid, (
                f"meth_id={mid} planted at p={p}; sample at center={p+k} "
                f"expected mc[{mc_idx}]=={mid} but got {mc[mc_idx]}. "
                f"Full mc={mc}"
            )


# ---------------------------------------------------------------------------
# slowed_split correctness
# ---------------------------------------------------------------------------

def _make_sample(ipd: float, mc: list[int], n_cols: int = 35) -> np.ndarray:
    """Fabricate a single sample row with arbitrary IPD and meth_context."""
    row = np.zeros(n_cols, dtype=np.float32)
    row[0] = ipd
    row[1] = 30.0          # PW (irrelevant for slowed_split)
    row[2] = 0.0           # fraction
    assert len(mc) == 11
    row[3:14] = mc
    # leave kinetic_profile (cols 14..31) and rev_meth (32..34) at 0 — slowed_split
    # only inspects cols 3..13 and col 0.
    return row


def test_slowed_split_classifies_and_filters_correctly():
    """End-to-end pass-2 contract: classify, cap baseline, threshold slowed."""
    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]

    # Config snippet enabling only m6A signatures (offsets 0 and 5). Offset 0
    # is invisible to a (kmer, 0) sample (center is always 0 there) — only
    # offset 5 should drive the classification.
    cfg = {
        "kinetic_signatures": {
            "m6A": {"signal_offsets": [0, 5]},
        },
    }
    targets = _build_upstream_signature_targets(cfg)
    assert targets == [(KMER_PRED_IDX - 5, m6a, "m6A", 5)], (
        f"unexpected targets: {targets}"
    )

    # Build a small none-bucket. mc[2] == m6A flags a slowed-by-m6A position;
    # all other entries are baseline. We use 60 baselines (cap=10 should drop
    # 50) and 8 slowed samples (4 high-IPD that survive the percentile, 4
    # low-IPD that get dropped).
    rng = np.random.default_rng(0)
    samples_kmer_A = []
    # 60 baseline (no upstream meth in window) with IPD ~ 50
    for _ in range(60):
        samples_kmer_A.append(_make_sample(ipd=50.0, mc=[0] * 11))
    # 4 slowed samples with high IPD (above p95 of baseline=50)
    high_mc = [0] * 11
    high_mc[KMER_PRED_IDX - 5] = m6a
    for _ in range(4):
        samples_kmer_A.append(_make_sample(ipd=200.0, mc=high_mc))
    # 4 slowed samples with low IPD (below p95 of baseline)
    for _ in range(4):
        samples_kmer_A.append(_make_sample(ipd=10.0, mc=high_mc))
    arr_A = np.stack(samples_kmer_A)

    # Build a kmer B with only baseline samples to verify cap on different kmers.
    samples_kmer_B = [_make_sample(ipd=40.0, mc=[0] * 11) for _ in range(20)]
    arr_B = np.stack(samples_kmer_B)

    none_buckets = {0xAAAA: arr_A, 0xBBBB: arr_B}

    new_buckets, stats = slowed_split(
        none_buckets, cfg,
        n_baseline_per_kmer=10,
        secondary_pct=95.0,
        rng=rng,
    )

    # All baselines have IPD in {50, 40} — strict majority is 50, so p95 ≈ 50.
    # Slowed samples at IPD=200 must survive; at IPD=10 must be dropped.
    assert stats["n_baseline_in"]    == 60 + 20
    assert stats["n_slowed_in"]      == 8
    assert stats["n_baseline_kept"]  == 10 + 10        # cap 10 per kmer
    assert stats["n_slowed_kept"]    == 4
    assert stats["n_slowed_dropped"] == 4
    assert stats["threshold"] is not None
    assert 30.0 <= stats["threshold"] <= 60.0          # p95 of (50.0)*60+(40.0)*20
    assert stats["offset_distribution"] == {"m6A+5": 8}

    # Per-bucket sizes: A has 10 baseline + 4 surviving slowed = 14; B has
    # only 10 surviving baseline.
    assert len(new_buckets[0xAAAA]) == 14
    assert len(new_buckets[0xBBBB]) == 10

    # IPDs of surviving slowed samples in A must all be >= threshold.
    a_arr = new_buckets[0xAAAA]
    is_slowed_a = (a_arr[:, 3 + (KMER_PRED_IDX - 5)] == m6a)
    slowed_in_out = a_arr[is_slowed_a]
    assert (slowed_in_out[:, 0] >= stats["threshold"]).all()


def test_slowed_split_no_signatures_skips_classification():
    """If no upstream signature offsets are configured (only k==0 entries),
    every sample is treated as baseline and only the cap applies."""
    cfg = {
        "kinetic_signatures": {
            "m4C": {"signal_offsets": [0]},   # only k==0 → invisible
        },
    }
    rng = np.random.default_rng(0)
    arr = np.stack([_make_sample(ipd=50.0, mc=[0] * 11) for _ in range(30)])
    new_buckets, stats = slowed_split(
        {1234: arr}, cfg,
        n_baseline_per_kmer=10,
        secondary_pct=95.0,
        rng=rng,
    )
    # All input samples are accounted for as baseline; no slowed.
    assert stats["n_baseline_in"]   == 30
    assert stats["n_slowed_in"]     == 0
    assert stats["n_baseline_kept"] == 30   # cap not applied: short-circuit
    assert stats["n_slowed_kept"]   == 0


if __name__ == "__main__":
    test_meth_context_encodes_upstream_meth_at_correct_offset()
    print("[pass] test_meth_context_encodes_upstream_meth_at_correct_offset")
    test_slowed_split_classifies_and_filters_correctly()
    print("[pass] test_slowed_split_classifies_and_filters_correctly")
    test_slowed_split_no_signatures_skips_classification()
    print("[pass] test_slowed_split_no_signatures_skips_classification")
