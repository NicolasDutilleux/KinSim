"""End-to-end unit tests for the storage / refine / analyze pipeline.

Three categories: baseline / slowed / near_meth. The methylation
centers themselves land in SLOWED or NEAR_METH depending on whether 0
is a signature offset for that type. Parent meth + offset attribution
is written at extract time into ``COL_PARENT_METH`` (col 36) and
``COL_PARENT_OFFSET`` (col 37). Refine and analyze bucket per
**(meth_type, parent_offset)** so a noisy offset of one meth type
never contaminates a clean offset of the same type.

Tests cover:
  1. Storage constants + get_categories on the 38-col layout.
  2. Refine: slowed_split filters CATEGORY_SLOWED below the
     per-kmer-mean baseline percentile; CATEGORY_BASELINE and
     CATEGORY_NEAR_METH pass through untouched.
  3. Refine GMM fits one model per (meth_type, offset) bucket.
  4. Analyze: compute_signature_profiles and
     compute_meth_context_distribution produce
     "baseline" / "slowed_by_<T>_at_+<O>" / "near_meth_by_<T>_at_+<O>"
     buckets using cols 36/37 directly (no mc[] inference).
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np

from kinsim.refine import refine_pkl, slowed_split, slowed_split_gmm
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
    COL_PW,
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
    # Both rows have COL_PARENT_OFFSET = 0 by default (np.zeros) → bucket
    # name encodes the offset in the suffix.
    assert "slowed_by_m6A_at_+0" in profiles
    assert "slowed_by_m5C_at_+0" in profiles
    assert profiles["slowed_by_m6A_at_+0"]["n_samples"] == 1
    assert profiles["slowed_by_m5C_at_+0"]["n_samples"] == 1


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


def test_refine_pkl_writes_output_p95():
    """Legacy p95 method end-to-end: loads, runs slowed_split, writes a clean pkl."""
    data = _build_v4_master(3)
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.pkl"
        out = Path(td) / "out.pkl"
        with open(inp, "wb") as f:
            pickle.dump(data, f)
        refine_pkl(inp, out, method="p95")
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
# Refine — GMM method (default)
# ---------------------------------------------------------------------------


def _build_gmm_dataset(
    n_kmers: int = 200,
    n_baseline_per_kmer: int = 50,
    n_slowed_per_kmer: int = 20,
    baseline_mu: float = 30.0,
    baseline_sigma: float = 5.0,
    slowed_mu: float = 90.0,
    slowed_sigma: float = 10.0,
    contamination_rate: float = 0.0,
    contamination_mu: float = 30.0,
    contamination_sigma: float = 5.0,
    parent_meth_id: int = 1,  # m6A
    seed: int = 0,
) -> dict:
    """Build a synthetic master with two clean Gaussians for GMM testing.

    ``contamination_rate`` of slowed rows are drawn from the baseline
    Gaussian instead of the slowed one — those are the FP motif matches
    the GMM should drop.
    """
    rng = np.random.default_rng(seed)
    data: dict = {}

    # PW is populated as a noisy linear function of IPD so the GMM's
    # covariance matrix is non-singular (otherwise the multivariate fit
    # degenerates on a 1D subspace).
    def _pw_from_ipd(ipd: float) -> float:
        return float(max(0.0, 0.5 * ipd + rng.normal(0.0, 1.5)))

    for kid in range(n_kmers):
        rows = []
        # Baseline rows
        base_ipd = rng.normal(baseline_mu, baseline_sigma, n_baseline_per_kmer)
        for ipd in base_ipd:
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            r[COL_IPD] = float(max(0.0, ipd))
            r[COL_PW] = _pw_from_ipd(r[COL_IPD])
            r[COL_CATEGORY] = CATEGORY_BASELINE
            rows.append(r)
        # Slowed rows — mostly real, with `contamination_rate` from baseline.
        for _ in range(n_slowed_per_kmer):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            if rng.random() < contamination_rate:
                ipd = float(max(0.0, rng.normal(contamination_mu, contamination_sigma)))
            else:
                ipd = float(max(0.0, rng.normal(slowed_mu, slowed_sigma)))
            r[COL_IPD] = ipd
            r[COL_PW] = _pw_from_ipd(ipd)
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = parent_meth_id
            r[COL_PARENT_OFFSET] = 0
            rows.append(r)
        data[kid] = np.stack(rows)
    return data


def test_gmm_k2_separates_clean_two_distributions():
    """K=2 GMM on well-separated baseline (μ=30) + slowed (μ=90) Gaussians:
    a 25 % contamination of slowed-from-baseline should be largely dropped."""
    data = _build_gmm_dataset(
        n_kmers=200,
        n_baseline_per_kmer=50,
        n_slowed_per_kmer=40,
        baseline_mu=30.0,
        baseline_sigma=5.0,
        slowed_mu=90.0,
        slowed_sigma=10.0,
        contamination_rate=0.25,
        seed=1,
    )
    # K=2 because the synthetic data is genuinely bimodal — K=3 would
    # over-fit the second baseline mode and the slowed Gaussian together.
    _new_data, stats = slowed_split_gmm(data, n_components=2, seed=42)

    # Baseline + near pass through.
    assert stats["n_baseline_in"] == stats["n_baseline_out"]
    assert stats["n_near_in"] == stats["n_near_out"]

    # Slowed: ~75 % real, ~25 % contamination. Keep should land near 75 %.
    n_in = stats["n_slowed_in"]
    n_kept = stats["n_slowed_kept"]
    survival = n_kept / n_in
    assert 0.65 < survival < 0.85, (
        f"survival {survival:.2%} outside expected 65-85% (kept {n_kept}/{n_in})"
    )

    # Per-bucket fit stored with reasonable means (lowest ~30, highest ~90).
    assert stats["method"] == "gmm_per_meth_offset"
    m6a_stats = stats["per_bucket"]["m6A@+0"]
    assert not m6a_stats["skipped"]
    # 2D fit: gmm_means is (K, 2) — read the IPD axis only for the assertion.
    ipd_means = sorted(row[0] for row in m6a_stats["gmm_means"])
    assert 25 < ipd_means[0] < 40, f"lowest-IPD-mean ~30 expected, got {ipd_means[0]}"
    assert 80 < ipd_means[-1] < 100, f"highest-IPD-mean ~90 expected, got {ipd_means[-1]}"


def _set_ipd_pw(r, ipd, rng):
    """Helper: set IPD and a PW that is a noisy linear function of IPD.

    The 2D GMM needs both axes populated and weakly correlated to avoid
    a singular covariance matrix on the PW axis.
    """
    r[COL_IPD] = float(max(0.0, ipd))
    r[COL_PW] = float(max(0.0, 0.5 * r[COL_IPD] + rng.normal(0.0, 1.5)))


def test_gmm_k3_handles_long_tail_baseline():
    """K=3 should cleanly handle a long-tailed baseline (PacBio-realistic):
    baseline is N(30, 5) + N(70, 15) tail; slowed is N(120, 10).
    The two lower components capture the bimodal baseline; the highest
    captures the meth signal. Validation passes; meth survives cleanly."""
    rng = np.random.default_rng(7)
    n_kmers = 100
    data: dict = {}
    for kid in range(n_kmers):
        rows = []
        # Bimodal baseline (PacBio-realistic): 70 % fast, 30 % long tail.
        for _ in range(100):
            mu, sig = (30.0, 5.0) if rng.random() < 0.7 else (70.0, 15.0)
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            _set_ipd_pw(r, rng.normal(mu, sig), rng)
            r[COL_CATEGORY] = CATEGORY_BASELINE
            rows.append(r)
        # Slowed: clean Gaussian at μ=120, well above the baseline tail.
        for _ in range(50):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            _set_ipd_pw(r, rng.normal(120.0, 10.0), rng)
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = 1  # m6A
            r[COL_PARENT_OFFSET] = 0
            rows.append(r)
        data[kid] = np.stack(rows)

    # K=3 (default) should handle this cleanly.
    _new_data, stats = slowed_split_gmm(data, n_components=3, seed=42)
    m6a_stats = stats["per_bucket"]["m6A@+0"]

    # Validation passes — baseline cleanly captured by the two lower comps.
    assert not m6a_stats["skipped"], (
        f"K=3 should pass validation on long-tail baseline; got "
        f"{m6a_stats.get('reason')} (pct_in_baseline="
        f"{m6a_stats.get('baseline_in_baseline_pct')})"
    )
    # Most of the slowed (which is at μ=120, far above baseline) should survive.
    survival = stats["n_slowed_kept"] / stats["n_slowed_in"]
    assert survival > 0.85, f"survival {survival:.2%} too low for clean meth signal"
    # The meth_idx component's IPD mean should be ~120 (far above baseline).
    means = m6a_stats["gmm_means"]  # (K, 2)
    meth_ipd_mean = means[m6a_stats["meth_idx"]][0]
    assert meth_ipd_mean > 100, f"meth-component IPD mean expected > 100, got {meth_ipd_mean}"


def test_gmm_validation_fails_when_meth_indistinguishable_from_baseline():
    """If 'slowed' overlaps the upper baseline mode (no real elevation),
    the GMM places it inside one of the baseline-like components and the
    fit produces no meth peak. The validation must reject (or the cut
    keeps essentially everything because no rows have posterior_meth ≥ 0.5).
    Either way the safety net prevents dropping valid baselines."""
    rng = np.random.default_rng(0)
    n_kmers = 100
    data: dict = {}
    for kid in range(n_kmers):
        rows = []
        # Tight baseline at μ=30
        for _ in range(100):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            _set_ipd_pw(r, rng.normal(30.0, 4.0), rng)
            r[COL_CATEGORY] = CATEGORY_BASELINE
            rows.append(r)
        # "Slowed" at μ=32 — indistinguishable from baseline
        for _ in range(200):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            _set_ipd_pw(r, rng.normal(32.0, 4.0), rng)
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = 1  # m6A
            r[COL_PARENT_OFFSET] = 0
            rows.append(r)
        data[kid] = np.stack(rows)

    _new_data, stats = slowed_split_gmm(data, n_components=3, seed=42)
    m6a_stats = stats["per_bucket"]["m6A@+0"]
    # Either validation rejects, OR no slowed gets cut because the meth
    # component sits inside the baseline range (P(meth | x) low everywhere).
    survival = stats["n_slowed_kept"] / stats["n_slowed_in"]
    # In both scenarios, we expect "keep all" (defensive) — the filter
    # should not aggressively drop slowed when the meth signal is absent.
    assert survival > 0.95 or m6a_stats["skipped"], (
        f"expected defensive behaviour (≥ 95 % survival or skip), got "
        f"survival={survival:.2%} skipped={m6a_stats.get('skipped')}"
    )


def test_gmm_bic_picks_k2_for_unimodal_baseline():
    """BIC over (2, 3) should select K=2 when baseline is unimodal —
    K=3 would over-fit the unimodal baseline into 3 sub-components."""
    data = _build_gmm_dataset(
        n_kmers=200,
        n_baseline_per_kmer=80,
        n_slowed_per_kmer=40,
        baseline_mu=30.0,
        baseline_sigma=4.0,  # unimodal, tight
        slowed_mu=100.0,
        slowed_sigma=8.0,
        contamination_rate=0.0,
        seed=11,
    )
    _new_data, stats = slowed_split_gmm(data, n_components=(2, 3), seed=42)
    m6a = stats["per_bucket"]["m6A@+0"]
    assert not m6a["skipped"]
    assert m6a["n_components_used"] == 2, (
        f"BIC should pick K=2 for unimodal baseline; picked K={m6a['n_components_used']} "
        f"(BICs: {m6a['bic_per_k']})"
    )


def test_gmm_bic_picks_k3_for_long_tail_baseline():
    """BIC over (2, 3) should select K=3 when baseline has a long tail —
    K=2 would conflate the baseline tail with the meth signal."""
    rng = np.random.default_rng(13)
    n_kmers = 100
    data: dict = {}
    for kid in range(n_kmers):
        rows = []
        # Bimodal baseline (PacBio-realistic)
        for _ in range(150):
            mu, sig = (30.0, 4.0) if rng.random() < 0.7 else (75.0, 12.0)
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            _set_ipd_pw(r, rng.normal(mu, sig), rng)
            r[COL_CATEGORY] = CATEGORY_BASELINE
            rows.append(r)
        # Strong meth signal at μ=140
        for _ in range(60):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            _set_ipd_pw(r, rng.normal(140.0, 10.0), rng)
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = 1  # m6A
            r[COL_PARENT_OFFSET] = 0
            rows.append(r)
        data[kid] = np.stack(rows)

    _new_data, stats = slowed_split_gmm(data, n_components=(2, 3), seed=42)
    m6a = stats["per_bucket"]["m6A@+0"]
    assert m6a["n_components_used"] == 3, (
        f"BIC should pick K=3 for long-tail baseline + clean meth signal; "
        f"picked K={m6a['n_components_used']} (BICs: {m6a['bic_per_k']})"
    )


def test_gmm_too_few_slowed_keeps_all():
    """A meth type with fewer than ``min_samples_for_gmm`` slowed rows
    should skip the fit entirely and keep them all."""
    data = _build_gmm_dataset(
        n_kmers=2,  # → only 40 slowed total
        n_baseline_per_kmer=100,
        n_slowed_per_kmer=20,
        seed=2,
    )
    _new_data, stats = slowed_split_gmm(data, min_samples_for_gmm=200, seed=42)
    m6a_stats = stats["per_bucket"]["m6A@+0"]
    assert m6a_stats["skipped"] is True
    assert m6a_stats["reason"] == "too_few_samples"
    assert stats["n_slowed_kept"] == stats["n_slowed_in"]


def test_gmm_per_offset_isolates_noisy_offset_from_clean_offset():
    """Two offsets of the same meth type are fit independently.

    Synthetic dataset: m6A@+0 has a clean meth signal at IPD≈100;
    m6A@+5 is indistinguishable from baseline (the +5 signature
    doesn't actually exist for this synthetic motif). The new
    per-(T, offset) refine should:
      - fit m6A@+0 cleanly and drop the contaminating tail
      - mark m6A@+5 as defensive (skipped or near-100% survival)
    so the noisy @+5 bucket cannot poison the @+0 bucket.
    """
    rng = np.random.default_rng(101)
    n_kmers = 80
    data: dict = {}
    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]
    for kid in range(n_kmers):
        rows = []
        # Clean baseline at μ=30
        for _ in range(80):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            _set_ipd_pw(r, rng.normal(30.0, 4.0), rng)
            r[COL_CATEGORY] = CATEGORY_BASELINE
            rows.append(r)
        # m6A@+0 — real meth at μ=100, with 25% baseline contamination
        for _ in range(40):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            ipd = rng.normal(30.0, 4.0) if rng.random() < 0.25 else rng.normal(100.0, 8.0)
            _set_ipd_pw(r, ipd, rng)
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = m6a
            r[COL_PARENT_OFFSET] = 0
            rows.append(r)
        # m6A@+5 — pure baseline (no real signature)
        for _ in range(40):
            r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
            _set_ipd_pw(r, rng.normal(30.0, 4.0), rng)
            r[COL_CATEGORY] = CATEGORY_SLOWED
            r[COL_PARENT_METH] = m6a
            r[COL_PARENT_OFFSET] = 5
            rows.append(r)
        data[kid] = np.stack(rows)

    _new_data, stats = slowed_split_gmm(data, n_components=(2, 3), seed=42)
    by_bucket = stats["per_bucket"]

    # Both buckets must be reported separately.
    assert "m6A@+0" in by_bucket, f"got buckets: {list(by_bucket)}"
    assert "m6A@+5" in by_bucket, f"got buckets: {list(by_bucket)}"

    b0 = by_bucket["m6A@+0"]
    b5 = by_bucket["m6A@+5"]

    # m6A@+0 must be a successful fit with most slowed surviving.
    assert not b0["skipped"], (
        f"m6A@+0 should fit cleanly (got skipped={b0.get('reason')})"
    )
    survival0 = b0["n_kept"] / b0["n_in"]
    assert 0.55 < survival0 < 0.95, (
        f"m6A@+0 survival {survival0:.2%} outside expected ~75% range"
    )

    # m6A@+5 is indistinguishable from baseline → defensive (skipped or
    # ~100 % kept). Either way, it must NOT have dropped m6A@+0.
    survival5 = b5["n_kept"] / b5["n_in"]
    assert b5.get("skipped") or survival5 > 0.95, (
        f"m6A@+5 should be defensive when meth signal is absent; "
        f"skipped={b5.get('skipped')} survival={survival5:.2%}"
    )


def test_refine_pkl_writes_output_gmm():
    """Default GMM end-to-end: writes a clean pkl with method recorded in __meta__."""
    data = _build_gmm_dataset(
        n_kmers=50,
        n_baseline_per_kmer=80,
        n_slowed_per_kmer=40,
        contamination_rate=0.3,
        seed=3,
    )
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.pkl"
        out = Path(td) / "out.pkl"
        with open(inp, "wb") as f:
            pickle.dump(data, f)
        refine_pkl(inp, out)  # default method=gmm
        with open(out, "rb") as f:
            refined = pickle.load(f)
        meta = refined.pop("__meta__")
        assert meta["method"] == "gmm_per_meth_offset"
        assert "m6A@+0" in meta["stats"]["per_bucket"]
        # Each kmer has at least baseline rows surviving.
        for kid in range(50):
            assert kid in refined
            assert refined[kid].shape[1] == SAMPLE_NCOLS


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
    """Profiles aggregated by CATEGORY × PARENT_METH × PARENT_OFFSET.

    The test fixture plants four parent buckets:
      - slowed_by_m6A_at_+0  (n=20 per kmer, IPD=200)
      - slowed_by_m6A_at_+5  (n=15 per kmer, IPD=150)
      - near_meth_by_m6A_at_+3  (n=10 per kmer, IPD=32)
      - near_meth_by_m5C_at_+0  (n=8  per kmer, IPD=35)
    Per-offset buckets keep clean and noisy offsets of the same meth
    type isolated in the report — m6A@+0 and m6A@+5 are reported
    separately so a flat +5 cannot mask an active +0.
    """
    from kinsim.analyze import compute_signature_profiles

    data = _build_v4_with_signatures(5)
    profiles = compute_signature_profiles(data)

    assert "baseline" in profiles
    assert "slowed_by_m6A_at_+0" in profiles
    assert "slowed_by_m6A_at_+5" in profiles
    assert "near_meth_by_m6A_at_+3" in profiles
    assert "near_meth_by_m5C_at_+0" in profiles

    base = profiles["baseline"]
    assert base["n_samples"] == 5 * 40
    for v in base["profile_ipd"]:
        assert abs(v - 30.0) < 0.1

    slow0 = profiles["slowed_by_m6A_at_+0"]
    assert slow0["n_samples"] == 5 * 20
    assert abs(slow0["profile_ipd"][0] - 200.0) < 0.5
    assert slow0["sig_offsets"] == [0]

    slow5 = profiles["slowed_by_m6A_at_+5"]
    assert slow5["n_samples"] == 5 * 15
    assert abs(slow5["profile_ipd"][0] - 150.0) < 0.5
    assert slow5["sig_offsets"] == [5]

    near_a = profiles["near_meth_by_m6A_at_+3"]
    assert near_a["n_samples"] == 5 * 10
    assert abs(near_a["profile_ipd"][0] - 32.0) < 0.1

    near_c = profiles["near_meth_by_m5C_at_+0"]
    assert near_c["n_samples"] == 5 * 8
    assert abs(near_c["profile_ipd"][0] - 35.0) < 0.1


def test_compute_meth_context_distribution():
    """v4 path: context buckets match the CATEGORY × parent_meth × offset split."""
    from kinsim.analyze import compute_meth_context_distribution

    data = _build_v4_with_signatures(5)
    ctx = compute_meth_context_distribution(data)

    assert "baseline" in ctx
    assert "slowed_by_m6A_at_+0" in ctx
    assert "slowed_by_m6A_at_+5" in ctx
    assert "near_meth_by_m6A_at_+3" in ctx
    assert "near_meth_by_m5C_at_+0" in ctx

    meth_ids = get_meth_ids()
    m5c = meth_ids["m5C"]

    # baseline: all-none
    base = ctx["baseline"]
    none_col = base["meth_ids"].index(0)
    for pos in range(11):
        assert abs(base["fractions"][pos, none_col] - 1.0) < 1e-6

    # near_meth_by_m5C@+0: m5C at center
    near_c = ctx["near_meth_by_m5C_at_+0"]
    m5c_col = near_c["meth_ids"].index(m5c)
    assert abs(near_c["fractions"][KMER_PRED_IDX, m5c_col] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Sharded refine + dataset + split helpers
# ---------------------------------------------------------------------------


def _write_shard(path: Path, data: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


# Tests below import kinsim.data.dataset which depends on torch. On
# environments without torch (Windows dev box, minimal CI), skip cleanly.
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_shard_sample_id_recovers_id_from_filename():
    if not HAS_TORCH:
        print("[skip] torch not installed — skipping shard helper tests")
        return
    from kinsim.data.dataset import shard_sample_id

    assert shard_sample_id("/data/run/shards/bc2034_shard.pkl") == "bc2034"
    assert shard_sample_id("/data/run/shards/bc2034_shard_clean.pkl") == "bc2034"
    assert shard_sample_id("plain_name.pkl") == "plain_name"


def test_split_shards_by_explicit_strain_list():
    if not HAS_TORCH:
        return
    from kinsim.data.dataset import split_shards

    paths = [
        "/x/bc2034_shard.pkl",
        "/x/bc2045_shard.pkl",
        "/x/bc2080_shard.pkl",
    ]
    train, test = split_shards(paths, test_strains=["bc2080"])
    assert len(train) == 2
    assert len(test) == 1
    assert "bc2080" in test[0]


def test_split_shards_by_random_fraction_is_reproducible():
    if not HAS_TORCH:
        return
    from kinsim.data.dataset import split_shards

    paths = [f"/x/bc{i:04d}_shard.pkl" for i in range(20)]
    train_a, test_a = split_shards(paths, test_fraction=0.2, seed=42)
    train_b, test_b = split_shards(paths, test_fraction=0.2, seed=42)
    assert train_a == train_b
    assert test_a == test_b
    assert len(test_a) == 4  # round(20 * 0.2)


def test_split_shards_unknown_test_strain_raises():
    if not HAS_TORCH:
        return
    from kinsim.data.dataset import split_shards

    paths = ["/x/bc2034_shard.pkl"]
    try:
        split_shards(paths, test_strains=["does_not_exist"])
    except ValueError as e:
        assert "no matching shard" in str(e)
    else:
        raise AssertionError("expected ValueError on missing test strain")


def test_sharded_signal_dataset_iterates_all_rows():
    """Two synthetic shards → ShardedSignalDataset yields every row exactly once
    (per epoch). Yields are (kmer_id, meth_full, log_signal, meth_id) tuples."""
    if not HAS_TORCH:
        return
    from kinsim.data.dataset import ShardedSignalDataset

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Shard A: 5 baseline rows (kmer_id=0)
        a = np.zeros((5, SAMPLE_NCOLS), dtype=np.float32)
        a[:, COL_IPD] = 30.0
        a[:, COL_PW] = 15.0
        _write_shard(td / "a_shard.pkl", {0: a})
        # Shard B: 3 slowed rows (kmer_id=1)
        b = np.zeros((3, SAMPLE_NCOLS), dtype=np.float32)
        b[:, COL_IPD] = 100.0
        b[:, COL_PW] = 50.0
        b[:, COL_CATEGORY] = CATEGORY_SLOWED
        b[:, COL_PARENT_METH] = 1
        _write_shard(td / "b_shard.pkl", {1: b})

        ds = ShardedSignalDataset(
            [str(td / "a_shard.pkl"), str(td / "b_shard.pkl")],
            shuffle=False,
        )
        items = list(ds)
        assert len(items) == 8  # 5 + 3
        # Spot-check first emit shape: kmer_id (Long), meth_full (K, 4),
        # log_signal (2,), meth_id (Long).
        kmer_id, meth_full, log_signal, meth_id = items[0]
        import torch as _torch

        assert isinstance(kmer_id, _torch.Tensor) and kmer_id.dtype == _torch.long
        assert meth_full.shape == (11, 4)
        assert log_signal.shape == (2,)
        assert isinstance(meth_id, _torch.Tensor)


def test_sharded_refine_writes_per_shard_clean_pkls():
    """slowed_split_gmm_shards over 2 synthetic shards produces 2 cleaned shards
    in the output dir, with __meta__["method"] = 'gmm_per_meth_offset'."""
    from kinsim.refine import slowed_split_gmm_shards

    meth_ids = get_meth_ids()
    m6a = meth_ids["m6A"]

    def _build_strain_shard(n_kmers: int, seed: int) -> dict:
        local = np.random.default_rng(seed)
        out = {}
        for kid in range(n_kmers):
            rows = []
            # Baseline: μ=30
            for _ in range(50):
                r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
                r[COL_IPD] = float(max(0.0, local.normal(30.0, 4.0)))
                r[COL_PW] = float(max(0.0, 0.5 * r[COL_IPD] + local.normal(0, 1.5)))
                r[COL_CATEGORY] = CATEGORY_BASELINE
                rows.append(r)
            # Slowed-by-m6A: μ=100, with 30 % contamination at μ=30
            for _ in range(40):
                r = np.zeros(SAMPLE_NCOLS, dtype=np.float32)
                ipd = local.normal(30.0, 4.0) if local.random() < 0.3 else local.normal(100.0, 8.0)
                r[COL_IPD] = float(max(0.0, ipd))
                r[COL_PW] = float(max(0.0, 0.5 * r[COL_IPD] + local.normal(0, 1.5)))
                r[COL_CATEGORY] = CATEGORY_SLOWED
                r[COL_PARENT_METH] = m6a
                r[COL_PARENT_OFFSET] = 0
                rows.append(r)
            out[kid] = np.stack(rows)
        return out

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shards_dir = td / "shards"
        out_dir = td / "refined"
        shards_dir.mkdir()

        # Two strain shards, 30 kmers each.
        _write_shard(shards_dir / "bc2034_shard.pkl", _build_strain_shard(30, seed=1))
        _write_shard(shards_dir / "bc2045_shard.pkl", _build_strain_shard(30, seed=2))

        stats = slowed_split_gmm_shards(shards_dir, out_dir, n_components=(2, 3), seed=42)

        # 2 cleaned shards on disk.
        cleaned = sorted((out_dir).glob("*_clean.pkl"))
        assert len(cleaned) == 2
        # Method recorded.
        assert stats["method"] == "gmm_per_meth_offset"
        assert stats["n_shards"] == 2
        # Each cleaned shard has __meta__ with refine info.
        for p in cleaned:
            with open(p, "rb") as f:
                d = pickle.load(f)
            assert "__meta__" in d
            assert d["__meta__"]["method"] == "gmm_per_meth_offset"


def test_concat_shards_in_analyze_merges_per_kmer_arrays():
    """_concat_shards stacks rows of the same kmer across shards."""
    from kinsim.analyze import _concat_shards

    a = np.zeros((3, SAMPLE_NCOLS), dtype=np.float32)
    a[:, COL_IPD] = 30.0
    b = np.zeros((4, SAMPLE_NCOLS), dtype=np.float32)
    b[:, COL_IPD] = 35.0
    c = np.zeros((2, SAMPLE_NCOLS), dtype=np.float32)
    c[:, COL_IPD] = 90.0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Shard 1 has kmer 0; shard 2 has kmer 0 AND kmer 1.
        _write_shard(td / "x_shard.pkl", {0: a})
        _write_shard(td / "y_shard.pkl", {0: b, 1: c})
        merged, _ = _concat_shards(td)
        assert merged[0].shape[0] == 7  # 3 + 4
        assert merged[1].shape[0] == 2


if __name__ == "__main__":
    test_category_constants_are_three()
    print("[pass] category constants")
    test_get_categories_reads_col35()
    print("[pass] get_categories")
    test_layout_column_contract()
    print("[pass] layout column contract (38 cols, 35 cat, 36 parent_meth, 37 parent_off)")
    test_slowed_split_only_filters_slowed()
    print("[pass] slowed_split (p95)")
    test_refine_pkl_writes_output_p95()
    print("[pass] refine_pkl (p95)")
    test_refine_fails_fast_on_obsolete_layout()
    print("[pass] refine fails fast on obsolete layout")
    test_gmm_k2_separates_clean_two_distributions()
    print("[pass] GMM K=2 separates two clean distributions")
    test_gmm_k3_handles_long_tail_baseline()
    print("[pass] GMM K=3 handles long-tail baseline (PacBio-realistic)")
    test_gmm_validation_fails_when_meth_indistinguishable_from_baseline()
    print("[pass] GMM defensive when meth indistinguishable from baseline")
    test_gmm_bic_picks_k2_for_unimodal_baseline()
    print("[pass] GMM BIC picks K=2 for unimodal baseline")
    test_gmm_bic_picks_k3_for_long_tail_baseline()
    print("[pass] GMM BIC picks K=3 for long-tail baseline")
    test_gmm_too_few_slowed_keeps_all()
    print("[pass] GMM too-few-slowed → keeps all")
    test_gmm_per_offset_isolates_noisy_offset_from_clean_offset()
    print("[pass] GMM per-(meth, offset) isolates noisy offsets from clean ones")
    test_refine_pkl_writes_output_gmm()
    print("[pass] refine_pkl (gmm)")
    test_compute_signature_profiles()
    print("[pass] compute_signature_profiles")
    test_analyze_uses_parent_meth_column_not_meth_context()
    print("[pass] analyze trusts COL_PARENT_METH over mc[]")
    test_compute_meth_context_distribution()
    print("[pass] compute_meth_context_distribution")

    # ── Sharded paths ─────────────────────────────────────────────────
    test_shard_sample_id_recovers_id_from_filename()
    print("[pass] shard_sample_id recovers ID from filename")
    test_split_shards_by_explicit_strain_list()
    print("[pass] split_shards by explicit --test-strains list")
    test_split_shards_by_random_fraction_is_reproducible()
    print("[pass] split_shards by random fraction is reproducible")
    test_split_shards_unknown_test_strain_raises()
    print("[pass] split_shards raises on unknown test strain")
    test_sharded_signal_dataset_iterates_all_rows()
    print("[pass] ShardedSignalDataset iterates all rows from all shards")
    test_sharded_refine_writes_per_shard_clean_pkls()
    print("[pass] sharded refine writes per-shard *_clean.pkl files")
    test_concat_shards_in_analyze_merges_per_kmer_arrays()
    print("[pass] _concat_shards in analyze merges per-kmer arrays")
    print("\nAll tests passed.")
