"""Per-sample column layout shared by extract / refine / dataset / generate.

Each stored row is `np.float32` with the following 36 columns (v4 format):

    Cols  | Contents
    ------+------------------------------------------------------------
    0     | IPD at prediction position
    1     | PW  at prediction position
    2     | stoichiometric fraction (0..1)
    3-13  | mc_0..mc_10 — meth_id at offsets [-7..+3] from prediction pos
    14-22 | profile_IPD_0..+8 — kinetic profile downstream
    23-31 | profile_PW_0..+8
    32-34 | rev_meth_-1, rev_meth_0, rev_meth_+1 — complementary-strand
            meth_id at active-site neighbours
    35    | category (v4) — three values only:
              0 = baseline   (far from any methylation; meth_context is empty)
              1 = slowed     (the methylation itself OR a downstream signature
                              offset of an upstream methylation. IPD elevation
                              is biophysically expected here.
                              ex. m6A at p has signature [0, 5] -> p and p+5
                                  are both SLOWED; the m6A position itself is
                                  classified by its OWN signature offset 0.)
              2 = near_meth  (close to a methylation but NOT at a signature
                              offset of it. IPD should look baseline-like.
                              Negative control teaching the model that meth-
                              in-context alone does NOT mean elevated IPD.
                              Window: [+1, +near_meth_max_dist] from the meth,
                              minus signature offsets.)
            (No separate "meth" category — methylation positions land in
             SLOWED if 0 ∈ signature_offsets[T] (m6A, m4C) or NEAR_METH
             otherwise (m5C, since signature is [+2, +6]).)

v3 format (pre-2026-05) used 35 columns. Loading code that may encounter
v3 pkls should treat samples with `arr.shape[1] == 35` as having an
implicit category derivable from `mc[7]` and signature offsets.

The v4 storage uses a `dict[kmer_id] -> ndarray(N, 36)` keying instead
of v3's `dict[(kmer_id, meth_id)] -> ndarray(N, 35)`. The meth_id at
center is already encoded in `mc[7]` (col 10), and the category column
distinguishes baseline from slowed when `mc[7]==0`. The (kmer, meth_id)
key was redundant.

The slicing helpers are pure Python so they can be unit-tested without
pulling in pysam (which extract.py depends on for BAM I/O).
"""

from __future__ import annotations

from .encoding import K, KMER_LEFT_PAD, KMER_PRED_IDX, KMER_RIGHT_PAD

# Methylation-context window: [-METH_CTX_LEFT, +METH_CTX_RIGHT] = 11 positions.
# Same asymmetric window as the kmer (KMER_LEFT_PAD=7, KMER_RIGHT_PAD=3).
METH_CTX_LEFT  = KMER_LEFT_PAD     # = 7
METH_CTX_RIGHT = KMER_RIGHT_PAD    # = 3
METH_CTX_LEN   = K                 # = 11

# Kinetic profile downstream of the prediction position. Used by refine to
# validate that the type-specific signature pattern (e.g. m5C at +2 and +6)
# is actually present in the sample's neighbourhood.
PROFILE_START = 0
PROFILE_END   = 8
PROFILE_LEN   = PROFILE_END - PROFILE_START + 1   # = 9

# Complementary-strand meth_id captured at the prediction position and its
# immediate neighbours (active-site footprint). Used to disambiguate
# bilateral methylation (palindromic R-M Type II sites).
REV_METH_OFFSETS = (-1, 0, 1)
REV_METH_LEN = len(REV_METH_OFFSETS)              # = 3

# Total per-sample column count: 35 v3-cols + 1 category col = 36 (v4).
# Code that loads pkls should accept both 35 (v3) and 36 (v4) and route
# accordingly via `infer_category()`.
SAMPLE_NCOLS_V3 = 3 + METH_CTX_LEN + 2 * PROFILE_LEN + REV_METH_LEN  # = 35
SAMPLE_NCOLS    = SAMPLE_NCOLS_V3 + 1                                # = 36 (v4)

# Convenience column-range constants — useful for refine / dataset code.
COL_IPD            = 0
COL_PW             = 1
COL_FRACTION       = 2
COL_METH_CTX_START = 3
COL_METH_CTX_END   = COL_METH_CTX_START + METH_CTX_LEN              # 14
COL_PROFILE_IPD    = COL_METH_CTX_END                                # 14
COL_PROFILE_PW     = COL_PROFILE_IPD + PROFILE_LEN                   # 23
COL_REV_METH       = COL_PROFILE_PW + PROFILE_LEN                    # 32
COL_CATEGORY       = COL_REV_METH + REV_METH_LEN                     # 35

# Category enum values for col 35.
CATEGORY_BASELINE  = 0   # far from any methylation, meth_context empty
CATEGORY_SLOWED    = 1   # at a signature offset of a meth (the meth itself
                         # included if 0 ∈ signature_offsets[T]). IPD elevated.
CATEGORY_NEAR_METH = 2   # close to a meth but not at a signature offset.
                         # IPD baseline-like. Negative control.
CATEGORY_NAMES = {
    CATEGORY_BASELINE:  "baseline",
    CATEGORY_SLOWED:    "slowed",
    CATEGORY_NEAR_METH: "near_meth",
}


def slice_meth_context(meth_status, center: int) -> list:
    """Return an 11-element list covering [-7, +3] around `center`.

    Out-of-range positions (start of read / end of read) are padded with 0
    (unmethylated) so every sample has the same fixed-length context.
    """
    n = len(meth_status)
    out = [0] * METH_CTX_LEN
    for k in range(METH_CTX_LEN):
        pos = center - METH_CTX_LEFT + k
        if 0 <= pos < n:
            out[k] = int(meth_status[pos])
    return out


def slice_rev_meth(meth_status_complement, center: int) -> list:
    """Return a list of 3 values: complementary-strand meth_id at offsets
    [-1, 0, +1] from `center`.

    `meth_status_complement` must already be expressed in this read's
    coordinates (reverse-mapped from rc_seq if needed). Out-of-range
    positions are padded with 0.
    """
    n = len(meth_status_complement)
    out = [0] * REV_METH_LEN
    for k, off in enumerate(REV_METH_OFFSETS):
        pos = center + off
        if 0 <= pos < n:
            out[k] = int(meth_status_complement[pos])
    return out


def slice_kinetic_profile(ipds, pws, center: int) -> list:
    """Return a list of 18 values: [profile_IPD_0..+8, profile_PW_0..+8].

    Out-of-range positions (past the end of the read) are padded with 0.
    """
    n_ipd = len(ipds)
    n_pw  = len(pws)
    ipd_prof = [0.0] * PROFILE_LEN
    pw_prof  = [0.0] * PROFILE_LEN
    for k in range(PROFILE_LEN):
        pos = center + PROFILE_START + k
        if 0 <= pos < n_ipd:
            ipd_prof[k] = float(ipds[pos])
        if 0 <= pos < n_pw:
            pw_prof[k]  = float(pws[pos])
    return ipd_prof + pw_prof


# ---------------------------------------------------------------------------
# Category inference (v3 fallback)
# ---------------------------------------------------------------------------

def is_v4_format(arr) -> bool:
    """Return True iff the array has the v4 36-column layout including
    an explicit category column. False for v3 35-col arrays."""
    import numpy as np
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return False
    return arr.shape[1] >= SAMPLE_NCOLS  # >= 36


def get_categories(arr, signature_offsets_by_meth: dict | None = None,
                    pred_idx: int | None = None,
                    near_meth_max_dist: int | None = None) -> "np.ndarray":
    """Return per-sample category as int8 ndarray of length N.

    For v4 arrays (36 cols), reads col `COL_CATEGORY` directly.
    For v3 arrays (35 cols), infers category from meth_context:

      The methylation at center (mc[pred_idx] > 0) is itself classified as
      SLOWED if its own offset 0 is in its signature_offsets, else NEAR_METH.
      Then any non-meth center sample is checked for upstream methylations:
      - upstream meth at a signature offset of it → CATEGORY_SLOWED
      - upstream meth in proximity window but NOT signature → CATEGORY_NEAR_METH
      - else → CATEGORY_BASELINE

    `signature_offsets_by_meth` maps meth_name -> list of int offsets.
    `near_meth_max_dist` defaults to METH_CTX_LEFT (= 7) — the meth_context
    only sees up to 7 bases upstream so that's the natural proximity window.
    """
    import numpy as np
    if pred_idx is None:
        pred_idx = METH_CTX_LEFT
    if near_meth_max_dist is None:
        near_meth_max_dist = METH_CTX_LEFT

    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    if is_v4_format(arr):
        return arr[:, COL_CATEGORY].astype(np.int8)

    # ---- v3 inference path ----
    cats = np.zeros(n, dtype=np.int8)
    mc = arr[:, COL_METH_CTX_START:COL_METH_CTX_END].astype(np.int32)
    center = mc[:, pred_idx]

    if not signature_offsets_by_meth:
        # Without signature info we cannot classify — meth at center counts
        # as slowed by default, everything else stays baseline.
        cats[center > 0] = CATEGORY_SLOWED
        return cats

    from .encoding import get_meth_ids
    meth_ids = get_meth_ids()

    # 1. Classify the meth center itself: SLOWED if 0 ∈ sig, else NEAR_METH.
    for mname, offsets in signature_offsets_by_meth.items():
        mid = meth_ids.get(mname)
        if not mid:
            continue
        sig_set = {int(k) for k in offsets if isinstance(k, (int, float))}
        target_cat = CATEGORY_SLOWED if 0 in sig_set else CATEGORY_NEAR_METH
        cats[(center == mid)] = target_cat

    # 2. For non-meth center samples, check upstream meth at sig vs proximity offsets.
    sig_probes  = []
    near_probes = []
    for mname, offsets in signature_offsets_by_meth.items():
        mid = meth_ids.get(mname)
        if not mid:
            continue
        sig_set = {int(k) for k in offsets if isinstance(k, (int, float))}
        for k in range(1, near_meth_max_dist + 1):
            idx = pred_idx - k
            if not (0 <= idx < METH_CTX_LEN):
                continue
            if k in sig_set:
                sig_probes.append((idx, mid))
            else:
                near_probes.append((idx, mid))

    base_mask = (cats == CATEGORY_BASELINE)   # only the truly-empty baselines
    if sig_probes:
        hit = np.zeros(n, dtype=bool)
        for mc_idx, mid in sig_probes:
            hit |= (mc[:, mc_idx] == mid)
        cats[base_mask & hit] = CATEGORY_SLOWED
    if near_probes:
        still_base = (cats == CATEGORY_BASELINE)
        hit = np.zeros(n, dtype=bool)
        for mc_idx, mid in near_probes:
            hit |= (mc[:, mc_idx] == mid)
        cats[still_base & hit] = CATEGORY_NEAR_METH

    return cats
