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
    35    | category (v4)
              0 = baseline   (mc[7]==0 AND no upstream signature meth in window)
              1 = meth       (mc[7]>0  — methylation at the prediction position)
              2 = slowed     (mc[7]==0 AND upstream confirmed meth at sig offset)

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
CATEGORY_BASELINE = 0   # truly unmethylated, no upstream signature meth
CATEGORY_METH     = 1   # methylated at center (mc[7] > 0)
CATEGORY_SLOWED   = 2   # unmethylated at center, upstream confirmed meth at sig offset
CATEGORY_NAMES = {
    CATEGORY_BASELINE: "baseline",
    CATEGORY_METH:     "meth",
    CATEGORY_SLOWED:   "slowed",
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
                    pred_idx: int | None = None) -> "np.ndarray":
    """Return per-sample category as int8 ndarray of length N.

    For v4 arrays (36 cols), reads col `COL_CATEGORY` directly.
    For v3 arrays (35 cols), infers category from meth_context:
      - mc[pred_idx] > 0 → CATEGORY_METH
      - mc[pred_idx] == 0 AND any signature offset upstream matches → CATEGORY_SLOWED
      - else → CATEGORY_BASELINE

    `signature_offsets_by_meth` maps meth_name -> list of int offsets.
    Required only for v3 inference; ignored for v4. If None for v3,
    every non-meth sample is classified as CATEGORY_BASELINE.

    `pred_idx` defaults to METH_CTX_LEFT (= 7).
    """
    import numpy as np
    if pred_idx is None:
        pred_idx = METH_CTX_LEFT

    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    if is_v4_format(arr):
        return arr[:, COL_CATEGORY].astype(np.int8)

    # ---- v3 inference path ----
    cats = np.zeros(n, dtype=np.int8)
    mc = arr[:, COL_METH_CTX_START:COL_METH_CTX_END].astype(np.int32)
    center = mc[:, pred_idx]
    cats[center > 0] = CATEGORY_METH

    if signature_offsets_by_meth:
        # Build (mc_idx, expected_meth_id) probes for each upstream sig offset.
        # Late import to keep this module pysam-free.
        from .encoding import get_meth_ids
        meth_ids = get_meth_ids()
        probes = []
        for mname, offsets in signature_offsets_by_meth.items():
            mid = meth_ids.get(mname)
            if not mid:
                continue
            for k in offsets:
                try:
                    k = int(k)
                except (TypeError, ValueError):
                    continue
                if k <= 0:
                    continue
                idx = pred_idx - k
                if 0 <= idx < METH_CTX_LEN:
                    probes.append((idx, mid))
        if probes:
            slowed_mask = (cats == CATEGORY_BASELINE)  # only check non-meth
            hit = np.zeros(n, dtype=bool)
            for mc_idx, mid in probes:
                hit |= (mc[:, mc_idx] == mid)
            cats[slowed_mask & hit] = CATEGORY_SLOWED

    return cats
