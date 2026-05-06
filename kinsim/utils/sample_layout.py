"""Per-sample column layout shared by extract / refine / dataset / generate.

Storage format: ``dict[kmer_id (int) -> np.ndarray(N, 20)]`` plus an
optional ``"__meta__"`` key with provenance.

Column reference
----------------

    Cols  | Contents
    ------+------------------------------------------------------------
    0     | IPD at prediction position (raw uint8 written as float32)
    1     | PW  at prediction position
    2     | stoichiometric fraction (0..1) — per-site motif occupancy
    3-13  | mc_0..mc_10  — meth_id at offsets [-7..+3] from prediction
    14-16 | rev_meth_-1, rev_meth_0, rev_meth_+1 — complementary-strand
            meth_id at active-site neighbours (FiLM consumes these)
    17    | CATEGORY — three values:
              0 = baseline   far from any methylation
              1 = slowed     at a signature offset of a methylation
              2 = near_meth  close to a methylation but NOT at a
                             signature offset (negative control)
    18    | PARENT_METH — meth_id of the canonical methylation that
              produced this slowed/near assignment (0 for baseline).
    19    | PARENT_OFFSET — row_pos − parent_meth_pos, in the polymerase
              frame. Used by refine for per-(meth, offset) GMM bucketing.

The methylation centres themselves land in SLOWED if 0 is in their
signature offsets (m6A, m4C) or NEAR_METH otherwise (m5C, sig [+2, +6]).

Helpers in this module are pure Python — extract.py needs pysam, but
refine / dataset / tests / scripts must be importable without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .encoding import KMER_LEFT_PAD, KMER_RIGHT_PAD, K

if TYPE_CHECKING:
    import numpy as np

# Methylation-context window: [-METH_CTX_LEFT, +METH_CTX_RIGHT] = 11 positions.
# Same asymmetric window as the kmer (KMER_LEFT_PAD=7, KMER_RIGHT_PAD=3).
METH_CTX_LEFT = KMER_LEFT_PAD  # = 7
METH_CTX_RIGHT = KMER_RIGHT_PAD  # = 3
METH_CTX_LEN = K  # = 11

# Complementary-strand meth_id captured at the prediction position and its
# immediate neighbours (active-site footprint). Used by FiLM to disambiguate
# bilateral methylation (palindromic R-M Type II sites).
REV_METH_OFFSETS = (-1, 0, 1)
REV_METH_LEN = len(REV_METH_OFFSETS)  # = 3

SAMPLE_NCOLS = 3 + METH_CTX_LEN + REV_METH_LEN + 3  # = 20

COL_IPD = 0
COL_PW = 1
COL_FRACTION = 2
COL_METH_CTX_START = 3
COL_METH_CTX_END = COL_METH_CTX_START + METH_CTX_LEN  # 14
COL_REV_METH = COL_METH_CTX_END  # 14
COL_CATEGORY = COL_REV_METH + REV_METH_LEN  # 17
COL_PARENT_METH = COL_CATEGORY + 1  # 18
COL_PARENT_OFFSET = COL_PARENT_METH + 1  # 19

# Category enum values written to col 17.
CATEGORY_BASELINE = 0
CATEGORY_SLOWED = 1
CATEGORY_NEAR_METH = 2
CATEGORY_NAMES = {
    CATEGORY_BASELINE: "baseline",
    CATEGORY_SLOWED: "slowed",
    CATEGORY_NEAR_METH: "near_meth",
}


def slice_meth_context(meth_status, center: int) -> list:
    """Return an 11-element list covering [-7, +3] around ``center``.

    Out-of-range positions are padded with 0 so every sample has the
    same fixed-length context.
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
    [-1, 0, +1] from ``center``.
    """
    n = len(meth_status_complement)
    out = [0] * REV_METH_LEN
    for k, off in enumerate(REV_METH_OFFSETS):
        pos = center + off
        if 0 <= pos < n:
            out[k] = int(meth_status_complement[pos])
    return out


def get_categories(arr) -> np.ndarray:
    """Return per-sample category as int8 ndarray of length N."""
    import numpy as np

    if len(arr) == 0:
        return np.empty(0, dtype=np.int8)
    return arr[:, COL_CATEGORY].astype(np.int8)
