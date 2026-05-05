"""Per-sample column layout shared by extract / refine / dataset / generate.

Storage format: ``dict[kmer_id (int)] -> np.ndarray(N, 37)`` plus an
optional ``"__meta__"`` key with provenance information.

Column reference:

    Cols  | Contents
    ------+------------------------------------------------------------
    0     | IPD at prediction position (raw uint8, stored as float32)
    1     | PW  at prediction position
    2     | stoichiometric fraction (0..1)
    3-13  | mc_0..mc_10  — meth_id at offsets [-7..+3] from prediction
    14-22 | profile_IPD_0..+8 — kinetic profile downstream
    23-31 | profile_PW_0..+8
    32-34 | rev_meth_-1, rev_meth_0, rev_meth_+1 — complementary-strand
            meth_id at active-site neighbours
    35    | CATEGORY — three values:
              0 = baseline   far from any methylation
              1 = slowed     at a signature offset of a methylation;
                             IPD elevation biophysically expected here
              2 = near_meth  close to a methylation but NOT at a
                             signature offset; IPD should look
                             baseline-like — used as negative control
    36    | PARENT_METH — meth_id of the methylation that produced this
              slowed/near assignment (0 for baseline; 1=m6A, 2=m4C,
              3=m5C — same enum as METH_IDS). Written by extract at
              emission time so analyze can group rows per parent meth
              vectorially without re-inferring from the meth_context.

The methylation centers themselves land in SLOWED if 0 is in their
signature offsets (m6A, m4C) or NEAR_METH otherwise (m5C, sig [+2, +6]).

Helpers in this module are pure Python so they remain importable
without pysam (which extract.py needs for BAM I/O).
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

# Kinetic profile downstream of the prediction position. Used by refine to
# validate that the type-specific signature pattern (e.g. m5C at +2 and +6)
# is actually present in the sample's neighbourhood.
PROFILE_START = 0
PROFILE_END = 8
PROFILE_LEN = PROFILE_END - PROFILE_START + 1  # = 9

# Complementary-strand meth_id captured at the prediction position and its
# immediate neighbours (active-site footprint). Used to disambiguate
# bilateral methylation (palindromic R-M Type II sites).
REV_METH_OFFSETS = (-1, 0, 1)
REV_METH_LEN = len(REV_METH_OFFSETS)  # = 3

SAMPLE_NCOLS = 3 + METH_CTX_LEN + 2 * PROFILE_LEN + REV_METH_LEN + 3  # = 38

# Column-range constants — used by refine / dataset / analyze.
COL_IPD = 0
COL_PW = 1
COL_FRACTION = 2
COL_METH_CTX_START = 3
COL_METH_CTX_END = COL_METH_CTX_START + METH_CTX_LEN  # 14
COL_PROFILE_IPD = COL_METH_CTX_END  # 14
COL_PROFILE_PW = COL_PROFILE_IPD + PROFILE_LEN  # 23
COL_REV_METH = COL_PROFILE_PW + PROFILE_LEN  # 32
COL_CATEGORY = COL_REV_METH + REV_METH_LEN  # 35
COL_PARENT_METH = COL_CATEGORY + 1  # 36 — meth_id of the parent meth (0 for baseline)
COL_PARENT_OFFSET = COL_PARENT_METH + 1  # 37 — offset (this row pos − parent meth pos)

# Category enum values written to col 35.
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

    ``meth_status_complement`` must already be expressed in this read's
    coordinates (reverse-mapped from rc_seq if needed).
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

    Out-of-range positions are padded with 0.
    """
    n_ipd = len(ipds)
    n_pw = len(pws)
    ipd_prof = [0.0] * PROFILE_LEN
    pw_prof = [0.0] * PROFILE_LEN
    for k in range(PROFILE_LEN):
        pos = center + PROFILE_START + k
        if 0 <= pos < n_ipd:
            ipd_prof[k] = float(ipds[pos])
        if 0 <= pos < n_pw:
            pw_prof[k] = float(pws[pos])
    return ipd_prof + pw_prof


def get_categories(arr) -> np.ndarray:
    """Return per-sample category as int8 ndarray of length N (reads col COL_CATEGORY)."""
    import numpy as np

    if len(arr) == 0:
        return np.empty(0, dtype=np.int8)
    return arr[:, COL_CATEGORY].astype(np.int8)
