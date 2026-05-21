"""Per-sample column layout — bilateral v2.

Each row stores both strands' kinetics and both strands' methylation
contexts derived from one raw HiFi aligned read at one genomic position.
The model predicts all four kinetic channels jointly.

Layout (K=11 → 33 cols; generic: ``n_cols = 11 + 2*K``)::

    0     | ipd_fwd  — kinetics for + ref strand methylations
    1     | pw_fwd
    2     | ipd_rev  — kinetics for - ref strand methylations
    3     | pw_rev
    4     | fraction (stoichiometry at the active site)
    5..K+4| mc_fwd  — + strand meth context, K positions
    K+5.. | mc_rev  — - strand meth context, K positions
    ...   | CATEGORY_FWD, PARENT_METH_FWD, PARENT_OFFSET_FWD
    ...   | CATEGORY_REV, PARENT_METH_REV, PARENT_OFFSET_REV

Kinetics are normalised at extract: ``ipd_fwd`` always means "kinetics
observed when polymerase read + strand template". Raw tags ``fi``/``ri``
are routed by ``read.is_reverse``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ExtractionParams, get_extraction_params

CATEGORY_BASELINE = 0
CATEGORY_SLOWED = 1
CATEGORY_NEAR_METH = 2
CATEGORY_NAMES = {
    CATEGORY_BASELINE: "baseline",
    CATEGORY_SLOWED: "slowed",
    CATEGORY_NEAR_METH: "near_meth",
}

METH_CTX_LEN = 11
SAMPLE_NCOLS = 33

COL_IPD_FWD = 0
COL_PW_FWD = 1
COL_IPD_REV = 2
COL_PW_REV = 3
COL_FRACTION = 4
COL_METH_CTX_FWD_START = 5
COL_METH_CTX_FWD_END = COL_METH_CTX_FWD_START + METH_CTX_LEN
COL_METH_CTX_REV_START = COL_METH_CTX_FWD_END
COL_METH_CTX_REV_END = COL_METH_CTX_REV_START + METH_CTX_LEN
COL_CATEGORY_FWD = COL_METH_CTX_REV_END
COL_PARENT_METH_FWD = COL_CATEGORY_FWD + 1
COL_PARENT_OFFSET_FWD = COL_CATEGORY_FWD + 2
COL_CATEGORY_REV = COL_PARENT_OFFSET_FWD + 1
COL_PARENT_METH_REV = COL_CATEGORY_REV + 1
COL_PARENT_OFFSET_REV = COL_CATEGORY_REV + 2


@dataclass(frozen=True)
class SampleLayout:
    """Bilateral column indices derived from :class:`ExtractionParams`."""

    params: ExtractionParams
    n_cols: int
    col_ipd_fwd: int
    col_pw_fwd: int
    col_ipd_rev: int
    col_pw_rev: int
    col_fraction: int
    col_meth_ctx_fwd_start: int
    col_meth_ctx_fwd_end: int
    col_meth_ctx_rev_start: int
    col_meth_ctx_rev_end: int
    col_category_fwd: int
    col_parent_meth_fwd: int
    col_parent_offset_fwd: int
    col_category_rev: int
    col_parent_meth_rev: int
    col_parent_offset_rev: int

    @classmethod
    def from_params(cls, params: ExtractionParams) -> SampleLayout:
        K = params.kmer_size
        c_mc_fwd_s = 5
        c_mc_fwd_e = c_mc_fwd_s + K
        c_mc_rev_s = c_mc_fwd_e
        c_mc_rev_e = c_mc_rev_s + K
        c_cat = c_mc_rev_e
        return cls(
            params=params,
            n_cols=params.sample_ncols,
            col_ipd_fwd=0,
            col_pw_fwd=1,
            col_ipd_rev=2,
            col_pw_rev=3,
            col_fraction=4,
            col_meth_ctx_fwd_start=c_mc_fwd_s,
            col_meth_ctx_fwd_end=c_mc_fwd_e,
            col_meth_ctx_rev_start=c_mc_rev_s,
            col_meth_ctx_rev_end=c_mc_rev_e,
            col_category_fwd=c_cat,
            col_parent_meth_fwd=c_cat + 1,
            col_parent_offset_fwd=c_cat + 2,
            col_category_rev=c_cat + 3,
            col_parent_meth_rev=c_cat + 4,
            col_parent_offset_rev=c_cat + 5,
        )

    @property
    def kmer_size(self) -> int:
        return self.params.kmer_size

    @property
    def upstream(self) -> int:
        return self.params.upstream

    @property
    def active_site_index(self) -> int:
        return self.params.active_site_index


def get_sample_layout(params: ExtractionParams | None = None) -> SampleLayout:
    """Return :class:`SampleLayout` for the given (or current YAML) params."""
    if params is None:
        params = get_extraction_params()
    return SampleLayout.from_params(params)


def get_categories_fwd(arr, layout: SampleLayout | None = None):
    """Forward-strand category per sample (int8)."""
    import numpy as np
    if len(arr) == 0:
        return np.empty(0, dtype=np.int8)
    if layout is None:
        layout = get_sample_layout()
    return arr[:, layout.col_category_fwd].astype(np.int8)


def get_categories_rev(arr, layout: SampleLayout | None = None):
    """Reverse-strand category per sample (int8)."""
    import numpy as np
    if len(arr) == 0:
        return np.empty(0, dtype=np.int8)
    if layout is None:
        layout = get_sample_layout()
    return arr[:, layout.col_category_rev].astype(np.int8)
