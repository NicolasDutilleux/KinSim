"""Per-sample column layout shared by extract / refine / dataset / generate.

Storage format
--------------

    dict[kmer_id (int) -> np.ndarray(N, SAMPLE_NCOLS)]

plus an optional ``"__meta__"`` key carrying provenance — including the
:class:`~kinsim.utils.config.ExtractionParams` the shard was built with,
serialised under ``__meta__["extraction_params"]``.

Column reference (legacy K=11 layout — the module-level ``COL_*`` constants
below resolve to these indices; for non-default geometries use
:class:`SampleLayout`)::

    Cols  | Contents
    ------+--------------------------------------------------------------
    0     | IPD at prediction position (raw uint8 written as float32)
    1     | PW  at prediction position
    2     | stoichiometric fraction (0..1)
    3..13 | mc_0..mc_10 (meth_id at offsets [-7, +3])
    14..16| rev_meth at offsets [-1, 0, +1]
    17    | CATEGORY (0=baseline, 1=slowed, 2=near_meth)
    18    | PARENT_METH
    19    | PARENT_OFFSET

Naming convention
-----------------

* ``UPPERCASE`` (``COL_IPD``, ``SAMPLE_NCOLS``, ``CATEGORY_BASELINE``):
  module-level integer constants for the **default K=11 layout**.
  Static — readable in tests / call sites that hard-target K=11.
* ``lowercase`` dataclass attributes (``layout.col_ipd``, ``layout.n_cols``):
  K-aware values from a :class:`SampleLayout` instance. Use these
  whenever the shard's geometry isn't statically known — i.e. read
  ``layout = get_sample_layout(read_shard_extraction_params(data))``
  then index ``arr[:, layout.col_category]``.

This module is pure Python (no pysam) so refine, dataset, tests, and
scripts can import it without the BAM dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ExtractionParams, get_extraction_params

# ---------------------------------------------------------------------------
# Category enum
# ---------------------------------------------------------------------------

CATEGORY_BASELINE = 0
CATEGORY_SLOWED = 1
CATEGORY_NEAR_METH = 2
CATEGORY_NAMES = {
    CATEGORY_BASELINE: "baseline",
    CATEGORY_SLOWED: "slowed",
    CATEGORY_NEAR_METH: "near_meth",
}


# ---------------------------------------------------------------------------
# Legacy K=11 layout — hardcoded for the historical defaults.
# New code should prefer :func:`get_sample_layout` so the layout adapts to
# ``kinsim_config.yaml``.
# ---------------------------------------------------------------------------

METH_CTX_LEFT = 7
METH_CTX_RIGHT = 3
METH_CTX_LEN = 11
REV_METH_OFFSETS = (-1, 0, 1)
REV_METH_LEN = 3

SAMPLE_NCOLS = 20
COL_IPD = 0
COL_PW = 1
COL_FRACTION = 2
COL_METH_CTX_START = 3
COL_METH_CTX_END = 14
COL_REV_METH = 14
COL_CATEGORY = 17
COL_PARENT_METH = 18
COL_PARENT_OFFSET = 19


# ---------------------------------------------------------------------------
# Dynamic SampleLayout — derives all column indices from ExtractionParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleLayout:
    """Column indices for the per-sample storage vector.

    All indices are derived from an :class:`ExtractionParams` so the layout
    automatically adapts when the user changes ``kmer_size`` or
    ``rev_meth_offsets`` in ``kinsim_config.yaml``.
    """

    params: ExtractionParams
    n_cols: int
    col_ipd: int
    col_pw: int
    col_fraction: int
    col_meth_ctx_start: int
    col_meth_ctx_end: int
    col_rev_meth: int
    col_category: int
    col_parent_meth: int
    col_parent_offset: int

    @classmethod
    def from_params(cls, params: ExtractionParams) -> SampleLayout:
        col_meth_ctx_start = 3
        col_meth_ctx_end = col_meth_ctx_start + params.kmer_size
        col_rev_meth = col_meth_ctx_end
        col_category = col_rev_meth + params.n_rev_meth
        return cls(
            params=params,
            n_cols=params.sample_ncols,
            col_ipd=0,
            col_pw=1,
            col_fraction=2,
            col_meth_ctx_start=col_meth_ctx_start,
            col_meth_ctx_end=col_meth_ctx_end,
            col_rev_meth=col_rev_meth,
            col_category=col_category,
            col_parent_meth=col_category + 1,
            col_parent_offset=col_category + 2,
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

    @property
    def n_rev_meth(self) -> int:
        return self.params.n_rev_meth

    @property
    def rev_meth_offsets(self) -> tuple[int, ...]:
        return self.params.rev_meth_offsets


def get_sample_layout(params: ExtractionParams | None = None) -> SampleLayout:
    """Return the :class:`SampleLayout` for the given (or current YAML) params."""
    if params is None:
        params = get_extraction_params()
    return SampleLayout.from_params(params)


def get_categories(arr, layout: SampleLayout | None = None):
    """Return per-sample category as int8 ndarray. K-aware via ``layout``
    (falls back to the YAML's current layout when omitted)."""
    import numpy as np

    if len(arr) == 0:
        return np.empty(0, dtype=np.int8)
    if layout is None:
        layout = get_sample_layout()
    return arr[:, layout.col_category].astype(np.int8)
