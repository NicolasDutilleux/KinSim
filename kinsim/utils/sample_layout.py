"""Per-sample column layout shared by extract / refine / dataset / generate.

Storage format
--------------

    dict[kmer_id (int) -> np.ndarray(N, SAMPLE_NCOLS)]

plus an optional ``"__meta__"`` key carrying provenance — including the
:class:`~kinsim.utils.config.ExtractionParams` the shard was built with,
serialised under ``__meta__["extraction_params"]``.

Column reference (for the historical K=11 layout — see
:class:`SampleLayout` for the dynamic version)::

    Cols  | Contents
    ------+--------------------------------------------------------------
    0     | IPD at prediction position (raw uint8 written as float32)
    1     | PW  at prediction position
    2     | stoichiometric fraction (0..1) — per-site motif occupancy
    3..3+K-1                | mc_0..mc_K-1
                            | meth_id at offsets [-upstream, +downstream]
                            | from the prediction position
    3+K..3+K+R-1            | rev_meth_<off_0>..rev_meth_<off_R-1>
                            | complementary-strand meth_id at the offsets
                            | declared in ExtractionParams.rev_meth_offsets
    NCOLS-3                 | CATEGORY  (0=baseline, 1=slowed, 2=near_meth)
    NCOLS-2                 | PARENT_METH   meth_id of the parent methylation
    NCOLS-1                 | PARENT_OFFSET row_pos − parent_meth_pos

The methylation centres themselves land in SLOWED if 0 is in their
signature offsets (m6A, m4C) or NEAR_METH otherwise (m5C, sig [+2, +6]).

Conventions
-----------

* All helpers are pure Python (no pysam) so refine, dataset, tests, and
  scripts can import this module without the BAM dependency.
* For new code, prefer :class:`SampleLayout` (driven by
  :class:`~kinsim.utils.config.ExtractionParams`) over the module-level
  ``COL_*`` constants. The constants are kept for backward compatibility
  with the historical K=11 layout and resolve to the same values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import ExtractionParams, get_extraction_params
from .encoding import KMER_LEFT_PAD, KMER_RIGHT_PAD, K  # noqa: F401 — kept for legacy imports

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Category enum — same regardless of window geometry
# ---------------------------------------------------------------------------

CATEGORY_BASELINE = 0
CATEGORY_SLOWED = 1
CATEGORY_NEAR_METH = 2
CATEGORY_NAMES = {
    CATEGORY_BASELINE: "baseline",
    CATEGORY_SLOWED:   "slowed",
    CATEGORY_NEAR_METH: "near_meth",
}


# ---------------------------------------------------------------------------
# Dynamic SampleLayout — derives all column indices from ExtractionParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleLayout:
    """Column indices for the per-sample storage vector.

    Every index is derived from an :class:`ExtractionParams` so the layout
    automatically adapts when the user changes ``kmer_size`` or
    ``rev_meth_offsets`` in ``kinsim_config.yaml``.

    Construct via :func:`get_sample_layout` (which reads the YAML) or
    :meth:`from_params` for explicit control.

    Attributes:
        params:               The :class:`ExtractionParams` this layout was
                              built from. Re-serialised into shard
                              ``__meta__`` so downstream consumers can
                              re-derive the layout from the shard alone.
        n_cols:               Total column count (= params.sample_ncols).
        col_ipd / col_pw / col_fraction:
                              Indices of the kinetic + occupancy columns.
        col_meth_ctx_start:   First column of the forward meth-context block.
        col_meth_ctx_end:     One past the last column of the forward block
                              (== col_rev_meth).
        col_rev_meth:         First column of the rev_meth block.
        col_category:         CATEGORY column.
        col_parent_meth:      PARENT_METH column.
        col_parent_offset:    PARENT_OFFSET column.
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

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_params(cls, params: ExtractionParams) -> "SampleLayout":
        """Build a :class:`SampleLayout` from a validated :class:`ExtractionParams`.

        Args:
            params: Validated window-geometry record.

        Returns:
            A frozen :class:`SampleLayout` with all column indices populated.
        """
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

    # ------------------------------------------------------------------
    # Geometry shortcuts — delegate to ExtractionParams for readability
    # ------------------------------------------------------------------

    @property
    def kmer_size(self) -> int:
        """Window size (forwarded from :attr:`params`)."""
        return self.params.kmer_size

    @property
    def upstream(self) -> int:
        """Bases of context before the active site."""
        return self.params.upstream

    @property
    def active_site_index(self) -> int:
        """Index of the active (prediction) position inside the kmer."""
        return self.params.active_site_index

    @property
    def n_rev_meth(self) -> int:
        """Number of complementary-strand meth positions captured."""
        return self.params.n_rev_meth

    @property
    def rev_meth_offsets(self) -> tuple[int, ...]:
        """Complementary-strand offsets captured at the active-site footprint."""
        return self.params.rev_meth_offsets

    # ------------------------------------------------------------------
    # Slicing helpers — operate on full sample arrays
    # ------------------------------------------------------------------

    def slice_meth_ctx(self, sample_row):
        """Return the forward meth-context block of a single sample row."""
        return sample_row[self.col_meth_ctx_start:self.col_meth_ctx_end]

    def slice_rev_meth(self, sample_row):
        """Return the rev_meth block of a single sample row."""
        return sample_row[self.col_rev_meth:self.col_category]


def get_sample_layout(params: ExtractionParams | None = None) -> SampleLayout:
    """Return the :class:`SampleLayout` for the given (or current YAML) params.

    Args:
        params: Optional explicit :class:`ExtractionParams`. If ``None``,
            the values are read from ``kinsim_config.yaml`` via
            :func:`~kinsim.utils.config.get_extraction_params`.
    """
    if params is None:
        params = get_extraction_params()
    return SampleLayout.from_params(params)


# ---------------------------------------------------------------------------
# Legacy module-level constants — preserved for the K=11 default layout.
#
# These resolve to the SAME values that the historical hardcoded layout
# produced (METH_CTX_LEN=11, REV_METH_LEN=3, SAMPLE_NCOLS=20). Code that
# still imports `COL_CATEGORY`, `SAMPLE_NCOLS`, etc. keeps working unchanged.
#
# New code should prefer `get_sample_layout()` so the indices adapt
# automatically when `kinsim_config.yaml` declares a different geometry.
# ---------------------------------------------------------------------------

_LEGACY_DEFAULT_PARAMS = ExtractionParams(
    kmer_size=11, upstream=7, downstream=3, rev_meth_offsets=(-1, 0, 1),
)
_LEGACY_LAYOUT = SampleLayout.from_params(_LEGACY_DEFAULT_PARAMS)

METH_CTX_LEFT = _LEGACY_DEFAULT_PARAMS.upstream            # = 7
METH_CTX_RIGHT = _LEGACY_DEFAULT_PARAMS.downstream         # = 3
METH_CTX_LEN = _LEGACY_DEFAULT_PARAMS.kmer_size            # = 11
REV_METH_OFFSETS = _LEGACY_DEFAULT_PARAMS.rev_meth_offsets  # = (-1, 0, 1)
REV_METH_LEN = _LEGACY_DEFAULT_PARAMS.n_rev_meth            # = 3

SAMPLE_NCOLS = _LEGACY_LAYOUT.n_cols                       # = 20
COL_IPD = _LEGACY_LAYOUT.col_ipd                           # = 0
COL_PW = _LEGACY_LAYOUT.col_pw                             # = 1
COL_FRACTION = _LEGACY_LAYOUT.col_fraction                 # = 2
COL_METH_CTX_START = _LEGACY_LAYOUT.col_meth_ctx_start     # = 3
COL_METH_CTX_END = _LEGACY_LAYOUT.col_meth_ctx_end         # = 14
COL_REV_METH = _LEGACY_LAYOUT.col_rev_meth                 # = 14
COL_CATEGORY = _LEGACY_LAYOUT.col_category                 # = 17
COL_PARENT_METH = _LEGACY_LAYOUT.col_parent_meth           # = 18
COL_PARENT_OFFSET = _LEGACY_LAYOUT.col_parent_offset       # = 19


# ---------------------------------------------------------------------------
# Pure helpers — used by extract.py to fill one row at a time
# ---------------------------------------------------------------------------


def slice_meth_context(meth_status, center: int,
                       layout: SampleLayout | None = None) -> list:
    """Return a kmer_size-element list of meth_id around ``center``.

    Args:
        meth_status: Per-position meth_id array (int-like, full length of
                     the read or contig).
        center:      Index of the prediction position in ``meth_status``.
        layout:      Optional :class:`SampleLayout` selecting the window
                     geometry. Defaults to the legacy K=11 layout for
                     backward compatibility.

    Returns:
        A list of length ``layout.kmer_size``; out-of-range positions are
        padded with 0 so every sample has the same fixed-length context.
    """
    lay = layout or _LEGACY_LAYOUT
    n = len(meth_status)
    out = [0] * lay.kmer_size
    for k in range(lay.kmer_size):
        pos = center - lay.upstream + k
        if 0 <= pos < n:
            out[k] = int(meth_status[pos])
    return out


def slice_rev_meth(meth_status_complement, center: int,
                   layout: SampleLayout | None = None) -> list:
    """Return the rev_meth meth_id values at the configured offsets around ``center``.

    Args:
        meth_status_complement: Per-position meth_id array on the
                                complementary strand.
        center:                 Index of the active (prediction) site.
        layout:                 Optional :class:`SampleLayout`. Defaults
                                to the legacy K=11 layout.

    Returns:
        A list of length ``layout.n_rev_meth``, in the same order as
        ``layout.rev_meth_offsets``. Out-of-range positions are padded
        with 0.
    """
    lay = layout or _LEGACY_LAYOUT
    n = len(meth_status_complement)
    out = [0] * lay.n_rev_meth
    for k, off in enumerate(lay.rev_meth_offsets):
        pos = center + off
        if 0 <= pos < n:
            out[k] = int(meth_status_complement[pos])
    return out


def get_categories(arr, layout: SampleLayout | None = None) -> np.ndarray:
    """Return per-sample category as int8 ndarray of length N.

    Args:
        arr:    Sample matrix of shape ``(N, layout.n_cols)``.
        layout: Optional :class:`SampleLayout`. Defaults to the legacy
                K=11 layout (uses ``COL_CATEGORY = 17``).

    Returns:
        ``np.ndarray`` of dtype ``int8`` and length ``N``.
    """
    import numpy as np

    lay = layout or _LEGACY_LAYOUT
    if len(arr) == 0:
        return np.empty(0, dtype=np.int8)
    return arr[:, lay.col_category].astype(np.int8)
