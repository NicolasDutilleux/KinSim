"""Shared encoding helpers for kinsim_NN.

A thin compatibility layer that re-exports the canonical base map from
``kinsim.utils.encoding`` (single source of truth) and provides
``encode_seq`` used by both extract and generate.
"""
from __future__ import annotations

import numpy as np

from kinsim.utils.encoding import BASE_MAP


# Case-insensitive base → integer code lookup (A=0, C=1, G=2, T=3).
# Non-ACGT bases (N, IUPAC ambiguity) silently encode as A (0); callers can
# track this via :data:`N_BASE_COUNT` if they care.
_BASE_TO_CODE: dict[str, int] = dict(BASE_MAP)
_BASE_TO_CODE.update({b.lower(): i for b, i in BASE_MAP.items()})

#: Per-process counter of non-ACGT bases silently encoded as A.
#: Wrapped in a list so consumers can mutate ``N_BASE_COUNT[0]`` from outside.
N_BASE_COUNT: list[int] = [0]


def encode_seq(seq: str, track_n: bool = True) -> np.ndarray:
    """Encode an ACGT(case-insensitive) string to uint8 codes.

    Non-ACGT bases (N, ambiguity codes) are silently encoded as A (0).
    When ``track_n=True``, the count is added to :data:`N_BASE_COUNT[0]`.
    """
    arr = np.empty(len(seq), dtype=np.uint8)
    n_count = 0
    for i, b in enumerate(seq):
        c = _BASE_TO_CODE.get(b)
        if c is None:
            arr[i] = 0
            n_count += 1
        else:
            arr[i] = c
    if track_n:
        N_BASE_COUNT[0] += n_count
    return arr


__all__ = ["BASE_MAP", "encode_seq", "N_BASE_COUNT"]
