"""Shared encoding helpers for kinsim_NN.

A thin compatibility layer that re-exports the canonical base map from
``kinsim.utils.encoding`` (single source of truth) and provides
``encode_seq`` used by both extract and generate.
"""
from __future__ import annotations

import numpy as np

from kinsim.utils.encoding import BASE_MAP


# 256-entry lookup table: byte (uppercase ACGT or lowercase variant) → code 0..3.
# Everything else (N, IUPAC ambiguity, gaps, whitespace) maps to 0 (A).
# Bit 5 (0x20) flips upper/lower case in ASCII so we set both cases.
_BYTE_TO_CODE: np.ndarray = np.zeros(256, dtype=np.uint8)
for _b, _v in BASE_MAP.items():
    _BYTE_TO_CODE[ord(_b)] = _v
    _BYTE_TO_CODE[ord(_b.lower())] = _v

# Reverse-complement table on integer codes: A↔T (0↔3), C↔G (1↔2).
BASE_RC: np.ndarray = np.array([3, 2, 1, 0], dtype=np.uint8)

#: Per-process counter of non-ACGT bases silently encoded as A.
#: Wrapped in a list so consumers can mutate ``N_BASE_COUNT[0]`` from outside.
N_BASE_COUNT: list[int] = [0]

# Mask of bytes that ARE valid ACGT (case-insensitive). Used to count N-bases
# at encode time without a per-character Python loop.
_VALID_BASE_MASK: np.ndarray = np.zeros(256, dtype=bool)
for _b in BASE_MAP:
    _VALID_BASE_MASK[ord(_b)] = True
    _VALID_BASE_MASK[ord(_b.lower())] = True


def encode_seq(seq: str, track_n: bool = True) -> np.ndarray:
    """Encode an ACGT(case-insensitive) string to uint8 codes.

    Non-ACGT bases (N, ambiguity codes) are silently encoded as A (0).
    When ``track_n=True``, the count is added to :data:`N_BASE_COUNT[0]`.

    Vectorised via numpy LUT — ~50× faster than per-character Python loop
    on K=21 windows.
    """
    if not seq:
        return np.empty(0, dtype=np.uint8)
    bytes_ = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    if track_n:
        n_count = int((~_VALID_BASE_MASK[bytes_]).sum())
        if n_count:
            N_BASE_COUNT[0] += n_count
    return _BYTE_TO_CODE[bytes_]


__all__ = ["BASE_MAP", "BASE_RC", "encode_seq", "N_BASE_COUNT"]
