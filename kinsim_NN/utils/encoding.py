"""Shared encoding helpers for kinsim_NN.

Self-contained: defines the base map and methylation-id map used across the
package, with no dependency on legacy kinsim modules.
"""
from __future__ import annotations

import numpy as np


# Single source of truth for base → integer code.
BASE_MAP: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}

# Default methylation type → integer mapping. Stable across runs because IDs
# are persisted in shard storage. Extended at runtime by ``get_meth_ids()``
# which reads any additional types declared in ``kinsim_nn_config.yaml``.
METH_IDS: dict[str, int] = {"none": 0, "m6A": 1, "m4C": 2, "m5C": 3}


def get_meth_ids() -> dict[str, int]:
    """Return ``{mod_type: int_id}`` from the kinsim_NN YAML.

    Pinned IDs (METH_IDS) win to keep older shards decodable; any extra
    type declared in YAML is assigned the next free integer in declaration
    order. Falls back to bare METH_IDS if the YAML cannot be loaded.
    """
    try:
        from .config import load_config  # lazy to avoid circular import
        cfg = load_config()
    except (ImportError, OSError, ValueError):
        return dict(METH_IDS)

    user_types = [t.name for t in cfg.methylation_types]
    out: dict[str, int] = {"none": 0}
    next_id = 1
    for pinned, pinned_id in METH_IDS.items():
        if pinned == "none":
            continue
        if pinned in user_types:
            out[pinned] = pinned_id
            next_id = max(next_id, pinned_id + 1)
    for name in user_types:
        if name in out:
            continue
        out[name] = next_id
        next_id += 1
    return out


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


__all__ = [
    "BASE_MAP",
    "BASE_RC",
    "METH_IDS",
    "N_BASE_COUNT",
    "encode_seq",
    "get_meth_ids",
]
