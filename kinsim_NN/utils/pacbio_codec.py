"""PacBio kinetics codec: uint8 ↔ frames.

PacBio stores per-base kinetics (IPD, PW) as ``uint8`` in BAM tags via a
non-linear codec with four ranges:

    byte 0..63     → frames 0..63        step 1
    byte 64..127   → frames 64..190      step 2
    byte 128..191  → frames 192..444     step 4
    byte 192..255  → frames 448..952     step 8

Reference: PacBio BAM specification, kinetics tag section.

Training is done in ``log1p(frames)`` space to capture the natural log
distribution of polymerase pause times. At BAM emission, we invert via
nearest-bucket lookup.
"""
from __future__ import annotations

import numpy as np


def _byte_to_frame_scalar(b: int) -> int:
    if b < 64:
        return b
    if b < 128:
        return 64 + 2 * (b - 64)
    if b < 192:
        return 192 + 4 * (b - 128)
    return 448 + 8 * (b - 192)


# Precomputed lookup tables — index by byte (0..255) or frame (clipped).
FRAMES_TABLE: np.ndarray = np.array(
    [_byte_to_frame_scalar(b) for b in range(256)],
    dtype=np.int32,
)  # shape (256,)

_FRAMES_TO_BYTE: np.ndarray = np.full(
    FRAMES_TABLE[-1] + 1, 0, dtype=np.uint8
)
# Build the inverse: for every frame value, find the largest byte b such
# that FRAMES_TABLE[b] <= frame (nearest-bucket from below). Then for
# values between two bucket centers we round to nearest at lookup time.
for b in range(256):
    f = int(FRAMES_TABLE[b])
    next_f = int(FRAMES_TABLE[b + 1]) if b < 255 else f + 1
    # Each byte b covers frames [FRAMES_TABLE[b], next_f). The bucket
    # center mapping rounds 'frame' to the nearest bucket.
    half = (next_f - f) // 2
    for v in range(f, min(next_f, _FRAMES_TO_BYTE.shape[0])):
        # Round halfway to the closer byte. For step=1 this is exact.
        # For larger steps, the lookup matches PacBio decoder behaviour.
        _FRAMES_TO_BYTE[v] = b if (v - f) < half + 1 else min(b + 1, 255)


def uint8_to_frames(arr: np.ndarray) -> np.ndarray:
    """Decode PacBio uint8 kinetics bytes → frames (int32).

    Vectorised lookup via :data:`FRAMES_TABLE`. Input dtype must be uint8
    or coercible to it; output is int32 with the same shape.
    """
    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.uint8)
    return FRAMES_TABLE[arr]


def frames_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Encode frame values → PacBio uint8 bytes.

    Values above ``FRAMES_TABLE[-1]`` (952) are clipped to byte 255.
    Negative values are clipped to byte 0.
    """
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr).astype(np.int32)
    arr = np.clip(arr, 0, _FRAMES_TO_BYTE.shape[0] - 1)
    return _FRAMES_TO_BYTE[arr]


def uint8_to_log1p_frames(arr: np.ndarray) -> np.ndarray:
    """Convenience: uint8 BAM byte → log1p(frames). Float32 output."""
    return np.log1p(uint8_to_frames(arr).astype(np.float32))


def log1p_frames_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convenience: log1p(frames) → uint8 BAM byte. Inverse of
    :func:`uint8_to_log1p_frames` up to bucket-rounding precision."""
    frames = np.expm1(arr)
    frames = np.maximum(frames, 0.0)
    return frames_to_uint8(np.rint(frames).astype(np.int32))


__all__ = [
    "FRAMES_TABLE",
    "uint8_to_frames",
    "frames_to_uint8",
    "uint8_to_log1p_frames",
    "log1p_frames_to_uint8",
]
