"""Compact shard schema for kinsim_NN.

Each strain's training data lives in a single pickle file:

    shards/<strain>_shard.pkl

The pickle contains a dict with these arrays (all length N):

    base_fwd   uint8  (N, K)            A/C/G/T codes 0..3
    meth_fwd   uint8  (N, K)            meth_id 0..M-1 (0 = none)
    meth_rev   uint8  (N, K)            meth_id 0..M-1 on reverse strand
    signal     uint8  (N, K, 4)         IPD_fwd, PW_fwd, IPD_rev, PW_rev (uint8 codec)
    category   uint8  (N,)              0 = baseline, 1 = meth-positive
    ref_id     uint16 (N,)              indexed into __meta__["ref_names"]
    ref_pos    int32  (N,)              0-based position of window center
    strand     int8   (N,)              +1 / -1
    zmw        int64  (N,)              ZMW number (16-byte read name hash if non-numeric)

plus a metadata dict::

    __meta__ = {
        "config_version": "kinsim_NN-1",
        "k": 21, "half_width": 10, "n_channels": 4,
        "n_meth_types": M,
        "meth_id_by_name": {"none": 0, "m6A": 1, ...},
        "ref_names": ["contig_1", "contig_2", ...],
        "strain_id": "strepto_bc2033",
        "git_sha": "...",
        "kinsim_nn_version": "0.1.0",
        "timestamp_utc": "...",
        "label_sources": [...],
    }

Storage: ~170 bytes per sample. ~500 MB per strain at 3M samples.
"""
from __future__ import annotations

import logging
import pickle
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


log = logging.getLogger(__name__)

SHARD_CONFIG_VERSION = "kinsim_NN-1"


@dataclass
class ShardData:
    """In-memory shard contents. All arrays length N."""

    base_fwd: np.ndarray   # (N, K) uint8
    meth_fwd: np.ndarray   # (N, K) uint8
    meth_rev: np.ndarray   # (N, K) uint8
    signal: np.ndarray     # (N, K, 4) uint8
    category: np.ndarray   # (N,) uint8
    ref_id: np.ndarray     # (N,) uint16
    ref_pos: np.ndarray    # (N,) int32
    strand: np.ndarray     # (N,) int8
    zmw: np.ndarray        # (N,) int64
    meta: dict[str, Any]

    @property
    def n(self) -> int:
        return self.base_fwd.shape[0]

    @property
    def k(self) -> int:
        return int(self.meta.get("k", self.base_fwd.shape[1]))


def _hash_zmw(name: str) -> int:
    """Stable 63-bit hash from a read name → int64-safe ZMW id."""
    return int(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF)


def write_shard(path: Path, shard: ShardData) -> None:
    """Serialise the shard to disk (pickle protocol 5)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "__meta__": shard.meta,
        "base_fwd": shard.base_fwd,
        "meth_fwd": shard.meth_fwd,
        "meth_rev": shard.meth_rev,
        "signal": shard.signal,
        "category": shard.category,
        "ref_id": shard.ref_id,
        "ref_pos": shard.ref_pos,
        "strand": shard.strand,
        "zmw": shard.zmw,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=5)
    bytes_ = sum(
        getattr(v, "nbytes", 0) for k, v in payload.items() if k != "__meta__"
    )
    log.info(
        "Wrote shard %s  N=%d  K=%d  payload=%.1f MB",
        path, shard.n, shard.k, bytes_ / 1e6,
    )


def read_shard(path: Path) -> ShardData:
    """Load a shard from disk."""
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    meta = payload.get("__meta__", {})
    cfg_v = meta.get("config_version")
    if cfg_v != SHARD_CONFIG_VERSION:
        raise ValueError(
            f"{path}: config_version mismatch (got {cfg_v!r}, "
            f"expected {SHARD_CONFIG_VERSION!r})"
        )
    return ShardData(
        base_fwd=payload["base_fwd"],
        meth_fwd=payload["meth_fwd"],
        meth_rev=payload["meth_rev"],
        signal=payload["signal"],
        category=payload["category"],
        ref_id=payload["ref_id"],
        ref_pos=payload["ref_pos"],
        strand=payload["strand"],
        zmw=payload["zmw"],
        meta=meta,
    )


def empty_shard(k: int) -> dict[str, list]:
    """Builder helper: return a dict of empty per-sample lists.

    Use during extract: append to each list per sample, then stack at the
    end via :func:`finalize_shard`.
    """
    return {
        "base_fwd": [],
        "meth_fwd": [],
        "meth_rev": [],
        "signal": [],
        "category": [],
        "ref_id": [],
        "ref_pos": [],
        "strand": [],
        "zmw": [],
    }


def finalize_shard(builder: dict[str, list], meta: dict[str, Any], k: int) -> ShardData:
    """Stack the per-sample lists into a :class:`ShardData`."""
    if not builder["base_fwd"]:
        # Empty shard
        return ShardData(
            base_fwd=np.empty((0, k), dtype=np.uint8),
            meth_fwd=np.empty((0, k), dtype=np.uint8),
            meth_rev=np.empty((0, k), dtype=np.uint8),
            signal=np.empty((0, k, 4), dtype=np.uint8),
            category=np.empty(0, dtype=np.uint8),
            ref_id=np.empty(0, dtype=np.uint16),
            ref_pos=np.empty(0, dtype=np.int32),
            strand=np.empty(0, dtype=np.int8),
            zmw=np.empty(0, dtype=np.int64),
            meta=meta,
        )
    return ShardData(
        base_fwd=np.stack(builder["base_fwd"]).astype(np.uint8),
        meth_fwd=np.stack(builder["meth_fwd"]).astype(np.uint8),
        meth_rev=np.stack(builder["meth_rev"]).astype(np.uint8),
        signal=np.stack(builder["signal"]).astype(np.uint8),
        category=np.asarray(builder["category"], dtype=np.uint8),
        ref_id=np.asarray(builder["ref_id"], dtype=np.uint16),
        ref_pos=np.asarray(builder["ref_pos"], dtype=np.int32),
        strand=np.asarray(builder["strand"], dtype=np.int8),
        zmw=np.asarray(builder["zmw"], dtype=np.int64),
        meta=meta,
    )


__all__ = [
    "SHARD_CONFIG_VERSION",
    "ShardData",
    "read_shard",
    "write_shard",
    "empty_shard",
    "finalize_shard",
    "_hash_zmw",
]
