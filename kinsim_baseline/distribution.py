"""Per-kmer empirical (IPD, PW) sample bank — pure storage + sampling.

The table holds, for every 11-mer index in [0, 4^11), up to ``n_per_kmer``
empirical (IPD, PW) byte pairs observed in real PacBio data. Sampling
draws one pair uniformly at random from a kmer's bucket. Kmers that were
never observed (count == 0) fall back to a global pool of (IPD, PW)
pairs sampled from the entire training baseline.

Storage layout (.npz):
    ipd:        (4194304, n_per_kmer) uint8
    pw:         (4194304, n_per_kmer) uint8
    count:      (4194304,)            uint16    — actual samples (≤ n_per_kmer)
    global_ipd: (n_global_pool,)      uint8     — cross-kmer fallback
    global_pw:  (n_global_pool,)      uint8

At n_per_kmer=200 the per-kmer arrays are 4M × 200 × 2 bytes ≈ 1.6 GB.
The global pool default 1M pairs adds ~2 MB.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from kinsim.utils.encoding import KMER_MASK

log = logging.getLogger(__name__)

NUM_KMERS = KMER_MASK + 1  # 4 ** 11 = 4 194 304


class KmerDistribution:
    """Per-kmer empirical (IPD, PW) bank with global-pool fallback.

    Optionally also carries a parallel **modified** sample bank populated by
    :func:`kinsim_baseline.calibrate.calibrate_from_bams`. The modified
    samples come from real BAM positions whose observed IPD exceeds the
    kmer's baseline 99th percentile — i.e. candidate-methylation positions.
    Per-kmer IPD ratio (modified mean / baseline mean) is computed
    on-demand via :meth:`ipd_ratio`.
    """

    def __init__(
        self,
        ipd: np.ndarray,
        pw: np.ndarray,
        count: np.ndarray,
        global_ipd: Optional[np.ndarray] = None,
        global_pw: Optional[np.ndarray] = None,
        # Optional modified bank (calibration output)
        modified_ipd: Optional[np.ndarray] = None,
        modified_pw: Optional[np.ndarray] = None,
        modified_count: Optional[np.ndarray] = None,
        threshold_percentile: Optional[float] = None,
    ):
        if ipd.shape != pw.shape:
            raise ValueError(f"ipd shape {ipd.shape} != pw shape {pw.shape}")
        if ipd.shape[0] != NUM_KMERS:
            raise ValueError(
                f"first axis must be NUM_KMERS={NUM_KMERS}, got {ipd.shape[0]}"
            )
        if count.shape != (NUM_KMERS,):
            raise ValueError(
                f"count shape must be ({NUM_KMERS},), got {count.shape}"
            )
        self.ipd = ipd            # (NUM_KMERS, n_per_kmer) uint8
        self.pw = pw              # (NUM_KMERS, n_per_kmer) uint8
        self.count = count        # (NUM_KMERS,) uint16
        self.n_per_kmer = ipd.shape[1]
        self._global_ipd = (
            global_ipd if global_ipd is not None and global_ipd.size > 0 else None
        )
        self._global_pw = (
            global_pw if global_pw is not None and global_pw.size > 0 else None
        )
        # Optional modified bank
        self.modified_ipd = modified_ipd        # (NUM_KMERS, n_mod_per_kmer) or None
        self.modified_pw = modified_pw
        self.modified_count = modified_count    # (NUM_KMERS,) or None
        self.threshold_percentile = threshold_percentile

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "KmerDistribution":
        d = np.load(str(path))
        return cls(
            ipd=d["ipd"],
            pw=d["pw"],
            count=d["count"],
            global_ipd=d["global_ipd"] if "global_ipd" in d.files else None,
            global_pw=d["global_pw"] if "global_pw" in d.files else None,
            modified_ipd=d["modified_ipd"] if "modified_ipd" in d.files else None,
            modified_pw=d["modified_pw"] if "modified_pw" in d.files else None,
            modified_count=d["modified_count"] if "modified_count" in d.files else None,
            threshold_percentile=(
                float(d["threshold_percentile"])
                if "threshold_percentile" in d.files
                else None
            ),
        )

    def save(self, path: str | Path) -> None:
        kw = dict(ipd=self.ipd, pw=self.pw, count=self.count)
        if self._global_ipd is not None:
            kw["global_ipd"] = self._global_ipd
            kw["global_pw"] = self._global_pw
        if self.modified_ipd is not None:
            kw["modified_ipd"] = self.modified_ipd
            kw["modified_pw"] = self.modified_pw
            kw["modified_count"] = self.modified_count
        if self.threshold_percentile is not None:
            kw["threshold_percentile"] = np.array(self.threshold_percentile)
        np.savez_compressed(str(path), **kw)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        kmer_ids: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample (IPD, PW) for an array of kmer IDs.

        Args:
            kmer_ids: (n,) integer array of 22-bit kmer indices.
            rng: optional ``np.random.Generator``.

        Returns:
            ``(ipds, pws)`` each (n,) uint8.
        """
        if rng is None:
            rng = np.random.default_rng()
        kmer_ids = np.asarray(kmer_ids, dtype=np.int64)
        n = kmer_ids.shape[0]
        out_ipd = np.empty(n, dtype=np.uint8)
        out_pw = np.empty(n, dtype=np.uint8)

        counts = self.count[kmer_ids]
        seen = counts > 0
        # Per-kmer empirical lookup for kmers we have data for.
        if seen.any():
            seen_kmers = kmer_ids[seen]
            seen_counts = counts[seen].astype(np.int64)
            # uniform sample index in [0, count) per kmer
            sample_idx = (rng.random(seen_counts.size) * seen_counts).astype(np.int64)
            out_ipd[seen] = self.ipd[seen_kmers, sample_idx]
            out_pw[seen] = self.pw[seen_kmers, sample_idx]

        # Fallback: kmers never observed in baseline data.
        unseen_n = int((~seen).sum())
        if unseen_n:
            if self._global_ipd is not None and self._global_ipd.size > 0:
                idx = rng.integers(0, self._global_ipd.size, unseen_n)
                out_ipd[~seen] = self._global_ipd[idx]
                out_pw[~seen] = self._global_pw[idx]
            else:
                # Very unlikely (no global pool). PacBio convention: 1 = "no data".
                out_ipd[~seen] = 1
                out_pw[~seen] = 1
        return out_ipd, out_pw

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage(self) -> float:
        """Fraction of the 4M kmer space with at least one sample."""
        return float((self.count > 0).mean())

    def n_global_pool(self) -> int:
        return 0 if self._global_ipd is None else int(self._global_ipd.size)

    def quantile(self, kmer_id: int, q: float, axis: str = "ipd") -> float | None:
        """Quantile of one kmer's empirical distribution. ``None`` if no data."""
        kmer_id = int(kmer_id)
        n = int(self.count[kmer_id])
        if n == 0:
            return None
        arr = self.ipd if axis == "ipd" else self.pw
        return float(np.quantile(arr[kmer_id, :n].astype(np.float32), q))

    def stats(self, kmer_id: int) -> dict | None:
        """Mean/std/quantiles for one kmer. ``None`` if no data."""
        kmer_id = int(kmer_id)
        n = int(self.count[kmer_id])
        if n == 0:
            return None
        ipd = self.ipd[kmer_id, :n].astype(np.float32)
        pw = self.pw[kmer_id, :n].astype(np.float32)
        return {
            "n": n,
            "ipd_mean": float(ipd.mean()),
            "ipd_std": float(ipd.std()),
            "ipd_p5": float(np.quantile(ipd, 0.05)),
            "ipd_p95": float(np.quantile(ipd, 0.95)),
            "pw_mean": float(pw.mean()),
            "pw_std": float(pw.std()),
        }

    # ------------------------------------------------------------------
    # Modified bank (populated by calibrate.py)
    # ------------------------------------------------------------------

    def has_modified(self) -> bool:
        return self.modified_ipd is not None and self.modified_count is not None

    def sample_modified(
        self,
        kmer_ids: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample (IPD, PW) from the modified bank.

        Same API as :meth:`sample` but draws from the modified pool.
        Falls back to the baseline pool when a kmer has no modified
        samples (so that boundary positions, or kmers absent from the
        calibration set, still get a reasonable kinetic value).
        """
        if not self.has_modified():
            return self.sample(kmer_ids, rng=rng)
        if rng is None:
            rng = np.random.default_rng()
        kmer_ids = np.asarray(kmer_ids, dtype=np.int64)
        n = kmer_ids.shape[0]
        out_ipd = np.empty(n, dtype=np.uint8)
        out_pw = np.empty(n, dtype=np.uint8)

        mod_counts = self.modified_count[kmer_ids]
        seen = mod_counts > 0
        if seen.any():
            seen_kmers = kmer_ids[seen]
            seen_counts = mod_counts[seen].astype(np.int64)
            sample_idx = (rng.random(seen_counts.size) * seen_counts).astype(np.int64)
            out_ipd[seen] = self.modified_ipd[seen_kmers, sample_idx]
            out_pw[seen] = self.modified_pw[seen_kmers, sample_idx]
        # Fallback: kmers with no modified samples → baseline.
        if not seen.all():
            fallback_kmers = kmer_ids[~seen]
            fb_ipd, fb_pw = self.sample(fallback_kmers, rng=rng)
            out_ipd[~seen] = fb_ipd
            out_pw[~seen] = fb_pw
        return out_ipd, out_pw

    def ipd_ratio(self) -> np.ndarray:
        """Per-kmer modified-mean / baseline-mean IPD ratio.

        Returns a ``(NUM_KMERS,)`` float32 array. NaN where either side
        has zero samples.
        """
        ratio = np.full(NUM_KMERS, np.nan, dtype=np.float32)
        if not self.has_modified():
            return ratio
        # Vectorised mean per row (where count > 0).
        # Baseline mean.
        base_n = self.count.astype(np.int64)
        base_seen = base_n > 0
        if base_seen.any():
            base_sum = self.ipd.astype(np.float32).sum(axis=1)
            # Subtract zero-padding contribution (zeros add nothing, so sum is fine
            # — but mean must divide by the actual count, not n_per_kmer).
            base_mean = np.where(base_seen, base_sum / np.where(base_n == 0, 1, base_n), np.nan)
        else:
            base_mean = np.full(NUM_KMERS, np.nan, dtype=np.float32)

        mod_n = self.modified_count.astype(np.int64)
        mod_seen = mod_n > 0
        mod_sum = self.modified_ipd.astype(np.float32).sum(axis=1)
        mod_mean = np.where(mod_seen, mod_sum / np.where(mod_n == 0, 1, mod_n), np.nan)

        valid = base_seen & mod_seen & (base_mean > 0)
        ratio[valid] = mod_mean[valid] / base_mean[valid]
        return ratio

    def n_modified_pool(self) -> int:
        return 0 if not self.has_modified() else int((self.modified_count > 0).sum())
