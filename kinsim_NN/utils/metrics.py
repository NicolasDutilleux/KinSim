"""Numerical metrics shared by train + evaluate."""
from __future__ import annotations

import numpy as np


def wasserstein_1d(a: np.ndarray, b: np.ndarray, n_quantiles: int = 1024) -> float:
    """1D Wasserstein-1 distance via sorted-quantile interpolation.

    Equivalent to ``scipy.stats.wasserstein_distance`` for 1D continuous
    distributions but without the SciPy dependency. Returns NaN for
    empty inputs.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.size, b.size, n_quantiles)
    qs = np.linspace(0.0, 1.0, n)
    aq = np.interp(qs, np.linspace(0.0, 1.0, a.size), np.sort(a))
    bq = np.interp(qs, np.linspace(0.0, 1.0, b.size), np.sort(b))
    return float(np.mean(np.abs(aq - bq)))


__all__ = ["wasserstein_1d"]
