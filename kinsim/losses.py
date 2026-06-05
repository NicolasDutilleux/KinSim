"""Training losses for kinsim.

After the step-130k audit (see thesis §5.4 + audit_kinsim_step130k.json):
the previous loss (bucketed ED² + L1 on per-bucket *mean*) reproduced
the bucket mean well but flattened the per-position signature — exactly
what the audit measured on SLOWED. Two changes:

1. Bucket by (category, parent_meth) instead of category alone — done
   in kinsim/train.py:_compute_bucket_id. Inside a 5-bucket
   stratification (BASELINE / NEAR_METH / SLOWED_m6A / SLOWED_m4C /
   SLOWED_m5C) each bucket's REAL per-position mean profile actually
   shows the signature peaks rather than averaging them out across
   meth types.

2. Replace the L1-on-bucket-mean anchor by a per-position 1D
   Wasserstein loss (``spatial_per_position_w1``). Sorted L1 between
   real and fake samples at each (bucket, position, channel) cell. This
   penalises mismatches in the per-position marginal distribution
   directly, not just in its first moment. It captures the missing
   peaks AND the under-produced tail at once.

The total loss is:

    L = ED²(joint, bucketed)   +   λ_pos * W1_per_position(bucketed)

ED² keeps the joint-structure pressure (correlations across positions
inside the same window are still constrained). W1-per-position adds the
explicit per-position marginal pressure the previous L1-on-mean term
could not provide.
"""
from __future__ import annotations

import torch


def _pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean pairwise distances between rows of x (N, D) and y (M, D)."""
    xx = (x * x).sum(dim=-1, keepdim=True)
    yy = (y * y).sum(dim=-1, keepdim=True).t()
    xy = x @ y.t()
    d2 = xx + yy - 2 * xy
    return d2.clamp_min(1e-12).sqrt()


def energy_distance(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Energy distance² between empirical distributions of real and fake.

    real, fake: (N, D) and (M, D) tensors. Returns a non-negative scalar.
    """
    if real.shape[0] < 2 or fake.shape[0] < 2:
        return real.new_zeros(())
    d_rf = _pairwise_distances(real, fake).mean()
    d_rr = _pairwise_distances(real, real).mean()
    d_ff = _pairwise_distances(fake, fake).mean()
    return 2.0 * d_rf - d_rr - d_ff


def bucketed_energy_distance(
    real: torch.Tensor,
    fake: torch.Tensor,
    bucket_id: torch.Tensor,
    n_buckets: int,
    min_samples: int = 4,
) -> tuple[torch.Tensor, dict[int, float]]:
    """Energy distance averaged across category buckets.

    real, fake: (B, D) tensors. bucket_id: (B,) int tensor in [0, n_buckets).
    """
    losses: list[torch.Tensor] = []
    diag: dict[int, float] = {}
    for b in range(n_buckets):
        mask = (bucket_id == b)
        if int(mask.sum().item()) < min_samples:
            continue
        r = real[mask]
        f = fake[mask]
        ed = energy_distance(r, f)
        losses.append(ed)
        diag[b] = float(ed.detach().item())
    if not losses:
        return real.new_zeros(()), diag
    return torch.stack(losses).mean(), diag


def spatial_per_position_w1(
    real: torch.Tensor,
    fake: torch.Tensor,
    bucket_id: torch.Tensor,
    n_buckets: int,
    min_samples: int = 4,
) -> tuple[torch.Tensor, dict[int, float]]:
    """Per-bucket per-position 1-D Wasserstein loss on (B, K, C) tiles.

    For each populated bucket and each (position, channel) cell, computes
    the empirical 1-D Wasserstein-1 distance via sorted L1 between the
    real and fake samples drawn from that bucket. The empirical W₁ between
    two equal-size samples in 1-D equals the L1 norm of the difference of
    their sorted values — directly differentiable via autograd through
    ``torch.sort``'s stable gradient.

    The result for each bucket is the mean across all (position, channel)
    cells; the final loss averages over populated buckets. Compared to
    the previous L1-on-bucket-mean anchor, this:

    * Looks at the FULL per-position marginal distribution, not just its
      first moment. The narrow-std / under-produced-tail pathology that
      the audit measured (std ratio 0.74, P[IPD>80] ratio 0.54 on SLOWED)
      is directly visible in sorted-L1 because the sorted quantiles
      differ when spread differs.
    * Penalises the per-position SIGNATURE shape directly. If the real
      SLOWED_m6A bucket has a peak at position 10, the sorted samples
      at position 10 sit at higher values; the sorted-L1 of a flat
      generator vs that bucket grows proportionally to the missing peak.

    Returns ``(loss, per_bucket_diag)`` analogous to
    :func:`bucketed_energy_distance`.
    """
    losses: list[torch.Tensor] = []
    diag: dict[int, float] = {}
    for b in range(n_buckets):
        mask = (bucket_id == b)
        if int(mask.sum().item()) < min_samples:
            continue
        r = real[mask]                           # (Nb, K, C)
        f = fake[mask]                           # (Nb, K, C)
        # Sort each (position, channel) column independently along the
        # sample axis. ``r`` and ``f`` have the same Nb by construction
        # (both come from the same mask), so sorted-L1 is the empirical
        # W1 between the two 1D marginals.
        r_sorted, _ = r.sort(dim=0)
        f_sorted, _ = f.sort(dim=0)
        # |r_sorted - f_sorted| has shape (Nb, K, C). The mean over Nb is
        # the per-cell sample-mean of the absolute deviation between
        # paired sorted values, equivalent to the empirical W1 in 1D up
        # to a normalising constant. Then mean over (K, C) aggregates
        # across all positions and channels.
        per_cell = (r_sorted - f_sorted).abs().mean(dim=0)   # (K, C)
        bucket_loss = per_cell.mean()
        losses.append(bucket_loss)
        diag[b] = float(bucket_loss.detach().item())
    if not losses:
        return real.new_zeros(()), diag
    return torch.stack(losses).mean(), diag


def tail_quantile_loss(
    real: torch.Tensor,
    fake: torch.Tensor,
    bucket_id: torch.Tensor,
    n_buckets: int,
    quantiles: tuple[float, ...] = (0.95, 0.99),
    min_samples: int = 8,
) -> tuple[torch.Tensor, dict[int, float]]:
    """Per-bucket per-position MSE between empirical tail quantiles.

    Sorted-L1 (spatial_per_position_w1) measures average sample-to-sample
    deviation between sorted real/fake values at each (position, channel),
    but the top-quantile mass is only a small fraction of the sorted array
    and its mismatch is averaged with the bulk. The audit showed the
    generator under-producing extreme values: P(IPD > 80) ratio G/R ≈
    0.54-0.58 on BASELINE and SLOWED. That tail under-production directly
    suppresses ipdSummary's Z-scores (which are dominated by the extreme
    end of the IPD distribution) and is the most likely cause of motif
    recovery failure.

    This loss penalises mismatch on the top quantiles SPECIFICALLY by
    computing the per-cell empirical quantile of ``real`` and ``fake`` at
    each requested level (default p95, p99) and MSE-comparing them.

    Returns ``(loss, per_bucket_diag)`` analogous to the other bucketed
    losses. Default ``min_samples=8`` higher than the others because a
    p99 estimate from < 8 samples is essentially the maximum and has
    high variance.
    """
    losses: list[torch.Tensor] = []
    diag: dict[int, float] = {}
    qs = torch.tensor(list(quantiles), device=real.device, dtype=real.dtype)
    for b in range(n_buckets):
        mask = (bucket_id == b)
        if int(mask.sum().item()) < min_samples:
            continue
        r = real[mask]                                # (Nb, K, C)
        f = fake[mask]
        # torch.quantile reduces the chosen dim; with a 1-D ``qs`` of
        # length Q the result has a leading Q dimension → (Q, K, C).
        r_q = torch.quantile(r, qs, dim=0)
        f_q = torch.quantile(f, qs, dim=0)
        bucket_loss = ((r_q - f_q) ** 2).mean()
        losses.append(bucket_loss)
        diag[b] = float(bucket_loss.detach().item())
    if not losses:
        return real.new_zeros(()), diag
    return torch.stack(losses).mean(), diag


__all__ = [
    "energy_distance",
    "bucketed_energy_distance",
    "spatial_per_position_w1",
    "tail_quantile_loss",
]
