"""Training losses for kinsim.

Two components, combined as

    L_total = L_ed + λ_mean * L_mean

L_ed (Energy Distance, Székely & Rizzo 2013) — captures the joint
distribution match over the full (K × n_channels)-dimensional tile,
within each category bucket. ED is a proper metric on distributions and
has no kernel-bandwidth hyperparameter, unlike MMD with a Gaussian
kernel:

    ED²(P, Q) = 2 E‖X − Y‖ − E‖X − X'‖ − E‖Y − Y'‖

where X, X' ~ P and Y, Y' ~ Q are independent. ED captures the joint
distribution including all moments and correlations between positions
and channels, which is what the downstream methylation chain probes.

L_mean (per-position L1 between conditional means) — a small auxiliary
that anchors the predicted bucket mean to the real bucket mean. ED
alone is theoretically sufficient but converges slowly because it has
no signal about per-sample correspondence; the L1 anchor pulls the
model into the right region of the joint quickly, then ED shapes the
spread and the correlations around it. Default λ_mean = 0.1.

The "bucket" is the emission category (BASELINE / SLOWED / NEAR_METH).
Splitting by category ensures the slowed-position bucket is not
swamped by the much larger baseline bucket when matching.
"""
from __future__ import annotations

import torch


def _pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean pairwise distances between rows of x (N, D) and y (M, D)."""
    # ||a - b||² = ||a||² + ||b||² - 2 a·b
    xx = (x * x).sum(dim=-1, keepdim=True)             # (N, 1)
    yy = (y * y).sum(dim=-1, keepdim=True).t()         # (1, M)
    xy = x @ y.t()                                     # (N, M)
    d2 = xx + yy - 2 * xy
    return d2.clamp_min(1e-12).sqrt()


def energy_distance(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Energy distance² between empirical distributions of real and fake.

    real, fake: (N, D) and (M, D) tensors. Returns a non-negative scalar.

    Implementation note: returns ED² (squared energy distance) for
    smoothness; the gradient is identical up to sign and the scalar is
    backprop-stable around 0. Both terms are required so that scaling
    the input doesn't trivially decrease the loss.
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
    Buckets with fewer than `min_samples` are skipped.

    Returns the mean ED² across populated buckets, and a diagnostic dict
    with per-bucket values (Python floats, detached).
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


def conditional_mean_l1(
    real: torch.Tensor,
    fake: torch.Tensor,
    bucket_id: torch.Tensor,
    n_buckets: int,
    min_samples: int = 4,
) -> torch.Tensor:
    """L1 distance between per-bucket means of real and fake tiles.

    real, fake: (B, K, C) tensors. Returns mean L1 across populated
    buckets and across all (K, C) positions.
    """
    losses: list[torch.Tensor] = []
    for b in range(n_buckets):
        mask = (bucket_id == b)
        if int(mask.sum().item()) < min_samples:
            continue
        r_mean = real[mask].mean(dim=0)               # (K, C)
        f_mean = fake[mask].mean(dim=0)
        losses.append((r_mean - f_mean).abs().mean())
    if not losses:
        return real.new_zeros(())
    return torch.stack(losses).mean()


__all__ = [
    "energy_distance",
    "bucketed_energy_distance",
    "conditional_mean_l1",
]
