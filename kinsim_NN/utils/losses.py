"""WGAN-GP loss and gradient penalty for kinsim_NN.

Gulrajani et al. 2017, "Improved Training of Wasserstein GANs". The
critic outputs an unbounded scalar; loss = Wasserstein distance
approximation. Gradient penalty enforces the 1-Lipschitz constraint on
the critic without weight clipping.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def wgan_gp_d_loss(
    d_real: torch.Tensor,
    d_fake: torch.Tensor,
    gp: torch.Tensor,
    gp_lambda: float = 10.0,
) -> torch.Tensor:
    """Discriminator (critic) loss.

    ``d_real`` and ``d_fake`` are critic outputs (B,) for real and fake
    samples respectively. ``gp`` is the gradient penalty term from
    :func:`gradient_penalty`. We minimise:

        L_D = mean(d_fake) - mean(d_real) + gp_lambda * gp
    """
    return d_fake.mean() - d_real.mean() + gp_lambda * gp


def wgan_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Generator loss.

    G wants D(fake) to be HIGH. We minimise ``-mean(d_fake)``.
    """
    return -d_fake.mean()


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond_kwargs: dict,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """One-sided gradient penalty for WGAN-GP.

    Interpolates between real and fake samples and penalises the
    deviation of ‖∇_x D(x_interp, cond)‖₂ from 1. ``cond_kwargs`` is the
    dict of conditioning tensors passed unchanged to ``discriminator``
    (e.g. ``base_fwd``, ``base_rev``, ``meth_fwd``, ``meth_rev``).
    """
    if real.shape != fake.shape:
        raise ValueError(f"shape mismatch real {real.shape} vs fake {fake.shape}")

    bsz = real.size(0)
    # Per-sample alpha in [0, 1], broadcast over signal dims
    alpha = torch.rand(bsz, *([1] * (real.ndim - 1)), device=device)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    d_interp = discriminator(signal=interp, **cond_kwargs)
    grad_outputs = torch.ones_like(d_interp)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(bsz, -1)
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


__all__ = ["wgan_gp_d_loss", "wgan_g_loss", "gradient_penalty"]
