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
    form: str = "one_sided",
) -> torch.Tensor:
    """Gradient penalty for WGAN-GP / WGAN-LP.

    Interpolates between real and fake samples conditioned on ``cond_kwargs``
    and penalises the squared deviation of ``‖∇_x D(x_interp, cond)‖₂`` from 1.

    Two penalty forms are supported:
      * ``"two_sided"`` — original WGAN-GP, ``mean((‖∇‖₂ − 1)²)``
        (Gulrajani et al., NeurIPS 2017).
      * ``"one_sided"`` — WGAN-LP, ``mean(max(0, ‖∇‖₂ − 1)²)``
        (Petzka et al., ICLR 2018). Only penalises when the gradient
        norm exceeds 1, so a degenerate "flat critic" optimum
        (``‖∇‖₂ → 0``) is no longer attractive — empirically observed
        as the failure mode of the two-sided form on this dataset.

    ``cond_kwargs`` is forwarded unchanged to the discriminator
    (``base_fwd_onehot``, ``base_rev_onehot``, ``meth_fwd_onehot``,
    ``meth_rev_onehot``).
    """
    if real.shape != fake.shape:
        raise ValueError(f"shape mismatch real {real.shape} vs fake {fake.shape}")
    if form not in ("one_sided", "two_sided"):
        raise ValueError(
            f"gradient_penalty form must be 'one_sided' or 'two_sided', got {form!r}"
        )

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
    if form == "two_sided":
        return ((grad_norm - 1.0) ** 2).mean()
    # one_sided (WGAN-LP)
    return torch.clamp(grad_norm - 1.0, min=0.0).pow(2).mean()


__all__ = ["wgan_gp_d_loss", "wgan_g_loss", "gradient_penalty"]
