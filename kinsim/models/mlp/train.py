"""Train the MLP predictor on merged raw kinetic samples.

Input format
------------
A single merged .pkl file produced by the shared data pipeline:

    kinsim extract <reads.bam> <motifs> <shard.pkl>   # one per BAM / SLURM task
    kinsim merge   <shards_dir/> <master_data.pkl>     # combine all shards

The .pkl maps (kmer_id: int, meth_id: int) → np.ndarray(N, 2) where columns
are [IPD, PW] in raw uint8 space [0, 255] as read from BAM fi/fp tags.
There is no separate extraction step for MLP — it reuses the cGAN pipeline.

Loss function
-------------
We use Gaussian Negative Log-Likelihood (GNLL) by default.  The model outputs
(μ, log_σ) for both IPD and PW, so the loss is:

    L = 0.5 * [ log(σ²) + (target - μ)² / σ² ]

This jointly optimises the predicted mean and spread, which is important because
signal variance is strongly context-dependent (e.g., methylated vs. plain sites).

Alternative losses (--loss mse or --loss huber) use only the μ head and ignore
the variance head — useful for ablations or when only mean accuracy matters.

Evaluation metrics (logged per epoch)
--------------------------------------
    MSE (IPD/PW)    — mean squared error in log1p space
    MAE (IPD/PW)    — mean absolute error in log1p space
    Pearson r (IPD) — Pearson correlation between predicted μ_ipd and true IPD
    Pearson r (PW)  — same for PW

All metrics are reported on both a 90 % training split and a 10 % validation split.
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from ...common.dataset import MLPSignalDataset
from .model import MLPPredictor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _gaussian_nll_loss(
    params: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL for (IPD, PW) jointly.

    Args:
        params:  Model output of shape (batch, 4) — [μ_ipd, μ_pw, log_σ_ipd, log_σ_pw].
        targets: Ground-truth signals of shape (batch, 2) — [IPD, PW] in log1p space.

    Returns:
        Scalar loss (mean over batch and signal dimensions).
    """
    mu      = params[:, :2]   # (batch, 2)
    log_sig = params[:, 2:]   # (batch, 2)

    # Clamp to a safe range: prevents log_σ from collapsing to -∞ or exploding
    log_sig = torch.clamp(log_sig, -6.0, 3.0)
    var = torch.exp(2.0 * log_sig)

    # Pointwise Gaussian NLL (constant term dropped as it doesn't affect gradient)
    nll = 0.5 * (log_sig * 2.0 + (targets - mu) ** 2 / var)
    return nll.mean()


def _mse_loss(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain MSE on the mean head only (μ_ipd, μ_pw)."""
    mu = params[:, :2]
    return nn.functional.mse_loss(mu, targets)


def _huber_loss(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Huber (smooth L1) loss on the mean head — less sensitive to outliers."""
    mu = params[:, :2]
    return nn.functional.huber_loss(mu, targets)


_LOSS_FUNCTIONS = {
    "gnll":  _gaussian_nll_loss,
    "mse":   _mse_loss,
    "huber": _huber_loss,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_metrics(
    model: MLPPredictor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute MSE, MAE, Pearson r, and 2σ calibration on a data split.

    Calibration (``calibration_ipd`` / ``calibration_pw``): fraction of actual
    observations that fall within the model's predicted interval [μ − 2σ, μ + 2σ].
    A perfectly calibrated Gaussian gives ~95.4 %.  Values significantly below
    95 % indicate the model underestimates uncertainty; above 95 % indicates
    over-dispersion.

    Args:
        model:  MLPPredictor in eval mode.
        loader: DataLoader for the split.
        device: Torch device.

    Returns:
        Dictionary with keys: mse_ipd, mse_pw, mae_ipd, mae_pw,
                              pearson_ipd, pearson_pw,
                              calibration_ipd, calibration_pw.
    """
    model.eval()
    all_mu    = []
    all_sigma = []
    all_true  = []

    for kmer_ids, meth_probs, signals in loader:
        kmer_ids   = kmer_ids.to(device)
        meth_probs = meth_probs.to(device)
        signals    = signals.to(device)

        params  = model(kmer_ids, meth_probs)
        mu      = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], -6.0, 3.0)
        sigma   = torch.exp(log_sig)

        all_mu.append(mu.cpu())
        all_sigma.append(sigma.cpu())
        all_true.append(signals.cpu())

    all_mu    = torch.cat(all_mu,    dim=0).numpy()   # (N, 2)
    all_sigma = torch.cat(all_sigma, dim=0).numpy()   # (N, 2)
    all_true  = torch.cat(all_true,  dim=0).numpy()   # (N, 2)

    diff  = all_mu - all_true
    mse   = (diff ** 2).mean(axis=0)   # [mse_ipd, mse_pw]
    mae   = np.abs(diff).mean(axis=0)  # [mae_ipd, mae_pw]

    # Calibration: fraction of real observations within predicted [μ-2σ, μ+2σ]
    in_2sigma = (np.abs(diff) <= 2.0 * all_sigma).mean(axis=0)

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "mse_ipd":          float(mse[0]),
        "mse_pw":           float(mse[1]),
        "mae_ipd":          float(mae[0]),
        "mae_pw":           float(mae[1]),
        "pearson_ipd":      _pearson(all_mu[:, 0], all_true[:, 0]),
        "pearson_pw":       _pearson(all_mu[:, 1], all_true[:, 1]),
        "calibration_ipd":  float(in_2sigma[0]),
        "calibration_pw":   float(in_2sigma[1]),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_mlp(
    pkl_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 4096,
    lr: float = 1e-3,
    kmer_embed_dim: int = 64,
    hidden_dim: int = 128,
    meth_proj_dim: int = 8,
    loss_name: str = "gnll",
    val_fraction: float = 0.10,
    checkpoint_every: int = 10,
    device: str = "cuda",
    resume_ckpt: str | None = None,
) -> None:
    """Train the MLPPredictor on merged raw kinetic samples.

    Args:
        pkl_path:         Path to merged .pkl file (from `kinsim merge`).
        output_dir:       Directory for checkpoints, logs, and model config.
        epochs:           Total number of training epochs.
        batch_size:       Mini-batch size (= number of unique keys per step).
        lr:               Initial learning rate for Adam.
        kmer_embed_dim:   Dimension of the 11-mer embedding (32 or 64).
        hidden_dim:       Width of the two hidden MLP layers.
        meth_proj_dim:    Output dimension of the methylation linear projection.
        loss_name:        Loss function: "gnll" | "mse" | "huber".
        val_fraction:     Fraction of data held out for validation.
        checkpoint_every: Save a checkpoint every N epochs (and at the final epoch).
        device:           "cuda" or "cpu".
        resume_ckpt:      Path to a previous checkpoint to resume from.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = MLPSignalDataset(pkl_path)    # random-shot, dynamic capping

    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val

    # Reproducible split (same split across restarts)
    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=rng).tolist()

    train_loader = DataLoader(
        Subset(dataset, indices[:n_train]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, indices[n_train:]),
        batch_size=batch_size * 4,   # larger batches are fine for eval
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # n_train / n_val = number of unique (kmer, meth) keys in each split.
    # Each step, the DataLoader draws one random signal per key from its pool,
    # so the effective data volume grows over epochs without repeating any fixed order.
    log.info("Keys (unique contexts) — train: %d, val: %d", n_train, n_val)

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    model = MLPPredictor(
        kmer_embed_dim=kmer_embed_dim,
        hidden_dim=hidden_dim,
        meth_proj_dim=meth_proj_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Halve LR when validation loss stops improving for 5 consecutive epochs.
    # verbose=True was removed in PyTorch 2.1+; we track LR changes manually.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    loss_fn = _LOSS_FUNCTIONS[loss_name]

    start_epoch = 0
    if resume_ckpt:
        log.info("Resuming from: %s", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        log.info("Resumed at epoch %d", start_epoch)

    # ------------------------------------------------------------------
    # TensorBoard / CSV logging
    # ------------------------------------------------------------------
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=output_dir / "runs")
        use_tb = True
    except ImportError:
        writer = None
        use_tb = False
        log.warning("TensorBoard not available — writing training_log.csv only.")

    # Persist model config before training so generate.py can reconstruct
    # the exact architecture even if training is interrupted.
    model_config = {
        "kmer_embed_dim": kmer_embed_dim,
        "hidden_dim":     hidden_dim,
        "meth_proj_dim":  meth_proj_dim,
    }
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    csv_path = output_dir / "training_log.csv"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log.info(
        "Starting MLP training — %d epochs, loss=%s, lr=%.2e",
        epochs, loss_name, lr,
    )
    log.info(
        "  batch_size=%d  embed_dim=%d  hidden=%d  meth_proj_dim=%d",
        batch_size, kmer_embed_dim, hidden_dim, meth_proj_dim,
    )

    csv_file = open(csv_path, "a", newline="")
    try:
        csv_writer = csv.writer(csv_file)
        if start_epoch == 0:
            csv_writer.writerow([
                "epoch", "train_loss",
                "val_mse_ipd", "val_mse_pw",
                "val_mae_ipd", "val_mae_pw",
                "val_pearson_ipd", "val_pearson_pw",
                "val_calib_ipd", "val_calib_pw",
                "lr",
            ])

        prev_lr = optimizer.param_groups[0]["lr"]

        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0.0

            for kmer_ids, meth_probs, signals in train_loader:
                kmer_ids   = kmer_ids.to(device)
                meth_probs = meth_probs.to(device)
                signals    = signals.to(device)

                optimizer.zero_grad()
                params = model(kmer_ids, meth_probs)
                loss   = loss_fn(params, signals)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validate and step the LR scheduler
            val_metrics = _compute_metrics(model, val_loader, device)
            scheduler.step(val_metrics["mse_ipd"] + val_metrics["mse_pw"])

            current_lr = optimizer.param_groups[0]["lr"]

            # Log LR reduction manually (verbose=True was removed in PyTorch 2.1+)
            if current_lr < prev_lr:
                log.info("LR reduced: %.2e → %.2e", prev_lr, current_lr)
                prev_lr = current_lr

            log.info(
                "Epoch [%3d/%d]  train_loss=%.4f  "
                "val_mse=(%.4f, %.4f)  pearson=(%.3f, %.3f)  "
                "calib=(%.1f%%, %.1f%%)  lr=%.2e",
                epoch + 1, epochs, avg_train_loss,
                val_metrics["mse_ipd"], val_metrics["mse_pw"],
                val_metrics["pearson_ipd"], val_metrics["pearson_pw"],
                val_metrics["calibration_ipd"] * 100,
                val_metrics["calibration_pw"]  * 100,
                current_lr,
            )

            # TensorBoard
            if use_tb:
                writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"Val/{k}", v, epoch + 1)
                writer.add_scalar("LR", current_lr, epoch + 1)

            # CSV — flush after every epoch so partial results survive a crash
            csv_writer.writerow([
                epoch + 1, avg_train_loss,
                val_metrics["mse_ipd"],        val_metrics["mse_pw"],
                val_metrics["mae_ipd"],        val_metrics["mae_pw"],
                val_metrics["pearson_ipd"],    val_metrics["pearson_pw"],
                val_metrics["calibration_ipd"], val_metrics["calibration_pw"],
                current_lr,
            ])
            csv_file.flush()

            # Checkpoint — include scheduler state for clean resume
            if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
                ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
                torch.save({
                    "epoch":     epoch + 1,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, ckpt_path)
                log.info("Checkpoint saved: %s", ckpt_path)

    finally:
        csv_file.close()
        if use_tb:
            writer.close()

    log.info("Training complete. Outputs in: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse
    from ...config import setup_logging, load_yaml_config

    parser = argparse.ArgumentParser(
        prog="kinsim train --model mlp",
        description=(
            "Train an MLPPredictor for kinetic signal generation.\n\n"
            "Input: a merged .pkl from the shared extraction pipeline:\n"
            "  kinsim extract reads.bam motifs shard.pkl   # repeat per BAM\n"
            "  kinsim merge   shards/    master_data.pkl   # combine all shards\n\n"
            "The .pkl maps (kmer_id, meth_id) -> np.ndarray(N, 2) [IPD, PW].\n"
            "MLP and cGAN share the same data pipeline — no separate extraction.\n\n"
            "All flags may be specified in a YAML config file (--config).\n"
            "Command-line flags override YAML values."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pkl",        nargs="?", default=None,
                        help="Merged training data .pkl file")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Directory for checkpoints and logs")

    parser.add_argument("--config",          default=None,
                        help="YAML config file — all flags can be set here for reproducibility")
    parser.add_argument("--epochs",          type=int,   default=None,  help="Training epochs (default: 50)")
    parser.add_argument("--batch-size",      type=int,   default=None,  help="Batch size (default: 4096)")
    parser.add_argument("--lr",              type=float, default=None,  help="Learning rate (default: 1e-3)")
    parser.add_argument("--kmer-embed-dim",  type=int,   default=None,  help="11-mer embedding dimension (default: 64; use 32 for ~0.5 GB RAM)")
    parser.add_argument("--hidden-dim",      type=int,   default=None,  help="Hidden layer width (default: 128)")
    parser.add_argument("--meth-proj-dim",   type=int,   default=None,  help="Methylation linear projection output dim (default: 8)")
    parser.add_argument("--loss",            default=None,
                        choices=["gnll", "mse", "huber"],
                        help="Loss function: gnll=Gaussian NLL (default), mse, huber")
    parser.add_argument("--val-fraction",    type=float, default=None,  help="Fraction for validation split (default: 0.10)")
    parser.add_argument("--checkpoint-every",type=int,   default=None,  help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--device",          default=None,
                        choices=["cuda", "cpu"],
                        help="Device (default: cuda, falls back to cpu automatically)")
    parser.add_argument("--resume",          dest="resume_ckpt",        help="Resume from a previous checkpoint .pt file")
    parser.add_argument("--verbose", "-v",   action="store_true",       help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    # ---- Merge YAML config (if provided) with CLI flags ----
    # Precedence: CLI flags > YAML > hard-coded defaults
    cfg: dict = {}
    if args.config:
        cfg = load_yaml_config(args.config)

    def _get(cli_val, key, default):
        """Return CLI value if given, else YAML value, else default."""
        if cli_val is not None:
            return cli_val
        return cfg.get(key, default)

    pkl_path   = args.pkl        or cfg.get("pkl")
    output_dir = args.output_dir or cfg.get("output_dir")

    if not pkl_path:
        parser.error("pkl is required (positional arg or 'pkl' in YAML config)")
    if not output_dir:
        parser.error("output_dir is required (positional arg or 'output_dir' in YAML config)")

    train_mlp(
        pkl_path         = pkl_path,
        output_dir       = output_dir,
        epochs           = _get(args.epochs,          "epochs",          50),
        batch_size       = _get(args.batch_size,      "batch_size",      4096),
        lr               = _get(args.lr,              "lr",              1e-3),
        kmer_embed_dim   = _get(args.kmer_embed_dim,  "kmer_embed_dim",  64),
        hidden_dim       = _get(args.hidden_dim,      "hidden_dim",      128),
        meth_proj_dim    = _get(args.meth_proj_dim,   "meth_proj_dim",   8),
        loss_name        = _get(args.loss,            "loss",            "gnll"),
        val_fraction     = _get(args.val_fraction,    "val_fraction",    0.10),
        checkpoint_every = _get(args.checkpoint_every,"checkpoint_every",10),
        device           = _get(args.device,          "device",          "cuda"),
        resume_ckpt      = args.resume_ckpt or cfg.get("resume"),
    )


if __name__ == "__main__":
    main()
