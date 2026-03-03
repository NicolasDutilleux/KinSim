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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from ...common.dataset import MLPSignalDataset
from .model import MLPPredictor


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
    """Compute MSE, MAE and Pearson r on a data split.

    Args:
        model:  MLPPredictor in eval mode.
        loader: DataLoader for the split.
        device: Torch device.

    Returns:
        Dictionary with keys: mse_ipd, mse_pw, mae_ipd, mae_pw,
                              pearson_ipd, pearson_pw.
    """
    model.eval()
    all_mu   = []
    all_true = []

    for kmer_ids, meth_probs, signals in loader:
        kmer_ids   = kmer_ids.to(device)
        meth_probs = meth_probs.to(device)
        signals    = signals.to(device)

        params = model(kmer_ids, meth_probs)
        mu     = params[:, :2]

        all_mu.append(mu.cpu())
        all_true.append(signals.cpu())

    all_mu   = torch.cat(all_mu,   dim=0).numpy()   # (N, 2)
    all_true = torch.cat(all_true, dim=0).numpy()   # (N, 2)

    diff  = all_mu - all_true
    mse   = (diff ** 2).mean(axis=0)   # [mse_ipd, mse_pw]
    mae   = np.abs(diff).mean(axis=0)  # [mae_ipd, mae_pw]

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "mse_ipd":    float(mse[0]),
        "mse_pw":     float(mse[1]),
        "mae_ipd":    float(mae[0]),
        "mae_pw":     float(mae[1]),
        "pearson_ipd": _pearson(all_mu[:, 0], all_true[:, 0]),
        "pearson_pw":  _pearson(all_mu[:, 1], all_true[:, 1]),
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
    print(f"Device: {device}")

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
    print(f"Keys (unique contexts) — train: {n_train:,}, val: {n_val:,}")

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
        print(f"Resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        print(f"  Resumed at epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=output_dir / "runs")
        use_tb = True
    except ImportError:
        writer = None
        use_tb = False
        print("TensorBoard not available, writing training_log.csv instead.")

    # Persist model config before training so generate.py can reconstruct
    # the exact architecture even if training is interrupted.
    config = {
        "kmer_embed_dim": kmer_embed_dim,
        "hidden_dim":     hidden_dim,
        "meth_proj_dim":  meth_proj_dim,
    }
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    csv_path = output_dir / "training_log.csv"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting MLP training — {epochs} epochs, loss={loss_name}, lr={lr}")
    print(f"  batch_size={batch_size}, embed_dim={kmer_embed_dim}, "
          f"hidden={hidden_dim}, meth_proj_dim={meth_proj_dim}")

    csv_file = open(csv_path, "a", newline="")
    try:
        csv_writer = csv.writer(csv_file)
        if start_epoch == 0:
            csv_writer.writerow([
                "epoch", "train_loss",
                "val_mse_ipd", "val_mse_pw",
                "val_mae_ipd", "val_mae_pw",
                "val_pearson_ipd", "val_pearson_pw",
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

            # Print LR reduction manually (verbose=True was removed in PyTorch 2.1+)
            if current_lr < prev_lr:
                print(f"  LR reduced: {prev_lr:.2e} → {current_lr:.2e}")
                prev_lr = current_lr

            print(
                f"Epoch [{epoch + 1:>3}/{epochs}]  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_mse=({val_metrics['mse_ipd']:.4f}, {val_metrics['mse_pw']:.4f})  "
                f"pearson=({val_metrics['pearson_ipd']:.3f}, {val_metrics['pearson_pw']:.3f})  "
                f"lr={current_lr:.2e}"
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
                val_metrics["mse_ipd"], val_metrics["mse_pw"],
                val_metrics["mae_ipd"], val_metrics["mae_pw"],
                val_metrics["pearson_ipd"], val_metrics["pearson_pw"],
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
                print(f"  Checkpoint saved: {ckpt_path}")

    finally:
        csv_file.close()
        if use_tb:
            writer.close()

    print(f"\nTraining complete. Outputs in: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim mlp train",
        description=(
            "Train an MLPPredictor for kinetic signal generation.\n\n"
            "Input: a merged .pkl from the shared extraction pipeline:\n"
            "  kinsim extract reads.bam motifs shard.pkl   # repeat per BAM\n"
            "  kinsim merge   shards/    master_data.pkl   # combine all shards\n\n"
            "The .pkl maps (kmer_id, meth_id) -> np.ndarray(N, 2) [IPD, PW].\n"
            "MLP and cGAN share the same data pipeline — no separate extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pkl",        help="Merged training data .pkl file")
    parser.add_argument("output_dir", help="Directory for checkpoints and logs")

    parser.add_argument("--epochs",          type=int,   default=50,    help="Training epochs (default: 50)")
    parser.add_argument("--batch-size",      type=int,   default=4096,  help="Batch size (default: 4096)")
    parser.add_argument("--lr",              type=float, default=1e-3,  help="Learning rate (default: 1e-3)")
    parser.add_argument("--kmer-embed-dim",  type=int,   default=64,    help="11-mer embedding dimension (default: 64; use 32 for ~0.5 GB RAM)")
    parser.add_argument("--hidden-dim",      type=int,   default=128,   help="Hidden layer width (default: 128)")
    parser.add_argument("--meth-proj-dim",   type=int,   default=8,     help="Methylation linear projection output dim (default: 8)")
    parser.add_argument("--loss",            default="gnll",
                        choices=["gnll", "mse", "huber"],
                        help="Loss function: gnll=Gaussian NLL (default), mse, huber")
    parser.add_argument("--val-fraction",    type=float, default=0.10,  help="Fraction for validation split (default: 0.10)")
    parser.add_argument("--checkpoint-every",type=int,   default=10,    help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--device",          default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device (default: cuda, falls back to cpu automatically)")
    parser.add_argument("--resume",          dest="resume_ckpt",        help="Resume from a previous checkpoint .pt file")

    args = parser.parse_args(argv)

    train_mlp(
        pkl_path         = args.pkl,
        output_dir       = args.output_dir,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        kmer_embed_dim   = args.kmer_embed_dim,
        hidden_dim       = args.hidden_dim,
        meth_proj_dim    = args.meth_proj_dim,
        loss_name        = args.loss,
        val_fraction     = args.val_fraction,
        checkpoint_every = args.checkpoint_every,
        device           = args.device,
        resume_ckpt      = args.resume_ckpt,
    )


if __name__ == "__main__":
    main()
