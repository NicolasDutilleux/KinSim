"""Train conditional GAN for kinetic signal generation using WGAN-GP.

Dataset: Flattened {(kmer, meth): np.ndarray(N, 2)} from parse_train.py
Training: WGAN-GP with gradient penalty, TensorBoard logging, checkpointing
"""

import os
import sys
import json
import pickle
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .model import Generator, Discriminator, log_transform


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KmerSignalDataset(Dataset):
    """Dataset of raw (IPD, PW) samples keyed by (kmer_id, meth_id).

    Loads the output of `kinsim cgan merge` and flattens into tensors.
    Signals are log-transformed at load time.
    """

    def __init__(self, pkl_path):
        """Load and flatten the training data.

        Args:
            pkl_path: Path to merged *_cgan.pkl file with structure:
                dict[(kmer_id, meth_id)] -> np.ndarray(N, 2)
        """
        print(f"Loading training data from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)

        kmer_ids = []
        meth_ids = []
        signals = []

        for (kmer_id, meth_id), samples in data_dict.items():
            n = len(samples)
            kmer_ids.extend([kmer_id] * n)
            meth_ids.extend([meth_id] * n)
            signals.append(samples)

        # Concatenate all samples
        self.kmer_ids = torch.tensor(kmer_ids, dtype=torch.long)
        self.meth_ids = torch.tensor(meth_ids, dtype=torch.long)
        self.signals = torch.from_numpy(np.concatenate(signals, axis=0)).float()

        # Log-transform signals
        self.signals = log_transform(self.signals)

        print(f"Dataset loaded: {len(self)} samples, "
              f"{len(data_dict)} unique (kmer, meth) contexts")

    def __len__(self):
        return len(self.kmer_ids)

    def __getitem__(self, idx):
        return self.kmer_ids[idx], self.meth_ids[idx], self.signals[idx]


# ---------------------------------------------------------------------------
# Gradient Penalty
# ---------------------------------------------------------------------------

def compute_gradient_penalty(D, real_signals, fake_signals, kmer_ids, meth_ids, device):
    """Compute WGAN-GP gradient penalty on interpolated samples.

    Args:
        D: Discriminator model
        real_signals: Real signal tensor (batch_size, 2) in log1p space
        fake_signals: Fake signal tensor (batch_size, 2) in log1p space
        kmer_ids: Kmer IDs (batch_size,)
        meth_ids: Methylation IDs (batch_size,)
        device: torch device

    Returns:
        Gradient penalty scalar
    """
    batch_size = real_signals.size(0)

    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = alpha * real_signals + (1 - alpha) * fake_signals
    interpolated.requires_grad_(True)

    # Discriminator score on interpolated samples
    d_interpolated = D(interpolated, kmer_ids, meth_ids)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Gradient penalty: (||grad|| - 1)^2
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()

    return penalty


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_cgan(
    pkl_path,
    output_dir,
    epochs=100,
    batch_size=4096,
    lr_g=1e-4,
    lr_d=1e-4,
    noise_dim=32,
    kmer_embed_dim=64,
    n_critic=5,
    lambda_gp=10,
    device='cuda',
    resume_ckpt=None
):
    """Train conditional GAN with WGAN-GP loss.

    Args:
        pkl_path: Path to merged training data .pkl
        output_dir: Directory for checkpoints and logs
        epochs: Number of training epochs
        batch_size: Training batch size
        lr_g: Generator learning rate
        lr_d: Discriminator learning rate
        noise_dim: Dimension of input noise vector
        kmer_embed_dim: Dimension of kmer embedding
        n_critic: Number of discriminator updates per generator update
        lambda_gp: Gradient penalty weight
        device: 'cuda' or 'cpu'
        resume_ckpt: Path to checkpoint to resume from
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = KmerSignalDataset(pkl_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Models
    G = Generator(noise_dim=noise_dim, kmer_embed_dim=kmer_embed_dim).to(device)
    D = Discriminator(kmer_embed_dim=kmer_embed_dim).to(device)

    # Optimizers
    opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.9))
    opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.9))

    start_epoch = 0
    global_step = 0

    # Resume from checkpoint if provided
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        G.load_state_dict(ckpt['generator'])
        D.load_state_dict(ckpt['discriminator'])
        opt_g.load_state_dict(ckpt['opt_g'])
        opt_d.load_state_dict(ckpt['opt_d'])
        start_epoch = ckpt['epoch']
        global_step = ckpt.get('step', 0)
        print(f"  Resumed from epoch {start_epoch}, step {global_step}")

    # TensorBoard logging (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=output_dir / "runs")
        use_tensorboard = True
    except ImportError:
        print("TensorBoard not available, using CSV logging only")
        writer = None
        use_tensorboard = False

    # CSV logging fallback
    csv_path = output_dir / "training_log.csv"
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if start_epoch == 0:
        csv_writer.writerow(['epoch', 'step', 'd_loss', 'g_loss', 'gp'])
    csv_file.flush()

    # Save model config
    config_path = output_dir / "model_config.json"
    config = {
        'noise_dim': noise_dim,
        'kmer_embed_dim': kmer_embed_dim,
        'hidden_dim': 128,
        'n_critic': n_critic,
        'lambda_gp': lambda_gp
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved to {config_path}")

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  n_critic: {n_critic}, lambda_gp: {lambda_gp}")
    print(f"  LR (G/D): {lr_g}/{lr_d}")

    for epoch in range(start_epoch, epochs):
        G.train()
        D.train()

        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_gp = 0
        n_batches = 0

        for batch_idx, (kmer_ids, meth_ids, real_signals) in enumerate(dataloader):
            kmer_ids = kmer_ids.to(device)
            meth_ids = meth_ids.to(device)
            real_signals = real_signals.to(device)
            batch_size_actual = real_signals.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_d.zero_grad()

            # Real samples
            d_real = D(real_signals, kmer_ids, meth_ids)

            # Fake samples
            z = torch.randn(batch_size_actual, noise_dim, device=device)
            fake_signals = G(z, kmer_ids, meth_ids)
            d_fake = D(fake_signals.detach(), kmer_ids, meth_ids)

            # Gradient penalty
            gp = compute_gradient_penalty(D, real_signals, fake_signals, kmer_ids, meth_ids, device)

            # WGAN-GP loss: -E[D(real)] + E[D(fake)] + lambda * GP
            d_loss = -d_real.mean() + d_fake.mean() + lambda_gp * gp

            d_loss.backward()
            opt_d.step()

            epoch_d_loss += d_loss.item()
            epoch_gp += gp.item()

            # ---------------------
            # Train Generator
            # ---------------------
            if (batch_idx + 1) % n_critic == 0:
                opt_g.zero_grad()

                z = torch.randn(batch_size_actual, noise_dim, device=device)
                fake_signals = G(z, kmer_ids, meth_ids)
                d_fake = D(fake_signals, kmer_ids, meth_ids)

                # Generator loss: -E[D(fake)]
                g_loss = -d_fake.mean()

                g_loss.backward()
                opt_g.step()

                epoch_g_loss += g_loss.item()

                # Logging
                if use_tensorboard:
                    writer.add_scalar('Loss/D', d_loss.item(), global_step)
                    writer.add_scalar('Loss/G', g_loss.item(), global_step)
                    writer.add_scalar('GP', gp.item(), global_step)

                csv_writer.writerow([epoch + 1, global_step, d_loss.item(), g_loss.item(), gp.item()])
                csv_file.flush()

                global_step += 1

            n_batches += 1

        # Epoch summary
        avg_d_loss = epoch_d_loss / n_batches
        g_steps = n_batches // n_critic
        avg_g_loss = epoch_g_loss / g_steps if g_steps > 0 else 0.0
        avg_gp = epoch_gp / n_batches

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, GP: {avg_gp:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'step': global_step,
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    csv_file.close()
    if use_tensorboard:
        writer.close()

    print(f"\nTraining complete. Checkpoints and logs saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim cgan train",
        description="Train a conditional GAN (WGAN-GP) for kinetic signal generation. "
                    "Requires merged training data from 'kinsim cgan merge'.",
    )
    parser.add_argument("pkl", help="Merged training data .pkl file")
    parser.add_argument("output_dir", help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Training batch size (default: 4096)")
    parser.add_argument("--lr-g", type=float, default=1e-4,
                        help="Generator learning rate (default: 1e-4)")
    parser.add_argument("--lr-d", type=float, default=1e-4,
                        help="Discriminator learning rate (default: 1e-4)")
    parser.add_argument("--noise-dim", type=int, default=32,
                        help="Noise vector dimension (default: 32)")
    parser.add_argument("--kmer-embed-dim", type=int, default=64,
                        help="Kmer embedding dimension (default: 64, use 32 for ~0.5 GB RAM)")
    parser.add_argument("--n-critic", type=int, default=5,
                        help="Discriminator updates per generator update (default: 5)")
    parser.add_argument("--lambda-gp", type=float, default=10,
                        help="Gradient penalty weight (default: 10)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--resume", dest="resume_ckpt",
                        help="Resume training from checkpoint .pt file")

    args = parser.parse_args(argv)

    train_cgan(
        pkl_path=args.pkl,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        noise_dim=args.noise_dim,
        kmer_embed_dim=args.kmer_embed_dim,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        device=args.device,
        resume_ckpt=args.resume_ckpt
    )


if __name__ == "__main__":
    main()
