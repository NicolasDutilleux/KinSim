"""Baseline 3: ConvPredictor without FiLM — post-hoc IPD ratio shift.

Same architecture as ConvPredictor (per-base embedding, positional embedding,
Conv1D backbone, dual readout, Gaussian output head) but with NO methylation
conditioning during the forward pass. The model learns sequence context only.

At generation time, methylation is applied as a post-hoc multiplicative shift
on the predicted mean, using learned IPD/PW ratios per methylation type
(computed from training data during fit).

This isolates the contribution of FiLM: comparing this baseline against the
full ConvPredictor shows whether learned multiplicative conditioning improves
over a simple global ratio shift.

Architecture (identical to ConvPredictor minus FiLM):

    bases (B, 11) int  -> Embedding(4, 16) + pos_embed -> (B, 11, 16)
                           (NO FiLM — methylation not seen)
    Conv1D x3 (k=3, BN, GELU) -> (B, 128, 11)
    Readout: center || global_pool -> (B, 256)
    Head: Linear(256→128) -> LN -> GELU -> Dropout -> Linear(128→4)
    Output: [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw]  (log1p space)

Post-hoc methylation:
    mu_ipd_final = mu_ipd * ipd_ratio[meth_id]
    mu_pw_final  = mu_pw  * pw_ratio[meth_id]
"""

import json
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinsim.data.dataset import inv_log_transform, MLPSignalDataset
from kinsim.utils.encoding import K as _DEFAULT_K, METH_IDS

log = logging.getLogger(__name__)

METH_NAMES = {v: k for k, v in METH_IDS.items()}


# =========================================================================
# Model: ConvPredictor without FiLM
# =========================================================================

class ConvNoFiLMPredictor(nn.Module):
    """ConvPredictor architecture with FiLM removed.

    The model sees only the 11-mer sequence context. Methylation is NOT
    provided as input — it is applied post-hoc as a ratio shift.

    The forward() signature matches ConvPredictor for compatibility:
    meth_probs is accepted but ignored.
    """

    def __init__(
        self,
        base_embed_dim: int = 16,
        conv_dim: int = 128,
        n_conv_layers: int = 3,
        kernel_size: int = 3,
        head_dim: int = 128,
        dropout: float = 0.1,
        kmer_size: int = _DEFAULT_K,
    ):
        super().__init__()

        self.base_embed_dim = base_embed_dim
        self.conv_dim = conv_dim
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.head_dim = head_dim
        self.dropout_p = dropout
        self.kmer_size = kmer_size

        # Per-base embedding
        self.base_embed = nn.Embedding(4, base_embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, kmer_size, base_embed_dim))

        # NO meth_proj, NO film_gamma, NO film_beta

        # Conv1D backbone (identical to ConvPredictor)
        conv_layers = []
        in_ch = base_embed_dim
        for _ in range(n_conv_layers):
            conv_layers.extend([
                nn.Conv1d(in_ch, conv_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(conv_dim),
                nn.GELU(),
            ])
            in_ch = conv_dim
        self.conv = nn.Sequential(*conv_layers)

        # Dual readout
        readout_dim = conv_dim * 2

        # Output head (identical to ConvPredictor)
        self.head = nn.Sequential(
            nn.Linear(readout_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 4),
        )

        # Bit-shift buffer for decoding kmer_ids
        self.register_buffer(
            "_shifts",
            torch.arange(kmer_size - 1, -1, -1) * 2,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _decode_kmer_ids(self, kmer_ids: torch.Tensor) -> torch.Tensor:
        return (kmer_ids.unsqueeze(1) >> self._shifts.unsqueeze(0)) & 3

    def forward(
        self,
        kmer_ids: torch.Tensor,
        meth_probs: torch.Tensor,  # accepted but IGNORED
    ) -> torch.Tensor:
        """Forward pass — sequence context only, no methylation."""
        bases = self._decode_kmer_ids(kmer_ids)

        # Per-base embedding + positional
        x = self.base_embed(bases) + self.pos_embed  # (B, 11, 16)

        # NO FiLM — go straight to conv
        x = x.transpose(1, 2)  # (B, 16, 11)
        x = self.conv(x)       # (B, conv_dim, 11)

        # Dual readout
        center = x[:, :, self.kmer_size // 2]
        global_pool = x.mean(dim=2)
        readout = torch.cat([center, global_pool], dim=1)

        return self.head(readout)  # (B, 4)

    @torch.no_grad()
    def sample(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        params = self.forward(kmer_ids, meth_probs)
        mu = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], -6.0, 3.0)
        sigma = torch.exp(log_sig)
        z = torch.randn_like(mu)
        return inv_log_transform(mu + sigma * z)

    @torch.no_grad()
    def predict_mean(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        params = self.forward(kmer_ids, meth_probs)
        return inv_log_transform(params[:, :2])

    def get_config(self) -> dict:
        return {
            "architecture": "conv_no_film",
            "kmer_size": self.kmer_size,
            "base_embed_dim": self.base_embed_dim,
            "conv_dim": self.conv_dim,
            "n_conv_layers": self.n_conv_layers,
            "kernel_size": self.kernel_size,
            "head_dim": self.head_dim,
            "dropout": self.dropout_p,
        }


# =========================================================================
# Training loop (standalone, no Lightning — keeps it simple for baseline)
# =========================================================================

def _gaussian_nll_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Same Gaussian NLL loss as kinsim/train.py."""
    mu = pred[:, :2]
    log_sigma = torch.clamp(pred[:, 2:], -6.0, 3.0)
    loss = 0.5 * (2.0 * log_sigma + (target - mu) ** 2 / torch.exp(2.0 * log_sigma))
    return loss.mean()


def fit(pkl_path: str, output_dir: str, epochs: int = 50,
        batch_size: int = 4096, lr: float = 1e-3, device_str: str = "auto"):
    """Train ConvNoFiLMPredictor + compute IPD ratios from training data.

    The model is trained on ALL data (unmethylated + methylated) to learn
    the baseline sequence-dependent signal. Then IPD ratios are computed
    by comparing methylated vs unmethylated predictions.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info("Device: %s", device)

    # Load data
    log.info("Loading training data: %s", pkl_path)
    dataset = MLPSignalDataset(pkl_path)
    log.info("Dataset: %d samples", len(dataset))

    # Split 90/10
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Model
    model = ConvNoFiLMPredictor().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("ConvNoFiLMPredictor: %d parameters", n_params)

    # Save config before training
    config = model.get_config()
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    log.info("Training for %d epochs", epochs)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for kmer_ids, meth_probs, targets, _meth_ids in train_loader:
            kmer_ids = kmer_ids.to(device)
            meth_probs = meth_probs.to(device)
            targets = targets.to(device)

            pred = model(kmer_ids, meth_probs)  # meth_probs ignored inside
            loss = _gaussian_nll_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for kmer_ids, meth_probs, targets, _meth_ids in val_loader:
                kmer_ids = kmer_ids.to(device)
                meth_probs = meth_probs.to(device)
                targets = targets.to(device)
                pred = model(kmer_ids, meth_probs)
                val_losses.append(_gaussian_nll_loss(pred, targets).item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        log.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                 epoch + 1, epochs, train_loss, val_loss, current_lr)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, "best_checkpoint.pt"))

        # Save periodic
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, f"checkpoint_epoch{epoch + 1}.pt"))

    log.info("Best val_loss: %.4f", best_val_loss)

    # --- Compute IPD/PW ratios from training data ---
    log.info("Computing post-hoc IPD/PW ratios from training data...")
    model.eval()
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_checkpoint.pt"),
                    map_location=device)['model']
    )

    # Collect per meth_id: model predictions (unmethylated context)
    # vs actual observed values
    meth_ipd_observed = {0: [], 1: [], 2: [], 3: []}
    meth_pw_observed = {0: [], 1: [], 2: [], 3: []}

    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)

    for key, arr in raw_data.items():
        if not isinstance(key, tuple):
            continue
        _kmer_id, meth_id = key
        if not isinstance(arr, np.ndarray) or len(arr) == 0:
            continue
        meth_ipd_observed[meth_id].extend(arr[:, 0].tolist())
        meth_pw_observed[meth_id].extend(arr[:, 1].tolist())

    # Compute ratios
    mean_unmeth_ipd = np.mean(meth_ipd_observed[0]) if meth_ipd_observed[0] else 10.0
    mean_unmeth_pw = np.mean(meth_pw_observed[0]) if meth_pw_observed[0] else 8.0

    ipd_ratios = {0: 1.0}
    pw_ratios = {0: 1.0}
    for meth_id in [1, 2, 3]:
        if meth_ipd_observed[meth_id]:
            ipd_ratios[meth_id] = float(np.mean(meth_ipd_observed[meth_id])) / max(mean_unmeth_ipd, 0.1)
            pw_ratios[meth_id] = float(np.mean(meth_pw_observed[meth_id])) / max(mean_unmeth_pw, 0.1)
        else:
            ipd_ratios[meth_id] = 1.0
            pw_ratios[meth_id] = 1.0

    ratios = {
        'ipd_ratios': {str(k): v for k, v in ipd_ratios.items()},
        'pw_ratios': {str(k): v for k, v in pw_ratios.items()},
        'mean_unmeth_ipd': mean_unmeth_ipd,
        'mean_unmeth_pw': mean_unmeth_pw,
    }
    ratios_path = os.path.join(output_dir, "meth_ratios.json")
    with open(ratios_path, "w") as f:
        json.dump(ratios, f, indent=2)

    log.info("IPD ratios: none=1.0, m6A=%.3f, m4C=%.3f, m5C=%.3f",
             ipd_ratios.get(1, 1.0), ipd_ratios.get(2, 1.0), ipd_ratios.get(3, 1.0))
    log.info("Saved ratios to %s", ratios_path)
    log.info("Done. Model: %s, Ratios: %s", output_dir, ratios_path)


# =========================================================================
# Evaluate: compare predictions with and without ratio shift
# =========================================================================

def evaluate(model_dir: str, test_pkl: str, device_str: str = "auto"):
    """Evaluate ConvNoFiLM + ratio shift on test data."""

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load model
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    model = ConvNoFiLMPredictor(
        base_embed_dim=config.get("base_embed_dim", 16),
        conv_dim=config.get("conv_dim", 128),
        n_conv_layers=config.get("n_conv_layers", 3),
        kernel_size=config.get("kernel_size", 3),
        head_dim=config.get("head_dim", 128),
        dropout=config.get("dropout", 0.1),
    ).to(device)

    ckpt = torch.load(os.path.join(model_dir, "best_checkpoint.pt"), map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Load ratios
    ratios_path = os.path.join(model_dir, "meth_ratios.json")
    with open(ratios_path) as f:
        ratios = json.load(f)
    ipd_ratios = {int(k): v for k, v in ratios['ipd_ratios'].items()}
    pw_ratios = {int(k): v for k, v in ratios['pw_ratios'].items()}

    # Load test data
    dataset = MLPSignalDataset(test_pkl)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4096, shuffle=False, num_workers=2,
    )

    all_pred_ipd = []
    all_pred_pw = []
    all_true_ipd = []
    all_true_pw = []
    all_meth_ids = []

    with torch.no_grad():
        for kmer_ids, meth_probs, targets, meth_ids in loader:
            kmer_ids = kmer_ids.to(device)
            meth_probs = meth_probs.to(device)

            # Model predicts in log space — get mean prediction
            params = model(kmer_ids, meth_probs)
            mu = params[:, :2]
            pred_raw = inv_log_transform(mu).cpu().numpy()

            # Apply post-hoc ratio shift
            meth_np = meth_ids.numpy()
            for i in range(len(pred_raw)):
                mid = int(meth_np[i])
                pred_raw[i, 0] *= ipd_ratios.get(mid, 1.0)
                pred_raw[i, 1] *= pw_ratios.get(mid, 1.0)
                pred_raw[i] = np.clip(pred_raw[i], 0, 255)

            true_raw = inv_log_transform(targets).numpy()

            all_pred_ipd.extend(pred_raw[:, 0].tolist())
            all_pred_pw.extend(pred_raw[:, 1].tolist())
            all_true_ipd.extend(true_raw[:, 0].tolist())
            all_true_pw.extend(true_raw[:, 1].tolist())
            all_meth_ids.extend(meth_np.tolist())

    pred_ipd = np.array(all_pred_ipd)
    pred_pw = np.array(all_pred_pw)
    true_ipd = np.array(all_true_ipd)
    true_pw = np.array(all_true_pw)
    meth_arr = np.array(all_meth_ids, dtype=int)

    # Overall metrics
    from scipy.stats import pearsonr
    r_ipd, _ = pearsonr(pred_ipd, true_ipd)
    r_pw, _ = pearsonr(pred_pw, true_pw)
    mae_ipd = np.mean(np.abs(pred_ipd - true_ipd))
    mae_pw = np.mean(np.abs(pred_pw - true_pw))

    print("\n=== ConvNoFiLM + Ratio Shift Evaluation ===")
    print(f"Test samples: {len(pred_ipd):,}")
    print(f"\n{'Metric':<15} {'IPD':>10} {'PW':>10}")
    print("-" * 38)
    print(f"{'Pearson r':<15} {r_ipd:>10.4f} {r_pw:>10.4f}")
    print(f"{'MAE':<15} {mae_ipd:>10.2f} {mae_pw:>10.2f}")

    # Per meth type
    print(f"\n{'Meth type':<10} {'N':>8} {'r_IPD':>8} {'MAE_IPD':>8} {'r_PW':>8} {'MAE_PW':>8}")
    print("-" * 50)
    for mid in range(4):
        mask = meth_arr == mid
        if mask.sum() < 10:
            continue
        r_i, _ = pearsonr(pred_ipd[mask], true_ipd[mask])
        r_p, _ = pearsonr(pred_pw[mask], true_pw[mask])
        mae_i = np.mean(np.abs(pred_ipd[mask] - true_ipd[mask]))
        mae_p = np.mean(np.abs(pred_pw[mask] - true_pw[mask]))
        name = METH_NAMES.get(mid, f"id={mid}")
        print(f"{name:<10} {mask.sum():>8} {r_i:>8.4f} {mae_i:>8.2f} {r_p:>8.4f} {mae_p:>8.2f}")

    print(f"\nIPD ratios used: {ipd_ratios}")
    print(f"PW ratios used:  {pw_ratios}")
