"""Train the MLP predictor with PyTorch Lightning + optional Optuna HPO.

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

    L = 0.5 * [ 2·log_σ + (target − μ)² / exp(2·log_σ) ]

This jointly optimises the predicted mean and spread, which is important because
signal variance is strongly context-dependent (e.g., methylated vs. plain sites).

Alternative losses (--loss mse or --loss huber) use only the μ head and ignore
the variance head — useful for ablations or when only mean accuracy matters.

Evaluation metrics (logged per epoch)
--------------------------------------
    MSE (IPD/PW)         — mean squared error in log1p space
    MAE (IPD/PW)         — mean absolute error in log1p space
    Pearson r (IPD/PW)   — correlation between predicted μ and true signal
    2σ calibration       — fraction of observations within [μ−2σ, μ+2σ] (~95.4 %)

All metrics are reported on the 10 % validation split.

Hyperparameter optimisation (Optuna)
-------------------------------------
Pass --optuna to run a search before the final training run:

    kinsim train --model mlp master.pkl ckpts/ --optuna --n-trials 20

Optuna searches over lr (log-uniform 1e-4..1e-2), kmer_embed_dim (32, 64),
hidden_dim (128, 256, 512).  Best values override CLI/YAML defaults.

Backward compatibility
----------------------
In addition to Lightning .ckpt files, this module writes checkpoint_epoch{N}.pt
files in the legacy format so that kinsim mlp generate and kinsim mlp evaluate
continue to work unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.callbacks.callback import Callback
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
except ImportError:
    try:
        import pytorch_lightning as L
        from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    except ImportError as exc:
        raise ImportError(
            "PyTorch Lightning is required for MLP training.\n"
            "Install with: pip install lightning"
        ) from exc

from ...common.dataset import MLPSignalDataset
from .model import MLPPredictor, ConvPredictor, create_from_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _gaussian_nll_loss(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL for (IPD, PW) jointly.

    Args:
        params:  Model output (batch, 4) — [μ_ipd, μ_pw, log_σ_ipd, log_σ_pw].
        targets: Ground-truth signals (batch, 2) — [IPD, PW] in log1p space.
    """
    mu      = params[:, :2]
    log_sig = torch.clamp(params[:, 2:], -6.0, 3.0)
    var     = torch.exp(2.0 * log_sig)
    return (0.5 * (log_sig * 2.0 + (targets - mu) ** 2 / var)).mean()


def _mse_loss(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain MSE on the mean head only (μ_ipd, μ_pw)."""
    return nn.functional.mse_loss(params[:, :2], targets)


def _huber_loss(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Huber (smooth L1) loss on the mean head — less sensitive to outliers."""
    return nn.functional.huber_loss(params[:, :2], targets)


_LOSS_FUNCTIONS = {
    "gnll":  _gaussian_nll_loss,
    "mse":   _mse_loss,
    "huber": _huber_loss,
}


# ---------------------------------------------------------------------------
# KineticDataModule
# ---------------------------------------------------------------------------

class KineticDataModule(L.LightningDataModule):
    """LightningDataModule for KinSim kinetic .pkl files.

    Wraps MLPSignalDataset with a reproducible train/val split.
    MLPSignalDataset draws one random (IPD, PW) per unique (kmer, meth) key
    per __getitem__ call, so effective data volume grows across epochs without
    storing all samples in RAM.

    Args:
        pkl_path:     Path to merged .pkl file (from kinsim merge).
        val_fraction: Fraction of unique (kmer, meth) keys for validation.
        batch_size:   Training DataLoader batch size.
        seed:         Random seed for reproducible train/val split.
    """

    def __init__(
        self,
        pkl_path: str,
        val_fraction: float = 0.10,
        batch_size: int = 4096,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.pkl_path     = pkl_path
        self.val_fraction = val_fraction
        self.batch_size   = batch_size
        self.seed         = seed
        self._train_subset = None
        self._val_subset   = None

    def setup(self, stage: str | None = None) -> None:
        dataset = MLPSignalDataset(self.pkl_path)
        n_val   = max(1, int(len(dataset) * self.val_fraction))
        n_train = len(dataset) - n_val
        rng     = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(dataset), generator=rng).tolist()
        self._train_subset = Subset(dataset, indices[:n_train])
        self._val_subset   = Subset(dataset, indices[n_train:])
        log.info("Data split — train: %d keys, val: %d keys", n_train, n_val)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_subset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# KineticPredictor (LightningModule)
# ---------------------------------------------------------------------------

class KineticPredictor(L.LightningModule):
    """Lightning module wrapping MLPPredictor.

    Training:
        - Gaussian NLL loss (or MSE/Huber via loss_name).
        - Adam optimiser with ReduceLROnPlateau (patience=5, factor=0.5).

    Validation (per epoch):
        - val_loss: GNLL on the validation split (used by EarlyStopping / scheduler).
        - val_mse_ipd / val_mse_pw: mean squared error in log1p space.
        - val_pearson_ipd / val_pearson_pw: Pearson r between predicted μ and truth.
        - val_calib_ipd / val_calib_pw: 2σ calibration coverage (~95.4 % expected).

    Args:
        model:     MLPPredictor instance (architecture defined externally).
        lr:        Initial Adam learning rate.
        loss_name: "gnll" (default) | "mse" | "huber".
    """

    def __init__(
        self,
        model: MLPPredictor,
        lr: float = 1e-3,
        loss_name: str = "gnll",
    ) -> None:
        super().__init__()
        self.model    = model
        self.lr       = lr
        self._loss_fn = _LOSS_FUNCTIONS[loss_name]
        # Accumulate per-batch val predictions for epoch-level metric computation
        self._val_mu:    list[torch.Tensor] = []
        self._val_sigma: list[torch.Tensor] = []
        self._val_true:  list[torch.Tensor] = []

    def forward(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        return self.model(kmer_ids, meth_probs)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        kmer_ids, meth_probs, signals = batch
        loss = self._loss_fn(self.model(kmer_ids, meth_probs), signals)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        kmer_ids, meth_probs, signals = batch
        params  = self.model(kmer_ids, meth_probs)
        loss    = self._loss_fn(params, signals)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Accumulate for epoch-level Pearson / calibration metrics
        mu      = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], -6.0, 3.0)
        sigma   = torch.exp(log_sig)
        self._val_mu.append(mu.detach().cpu())
        self._val_sigma.append(sigma.detach().cpu())
        self._val_true.append(signals.detach().cpu())
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_mu:
            return

        all_mu    = torch.cat(self._val_mu,    dim=0).numpy()   # (N, 2)
        all_sigma = torch.cat(self._val_sigma, dim=0).numpy()   # (N, 2)
        all_true  = torch.cat(self._val_true,  dim=0).numpy()   # (N, 2)

        diff    = all_mu - all_true
        mse     = (diff ** 2).mean(axis=0)
        mae     = np.abs(diff).mean(axis=0)
        in_2sig = (np.abs(diff) <= 2.0 * all_sigma).mean(axis=0)

        def _pearson(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0

        metrics = {
            "val_mse_ipd":     float(mse[0]),
            "val_mse_pw":      float(mse[1]),
            "val_mae_ipd":     float(mae[0]),
            "val_mae_pw":      float(mae[1]),
            "val_pearson_ipd": _pearson(all_mu[:, 0], all_true[:, 0]),
            "val_pearson_pw":  _pearson(all_mu[:, 1], all_true[:, 1]),
            "val_calib_ipd":   float(in_2sig[0]),
            "val_calib_pw":    float(in_2sig[1]),
        }
        self.log_dict(metrics, on_epoch=True)
        log.info(
            "  val — mse=(%.4f, %.4f)  pearson=(%.3f, %.3f)  calib=(%.1f%%, %.1f%%)",
            metrics["val_mse_ipd"], metrics["val_mse_pw"],
            metrics["val_pearson_ipd"], metrics["val_pearson_pw"],
            metrics["val_calib_ipd"] * 100, metrics["val_calib_pw"] * 100,
        )
        self._val_mu.clear()
        self._val_sigma.clear()
        self._val_true.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val_loss",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def sample(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        """Stochastic inference — delegates to MLPPredictor.sample()."""
        return self.model.sample(kmer_ids, meth_probs)


# ---------------------------------------------------------------------------
# LegacyCheckpointCallback
# ---------------------------------------------------------------------------

class LegacyCheckpointCallback(Callback):
    """Write checkpoint_epoch{N}.pt in the legacy format.

    Allows kinsim mlp generate and kinsim mlp evaluate to work unchanged
    alongside the new Lightning checkpoint infrastructure.  The legacy format
    stores only the MLPPredictor state dict (no 'model.' prefix), matching
    what generate.py and evaluate.py expect:

        {
            "epoch":     int,
            "model":     MLPPredictor.state_dict(),   # no "model." prefix
            "optimizer": ...,
            "scheduler": ...,                         # when available
        }

    Saves every checkpoint_every epochs.  Always saves at the very end of
    training to handle early stopping (which may stop mid-interval).

    Args:
        output_dir:       Directory to write checkpoint_epoch*.pt files.
        checkpoint_every: Save every N epochs (default 10).
    """

    def __init__(self, output_dir: Path, checkpoint_every: int = 10) -> None:
        self.output_dir       = Path(output_dir)
        self.checkpoint_every = checkpoint_every

    def _save(
        self,
        trainer: "L.Trainer",
        pl_module: "KineticPredictor",
        epoch: int,
    ) -> None:
        state: dict = {
            "epoch": epoch,
            "model": pl_module.model.state_dict(),
            "optimizer": trainer.optimizers[0].state_dict(),
        }
        if trainer.lr_scheduler_configs:
            state["scheduler"] = trainer.lr_scheduler_configs[0].scheduler.state_dict()
        path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(state, path)
        log.info("Legacy checkpoint saved: %s", path)

    def on_train_epoch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "KineticPredictor",
    ) -> None:
        # trainer.current_epoch is 0-indexed during on_train_epoch_end
        epoch = trainer.current_epoch + 1
        if epoch % self.checkpoint_every == 0:
            self._save(trainer, pl_module, epoch)

    def on_train_end(
        self,
        trainer: "L.Trainer",
        pl_module: "KineticPredictor",
    ) -> None:
        """Save the final epoch — handles early stopping stopping mid-interval."""
        # After the last on_train_epoch_end, current_epoch is incremented by Lightning.
        # So at on_train_end, current_epoch == number of completed epochs (1-indexed).
        epoch     = trainer.current_epoch
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        if not ckpt_path.exists():
            self._save(trainer, pl_module, epoch)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_model_config(output_dir: Path, model: nn.Module) -> None:
    """Write model_config.json before training starts.

    generate.py and evaluate.py both require this file to reconstruct the
    model architecture.  Writing it before the first epoch ensures it
    exists even if training is interrupted.
    """
    cfg = model.get_config()
    path = output_dir / "model_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    log.info("Model config saved: %s  (architecture=%s)", path, cfg.get("architecture"))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(
    trial,
    pkl_path: str,
    output_dir: Path,
    architecture: str = "conv",
    optuna_epochs: int = 20,
    batch_size: int = 4096,
    val_fraction: float = 0.10,
    loss_name: str = "gnll",
    device: str = "cuda",
) -> float:
    """Optuna objective — returns best val_loss (GNLL) for a trial.

    Search space depends on architecture:
        conv: lr, base_embed_dim, conv_dim, head_dim, kernel_size, dropout
        mlp:  lr, kmer_embed_dim, hidden_dim, dropout

    Args:
        trial:         Optuna Trial object.
        pkl_path:      Merged .pkl file path.
        output_dir:    Root dir for trial subdirectories.
        architecture:  "conv" (default) or "mlp".
        optuna_epochs: Max epochs per trial (shorter than final run).
        batch_size:    DataLoader batch size.
        val_fraction:  Fraction of keys for validation.
        loss_name:     Loss function.
        device:        "cuda" or "cpu".

    Returns:
        Best val_loss seen during this trial (lower is better).
    """
    lr      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    if architecture == "conv":
        base_embed_dim = trial.suggest_categorical("base_embed_dim", [8, 16])
        conv_dim       = trial.suggest_categorical("conv_dim", [64, 128])
        head_dim       = trial.suggest_categorical("head_dim", [64, 128, 256])
        kernel_size    = trial.suggest_categorical("kernel_size", [3, 5])
        model = ConvPredictor(
            base_embed_dim=base_embed_dim,
            conv_dim=conv_dim,
            head_dim=head_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    else:
        kmer_embed_dim = trial.suggest_categorical("kmer_embed_dim", [32, 64])
        hidden_dim     = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        model = MLPPredictor(
            kmer_embed_dim=kmer_embed_dim,
            hidden_dim=hidden_dim,
            meth_proj_dim=8,
            dropout=dropout,
        )

    lm = KineticPredictor(model, lr=lr, loss_name=loss_name)
    dm = KineticDataModule(
        pkl_path=pkl_path,
        val_fraction=val_fraction,
        batch_size=batch_size,
    )

    callbacks: list = [EarlyStopping(monitor="val_loss", patience=5, mode="min")]
    try:
        from optuna.integration import PyTorchLightningPruningCallback
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))
    except ImportError:
        pass  # Pruning skipped — optuna.integration not available

    trial_dir = output_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if device == "cuda" and torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=optuna_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.5,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=CSVLogger(str(trial_dir)),
        callbacks=callbacks,
        log_every_n_steps=1,
    )
    trainer.fit(lm, datamodule=dm)

    return float(trainer.callback_metrics.get("val_loss", float("inf")))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_mlp(
    pkl_path: str,
    output_dir: str,
    architecture: str = "conv",
    epochs: int = 50,
    batch_size: int = 4096,
    lr: float = 1e-3,
    # Conv architecture params
    base_embed_dim: int = 16,
    conv_dim: int = 128,
    n_conv_layers: int = 3,
    kernel_size: int = 3,
    head_dim: int = 128,
    # MLP architecture params (legacy)
    kmer_embed_dim: int = 64,
    hidden_dim: int = 128,
    # Shared params
    meth_proj_dim: int = 8,
    dropout: float = 0.1,
    loss_name: str = "gnll",
    val_fraction: float = 0.10,
    checkpoint_every: int = 10,
    device: str = "cuda",
    resume_ckpt: str | None = None,
    run_optuna: bool = False,
    n_trials: int = 20,
    optuna_epochs: int = 20,
) -> None:
    """Train kinetic predictor using PyTorch Lightning.

    Supports two architectures:
        "conv" (default): ConvPredictor — per-base embeddings + 1D conv + FiLM.
                          ~140K params.  Learns compositional spatial rules.
        "mlp"  (legacy):  MLPPredictor — flat 4.2M k-mer embedding + MLP.
                          ~268M params.  Fast lookup, but memorises each 11-mer.

    If run_optuna=True, an Optuna HPO study runs first.  Best values override
    defaults for the final training run.

    Args:
        pkl_path:         Merged .pkl file from kinsim merge.
        output_dir:       Directory for checkpoints, logs, and model config.
        architecture:     "conv" (default) or "mlp".
        epochs:           Total training epochs for the final run.
        batch_size:       Mini-batch size (unique (kmer, meth) keys per step).
        lr:               Initial Adam learning rate.
        base_embed_dim:   [conv] Per-base embedding dimension (default 16).
        conv_dim:         [conv] Conv channel width (default 128).
        n_conv_layers:    [conv] Number of conv layers (default 3).
        kernel_size:      [conv] Conv kernel size (default 3).
        head_dim:         [conv] Head hidden layer width (default 128).
        kmer_embed_dim:   [mlp] 11-mer embedding dimension (32 or 64).
        hidden_dim:       [mlp] Hidden layer width.
        meth_proj_dim:    Methylation projection output dimension.
        dropout:          Dropout probability (default 0.1 for conv, 0.0 for mlp).
        loss_name:        "gnll" (default) | "mse" | "huber".
        val_fraction:     Fraction of keys reserved for validation.
        checkpoint_every: Save legacy checkpoint_epoch*.pt every N epochs.
        device:           "cuda" or "cpu".
        resume_ckpt:      Legacy .pt or Lightning .ckpt to load weights from.
        run_optuna:       Run Optuna HPO before the final training run.
        n_trials:         Number of Optuna trials.
        optuna_epochs:    Epochs per trial (typically much shorter than epochs).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Optuna HPO ────────────────────────────────────────────────────────
    if run_optuna:
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for HPO. Install with: pip install optuna"
            )

        log.info("Optuna HPO — arch=%s  %d trials × %d epochs",
                 architecture, n_trials, optuna_epochs)
        optuna_dir = output_dir / "optuna"
        optuna_dir.mkdir(exist_ok=True)

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            study_name="kinsim_mlp",
        )
        study.optimize(
            lambda trial: objective(
                trial,
                pkl_path=pkl_path,
                output_dir=optuna_dir,
                architecture=architecture,
                optuna_epochs=optuna_epochs,
                batch_size=batch_size,
                val_fraction=val_fraction,
                loss_name=loss_name,
                device=device,
            ),
            n_trials=n_trials,
        )

        best = study.best_params
        log.info("Optuna best — val_loss=%.6f  params=%s", study.best_value, best)
        # Override with Optuna's best hyperparameters
        lr      = best["lr"]
        dropout = best.get("dropout", dropout)
        if architecture == "conv":
            base_embed_dim = best.get("base_embed_dim", base_embed_dim)
            conv_dim       = best.get("conv_dim", conv_dim)
            head_dim       = best.get("head_dim", head_dim)
            kernel_size    = best.get("kernel_size", kernel_size)
        else:
            kmer_embed_dim = best.get("kmer_embed_dim", kmer_embed_dim)
            hidden_dim     = best.get("hidden_dim", hidden_dim)
        (output_dir / "optuna_best_params.json").write_text(
            json.dumps({"best_val_loss": study.best_value, **best}, indent=2)
        )

    # ── Build model ───────────────────────────────────────────────────────
    accelerator = "gpu" if device == "cuda" and torch.cuda.is_available() else "cpu"

    if architecture == "conv":
        log.info(
            "Training — arch=conv  %d epochs  loss=%s  lr=%.2e  base_embed=%d  "
            "conv_dim=%d  n_layers=%d  k=%d  head=%d  meth_proj=%d  dropout=%.2f  accel=%s",
            epochs, loss_name, lr, base_embed_dim, conv_dim, n_conv_layers,
            kernel_size, head_dim, meth_proj_dim, dropout, accelerator,
        )
        model = ConvPredictor(
            base_embed_dim=base_embed_dim,
            meth_proj_dim=meth_proj_dim,
            conv_dim=conv_dim,
            n_conv_layers=n_conv_layers,
            kernel_size=kernel_size,
            head_dim=head_dim,
            dropout=dropout,
        )
    else:
        log.info(
            "Training — arch=mlp  %d epochs  loss=%s  lr=%.2e  embed=%d  hidden=%d  "
            "meth_proj=%d  dropout=%.2f  accel=%s",
            epochs, loss_name, lr, kmer_embed_dim, hidden_dim, meth_proj_dim,
            dropout, accelerator,
        )
        model = MLPPredictor(
            kmer_embed_dim=kmer_embed_dim,
            hidden_dim=hidden_dim,
            meth_proj_dim=meth_proj_dim,
            dropout=dropout,
        )

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s (%s)", f"{n_params:,}", architecture)

    # Save model config BEFORE first epoch — generate.py needs it even if interrupted
    _save_model_config(output_dir, model)

    if resume_ckpt:
        log.info("Loading weights from: %s", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            # Legacy format: direct MLPPredictor state dict (no "model." prefix)
            model.load_state_dict(ckpt["model"])
        elif "state_dict" in ckpt:
            # Lightning format: strip "model." prefix added by KineticPredictor wrapper
            state_dict = {
                k[len("model."):]: v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.")
            }
            model.load_state_dict(state_dict)
        else:
            raise ValueError(
                f"Unrecognized checkpoint format in {resume_ckpt}.\n"
                "Expected 'model' key (legacy) or 'state_dict' key (Lightning)."
            )
        log.info("Weights loaded.")

    lm = KineticPredictor(model, lr=lr, loss_name=loss_name)
    dm = KineticDataModule(
        pkl_path=pkl_path,
        val_fraction=val_fraction,
        batch_size=batch_size,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    lightning_ckpt = ModelCheckpoint(
        dirpath=str(output_dir / "lightning_ckpts"),
        filename="ckpt-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    legacy_ckpt = LegacyCheckpointCallback(
        output_dir=output_dir,
        checkpoint_every=checkpoint_every,
    )

    # ── Loggers ───────────────────────────────────────────────────────────
    loggers: list = [CSVLogger(str(output_dir), name="logs")]
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        loggers.append(TensorBoardLogger(str(output_dir), name="runs"))
    except ImportError:
        log.warning("TensorBoard not available — CSV logger only.")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.5,
        callbacks=[early_stop, lightning_ckpt, legacy_ckpt],
        logger=loggers,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(lm, datamodule=dm)
    log.info("Training complete. Outputs in: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse
    from ...config import load_yaml_config, setup_logging

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
            "Command-line flags override YAML values.\n\n"
            "Optuna HPO:\n"
            "  kinsim train --model mlp master.pkl ckpts/ --optuna --n-trials 20"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pkl",        nargs="?", default=None,
                        help="Merged training data .pkl file")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Directory for checkpoints and logs")

    # Architecture selection
    parser.add_argument("--architecture",     default=None,
                        choices=["conv", "mlp"],
                        help="Model architecture: conv (default, 1D-conv + FiLM) or mlp (legacy embedding)")

    # Training hyperparameters
    parser.add_argument("--config",           default=None,
                        help="YAML config file (all flags can be set here)")
    parser.add_argument("--epochs",           type=int,   default=None,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch-size",       type=int,   default=None,
                        help="Batch size (default: 4096)")
    parser.add_argument("--lr",               type=float, default=None,
                        help="Learning rate (default: 1e-3)")

    # Conv architecture params
    parser.add_argument("--base-embed-dim",   type=int,   default=None,
                        help="[conv] Per-base embedding dimension (default: 16)")
    parser.add_argument("--conv-dim",         type=int,   default=None,
                        help="[conv] Conv channel width (default: 128)")
    parser.add_argument("--n-conv-layers",    type=int,   default=None,
                        help="[conv] Number of conv layers (default: 3)")
    parser.add_argument("--kernel-size",      type=int,   default=None,
                        help="[conv] Conv kernel size (default: 3)")
    parser.add_argument("--head-dim",         type=int,   default=None,
                        help="[conv] Head hidden layer width (default: 128)")

    # MLP architecture params (legacy)
    parser.add_argument("--kmer-embed-dim",   type=int,   default=None,
                        help="[mlp] 11-mer embedding dimension (default: 64)")
    parser.add_argument("--hidden-dim",       type=int,   default=None,
                        help="[mlp] Hidden layer width (default: 128)")

    # Shared params
    parser.add_argument("--meth-proj-dim",    type=int,   default=None,
                        help="Methylation projection output dim (default: 8)")
    parser.add_argument("--dropout",          type=float, default=None,
                        help="Dropout probability (default: 0.1 for conv, 0.0 for mlp)")
    parser.add_argument("--loss",             default=None,
                        choices=["gnll", "mse", "huber"],
                        help="Loss function: gnll=Gaussian NLL (default), mse, huber")
    parser.add_argument("--val-fraction",     type=float, default=None,
                        help="Fraction for validation split (default: 0.10)")
    parser.add_argument("--checkpoint-every", type=int,   default=None,
                        help="Save legacy checkpoint every N epochs (default: 10)")
    parser.add_argument("--device",           default=None,
                        choices=["cuda", "cpu"],
                        help="Device (default: cuda, falls back to cpu automatically)")
    parser.add_argument("--resume",           dest="resume_ckpt",
                        help="Resume weights from a checkpoint .pt or .ckpt file")

    # Optuna HPO flags
    parser.add_argument("--optuna",           action="store_true",
                        help="Run Optuna HPO before the final training run")
    parser.add_argument("--n-trials",         type=int,   default=None,
                        help="Number of Optuna trials (default: 20)")
    parser.add_argument("--optuna-epochs",    type=int,   default=None,
                        help="Epochs per Optuna trial (default: 20, shorter than --epochs)")

    parser.add_argument("--verbose", "-v",    action="store_true",
                        help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    # Merge YAML config with CLI flags — precedence: CLI > YAML > hard-coded defaults
    cfg: dict = {}
    if args.config:
        cfg = load_yaml_config(args.config)

    def _get(cli_val, key, default):
        return cli_val if cli_val is not None else cfg.get(key, default)

    pkl_path   = args.pkl        or cfg.get("pkl")
    output_dir = args.output_dir or cfg.get("output_dir")

    if not pkl_path:
        parser.error("pkl is required (positional arg or 'pkl' in YAML config)")
    if not output_dir:
        parser.error("output_dir is required (positional arg or 'output_dir' in YAML config)")

    architecture = _get(args.architecture, "architecture", "conv")
    # Default dropout depends on architecture
    default_dropout = 0.1 if architecture == "conv" else 0.0

    train_mlp(
        pkl_path         = pkl_path,
        output_dir       = output_dir,
        architecture     = architecture,
        epochs           = _get(args.epochs,          "epochs",          50),
        batch_size       = _get(args.batch_size,      "batch_size",      4096),
        lr               = _get(args.lr,              "lr",              1e-3),
        # Conv params
        base_embed_dim   = _get(args.base_embed_dim,  "base_embed_dim",  16),
        conv_dim         = _get(args.conv_dim,        "conv_dim",        128),
        n_conv_layers    = _get(args.n_conv_layers,   "n_conv_layers",   3),
        kernel_size      = _get(args.kernel_size,     "kernel_size",     3),
        head_dim         = _get(args.head_dim,        "head_dim",        128),
        # MLP params
        kmer_embed_dim   = _get(args.kmer_embed_dim,  "kmer_embed_dim",  64),
        hidden_dim       = _get(args.hidden_dim,      "hidden_dim",      128),
        # Shared
        meth_proj_dim    = _get(args.meth_proj_dim,   "meth_proj_dim",   8),
        dropout          = _get(args.dropout,         "dropout",         default_dropout),
        loss_name        = _get(args.loss,            "loss",            "gnll"),
        val_fraction     = _get(args.val_fraction,    "val_fraction",    0.10),
        checkpoint_every = _get(args.checkpoint_every,"checkpoint_every",10),
        device           = _get(args.device,          "device",          "cuda"),
        resume_ckpt      = args.resume_ckpt or cfg.get("resume"),
        run_optuna       = args.optuna or cfg.get("optuna", False),
        n_trials         = _get(args.n_trials,        "n_trials",        20),
        optuna_epochs    = _get(args.optuna_epochs,   "optuna_epochs",   20),
    )


if __name__ == "__main__":
    main()
