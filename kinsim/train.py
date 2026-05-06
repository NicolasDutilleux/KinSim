"""``kinsim train`` — train the kinetic predictor on shard pkls.

Input
-----
Either:
  - a directory of ``*_shard.pkl`` files (sharded mode, recommended) — uses
    ``ShardedSignalDataset``, never holds the corpus in RAM
  - a single ``shard.pkl`` (small datasets / debugging) — loads into RAM

Each shard is ``dict[kmer_id (int) → np.ndarray(N, 38)]`` with the column
layout from :mod:`kinsim.utils.sample_layout`. Produced by
``kinsim extract`` and optionally filtered by ``kinsim refine``.

Loss
----
Default is Gaussian Negative Log-Likelihood (GNLL). The model outputs
``(μ, log_σ)`` for both IPD and PW in log1p space, so::

    L = 0.5 * [ 2·log_σ + (target − μ)² / exp(2·log_σ) ]

Alternatives (``--loss mse`` / ``--loss huber``) use only the μ head — for
ablations or when variance modelling isn't needed.

Metrics (per epoch on the validation split)
-------------------------------------------
- MSE / MAE (IPD/PW) — in log1p space
- Pearson r (IPD/PW) — predicted μ vs target
- 2σ calibration — fraction of observations within [μ−2σ, μ+2σ] (~95.4 %)

Train/test split
----------------
- ``--test-strains bc2080,bc2081`` — explicit by-sample-id holdout
- ``--test-fraction 0.10 --split-seed 42`` — random per-shard split
- For single-pkl input — random 90/10 row split

Hyperparameter search
---------------------
``--optuna --n-trials N`` runs an Optuna search over lr / kmer_embed_dim /
hidden_dim before the final training run.

Checkpoints
-----------
Writes ``checkpoint_epoch{N}.pt`` + ``model_config.json`` to the output
directory. ``kinsim generate`` / ``kinsim evaluate`` consume these.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, Subset

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
            "PyTorch Lightning is required for MLP training.\nInstall with: pip install lightning"
        ) from exc

from .data.dataset import (
    MLPSignalDataset,
    ShardedSignalDataset,
    list_shards,
    split_shards,
)
from .models.predictor import ConvPredictor, MLPPredictor

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
    mu = params[:, :2]
    log_sig = torch.clamp(params[:, 2:], -6.0, 3.0)
    var = torch.exp(2.0 * log_sig)
    return (0.5 * (log_sig * 2.0 + (targets - mu) ** 2 / var)).mean()


def _mse_loss(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain MSE on the mean head only (μ_ipd, μ_pw)."""
    return nn.functional.mse_loss(params[:, :2], targets)


def _huber_loss(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Huber (smooth L1) loss on the mean head — less sensitive to outliers."""
    return nn.functional.huber_loss(params[:, :2], targets)


_LOSS_FUNCTIONS = {
    "gnll": _gaussian_nll_loss,
    "mse": _mse_loss,
    "huber": _huber_loss,
}

def _meth_names() -> dict[int, str]:
    """Return ``{meth_id: meth_name}`` derived from kinsim_config.yaml.

    Lazy: rebuilt on each call so changes to the YAML during a long
    process (rare) take effect. Used only for human-readable metric
    labels in :func:`_compute_metrics` — not for ID encoding.
    """
    from .utils.encoding import get_meth_ids

    return {v: k for k, v in get_meth_ids().items()}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0


def _compute_metrics(
    all_mu: np.ndarray,
    all_sigma: np.ndarray,
    all_true: np.ndarray,
    all_meth_ids: np.ndarray,
    prefix: str,
) -> dict:
    """Compute overall + per-meth-type metrics.

    Returns a flat dict of floats for logging, plus a nested 'by_type' dict
    for human-readable output.
    """
    diff = all_mu - all_true
    result: dict = {}

    # Overall metrics
    mse = (diff**2).mean(axis=0)
    mae = np.abs(diff).mean(axis=0)
    in_1sig = (np.abs(diff) <= 1.0 * all_sigma).mean(axis=0)
    in_2sig = (np.abs(diff) <= 2.0 * all_sigma).mean(axis=0)
    in_3sig = (np.abs(diff) <= 3.0 * all_sigma).mean(axis=0)
    result.update(
        {
            f"{prefix}_mse_ipd": float(mse[0]),
            f"{prefix}_mse_pw": float(mse[1]),
            f"{prefix}_mae_ipd": float(mae[0]),
            f"{prefix}_mae_pw": float(mae[1]),
            f"{prefix}_pearson_ipd": _pearson(all_mu[:, 0], all_true[:, 0]),
            f"{prefix}_pearson_pw": _pearson(all_mu[:, 1], all_true[:, 1]),
            f"{prefix}_calib_1sig_ipd": float(in_1sig[0]),
            f"{prefix}_calib_1sig_pw": float(in_1sig[1]),
            f"{prefix}_calib_2sig_ipd": float(in_2sig[0]),
            f"{prefix}_calib_2sig_pw": float(in_2sig[1]),
            f"{prefix}_calib_3sig_ipd": float(in_3sig[0]),
            f"{prefix}_calib_3sig_pw": float(in_3sig[1]),
        }
    )

    # Per-meth-type breakdown
    by_type: dict = {}
    for meth_id in sorted(np.unique(all_meth_ids)):
        mask = all_meth_ids == meth_id
        if mask.sum() < 2:
            continue
        mu_m = all_mu[mask]
        sig_m = all_sigma[mask]
        true_m = all_true[mask]
        diff_m = mu_m - true_m
        name = _meth_names().get(int(meth_id), f"meth{meth_id}")
        by_type[name] = {
            "n": int(mask.sum()),
            "pearson_ipd": _pearson(mu_m[:, 0], true_m[:, 0]),
            "pearson_pw": _pearson(mu_m[:, 1], true_m[:, 1]),
            "mae_ipd": float(np.abs(diff_m[:, 0]).mean()),
            "mae_pw": float(np.abs(diff_m[:, 1]).mean()),
            "calib_1sig": float((np.abs(diff_m) <= 1.0 * sig_m).mean()),
            "calib_2sig": float((np.abs(diff_m) <= 2.0 * sig_m).mean()),
            "calib_3sig": float((np.abs(diff_m) <= 3.0 * sig_m).mean()),
        }
        # Also log per-type scalars for CSV/TensorBoard
        result[f"{prefix}_pearson_ipd_{name}"] = by_type[name]["pearson_ipd"]
        result[f"{prefix}_pearson_pw_{name}"] = by_type[name]["pearson_pw"]
        result[f"{prefix}_calib_2sig_{name}"] = by_type[name]["calib_2sig"]

    result["_by_type"] = by_type  # human-readable, not logged as scalar

    # ── Pearson oracle: theoretical ceiling for a perfect distributional model ─
    # r_oracle = Var(μ) / (Var(μ) + E[σ²])
    var_mu_ipd = np.var(all_mu[:, 0])
    var_mu_pw = np.var(all_mu[:, 1])
    e_sig2_ipd = np.mean(all_sigma[:, 0] ** 2)
    e_sig2_pw = np.mean(all_sigma[:, 1] ** 2)
    oracle_ipd = var_mu_ipd / (var_mu_ipd + e_sig2_ipd) if (var_mu_ipd + e_sig2_ipd) > 0 else 0.0
    oracle_pw = var_mu_pw / (var_mu_pw + e_sig2_pw) if (var_mu_pw + e_sig2_pw) > 0 else 0.0
    result[f"{prefix}_oracle_ipd"] = float(oracle_ipd)
    result[f"{prefix}_oracle_pw"] = float(oracle_pw)

    # ── Distribution samples: 10 random per meth type ─────────────────────────
    dist_samples: dict = {}
    rng = np.random.default_rng(42)
    for meth_id in sorted(np.unique(all_meth_ids)):
        mask = all_meth_ids == meth_id
        n_avail = int(mask.sum())
        if n_avail < 1:
            continue
        n_pick = min(10, n_avail)
        idx = rng.choice(n_avail, n_pick, replace=False)
        mu_sel = all_mu[mask][idx]  # (n_pick, 2)
        sig_sel = all_sigma[mask][idx]  # (n_pick, 2)
        name = _meth_names().get(int(meth_id), f"meth{meth_id}")
        dist_samples[name] = {
            "mu_ipd": mu_sel[:, 0],
            "mu_pw": mu_sel[:, 1],
            "sigma_ipd": sig_sel[:, 0],
            "sigma_pw": sig_sel[:, 1],
        }
    result["_dist_samples"] = dist_samples

    return result


def _grade(value: float, good: float, ok: float, higher_is_better: bool = True) -> str:
    """Return GOOD / OK / POOR based on thresholds."""
    if higher_is_better:
        if value >= good:
            return "GOOD"
        if value >= ok:
            return "OK  "
        return "POOR"
    else:
        if value <= good:
            return "GOOD"
        if value <= ok:
            return "OK  "
        return "POOR"


def _log_metrics(metrics: dict, prefix: str) -> None:
    """Print a compact, self-interpreting summary of metrics.

    Reference thresholds (log1p-space, PacBio IPD/PW):
      Pearson r IPD : GOOD ≥ 0.70  OK ≥ 0.50  POOR < 0.50
      Pearson r PW  : GOOD ≥ 0.60  OK ≥ 0.40  POOR < 0.40
      2σ calibration: GOOD 90–99%  OK 80–90%  POOR otherwise
        (ideal 95.4% — if overconfident model: <80%, underconfident: >99%)
      GNLL loss     : GOOD ≤ 1.0   OK ≤ 1.5   POOR > 1.5
        (lower = better; well-calibrated model on log1p signals ≈ 0.5–1.0)
    """
    W = 72
    log.info("─" * W)
    log.info("  %s  [GOOD/OK/POOR thresholds shown]", prefix.upper())
    log.info("─" * W)

    p_ipd = metrics.get(f"{prefix}_pearson_ipd", 0.0)
    p_pw = metrics.get(f"{prefix}_pearson_pw", 0.0)
    m_ipd = metrics.get(f"{prefix}_mae_ipd", 0.0)
    m_pw = metrics.get(f"{prefix}_mae_pw", 0.0)
    c1i = metrics.get(f"{prefix}_calib_1sig_ipd", 0.0) * 100
    c2i = metrics.get(f"{prefix}_calib_2sig_ipd", 0.0) * 100
    c3i = metrics.get(f"{prefix}_calib_3sig_ipd", 0.0) * 100
    c1p = metrics.get(f"{prefix}_calib_1sig_pw", 0.0) * 100
    c2p = metrics.get(f"{prefix}_calib_2sig_pw", 0.0) * 100
    c3p = metrics.get(f"{prefix}_calib_3sig_pw", 0.0) * 100

    # Calibration grade: ideally near 95.4% for 2σ
    calib_grade = _grade(c2i, 90.0, 80.0, higher_is_better=True)
    if c2i > 99.0:
        calib_grade = "POOR"  # underconfident

    o_ipd = metrics.get(f"{prefix}_oracle_ipd", 0.0)
    o_pw = metrics.get(f"{prefix}_oracle_pw", 0.0)
    log.info(
        "  Pearson  IPD=%.3f [%s ≥0.70]   PW=%.3f [%s ≥0.60]",
        p_ipd,
        _grade(p_ipd, 0.70, 0.50),
        p_pw,
        _grade(p_pw, 0.60, 0.40),
    )
    log.info(
        "  Oracle   IPD=%.3f              PW=%.3f  (theoretical ceiling)",
        o_ipd,
        o_pw,
    )
    log.info(
        "  MAE      IPD=%.4f               PW=%.4f  (log1p space)",
        m_ipd,
        m_pw,
    )
    log.info(
        "  Calib    IPD: 1σ=%.1f%%  2σ=%.1f%% [%s 90-99%%]  3σ=%.1f%%",
        c1i,
        c2i,
        calib_grade,
        c3i,
    )
    log.info(
        "           PW:  1σ=%.1f%%  2σ=%.1f%%               3σ=%.1f%%",
        c1p,
        c2p,
        c3p,
    )
    log.info("  (ideal calibration: 1σ=68%%  2σ=95.4%%  3σ=99.7%%)")

    by_type = metrics.get("_by_type", {})
    if by_type:
        log.info("  Per-type breakdown:")
        for name, t in by_type.items():
            cg = _grade(t["calib_2sig"] * 100, 90.0, 80.0)
            if t["calib_2sig"] * 100 > 99.0:
                cg = "POOR"
            log.info(
                "    %-6s  n=%-7d  pearson=(IPD %.3f [%s] PW %.3f [%s])  2σ=%.1f%% [%s]",
                name,
                t["n"],
                t["pearson_ipd"],
                _grade(t["pearson_ipd"], 0.70, 0.50),
                t["pearson_pw"],
                _grade(t["pearson_pw"], 0.60, 0.40),
                t["calib_2sig"] * 100,
                cg,
            )
    # ── Distribution samples (10 per meth type) ─────────────────────────────
    dist_samples = metrics.get("_dist_samples", {})
    if dist_samples:
        log.info("  Distribution samples (predicted μ ± σ in log1p space):")
        for name, s in dist_samples.items():
            mu_ipd_mean = float(np.mean(s["mu_ipd"]))
            mu_pw_mean = float(np.mean(s["mu_pw"]))
            sig_ipd_mean = float(np.mean(s["sigma_ipd"]))
            sig_pw_mean = float(np.mean(s["sigma_pw"]))
            log.info(
                "    %-6s  IPD: μ=%.3f ± σ=%.3f   PW: μ=%.3f ± σ=%.3f  (avg of %d samples)",
                name,
                mu_ipd_mean,
                sig_ipd_mean,
                mu_pw_mean,
                sig_pw_mean,
                len(s["mu_ipd"]),
            )
    log.info("─" * W)


# ---------------------------------------------------------------------------
# KineticDataModule
# ---------------------------------------------------------------------------


class KineticDataModule(L.LightningDataModule):
    """LightningDataModule for KinSim kinetic data — supports both layouts.

    Two input modes (auto-detected from ``input_path``):

    * **Single .pkl** — loads :class:`MLPSignalDataset` once into RAM,
      then random train/val split. Optional separate ``test_pkl`` for a
      held-out evaluation set. Best for small datasets.

    * **Shards directory** — uses :class:`ShardedSignalDataset` over a
      list of per-strain shard pkls. The shard list is split:

      - ``test_strains`` (explicit list) OR ``test_fraction`` (random
        per-shard) carves the held-out **test** set.
      - The remaining shards are split into train + val by
        ``val_fraction`` (random per-shard with ``seed``).

      Peak RAM is bounded by ``num_workers × shard_size``, regardless of
      corpus size — the right path for ≥ 10 strains.

    Args:
        input_path:    Path to a master .pkl OR a shards directory.
        test_pkl:      (single-pkl mode) path to a separate held-out test .pkl.
        test_strains:  (sharded mode) list of sample_ids to hold out as test.
        test_fraction: (sharded mode) random fraction of shards held out as test.
        val_fraction:  Fraction held out as validation (after test split).
        batch_size:    Training DataLoader batch size.
        seed:          Random seed for reproducible splits + sharded shuffler.
    """

    def __init__(
        self,
        input_path: str,
        test_pkl: str | None = None,
        test_strains: list | None = None,
        test_fraction: float | None = None,
        val_fraction: float = 0.10,
        batch_size: int = 4096,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_path = str(input_path)
        self.test_pkl = test_pkl
        self.test_strains = list(test_strains) if test_strains else None
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.seed = seed
        # populated in setup()
        self._train_subset = None
        self._val_subset = None
        self._test_dataset = None
        self._sharded = Path(self.input_path).is_dir()

    def setup(self, stage: str | None = None) -> None:
        if self._sharded:
            self._setup_sharded(stage)
        else:
            self._setup_monolithic(stage)

    # ── Single-pkl path ────────────────────────────────────────────────
    def _setup_monolithic(self, stage: str | None) -> None:
        if stage in ("fit", None):
            dataset = MLPSignalDataset(self.input_path)
            n_val = max(1, int(len(dataset) * self.val_fraction))
            n_train = len(dataset) - n_val
            rng = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(dataset), generator=rng).tolist()
            self._train_subset = Subset(dataset, indices[:n_train])
            self._val_subset = Subset(dataset, indices[n_train:])
            log.info("Data split — train: %d samples, val: %d samples", n_train, n_val)
        if stage in ("test", None) and self.test_pkl:
            self._test_dataset = MLPSignalDataset(self.test_pkl)
            log.info("Test set: %d keys from %s", len(self._test_dataset), self.test_pkl)

    # ── Sharded path ───────────────────────────────────────────────────
    def _setup_sharded(self, stage: str | None) -> None:
        all_shards = list_shards(self.input_path)
        if not all_shards:
            raise FileNotFoundError(f"No *_shard.pkl in {self.input_path}")

        train_shards, test_shards = split_shards(
            all_shards,
            test_strains=self.test_strains,
            test_fraction=self.test_fraction,
            seed=self.seed,
        )
        # Carve a val set out of train_shards (per-shard split, reproducible).
        rng = np.random.default_rng(self.seed + 1)
        idx = np.arange(len(train_shards))
        rng.shuffle(idx)
        n_val = max(1, round(len(train_shards) * self.val_fraction))
        val_idx = set(int(i) for i in idx[:n_val])
        val_shards = [s for i, s in enumerate(train_shards) if i in val_idx]
        train_only_shards = [s for i, s in enumerate(train_shards) if i not in val_idx]

        log.info(
            "Sharded split — train: %d shards, val: %d shards, test: %d shards",
            len(train_only_shards),
            len(val_shards),
            len(test_shards),
        )
        log.info("  train: %s", [Path(s).stem for s in train_only_shards])
        log.info("  val:   %s", [Path(s).stem for s in val_shards])
        log.info("  test:  %s", [Path(s).stem for s in test_shards])

        if stage in ("fit", None):
            self._train_subset = ShardedSignalDataset(
                train_only_shards,
                shuffle=True,
                seed=self.seed,
            )
            self._val_subset = ShardedSignalDataset(
                val_shards,
                shuffle=False,
                seed=self.seed + 100,
            )
        if stage in ("test", None) and test_shards:
            self._test_dataset = ShardedSignalDataset(
                test_shards,
                shuffle=False,
                seed=self.seed + 200,
            )

    def train_dataloader(self) -> DataLoader:
        # IterableDataset shuffles inside __iter__ — DataLoader must NOT.
        is_iter = isinstance(self._train_subset, IterableDataset)
        return DataLoader(
            self._train_subset,
            batch_size=self.batch_size,
            shuffle=not is_iter,
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

    def test_dataloader(self) -> DataLoader | None:
        if self._test_dataset is None:
            return None
        return DataLoader(
            self._test_dataset,
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
        self.model = model
        self.lr = lr
        self._loss_fn = _LOSS_FUNCTIONS[loss_name]
        # Accumulate per-batch predictions for epoch-level metrics
        self._val_mu: list[torch.Tensor] = []
        self._val_sigma: list[torch.Tensor] = []
        self._val_true: list[torch.Tensor] = []
        self._val_meth_ids: list[torch.Tensor] = []
        self._test_mu: list[torch.Tensor] = []
        self._test_sigma: list[torch.Tensor] = []
        self._test_true: list[torch.Tensor] = []
        self._test_meth_ids: list[torch.Tensor] = []

    def forward(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        return self.model(kmer_ids, meth_probs)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        kmer_ids, meth_probs, signals, _meth_ids = batch
        loss = self._loss_fn(self.model(kmer_ids, meth_probs), signals)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        kmer_ids, meth_probs, signals, meth_ids = batch
        params = self.model(kmer_ids, meth_probs)
        loss = self._loss_fn(params, signals)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        mu = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], -6.0, 3.0)
        sigma = torch.exp(log_sig)
        self._val_mu.append(mu.detach().cpu())
        self._val_sigma.append(sigma.detach().cpu())
        self._val_true.append(signals.detach().cpu())
        self._val_meth_ids.append(meth_ids.detach().cpu())
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_mu:
            return
        all_mu = torch.cat(self._val_mu).numpy()  # (N, 2)
        all_sigma = torch.cat(self._val_sigma).numpy()  # (N, 2)
        all_true = torch.cat(self._val_true).numpy()  # (N, 2)
        all_meth_ids = torch.cat(self._val_meth_ids).numpy()  # (N,)

        val_loss = self.trainer.callback_metrics.get("val_loss", float("nan"))
        gnll_grade = _grade(float(val_loss), 1.0, 1.5, higher_is_better=False)
        log.info(
            "Epoch %d  val_loss(GNLL)=%.4f [%s ≤1.0]",
            self.current_epoch,
            float(val_loss),
            gnll_grade,
        )
        metrics = _compute_metrics(all_mu, all_sigma, all_true, all_meth_ids, prefix="val")
        self.log_dict(
            {k: v for k, v in metrics.items() if isinstance(v, float)},
            on_epoch=True,
        )
        _log_metrics(metrics, prefix="val")

        self._val_mu.clear()
        self._val_sigma.clear()
        self._val_true.clear()
        self._val_meth_ids.clear()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        kmer_ids, meth_probs, signals, meth_ids = batch
        params = self.model(kmer_ids, meth_probs)
        mu = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], -6.0, 3.0)
        sigma = torch.exp(log_sig)
        self._test_mu.append(mu.detach().cpu())
        self._test_sigma.append(sigma.detach().cpu())
        self._test_true.append(signals.detach().cpu())
        self._test_meth_ids.append(meth_ids.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_mu:
            return
        all_mu = torch.cat(self._test_mu).numpy()
        all_sigma = torch.cat(self._test_sigma).numpy()
        all_true = torch.cat(self._test_true).numpy()
        all_meth_ids = torch.cat(self._test_meth_ids).numpy()

        metrics = _compute_metrics(all_mu, all_sigma, all_true, all_meth_ids, prefix="test")
        self.log_dict(
            {k: v for k, v in metrics.items() if isinstance(v, float)},
        )
        _log_metrics(metrics, prefix="test")

        self._test_mu.clear()
        self._test_sigma.clear()
        self._test_true.clear()
        self._test_meth_ids.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
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
        self.output_dir = Path(output_dir)
        self.checkpoint_every = checkpoint_every

    def _save(
        self,
        trainer: L.Trainer,
        pl_module: KineticPredictor,
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
        trainer: L.Trainer,
        pl_module: KineticPredictor,
    ) -> None:
        # trainer.current_epoch is 0-indexed during on_train_epoch_end
        epoch = trainer.current_epoch + 1
        if epoch % self.checkpoint_every == 0:
            self._save(trainer, pl_module, epoch)

    def on_train_end(
        self,
        trainer: L.Trainer,
        pl_module: KineticPredictor,
    ) -> None:
        """Save the final epoch — handles early stopping stopping mid-interval."""
        # After the last on_train_epoch_end, current_epoch is incremented by Lightning.
        # So at on_train_end, current_epoch == number of completed epochs (1-indexed).
        epoch = trainer.current_epoch
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        if not ckpt_path.exists():
            self._save(trainer, pl_module, epoch)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _read_pkl_meta(pkl_path: str) -> dict:
    """Return the ``__meta__`` provenance dict from a training .pkl, or {}.

    For sharded mode (pkl_path is a directory), reads the meta from the
    first ``*_shard.pkl`` found — refine writes the same global stats
    (including ``per_bucket`` p_fire) into every shard's __meta__.
    """
    p = Path(pkl_path)
    if p.is_dir():
        shard = next(iter(sorted(p.glob("*_shard.pkl"))), None)
        if shard is None:
            return {}
        p = shard
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
    except (OSError, pickle.UnpicklingError) as exc:
        log.warning("Could not read __meta__ from %s: %s", p, exc)
        return {}
    return data.get("__meta__", {}) if isinstance(data, dict) else {}


def _extract_p_fire(meta: dict) -> dict[str, float]:
    """Pull a flat {bucket_label: p_fire} dict out of a refined shard's meta.

    Returns ``{}`` if the meta isn't from a refined shard (no per_bucket key).
    """
    per_bucket = (meta.get("stats") or {}).get("per_bucket") or {}
    return {
        label: float(b["p_fire"])
        for label, b in per_bucket.items()
        if isinstance(b, dict) and "p_fire" in b
    }


def _extract_mean_occupancy(meta: dict) -> dict[str, float]:
    """Pull {bucket_label: mean_occupancy} from refined meta.

    Empty when the meta predates the mean_occupancy field. Generate uses
    it together with target-genome per-site fractions to apply
    ``p_fire = target_frac × (p_fire / mean_occupancy)`` as the firing
    Bernoulli rate.
    """
    per_bucket = (meta.get("stats") or {}).get("per_bucket") or {}
    return {
        label: float(b["mean_occupancy"])
        for label, b in per_bucket.items()
        if isinstance(b, dict) and "mean_occupancy" in b
    }


def _save_model_config(
    output_dir: Path,
    model: nn.Module,
    meth_types: list[str] | None = None,
    p_fire: dict[str, float] | None = None,
    mean_occupancy: dict[str, float] | None = None,
) -> None:
    """Write model_config.json before training starts.

    generate.py and evaluate.py both require this file to reconstruct the
    model architecture.  Writing it before the first epoch ensures it
    exists even if training is interrupted.

    ``meth_types`` — the alphabet the training .pkl was extracted with — is
    persisted so that generate.py can warn when the caller tries to simulate a
    type that the model never saw during training.
    """
    cfg = model.get_config()
    if meth_types is not None:
        cfg["meth_types"] = sorted(meth_types)
    if p_fire:
        cfg["p_fire"] = p_fire
    if mean_occupancy:
        cfg["mean_occupancy"] = mean_occupancy
    path = output_dir / "model_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    log.info(
        "Model config saved: %s  (architecture=%s, meth_types=%s, p_fire=%s, mean_occ=%s)",
        path,
        cfg.get("architecture"),
        cfg.get("meth_types", "all"),
        f"{len(p_fire)} buckets" if p_fire else "none",
        f"{len(mean_occupancy)} buckets" if mean_occupancy else "none",
    )


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
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    if architecture == "conv":
        base_embed_dim = trial.suggest_categorical("base_embed_dim", [8, 16])
        conv_dim = trial.suggest_categorical("conv_dim", [64, 128])
        head_dim = trial.suggest_categorical("head_dim", [64, 128, 256])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
        model = ConvPredictor(
            base_embed_dim=base_embed_dim,
            conv_dim=conv_dim,
            head_dim=head_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    else:
        kmer_embed_dim = trial.suggest_categorical("kmer_embed_dim", [32, 64])
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        model = MLPPredictor(
            kmer_embed_dim=kmer_embed_dim,
            hidden_dim=hidden_dim,
            meth_proj_dim=8,
            dropout=dropout,
        )

    lm = KineticPredictor(model, lr=lr, loss_name=loss_name)
    dm = KineticDataModule(
        input_path=pkl_path,
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
    test_pkl: str | None = None,
    test_strains: list | None = None,
    test_fraction: float | None = None,
    split_seed: int = 42,
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
        pkl_path:         Shard .pkl OR a directory of shard.pkl files (sharded mode).
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
        except ImportError as exc:
            raise ImportError(
                "Optuna is required for HPO. Install with: pip install optuna"
            ) from exc

        log.info(
            "Optuna HPO — arch=%s  %d trials × %d epochs", architecture, n_trials, optuna_epochs
        )
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
        lr = best["lr"]
        dropout = best.get("dropout", dropout)
        if architecture == "conv":
            base_embed_dim = best.get("base_embed_dim", base_embed_dim)
            conv_dim = best.get("conv_dim", conv_dim)
            head_dim = best.get("head_dim", head_dim)
            kernel_size = best.get("kernel_size", kernel_size)
        else:
            kmer_embed_dim = best.get("kmer_embed_dim", kmer_embed_dim)
            hidden_dim = best.get("hidden_dim", hidden_dim)
        (output_dir / "optuna_best_params.json").write_text(
            json.dumps({"best_val_loss": study.best_value, **best}, indent=2)
        )

    # ── Build model ───────────────────────────────────────────────────────
    accelerator = "gpu" if device == "cuda" and torch.cuda.is_available() else "cpu"

    if architecture == "conv":
        log.info(
            "Training — arch=conv  %d epochs  loss=%s  lr=%.2e  base_embed=%d  "
            "conv_dim=%d  n_layers=%d  k=%d  head=%d  meth_proj=%d  dropout=%.2f  accel=%s",
            epochs,
            loss_name,
            lr,
            base_embed_dim,
            conv_dim,
            n_conv_layers,
            kernel_size,
            head_dim,
            meth_proj_dim,
            dropout,
            accelerator,
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
            epochs,
            loss_name,
            lr,
            kmer_embed_dim,
            hidden_dim,
            meth_proj_dim,
            dropout,
            accelerator,
        )
        model = MLPPredictor(
            kmer_embed_dim=kmer_embed_dim,
            hidden_dim=hidden_dim,
            meth_proj_dim=meth_proj_dim,
            dropout=dropout,
        )

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s (%s)", f"{n_params:,}", architecture)

    # Surface the meth-alphabet used for extraction so training logs make it
    # explicit which modification types this checkpoint is valid for.
    meta = _read_pkl_meta(pkl_path)
    pkl_meth_types = meta.get("meth_types")  # list[str] | None
    if pkl_meth_types:
        log.info("Training alphabet (from %s __meta__): %s", Path(pkl_path).name, pkl_meth_types)
    else:
        log.info(
            "Training alphabet: all types in .pkl (no --meth-types filter "
            "was applied during extraction)"
        )

    # GMM survival rate + mean motif occupancy per (meth, offset) bucket —
    # generate.py decomposes them as p_efficiency = p_fire / mean_occupancy
    # and applies p_fire(target site) = target_frac × p_efficiency at
    # inference time, so synthetic reads track per-strain occupancy
    # (not just the training-corpus average rate).
    p_fire = _extract_p_fire(meta)
    mean_occ = _extract_mean_occupancy(meta)
    if p_fire:
        log.info(
            "p_fire (GMM survival): %s",
            ", ".join(f"{k}={v:.2f}" for k, v in sorted(p_fire.items())),
        )
    if mean_occ:
        log.info(
            "mean_occupancy (training corpus): %s",
            ", ".join(f"{k}={v:.2f}" for k, v in sorted(mean_occ.items())),
        )

    # Save model config BEFORE first epoch — generate.py needs it even if interrupted
    _save_model_config(
        output_dir, model,
        meth_types=pkl_meth_types,
        p_fire=p_fire,
        mean_occupancy=mean_occ,
    )

    if resume_ckpt:
        log.info("Loading weights from: %s", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            # Legacy format: direct MLPPredictor state dict (no "model." prefix)
            model.load_state_dict(ckpt["model"])
        elif "state_dict" in ckpt:
            # Lightning format: strip "model." prefix added by KineticPredictor wrapper
            state_dict = {
                k[len("model.") :]: v
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
        input_path=pkl_path,
        test_pkl=test_pkl,
        test_strains=test_strains,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        batch_size=batch_size,
        seed=split_seed,
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

    if test_pkl:
        log.info("Running evaluation on held-out test set: %s", test_pkl)
        trainer.test(lm, datamodule=dm)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .utils.config import load_yaml_config, setup_logging

    parser = argparse.ArgumentParser(
        prog="kinsim train",
        description=(
            "Train ConvPredictor / MLPPredictor on shard pkls.\n\n"
            "Input: a directory of refined *_shard.pkl (sharded mode, recommended)\n"
            "or a single shard.pkl (small datasets, debugging). Auto-detected.\n\n"
            "Pipeline:\n"
            "  kinsim extract --manifest manifest.csv --task N --output-dir shards/\n"
            "  kinsim refine  shards/   refined/\n"
            "  kinsim train   refined/  checkpoints/\n\n"
            "All flags may be specified in a YAML config file (--config).\n"
            "Command-line flags override YAML values.\n\n"
            "Optuna HPO: --optuna --n-trials N  (searches lr / kmer_embed_dim / hidden_dim)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pkl",
        nargs="?",
        default=None,
        help="Training input — either a single shard.pkl (in-memory) or a directory "
        "of refined *_shard.pkl files (sharded streaming). Auto-detected from the path.",
    )
    parser.add_argument(
        "output_dir", nargs="?", default=None, help="Directory for checkpoints and logs"
    )
    parser.add_argument(
        "--test-pkl",
        default=None,
        help="(single-pkl mode) held-out test .pkl — evaluated once after training",
    )
    parser.add_argument(
        "--test-strains",
        default=None,
        help="(sharded mode) comma-separated sample_ids to hold out as the test set "
        "(e.g. --test-strains bc2080,bc2081,bc2082). Their shards never enter training.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="(sharded mode) random per-shard fraction held out as test "
        "(default: 0.10 if neither --test-strains nor --test-pkl is given).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for the random shard split (sharded mode only).",
    )

    # Architecture selection
    parser.add_argument(
        "--architecture",
        default=None,
        choices=["conv", "mlp"],
        help="Model architecture: conv (default, 1D-conv + FiLM) or mlp (legacy embedding)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--config", default=None, help="YAML config file (all flags can be set here)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 4096)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-3)")

    # Conv architecture params
    parser.add_argument(
        "--base-embed-dim",
        type=int,
        default=None,
        help="[conv] Per-base embedding dimension (default: 16)",
    )
    parser.add_argument(
        "--conv-dim", type=int, default=None, help="[conv] Conv channel width (default: 128)"
    )
    parser.add_argument(
        "--n-conv-layers", type=int, default=None, help="[conv] Number of conv layers (default: 3)"
    )
    parser.add_argument(
        "--kernel-size", type=int, default=None, help="[conv] Conv kernel size (default: 3)"
    )
    parser.add_argument(
        "--head-dim", type=int, default=None, help="[conv] Head hidden layer width (default: 128)"
    )

    # MLP architecture params (legacy)
    parser.add_argument(
        "--kmer-embed-dim",
        type=int,
        default=None,
        help="[mlp] 11-mer embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None, help="[mlp] Hidden layer width (default: 128)"
    )

    # Shared params
    parser.add_argument(
        "--meth-proj-dim",
        type=int,
        default=None,
        help="Methylation projection output dim (default: 8)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability (default: 0.1 for conv, 0.0 for mlp)",
    )
    parser.add_argument(
        "--loss",
        default=None,
        choices=["gnll", "mse", "huber"],
        help="Loss function: gnll=Gaussian NLL (default), mse, huber",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Fraction for validation split (default: 0.10)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save legacy checkpoint every N epochs (default: 10)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device (default: cuda, falls back to cpu automatically)",
    )
    parser.add_argument(
        "--resume", dest="resume_ckpt", help="Resume weights from a checkpoint .pt or .ckpt file"
    )

    # Optuna HPO flags
    parser.add_argument(
        "--optuna", action="store_true", help="Run Optuna HPO before the final training run"
    )
    parser.add_argument(
        "--n-trials", type=int, default=None, help="Number of Optuna trials (default: 20)"
    )
    parser.add_argument(
        "--optuna-epochs",
        type=int,
        default=None,
        help="Epochs per Optuna trial (default: 20, shorter than --epochs)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    # Merge YAML config with CLI flags — precedence: CLI > YAML > hard-coded defaults
    cfg: dict = {}
    if args.config:
        cfg = load_yaml_config(args.config)

    def _get(cli_val, key, default):
        return cli_val if cli_val is not None else cfg.get(key, default)

    pkl_path = args.pkl or cfg.get("pkl")
    output_dir = args.output_dir or cfg.get("output_dir")

    if not pkl_path:
        parser.error("pkl is required (positional arg or 'pkl' in YAML config)")
    if not output_dir:
        parser.error("output_dir is required (positional arg or 'output_dir' in YAML config)")

    architecture = _get(args.architecture, "architecture", "conv")
    # Default dropout depends on architecture
    default_dropout = 0.1 if architecture == "conv" else 0.0

    # Parse --test-strains "a,b,c" into a list (sharded mode only).
    test_strains_arg = args.test_strains or cfg.get("test_strains")
    if isinstance(test_strains_arg, str):
        test_strains_list = [s.strip() for s in test_strains_arg.split(",") if s.strip()]
    elif isinstance(test_strains_arg, list):
        test_strains_list = list(test_strains_arg)
    else:
        test_strains_list = None

    train_mlp(
        pkl_path=pkl_path,
        output_dir=output_dir,
        test_pkl=args.test_pkl or cfg.get("test_pkl"),
        test_strains=test_strains_list,
        test_fraction=_get(args.test_fraction, "test_fraction", None),
        split_seed=_get(args.split_seed, "split_seed", 42),
        architecture=architecture,
        epochs=_get(args.epochs, "epochs", 50),
        batch_size=_get(args.batch_size, "batch_size", 4096),
        lr=_get(args.lr, "lr", 1e-3),
        # Conv params
        base_embed_dim=_get(args.base_embed_dim, "base_embed_dim", 16),
        conv_dim=_get(args.conv_dim, "conv_dim", 128),
        n_conv_layers=_get(args.n_conv_layers, "n_conv_layers", 3),
        kernel_size=_get(args.kernel_size, "kernel_size", 3),
        head_dim=_get(args.head_dim, "head_dim", 128),
        # MLP params
        kmer_embed_dim=_get(args.kmer_embed_dim, "kmer_embed_dim", 64),
        hidden_dim=_get(args.hidden_dim, "hidden_dim", 128),
        # Shared
        meth_proj_dim=_get(args.meth_proj_dim, "meth_proj_dim", 8),
        dropout=_get(args.dropout, "dropout", default_dropout),
        loss_name=_get(args.loss, "loss", "gnll"),
        val_fraction=_get(args.val_fraction, "val_fraction", 0.10),
        checkpoint_every=_get(args.checkpoint_every, "checkpoint_every", 10),
        device=_get(args.device, "device", "cuda"),
        resume_ckpt=args.resume_ckpt or cfg.get("resume"),
        run_optuna=args.optuna or cfg.get("optuna", False),
        n_trials=_get(args.n_trials, "n_trials", 20),
        optuna_epochs=_get(args.optuna_epochs, "optuna_epochs", 20),
    )


if __name__ == "__main__":
    main()
