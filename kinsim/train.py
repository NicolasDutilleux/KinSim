"""``kinsim train`` — train the kinetic predictor on shard pkls.

Input
-----
Either:
  - a directory of ``*_shard.pkl`` files (sharded mode, recommended) — uses
    ``ShardedSignalDataset``, never holds the corpus in RAM
  - a single ``shard.pkl`` (small datasets / debugging) — loads into RAM

Each shard is ``dict[kmer_id (int) → np.ndarray(N, 20)]`` with the column
layout from :mod:`kinsim.utils.sample_layout`. Produced by
``kinsim extract`` and optionally filtered by ``kinsim refine``.

Loss
----
Default is Beta-NLL (``betanll``, β=0.5) — a scale-corrected variant of
the Gaussian NLL that down-weights samples in regions of high predicted
σ², avoiding the model gaming σ in place of fitting μ. The model outputs
``(μ, log_σ)`` for both IPD and PW in log1p space.

Alternatives: ``--loss gnll`` (vanilla Gaussian NLL),
``--loss betanll_0.3 / _0.5 / _1.0`` (β override), ``--loss mse`` /
``--loss huber`` (only the μ head — for ablations).

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
``--optuna --n-trials N`` runs an Optuna search over lr / base_embed_dim /
conv_dim / head_dim / kernel_size / dropout before the final training run.

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
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
except ImportError:
    try:
        import pytorch_lightning as L
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    except ImportError as exc:
        raise ImportError(
            "PyTorch Lightning is required for training.\nInstall with: pip install lightning"
        ) from exc

from .data.dataset import (
    ShardedSignalDataset,
    SignalDataset,
    list_shards,
    split_shards,
)
from .models.predictor import ConvPredictor

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Tunables (formerly magic numbers)
# ─────────────────────────────────────────────────────────────────────────
# Optuna trials cut short — short runs that don't improve fast aren't
# worth a full 10-epoch wait.
EARLY_STOP_PATIENCE_TRIAL: int = 5
# Full training run — let val_loss plateaus actually exit a saddle.
EARLY_STOP_PATIENCE_FULL: int = 10
# Per-epoch budget. IterableDataset has no ``__len__`` so Lightning
# would iterate the full corpus per epoch (~2.5B rows on a 49-strain
# Strepto+Vega corpus). 50k batches × ~2048 ≈ 100M rows/epoch.
TRAIN_BATCHES_PER_EPOCH: int = 50_000
VAL_BATCHES_PER_EPOCH: int = 2_000


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _gaussian_nll_loss(
    params: torch.Tensor,
    targets: torch.Tensor,
    log_sigma_min: float = -6.0,
    log_sigma_max: float = 3.0,
) -> torch.Tensor:
    """Gaussian NLL for (IPD, PW) jointly.

    Args:
        params:        Model output (batch, 4) — [μ_ipd, μ_pw, log_σ_ipd, log_σ_pw].
        targets:       Ground-truth signals (batch, 2) — [IPD, PW] in log1p space.
        log_sigma_min: Lower clamp on log σ (default -6).
        log_sigma_max: Upper clamp on log σ (default 3, tighten to 1.5 to
                       prevent the model from gaming σ instead of fitting μ).
    """
    mu = params[:, :2]
    log_sig = torch.clamp(params[:, 2:], log_sigma_min, log_sigma_max)
    var = torch.exp(2.0 * log_sig)
    return (0.5 * (log_sig * 2.0 + (targets - mu) ** 2 / var)).mean()


def _beta_nll_loss(
    params: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.5,
    log_sigma_min: float = -6.0,
    log_sigma_max: float = 3.0,
) -> torch.Tensor:
    """Beta-NLL (Seitzer+ 2022) — reweights per-sample GNLL by σ^(2β).

    Standard GNLL allows the network to "cheat" by inflating σ on hard
    samples instead of improving μ. Beta-NLL multiplies the per-sample
    NLL by ``σ²ᵝ`` (detached from the graph) — large-σ samples get
    proportionally more gradient on μ, eliminating the shortcut.

    β=0   → identity (vanilla GNLL).
    β=0.5 → recommended default in the paper.
    β=1   → equivalent to plain MSE on μ (σ gradient cancelled).
    """
    mu = params[:, :2]
    log_sig = torch.clamp(params[:, 2:], log_sigma_min, log_sigma_max)
    var = torch.exp(2.0 * log_sig)
    nll = 0.5 * (log_sig * 2.0 + (targets - mu) ** 2 / var)
    # Stop-grad reweighting term: σ²ᵝ as a fixed scalar per element.
    weight = var.detach() ** beta
    return (nll * weight).mean()


def _per_sample_gaussian_nll(
    mu: np.ndarray,
    sigma: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Per-sample GNLL on (IPD, PW) — same math as _gaussian_nll_loss but
    returns a (N,) array (mean over the 2 dims). Used to bucket val/test
    losses by (category, parent_meth, parent_offset).
    """
    sigma_safe = np.maximum(sigma, 1e-6)
    log_sig = np.log(sigma_safe)
    var = sigma_safe**2
    nll = 0.5 * (log_sig * 2.0 + (targets - mu) ** 2 / var)
    return nll.mean(axis=1)


def _mse_loss(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain MSE on the mean head only (μ_ipd, μ_pw)."""
    return nn.functional.mse_loss(params[:, :2], targets)


def _huber_loss(params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Huber (smooth L1) loss on the mean head — less sensitive to outliers."""
    return nn.functional.huber_loss(params[:, :2], targets)


_LOSS_FUNCTIONS = {
    "gnll": _gaussian_nll_loss,
    "betanll": lambda p, t, **kw: _beta_nll_loss(p, t, beta=0.5, **kw),
    "betanll_0.3": lambda p, t, **kw: _beta_nll_loss(p, t, beta=0.3, **kw),
    "betanll_0.5": lambda p, t, **kw: _beta_nll_loss(p, t, beta=0.5, **kw),
    "betanll_1.0": lambda p, t, **kw: _beta_nll_loss(p, t, beta=1.0, **kw),
    # MSE/huber don't use σ — accept and ignore the clamp kwargs.
    "mse": lambda p, t, **_kw: _mse_loss(p, t),
    "huber": lambda p, t, **_kw: _huber_loss(p, t),
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
    kmer_ids: np.ndarray | None = None,
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
            # Single μ-only checkpoint metric: average MSE of IPD and PW.
            # Best checkpoint by val_mse_mu picks the model that fits μ
            # best regardless of σ-cheating (Seitzer 2022 motivation).
            f"{prefix}_mse_mu": float((mse[0] + mse[1]) / 2.0),
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

    # ── Pearson oracle: theoretical ceiling for a perfect distributional model
    # Two flavours are logged side-by-side:
    #
    #   r_self_oracle = Var(μ_pred) / (Var(μ_pred) + E[σ_pred²])
    #       Uses the model's own predicted σ. Useful as a diagnostic but
    #       CIRCULAR: a lazy model that inflates σ on hard samples lowers
    #       its own ceiling, then matches it — looking healthy when it's
    #       only consistent with its own laziness.
    #
    #   r_empirical_oracle = Var(per-bucket μ_true) / (Var + mean(per-bucket σ_true²))
    #       Per-(kmer_id, meth_id) bucket from the OBSERVED signals; the
    #       true irreducible noise. Requires kmer_ids and at least
    #       _ORACLE_MIN_PER_BUCKET samples per bucket; otherwise skipped.
    var_mu_ipd = np.var(all_mu[:, 0])
    var_mu_pw = np.var(all_mu[:, 1])
    e_sig2_ipd = np.mean(all_sigma[:, 0] ** 2)
    e_sig2_pw = np.mean(all_sigma[:, 1] ** 2)
    self_oracle_ipd = (
        var_mu_ipd / (var_mu_ipd + e_sig2_ipd) if (var_mu_ipd + e_sig2_ipd) > 0 else 0.0
    )
    self_oracle_pw = var_mu_pw / (var_mu_pw + e_sig2_pw) if (var_mu_pw + e_sig2_pw) > 0 else 0.0
    result[f"{prefix}_oracle_ipd"] = float(self_oracle_ipd)
    result[f"{prefix}_oracle_pw"] = float(self_oracle_pw)

    if kmer_ids is not None and len(kmer_ids) == len(all_true):
        _ORACLE_MIN_PER_BUCKET = 5
        bucket_keys = (kmer_ids.astype(np.int64) << 8) | (all_meth_ids.astype(np.int64) & 0xFF)
        uniq, inv, counts = np.unique(bucket_keys, return_inverse=True, return_counts=True)
        keep_mask = counts[inv] >= _ORACLE_MIN_PER_BUCKET
        if keep_mask.any():
            mu_true_ipd_per = np.bincount(
                inv[keep_mask], weights=all_true[keep_mask, 0]
            ) / np.maximum(np.bincount(inv[keep_mask]), 1)
            mu_true_pw_per = np.bincount(
                inv[keep_mask], weights=all_true[keep_mask, 1]
            ) / np.maximum(np.bincount(inv[keep_mask]), 1)
            # Per-bucket variance via E[X²] - (E[X])²
            sqr_ipd_per = np.bincount(
                inv[keep_mask], weights=all_true[keep_mask, 0] ** 2
            ) / np.maximum(np.bincount(inv[keep_mask]), 1)
            sqr_pw_per = np.bincount(
                inv[keep_mask], weights=all_true[keep_mask, 1] ** 2
            ) / np.maximum(np.bincount(inv[keep_mask]), 1)
            var_ipd_per = np.maximum(sqr_ipd_per - mu_true_ipd_per**2, 0.0)
            var_pw_per = np.maximum(sqr_pw_per - mu_true_pw_per**2, 0.0)
            n_valid_buckets = int((np.bincount(inv[keep_mask]) > 0).sum())
            if n_valid_buckets >= 2:
                # Only buckets that actually have samples in the kept rows
                nonempty = np.bincount(inv[keep_mask]) > 0
                v_mu_ipd_emp = float(np.var(mu_true_ipd_per[nonempty]))
                v_mu_pw_emp = float(np.var(mu_true_pw_per[nonempty]))
                e_var_ipd_emp = float(np.mean(var_ipd_per[nonempty]))
                e_var_pw_emp = float(np.mean(var_pw_per[nonempty]))
                emp_oracle_ipd = (
                    v_mu_ipd_emp / (v_mu_ipd_emp + e_var_ipd_emp)
                    if (v_mu_ipd_emp + e_var_ipd_emp) > 0
                    else 0.0
                )
                emp_oracle_pw = (
                    v_mu_pw_emp / (v_mu_pw_emp + e_var_pw_emp)
                    if (v_mu_pw_emp + e_var_pw_emp) > 0
                    else 0.0
                )
                result[f"{prefix}_oracle_empirical_ipd"] = float(emp_oracle_ipd)
                result[f"{prefix}_oracle_empirical_pw"] = float(emp_oracle_pw)
                result[f"{prefix}_oracle_empirical_n_buckets"] = float(n_valid_buckets)

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
    """Print a compact summary. Pearson thresholds match PacBio noise floor."""
    P_IPD_GOOD, P_IPD_OK = 0.35, 0.20
    P_PW_GOOD,  P_PW_OK  = 0.30, 0.15

    def g(v, good, ok, lower=False):
        return _grade(v, good, ok, higher_is_better=not lower)

    def get(k, default=0.0):
        return metrics.get(f"{prefix}_{k}", default)

    p_ipd, p_pw = get("pearson_ipd"), get("pearson_pw")
    c2i, c2p    = get("calib_2sig_ipd") * 100, get("calib_2sig_pw") * 100
    calib_grade = "POOR" if c2i > 99.0 else g(c2i, 90.0, 80.0)

    log.info("─" * 72)
    log.info("  %s metrics", prefix.upper())
    log.info("─" * 72)
    log.info("  Pearson  IPD=%.3f [%s >=%.2f]   PW=%.3f [%s >=%.2f]",
             p_ipd, g(p_ipd, P_IPD_GOOD, P_IPD_OK), P_IPD_GOOD,
             p_pw,  g(p_pw,  P_PW_GOOD,  P_PW_OK),  P_PW_GOOD)

    emp_ipd = metrics.get(f"{prefix}_oracle_empirical_ipd")
    emp_pw  = metrics.get(f"{prefix}_oracle_empirical_pw")
    n_buck  = int(metrics.get(f"{prefix}_oracle_empirical_n_buckets") or 0)
    if emp_ipd is not None and emp_ipd > 0:
        pct_i = 100.0 * p_ipd / emp_ipd
        pct_p = 100.0 * p_pw / emp_pw if emp_pw and emp_pw > 0 else 0.0
        log.info("  Ceiling  IPD=%.3f  PW=%.3f  (empirical, %d buckets)  achieved=%.0f%%/%.0f%%",
                 emp_ipd, emp_pw, n_buck, pct_i, pct_p)
    else:
        log.info("  Oracle   IPD=%.3f  PW=%.3f  (self, circular if sigma inflated)",
                 get("oracle_ipd"), get("oracle_pw"))

    log.info("  MAE      IPD=%.4f  PW=%.4f  (log1p)", get("mae_ipd"), get("mae_pw"))
    log.info("  Calib    IPD 2sigma=%.1f%% [%s]   PW 2sigma=%.1f%%   (ideal=95.4%%)",
             c2i, calib_grade, c2p)

    by_type = metrics.get("_by_type", {})
    if by_type:
        log.info("  Per-type (%d in data: %s):", len(by_type), ", ".join(by_type.keys()))
        for name, t in by_type.items():
            c2 = t["calib_2sig"] * 100
            cg = "POOR" if c2 > 99.0 else g(c2, 90.0, 80.0)
            log.info("    %-6s  n=%-7d  IPD=%.3f [%s]  PW=%.3f [%s]  2sigma=%.1f%% [%s]",
                     name, t["n"],
                     t["pearson_ipd"], g(t["pearson_ipd"], P_IPD_GOOD, P_IPD_OK),
                     t["pearson_pw"],  g(t["pearson_pw"],  P_PW_GOOD,  P_PW_OK),
                     c2, cg)

    samples = metrics.get("_dist_samples", {})
    if samples:
        log.info("  Mean predicted mu/sigma per meth type (log1p):")
        for name, s in samples.items():
            log.info("    %-6s  IPD mu=%.3f sigma=%.3f   PW mu=%.3f sigma=%.3f",
                     name, float(np.mean(s["mu_ipd"])), float(np.mean(s["sigma_ipd"])),
                     float(np.mean(s["mu_pw"])),  float(np.mean(s["sigma_pw"])))
    log.info("─" * 72)


# ---------------------------------------------------------------------------
# Per-(category, parent_meth, parent_offset) bucket metrics
# ---------------------------------------------------------------------------


def _bucket_keys_from_arrays(
    categories: np.ndarray,
    parent_meths: np.ndarray,
    parent_offsets: np.ndarray,
) -> np.ndarray:
    """Map per-sample (category, parent_meth, parent_offset) → bucket key.

    Compact view (9 buckets typical for m6A + m4C + m5C YAML):
      CATEGORY=0 (BASELINE)  → 'BASELINE'
      CATEGORY=1 (SLOWED)    → 'SLOWED/<meth_name>@<+offset>'
      CATEGORY=2 (NEAR_METH) → 'NEAR_METH/<meth_name>'  (aggregated across offsets)

    Aggregating NEAR_METH per meth type instead of per offset keeps the
    log table readable (otherwise we'd have ~24 buckets — one per non-
    signature offset per meth type, which is just noise control).
    """
    from .utils.encoding import get_meth_ids

    names_by_id = {v: k for k, v in get_meth_ids().items()}

    keys = np.empty(len(categories), dtype=object)
    keys[:] = "OTHER"
    keys[categories == 0] = "BASELINE"

    for meth_id, name in names_by_id.items():
        if meth_id == 0:
            continue
        slowed_mask = (categories == 1) & (parent_meths == meth_id)
        if slowed_mask.any():
            unique_offsets = np.unique(parent_offsets[slowed_mask])
            for k in unique_offsets:
                mask_k = slowed_mask & (parent_offsets == k)
                keys[mask_k] = f"SLOWED/{name}@{int(k):+d}"
        near_mask = (categories == 2) & (parent_meths == meth_id)
        if near_mask.any():
            keys[near_mask] = f"NEAR_METH/{name}"

    return keys


def _safe_key(s: str) -> str:
    """Make a bucket key safe as a TensorBoard / CSV metric key."""
    return s.replace("/", "_").replace("@", "_at_").replace("+", "p").replace("-", "m")


def _compute_bucket_metrics(
    all_mu: np.ndarray,
    all_sigma: np.ndarray,
    all_true: np.ndarray,
    all_categories: np.ndarray,
    all_parent_meths: np.ndarray,
    all_parent_offsets: np.ndarray,
    prefix: str,
) -> dict:
    """Per-bucket val/test metrics: loss, MAE, Pearson, 2σ calibration."""
    bucket_keys = _bucket_keys_from_arrays(
        all_categories,
        all_parent_meths,
        all_parent_offsets,
    )
    per_loss = _per_sample_gaussian_nll(all_mu, all_sigma, all_true)  # (N,)

    result: dict = {}
    by_bucket: dict = {}
    for bk in np.unique(bucket_keys):
        mask = bucket_keys == bk
        n = int(mask.sum())
        if n < 10:
            continue
        mu_b = all_mu[mask]
        true_b = all_true[mask]
        sig_b = all_sigma[mask]
        diff_b = mu_b - true_b
        rec = {
            "n": n,
            "loss": float(per_loss[mask].mean()),
            "pearson_ipd": _pearson(mu_b[:, 0], true_b[:, 0]),
            "pearson_pw": _pearson(mu_b[:, 1], true_b[:, 1]),
            "mae_ipd": float(np.abs(diff_b[:, 0]).mean()),
            "mae_pw": float(np.abs(diff_b[:, 1]).mean()),
            "calib_2sig": float((np.abs(diff_b) <= 2.0 * sig_b).mean()),
        }
        by_bucket[bk] = rec
        sk = _safe_key(bk)
        result[f"{prefix}_bucket_loss_{sk}"] = rec["loss"]
        result[f"{prefix}_bucket_pearson_ipd_{sk}"] = rec["pearson_ipd"]
        result[f"{prefix}_bucket_pearson_pw_{sk}"] = rec["pearson_pw"]
        result[f"{prefix}_bucket_mae_ipd_{sk}"] = rec["mae_ipd"]
        result[f"{prefix}_bucket_calib2_{sk}"] = rec["calib_2sig"]
    result["_by_bucket"] = by_bucket
    return result


def _bucket_sort_key(bk: str) -> tuple:
    """Sort: BASELINE, then SLOWED grouped by meth then offset, then NEAR_METH."""
    if bk == "BASELINE":
        return (0, "", 0)
    if bk.startswith("SLOWED/"):
        rest = bk[len("SLOWED/") :]  # e.g. m6A@+0
        if "@" in rest:
            T, off = rest.split("@")
            try:
                return (1, T, int(off))
            except ValueError:
                return (1, T, 0)
        return (1, rest, 0)
    if bk.startswith("NEAR_METH/"):
        return (2, bk[len("NEAR_METH/") :], 0)
    return (9, bk, 0)


def _log_bucket_metrics(bucket_metrics: dict, prefix: str) -> None:
    """Pretty-print per-bucket metrics to the INFO log."""
    by_bucket = bucket_metrics.get("_by_bucket", {})
    if not by_bucket:
        return
    W = 88
    log.info("─" * W)
    log.info("  %s  per-(category, parent_meth, parent_offset) breakdown", prefix.upper())
    log.info("─" * W)
    log.info(
        "    %-26s  %12s  %8s  %8s  %8s  %6s",
        "bucket",
        "n",
        "loss",
        "pIPD",
        "pPW",
        "2σ%",
    )
    for bk in sorted(by_bucket.keys(), key=_bucket_sort_key):
        r = by_bucket[bk]
        log.info(
            "    %-26s  %12d  %+8.4f  %8.3f  %8.3f  %5.1f%%",
            bk,
            r["n"],
            r["loss"],
            r["pearson_ipd"],
            r["pearson_pw"],
            r["calib_2sig"] * 100,
        )
    log.info("─" * W)


# ---------------------------------------------------------------------------
# KineticDataModule
# ---------------------------------------------------------------------------


class KineticDataModule(L.LightningDataModule):
    """LightningDataModule for KinSim kinetic data — supports both layouts.

    Two input modes (auto-detected from ``input_path``):

    * **Single .pkl** — loads :class:`SignalDataset` once into RAM,
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
        num_meth_types: int | None = None,
        seed: int = 42,
        max_rows_per_shard: int | None = None,
        num_workers: int = 2,
        augment: bool = True,
        balance_kmers: bool = True,
        params=None,  # ExtractionParams | None
    ) -> None:
        super().__init__()
        self.input_path = str(input_path)
        self.test_pkl = test_pkl
        self.test_strains = list(test_strains) if test_strains else None
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        # ``num_meth_types`` defaults to whatever the YAML declares right now
        # (max meth_id + 1 from get_meth_ids()). Caller can override for tests
        # or to pin a value matching a pre-existing checkpoint.
        if num_meth_types is None:
            from .utils.encoding import get_meth_ids

            num_meth_types = max(get_meth_ids().values()) + 1
        self.num_meth_types = int(num_meth_types)
        self.seed = seed
        # ``max_rows_per_shard``: cap yielded rows per shard per worker pass.
        # Forces visiting all shards per epoch (with limit_train_batches active)
        # instead of one big shard hogging the budget. 2M rows/shard × 49 shards
        # / 2 workers ≈ 50M rows/epoch — fits the 50k batches × 1024 budget below
        # while guaranteeing each shard is seen at least once.
        self.max_rows_per_shard = max_rows_per_shard
        self.num_workers = int(num_workers)
        # Augmentation / balancing flags applied only to the TRAIN split —
        # val and test stay clean for honest metrics.
        self.augment = bool(augment)
        self.balance_kmers = bool(balance_kmers)
        # Window geometry: forwarded into every Dataset constructed in setup()
        # so the column layout is fixed and verified per shard.
        self.params = params
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
            dataset = SignalDataset(
                self.input_path,
                num_meth_types=self.num_meth_types,
                augment=self.augment,
                augment_seed=self.seed,
                params=self.params,
            )
            n_val = max(1, int(len(dataset) * self.val_fraction))
            n_train = len(dataset) - n_val
            rng = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(dataset), generator=rng).tolist()
            self._train_subset = Subset(dataset, indices[:n_train])
            self._val_subset = Subset(dataset, indices[n_train:])
            log.info("Data split — train: %d samples, val: %d samples", n_train, n_val)
        if stage in ("test", None) and self.test_pkl:
            self._test_dataset = SignalDataset(
                self.test_pkl,
                num_meth_types=self.num_meth_types,
                params=self.params,
            )
            log.info("Test set: %d keys from %s", len(self._test_dataset), self.test_pkl)

    # ── Sharded path ───────────────────────────────────────────────────
    def _setup_sharded(self, stage: str | None) -> None:
        all_shards = list_shards(self.input_path)
        if not all_shards:
            raise FileNotFoundError(f"No *_shard*.pkl (raw or refined) in {self.input_path}")

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
                num_meth_types=self.num_meth_types,
                seed=self.seed,
                augment=self.augment,
                balance_kmers=self.balance_kmers,
                params=self.params,
            )
            # Per-shard cap on train so every shard is visited per epoch
            # (rather than burning the full budget on one big shard).
            self._train_subset._max_rows_per_shard = self.max_rows_per_shard
            self._val_subset = ShardedSignalDataset(
                val_shards,
                shuffle=False,
                num_meth_types=self.num_meth_types,
                seed=self.seed + 100,
                params=self.params,
            )
        if stage in ("test", None) and test_shards:
            self._test_dataset = ShardedSignalDataset(
                test_shards,
                shuffle=False,
                num_meth_types=self.num_meth_types,
                seed=self.seed + 200,
                params=self.params,
            )

    def train_dataloader(self) -> DataLoader:
        # IterableDataset shuffles inside __iter__ — DataLoader must NOT.
        # Keep the DataLoader config minimal and identical to the
        # known-working v6 baseline: 2 workers, pin_memory, no
        # persistent_workers (caused steady-state slowdown on this
        # cluster's IPC), no prefetch_factor override.
        is_iter = isinstance(self._train_subset, IterableDataset)
        return DataLoader(
            self._train_subset,
            batch_size=self.batch_size,
            shuffle=not is_iter,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_subset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self._test_dataset is None:
            return None
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# KineticPredictor (LightningModule)
# ---------------------------------------------------------------------------


class KineticPredictor(L.LightningModule):
    """Lightning module wrapping :class:`ConvPredictor`.

    Validation (per epoch):
        - val_loss: GNLL / Beta-NLL on the validation split.
        - val_mse_ipd / val_mse_pw: mean squared error in log1p space.
        - val_pearson_ipd / val_pearson_pw: Pearson r between predicted μ and truth.
        - val_calib_ipd / val_calib_pw: 2σ calibration coverage (~95.4 % expected).
    """

    def __init__(
        self,
        model: ConvPredictor,
        lr: float = 1e-3,
        loss_name: str = "betanll",
        lr_schedule: str = "cosine",
        max_epochs: int = 50,
        warmup_epochs: int = 3,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_name = loss_name
        self._loss_fn = _LOSS_FUNCTIONS[loss_name]
        self.lr_schedule = str(lr_schedule)
        self.max_epochs = int(max_epochs)
        self.warmup_epochs = int(warmup_epochs)
        # Accumulate per-batch predictions for epoch-level metrics
        self._val_mu: list[torch.Tensor] = []
        self._val_sigma: list[torch.Tensor] = []
        self._val_true: list[torch.Tensor] = []
        self._val_meth_ids: list[torch.Tensor] = []
        self._val_parent_meths: list[torch.Tensor] = []
        self._val_parent_offsets: list[torch.Tensor] = []
        self._val_categories: list[torch.Tensor] = []
        self._val_kmer_ids: list[torch.Tensor] = []
        self._test_mu: list[torch.Tensor] = []
        self._test_sigma: list[torch.Tensor] = []
        self._test_true: list[torch.Tensor] = []
        self._test_meth_ids: list[torch.Tensor] = []
        self._test_parent_meths: list[torch.Tensor] = []
        self._test_parent_offsets: list[torch.Tensor] = []
        self._test_categories: list[torch.Tensor] = []
        self._test_kmer_ids: list[torch.Tensor] = []

    def forward(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        return self.model(kmer_ids, meth_probs)

    def _clamp_kwargs(self) -> dict:
        """Return the model's log-σ clamp range as kwargs for the loss fn."""
        return {
            "log_sigma_min": float(self.model.log_sigma_clamp_min),
            "log_sigma_max": float(self.model.log_sigma_clamp_max),
        }

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        # 7-tuple now; training ignores parent_meth / parent_offset / category.
        kmer_ids, meth_probs, signals, *_extras = batch
        loss = self._loss_fn(
            self.model(kmer_ids, meth_probs),
            signals,
            **self._clamp_kwargs(),
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        (kmer_ids, meth_probs, signals, meth_ids, parent_meths, parent_offsets, categories) = batch
        params = self.model(kmer_ids, meth_probs)
        clamp = self._clamp_kwargs()
        loss = self._loss_fn(params, signals, **clamp)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        mu = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], clamp["log_sigma_min"], clamp["log_sigma_max"])
        sigma = torch.exp(log_sig)
        # ``.float()`` upcasts bf16 → fp32 so ``.numpy()`` works at epoch end
        # (numpy has no native BFloat16 dtype). Without this, bf16-mixed
        # precision crashes in on_validation_epoch_end.
        self._val_mu.append(mu.detach().float().cpu())
        self._val_sigma.append(sigma.detach().float().cpu())
        self._val_true.append(signals.detach().float().cpu())
        self._val_meth_ids.append(meth_ids.detach().cpu())
        self._val_parent_meths.append(parent_meths.detach().cpu())
        self._val_parent_offsets.append(parent_offsets.detach().cpu())
        self._val_categories.append(categories.detach().cpu())
        self._val_kmer_ids.append(kmer_ids.detach().cpu())
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_mu:
            return
        all_mu = torch.cat(self._val_mu).numpy()  # (N, 2)
        all_sigma = torch.cat(self._val_sigma).numpy()  # (N, 2)
        all_true = torch.cat(self._val_true).numpy()  # (N, 2)
        all_meth_ids = torch.cat(self._val_meth_ids).numpy()  # (N,)
        all_pm = torch.cat(self._val_parent_meths).numpy()  # (N,)
        all_po = torch.cat(self._val_parent_offsets).numpy()  # (N,)
        all_cat = torch.cat(self._val_categories).numpy()  # (N,)
        all_kmer = torch.cat(self._val_kmer_ids).numpy()  # (N,)

        val_loss = self.trainer.callback_metrics.get("val_loss", float("nan"))
        gnll_grade = _grade(float(val_loss), 1.0, 1.5, higher_is_better=False)
        log.info(
            "Epoch %d  val_loss(GNLL)=%.4f [%s ≤1.0]",
            self.current_epoch,
            float(val_loss),
            gnll_grade,
        )
        metrics = _compute_metrics(
            all_mu,
            all_sigma,
            all_true,
            all_meth_ids,
            prefix="val",
            kmer_ids=all_kmer,
        )
        bucket_metrics = _compute_bucket_metrics(
            all_mu,
            all_sigma,
            all_true,
            all_cat,
            all_pm,
            all_po,
            prefix="val",
        )
        metrics.update(bucket_metrics)
        self.log_dict(
            {k: v for k, v in metrics.items() if isinstance(v, float)},
            on_epoch=True,
        )
        _log_metrics(metrics, prefix="val")
        _log_bucket_metrics(bucket_metrics, prefix="val")

        self._val_mu.clear()
        self._val_sigma.clear()
        self._val_true.clear()
        self._val_meth_ids.clear()
        self._val_parent_meths.clear()
        self._val_parent_offsets.clear()
        self._val_categories.clear()
        self._val_kmer_ids.clear()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        (kmer_ids, meth_probs, signals, meth_ids, parent_meths, parent_offsets, categories) = batch
        params = self.model(kmer_ids, meth_probs)
        clamp = self._clamp_kwargs()
        mu = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], clamp["log_sigma_min"], clamp["log_sigma_max"])
        sigma = torch.exp(log_sig)
        # ``.float()`` upcasts bf16 → fp32 so ``.numpy()`` works at epoch end.
        self._test_mu.append(mu.detach().float().cpu())
        self._test_sigma.append(sigma.detach().float().cpu())
        self._test_true.append(signals.detach().float().cpu())
        self._test_meth_ids.append(meth_ids.detach().cpu())
        self._test_parent_meths.append(parent_meths.detach().cpu())
        self._test_parent_offsets.append(parent_offsets.detach().cpu())
        self._test_categories.append(categories.detach().cpu())
        self._test_kmer_ids.append(kmer_ids.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_mu:
            return
        all_mu = torch.cat(self._test_mu).numpy()
        all_sigma = torch.cat(self._test_sigma).numpy()
        all_true = torch.cat(self._test_true).numpy()
        all_meth_ids = torch.cat(self._test_meth_ids).numpy()
        all_pm = torch.cat(self._test_parent_meths).numpy()
        all_po = torch.cat(self._test_parent_offsets).numpy()
        all_cat = torch.cat(self._test_categories).numpy()
        all_kmer = torch.cat(self._test_kmer_ids).numpy()

        metrics = _compute_metrics(
            all_mu,
            all_sigma,
            all_true,
            all_meth_ids,
            prefix="test",
            kmer_ids=all_kmer,
        )
        bucket_metrics = _compute_bucket_metrics(
            all_mu,
            all_sigma,
            all_true,
            all_cat,
            all_pm,
            all_po,
            prefix="test",
        )
        metrics.update(bucket_metrics)
        self.log_dict(
            {k: v for k, v in metrics.items() if isinstance(v, float)},
        )
        _log_metrics(metrics, prefix="test")
        _log_bucket_metrics(bucket_metrics, prefix="test")

        self._test_mu.clear()
        self._test_sigma.clear()
        self._test_true.clear()
        self._test_meth_ids.clear()
        self._test_parent_meths.clear()
        self._test_parent_offsets.clear()
        self._test_categories.clear()
        self._test_kmer_ids.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if self.lr_schedule == "cosine":
            # Linear warmup for `warmup_epochs`, then cosine decay over the
            # remaining epochs down to lr * 0.01. Decoupled from val_loss so
            # the LR schedule is fully deterministic and reproducible.
            import math

            max_epochs = max(1, self.max_epochs)
            warmup = max(0, min(self.warmup_epochs, max_epochs - 1))
            min_factor = 0.01

            def _lr_lambda(epoch: int) -> float:
                if warmup > 0 and epoch < warmup:
                    return float(epoch + 1) / float(warmup)
                progress = (epoch - warmup) / max(1, max_epochs - warmup)
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        # Legacy: ReduceLROnPlateau on val_loss.
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
        """Stochastic inference — delegates to ConvPredictor.sample()."""
        return self.model.sample(kmer_ids, meth_probs)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _infer_num_meth_types_from_data(pkl_path: str) -> int:
    """Derive ``num_meth_types`` from the actual training data, ignoring the YAML.

    Strategy (cheapest first):
      1. **First shard's ``__meta__["meth_id_map"]``** — extract writes this
         since v0.4.0. Instant lookup, no data scan.
      2. **Scan all shards' meth-ID columns** (mc_0..mc_10, rev_meth_-1..+1,
         PARENT_METH) and take ``max + 1``. Slow but always correct, used
         when ``__meta__`` is from an older extract.

    Returns at least 4 (the canonical baseline + m6A/m4C/m5C alphabet) so
    pre-existing checkpoints stay loadable even with shards that happen
    to contain only baseline rows.
    """
    from .utils.sample_layout import (
        COL_METH_CTX_START,
        COL_PARENT_METH,
        COL_REV_METH,
        REV_METH_LEN,
        SAMPLE_NCOLS,
    )

    p = Path(pkl_path)
    if p.is_dir():
        paths = [Path(s) for s in list_shards(str(p))]
    else:
        paths = [p]

    if not paths:
        log.warning("No shards/pkl found for num_meth_types inference; defaulting to 4")
        return 4

    # ── Fast path: first-shard __meta__ ───────────────────────────────
    try:
        with open(paths[0], "rb") as f:
            first = pickle.load(f)
        meta = first.get("__meta__", {}) if isinstance(first, dict) else {}
        meth_id_map = meta.get("meth_id_map") if isinstance(meta, dict) else None
        if isinstance(meth_id_map, dict) and meth_id_map:
            n = max(int(v) for v in meth_id_map.values()) + 1
            log.info(
                "num_meth_types from %s __meta__['meth_id_map'] = %s -> n=%d",
                paths[0].name,
                meth_id_map,
                n,
            )
            return max(n, 4)
        del first  # release if we don't return
    except (OSError, pickle.UnpicklingError) as exc:
        log.warning("Could not read __meta__ from %s: %s", paths[0], exc)

    # ── Slow path: scan meth-ID columns across all shards ─────────────
    log.info(
        "No meth_id_map in __meta__; scanning %d shard(s) for max meth_id ...",
        len(paths),
    )
    max_id = 0
    for i, path in enumerate(paths, 1):
        with open(path, "rb") as f:
            data = pickle.load(f)
        for k, arr in data.items():
            if not isinstance(k, (int, np.integer)) or not isinstance(arr, np.ndarray):
                continue
            if arr.size == 0 or arr.shape[1] < 6:
                continue
            # K-aware: mc + rev_meth columns are cols 3..(NCOLS-3). The last
            # 3 cols are CATEGORY / PARENT_METH / PARENT_OFFSET regardless of
            # K. Replaces the legacy ``COL_REV_METH + REV_METH_LEN`` (= 17
            # for K=11). For K=21 it adapts to 27 automatically.
            meth_cols_end = arr.shape[1] - 3
            mc_block_max = int(arr[:, COL_METH_CTX_START:meth_cols_end].max())
            pm_max = int(arr[:, arr.shape[1] - 2].max())  # COL_PARENT_METH
            shard_max = max(mc_block_max, pm_max)
            if shard_max > max_id:
                max_id = shard_max
        log.info("  scanned %d/%d  current max meth_id = %d", i, len(paths), max_id)
        del data

    n = max(max_id + 1, 4)
    log.info("num_meth_types (data scan) = %d (max meth_id seen = %d)", n, max_id)
    return n


def _read_pkl_meta(pkl_path: str) -> dict:
    """Return the ``__meta__`` provenance dict from a training .pkl, or {}.

    For sharded mode (pkl_path is a directory), reads the meta from the
    first ``*_shard*.pkl`` found (matches both raw ``<sample>_shard.pkl``
    and refined ``<sample>_shard_clean.pkl``) — refine writes the same
    global stats (including ``per_bucket`` p_fire) into every shard's
    __meta__.
    """
    p = Path(pkl_path)
    if p.is_dir():
        shard = next(iter(sorted(p.glob("*_shard*.pkl"))), None)
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
    model architecture. Writing it before the first epoch ensures it
    exists even if training is interrupted.

    The config carries:
    - architecture + hyperparameters (from ``model.get_config()``)
    - ``meth_types`` — alphabet the training .pkl was extracted with
    - ``meth_id_map`` — frozen ``{name: int}`` mapping at training time, so
      generate / evaluate can decode integer meth IDs without re-deriving
      from the YAML at inference (the YAML may add new types between train
      and generate, but THIS model's IDs stay pinned in this config).
    - ``p_fire`` and ``mean_occupancy`` per (meth, offset) bucket — feed
      the statistical-firing decomposition at generate time.
    """
    import datetime
    import subprocess

    from . import __version__
    from .utils.encoding import get_meth_ids

    cfg = model.get_config()
    if meth_types is not None:
        cfg["meth_types"] = sorted(meth_types)
    if p_fire:
        cfg["p_fire"] = p_fire
    if mean_occupancy:
        cfg["mean_occupancy"] = mean_occupancy
    # Freeze the meth_id mapping at train time so generate/evaluate use
    # the same integer IDs even if kinsim_config.yaml changes later.
    cfg["meth_id_map"] = get_meth_ids()
    # Provenance: git_sha + kinsim version + UTC timestamp. Lets a reviewer
    # reconstruct the exact code that produced this checkpoint even after
    # the repo evolves. "unknown" sha when not under a git checkout.
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        sha = "unknown"
    cfg["git_sha"] = sha
    cfg["kinsim_version"] = __version__
    cfg["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path = output_dir / "model_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    log.info(
        "Model config saved: %s  (kinsim=%s  sha=%s  meth_types=%s  meth_id_map=%s  "
        "p_fire=%s  mean_occ=%s)",
        path,
        cfg["kinsim_version"],
        cfg["git_sha"][:10],
        cfg.get("meth_types", "all"),
        cfg["meth_id_map"],
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
    optuna_epochs: int = 20,
    batch_size: int = 4096,
    val_fraction: float = 0.10,
    loss_name: str = "betanll",
    device: str = "cuda",
) -> float:
    """Optuna objective — returns best val_loss (GNLL) for a trial.

    Search space: lr, base_embed_dim, conv_dim, head_dim, kernel_size, dropout.
    """
    # Derive from the data, not the YAML — see _infer_num_meth_types_from_data.
    num_meth_types = _infer_num_meth_types_from_data(pkl_path)

    # Read geometry from the shard meta so non-default K (e.g. K=21) works.
    # Otherwise ConvPredictor builds at K=11 default and KineticDataModule
    # raises a shape mismatch on the first batch.
    from .utils.config import get_extraction_params as _get_yaml_params
    _meta = _read_pkl_meta(pkl_path)
    _ext = (_meta or {}).get("extraction_params", {}) if _meta else {}
    _kmer_size = int(_ext.get("kmer_size", 11))
    _active_idx = int(_ext.get("active_site_index", _ext.get("upstream", 7)))
    _n_rev = int(_ext.get("n_rev_meth", 3))
    _params = _get_yaml_params()

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    base_embed_dim = trial.suggest_categorical("base_embed_dim", [8, 16])
    conv_dim = trial.suggest_categorical("conv_dim", [64, 128])
    head_dim = trial.suggest_categorical("head_dim", [64, 128, 256])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    model = ConvPredictor(
        base_embed_dim=base_embed_dim,
        conv_dim=conv_dim,
        head_dim=head_dim,
        kernel_size=kernel_size,
        num_meth_types=num_meth_types,
        dropout=dropout,
        kmer_size=_kmer_size,
        active_site_index=_active_idx,
        n_rev_meth=_n_rev,
    )

    lm = KineticPredictor(model, lr=lr, loss_name=loss_name)
    dm = KineticDataModule(
        input_path=pkl_path,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_meth_types=num_meth_types,
        params=_params,
    )

    callbacks: list = [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE_TRIAL, mode="min")
    ]
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


def train_main(
    pkl_path: str,
    output_dir: str,
    test_pkl: str | None = None,
    test_strains: list | None = None,
    test_fraction: float | None = None,
    split_seed: int = 42,
    epochs: int = 50,
    batch_size: int = 4096,
    lr: float = 1e-3,
    base_embed_dim: int = 16,
    conv_dim: int = 128,
    n_conv_layers: int = 3,
    kernel_size: int = 3,
    head_dim: int = 128,
    meth_proj_dim: int = 8,
    dropout: float = 0.1,
    loss_name: str = "betanll",
    val_fraction: float = 0.10,
    device: str = "cuda",
    resume_ckpt: str | None = None,
    run_optuna: bool = False,
    n_trials: int = 20,
    optuna_epochs: int = 20,
    augment: bool = True,
    balance_kmers: bool = True,
    biology_mask: bool = False,
    log_sigma_clamp_max: float = 1.5,
    lr_schedule: str = "cosine",
    warmup_epochs: int = 3,
) -> None:
    """Train :class:`ConvPredictor` using PyTorch Lightning."""
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

        log.info("Optuna HPO — %d trials × %d epochs", n_trials, optuna_epochs)
        optuna_dir = output_dir / "optuna"
        optuna_dir.mkdir(exist_ok=True)

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            study_name="kinsim",
        )
        study.optimize(
            lambda trial: objective(
                trial,
                pkl_path=pkl_path,
                output_dir=optuna_dir,
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
        lr = best["lr"]
        dropout = best.get("dropout", dropout)
        base_embed_dim = best.get("base_embed_dim", base_embed_dim)
        conv_dim = best.get("conv_dim", conv_dim)
        head_dim = best.get("head_dim", head_dim)
        kernel_size = best.get("kernel_size", kernel_size)
        (output_dir / "optuna_best_params.json").write_text(
            json.dumps({"best_val_loss": study.best_value, **best}, indent=2)
        )

    # ── Build model ───────────────────────────────────────────────────────
    accelerator = "gpu" if device == "cuda" and torch.cuda.is_available() else "cpu"

    # Number of methylation states is **data-driven**, NOT YAML-driven.
    # Reasoning: the YAML at train time may have been edited / drifted /
    # be missing relative to extract time. The shards themselves are the
    # authoritative record of what meth IDs exist in this corpus. We
    # check (in order):
    #   1. __meta__["meth_id_map"] from the first shard, if extract wrote it.
    #   2. Else: scan the meth-ID columns across all shards for the max.
    # The value is persisted in model_config.json via model.get_config()
    # so generate.py reads it back from the trained model at inference,
    # not from any YAML.
    num_meth_types = _infer_num_meth_types_from_data(pkl_path)
    log.info("num_meth_types (data-derived): %d", num_meth_types)

    # Resolve the window geometry from the shards (authoritative) and reject
    # any disagreement with the active YAML. The shard's meta wins because
    # the data was physically extracted under that geometry — training on
    # a model sized to a different geometry would silently produce garbage.
    from .data.dataset import _peek_shard_extraction_params
    from .utils.config import get_extraction_params as _get_yaml_params

    shard_params = None
    if Path(pkl_path).is_dir():
        shards = list_shards(pkl_path)
        if shards:
            shard_params = _peek_shard_extraction_params(shards[0])
    if shard_params is None:
        # Single-pkl mode or pre-v0.5 shard — try the YAML.
        resolved_params = _get_yaml_params()
        log.info(
            "extraction params: read from kinsim_config.yaml — "
            "kmer_size=%d  upstream=%d  downstream=%d  rev_meth=%s",
            resolved_params.kmer_size,
            resolved_params.upstream,
            resolved_params.downstream,
            list(resolved_params.rev_meth_offsets),
        )
    else:
        yaml_params = _get_yaml_params()
        if shard_params != yaml_params:
            log.warning(
                "Shard ExtractionParams differ from kinsim_config.yaml; "
                "shard wins:\n  shard:  %s\n  YAML:   %s",
                shard_params.to_dict(),
                yaml_params.to_dict(),
            )
        resolved_params = shard_params
        log.info(
            "extraction params: read from shards — "
            "kmer_size=%d  upstream=%d  downstream=%d  rev_meth=%s",
            resolved_params.kmer_size,
            resolved_params.upstream,
            resolved_params.downstream,
            list(resolved_params.rev_meth_offsets),
        )

    log.info(
        "Training — %d epochs  loss=%s  lr=%.2e  base_embed=%d  "
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
        num_meth_types=num_meth_types,
        conv_dim=conv_dim,
        n_conv_layers=n_conv_layers,
        kernel_size=kernel_size,
        head_dim=head_dim,
        dropout=dropout,
        biology_mask=biology_mask,
        log_sigma_clamp_max=log_sigma_clamp_max,
        kmer_size=resolved_params.kmer_size,
        active_site_index=resolved_params.active_site_index,
        n_rev_meth=resolved_params.n_rev_meth,
    )

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{n_params:,}")

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
        output_dir,
        model,
        meth_types=pkl_meth_types,
        p_fire=p_fire,
        mean_occupancy=mean_occ,
    )

    if resume_ckpt:
        from .models.predictor import load_state_dict_from_ckpt

        log.info("Loading weights from: %s", resume_ckpt)
        model.load_state_dict(load_state_dict_from_ckpt(resume_ckpt))
        log.info("Weights loaded.")

    lm = KineticPredictor(
        model,
        lr=lr,
        loss_name=loss_name,
        lr_schedule=lr_schedule,
        max_epochs=epochs,
        warmup_epochs=warmup_epochs,
    )
    dm = KineticDataModule(
        input_path=pkl_path,
        test_pkl=test_pkl,
        test_strains=test_strains,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_meth_types=num_meth_types,
        seed=split_seed,
        augment=augment,
        balance_kmers=balance_kmers,
        params=resolved_params,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    early_stop = EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE_FULL, mode="min")
    lightning_ckpt = ModelCheckpoint(
        dirpath=str(output_dir / "lightning_ckpts"),
        filename="ckpt-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    # Second checkpoint tracking μ-only MSE — picks the model that fits the
    # mean best, independent of how σ is being shaped by GNLL/Beta-NLL.
    mu_ckpt = ModelCheckpoint(
        dirpath=str(output_dir / "lightning_ckpts"),
        filename="best_mu-{epoch:03d}-{val_mse_mu:.4f}",
        monitor="val_mse_mu",
        mode="min",
        save_top_k=1,
    )
    # Snapshot the active YAML next to model_config.json so downstream
    # tooling can reproduce the geometry without re-reading the live config.
    try:
        import yaml as _yaml

        from .utils.config import load_kinsim_config as _load_yaml

        (output_dir / "kinsim_config.snapshot.yaml").write_text(
            _yaml.safe_dump(dict(_load_yaml()), sort_keys=False),
            encoding="utf-8",
        )
    except (ImportError, OSError, ValueError) as _exc:
        # ImportError: PyYAML missing; OSError: write failed;
        # ValueError: YAML round-trip failed.
        log.warning("Could not snapshot kinsim_config.yaml: %s", _exc)

    # ── Loggers ───────────────────────────────────────────────────────────
    loggers: list = [CSVLogger(str(output_dir), name="logs")]
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401

        loggers.append(TensorBoardLogger(str(output_dir), name="runs"))
    except ImportError:
        log.warning("TensorBoard not available — CSV logger only.")

    # ── Trainer ───────────────────────────────────────────────────────────
    # ``limit_train_batches`` caps per-epoch iterations. Without it, Lightning
    # iterates the FULL IterableDataset per epoch (~2.5 B rows on a 49-strain
    # corpus → ~50 h/epoch × 50 epochs = unusable). 50 000 batches at batch
    # size 2048 ≈ 100 M rows/epoch ≈ 1 h/epoch on GPU with num_workers=2.
    # That gives ~50 h max for the full 50-epoch budget — but
    # ``early_stop`` will usually cut it off after 10–20 epochs once val_loss
    # plateaus.
    #
    # ``limit_val_batches`` is symmetric for val: val IterableDataset has no
    # __len__, so unbounded val iterates the whole holdout per epoch. With
    # num_workers=1 on val (worker leak avoidance) that's the dominant slowdown.
    # 2 000 batches × 2048 ≈ 4 M val rows — enough for a stable val_loss
    # while keeping val under ~20 min/epoch.
    # Note: bf16-mixed was tried but caused 10-20× slowdown on this
    # cluster's GPUs (likely older Ampere without efficient bf16 + slow
    # autocast). Stay on default fp32 — the data path is already
    # CPU-bound, GPU precision isn't the bottleneck.
    trainer = L.Trainer(
        max_epochs=epochs,
        limit_train_batches=TRAIN_BATCHES_PER_EPOCH,
        limit_val_batches=VAL_BATCHES_PER_EPOCH,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.5,
        callbacks=[early_stop, lightning_ckpt, mu_ckpt],
        logger=loggers,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(lm, datamodule=dm)
    log.info("Training complete. Outputs in: %s", output_dir)

    # Run final test pass when EITHER a separate test pkl OR a sharded-mode
    # holdout (test_strains / test_fraction) was provided. Before this fix,
    # `trainer.test()` only ran for ``test_pkl`` which silently dropped the
    # `--test-strains` holdout — leaving users with no per-meth-type test
    # metrics when they used the sharded mode.
    has_test_set = (
        bool(test_pkl) or bool(test_strains) or (test_fraction is not None and test_fraction > 0)
    )
    if has_test_set:
        log.info("Running evaluation on held-out test set")
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
            "Train ConvPredictor on shard pkls.\n\n"
            "Input: a directory of refined *_shard.pkl (sharded mode, recommended)\n"
            "or a single shard.pkl (small datasets, debugging). Auto-detected.\n\n"
            "Pipeline:\n"
            "  kinsim extract --manifest manifest.csv --task N --output-dir shards/\n"
            "  kinsim refine  shards/   refined/\n"
            "  kinsim train   refined/  checkpoints/\n\n"
            "All flags may be specified in a YAML config file (--config).\n"
            "Command-line flags override YAML values."
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

    # Training hyperparameters
    parser.add_argument(
        "--config", default=None, help="YAML config file (all flags can be set here)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 4096)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-3)")

    parser.add_argument(
        "--base-embed-dim",
        type=int,
        default=None,
        help="Per-base embedding dimension (default: 16)",
    )
    parser.add_argument(
        "--conv-dim", type=int, default=None, help="Conv channel width (default: 128)"
    )
    parser.add_argument(
        "--n-conv-layers", type=int, default=None, help="Number of conv layers (default: 3)"
    )
    parser.add_argument(
        "--kernel-size", type=int, default=None, help="Conv kernel size (default: 3)"
    )
    parser.add_argument(
        "--head-dim", type=int, default=None, help="Head hidden layer width (default: 128)"
    )
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
        help="Dropout probability (default: 0.1)",
    )
    parser.add_argument(
        "--loss",
        default=None,
        choices=["gnll", "betanll", "betanll_0.3", "betanll_0.5", "betanll_1.0", "mse", "huber"],
        help="Loss function: gnll=Gaussian NLL, betanll=Beta-NLL β=0.5 (default), "
        "betanll_<β> for explicit β, mse, huber",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Fraction for validation split (default: 0.10)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device (default: cuda, falls back to cpu automatically)",
    )
    parser.add_argument(
        "--resume", dest="resume_ckpt", help="Resume weights from a checkpoint .ckpt or .pt file"
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

    # ── Accuracy improvements (all ON by default — flags opt OUT) ────────
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        default=None,
        help="Disable paired-positive augmentation (default: ON). When on, "
        "each non-baseline row is paired with a real baseline of the "
        "same kmer — forces the meth/no-meth contrast on the same "
        "sequence (Khosla 2020).",
    )
    parser.add_argument(
        "--no-balance-kmers",
        dest="balance_kmers",
        action="store_false",
        default=None,
        help="Disable per-(kmer, category) inverse-frequency sampling "
        "(default: ON). When on, rare kmers / rare categories get "
        "proportionally more gradient (He+Garcia 2009, Cui 2019).",
    )
    parser.add_argument(
        "--no-biology-mask",
        dest="biology_mask",
        action="store_false",
        default=None,
        help="Disable architectural biology mask (default: OFF per "
        "kinsim_config.yaml, model.biology_mask=false — v0.5.0 had a "
        "subtle synthesized-vs-template-strand bug; extract already "
        "enforces base/meth chemistry so the mask is a redundant "
        "safety net). Pass --biology-mask to force it ON.",
    )
    parser.add_argument(
        "--log-sigma-clamp-max",
        type=float,
        default=None,
        help="Upper clamp on log σ (default: 1.5). Tighten to prevent the "
        "model from gaming σ instead of fitting μ. Legacy was 3.0.",
    )
    parser.add_argument(
        "--lr-schedule",
        default=None,
        choices=["cosine", "plateau"],
        help="LR schedule: cosine (default) with linear warmup + cosine "
        "decay to 1%% lr, or plateau (legacy ReduceLROnPlateau).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Linear warmup epochs for cosine schedule (default: 3).",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    # Global determinism. Seeds torch + numpy + Python random + Lightning DataLoader
    # workers. Same shard set + same hyperparameters now produce the same weights.
    seed = args.split_seed if args.split_seed is not None else 42
    L.seed_everything(int(seed), workers=True)

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

    # Parse --test-strains "a,b,c" into a list (sharded mode only).
    test_strains_arg = args.test_strains or cfg.get("test_strains")
    if isinstance(test_strains_arg, str):
        test_strains_list = [s.strip() for s in test_strains_arg.split(",") if s.strip()]
    elif isinstance(test_strains_arg, list):
        test_strains_list = list(test_strains_arg)
    else:
        test_strains_list = None

    train_main(
        pkl_path=pkl_path,
        output_dir=output_dir,
        test_pkl=args.test_pkl or cfg.get("test_pkl"),
        test_strains=test_strains_list,
        test_fraction=_get(args.test_fraction, "test_fraction", None),
        split_seed=_get(args.split_seed, "split_seed", 42),
        epochs=_get(args.epochs, "epochs", 50),
        batch_size=_get(args.batch_size, "batch_size", 4096),
        lr=_get(args.lr, "lr", 1e-3),
        base_embed_dim=_get(args.base_embed_dim, "base_embed_dim", 16),
        conv_dim=_get(args.conv_dim, "conv_dim", 128),
        n_conv_layers=_get(args.n_conv_layers, "n_conv_layers", 3),
        kernel_size=_get(args.kernel_size, "kernel_size", 3),
        head_dim=_get(args.head_dim, "head_dim", 128),
        meth_proj_dim=_get(args.meth_proj_dim, "meth_proj_dim", 8),
        dropout=_get(args.dropout, "dropout", 0.1),
        loss_name=_get(args.loss, "loss", "betanll"),
        val_fraction=_get(args.val_fraction, "val_fraction", 0.10),
        device=_get(args.device, "device", "cuda"),
        resume_ckpt=args.resume_ckpt or cfg.get("resume"),
        run_optuna=args.optuna or cfg.get("optuna", False),
        n_trials=_get(args.n_trials, "n_trials", 20),
        optuna_epochs=_get(args.optuna_epochs, "optuna_epochs", 20),
        augment=_get(args.augment, "augment", True),
        balance_kmers=_get(args.balance_kmers, "balance_kmers", True),
        biology_mask=_get(args.biology_mask, "biology_mask", False),
        log_sigma_clamp_max=_get(args.log_sigma_clamp_max, "log_sigma_clamp_max", 1.5),
        lr_schedule=_get(args.lr_schedule, "lr_schedule", "cosine"),
        warmup_epochs=_get(args.warmup_epochs, "warmup_epochs", 3),
    )


if __name__ == "__main__":
    main()
