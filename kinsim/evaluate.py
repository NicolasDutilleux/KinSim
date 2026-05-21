"""``kinsim evaluate`` — post-training calibration report.

Usage
-----
::

    kinsim evaluate <checkpoint_dir> <shard.pkl>
        Full calibration report over the dataset. Prints per-metric numbers
        and saves evaluation_report.txt.

    kinsim evaluate <checkpoint_dir> <shard.pkl> \\
        --kmer GATCGATCGAT --meth m6A --plot gatc_m6A.png
        Plot predicted N(μ, σ²) vs actual observations for one specific
        (kmer, methylation) context.

Metrics
-------
Pearson r
    Correlation between predicted mean μ and actual signals across ALL contexts
    in the dataset (one random sample per key).  Target: > 0.9.

Pearson oracle
    Theoretical ceiling: r_oracle = Var(μ) / (Var(μ) + E[σ²]).
    Even a perfect model cannot exceed this — σ² is irreducible noise.

Random-from-distribution Pearson
    Sample z ~ N(μ, σ²) for each context, then Pearson(z, true).
    This is the noise floor: what you'd get by randomly drawing from the
    correct distribution.  Model Pearson should be well above this.

Efficiency
    (r_model − r_random) / (r_oracle − r_random) × 100%.
    Shows what fraction of the "oracle gap" the model captures.

MAE (log-space)
    Mean Absolute Error in log1p space.  0.0 is perfect; ~0.1 is excellent.

2σ Calibration
    Fraction of actual observations falling within [μ − 2σ, μ + 2σ].
    Expected for a correctly calibrated Gaussian: 95.4 %.
    - Below 90 %: model underestimates uncertainty (σ too small, overconfident)
    - Above 99 %: model over-disperses (σ too large, too conservative)

Per-methylation-type breakdown
    Pearson, MAE, calibration, and σ broken down by none/m6A/m4C/m5C.
    Methylated types should show larger σ (learned heteroscedasticity).

σ Histogram
    Distribution of predicted σ values.  A well-trained model shows different σ
    for methylated vs unmethylated contexts — it has learned that m6A pauses are
    noisier than background.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import SignalDataset
from .models.predictor import (
    ConvPredictor,
    create_from_config,
    load_state_dict_from_ckpt,
)
from .utils.encoding import METH_IDS, encode_kmer

log = logging.getLogger(__name__)

_SIGMA_CLAMP_LEGACY = (-6.0, 3.0)  # legacy default — replaced at runtime by model config


def _sigma_clamp_from_model(model) -> tuple[float, float]:
    """Return (min, max) log-sigma clamp matching what the model produced.

    ``getattr(model, name, default)`` is total: it never raises. The clamp
    must match what the model emitted at training time so calibration /
    2σ stats line up with what ``generate.sample()`` consumes.
    """
    return (
        float(getattr(model, "log_sigma_clamp_min", _SIGMA_CLAMP_LEGACY[0])),
        float(getattr(model, "log_sigma_clamp_max", _SIGMA_CLAMP_LEGACY[1])),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Pick the most-recent checkpoint, preferring Lightning ``.ckpt`` files.

    Search order:
      1. ``lightning_ckpts/last.ckpt`` (always the most recent Lightning epoch).
      2. ``lightning_ckpts/*.ckpt`` sorted by mtime.
      3. ``checkpoint_epoch*.pt`` sorted by epoch number (legacy).
    """
    lightning_dir = checkpoint_dir / "lightning_ckpts"
    if (lightning_dir / "last.ckpt").exists():
        return lightning_dir / "last.ckpt"
    ckpts = sorted(lightning_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        return ckpts[-1]

    def _epoch_num(p: Path) -> int:
        try:
            return int(p.stem.removeprefix("checkpoint_epoch"))
        except ValueError:
            return -1

    pts = list(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    if pts:
        return max(pts, key=_epoch_num)
    raise FileNotFoundError(
        f"No checkpoint files found in {checkpoint_dir} "
        f"(looked for lightning_ckpts/*.ckpt and checkpoint_epoch*.pt)"
    )


def _load_model(checkpoint_dir: str | Path, device: torch.device) -> torch.nn.Module:
    """Load a ConvPredictor from a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    cfg_path = checkpoint_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found in {checkpoint_dir}.\n"
            "Ensure training completed at least one epoch."
        )
    cfg = json.loads(cfg_path.read_text())

    model = create_from_config(cfg).to(device)
    ckpt_path = _find_latest_checkpoint(checkpoint_dir)
    model.load_state_dict(load_state_dict_from_ckpt(ckpt_path))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "Model loaded: params=%s  checkpoint=%s",
        f"{n_params:,}",
        ckpt_path.name,
    )
    return model


# ---------------------------------------------------------------------------
# Full calibration report
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: ConvPredictor,
    pkl_path: str | Path,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, float]:
    """Compute the full evaluation suite on a merged dataset.

    Iterates over every (kmer, meth) key in the dataset, draws one random
    signal per key, runs the model, and aggregates:

        - MAE, MSE in log1p space
        - Pearson r between predicted μ and actual signal
        - 2σ calibration coverage
        - Median and mean predicted σ (to check heteroscedasticity)

    Args:
        model:      ConvPredictor in eval mode.
        pkl_path:   Path to merged .pkl file.
        device:     Torch device.
        batch_size: Inference batch size.

    Returns:
        Dictionary of metric names → float values.
    """
    dataset = SignalDataset(str(pkl_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_mu = []
    all_sigma = []
    all_true = []
    all_meth_ids = []
    sigma_clamp = _sigma_clamp_from_model(model)

    for kmer_ids, meth_probs, signals, meth_ids, *_extras in loader:
        kmer_ids = kmer_ids.to(device)
        meth_probs = meth_probs.to(device)

        params = model(kmer_ids, meth_probs)
        mu = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], *sigma_clamp)
        sigma = torch.exp(log_sig)

        all_mu.append(mu.cpu().numpy())
        all_sigma.append(sigma.cpu().numpy())
        all_true.append(signals.numpy())
        all_meth_ids.append(meth_ids.numpy())

    mu = np.concatenate(all_mu, axis=0)  # (N, 2) log1p space
    sigma = np.concatenate(all_sigma, axis=0)  # (N, 2) log1p space
    true = np.concatenate(all_true, axis=0)  # (N, 2) log1p space
    meth_ids = np.concatenate(all_meth_ids, axis=0)  # (N,)

    diff = mu - true
    mse = (diff**2).mean(axis=0)
    mae = np.abs(diff).mean(axis=0)

    from .utils.encoding import get_meth_ids as _gmi
    _METH_NAMES = {v: k for k, v in _gmi().items()}

    def _pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0

    # Calibration at 1σ, 2σ, 3σ (expected: 68.3%, 95.4%, 99.7%)
    def _calib(n_sigma, d=diff, s=sigma):
        return (np.abs(d) <= n_sigma * s).mean(axis=0)

    calib_1s = _calib(1)
    calib_2s = _calib(2)
    calib_3s = _calib(3)

    # ── Pearson oracle: theoretical ceiling for a distributional model ────
    # r_oracle = Var(μ) / (Var(μ) + E[σ²])
    # Even a *perfect* model cannot exceed this because σ² is irreducible noise.
    var_mu_ipd = np.var(mu[:, 0])
    var_mu_pw = np.var(mu[:, 1])
    e_sig2_ipd = np.mean(sigma[:, 0] ** 2)
    e_sig2_pw = np.mean(sigma[:, 1] ** 2)
    oracle_ipd = var_mu_ipd / (var_mu_ipd + e_sig2_ipd) if (var_mu_ipd + e_sig2_ipd) > 0 else 0.0
    oracle_pw = var_mu_pw / (var_mu_pw + e_sig2_pw) if (var_mu_pw + e_sig2_pw) > 0 else 0.0

    # ── Random-from-distribution baseline ─────────────────────────────────
    # Sample z ~ N(μ, σ²) and compute Pearson(z, true).
    # This is what you'd get if you randomly drew from the *correct* distribution.
    # It shows the noise floor: even perfect distributions yield low Pearson
    # when σ is large relative to the spread of μ.
    rng = np.random.default_rng(42)
    z_random = mu + sigma * rng.standard_normal(mu.shape).astype(np.float32)
    rand_pearson_ipd = _pearson(z_random[:, 0], true[:, 0])
    rand_pearson_pw = _pearson(z_random[:, 1], true[:, 1])

    # ── Per-meth-type breakdown ───────────────────────────────────────────
    by_type = {}
    for mid in sorted(np.unique(meth_ids)):
        mask = meth_ids == mid
        if mask.sum() < 2:
            continue
        name = _METH_NAMES.get(int(mid), f"meth{int(mid)}")
        mu_m, sig_m, true_m = mu[mask], sigma[mask], true[mask]
        diff_m = mu_m - true_m
        c1 = (np.abs(diff_m) <= 1.0 * sig_m).mean(axis=0)
        c2 = (np.abs(diff_m) <= 2.0 * sig_m).mean(axis=0)
        c3 = (np.abs(diff_m) <= 3.0 * sig_m).mean(axis=0)
        by_type[name] = {
            "n": int(mask.sum()),
            "pearson_ipd": _pearson(mu_m[:, 0], true_m[:, 0]),
            "pearson_pw": _pearson(mu_m[:, 1], true_m[:, 1]),
            "mae_ipd": float(np.abs(diff_m[:, 0]).mean()),
            "mae_pw": float(np.abs(diff_m[:, 1]).mean()),
            "calib_1s": float((c1[0] + c1[1]) / 2),
            "calib_2s": float((c2[0] + c2[1]) / 2),
            "calib_3s": float((c3[0] + c3[1]) / 2),
            "mean_sigma_ipd": float(sig_m[:, 0].mean()),
            "mean_sigma_pw": float(sig_m[:, 1].mean()),
        }

    return {
        # Mean / spread quality
        "mse_ipd": float(mse[0]),
        "mse_pw": float(mse[1]),
        "mae_ipd": float(mae[0]),
        "mae_pw": float(mae[1]),
        "pearson_ipd": _pearson(mu[:, 0], true[:, 0]),
        "pearson_pw": _pearson(mu[:, 1], true[:, 1]),
        # Pearson oracle (theoretical ceiling)
        "oracle_ipd": float(oracle_ipd),
        "oracle_pw": float(oracle_pw),
        # Random-from-distribution baseline (noise floor)
        "rand_pearson_ipd": rand_pearson_ipd,
        "rand_pearson_pw": rand_pearson_pw,
        # Calibration coverage
        "calib_1s_ipd": float(calib_1s[0]),
        "calib_1s_pw": float(calib_1s[1]),
        "calib_2s_ipd": float(calib_2s[0]),
        "calib_2s_pw": float(calib_2s[1]),
        "calib_3s_ipd": float(calib_3s[0]),
        "calib_3s_pw": float(calib_3s[1]),
        # Heteroscedasticity check — σ spread
        "mean_sigma_ipd": float(sigma[:, 0].mean()),
        "mean_sigma_pw": float(sigma[:, 1].mean()),
        "median_sigma_ipd": float(np.median(sigma[:, 0])),
        "median_sigma_pw": float(np.median(sigma[:, 1])),
        # Per-type breakdown
        "by_type": by_type,
        "n_contexts": len(mu),
    }


def print_report(metrics: dict) -> str:
    """Format evaluation metrics as a human-readable report string."""
    by_type = metrics.get("by_type", {})

    lines = [
        "=" * 70,
        "  KinSim — Evaluation Report",
        "=" * 70,
        "",
        f"  Contexts evaluated : {metrics['n_contexts']:,}",
        "",
        "  ── Mean prediction quality (log1p space) ─────────────────────",
        f"  MAE   IPD / PW  :  {metrics['mae_ipd']:.4f}  /  {metrics['mae_pw']:.4f}",
        f"  MSE   IPD / PW  :  {metrics['mse_ipd']:.4f}  /  {metrics['mse_pw']:.4f}",
        f"  Pearson IPD / PW:  {metrics['pearson_ipd']:.4f}  /  {metrics['pearson_pw']:.4f}",
        "",
        "  ── Pearson context ────────────────────────────────────────────",
        f"  Model Pearson     IPD = {metrics['pearson_ipd']:.4f}    PW = {metrics['pearson_pw']:.4f}",
        f"  Oracle (ceiling)  IPD = {metrics['oracle_ipd']:.4f}    PW = {metrics['oracle_pw']:.4f}",
        f"  Random sample     IPD = {metrics['rand_pearson_ipd']:.4f}    PW = {metrics['rand_pearson_pw']:.4f}",
        "",
        "  Oracle = Var(μ)/(Var(μ)+E[σ²]) — max achievable Pearson.",
        "  Random = Pearson between z~N(μ,σ²) and truth — noise floor.",
        "  Model should be close to Oracle and well above Random.",
    ]

    # Efficiency ratio: how close model is to oracle vs random
    for signal in ("ipd", "pw"):
        r_model = metrics[f"pearson_{signal}"]
        r_oracle = metrics[f"oracle_{signal}"]
        r_random = metrics[f"rand_pearson_{signal}"]
        if r_oracle - r_random > 1e-6:
            efficiency = (r_model - r_random) / (r_oracle - r_random)
            lines.append(
                f"  Efficiency {signal.upper():>3s}     {efficiency * 100:.1f}% of oracle gap captured"
            )
        else:
            lines.append(f"  Efficiency {signal.upper():>3s}     N/A (oracle ≈ random)")

    lines += [
        "",
        "  ── Calibration coverage (% within nσ of μ) ──────────────────",
        "  Coverage     IPD     PW    Expected",
        f"  1σ (68%)  {metrics['calib_1s_ipd'] * 100:6.1f}%  {metrics['calib_1s_pw'] * 100:6.1f}%    68.3%",
        f"  2σ (95%)  {metrics['calib_2s_ipd'] * 100:6.1f}%  {metrics['calib_2s_pw'] * 100:6.1f}%    95.4%",
        f"  3σ (99%)  {metrics['calib_3s_ipd'] * 100:6.1f}%  {metrics['calib_3s_pw'] * 100:6.1f}%    99.7%",
        "",
        "  Interpretation: 2σ ≈ 95% = well-calibrated.",
        "    > 98%: σ too large (underconfident). < 90%: σ too small (overconfident).",
        "",
        "  ── Predicted σ (log1p space) — heteroscedasticity ───────────",
        f"  Mean σ  IPD / PW  :  {metrics['mean_sigma_ipd']:.4f}  /  {metrics['mean_sigma_pw']:.4f}",
        f"  Median σ IPD / PW :  {metrics['median_sigma_ipd']:.4f}  /  {metrics['median_sigma_pw']:.4f}",
    ]

    if by_type:
        lines += [
            "",
            "  ── Per-methylation-type breakdown ────────────────────────────",
            f"  {'Type':<8} {'N':>8}  {'r_IPD':>6} {'r_PW':>6}  {'MAE_I':>6} {'MAE_P':>6}"
            f"  {'2σ%':>5}  {'σ_IPD':>6} {'σ_PW':>6}",
            "  " + "-" * 66,
        ]
        for name in ["none", "m6A", "m4C", "m5C"]:
            if name not in by_type:
                continue
            t = by_type[name]
            lines.append(
                f"  {name:<8} {t['n']:>8}  {t['pearson_ipd']:>6.3f} {t['pearson_pw']:>6.3f}"
                f"  {t['mae_ipd']:>6.4f} {t['mae_pw']:>6.4f}"
                f"  {t['calib_2s'] * 100:>4.1f}%"
                f"  {t['mean_sigma_ipd']:>6.4f} {t['mean_sigma_pw']:>6.4f}"
            )
        lines += [
            "",
            "  Methylated types should show larger σ than 'none' (learned",
            "  that modified bases have higher variance in kinetic signals).",
        ]

    lines += ["", "=" * 70]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# K-mer distribution plot
# ---------------------------------------------------------------------------


@torch.no_grad()
def plot_kmer_distribution(
    model: ConvPredictor,
    data: dict,
    kmer_str: str,
    meth_name: str = "none",
    device: torch.device = torch.device("cpu"),
    output_path: str | None = None,
) -> None:
    """Plot predicted N(μ, σ²) vs actual observations for one (kmer, meth) context.

    Produces a 2-panel figure (IPD left, PW right), each showing:
      - Histogram of actual log1p values from the training data
      - Predicted Gaussian PDF (red curve)
      - μ ± 2σ interval (orange dashed lines)

    Args:
        model:       ConvPredictor in eval mode.
        data:        Raw dict from .pkl file: (kmer_id, meth_id) → np.ndarray(N, 2/3/14).
        kmer_str:    11-mer string, e.g. "GGATCCTGCAT".
        meth_name:   One of "none", "m6A", "m4C", "m5C".
        device:      Torch device for model inference.
        output_path: If given, save figure to this path; otherwise display interactively.

    Raises:
        KeyError:   If (kmer_str, meth_name) is not present in data.
        ValueError: If kmer_str is not exactly 11 bases.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    from .utils.encoding import K as _K
    from .utils.encoding import get_meth_ids as _gmi
    from .utils.sample_layout import (
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
        COL_PARENT_METH,
        COL_PW,
    )
    _METH_IDS_RUNTIME = _gmi()
    # K-aware: prefer the model's actual kmer_size, fall back to module K.
    _ckpt_k = int(model.get_config().get("kmer_size", _K))
    if len(kmer_str) != _ckpt_k:
        raise ValueError(f"kmer_str must be exactly {_ckpt_k} bases, got {len(kmer_str)}")
    if meth_name not in _METH_IDS_RUNTIME:
        raise ValueError(
            f"meth_name must be one of {list(_METH_IDS_RUNTIME)}, got '{meth_name}'"
        )

    kmer_id = encode_kmer(kmer_str)
    meth_id = _METH_IDS_RUNTIME[meth_name]

    # Current shard format: int kmer_id → ndarray with meth in COL_PARENT_METH.
    # Filter rows of THIS kmer where the parent meth matches the requested name,
    # category=SLOWED (the rows the user actually wants to see for non-none
    # meth_name; for meth_name="none" we filter to BASELINE rows).
    if kmer_id not in data or not isinstance(data[kmer_id], np.ndarray):
        raise KeyError(
            f"k-mer '{kmer_str}' not found in dataset (int-keyed format)."
        )
    arr = data[kmer_id]
    if meth_id > 0:
        sel = (arr[:, COL_CATEGORY].astype(np.int8) == CATEGORY_SLOWED) & (
            arr[:, COL_PARENT_METH].astype(np.int8) == meth_id
        )
    else:
        from .utils.sample_layout import CATEGORY_BASELINE
        sel = arr[:, COL_CATEGORY].astype(np.int8) == CATEGORY_BASELINE
    if not sel.any():
        raise KeyError(
            f"Context '{kmer_str}' / '{meth_name}' has 0 matching rows."
        )
    samples = arr[sel].astype(np.float32)
    actual_raw = np.stack([samples[:, COL_IPD], samples[:, COL_PW]], axis=1)
    actual_log = np.log1p(actual_raw)

    # Build a 3-D meth_probs (B, K+rev, M) — ConvPredictor refuses 2-D input.
    # Mark the active site (slot KMER_PRED_IDX) with the meth_id's full prob.
    from .utils.encoding import KMER_PRED_IDX
    _M = len(_METH_IDS_RUNTIME)  # number of meth types (incl. none)
    mcfg = model.get_config()
    _kmer_size = int(mcfg.get("kmer_size", _K))
    _n_rev = int(mcfg.get("n_rev_meth", 3))
    meth_probs = torch.zeros(1, _kmer_size + _n_rev, _M, device=device)
    if meth_id > 0:
        meth_probs[0, KMER_PRED_IDX, meth_id] = 1.0
    kmer_tensor = torch.tensor([kmer_id], dtype=torch.long, device=device)

    params = model(kmer_tensor, meth_probs)  # (1, 4)
    mu_log = params[0, :2].cpu().numpy()  # [μ_ipd, μ_pw] log1p
    log_sig = np.clip(params[0, 2:].cpu().numpy(), *_sigma_clamp_from_model(model))
    sigma = np.exp(log_sig)  # [σ_ipd, σ_pw] log1p

    # Gaussian PDF (pure numpy, no scipy dependency)
    def _gauss_pdf(x, mu, sig):
        return (1.0 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = ["IPD", "PW"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        col = actual_log[:, i]
        mu_i = mu_log[i]
        sig_i = sigma[i]

        # Histogram of actual data
        ax.hist(
            col,
            bins=50,
            density=True,
            alpha=0.55,
            color="steelblue",
            edgecolor="white",
            linewidth=0.4,
            label=f"Actual  (n={len(col):,})",
        )

        # Predicted Gaussian PDF
        x_range = np.linspace(
            min(col.min(), mu_i - 4 * sig_i),
            max(col.max(), mu_i + 4 * sig_i),
            500,
        )
        ax.plot(
            x_range,
            _gauss_pdf(x_range, mu_i, sig_i),
            "r-",
            lw=2.5,
            label=f"Predicted  μ={mu_i:.3f}  σ={sig_i:.3f}",
        )

        # μ ± 2σ shaded region
        x_fill = np.linspace(mu_i - 2 * sig_i, mu_i + 2 * sig_i, 300)
        ax.fill_between(
            x_fill,
            _gauss_pdf(x_fill, mu_i, sig_i),
            alpha=0.18,
            color="orange",
            label="μ ± 2σ  (~95.4%)",
        )
        ax.axvline(mu_i - 2 * sig_i, color="orange", ls="--", lw=1.4)
        ax.axvline(mu_i + 2 * sig_i, color="orange", ls="--", lw=1.4)
        ax.axvline(mu_i, color="red", ls="-", lw=1.2, alpha=0.7)

        # Calibration annotation
        in_2s = float(np.mean(np.abs(col - mu_i) <= 2 * sig_i))
        ax.text(
            0.97,
            0.95,
            f"2σ coverage: {in_2s * 100:.1f}%\n(expected 95.4%)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="darkorange",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Signal value (log1p space)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{label}  ·  {kmer_str}  [{meth_name}]", fontsize=11)
        ax.legend(fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Convert μ back to linear space for the subtitle
    mu_linear = np.expm1(mu_log)
    fig.suptitle(
        f"KinSim MLP — Predicted distribution vs actual data\n"
        f"μ_IPD ≈ {mu_linear[0]:.1f}   μ_PW ≈ {mu_linear[1]:.1f}  (linear scale)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info("Plot saved: %s", output_path)
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    import argparse

    from .utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="kinsim evaluate",
        description=(
            "Evaluate a trained kinetic predictor and inspect its predicted distributions.\n\n"
            "Full calibration report:\n"
            "  kinsim evaluate checkpoints/ shard.pkl\n\n"
            "Distribution plot for one (kmer, meth):\n"
            "  kinsim evaluate checkpoints/ shard.pkl \\\n"
            "      --kmer GGATCCTGCAT --meth m6A --plot gatc_m6A.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_dir", help="Directory with model_config.json and checkpoint_epoch*.pt"
    )
    parser.add_argument("pkl", help="Shard .pkl from kinsim extract (or refined)")
    parser.add_argument("--kmer", default=None, help="11-mer string to inspect (e.g. GGATCCTGCAT)")
    parser.add_argument(
        "--meth",
        default="none",
        choices=list(METH_IDS),
        help="Methylation state for the k-mer plot (default: none)",
    )
    parser.add_argument(
        "--plot",
        default=None,
        metavar="FILE",
        help="Save distribution plot to FILE instead of displaying it",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="TXT",
        help="Save full report to FILE (default: <checkpoint_dir>/evaluation_report.txt)",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log.info("Device: %s", device)

    model = _load_model(args.checkpoint_dir, device)

    if args.kmer:
        # ── Plot one k-mer distribution ─────────────────────────────────────
        log.info("Loading data for k-mer plot: %s", args.pkl)
        with open(args.pkl, "rb") as f:
            import pickle

            raw_data = pickle.load(f)

        # Keep int-keyed entries (current shard format); drop __meta__ etc.
        data = {
            k: v
            for k, v in raw_data.items()
            if isinstance(k, (int, np.integer)) and isinstance(v, np.ndarray)
        }

        plot_kmer_distribution(
            model,
            data,
            kmer_str=args.kmer,
            meth_name=args.meth,
            device=device,
            output_path=args.plot,
        )

    else:
        # ── Full calibration report ──────────────────────────────────────────
        log.info("Running full evaluation on: %s", args.pkl)
        metrics = evaluate(model, args.pkl, device, batch_size=args.batch_size)
        report = print_report(metrics)
        print(report)

        out_path = args.output or str(Path(args.checkpoint_dir) / "evaluation_report.txt")
        Path(out_path).write_text(report)
        log.info("Report saved: %s", out_path)
