"""Post-training evaluation and visualisation for the MLPPredictor.

Commands
--------
kinsim mlp evaluate <checkpoint_dir> <master_data.pkl>
    Full calibration report over the entire dataset.
    Prints per-metric numbers and saves evaluation_report.txt.

kinsim mlp evaluate <checkpoint_dir> <master_data.pkl> \\
    --kmer GATCGATCGAT --meth m6A --plot gatc_m6A.png
    Plot the predicted N(μ, σ²) distribution vs actual observations for one
    specific (k-mer, methylation) context.

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
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import MLPSignalDataset
from .models.predictor import MLPPredictor, create_from_config
from .utils.encoding import METH_IDS, encode_kmer

log = logging.getLogger(__name__)

_SIGMA_CLAMP = (-6.0, 3.0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(checkpoint_dir: str | Path, device: torch.device) -> torch.nn.Module:
    """Load model from a checkpoint directory (supports conv and mlp)."""
    checkpoint_dir = Path(checkpoint_dir)
    cfg_path = checkpoint_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found in {checkpoint_dir}.\n"
            "Ensure training completed at least one epoch."
        )
    cfg = json.loads(cfg_path.read_text())

    model = create_from_config(cfg).to(device)

    # Find the latest checkpoint
    pts = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    if not pts:
        raise FileNotFoundError(f"No checkpoint_epoch*.pt files found in {checkpoint_dir}")
    ckpt_path = pts[-1]

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    arch = cfg.get("architecture", "mlp")
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "Model loaded: architecture=%s  params=%s  checkpoint=%s",
        arch,
        f"{n_params:,}",
        ckpt_path.name,
    )
    return model


# ---------------------------------------------------------------------------
# Full calibration report
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: MLPPredictor,
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
        model:      MLPPredictor in eval mode.
        pkl_path:   Path to merged .pkl file.
        device:     Torch device.
        batch_size: Inference batch size.

    Returns:
        Dictionary of metric names → float values.
    """
    dataset = MLPSignalDataset(str(pkl_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_mu = []
    all_sigma = []
    all_true = []
    all_meth_ids = []

    for kmer_ids, meth_probs, signals, meth_ids in loader:
        kmer_ids = kmer_ids.to(device)
        meth_probs = meth_probs.to(device)

        params = model(kmer_ids, meth_probs)
        mu = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], *_SIGMA_CLAMP)
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

    _METH_NAMES = {v: k for k, v in METH_IDS.items()}

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
    model: MLPPredictor,
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
        model:       MLPPredictor in eval mode.
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

    if len(kmer_str) != 11:
        raise ValueError(f"kmer_str must be exactly 11 bases, got {len(kmer_str)}")
    if meth_name not in METH_IDS:
        raise ValueError(f"meth_name must be one of {list(METH_IDS)}, got '{meth_name}'")

    kmer_id = encode_kmer(kmer_str)
    meth_id = METH_IDS[meth_name]
    key = (kmer_id, meth_id)

    if key not in data:
        raise KeyError(
            f"Context '{kmer_str}' / '{meth_name}' not found in dataset.\n"
            f"Available meth states for this k-mer: "
            + str([m_name for m_name, m_id in METH_IDS.items() if (kmer_id, m_id) in data])
        )

    # Actual data: raw uint8 → log1p (use only IPD/PW columns)
    samples = data[key].astype(np.float32)
    actual_raw = samples[:, :2]  # (N, 2) [IPD, PW]
    actual_log = np.log1p(actual_raw)  # (N, 2) log1p space

    # Build stoichiometric meth_probs from stored fraction (3rd column)
    # For legacy 2-column data, default to 1.0 for methylated.
    if samples.shape[1] >= 3:
        fraction = float(samples[0, 2])
    else:
        fraction = 1.0 if meth_id > 0 else 0.0

    kmer_tensor = torch.tensor([kmer_id], dtype=torch.long, device=device)
    meth_probs = torch.zeros(1, 4, device=device)
    if meth_id > 0:
        meth_probs[0, meth_id] = fraction

    params = model(kmer_tensor, meth_probs)  # (1, 4)
    mu_log = params[0, :2].cpu().numpy()  # [μ_ipd, μ_pw] log1p
    log_sig = np.clip(params[0, 2:].cpu().numpy(), *_SIGMA_CLAMP)
    sigma = np.exp(log_sig)  # [σ_ipd, σ_pw] log1p

    # Gaussian PDF (pure numpy, no scipy dependency)
    def _gauss_pdf(x, mu, sig):
        return (1.0 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = ["IPD", "PW"]

    for i, (ax, label) in enumerate(zip(axes, labels, strict=False)):
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
# Baseline comparison
# ---------------------------------------------------------------------------


def _evaluate_predictions(
    mu: np.ndarray,
    sigma: np.ndarray,
    true: np.ndarray,
    meth_ids: np.ndarray,
) -> dict:
    """Compute metrics from pre-computed predictions (shared by main + baselines).

    All arrays in log1p space. mu/sigma/true: (N, 2), meth_ids: (N,).
    """
    _METH_NAMES = {v: k for k, v in METH_IDS.items()}

    def _pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0

    diff = mu - true
    (diff**2).mean(axis=0)
    mae = np.abs(diff).mean(axis=0)

    def _calib(n_sigma):
        return (np.abs(diff) <= n_sigma * sigma).mean(axis=0)

    calib_2s = _calib(2)

    return {
        "mae_ipd": float(mae[0]),
        "mae_pw": float(mae[1]),
        "pearson_ipd": _pearson(mu[:, 0], true[:, 0]),
        "pearson_pw": _pearson(mu[:, 1], true[:, 1]),
        "calib_2s_ipd": float(calib_2s[0]),
        "calib_2s_pw": float(calib_2s[1]),
    }


def evaluate_baselines(
    pkl_path: str | Path,
    baselines_dir: str | Path,
) -> dict[str, dict]:
    """Evaluate baseline models on the same .pkl for side-by-side comparison.

    Expects baselines_dir to contain subdirectories:
      - global_gaussian/global_gaussian.json
      - kmer_gaussian/model_meta.json + kmer_stats.pkl
      - conv_no_film/model_config.json + best_checkpoint.pt + meth_ratios.json

    Returns dict mapping baseline name -> metrics dict.
    """
    import sys

    baselines_dir = Path(baselines_dir)
    results = {}

    # Load raw pkl for baselines 1 & 2 (they predict in raw space)
    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)

    # Collect (kmer_id, meth_id, true_log, fraction) from pkl
    true_log_list = []
    kmer_list = []
    meth_list = []
    frac_list = []
    for key, arr in raw_data.items():
        if not isinstance(key, tuple):
            continue
        kmer_id, meth_id = key
        if not isinstance(arr, np.ndarray) or len(arr) == 0:
            continue
        # Pick one random sample per key (same as evaluate())
        rng = np.random.default_rng(42 + kmer_id + meth_id)
        idx = rng.integers(len(arr))
        sample = arr[idx].astype(np.float32)
        ipd_raw, pw_raw = sample[0], sample[1]
        frac = float(sample[2]) if len(sample) >= 3 else (1.0 if meth_id > 0 else 0.0)
        true_log_list.append([np.log1p(ipd_raw), np.log1p(pw_raw)])
        kmer_list.append(kmer_id)
        meth_list.append(meth_id)
        frac_list.append(frac)

    true_log = np.array(true_log_list, dtype=np.float32)
    kmer_arr = np.array(kmer_list, dtype=np.int64)
    meth_arr = np.array(meth_list, dtype=np.int64)
    np.array(frac_list, dtype=np.float32)
    n = len(true_log)

    # ── Baseline 1: Global Gaussian ──────────────────────────────────────
    gg_path = baselines_dir / "global_gaussian" / "global_gaussian.json"
    if gg_path.exists():
        log.info("Evaluating baseline: Global Gaussian")
        gg_model = json.loads(gg_path.read_text())
        gg_model = {int(k): v for k, v in gg_model.items()}

        mu_raw = np.zeros((n, 2), dtype=np.float32)
        sigma_raw = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            mid = int(meth_arr[i])
            m = gg_model.get(mid, gg_model.get(0, {}))
            mu_raw[i, 0] = m.get("mu_ipd", 10.0)
            mu_raw[i, 1] = m.get("mu_pw", 8.0)
            sigma_raw[i, 0] = max(m.get("sigma_ipd", 5.0), 0.1)
            sigma_raw[i, 1] = max(m.get("sigma_pw", 4.0), 0.1)

        # Convert to log1p space for fair comparison
        mu_log = np.log1p(mu_raw)
        # Approximate sigma in log1p space: d/dx log1p(x) = 1/(1+x)
        sigma_log = sigma_raw / (1.0 + mu_raw)

        results["Global Gaussian"] = _evaluate_predictions(mu_log, sigma_log, true_log, meth_arr)
    else:
        log.info("Skipping Global Gaussian (not found: %s)", gg_path)

    # ── Baseline 2: Per-kmer Gaussian ────────────────────────────────────
    kg_meta = baselines_dir / "kmer_gaussian" / "model_meta.json"
    kg_kmer = baselines_dir / "kmer_gaussian" / "kmer_stats.pkl"
    if kg_meta.exists() and kg_kmer.exists():
        log.info("Evaluating baseline: Per-kmer Gaussian")
        with open(kg_meta) as f:
            meta = json.load(f)
        with open(kg_kmer, "rb") as f:
            kmer_stats = pickle.load(f)

        ipd_ratios = {int(k): v for k, v in meta["ipd_ratios"].items()}
        pw_ratios = {int(k): v for k, v in meta["pw_ratios"].items()}
        global_u = meta["global_unmeth"]

        mu_raw = np.zeros((n, 2), dtype=np.float32)
        sigma_raw = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            kid = int(kmer_arr[i])
            mid = int(meth_arr[i])
            if kid in kmer_stats:
                s = kmer_stats[kid]
            else:
                s = global_u
            mu_ipd = s.get("mu_ipd", global_u["mu_ipd"])
            mu_pw = s.get("mu_pw", global_u["mu_pw"])
            sig_ipd = max(s.get("sigma_ipd", global_u["sigma_ipd"]), 0.1)
            sig_pw = max(s.get("sigma_pw", global_u["sigma_pw"]), 0.1)
            if mid > 0:
                mu_ipd *= ipd_ratios.get(mid, 1.0)
                mu_pw *= pw_ratios.get(mid, 1.0)
            mu_raw[i] = [mu_ipd, mu_pw]
            sigma_raw[i] = [sig_ipd, sig_pw]

        mu_log = np.log1p(mu_raw)
        sigma_log = sigma_raw / (1.0 + mu_raw)

        results["Per-kmer Gaussian"] = _evaluate_predictions(mu_log, sigma_log, true_log, meth_arr)
    else:
        log.info("Skipping Per-kmer Gaussian (not found: %s)", kg_meta)

    # ── Baseline 3: ConvNoFiLM ───────────────────────────────────────────
    cnf_cfg = baselines_dir / "conv_no_film" / "model_config.json"
    cnf_ckpt = baselines_dir / "conv_no_film" / "best_checkpoint.pt"
    cnf_ratios = baselines_dir / "conv_no_film" / "meth_ratios.json"
    if cnf_cfg.exists() and cnf_ckpt.exists() and cnf_ratios.exists():
        log.info("Evaluating baseline: ConvNoFiLM")
        # Import here to avoid circular dependency at module level
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from baseline.conv_no_film import ConvNoFiLMPredictor

        cfg = json.loads(cnf_cfg.read_text())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnf_model = ConvNoFiLMPredictor(
            base_embed_dim=cfg.get("base_embed_dim", 16),
            conv_dim=cfg.get("conv_dim", 128),
            n_conv_layers=cfg.get("n_conv_layers", 3),
            kernel_size=cfg.get("kernel_size", 3),
            head_dim=cfg.get("head_dim", 128),
            dropout=cfg.get("dropout", 0.1),
        ).to(device)
        ckpt = torch.load(str(cnf_ckpt), map_location=device)
        cnf_model.load_state_dict(ckpt["model"])
        cnf_model.eval()

        with open(cnf_ratios) as f:
            ratios = json.load(f)
        ipd_ratios = {int(k): v for k, v in ratios["ipd_ratios"].items()}
        pw_ratios = {int(k): v for k, v in ratios["pw_ratios"].items()}

        # Run through the same dataset
        dataset = MLPSignalDataset(str(pkl_path))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=2)

        cnf_mu_list = []
        cnf_sigma_list = []
        cnf_true_list = []
        cnf_meth_list = []

        with torch.no_grad():
            for kmer_ids, meth_probs, signals, m_ids in loader:
                kmer_ids = kmer_ids.to(device)
                meth_probs = meth_probs.to(device)
                params = cnf_model(kmer_ids, meth_probs)
                mu = params[:, :2]
                log_sig = torch.clamp(params[:, 2:], *_SIGMA_CLAMP)
                sigma = torch.exp(log_sig)

                # Apply post-hoc ratio shift to mu (in log1p space)
                mu_np = mu.cpu().numpy()
                sig_np = sigma.cpu().numpy()
                m_np = m_ids.numpy()
                for i in range(len(mu_np)):
                    mid = int(m_np[i])
                    if mid > 0:
                        # Convert to raw, apply ratio, convert back
                        mu_raw = np.expm1(mu_np[i])
                        mu_raw[0] *= ipd_ratios.get(mid, 1.0)
                        mu_raw[1] *= pw_ratios.get(mid, 1.0)
                        mu_np[i] = np.log1p(np.clip(mu_raw, 0, 255))

                cnf_mu_list.append(mu_np)
                cnf_sigma_list.append(sig_np)
                cnf_true_list.append(signals.numpy())
                cnf_meth_list.append(m_np)

        cnf_mu = np.concatenate(cnf_mu_list)
        cnf_sigma = np.concatenate(cnf_sigma_list)
        cnf_true = np.concatenate(cnf_true_list)
        cnf_meth = np.concatenate(cnf_meth_list)

        results["ConvNoFiLM"] = _evaluate_predictions(cnf_mu, cnf_sigma, cnf_true, cnf_meth)
    else:
        log.info("Skipping ConvNoFiLM (not found: %s)", cnf_cfg)

    return results


def print_comparison(main_metrics: dict, baseline_metrics: dict[str, dict]) -> str:
    """Format a side-by-side comparison table."""
    lines = [
        "",
        "=" * 70,
        "  Model Comparison",
        "=" * 70,
        "",
        f"  {'Model':<22} {'r_IPD':>7} {'r_PW':>7}  {'MAE_I':>7} {'MAE_P':>7}"
        f"  {'2σ_IPD':>7} {'2σ_PW':>7}",
        "  " + "-" * 66,
    ]

    # Main model
    m = main_metrics
    lines.append(
        f"  {'Trained model':<22} {m['pearson_ipd']:>7.4f} {m['pearson_pw']:>7.4f}"
        f"  {m['mae_ipd']:>7.4f} {m['mae_pw']:>7.4f}"
        f"  {m['calib_2s_ipd'] * 100:>6.1f}% {m['calib_2s_pw'] * 100:>6.1f}%"
    )

    # Baselines
    for name, bm in baseline_metrics.items():
        lines.append(
            f"  {name:<22} {bm['pearson_ipd']:>7.4f} {bm['pearson_pw']:>7.4f}"
            f"  {bm['mae_ipd']:>7.4f} {bm['mae_pw']:>7.4f}"
            f"  {bm['calib_2s_ipd'] * 100:>6.1f}% {bm['calib_2s_pw'] * 100:>6.1f}%"
        )

    # Oracle / random for context
    lines += [
        "  " + "-" * 66,
        f"  {'Oracle (ceiling)':<22} {m['oracle_ipd']:>7.4f} {m['oracle_pw']:>7.4f}",
        f"  {'Random sample':<22} {m['rand_pearson_ipd']:>7.4f} {m['rand_pearson_pw']:>7.4f}",
        "",
        "  Higher Pearson = better.  2σ ≈ 95% = well-calibrated.",
        "=" * 70,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    import argparse

    from .utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="kinsim mlp evaluate",
        description=(
            "Evaluate a trained MLP and inspect its predicted distributions.\n\n"
            "Full report mode:\n"
            "  kinsim mlp evaluate checkpoints_mlp/ master_data.pkl\n\n"
            "Distribution plot for one k-mer:\n"
            "  kinsim mlp evaluate checkpoints_mlp/ master_data.pkl \\\n"
            "      --kmer GGATCCTGCAT --meth m6A --plot gatc_m6A.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_dir", help="Directory with model_config.json and checkpoint_epoch*.pt"
    )
    parser.add_argument("pkl", help="Merged master .pkl file (from kinsim merge)")
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
    parser.add_argument(
        "--baselines-dir",
        default=None,
        metavar="DIR",
        help="Directory with baseline subdirs (global_gaussian/, kmer_gaussian/, conv_no_film/)",
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

        # Remove non-tuple keys (__meta__, etc.)
        data = {k: v for k, v in raw_data.items() if isinstance(k, tuple)}

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

        # ── Baseline comparison (if available) ───────────────────────────────
        if args.baselines_dir:
            log.info("Evaluating baselines from: %s", args.baselines_dir)
            baseline_metrics = evaluate_baselines(args.pkl, args.baselines_dir)
            if baseline_metrics:
                comparison = print_comparison(metrics, baseline_metrics)
                report += "\n" + comparison

        print(report)

        out_path = args.output or str(Path(args.checkpoint_dir) / "evaluation_report.txt")
        Path(out_path).write_text(report)
        log.info("Report saved: %s", out_path)
