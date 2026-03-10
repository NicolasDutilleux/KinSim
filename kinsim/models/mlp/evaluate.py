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

MAE (log-space)
    Mean Absolute Error in log1p space.  0.0 is perfect; ~0.1 is excellent.

2σ Calibration
    Fraction of actual observations falling within [μ − 2σ, μ + 2σ].
    Expected for a correctly calibrated Gaussian: 95.4 %.
    - Below 90 %: model underestimates uncertainty (σ too small, overconfident)
    - Above 99 %: model over-disperses (σ too large, too conservative)

σ Histogram
    Distribution of predicted σ values.  A well-trained model shows different σ
    for methylated vs unmethylated contexts — it has learned that m6A pauses are
    noisier than background.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...common.dataset import MLPSignalDataset, inv_log_transform, log_transform
from ...encoding import METH_IDS, encode_kmer
from .model import MLPPredictor

log = logging.getLogger(__name__)

_SIGMA_CLAMP = (-6.0, 3.0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint_dir: str | Path, device: torch.device) -> MLPPredictor:
    """Load MLPPredictor from a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    cfg_path = checkpoint_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found in {checkpoint_dir}.\n"
            "Ensure training completed at least one epoch."
        )
    cfg = json.loads(cfg_path.read_text())

    model = MLPPredictor(
        kmer_embed_dim=cfg["kmer_embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        meth_proj_dim=cfg.get("meth_proj_dim", 8),
        dropout=cfg.get("dropout", 0.0),
    ).to(device)

    # Find the latest checkpoint
    pts = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    if not pts:
        raise FileNotFoundError(f"No checkpoint_epoch*.pt files found in {checkpoint_dir}")
    ckpt_path = pts[-1]
    log.info("Loading checkpoint: %s", ckpt_path.name)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
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
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_mu    = []
    all_sigma = []
    all_true  = []

    for kmer_ids, meth_probs, signals in loader:
        kmer_ids   = kmer_ids.to(device)
        meth_probs = meth_probs.to(device)

        params  = model(kmer_ids, meth_probs)
        mu      = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], *_SIGMA_CLAMP)
        sigma   = torch.exp(log_sig)

        all_mu.append(mu.cpu().numpy())
        all_sigma.append(sigma.cpu().numpy())
        all_true.append(signals.numpy())

    mu    = np.concatenate(all_mu,    axis=0)   # (N, 2) log1p space
    sigma = np.concatenate(all_sigma, axis=0)   # (N, 2) log1p space
    true  = np.concatenate(all_true,  axis=0)   # (N, 2) log1p space

    diff = mu - true
    mse  = (diff ** 2).mean(axis=0)
    mae  = np.abs(diff).mean(axis=0)

    def _pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0

    # Calibration at 1σ, 2σ, 3σ (expected: 68.3%, 95.4%, 99.7%)
    def _calib(n_sigma):
        return (np.abs(diff) <= n_sigma * sigma).mean(axis=0)

    calib_1s = _calib(1)
    calib_2s = _calib(2)
    calib_3s = _calib(3)

    return {
        # Mean / spread quality
        "mse_ipd":        float(mse[0]),
        "mse_pw":         float(mse[1]),
        "mae_ipd":        float(mae[0]),
        "mae_pw":         float(mae[1]),
        "pearson_ipd":    _pearson(mu[:, 0], true[:, 0]),
        "pearson_pw":     _pearson(mu[:, 1], true[:, 1]),
        # Calibration coverage
        "calib_1s_ipd":   float(calib_1s[0]),
        "calib_1s_pw":    float(calib_1s[1]),
        "calib_2s_ipd":   float(calib_2s[0]),
        "calib_2s_pw":    float(calib_2s[1]),
        "calib_3s_ipd":   float(calib_3s[0]),
        "calib_3s_pw":    float(calib_3s[1]),
        # Heteroscedasticity check — σ spread
        "mean_sigma_ipd": float(sigma[:, 0].mean()),
        "mean_sigma_pw":  float(sigma[:, 1].mean()),
        "median_sigma_ipd": float(np.median(sigma[:, 0])),
        "median_sigma_pw":  float(np.median(sigma[:, 1])),
        "n_contexts":     len(mu),
    }


def print_report(metrics: dict[str, float]) -> str:
    """Format evaluation metrics as a human-readable report string."""
    lines = [
        "=" * 60,
        "  KinSim MLP — Evaluation Report",
        "=" * 60,
        "",
        f"  Contexts evaluated : {metrics['n_contexts']:,}",
        "",
        "  ── Mean prediction quality (log1p space) ──────────────",
        f"  MAE   IPD / PW  :  {metrics['mae_ipd']:.4f}  /  {metrics['mae_pw']:.4f}",
        f"  MSE   IPD / PW  :  {metrics['mse_ipd']:.4f}  /  {metrics['mse_pw']:.4f}",
        f"  Pearson IPD / PW:  {metrics['pearson_ipd']:.4f}  /  {metrics['pearson_pw']:.4f}",
        "",
        "  ── Calibration coverage (% within nσ of μ) ─────────────",
        "  Coverage     IPD     PW    Expected",
        f"  1σ (68%)  {metrics['calib_1s_ipd']*100:6.1f}%  {metrics['calib_1s_pw']*100:6.1f}%    68.3%",
        f"  2σ (95%)  {metrics['calib_2s_ipd']*100:6.1f}%  {metrics['calib_2s_pw']*100:6.1f}%    95.4%",
        f"  3σ (99%)  {metrics['calib_3s_ipd']*100:6.1f}%  {metrics['calib_3s_pw']*100:6.1f}%    99.7%",
        "",
        "  ── Predicted σ (log1p space) — heteroscedasticity ──────",
        f"  Mean σ  IPD / PW  :  {metrics['mean_sigma_ipd']:.4f}  /  {metrics['mean_sigma_pw']:.4f}",
        f"  Median σ IPD / PW :  {metrics['median_sigma_ipd']:.4f}  /  {metrics['median_sigma_pw']:.4f}",
        "",
        "  Calibration interpretation:",
        "    > 95%  overconfident (σ too small) — model underestimates noise",
        "    < 95%  well-calibrated or over-dispersed",
        "=" * 60,
    ]
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
        data:        Raw dict from .pkl file: (kmer_id, meth_id) → np.ndarray(N, 2).
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
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    if len(kmer_str) != 11:
        raise ValueError(f"kmer_str must be exactly 11 bases, got {len(kmer_str)}")
    if meth_name not in METH_IDS:
        raise ValueError(f"meth_name must be one of {list(METH_IDS)}, got '{meth_name}'")

    kmer_id = encode_kmer(kmer_str)
    meth_id = METH_IDS[meth_name]
    key     = (kmer_id, meth_id)

    if key not in data:
        raise KeyError(
            f"Context '{kmer_str}' / '{meth_name}' not found in dataset.\n"
            f"Available meth states for this k-mer: "
            + str([m for m_name, m_id in METH_IDS.items() if (kmer_id, m_id) in data])
        )

    # Actual data: raw uint8 → log1p (use only IPD/PW columns)
    samples    = data[key].astype(np.float32)
    actual_raw = samples[:, :2]                     # (N, 2) [IPD, PW]
    actual_log = np.log1p(actual_raw)               # (N, 2) log1p space

    # Build stoichiometric meth_probs from stored fraction (3rd column)
    # For legacy 2-column data, default to 1.0 for methylated.
    if samples.shape[1] >= 3:
        fraction = float(samples[0, 2])
    else:
        fraction = 1.0 if meth_id > 0 else 0.0

    kmer_tensor = torch.tensor([kmer_id], dtype=torch.long, device=device)
    meth_probs  = torch.zeros(1, 4, device=device)
    if meth_id > 0:
        meth_probs[0, meth_id] = fraction

    params  = model(kmer_tensor, meth_probs)       # (1, 4)
    mu_log  = params[0, :2].cpu().numpy()          # [μ_ipd, μ_pw] log1p
    log_sig = np.clip(params[0, 2:].cpu().numpy(), *_SIGMA_CLAMP)
    sigma   = np.exp(log_sig)                      # [σ_ipd, σ_pw] log1p

    # Gaussian PDF (pure numpy, no scipy dependency)
    def _gauss_pdf(x, mu, sig):
        return (1.0 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels    = ["IPD", "PW"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        col   = actual_log[:, i]
        mu_i  = mu_log[i]
        sig_i = sigma[i]

        # Histogram of actual data
        ax.hist(
            col, bins=50, density=True, alpha=0.55,
            color="steelblue", edgecolor="white", linewidth=0.4,
            label=f"Actual  (n={len(col):,})",
        )

        # Predicted Gaussian PDF
        x_range = np.linspace(
            min(col.min(), mu_i - 4 * sig_i),
            max(col.max(), mu_i + 4 * sig_i),
            500,
        )
        ax.plot(
            x_range, _gauss_pdf(x_range, mu_i, sig_i),
            "r-", lw=2.5,
            label=f"Predicted  μ={mu_i:.3f}  σ={sig_i:.3f}",
        )

        # μ ± 2σ shaded region
        x_fill = np.linspace(mu_i - 2 * sig_i, mu_i + 2 * sig_i, 300)
        ax.fill_between(
            x_fill, _gauss_pdf(x_fill, mu_i, sig_i),
            alpha=0.18, color="orange", label="μ ± 2σ  (~95.4%)",
        )
        ax.axvline(mu_i - 2 * sig_i, color="orange", ls="--", lw=1.4)
        ax.axvline(mu_i + 2 * sig_i, color="orange", ls="--", lw=1.4)
        ax.axvline(mu_i,             color="red",    ls="-",  lw=1.2, alpha=0.7)

        # Calibration annotation
        in_2s = float(np.mean(np.abs(col - mu_i) <= 2 * sig_i))
        ax.text(
            0.97, 0.95, f"2σ coverage: {in_2s*100:.1f}%\n(expected 95.4%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="darkorange",
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
        fontsize=12, fontweight="bold",
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
    from ...config import setup_logging

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
    parser.add_argument("checkpoint_dir",
                        help="Directory with model_config.json and checkpoint_epoch*.pt")
    parser.add_argument("pkl",
                        help="Merged master .pkl file (from kinsim merge)")
    parser.add_argument("--kmer",   default=None,
                        help="11-mer string to inspect (e.g. GGATCCTGCAT)")
    parser.add_argument("--meth",   default="none",
                        choices=list(METH_IDS),
                        help="Methylation state for the k-mer plot (default: none)")
    parser.add_argument("--plot",   default=None, metavar="FILE",
                        help="Save distribution plot to FILE instead of displaying it")
    parser.add_argument("--output", default=None, metavar="TXT",
                        help="Save full report to FILE (default: <checkpoint_dir>/evaluation_report.txt)")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    device_str = args.device if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)
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
            model, data,
            kmer_str=args.kmer,
            meth_name=args.meth,
            device=device,
            output_path=args.plot,
        )

    else:
        # ── Full calibration report ──────────────────────────────────────────
        log.info("Running full evaluation on: %s", args.pkl)
        metrics = evaluate(model, args.pkl, device, batch_size=args.batch_size)
        report  = print_report(metrics)
        print(report)

        out_path = args.output or str(Path(args.checkpoint_dir) / "evaluation_report.txt")
        Path(out_path).write_text(report)
        log.info("Report saved: %s", out_path)
