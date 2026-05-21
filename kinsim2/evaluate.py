"""``kinsim evaluate`` — post-training calibration report (bilateral v2).

Usage::

    kinsim evaluate <checkpoint_dir> <shard.pkl_or_dir>
        Full bilateral calibration report. Aggregates over fwd+rev strands.

    kinsim evaluate <checkpoint_dir> <shard.pkl> \\
        --kmer GATCGATCGAT --meth m6A --plot gatc_m6A.png
        Plot N(mu, sigma^2) vs actual for one (kmer, meth) bucket.

Metrics are 4-channel: IPD aggregates fwd+rev IPD; PW aggregates fwd+rev
PW. Calibration / oracle / random-from-distribution use the unified
4-channel view.
"""

from __future__ import annotations

import json
import logging
import pickle
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
from .utils.encoding import encode_kmer, get_meth_ids

log = logging.getLogger(__name__)

_SIGMA_CLAMP_LEGACY = (-6.0, 3.0)
_IPD_COLS = [0, 2]
_PW_COLS = [1, 3]


def _sigma_clamp_from_model(model) -> tuple[float, float]:
    return (
        float(getattr(model, "log_sigma_clamp_min", _SIGMA_CLAMP_LEGACY[0])),
        float(getattr(model, "log_sigma_clamp_max", _SIGMA_CLAMP_LEGACY[1])),
    )


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
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
    checkpoint_dir = Path(checkpoint_dir)
    cfg_path = checkpoint_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found in {checkpoint_dir}; train must have written it."
        )
    cfg = json.loads(cfg_path.read_text())
    model = create_from_config(cfg).to(device)
    ckpt_path = _find_latest_checkpoint(checkpoint_dir)
    model.load_state_dict(load_state_dict_from_ckpt(ckpt_path))
    model.eval()
    log.info("Model loaded: %d params  checkpoint=%s",
             sum(p.numel() for p in model.parameters()), ckpt_path.name)
    return model


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0


@torch.no_grad()
def evaluate(
    model: ConvPredictor,
    pkl_path: str | Path,
    device: torch.device,
    batch_size: int = 4096,
) -> dict:
    """Run the full bilateral calibration suite on a shard."""
    dataset = SignalDataset(str(pkl_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_mu, all_sigma, all_true, all_cat = [], [], [], []
    sigma_clamp = _sigma_clamp_from_model(model)

    for batch in loader:
        kmer_ids, mc_fwd, mc_rev, signals, cat_fwd, _cat_rev = batch
        kmer_ids = kmer_ids.to(device)
        mc_fwd = mc_fwd.to(device)
        mc_rev = mc_rev.to(device)
        params = model(kmer_ids, mc_fwd, mc_rev)
        mu = params[:, :4]
        log_sig = torch.clamp(params[:, 4:], *sigma_clamp)
        sigma = torch.exp(log_sig)
        all_mu.append(mu.cpu().numpy())
        all_sigma.append(sigma.cpu().numpy())
        all_true.append(signals.numpy())
        all_cat.append(cat_fwd.numpy())

    mu = np.concatenate(all_mu, axis=0)
    sigma = np.concatenate(all_sigma, axis=0)
    true = np.concatenate(all_true, axis=0)
    cat = np.concatenate(all_cat, axis=0)

    diff = mu - true
    mse = (diff**2).mean(axis=0)
    mae = np.abs(diff).mean(axis=0)

    def _calib(n: float):
        return (np.abs(diff) <= n * sigma).mean(axis=0)

    c1, c2, c3 = _calib(1), _calib(2), _calib(3)
    mse_ipd = float((mse[0] + mse[2]) / 2)
    mse_pw = float((mse[1] + mse[3]) / 2)
    mae_ipd = float((mae[0] + mae[2]) / 2)
    mae_pw = float((mae[1] + mae[3]) / 2)

    p_ipd = _pearson(mu[:, _IPD_COLS].ravel(), true[:, _IPD_COLS].ravel())
    p_pw = _pearson(mu[:, _PW_COLS].ravel(), true[:, _PW_COLS].ravel())
    var_mu_ipd = float(np.var(mu[:, _IPD_COLS]))
    var_mu_pw = float(np.var(mu[:, _PW_COLS]))
    e_sig2_ipd = float(np.mean(sigma[:, _IPD_COLS] ** 2))
    e_sig2_pw = float(np.mean(sigma[:, _PW_COLS] ** 2))
    oracle_ipd = var_mu_ipd / (var_mu_ipd + e_sig2_ipd) if (var_mu_ipd + e_sig2_ipd) > 0 else 0.0
    oracle_pw = var_mu_pw / (var_mu_pw + e_sig2_pw) if (var_mu_pw + e_sig2_pw) > 0 else 0.0

    rng = np.random.default_rng(42)
    z_random = mu + sigma * rng.standard_normal(mu.shape).astype(np.float32)
    rand_pearson_ipd = _pearson(z_random[:, _IPD_COLS].ravel(), true[:, _IPD_COLS].ravel())
    rand_pearson_pw = _pearson(z_random[:, _PW_COLS].ravel(), true[:, _PW_COLS].ravel())

    by_cat: dict = {}
    cat_names = {0: "baseline", 1: "slowed", 2: "near_meth"}
    for cid in sorted(np.unique(cat)):
        mask = cat == cid
        if mask.sum() < 2:
            continue
        name = cat_names.get(int(cid), f"cat{int(cid)}")
        mu_m = mu[mask]
        sig_m = sigma[mask]
        true_m = true[mask]
        diff_m = mu_m - true_m
        by_cat[name] = {
            "n": int(mask.sum()),
            "pearson_ipd": _pearson(mu_m[:, _IPD_COLS].ravel(), true_m[:, _IPD_COLS].ravel()),
            "pearson_pw": _pearson(mu_m[:, _PW_COLS].ravel(), true_m[:, _PW_COLS].ravel()),
            "mae_ipd": float(np.abs(diff_m[:, _IPD_COLS]).mean()),
            "mae_pw": float(np.abs(diff_m[:, _PW_COLS]).mean()),
            "calib_2s": float((np.abs(diff_m) <= 2 * sig_m).mean()),
            "mean_sigma_ipd": float(sig_m[:, _IPD_COLS].mean()),
            "mean_sigma_pw": float(sig_m[:, _PW_COLS].mean()),
        }

    return {
        "mse_ipd": mse_ipd,
        "mse_pw": mse_pw,
        "mae_ipd": mae_ipd,
        "mae_pw": mae_pw,
        "pearson_ipd": p_ipd,
        "pearson_pw": p_pw,
        "oracle_ipd": oracle_ipd,
        "oracle_pw": oracle_pw,
        "rand_pearson_ipd": rand_pearson_ipd,
        "rand_pearson_pw": rand_pearson_pw,
        "calib_1s_ipd": float((c1[0] + c1[2]) / 2),
        "calib_1s_pw": float((c1[1] + c1[3]) / 2),
        "calib_2s_ipd": float((c2[0] + c2[2]) / 2),
        "calib_2s_pw": float((c2[1] + c2[3]) / 2),
        "calib_3s_ipd": float((c3[0] + c3[2]) / 2),
        "calib_3s_pw": float((c3[1] + c3[3]) / 2),
        "mean_sigma_ipd": float(sigma[:, _IPD_COLS].mean()),
        "mean_sigma_pw": float(sigma[:, _PW_COLS].mean()),
        "median_sigma_ipd": float(np.median(sigma[:, _IPD_COLS])),
        "median_sigma_pw": float(np.median(sigma[:, _PW_COLS])),
        "by_category": by_cat,
        "n_contexts": len(mu),
    }


def print_report(metrics: dict) -> str:
    by_cat = metrics.get("by_category", {})
    lines = [
        "=" * 70,
        "  KinSim2 - Bilateral Evaluation Report",
        "=" * 70,
        "",
        f"  Contexts evaluated : {metrics['n_contexts']:,}",
        "",
        "  -- Mean prediction quality (log1p space, per-strand aggregated) --",
        f"  MAE   IPD / PW  :  {metrics['mae_ipd']:.4f}  /  {metrics['mae_pw']:.4f}",
        f"  MSE   IPD / PW  :  {metrics['mse_ipd']:.4f}  /  {metrics['mse_pw']:.4f}",
        f"  Pearson IPD/PW  :  {metrics['pearson_ipd']:.4f}  /  {metrics['pearson_pw']:.4f}",
        "",
        "  -- Pearson context --",
        f"  Model  IPD={metrics['pearson_ipd']:.4f}  PW={metrics['pearson_pw']:.4f}",
        f"  Oracle IPD={metrics['oracle_ipd']:.4f}  PW={metrics['oracle_pw']:.4f}",
        f"  Random IPD={metrics['rand_pearson_ipd']:.4f}  PW={metrics['rand_pearson_pw']:.4f}",
    ]
    for signal in ("ipd", "pw"):
        r_model = metrics[f"pearson_{signal}"]
        r_oracle = metrics[f"oracle_{signal}"]
        r_random = metrics[f"rand_pearson_{signal}"]
        if r_oracle - r_random > 1e-6:
            efficiency = (r_model - r_random) / (r_oracle - r_random)
            lines.append(f"  Efficiency {signal.upper()}: {efficiency * 100:.1f}% of oracle gap")
        else:
            lines.append(f"  Efficiency {signal.upper()}: N/A (oracle ~ random)")

    lines += [
        "",
        "  -- Calibration coverage (% within nsigma of mu) --",
        "  Coverage     IPD     PW    Expected",
        f"  1sigma   {metrics['calib_1s_ipd'] * 100:6.1f}%  {metrics['calib_1s_pw'] * 100:6.1f}%    68.3%",
        f"  2sigma   {metrics['calib_2s_ipd'] * 100:6.1f}%  {metrics['calib_2s_pw'] * 100:6.1f}%    95.4%",
        f"  3sigma   {metrics['calib_3s_ipd'] * 100:6.1f}%  {metrics['calib_3s_pw'] * 100:6.1f}%    99.7%",
        "",
        f"  Mean sigma  IPD={metrics['mean_sigma_ipd']:.4f}  PW={metrics['mean_sigma_pw']:.4f}",
        f"  Median sigma IPD={metrics['median_sigma_ipd']:.4f}  PW={metrics['median_sigma_pw']:.4f}",
    ]

    if by_cat:
        lines += ["", "  -- Per-category (fwd strand) --",
                  "  Type        N     r_IPD  r_PW   MAE_I  MAE_P   2s%"]
        for name, t in by_cat.items():
            lines.append(
                f"  {name:<10} {t['n']:>7}  {t['pearson_ipd']:.3f} {t['pearson_pw']:.3f}"
                f"  {t['mae_ipd']:.4f} {t['mae_pw']:.4f}  {t['calib_2s'] * 100:>4.1f}%"
            )
    lines += ["", "=" * 70]
    return "\n".join(lines)


@torch.no_grad()
def plot_kmer_distribution(
    model: ConvPredictor,
    data: dict,
    kmer_str: str,
    meth_name: str = "none",
    device: torch.device = torch.device("cpu"),
    output_path: str | None = None,
    layout=None,
) -> None:
    """Plot predicted N(mu, sigma^2) vs actual observations for one bucket.

    Bilateral plot: 2x2 grid (rows = IPD/PW, cols = fwd/rev strand).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    from .utils.sample_layout import CATEGORY_BASELINE, CATEGORY_SLOWED, get_sample_layout

    meth_ids_map = get_meth_ids()
    if meth_name not in meth_ids_map:
        raise ValueError(f"meth_name must be one of {list(meth_ids_map)}, got '{meth_name}'")
    meth_id = meth_ids_map[meth_name]

    cfg = model.get_config()
    K = int(cfg.get("kmer_size", 11))
    M = int(cfg.get("num_meth_types", max(meth_ids_map.values()) + 1))
    if len(kmer_str) != K:
        raise ValueError(f"kmer_str must be {K} bases, got {len(kmer_str)}")
    kmer_id = encode_kmer(kmer_str)
    if kmer_id not in data:
        raise KeyError(f"kmer '{kmer_str}' not present in dataset")
    arr = data[kmer_id]

    if layout is None:
        layout = get_sample_layout()
    cats = arr[:, layout.col_category_fwd].astype(np.int8)
    pm = arr[:, layout.col_parent_meth_fwd].astype(np.int8)
    if meth_id > 0:
        sel = (cats == CATEGORY_SLOWED) & (pm == meth_id)
    else:
        sel = cats == CATEGORY_BASELINE
    if not sel.any():
        raise KeyError(f"context '{kmer_str}/{meth_name}' has 0 matching rows")

    samples = arr[sel].astype(np.float32)
    actual_raw = samples[:, [
        layout.col_ipd_fwd, layout.col_pw_fwd, layout.col_ipd_rev, layout.col_pw_rev,
    ]]
    actual_log = np.log1p(actual_raw)

    mc_fwd = torch.zeros(1, K, M, device=device)
    mc_rev = torch.zeros(1, K, M, device=device)
    active = int(cfg.get("active_site_index", K // 2))
    if meth_id > 0:
        mc_fwd[0, active, meth_id] = 1.0
    kmer_t = torch.tensor([kmer_id], dtype=torch.long, device=device)
    params = model(kmer_t, mc_fwd, mc_rev)
    mu_log = params[0, :4].cpu().numpy()
    log_sig = np.clip(params[0, 4:].cpu().numpy(), *_sigma_clamp_from_model(model))
    sigma = np.exp(log_sig)

    def _gauss_pdf(x, mu, sig):
        return (1.0 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    titles = ["IPD fwd", "PW fwd", "IPD rev", "PW rev"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (ax, title) in enumerate(zip(axes.ravel(), titles)):
        col = actual_log[:, i]
        m, s = mu_log[i], sigma[i]
        ax.hist(col, bins=40, density=True, alpha=0.5, color="steelblue")
        x = np.linspace(min(col.min(), m - 4 * s), max(col.max(), m + 4 * s), 400)
        ax.plot(x, _gauss_pdf(x, m, s), "r-", lw=2,
                label=f"mu={m:.3f}  sigma={s:.3f}")
        ax.set_title(f"{title}  n={len(col)}")
        ax.legend(fontsize=8)
        ax.set_xlabel("log1p signal")
    fig.suptitle(f"{kmer_str}  [{meth_name}]")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        log.info("Plot saved: %s", output_path)
    else:
        plt.show()
    plt.close()


def main(argv=None) -> None:
    import argparse

    from .utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="kinsim evaluate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint_dir")
    parser.add_argument("pkl")
    parser.add_argument("--kmer", default=None)
    parser.add_argument("--meth", default="none", choices=list(get_meth_ids()))
    parser.add_argument("--plot", default=None, metavar="FILE")
    parser.add_argument("--output", default=None, metavar="TXT")
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
        with open(args.pkl, "rb") as f:
            raw_data = pickle.load(f)
        data = {
            k: v for k, v in raw_data.items()
            if isinstance(k, (int, np.integer)) and isinstance(v, np.ndarray)
        }
        plot_kmer_distribution(
            model, data, kmer_str=args.kmer, meth_name=args.meth,
            device=device, output_path=args.plot,
        )
    else:
        metrics = evaluate(model, args.pkl, device, batch_size=args.batch_size)
        report = print_report(metrics)
        print(report)
        out_path = args.output or str(Path(args.checkpoint_dir) / "evaluation_report.txt")
        Path(out_path).write_text(report)
        log.info("Report saved: %s", out_path)


if __name__ == "__main__":
    main()
