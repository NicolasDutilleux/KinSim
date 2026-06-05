"""Training loop for kinsim.

Direct (non-adversarial) training of a transformer generator under

    L = Energy_Distance²(real_tile, fake_tile)   bucketed by category
        + λ_mean * L1(per-bucket means)

The latent z provides stochasticity; the per-sample correspondence
between (cond, real_signal) drawn from the shard and (cond, fake_signal)
emitted by the generator is what enables the bucketed match — each
batch contains a mix of categories, and the loss is computed per
bucket so the minority methylation buckets pull on the model with
equal weight to the much larger baseline bucket.

Usage:

    python -m kinsim train <shards_dir> <ckpt_dir> [--config kinsim/config.yaml]

See kinsim/config.yaml for hyperparameters. Resuming a run is supported
via --resume (loads the latest checkpoint in <ckpt_dir>).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from .data import build_train_dataset
from .losses import bucketed_energy_distance, conditional_mean_l1
from .model import GeneratorConfig, TransformerGenerator


log = logging.getLogger("kinsim.train")


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_checkpoint(ckpt_dir: Path, step: int, model: TransformerGenerator,
                     opt: optim.Optimizer, cfg: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"G_step{step:08d}.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "model_config": asdict(model.cfg),
        "train_config": cfg,
    }, path)
    # Also write a model_config.json once at the first checkpoint so the
    # architecture is reproducible from the file alone.
    cfg_path = ckpt_dir / "model_config.json"
    if not cfg_path.exists():
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(model.cfg), f, indent=2)
    log.info("Wrote checkpoint %s", path)


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    cands = sorted(ckpt_dir.glob("G_step*.pt"))
    return cands[-1] if cands else None


def _infinite_loader(loader: DataLoader, dataset, epoch_start: int = 0):
    """Yield batches forever, advancing the dataset's epoch counter."""
    epoch = epoch_start
    while True:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def train(shards_dir: Path, ckpt_dir: Path, config_path: Path,
          resume: bool = False, device: str | None = None) -> None:
    cfg = _load_yaml(config_path)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    log.info("Device: %s", dev)

    # ----- model -----
    g_cfg = GeneratorConfig(
        k=int(cfg["window"]["k"]),
        n_channels=int(cfg["window"]["n_channels"]),
        n_meth_types=int(cfg["methylation"]["n_meth_types"]),
        d_model=int(cfg["model"]["d_model"]),
        n_layers=int(cfg["model"]["n_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),
        mlp_ratio=float(cfg["model"]["mlp_ratio"]),
        z_dim=int(cfg["model"]["z_dim"]),
        pos_embed_dim=int(cfg["model"]["pos_embed_dim"]),
        drop_rate=float(cfg["model"]["drop_rate"]),
    )
    model = TransformerGenerator(g_cfg).to(dev)
    log.info("Generator: %.2f M parameters", model.num_parameters() / 1e6)

    # ----- optimiser -----
    t_cfg = cfg["train"]
    opt = optim.AdamW(
        model.parameters(),
        lr=float(t_cfg["lr"]),
        betas=(float(t_cfg["beta1"]), float(t_cfg["beta2"])),
        weight_decay=float(t_cfg["weight_decay"]),
    )

    # ----- resume -----
    start_step = 0
    if resume:
        latest = _latest_checkpoint(Path(ckpt_dir))
        if latest is not None:
            ckpt = torch.load(latest, map_location=dev)
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["opt_state"])
            start_step = int(ckpt["step"]) + 1
            log.info("Resumed from %s at step %d", latest, start_step)
        else:
            log.info("No checkpoint to resume from in %s; starting fresh.", ckpt_dir)

    # ----- data -----
    test_strains = cfg.get("split", {}).get("test_strains", [])
    dataset = build_train_dataset(
        Path(shards_dir),
        n_meth_types=g_cfg.n_meth_types,
        test_strains=test_strains,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(t_cfg["batch_size"]),
        num_workers=int(t_cfg["num_workers"]),
        pin_memory=bool(t_cfg["pin_memory"]),
    )
    batches = _infinite_loader(loader, dataset, epoch_start=0)

    n_buckets = int(t_cfg.get("n_buckets", 3))
    bucket_min_samples = int(t_cfg.get("bucket_min_samples", 4))
    lambda_mean = float(t_cfg.get("lambda_mean", 0.1))

    log_every = int(t_cfg["log_every"])
    ckpt_every = int(t_cfg["checkpoint_every"])
    n_steps = int(t_cfg["n_steps"])

    last_log = time.time()
    running = {"ed": 0.0, "mean": 0.0, "loss": 0.0, "n": 0}

    for step in range(start_step, n_steps):
        batch = next(batches)
        base_fwd = batch["base_fwd_onehot"].to(dev, non_blocking=True)
        base_rev = batch["base_rev_onehot"].to(dev, non_blocking=True)
        meth_fwd = batch["meth_fwd_onehot"].to(dev, non_blocking=True)
        meth_rev = batch["meth_rev_onehot"].to(dev, non_blocking=True)
        real = batch["signal"].to(dev, non_blocking=True)               # (B, K, C)
        cat = batch["category"].to(dev, non_blocking=True).long()

        B = real.shape[0]
        z = torch.randn(B, g_cfg.z_dim, device=dev)
        fake = model(base_fwd, base_rev, meth_fwd, meth_rev, z)         # (B, K, C)

        real_flat = real.reshape(B, -1)
        fake_flat = fake.reshape(B, -1)

        ed_loss, _ed_per_bucket = bucketed_energy_distance(
            real_flat, fake_flat, cat, n_buckets, min_samples=bucket_min_samples
        )
        mean_loss = conditional_mean_l1(
            real, fake, cat, n_buckets, min_samples=bucket_min_samples
        )
        loss = ed_loss + lambda_mean * mean_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        running["ed"] += float(ed_loss.detach().item())
        running["mean"] += float(mean_loss.detach().item())
        running["loss"] += float(loss.detach().item())
        running["n"] += 1

        if step % log_every == 0 and running["n"] > 0:
            now = time.time()
            steps_per_sec = running["n"] / max(now - last_log, 1e-6)
            log.info(
                "step %6d  loss=%.4f  ed=%.4f  mean_l1=%.4f  steps/s=%.1f",
                step, running["loss"] / running["n"],
                running["ed"] / running["n"],
                running["mean"] / running["n"],
                steps_per_sec,
            )
            running = {"ed": 0.0, "mean": 0.0, "loss": 0.0, "n": 0}
            last_log = now

        if step > 0 and step % ckpt_every == 0:
            _save_checkpoint(Path(ckpt_dir), step, model, opt, cfg)

    # final checkpoint
    _save_checkpoint(Path(ckpt_dir), n_steps - 1, model, opt, cfg)
    log.info("Training complete.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the kinsim transformer generator.")
    p.add_argument("shards_dir", type=Path, help="Directory containing *_shard.pkl files.")
    p.add_argument("ckpt_dir", type=Path, help="Where to write G_step*.pt checkpoints.")
    p.add_argument("--config", type=Path,
                   default=Path(__file__).resolve().parent / "config.yaml",
                   help="Path to the kinsim YAML config (default: kinsim/config.yaml).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint in ckpt_dir, if any.")
    p.add_argument("--device", type=str, default=None,
                   help="Override device (cuda / cpu). Default: auto-detect.")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    _setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    train(
        shards_dir=args.shards_dir,
        ckpt_dir=args.ckpt_dir,
        config_path=args.config,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
