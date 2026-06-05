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
from .evaluate import find_test_shard_paths, held_out_w1
from .losses import bucketed_energy_distance, spatial_per_position_w1
from .model import GeneratorConfig, TransformerGenerator


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Explicit collate for the dict-of-tensors items produced by
    ShardedDataset. Matches kinsim_NN's collate to avoid relying on
    PyTorch's default collate quirks with IterableDataset + num_workers.
    """
    out = {}
    for k in batch[0]:
        if k in ("category", "parent_meth"):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def _compute_bucket_id(category: torch.Tensor, parent_meth: torch.Tensor) -> torch.Tensor:
    """Map (category, parent_meth) → integer bucket id in [0, 4].

    5 buckets, each homogeneous so its per-position mean profile
    actually shows the signature peaks the model must reproduce:

        0 = BASELINE       (category == 0)
        1 = NEAR_METH      (category == 2)
        2 = SLOWED_m6A     (category == 1, parent_meth == 1)
        3 = SLOWED_m4C     (category == 1, parent_meth == 2)
        4 = SLOWED_m5C     (category == 1, parent_meth == 3)

    Rationale: the previous 3-bucket scheme (BASELINE/SLOWED/NEAR_METH)
    averaged the m6A signature (offsets 0, +5), m4C (offset 0), m5C
    (offsets +2, +6) inside a single SLOWED bucket. The per-bucket mean
    profile that the L1 anchor matched was therefore the average of three
    different signatures — i.e. mostly flat. The audit on the previous
    run (W1=1.05 best ckpt, step 130k) showed exactly this pathology:
    the real SLOWED mean profile had small peaks at positions 4/9/16
    while the generator produced a flat profile. Per-meth bucketing
    gives each SLOWED_X bucket a CONSISTENT spatial signature that the
    L1 anchor can pull the generator toward.
    """
    # Sentinel for SLOWED with unknown parent_meth (shouldn't happen
    # in practice but be defensive: clamp to m6A bucket which is the
    # most common).
    slowed_id = torch.clamp(parent_meth, min=1, max=3) + 1  # 1→2, 2→3, 3→4
    return torch.where(
        category == 0, torch.zeros_like(category),
        torch.where(category == 2, torch.ones_like(category), slowed_id),
    ).long()


log = logging.getLogger("kinsim.train")


def _setup_logging(level: int = logging.INFO) -> None:
    # Force line-buffered stdout so SLURM-redirected logs appear in real
    # time rather than in 4 KB chunks. Without this, the first ~30
    # log lines accumulate in the kernel pipe buffer and only flush
    # when full — making the job look hung for the first few minutes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
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
    num_workers = int(t_cfg["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(t_cfg["batch_size"]),
        num_workers=num_workers,
        pin_memory=bool(t_cfg["pin_memory"]),
        collate_fn=_collate,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    batches = _infinite_loader(loader, dataset, epoch_start=0)

    # 5 buckets after the audit-driven per-meth refinement (see
    # _compute_bucket_id docstring): BASELINE / NEAR_METH /
    # SLOWED_m6A / SLOWED_m4C / SLOWED_m5C.
    n_buckets = int(t_cfg.get("n_buckets", 5))
    bucket_min_samples = int(t_cfg.get("bucket_min_samples", 4))
    lambda_pos = float(t_cfg.get("lambda_pos", 1.0))

    log_every = int(t_cfg["log_every"])
    ckpt_every = int(t_cfg["checkpoint_every"])
    eval_every = int(t_cfg.get("eval_every", ckpt_every))
    n_steps = int(t_cfg["n_steps"])

    # Cosine LR decay so the late-training oscillation observed at the
    # previous run (W1 1.05 → 3.72 between step 130k and step 199999)
    # doesn't recur. Decays from base lr at step 0 to base lr × 0.05
    # at step n_steps. If lr_schedule is "constant" the multiplier
    # stays at 1.0.
    lr_schedule = str(t_cfg.get("lr_schedule", "cosine")).lower()
    lr_min_frac = float(t_cfg.get("lr_min_frac", 0.05))
    import math
    def _lr_mult(step_idx: int) -> float:
        if lr_schedule == "cosine":
            t = step_idx / max(1, n_steps - 1)
            return lr_min_frac + 0.5 * (1.0 - lr_min_frac) * (1.0 + math.cos(math.pi * t))
        return 1.0
    base_lr = float(t_cfg["lr"])

    # Held-out test shards (looked up in the SAME shards dir as training —
    # list_shards excludes them from the training loader, so they live there
    # but never enter a training batch).
    test_shard_paths = find_test_shard_paths(Path(shards_dir), test_strains)
    if test_shard_paths:
        log.info("Held-out test shards: %d (%s)",
                 len(test_shard_paths), [p.name for p in test_shard_paths])
    else:
        log.warning("No held-out test shards found for test_strains=%s — "
                    "periodic W1 eval and best_G.pt selection disabled.",
                    test_strains)
    best_w1 = float("inf")

    last_log = time.time()
    running = {"ed": 0.0, "pos": 0.0, "loss": 0.0, "n": 0}

    for step in range(start_step, n_steps):
        # Apply cosine LR multiplier (does nothing if lr_schedule="constant")
        for pg in opt.param_groups:
            pg["lr"] = base_lr * _lr_mult(step)

        batch = next(batches)
        base_fwd = batch["base_fwd_onehot"].to(dev, non_blocking=True)
        base_rev = batch["base_rev_onehot"].to(dev, non_blocking=True)
        meth_fwd = batch["meth_fwd_onehot"].to(dev, non_blocking=True)
        meth_rev = batch["meth_rev_onehot"].to(dev, non_blocking=True)
        real = batch["signal"].to(dev, non_blocking=True)               # (B, K, C)
        cat = batch["category"].to(dev, non_blocking=True).long()
        parent_meth = batch["parent_meth"].to(dev, non_blocking=True).long()
        # 5-bucket id from (category, parent_meth) — see _compute_bucket_id
        bucket = _compute_bucket_id(cat, parent_meth)

        B = real.shape[0]
        z = torch.randn(B, g_cfg.z_dim, device=dev)
        fake = model(base_fwd, base_rev, meth_fwd, meth_rev, z)         # (B, K, C)

        real_flat = real.reshape(B, -1)
        fake_flat = fake.reshape(B, -1)

        ed_loss, _ = bucketed_energy_distance(
            real_flat, fake_flat, bucket, n_buckets, min_samples=bucket_min_samples
        )
        pos_loss, _ = spatial_per_position_w1(
            real, fake, bucket, n_buckets, min_samples=bucket_min_samples
        )
        loss = ed_loss + lambda_pos * pos_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        running["ed"] += float(ed_loss.detach().item())
        running["pos"] += float(pos_loss.detach().item())
        running["loss"] += float(loss.detach().item())
        running["n"] += 1

        if step % log_every == 0 and running["n"] > 0:
            now = time.time()
            steps_per_sec = running["n"] / max(now - last_log, 1e-6)
            log.info(
                "step %6d  loss=%.4f  ed=%.4f  pos_w1=%.4f  lr=%.2e  steps/s=%.1f",
                step, running["loss"] / running["n"],
                running["ed"] / running["n"],
                running["pos"] / running["n"],
                base_lr * _lr_mult(step),
                steps_per_sec,
            )
            running = {"ed": 0.0, "pos": 0.0, "loss": 0.0, "n": 0}
            last_log = now

        if step > 0 and step % ckpt_every == 0:
            _save_checkpoint(Path(ckpt_dir), step, model, opt, cfg)

        # Held-out W1 eval (same bucketing as kinsim_NN — numbers directly
        # comparable to v6 thesis W1=2.017). Saves best_G.pt on improvement.
        if step > 0 and step % eval_every == 0 and test_shard_paths:
            metrics = held_out_w1(
                model, test_shard_paths,
                g_cfg.n_meth_types, g_cfg.z_dim, dev,
            )
            w1 = metrics.get("w1_overall", float("nan"))
            log.info(
                "EVAL step %d  w1_overall=%.3f  %s",
                step, w1,
                "  ".join(f"{k}={v:.2f}" for k, v in metrics.items()
                          if k != "w1_overall"),
            )
            if not (w1 != w1) and w1 < best_w1:        # not NaN and lower
                best_w1 = w1
                best_path = Path(ckpt_dir) / "best_G.pt"
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "model_config": asdict(model.cfg),
                    "train_config": cfg,
                    "w1_overall": float(w1),
                }, best_path)
                log.info("New best G  (w1_overall=%.3f)  → %s", best_w1, best_path)

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
