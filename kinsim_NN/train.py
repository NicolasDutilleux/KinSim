"""``kinsim_nn train`` — pure-PyTorch cGAN training loop (WGAN-GP).

Trains a :class:`TransformerGenerator` against a
:class:`TransformerDiscriminator` on the shards produced by
``kinsim_nn extract``. No PyTorch Lightning — explicit loop because GAN
training benefits from fine control over optimizer steps, EMA, and
checkpoint policy.

Outputs (under ``--ckpt-dir``):
    G.pt                       latest generator state
    D.pt                       latest discriminator state
    optG.pt / optD.pt          optimizer states
    best_G.pt                  generator with best held-out Wasserstein
    model_config.json          frozen architecture + config (for generate/evaluate)
    metrics.csv                per-step losses
    tb/...                     TensorBoard logs
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import __version__
from .data.dataset import MultiShardDataset, ShardedDataset, list_shards
from .data.shard import read_shard
from .models.discriminator import TransformerDiscriminator
from .models.generator import TransformerGenerator
from .utils.config import KinsimNNConfig, load_config, setup_logging
from .utils.losses import gradient_penalty, wgan_g_loss, wgan_gp_d_loss
from .utils.pacbio_codec import log1p_frames_to_uint8


log = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _save_model_config(ckpt_dir: Path, cfg: KinsimNNConfig) -> None:
    payload = {
        "kinsim_nn_version": __version__,
        "git_sha": _git_sha(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config_version": "kinsim_NN-1",
        "k": cfg.window.k,
        "half_width": cfg.window.half_width,
        "n_channels": cfg.window.n_channels,
        "n_meth_types": cfg.n_meth_types,
        "meth_id_by_name": cfg.meth_id_by_name,
        "treat_modified_base_as": cfg.treat_modified_base_as,
        "generator": cfg.model.generator.__dict__,
        "discriminator": cfg.model.discriminator.__dict__,
        "train": cfg.train.__dict__,
    }
    out = ckpt_dir / "model_config.json"
    out.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", out)


def _build_models(cfg: KinsimNNConfig, device: torch.device):
    g = TransformerGenerator(
        k=cfg.window.k,
        n_meth_types=cfg.n_meth_types,
        d_model=cfg.model.generator.d_model,
        n_layers=cfg.model.generator.n_layers,
        n_heads=cfg.model.generator.n_heads,
        z_dim=cfg.model.generator.z_dim,
        pos_embed_dim=cfg.model.generator.pos_embed_dim,
        drop_rate=cfg.model.generator.drop_rate,
    ).to(device)
    d = TransformerDiscriminator(
        k=cfg.window.k,
        n_meth_types=cfg.n_meth_types,
        d_model=cfg.model.discriminator.d_model,
        n_layers=cfg.model.discriminator.n_layers,
        n_heads=cfg.model.discriminator.n_heads,
        spectral_norm=cfg.model.discriminator.spectral_norm,
        pos_embed_dim=cfg.model.discriminator.pos_embed_dim,
        drop_rate=0.0,
    ).to(device)
    n_g = sum(p.numel() for p in g.parameters())
    n_d = sum(p.numel() for p in d.parameters())
    log.info("Generator params: %.2f M", n_g / 1e6)
    log.info("Discriminator params: %.2f M", n_d / 1e6)
    return g, d


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    out = {}
    for k in batch[0]:
        if k == "category":
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _g_forward(g, batch, z):
    return g(
        z=z,
        base_fwd_onehot=batch["base_fwd_onehot"],
        base_rev_onehot=batch["base_rev_onehot"],
        meth_fwd_onehot=batch["meth_fwd_onehot"],
        meth_rev_onehot=batch["meth_rev_onehot"],
    )


def _d_forward(d, signal, batch):
    return d(
        signal=signal,
        base_fwd_onehot=batch["base_fwd_onehot"],
        base_rev_onehot=batch["base_rev_onehot"],
        meth_fwd_onehot=batch["meth_fwd_onehot"],
        meth_rev_onehot=batch["meth_rev_onehot"],
    )


def _cond_kwargs(batch):
    return {
        "base_fwd_onehot": batch["base_fwd_onehot"],
        "base_rev_onehot": batch["base_rev_onehot"],
        "meth_fwd_onehot": batch["meth_fwd_onehot"],
        "meth_rev_onehot": batch["meth_rev_onehot"],
    }


def _save_checkpoint(path: Path, model, optimizer, step: int) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, path)


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein-1 distance via sorted-quantile interpolation."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.size, b.size, 1024)
    qs = np.linspace(0.0, 1.0, n)
    aq = np.interp(qs, np.linspace(0.0, 1.0, a.size), np.sort(a))
    bq = np.interp(qs, np.linspace(0.0, 1.0, b.size), np.sort(b))
    return float(np.mean(np.abs(aq - bq)))


@torch.no_grad()
def _evaluate_on_shards(
    g: TransformerGenerator,
    test_shard_paths: list[Path],
    n_meth_types: int,
    device: torch.device,
    max_samples: int = 4000,
) -> dict[str, float]:
    """Compute per-meth-type Wasserstein-1 on held-out test shards.

    Returns ``{"w1_overall": ..., "w1_baseline": ..., "w1_m6A": ...}`` etc.
    Capped at ``max_samples`` per type to keep eval fast (~seconds).
    """
    g.eval()
    real_by_m: dict[int, list[int]] = {}
    gen_by_m: dict[int, list[int]] = {}
    for p in test_shard_paths:
        if sum(len(v) for v in real_by_m.values()) >= max_samples * 4:
            break
        try:
            shard = read_shard(p)
        except (OSError, EOFError, ValueError):
            continue
        if shard.n == 0:
            continue
        ds = ShardedDataset(shard, n_meth_types)
        idxs = np.random.default_rng(0).permutation(shard.n)[:max_samples]
        batch_items = [ds[int(i)] for i in idxs]
        batch = {
            k: torch.stack([b[k] for b in batch_items]) if k != "category"
            else torch.tensor([b[k] for b in batch_items])
            for k in batch_items[0]
        }
        z = g.sample_z(batch["signal"].size(0), device=device)
        gen = g(
            z,
            batch["base_fwd_onehot"].to(device),
            batch["base_rev_onehot"].to(device),
            batch["meth_fwd_onehot"].to(device),
            batch["meth_rev_onehot"].to(device),
        )
        half = shard.k // 2
        gen_center = gen[:, half].cpu().numpy()                     # (B, 4)
        gen_u8 = log1p_frames_to_uint8(gen_center)
        real_u8 = shard.signal[idxs, half]                          # (B, 4)
        mf = shard.meth_fwd[idxs, half]
        mr = shard.meth_rev[idxs, half]
        for i in range(real_u8.shape[0]):
            if mf[i] > 0:
                m_id, ch = int(mf[i]), 0
            elif mr[i] > 0:
                m_id, ch = int(mr[i]), 2
            else:
                m_id, ch = 0, 0
            real_by_m.setdefault(m_id, []).append(int(real_u8[i, ch]))
            gen_by_m.setdefault(m_id, []).append(int(gen_u8[i, ch]))
    g.train()
    out: dict[str, float] = {}
    all_real, all_gen = [], []
    for m_id in sorted(set(real_by_m) | set(gen_by_m)):
        r = np.asarray(real_by_m.get(m_id, []), dtype=np.float32)
        gg = np.asarray(gen_by_m.get(m_id, []), dtype=np.float32)
        out[f"w1_meth{m_id}"] = _wasserstein_1d(r, gg)
        all_real.extend(r.tolist())
        all_gen.extend(gg.tolist())
    out["w1_overall"] = _wasserstein_1d(
        np.asarray(all_real, dtype=np.float32),
        np.asarray(all_gen, dtype=np.float32),
    )
    return out


def train(
    shards_dir: Path,
    ckpt_dir: Path,
    cfg: KinsimNNConfig,
    test_strains: tuple[str, ...] | None,
    resume: bool = False,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Data
    test_strains = test_strains or cfg.split.test_strains
    shards = list_shards(shards_dir, exclude_strains=set(test_strains))
    if not shards:
        sys.exit(f"No shards under {shards_dir} (after excluding {test_strains})")
    log.info("Training shards: %d  test_strains=%s", len(shards), test_strains)

    dataset = MultiShardDataset(
        shard_paths=shards,
        n_meth_types=cfg.n_meth_types,
        shuffle_shards=True,
        shuffle_rows=True,
        seed=cfg.train.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        collate_fn=_collate,
        drop_last=True,
    )

    # Models
    g, d = _build_models(cfg, device)
    opt_g = torch.optim.Adam(g.parameters(), lr=cfg.train.lr_g,
                             betas=(cfg.train.beta1, cfg.train.beta2))
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg.train.lr_d,
                             betas=(cfg.train.beta1, cfg.train.beta2))

    start_step = 0
    if resume:
        g_path = ckpt_dir / "G.pt"
        d_path = ckpt_dir / "D.pt"
        if not (g_path.is_file() and d_path.is_file()):
            raise FileNotFoundError(
                f"--resume requires BOTH G.pt and D.pt under {ckpt_dir} "
                f"(found G.pt={g_path.is_file()} D.pt={d_path.is_file()}). "
                f"Resuming G alone breaks WGAN-GP's critic Lipschitz constraint. "
                f"Either copy the missing file or start fresh (drop --resume)."
            )
        for label, model, opt, fname in [
            ("G", g, opt_g, "G.pt"), ("D", d, opt_d, "D.pt")
        ]:
            p = ckpt_dir / fname
            ckpt = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            opt.load_state_dict(ckpt["optimizer"])
            start_step = max(start_step, int(ckpt.get("step", 0)))
            log.info("Resumed %s from %s (step %d)", label, p, start_step)

    # Don't overwrite model_config.json on resume — silently bumping it can
    # break a checkpoint if the YAML changed between runs.
    cfg_json = ckpt_dir / "model_config.json"
    if not (resume and cfg_json.is_file()):
        _save_model_config(ckpt_dir, cfg)
    else:
        log.info("Resume: keeping existing %s (skipping rewrite)", cfg_json)

    # Held-out test shards for periodic eval
    test_shard_paths: list[Path] = []
    for sid in test_strains:
        p = Path(shards_dir) / f"{sid}_shard.pkl"
        if p.is_file():
            test_shard_paths.append(p)
    if test_shard_paths:
        log.info("Eval shards: %d", len(test_shard_paths))
    else:
        log.warning("No eval shards found for test_strains=%s", test_strains)

    tb = SummaryWriter(str(ckpt_dir / "tb"))
    csv_path = ckpt_dir / "metrics.csv"
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if start_step == 0:
        csv_w.writerow(["step", "phase", "d_loss", "g_loss", "d_real",
                        "d_fake", "gp", "w1_overall"])

    g.train(); d.train()
    step = start_step
    n_steps = cfg.train.n_steps
    best_w1 = float("inf")

    # On resume, jump the epoch counter forward so shuffle order differs from
    # the freshly-loaded shard order. Otherwise stay at 0 (the default).
    epoch = start_step // 1000 if start_step > 0 else 0
    if epoch > 0:
        dataset.set_epoch(epoch)
    data_iter = iter(loader)
    try:
        while step < n_steps:
            # --- D step (n_critic times) ---
            d_losses, d_reals, d_fakes, gps = [], [], [], []
            for _ in range(cfg.train.n_critic):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    dataset.set_epoch(epoch)
                    data_iter = iter(loader)
                    batch = next(data_iter)
                batch = _to_device(batch, device)
                bsz = batch["signal"].size(0)
                z = g.sample_z(bsz, device=device)
                with torch.no_grad():
                    fake = _g_forward(g, batch, z)
                d_real = _d_forward(d, batch["signal"], batch)
                d_fake = _d_forward(d, fake, batch)
                gp = gradient_penalty(
                    d, batch["signal"], fake,
                    cond_kwargs=_cond_kwargs(batch),
                    device=device,
                )
                d_loss = wgan_gp_d_loss(d_real, d_fake, gp,
                                        gp_lambda=cfg.train.gradient_penalty_lambda)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

                d_losses.append(d_loss.item())
                d_reals.append(d_real.mean().item())
                d_fakes.append(d_fake.mean().item())
                gps.append(gp.item())

            # --- G step ---
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                dataset.set_epoch(epoch)
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = _to_device(batch, device)
            bsz = batch["signal"].size(0)
            z = g.sample_z(bsz, device=device)
            fake = _g_forward(g, batch, z)
            d_fake = _d_forward(d, fake, batch)
            g_loss = wgan_g_loss(d_fake)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            step += 1

            # Logging
            if step % cfg.train.log_every == 0:
                d_loss_avg = float(np.mean(d_losses))
                d_real_avg = float(np.mean(d_reals))
                d_fake_avg = float(np.mean(d_fakes))
                gp_avg = float(np.mean(gps))
                g_loss_v = float(g_loss.item())
                tb.add_scalar("loss/D", d_loss_avg, step)
                tb.add_scalar("loss/G", g_loss_v, step)
                tb.add_scalar("D/real", d_real_avg, step)
                tb.add_scalar("D/fake", d_fake_avg, step)
                tb.add_scalar("D/gp", gp_avg, step)
                csv_w.writerow([step, "train", d_loss_avg, g_loss_v,
                                d_real_avg, d_fake_avg, gp_avg])
                csv_f.flush()
                log.info(
                    "step %d  D=%+.4f  G=%+.4f  d_real=%+.3f  d_fake=%+.3f  gp=%.3f",
                    step, d_loss_avg, g_loss_v, d_real_avg, d_fake_avg, gp_avg,
                )

            # Checkpoint
            if step % cfg.train.checkpoint_every == 0:
                _save_checkpoint(ckpt_dir / "G.pt", g, opt_g, step)
                _save_checkpoint(ckpt_dir / "D.pt", d, opt_d, step)
                log.info("Checkpointed at step %d", step)

            # Held-out evaluation (Wasserstein-1 per meth type)
            if (
                step % cfg.train.eval_every == 0
                and test_shard_paths
            ):
                eval_metrics = _evaluate_on_shards(
                    g, test_shard_paths, cfg.n_meth_types, device,
                )
                w1_overall = eval_metrics.get("w1_overall", float("nan"))
                for k, v in eval_metrics.items():
                    tb.add_scalar(f"eval/{k}", v, step)
                csv_w.writerow([step, "eval", "", "", "", "", "", w1_overall])
                csv_f.flush()
                log.info(
                    "EVAL step %d  w1_overall=%.3f  %s",
                    step, w1_overall,
                    "  ".join(f"{k}={v:.2f}" for k, v in eval_metrics.items()
                              if k != "w1_overall"),
                )
                if w1_overall < best_w1 and not np.isnan(w1_overall):
                    best_w1 = w1_overall
                    _save_checkpoint(ckpt_dir / "best_G.pt", g, opt_g, step)
                    log.info("New best G (w1_overall=%.3f) at step %d", best_w1, step)

    finally:
        _save_checkpoint(ckpt_dir / "G.pt", g, opt_g, step)
        _save_checkpoint(ckpt_dir / "D.pt", d, opt_d, step)
        csv_f.close()
        tb.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kinsim_nn train", description=__doc__)
    ap.add_argument("shards_dir", help="Directory containing extracted shards")
    ap.add_argument("ckpt_dir", help="Output directory for checkpoints")
    ap.add_argument("--config", default=None, help="kinsim_nn_config.yaml path")
    ap.add_argument("--test-strains", default=None,
                    help="Comma-separated sample_ids to exclude from training (overrides YAML)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing G.pt / D.pt")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(verbose=args.verbose)

    cfg = load_config(args.config)
    test_strains = (
        tuple(s.strip() for s in args.test_strains.split(",") if s.strip())
        if args.test_strains else None
    )

    train(
        shards_dir=Path(args.shards_dir),
        ckpt_dir=Path(args.ckpt_dir),
        cfg=cfg,
        test_strains=test_strains,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
