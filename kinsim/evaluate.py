"""Held-out W1 evaluation for kinsim checkpoints.

Reproduces the same per-meth and per-category W1 bucketing as
kinsim_NN's _evaluate_on_shards so the numbers reported here are
directly comparable to v6's W1 = 2.017 (thesis §5.3).

Two entry points:

* :func:`evaluate_checkpoint` — load one .pt, return the W1 dict.
* :func:`evaluate_directory` — iterate over every G_step*.pt in a
  ckpt dir, report the trajectory, identify the best.

CLI:

    python -m kinsim evaluate <ckpt_or_dir> <shards_dir> [--config kinsim/config.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from kinsim_NN.data.dataset import ShardedDataset
from kinsim_NN.data.shard import read_shard
from kinsim_NN.utils.metrics import wasserstein_1d
from kinsim_NN.utils.pacbio_codec import log1p_frames_to_uint8

from .model import GeneratorConfig, TransformerGenerator


log = logging.getLogger("kinsim.evaluate")


def find_test_shard_paths(shards_dir: Path, test_strains: list[str]) -> list[Path]:
    """Find shards matching test_strains by trailing-component match.

    Mirrors the eval-side glob convention from kinsim_NN: a strain entry
    like ``bc2034`` matches ``strepto_bc2034_shard.pkl`` AND
    ``vega_bc2034_shard.pkl`` (both lineages).
    """
    shards_dir = Path(shards_dir)
    paths: list[Path] = []
    for sid in test_strains:
        paths.extend(sorted(shards_dir.glob(f"*{sid}_shard.pkl")))
    return paths


@torch.no_grad()
def _evaluate(
    generator: TransformerGenerator,
    test_shard_paths: list[Path],
    n_meth_types: int,
    z_dim: int,
    device: torch.device,
    max_samples_per_shard: int = 4000,
) -> dict[str, float]:
    """Compute W1 per meth_id (centre) and per category, plus w1_overall.

    Bucketing logic identical to kinsim_NN._evaluate_on_shards (Bug 7 + 8
    in BUGS_FOUND.md): per-meth buckets sum both strands when palindromic;
    per-category buckets count one contribution per row.
    """
    generator.eval()

    real_by_m: dict[int, list[int]] = {}
    gen_by_m: dict[int, list[int]] = {}
    cat_names = {0: "baseline", 1: "slowed", 2: "near_meth"}
    real_by_cat: dict[int, list[int]] = {0: [], 1: [], 2: []}
    gen_by_cat: dict[int, list[int]] = {0: [], 1: [], 2: []}

    for p in test_shard_paths:
        try:
            shard = read_shard(p)
        except (OSError, EOFError, ValueError, KeyError) as e:
            log.warning("Skipping unreadable test shard %s: %s", p, e)
            continue
        if shard.n == 0:
            continue
        ds = ShardedDataset(shard, n_meth_types)
        idxs = np.random.default_rng(0).permutation(shard.n)[:max_samples_per_shard]
        batch_items = [ds[int(i)] for i in idxs]
        batch = {
            k: (torch.stack([b[k] for b in batch_items])
                if k not in ("category", "parent_meth")
                else torch.tensor([b[k] for b in batch_items]))
            for k in batch_items[0]
        }
        B = batch["signal"].size(0)
        z = torch.randn(B, z_dim, device=device)
        fake = generator(
            batch["base_fwd_onehot"].to(device),
            batch["base_rev_onehot"].to(device),
            batch["meth_fwd_onehot"].to(device),
            batch["meth_rev_onehot"].to(device),
            z,
        ).float()
        half = shard.k // 2
        gen_center = fake[:, half].cpu().numpy()
        gen_u8 = log1p_frames_to_uint8(gen_center)
        real_u8 = shard.signal[idxs, half]
        mf = shard.meth_fwd[idxs, half]
        mr = shard.meth_rev[idxs, half]
        cats = shard.category[idxs]
        for i in range(real_u8.shape[0]):
            cat = int(cats[i])
            contributed = False
            if mf[i] > 0:
                m_id = int(mf[i])
                real_by_m.setdefault(m_id, []).append(int(real_u8[i, 0]))
                gen_by_m.setdefault(m_id, []).append(int(gen_u8[i, 0]))
                if not contributed and cat in real_by_cat:
                    real_by_cat[cat].append(int(real_u8[i, 0]))
                    gen_by_cat[cat].append(int(gen_u8[i, 0]))
                    contributed = True
            if mr[i] > 0:
                m_id = int(mr[i])
                real_by_m.setdefault(m_id, []).append(int(real_u8[i, 2]))
                gen_by_m.setdefault(m_id, []).append(int(gen_u8[i, 2]))
                if not contributed and cat in real_by_cat:
                    real_by_cat[cat].append(int(real_u8[i, 2]))
                    gen_by_cat[cat].append(int(gen_u8[i, 2]))
                    contributed = True
            if mf[i] == 0 and mr[i] == 0:
                real_by_m.setdefault(0, []).append(int(real_u8[i, 0]))
                gen_by_m.setdefault(0, []).append(int(gen_u8[i, 0]))
                if cat in real_by_cat:
                    real_by_cat[cat].append(int(real_u8[i, 0]))
                    gen_by_cat[cat].append(int(gen_u8[i, 0]))

    generator.train()

    out: dict[str, float] = {}
    all_real, all_gen = [], []
    for m_id in sorted(set(real_by_m) | set(gen_by_m)):
        r = np.asarray(real_by_m.get(m_id, []), dtype=np.float32)
        g = np.asarray(gen_by_m.get(m_id, []), dtype=np.float32)
        if r.size > 0 and g.size > 0:
            out[f"w1_meth{m_id}"] = float(wasserstein_1d(r, g))
        all_real.extend(r.tolist())
        all_gen.extend(g.tolist())
    for cat_id, cat_name in cat_names.items():
        r = np.asarray(real_by_cat.get(cat_id, []), dtype=np.float32)
        g = np.asarray(gen_by_cat.get(cat_id, []), dtype=np.float32)
        if r.size > 0 and g.size > 0:
            out[f"w1_{cat_name}"] = float(wasserstein_1d(r, g))
        else:
            out[f"w1_{cat_name}"] = float("nan")
    if all_real and all_gen:
        out["w1_overall"] = float(wasserstein_1d(
            np.asarray(all_real, dtype=np.float32),
            np.asarray(all_gen, dtype=np.float32),
        ))
    else:
        out["w1_overall"] = float("nan")
    return out


def held_out_w1(
    generator: TransformerGenerator,
    test_shard_paths: list[Path],
    n_meth_types: int,
    z_dim: int,
    device: torch.device,
    max_samples_per_shard: int = 4000,
) -> dict[str, float]:
    """Public entry point used by both the standalone evaluator and the
    training loop. Thin wrapper that keeps imports tidy."""
    return _evaluate(
        generator, test_shard_paths, n_meth_types, z_dim, device,
        max_samples_per_shard=max_samples_per_shard,
    )


def evaluate_checkpoint(
    ckpt_path: Path,
    shards_dir: Path,
    test_strains: list[str],
    device: torch.device | None = None,
    max_samples_per_shard: int = 4000,
) -> dict[str, float]:
    """Load a single G_step*.pt checkpoint and compute W1 on test_strains."""
    ckpt_path = Path(ckpt_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GeneratorConfig(**ckpt["model_config"])
    g = TransformerGenerator(cfg).to(device)
    g.load_state_dict(ckpt["model_state"])
    paths = find_test_shard_paths(shards_dir, test_strains)
    if not paths:
        raise FileNotFoundError(
            f"No test shards found in {shards_dir} for test_strains={test_strains}"
        )
    return held_out_w1(g, paths, cfg.n_meth_types, cfg.z_dim, device,
                       max_samples_per_shard=max_samples_per_shard)


def evaluate_directory(
    ckpt_dir: Path,
    shards_dir: Path,
    test_strains: list[str],
    device: torch.device | None = None,
    max_samples_per_shard: int = 4000,
) -> list[tuple[int, dict[str, float]]]:
    """Evaluate every G_step*.pt in ckpt_dir; return [(step, metrics), ...]
    sorted by step."""
    ckpt_dir = Path(ckpt_dir)
    paths = sorted(ckpt_dir.glob("G_step*.pt"))
    if not paths:
        raise FileNotFoundError(f"No G_step*.pt under {ckpt_dir}")
    out: list[tuple[int, dict[str, float]]] = []
    for p in paths:
        step = int(p.stem.removeprefix("G_step"))
        try:
            m = evaluate_checkpoint(
                p, shards_dir, test_strains, device,
                max_samples_per_shard=max_samples_per_shard,
            )
        except (OSError, RuntimeError, KeyError) as e:
            log.warning("Skipping %s: %s", p.name, e)
            continue
        out.append((step, m))
        log.info(
            "step %7d  w1_overall=%.3f  %s",
            step, m.get("w1_overall", float("nan")),
            "  ".join(f"{k}={v:.2f}" for k, v in m.items() if k != "w1_overall"),
        )
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute held-out W1 on kinsim checkpoints."
    )
    p.add_argument("ckpt_or_dir", type=Path,
                   help="Either a single G_step*.pt file or a directory of them.")
    p.add_argument("shards_dir", type=Path,
                   help="Directory of *_shard.pkl files (same one used for training).")
    p.add_argument("--config", type=Path,
                   default=Path(__file__).resolve().parent / "config.yaml",
                   help="YAML config (for split.test_strains).")
    p.add_argument("--test-strains", default=None,
                   help="Comma-separated override for split.test_strains.")
    p.add_argument("--max-samples", type=int, default=4000,
                   help="Max samples per test shard to evaluate on (default 4000).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        stream=sys.stdout,
    )
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.test_strains:
        test_strains = [s.strip() for s in args.test_strains.split(",") if s.strip()]
    else:
        test_strains = list(cfg.get("split", {}).get("test_strains", []))
    if not test_strains:
        sys.exit("No test_strains configured (set --test-strains or split.test_strains in YAML).")

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s, test_strains=%s", device, test_strains)

    if args.ckpt_or_dir.is_dir():
        results = evaluate_directory(
            args.ckpt_or_dir, args.shards_dir, test_strains, device,
            max_samples_per_shard=args.max_samples,
        )
        if not results:
            sys.exit("No usable checkpoints found.")
        # Identify the best (lowest w1_overall, ignoring NaN).
        valid = [(s, m) for s, m in results
                 if not (m["w1_overall"] != m["w1_overall"])]
        if valid:
            best_step, best_m = min(valid, key=lambda x: x[1]["w1_overall"])
            log.info("")
            log.info("=== BEST ===")
            log.info(
                "step %d  w1_overall=%.4f  %s",
                best_step, best_m["w1_overall"],
                "  ".join(f"{k}={v:.3f}" for k, v in best_m.items() if k != "w1_overall"),
            )
        # Dump full trajectory as JSON next to ckpt_dir for downstream plotting.
        out_path = args.ckpt_or_dir / "eval_w1.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"step": s, **m} for s, m in results],
                f, indent=2,
            )
        log.info("Wrote %s", out_path)
    else:
        metrics = evaluate_checkpoint(
            args.ckpt_or_dir, args.shards_dir, test_strains, device,
            max_samples_per_shard=args.max_samples,
        )
        log.info(
            "w1_overall=%.4f  %s",
            metrics["w1_overall"],
            "  ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "w1_overall"),
        )


if __name__ == "__main__":
    main()
