"""``kinsim_nn evaluate`` — distribution-level metrics on held-out shards.

For each test shard, we compute:
  * Per-kmer Wasserstein-1 distance between real per-read signal and
    a generated signal sample of equal size.
  * Per-meth-type pooled summary (median, IQR, mean) of real vs gen IPD
    distributions.

Output: HTML report + TSV of stats.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .data.dataset import ShardedDataset
from .data.shard import read_shard
from .models.generator import TransformerGenerator
from .utils.config import load_config, setup_logging
from .utils.metrics import wasserstein_1d as _wasserstein_1d
from .utils.pacbio_codec import log1p_frames_to_uint8


log = logging.getLogger(__name__)


def _load_generator(ckpt_dir: Path, device: torch.device):
    config_path = ckpt_dir / "model_config.json"
    cfg = json.loads(config_path.read_text())
    g = TransformerGenerator(
        k=cfg["k"],
        n_meth_types=cfg["n_meth_types"],
        d_model=cfg["generator"]["d_model"],
        n_layers=cfg["generator"]["n_layers"],
        n_heads=cfg["generator"]["n_heads"],
        z_dim=cfg["generator"]["z_dim"],
        pos_embed_dim=cfg["generator"]["pos_embed_dim"],
        drop_rate=cfg["generator"].get("drop_rate", 0.0),
    ).to(device)
    # Prefer best_G.pt > G.pt > most recent .pt (same precedence as generate.py)
    best = ckpt_dir / "best_G.pt"
    latest = ckpt_dir / "G.pt"
    if best.is_file():
        ckpt_path = best
    elif latest.is_file():
        ckpt_path = latest
    else:
        candidates = sorted(ckpt_dir.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
        ckpt_path = candidates[-1]
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    g.load_state_dict(state["state_dict"])
    g.eval()
    return g, cfg


@torch.no_grad()
def _generate_batch(g, batch, device, n_meth_types: int) -> np.ndarray:
    """Return per-token center-channel uint8 predictions for the batch.

    Shape: (B, 4) — IPD_fwd, PW_fwd, IPD_rev, PW_rev (uint8).
    """
    z = g.sample_z(batch["signal"].size(0), device=device)
    out = g(
        z,
        batch["base_fwd_onehot"].to(device),
        batch["base_rev_onehot"].to(device),
        batch["meth_fwd_onehot"].to(device),
        batch["meth_rev_onehot"].to(device),
    )
    K = out.shape[1]
    center = out[:, K // 2].cpu().numpy()                    # (B, 4)
    return log1p_frames_to_uint8(center)


def evaluate(
    ckpt_dir: Path,
    shards_dir: Path,
    output_prefix: Path,
    batch_size: int = 256,
) -> None:
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    g, model_cfg = _load_generator(ckpt_dir, device)
    n_meth_types = int(model_cfg["n_meth_types"])
    meth_name_by_id = {int(v): k for k, v in model_cfg["meth_id_by_name"].items()}

    shards = sorted(shards_dir.glob("*_shard.pkl"))
    if not shards:
        raise FileNotFoundError(f"No shards in {shards_dir}")

    # Pooled per-meth_type stats
    pooled_real: dict[int, list[int]] = defaultdict(list)
    pooled_gen: dict[int, list[int]] = defaultdict(list)

    for p in shards:
        log.info("Evaluating %s", p.name)
        shard = read_shard(p)
        if shard.n == 0:
            continue
        ds = ShardedDataset(shard, n_meth_types)
        # Iterate batches
        n = len(ds)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch_items = [ds[i] for i in range(start, stop)]
            batch = {
                key: torch.stack([item[key] for item in batch_items])
                if key != "category"
                else torch.tensor([item[key] for item in batch_items])
                for key in batch_items[0]
            }
            gen_centers = _generate_batch(g, batch, device, n_meth_types)
            half = shard.k // 2
            real_centers = shard.signal[start:stop, half]      # (B, 4) = (IPD_fwd, PW_fwd, IPD_rev, PW_rev)
            mf_center = shard.meth_fwd[start:stop, half]
            mr_center = shard.meth_rev[start:stop, half]
            # Strand pooling per sample (same rules as the in-training eval
            # in train.py): every sample contributes exactly one value per
            # bucket so per-meth_id W1 estimates are comparable across
            # buckets. Palindromic sites (both meth_fwd > 0 AND meth_rev > 0)
            # contribute to BOTH the fwd-meth and the rev-meth buckets so
            # neither strand is silently dropped.
            for i in range(real_centers.shape[0]):
                added = False
                if mf_center[i] > 0:
                    m_id = int(mf_center[i])
                    pooled_real[m_id].append(int(real_centers[i, 0]))
                    pooled_gen[m_id].append(int(gen_centers[i, 0]))
                    added = True
                if mr_center[i] > 0:
                    m_id = int(mr_center[i])
                    pooled_real[m_id].append(int(real_centers[i, 2]))
                    pooled_gen[m_id].append(int(gen_centers[i, 2]))
                    added = True
                if not added:
                    # Baseline: deterministic strand pick (channel 0) keeps
                    # the baseline count equal to the per-meth count.
                    pooled_real[0].append(int(real_centers[i, 0]))
                    pooled_gen[0].append(int(gen_centers[i, 0]))

    # Write summary TSV
    out = Path(str(output_prefix) + "_stats.tsv")
    with open(out, "w") as f:
        f.write("meth_id\tmeth_name\tn\treal_median\treal_mean\treal_sigma\t"
                "gen_median\tgen_mean\tgen_sigma\twasserstein_1d\n")
        for m_id in sorted(set(pooled_real.keys()) | set(pooled_gen.keys())):
            r = np.asarray(pooled_real.get(m_id, []))
            g_arr = np.asarray(pooled_gen.get(m_id, []))
            w1 = _wasserstein_1d(r.astype(np.float32), g_arr.astype(np.float32))
            f.write(
                f"{m_id}\t{meth_name_by_id.get(m_id, '?')}\t{r.size}\t"
                f"{np.median(r) if r.size else float('nan'):.2f}\t"
                f"{r.mean() if r.size else float('nan'):.2f}\t"
                f"{r.std() if r.size else float('nan'):.2f}\t"
                f"{np.median(g_arr) if g_arr.size else float('nan'):.2f}\t"
                f"{g_arr.mean() if g_arr.size else float('nan'):.2f}\t"
                f"{g_arr.std() if g_arr.size else float('nan'):.2f}\t"
                f"{w1:.3f}\n"
            )
    log.info("Wrote %s", out)
    for m_id in sorted(set(pooled_real.keys()) | set(pooled_gen.keys())):
        r = np.asarray(pooled_real.get(m_id, []))
        g_arr = np.asarray(pooled_gen.get(m_id, []))
        w1 = _wasserstein_1d(r.astype(np.float32), g_arr.astype(np.float32))
        log.info(
            "  meth=%s n=%d  real(med=%.1f σ=%.1f) gen(med=%.1f σ=%.1f) W1=%.2f",
            meth_name_by_id.get(m_id, "?"), r.size,
            np.median(r) if r.size else float("nan"),
            r.std() if r.size else float("nan"),
            np.median(g_arr) if g_arr.size else float("nan"),
            g_arr.std() if g_arr.size else float("nan"),
            w1,
        )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kinsim_nn evaluate", description=__doc__)
    ap.add_argument("ckpt_dir")
    ap.add_argument("shards_dir")
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.verbose:
        setup_logging(verbose=True)
    evaluate(
        ckpt_dir=Path(args.ckpt_dir),
        shards_dir=Path(args.shards_dir),
        output_prefix=Path(args.output_prefix),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
