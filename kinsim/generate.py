"""``kinsim generate`` — emit a BAM with synthetic kinetics from a kinsim ckpt.

Reuses **the entire** kinsim_NN/generate.py pipeline for motif scanning,
BAM I/O, alignment-aware ref/query mapping, fn/rn handling, codec
encoding, multiprocess sharding — unchanged. The only thing this module
swaps is the model loader: where kinsim_NN.generate loads its WGAN-GP
generator, we load a kinsim (energy-distance) checkpoint and wrap it to
match the same external API.

CLI:

    python -m kinsim generate <input.bam> <ref.fa> <ckpt> <motifs.csv> <out.bam>

``<ckpt>`` can be either:

* a directory containing a single G_step*.pt + model_config.json (we
  pick the latest step), OR
* an explicit path to a G_step*.pt file (we look for model_config.json
  next to it).

Everything else (--n-workers, --batch-size, --seed, ...) is passed
through to kinsim_NN.generate.main.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

import kinsim_NN.generate as _v6_gen
from kinsim_NN.models.generator import TransformerGenerator as _V6Gen

from .model import GeneratorConfig, TransformerGenerator as _KinsimGen


log = logging.getLogger("kinsim.generate")


class _AdaptedKinsimGenerator(_V6Gen):
    """Subclass of kinsim_NN's TransformerGenerator that delegates the
    forward pass to a kinsim TransformerGenerator.

    Subclassing the v6 class (rather than duck-typing) avoids any issubclass
    checks downstream and keeps ``sample_z`` consistent. We only override
    ``__init__`` (to skip the v6 nn.Module setup since we hold a kinsim model
    instead) and ``forward``.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, kinsim_model: _KinsimGen):
        torch.nn.Module.__init__(self)  # raw nn.Module init, skip v6's setup
        self._g = kinsim_model
        # kinsim_NN reads g.z_dim and g.k for window sizing / sampling.
        self.z_dim = kinsim_model.cfg.z_dim
        self.k = kinsim_model.cfg.k
        self.n_meth_types = kinsim_model.cfg.n_meth_types
        self.d_model = kinsim_model.cfg.d_model

    def forward(
        self,
        z: torch.Tensor,
        base_fwd_onehot: torch.Tensor,
        base_rev_onehot: torch.Tensor,
        meth_fwd_onehot: torch.Tensor,
        meth_rev_onehot: torch.Tensor,
    ) -> torch.Tensor:
        # kinsim's forward takes z LAST (style choice), v6 takes z FIRST.
        return self._g(
            base_fwd_onehot,
            base_rev_onehot,
            meth_fwd_onehot,
            meth_rev_onehot,
            z,
        )

    def sample_z(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=device)


def _resolve_ckpt(ckpt_path: Path) -> tuple[Path, Path]:
    """Return ``(ckpt_file, model_config_json_path)``.

    Accepts either a directory or an explicit G_step*.pt file.
    """
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_dir():
        cands = sorted(ckpt_path.glob("G_step*.pt"))
        if not cands:
            raise FileNotFoundError(f"No G_step*.pt under {ckpt_path}")
        # Prefer best_G.pt if it exists.
        best = ckpt_path / "best_G.pt"
        ckpt_file = best if best.is_file() else cands[-1]
        cfg_file = ckpt_path / "model_config.json"
    else:
        ckpt_file = ckpt_path
        cfg_file = ckpt_path.parent / "model_config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(
            f"{cfg_file} not found — kinsim ckpts need model_config.json "
            f"alongside G_step*.pt. Either point --ckpt to the ckpt dir, "
            f"or move model_config.json next to the .pt file."
        )
    return ckpt_file, cfg_file


def _load_kinsim_generator(ckpt_path: Path, device: torch.device):
    """Custom loader that replaces kinsim_NN.generate._load_generator.

    Returns ``(adapted_g, v6_compat_cfg)`` where ``v6_compat_cfg`` is a dict
    shaped like the v6 model_config.json so the rest of kinsim_NN.generate
    can read fields by the same keys.
    """
    ckpt_file, cfg_file = _resolve_ckpt(Path(ckpt_path))
    with open(cfg_file, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    # kinsim model_config.json is the flat dataclass dump.
    gcfg = GeneratorConfig(**raw_cfg)
    g_real = _KinsimGen(gcfg).to(device)
    state = torch.load(ckpt_file, map_location=device, weights_only=False)
    g_real.load_state_dict(state["model_state"])
    g_real.eval()
    g = _AdaptedKinsimGenerator(g_real).to(device)
    g.eval()
    # Re-shape config to the v6 layout so any downstream cfg["generator"]["x"]
    # lookups in kinsim_NN.generate keep working.
    v6_cfg = {
        "k": gcfg.k,
        "n_meth_types": gcfg.n_meth_types,
        "generator": {
            "d_model": gcfg.d_model,
            "n_layers": gcfg.n_layers,
            "n_heads": gcfg.n_heads,
            "z_dim": gcfg.z_dim,
            "pos_embed_dim": gcfg.pos_embed_dim,
            "drop_rate": gcfg.drop_rate,
        },
        "meth_id_by_name": {"none": 0, "m6A": 1, "m4C": 2, "m5C": 3},
    }
    log.info("Loaded kinsim G from %s  (k=%d, z_dim=%d, d_model=%d, params=%.2fM)",
             ckpt_file, gcfg.k, gcfg.z_dim, gcfg.d_model,
             g_real.num_parameters() / 1e6)
    return g, v6_cfg


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — monkey-patches kinsim_NN.generate._load_generator
    to use kinsim's loader, then defers to kinsim_NN.generate.main for all
    BAM / motif / pipeline logic."""
    # Pre-parse just enough to find --log-level / --help. The full arg parse
    # is done by kinsim_NN.generate.main.
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        # Show kinsim_NN's help — it documents every flag.
        _v6_gen.main(["--help"])
        return

    # Swap the model loader.
    _v6_gen._load_generator = _load_kinsim_generator  # type: ignore[attr-defined]

    _v6_gen.main(argv)


if __name__ == "__main__":
    main()
