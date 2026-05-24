"""YAML loader + dataclass schema for kinsim_NN.

The config file is ``kinsim_nn_config.yaml`` at the repo root by default,
overridable via ``--config`` flag or ``KINSIM_NN_CONFIG`` env var.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass schema (read after YAML load)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowParams:
    half_width: int = 10
    k: int = 21
    n_channels: int = 4


@dataclass(frozen=True)
class MethylationType:
    name: str
    id: int
    modified_base: str | None = None
    label_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractParams:
    baseline_min_dist: int = 20
    baseline_per_strain: int = 50_000
    reads_cap_per_position: int = 20
    min_mapq: int = 20            # was min_read_qv — it's MAPQ, not base QV
    bystrandify_pairing: bool = True
    meth_per_strain_cap: int = 0  # 0 = no cap; else random subsample meth positions


@dataclass(frozen=True)
class SplitParams:
    test_strains: tuple[str, ...] = ()
    test_fraction: float | None = None
    split_seed: int = 42


@dataclass(frozen=True)
class GeneratorParams:
    d_model: int = 192
    n_layers: int = 6
    n_heads: int = 6
    z_dim: int = 64
    drop_rate: float = 0.0
    pos_embed_dim: int = 16


@dataclass(frozen=True)
class DiscriminatorParams:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    spectral_norm: bool = True
    pos_embed_dim: int = 16
    drop_rate: float = 0.0


@dataclass(frozen=True)
class ModelParams:
    generator: GeneratorParams = field(default_factory=GeneratorParams)
    discriminator: DiscriminatorParams = field(default_factory=DiscriminatorParams)


@dataclass(frozen=True)
class TrainParams:
    loss: str = "wgan_gp"
    batch_size: int = 256
    n_critic: int = 5
    gradient_penalty_lambda: float = 10.0
    lr_g: float = 1e-4
    lr_d: float = 4e-4
    beta1: float = 0.0
    beta2: float = 0.9
    n_steps: int = 200_000
    checkpoint_every: int = 5000
    eval_every: int = 5000
    log_every: int = 100
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True


@dataclass(frozen=True)
class GenerateParams:
    use_fraction_bernoulli: bool = True
    n_context_skip: int = 10
    default_fi_for_unknown: int = 1


@dataclass(frozen=True)
class KinsimNNConfig:
    window: WindowParams
    methylation_types: tuple[MethylationType, ...]
    treat_modified_base_as: str | None
    labelers: tuple[dict, ...]
    extract: ExtractParams
    split: SplitParams
    model: ModelParams
    train: TrainParams
    generate: GenerateParams

    @property
    def n_meth_types(self) -> int:
        return max(t.id for t in self.methylation_types) + 1

    @property
    def meth_id_by_name(self) -> dict[str, int]:
        return {t.name: t.id for t in self.methylation_types}

    @property
    def meth_name_by_id(self) -> dict[int, str]:
        return {t.id: t.name for t in self.methylation_types}


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _default_config_path() -> Path:
    env = os.environ.get("KINSIM_NN_CONFIG")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    return here.parent.parent / "kinsim_nn_config.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> KinsimNNConfig:
    """Load and parse the YAML config into a frozen :class:`KinsimNNConfig`.

    Cached per resolved path. Pass ``None`` to use the default location
    (``KINSIM_NN_CONFIG`` env var or ``kinsim_nn_config.yaml`` next to
    the package).
    """
    if path is None:
        path = _default_config_path()
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"kinsim_NN config not found: {path}")
    log.info("Loading kinsim_NN config: %s", path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")

    win = WindowParams(**(raw.get("window") or {}))
    if win.k != 2 * win.half_width + 1:
        raise ValueError(
            f"window.k ({win.k}) must equal 2*half_width+1 ({2*win.half_width+1})"
        )

    meth_types = []
    raw_types = raw.get("methylation_types") or {}
    for name, body in raw_types.items():
        body = body or {}
        meth_types.append(MethylationType(
            name=name,
            id=int(body["id"]),
            modified_base=body.get("modified_base"),
            label_sources=tuple(body.get("label_sources") or ()),
        ))
    if not meth_types:
        raise ValueError("methylation_types must define at least 'none'")
    if not any(t.name == "none" for t in meth_types):
        raise ValueError("methylation_types must include 'none' (id=0)")

    # Normalise split.test_strains to a tuple of stripped strings (YAML
    # returns a list which would diverge from the CLI tuple).
    split_raw = dict(raw.get("split") or {})
    if "test_strains" in split_raw and split_raw["test_strains"] is not None:
        split_raw["test_strains"] = tuple(
            str(s).strip() for s in split_raw["test_strains"] if str(s).strip()
        )

    return KinsimNNConfig(
        window=win,
        methylation_types=tuple(meth_types),
        treat_modified_base_as=raw.get("treat_modified_base_as"),
        labelers=tuple(raw.get("labelers") or []),
        extract=ExtractParams(**(raw.get("extract") or {})),
        split=SplitParams(**split_raw),
        model=ModelParams(
            generator=GeneratorParams(**(raw.get("model", {}).get("generator") or {})),
            discriminator=DiscriminatorParams(**(raw.get("model", {}).get("discriminator") or {})),
        ),
        train=TrainParams(**(raw.get("train") or {})),
        generate=GenerateParams(**(raw.get("generate") or {})),
    )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quiet very chatty libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


__all__ = [
    "WindowParams",
    "MethylationType",
    "ExtractParams",
    "SplitParams",
    "GeneratorParams",
    "DiscriminatorParams",
    "ModelParams",
    "TrainParams",
    "GenerateParams",
    "KinsimNNConfig",
    "load_config",
    "setup_logging",
]
