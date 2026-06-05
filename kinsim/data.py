"""Dataset wrapper for kinsim.

Reuses the shard format and the streaming MultiShardDataset from
kinsim_NN — no extraction code is duplicated. The shards are written by
``kinsim_nn extract`` and consumed here unchanged.
"""
from __future__ import annotations

from pathlib import Path

from kinsim_NN.data.dataset import MultiShardDataset, list_shards


def build_train_dataset(
    shards_dir: Path,
    n_meth_types: int,
    test_strains: list[str] | set[str],
    seed: int = 42,
) -> MultiShardDataset:
    """Return a MultiShardDataset that excludes the test strains."""
    exclude = set(test_strains or [])
    paths = list_shards(Path(shards_dir), exclude_strains=exclude)
    if not paths:
        raise FileNotFoundError(
            f"No training shards found in {shards_dir} after excluding "
            f"test_strains={sorted(exclude)}"
        )
    return MultiShardDataset(
        shard_paths=paths,
        n_meth_types=n_meth_types,
        shuffle_shards=True,
        shuffle_rows=True,
        seed=seed,
    )


__all__ = ["build_train_dataset"]
