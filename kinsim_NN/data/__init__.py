"""kinsim_NN data layer: shard schema (numpy-only) + PyTorch datasets.

The dataset classes (``ShardedDataset``, ``MultiShardDataset``) require
``torch`` at import time. To keep ``extract`` runnable on CPU-only,
torch-less nodes, this package does NOT eagerly import ``dataset``. Use
the explicit import paths:

    from kinsim_NN.data.shard import read_shard, write_shard         # numpy + pickle only
    from kinsim_NN.data.dataset import ShardedDataset, MultiShardDataset  # needs torch
"""
from .shard import (
    SHARD_CONFIG_VERSION,
    ShardData,
    read_shard,
    write_shard,
    empty_shard,
    finalize_shard,
)

__all__ = [
    "SHARD_CONFIG_VERSION",
    "ShardData",
    "read_shard",
    "write_shard",
    "empty_shard",
    "finalize_shard",
]
