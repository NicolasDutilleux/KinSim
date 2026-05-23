"""kinsim_NN data layer: shard schema + PyTorch datasets."""
from .dataset import (
    ShardedDataset,
    MultiShardDataset,
    list_shards,
    shard_sample_id,
)
from .shard import (
    SHARD_CONFIG_VERSION,
    ShardData,
    read_shard,
    write_shard,
    empty_shard,
    finalize_shard,
)

__all__ = [
    "ShardedDataset",
    "MultiShardDataset",
    "list_shards",
    "shard_sample_id",
    "SHARD_CONFIG_VERSION",
    "ShardData",
    "read_shard",
    "write_shard",
    "empty_shard",
    "finalize_shard",
]
