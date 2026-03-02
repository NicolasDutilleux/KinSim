"""Backward-compatibility shim — the real implementation lives in kinsim.common.extract.

All logic (extract_samples_from_bam, merge_shards, CLI) was moved to the shared
common/ package so both cGAN and MLP can use the same data pipeline.

The CLI entry points 'kinsim cgan extract' and 'kinsim cgan merge' are preserved
here for backward compatibility with existing SLURM scripts.
"""

from ...common.extract import (  # noqa: F401  (re-exported)
    extract_samples_from_bam,
    merge_shards,
    main,
)
