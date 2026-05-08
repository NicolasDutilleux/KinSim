"""kinsim_baseline — purely statistical per-kmer (IPD, PW) distribution.

No neural network. No methylation conditioning. For every 11-mer that
appears in the BASELINE category of a KinSim extract shard, this stores a
small empirical sample of (IPD, PW) values from real PacBio data and
generates kinetics by drawing from those samples directly.

This is the **null comparison baseline** for the meth-conditioned KinSim
neural model. Same input shards, same BAM output format → side-by-side
``kinsim verify-generate`` runs produce a clean apples-to-apples
comparison: where does the neural conditioning actually win, and where
does pure per-kmer empirical sampling already capture the kinetics?

Pipeline:

    extract shards/ ──► kinsim_baseline build  ──► kmer_table.npz
                                                         │
    input.bam ────────► kinsim_baseline generate ◄──────┘
                                 │
                                 ▼
                          output.bam (fi/fp tags)

Module layout:

    distribution.py   — KmerDistribution class (storage + sampling)
    build_table.py    — build a table from a shards/ directory
    generate.py       — produce a BAM with fi/fp tags from an input BAM
    __main__.py       — ``python -m kinsim_baseline {build,generate}``
"""

from .distribution import KmerDistribution

__all__ = ["KmerDistribution"]
