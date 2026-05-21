"""Hardcoded fallback constants for kmer geometry (bilateral v2).

These kick in ONLY when ``kinsim_config.yaml::extraction`` is unreachable.
YAML is the source of truth at runtime. Bilateral v2 captures the full
K-position reverse-strand meth context — no ``rev_meth_offsets``.
"""

DEFAULT_KMER_SIZE: int = 11
DEFAULT_UPSTREAM: int = 7
DEFAULT_DOWNSTREAM: int = 3

assert DEFAULT_UPSTREAM + 1 + DEFAULT_DOWNSTREAM == DEFAULT_KMER_SIZE

BAM_TAG_MAX: int = 255
