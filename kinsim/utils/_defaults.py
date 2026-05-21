"""Hardcoded fallback constants for kmer geometry.

These kick in ONLY when ``kinsim_config.yaml::extraction`` is unreachable
(missing file, unreadable, parse error). The YAML is the source of truth
at runtime; these constants exist so the package stays importable when
the YAML is absent (early bootstrap, tests, packaging).

Single source — both ``utils.encoding`` and ``utils.config`` import from
here to avoid drift. Neutral module with no kinsim imports → safe from
circular-import issues during module init.
"""

DEFAULT_KMER_SIZE: int = 11
DEFAULT_UPSTREAM: int = 7
DEFAULT_DOWNSTREAM: int = 3
DEFAULT_REV_METH_OFFSETS: tuple[int, ...] = (-1, 0, 1)

assert DEFAULT_UPSTREAM + 1 + DEFAULT_DOWNSTREAM == DEFAULT_KMER_SIZE, (
    "Hardcoded defaults broken: upstream + 1 + downstream != kmer_size"
)

# BAM kinetic-tag value range (uint8 PacBio convention).
BAM_TAG_MAX: int = 255
