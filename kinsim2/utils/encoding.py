"""K-mer bit-packing: encode/decode DNA k-mers as 2k-bit integers.

Module-level ``K``, ``KMER_PRED_IDX`` etc. are read from
``kinsim_config.yaml::extraction`` at import time, falling back to the
historical K=11 / [-7, +3] window if the YAML is unreachable. Consumers
that need to mix K across calls should pass an explicit ``k`` arg to
``encode_kmer`` / ``decode_kmer`` / ``kmer_mask`` — all three accept one.

For shard-level work, prefer ``utils.sample_layout.get_sample_layout``
which reads ExtractionParams from the shard's __meta__.
"""

import numpy as np


def _load_kmer_geometry() -> tuple[int, int, int]:
    """Best-effort read of K / upstream / downstream from the YAML.

    Falls back to the hardcoded ``_defaults`` if the YAML can't be loaded
    (config.py imports only stdlib, so this is safe at module-init time
    — no circular).
    """
    from ._defaults import DEFAULT_DOWNSTREAM, DEFAULT_KMER_SIZE, DEFAULT_UPSTREAM
    try:
        from .config import load_kinsim_config
        ext = (load_kinsim_config().get("extraction") or {})
        return (
            int(ext.get("kmer_size", DEFAULT_KMER_SIZE)),
            int(ext.get("upstream", DEFAULT_UPSTREAM)),
            int(ext.get("downstream", DEFAULT_DOWNSTREAM)),
        )
    except (ImportError, OSError, ValueError, TypeError):
        return DEFAULT_KMER_SIZE, DEFAULT_UPSTREAM, DEFAULT_DOWNSTREAM


K, KMER_LEFT_PAD, KMER_RIGHT_PAD = _load_kmer_geometry()
KMER_BITS = 2 * K
KMER_MASK = (1 << KMER_BITS) - 1
KMER_PRED_IDX = KMER_LEFT_PAD
assert KMER_LEFT_PAD + 1 + KMER_RIGHT_PAD == K, (
    f"KMER_PAD must sum to K: {KMER_LEFT_PAD} + 1 + {KMER_RIGHT_PAD} != {K}"
)

BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
VALID_BASES = set("ACGT")

# Default methylation type → integer mapping. The three standard PacBio
# modifications (m6A=1, m4C=2, m5C=3) are listed here so that pickled
# datasets remain readable even when kinsim_config.yaml is unavailable.
# At runtime, `get_meth_ids()` extends this with any additional types
# declared in the YAML (`kinetic_signatures.<type>`), with stable IDs
# assigned in YAML-declaration order.
#
# Stability of IDs across runs matters because IDs end up encoded in the
# stored pkls (col COL_PARENT_METH); changing a type's ID would silently
# misinterpret older shards. We therefore keep the three standard IDs
# pinned by name and only auto-assign IDs for types not in this dict.
METH_IDS = {"none": 0, "m6A": 1, "m4C": 2, "m5C": 3}


def get_meth_ids() -> dict:
    """Return ``{mod_type: int_id}`` from the YAML, with stable IDs.

    Algorithm:
      1. Always include ``'none' = 0``.
      2. For each pinned-ID type in :data:`METH_IDS` (m6A=1, m4C=2,
         m5C=3) that is also declared in the YAML, keep its pinned ID.
      3. Walk the YAML's ``kinetic_signatures`` keys in declaration order
         and assign the next free integer to each as-yet-unassigned type.

    This guarantees:
      - Older pkls (which encoded m6A=1, m4C=2, m5C=3) remain decodable.
      - Adding a new type ``m4mC`` to the YAML is a config-only change;
        it will get the next free ID (typically 4) automatically and
        every consumer (extract, refine, train, analyze) picks it up
        with no code change.

    Falls back to the bare :data:`METH_IDS` if the YAML can't be loaded
    (e.g. during early imports before the config file is on disk) — this
    is intentional: the encoding module must be importable in any state.
    """
    try:
        # Lazy import to avoid circular dependency: utils.config imports motifs
        # which imports encoding.
        from .config import load_kinsim_config

        cfg = load_kinsim_config()
    except (ImportError, OSError, ValueError) as exc:
        # ImportError: lazy import or PyYAML missing.
        # OSError:     YAML file unreadable / missing.
        # ValueError:  YAML parse error (yaml.YAMLError subclasses it).
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "get_meth_ids: could not load kinsim_config.yaml (%s) — "
            "falling back to the built-in METH_IDS=%s",
            exc,
            METH_IDS,
        )
        return dict(METH_IDS)

    user_types = list(cfg.get("kinetic_signatures") or {})
    out: dict[str, int] = {"none": 0}
    next_id = 1
    # Pinned IDs first (so they keep their canonical integer regardless
    # of where they sit in the YAML).
    for pinned, pinned_id in METH_IDS.items():
        if pinned == "none":
            continue
        if pinned in user_types:
            out[pinned] = pinned_id
            next_id = max(next_id, pinned_id + 1)
    # Then assign sequential IDs to any additional user-declared types.
    for name in user_types:
        if name in out:
            continue
        out[name] = next_id
        next_id += 1
    return out


TOTAL_POSSIBLE_KMERS = 4**K  # 4,194,304 for K=11


def kmer_mask(k: int = K) -> int:
    """Return the bit-mask for a k-mer of size k (= (1 << 2k) - 1)."""
    return (1 << (2 * k)) - 1


def encode_kmer(seq: str, k: int = K) -> int:
    """Encode a k-mer string to a 2k-bit integer.

    Args:
        seq: DNA string of length k (ACGT only).
        k:   K-mer size (default K=11).

    Returns:
        Integer in [0, 4^k).
    """
    val = 0
    for base in seq:
        val = (val << 2) | BASE_MAP[base]
    return val & kmer_mask(k)


def decode_kmer(val: int, k: int = K) -> str:
    """Decode a 2k-bit integer back to a k-mer string.

    Args:
        val: Integer in [0, 4^k).
        k:   K-mer size (default K=11).

    Returns:
        DNA string of length k.
    """
    bases = []
    for _ in range(k):
        bases.append(INT_TO_BASE[val & 3])
        val >>= 2
    return "".join(reversed(bases))


def get_ipd_stats(acc):
    """Extract (mu_ipd, sigma_ipd) from accumulator [n, sum_ipd, sum_ipd2, sum_pw, sum_pw2]."""
    n = acc[0]
    if n < 1:
        return 1.0, 0.1
    mu = acc[1] / n
    var = max(0, (acc[2] / n) - mu**2)
    return mu, np.sqrt(var)


def get_pw_stats(acc):
    """Extract (mu_pw, sigma_pw) from accumulator [n, sum_ipd, sum_ipd2, sum_pw, sum_pw2]."""
    n = acc[0]
    if n < 1:
        return 1.0, 0.1
    mu = acc[3] / n
    var = max(0, (acc[4] / n) - mu**2)
    return mu, np.sqrt(var)
