"""K-mer bit-packing: encode/decode DNA k-mers as 2k-bit integers.

Default k=11 (22-bit integers, 4^11 = 4,194,304 possible k-mers).
All functions accept an optional ``k`` parameter so the window size
can be changed without touching call sites that use the default.
"""

import numpy as np

# Default k-mer size — change here to shift the whole pipeline default.
K = 11
KMER_BITS = 2 * K  # 22 for K=11
KMER_MASK = (1 << KMER_BITS) - 1  # mask for default K

# Asymmetric kmer/meth-context window around the prediction position.
# Window covers [-KMER_LEFT_PAD, +KMER_RIGHT_PAD] from the prediction position.
# Polymerase has read more bases UPSTREAM than DOWNSTREAM at any moment, and
# all kinetic signatures are downstream of the modification — so to predict
# IPD/PW at position Y we want more upstream context than downstream.
# Inspired by Feng et al. 2013 (kineticsTools/ipdSummary, [-7, +2] for
# unmodified DNA), extended to 11 bases for our k-mer.
KMER_LEFT_PAD = 7  # bases before prediction position
KMER_RIGHT_PAD = 3  # bases after prediction position
KMER_PRED_IDX = KMER_LEFT_PAD  # = 7
assert KMER_LEFT_PAD + 1 + KMER_RIGHT_PAD == K, "KMER_PAD must sum to K"

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
    except Exception:
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
