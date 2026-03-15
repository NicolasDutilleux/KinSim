"""11-mer bit-packing: encode/decode DNA k-mers as 22-bit integers."""

import numpy as np

K = 11
KMER_BITS = 2 * K  # 22
KMER_MASK = (1 << KMER_BITS) - 1

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INT_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
VALID_BASES = set('ACGT')

METH_IDS = {'none': 0, 'm6A': 1, 'm4C': 2, 'm5C': 3}
TOTAL_POSSIBLE_KMERS = 4 ** K  # 4,194,304


def encode_kmer(seq):
    """Encode an 11-mer string to a 22-bit integer."""
    val = 0
    for base in seq:
        val = (val << 2) | BASE_MAP[base]
    return val


def decode_kmer(val):
    """Decode a 22-bit integer back to an 11-mer string."""
    bases = []
    for _ in range(K):
        bases.append(INT_TO_BASE[val & 3])
        val >>= 2
    return ''.join(reversed(bases))


def get_ipd_stats(acc):
    """Extract (mu_ipd, sigma_ipd) from accumulator [n, sum_ipd, sum_ipd2, sum_pw, sum_pw2]."""
    n = acc[0]
    if n < 1:
        return 1.0, 0.1
    mu = acc[1] / n
    var = max(0, (acc[2] / n) - mu ** 2)
    return mu, np.sqrt(var)


def get_pw_stats(acc):
    """Extract (mu_pw, sigma_pw) from accumulator [n, sum_ipd, sum_ipd2, sum_pw, sum_pw2]."""
    n = acc[0]
    if n < 1:
        return 1.0, 0.1
    mu = acc[3] / n
    var = max(0, (acc[4] / n) - mu ** 2)
    return mu, np.sqrt(var)
