"""Generate IPD/PW kinetics by sampling from the per-kmer table.

Reads an input BAM (typically a stripped real BAM, or PBSIM3 simulated
reads), and for each base position samples (IPD, PW) from the empirical
per-kmer distribution. Writes an output BAM with PacBio ``fi``/``fp``
tags. Output flag is set to 4 (unmapped) to match the contract of
``kinsim generate``.

The kmer at each position is the centred 11-mer extracted from the
read sequence. Boundary positions (the first and last K//2 = 5 bases)
have no full kmer available — their fi/fp values are set to 1 (the
PacBio convention for "no data").

CLI::

    python -m kinsim_baseline generate INPUT_BAM TABLE_NPZ OUTPUT_BAM
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from kinsim.utils.encoding import BASE_MAP, K

from .distribution import KmerDistribution

log = logging.getLogger(__name__)


def encode_sequence_to_kmers(seq: str) -> np.ndarray:
    """Encode a sequence to a sliding array of 22-bit kmer IDs.

    Returns:
        (L - K + 1,) int64 array. ``kmer_ids[i]`` corresponds to
        ``seq[i : i+K]``. The centre base of that kmer is at sequence
        position ``i + K // 2``.
    """
    L = len(seq)
    if L < K:
        return np.array([], dtype=np.int64)
    base_ids = np.fromiter(
        (BASE_MAP.get(c.upper(), 0) for c in seq), dtype=np.int64, count=L
    )
    n_kmers = L - K + 1
    kmer_ids = np.zeros(n_kmers, dtype=np.int64)
    for i in range(K):
        kmer_ids = (kmer_ids << 2) | base_ids[i : i + n_kmers]
    return kmer_ids


def generate_bam(
    input_bam: Path,
    output_bam: Path,
    table: KmerDistribution,
    seed: int = 42,
    progress_every: int = 10_000,
) -> dict:
    """Sample kinetics for every read in ``input_bam`` and write to ``output_bam``."""
    import pysam  # imported here so the package import doesn't require pysam

    rng = np.random.default_rng(seed)
    centre_offset = K // 2  # 5 for K=11

    n_reads = 0
    n_short = 0
    with pysam.AlignmentFile(str(input_bam), "rb", check_sq=False) as bam_in:
        header = bam_in.header.to_dict()
        with pysam.AlignmentFile(str(output_bam), "wb", header=header) as bam_out:
            for read in bam_in:
                seq = read.query_sequence
                if seq is None or len(seq) < K:
                    n_short += 1
                    bam_out.write(read)
                    continue
                L = len(seq)

                # Initialise with PacBio "no data" sentinel (1).
                ipds = np.ones(L, dtype=np.uint8)
                pws = np.ones(L, dtype=np.uint8)

                kmer_ids = encode_sequence_to_kmers(seq)
                if kmer_ids.size > 0:
                    sampled_ipd, sampled_pw = table.sample(kmer_ids, rng=rng)
                    ipds[centre_offset : centre_offset + sampled_ipd.size] = sampled_ipd
                    pws[centre_offset : centre_offset + sampled_pw.size] = sampled_pw

                # Write fi (IPD) and fp (PW) as B:C arrays. Match
                # ``kinsim generate``'s output contract: unmapped flag.
                read.set_tag("fi", ipds.tolist(), value_type="C")
                read.set_tag("fp", pws.tolist(), value_type="C")
                read.flag = 4  # unmapped
                bam_out.write(read)
                n_reads += 1
                if n_reads % progress_every == 0:
                    log.info("processed %d reads", n_reads)

    log.info("Wrote %d reads to %s  (%d skipped: shorter than K=%d)",
             n_reads, output_bam, n_short, K)
    return {"n_reads": n_reads, "n_short": n_short}


def main(argv=None):
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python -m kinsim_baseline generate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_bam", help="Input BAM (typically stripped or simulated reads)")
    p.add_argument("table_npz", help="Per-kmer table from `kinsim_baseline build`")
    p.add_argument("output_bam", help="Output BAM with fi/fp tags")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    log.info("Loading table: %s", args.table_npz)
    table = KmerDistribution.load(args.table_npz)
    log.info("Table coverage: %.2f%% of 4M kmers, %d global-pool entries",
             100.0 * table.coverage(), table.n_global_pool())

    generate_bam(
        Path(args.input_bam),
        Path(args.output_bam),
        table,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
