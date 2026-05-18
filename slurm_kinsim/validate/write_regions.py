"""Split a BAM's references into roughly equal-sized regions.

Writes ``<task_id>\\t<contig>:<start>-<end>`` lines to ``REGIONS_FILE``.
The launch script's array job awks its task id against column 1 to pick
its region. If the genome is too small for ``N_SHARDS`` regions the
write loop simply emits fewer lines; surplus array tasks see an empty
region and exit cleanly.

Usage:
    python write_regions.py <BAM> <REGIONS_FILE> <N_SHARDS>
"""
from __future__ import annotations

import math
import sys

import pysam


def main() -> None:
    bam_path, regions_path, n_shards = sys.argv[1], sys.argv[2], int(sys.argv[3])
    bam = pysam.AlignmentFile(bam_path, "rb")
    contigs = list(zip(bam.references, bam.lengths))
    per_shard = max(1, math.ceil(sum(L for _, L in contigs) / n_shards))

    shard = 0
    cursor = 0
    with open(regions_path, "w") as out:
        for name, L in contigs:
            pos = 0
            while pos < L:
                take = min(per_shard - cursor, L - pos)
                out.write(f"{shard}\t{name}:{pos+1}-{pos+take}\n")
                pos += take
                cursor += take
                if cursor >= per_shard:
                    shard += 1
                    cursor = 0
    print(f"Wrote {shard + 1} regions to {regions_path}")


if __name__ == "__main__":
    main()
