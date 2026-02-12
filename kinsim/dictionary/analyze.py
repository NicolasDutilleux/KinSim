"""Analyze a trained 11-mer kinetic dictionary (.pkl) for coverage stats."""

import sys
import pickle
import numpy as np

from ..encoding import TOTAL_POSSIBLE_KMERS, METH_IDS


def analyze_dict(pkl_path):
    """Load a .pkl dictionary and print coverage statistics.

    Splits entries by methylation status (unmethylated vs each meth type)
    and reports: entry count, % of 4^11 covered, mean/median/min/max sample count.
    """
    with open(pkl_path, 'rb') as f:
        lookup = pickle.load(f)

    if not lookup:
        print("Dictionary is empty.")
        return

    # Partition entries by methylation id
    groups = {}  # meth_id -> list of sample counts (n)
    for (kmer, meth_id), acc in lookup.items():
        n = acc[0]
        if meth_id not in groups:
            groups[meth_id] = []
        groups[meth_id].append(n)

    # Reverse map: id -> name
    id_to_name = {v: k for k, v in METH_IDS.items()}

    total_entries = len(lookup)
    total_samples = sum(acc[0] for acc in lookup.values())

    print(f"Dictionary: {pkl_path}")
    print(f"Total entries: {total_entries:,}")
    print(f"Total samples: {total_samples:,.0f}")
    print(f"Possible 11-mer contexts: {TOTAL_POSSIBLE_KMERS:,}")
    print("-" * 60)

    for meth_id in sorted(groups.keys()):
        counts = np.array(groups[meth_id])
        name = id_to_name.get(meth_id, f"id={meth_id}")
        label = "Unmethylated" if meth_id == 0 else f"Methylated ({name})"
        n_entries = len(counts)
        coverage = 100.0 * n_entries / TOTAL_POSSIBLE_KMERS

        print(f"\n{label}:")
        print(f"  Unique 11-mers : {n_entries:,} / {TOTAL_POSSIBLE_KMERS:,} ({coverage:.2f}%)")
        print(f"  Sample count   : mean={np.mean(counts):.1f}  median={np.median(counts):.1f}"
              f"  min={np.min(counts):.0f}  max={np.max(counts):.0f}")
        print(f"  Total samples  : {np.sum(counts):,.0f}")

    # Aggregate methylated view
    meth_counts = []
    for mid, cnts in groups.items():
        if mid != 0:
            meth_counts.extend(cnts)

    if meth_counts:
        meth_counts = np.array(meth_counts)
        meth_entries = len(meth_counts)
        print(f"\nAll methylated (combined):")
        print(f"  Unique 11-mers : {meth_entries:,}")
        print(f"  Sample count   : mean={np.mean(meth_counts):.1f}  median={np.median(meth_counts):.1f}"
              f"  min={np.min(meth_counts):.0f}  max={np.max(meth_counts):.0f}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim dictionary analyze",
        description="Analyze a trained 11-mer kinetic dictionary for coverage statistics. "
                    "Reports per-methylation-state: percent of 4^11 possible 11-mers covered, "
                    "mean/median/min/max sample counts.",
    )
    parser.add_argument("pkl", help="Path to the trained dictionary (.pkl file)")
    args = parser.parse_args(argv)
    analyze_dict(args.pkl)


if __name__ == "__main__":
    main()
