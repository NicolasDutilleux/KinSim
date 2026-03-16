"""Randomly subsample a dictionary .pkl for train/test splits.

Usage examples:

    # Sample 500 observations per key from a dictionary
    kinsim sample general.pkl sampled.pkl --n-samples 500

    # Sample with a specific seed (reproducible)
    kinsim sample general.pkl sampled.pkl --n-samples 500 --seed 42

    # Create multiple samples and merge them into one dictionary
    kinsim sample msa_dict.pkl   samples/msa_sample.pkl   --n-samples 1000
    kinsim sample strep_dict.pkl samples/strep_sample.pkl --n-samples 1000
    kinsim merge samples/ combined_training.pkl

    # Train/test split: sample test set, remainder is training
    kinsim sample general.pkl test.pkl --n-samples 100 --seed 42
    kinsim sample general.pkl train.pkl --n-samples 5000 --exclude test.pkl
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def sample_pkl(
    input_path: str,
    output_path: str,
    n_samples: int,
    seed: int = 0,
    exclude_path: str | None = None,
) -> dict:
    """Randomly subsample a dictionary .pkl.

    For each (kmer_id, meth_id) key, randomly selects up to n_samples
    observations. Keys with fewer samples than n_samples keep all their data.

    Args:
        input_path:   Input .pkl file.
        output_path:  Output .pkl file.
        n_samples:    Max samples to keep per key.
        seed:         Random seed for reproducibility.
        exclude_path: Optional .pkl whose keys will be excluded from output
                      (useful for creating training set after sampling test set).

    Returns:
        Stats dict: {keys_in, keys_out, samples_in, samples_out}.
    """
    log.info("Loading: %s", input_path)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    # Load exclude keys if provided
    exclude_keys: set = set()
    if exclude_path:
        log.info("Loading exclude keys from: %s", exclude_path)
        with open(exclude_path, "rb") as f:
            exc_data = pickle.load(f)
        exclude_keys = {k for k in exc_data if isinstance(k, tuple)}
        log.info("  Excluding %d keys", len(exclude_keys))

    rng = np.random.default_rng(seed=seed)

    result = {}
    keys_in = keys_out = samples_in = samples_out = 0

    for key, samples in data.items():
        if not isinstance(key, tuple):
            # Preserve metadata
            result[key] = samples
            continue

        keys_in += 1
        samples_in += len(samples)

        if key in exclude_keys:
            continue

        if len(samples) <= n_samples:
            result[key] = samples
            keys_out += 1
            samples_out += len(samples)
        else:
            idx = rng.choice(len(samples), size=n_samples, replace=False)
            result[key] = samples[idx]
            keys_out += 1
            samples_out += n_samples

    # Update metadata
    result["__meta__"] = (
        f"sampled from {input_path} | n_samples={n_samples} seed={seed} | "
        f"{keys_out} keys, {samples_out} samples"
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(result, f)

    stats = {
        "keys_in": keys_in,
        "keys_out": keys_out,
        "samples_in": samples_in,
        "samples_out": samples_out,
    }
    log.info("Sampled: %d/%d keys, %d/%d samples -> %s",
             keys_out, keys_in, samples_out, samples_in, output_path)
    return stats


def main(argv=None):
    import argparse
    from .utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="kinsim sample",
        description=(
            "Randomly subsample a dictionary .pkl.\n\n"
            "Useful for:\n"
            "  - Creating train/test splits\n"
            "  - Downsampling large dictionaries for faster training\n"
            "  - Combining samples from multiple species dictionaries\n\n"
            "Examples:\n"
            "  kinsim sample general.pkl train.pkl --n-samples 5000\n"
            "  kinsim sample general.pkl test.pkl --n-samples 100 --seed 42\n"
            "  kinsim sample general.pkl train.pkl --n-samples 5000 --exclude test.pkl\n\n"
            "To combine multiple sampled dictionaries:\n"
            "  mkdir samples/\n"
            "  kinsim sample msa.pkl   samples/msa.pkl   --n-samples 1000\n"
            "  kinsim sample strep.pkl samples/strep.pkl --n-samples 1000\n"
            "  kinsim merge samples/ combined.pkl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input dictionary .pkl file")
    parser.add_argument("output", help="Output sampled .pkl file")
    parser.add_argument("--n-samples", type=int, required=True,
                        help="Max samples to keep per (kmer, meth) key")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--exclude", default=None, metavar="PKL",
                        help="Exclude keys present in this .pkl "
                             "(e.g. exclude test set keys from training)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    if not Path(args.input).is_file():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    stats = sample_pkl(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        exclude_path=args.exclude,
    )

    print(f"Input:   {stats['keys_in']:,} keys, {stats['samples_in']:,} samples")
    print(f"Output:  {stats['keys_out']:,} keys, {stats['samples_out']:,} samples")
    print(f"Written: {args.output}")
