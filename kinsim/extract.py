"""Extract raw IPD/PW training samples from BAM files.

This is the data-preparation pipeline for KinSim MLP training.
It has no dependency on the model itself.

Data format
-----------
Each shard is a pickle file containing:

    dict[(kmer_id: int, meth_id: int)] -> np.ndarray(N, 14)

where columns are [IPD, PW, fraction, mc_0, mc_1, ..., mc_10] as raw float32.
IPD and PW are read from the fi/fp BAM tags (uint8 [0, 255]).
Column 2 is the stoichiometric methylation fraction.
Columns 3–13 are the methylation IDs (0=none,1=m6A,2=m4C,3=m5C) for each
of the 11 positions in the k-mer window, with the active site at mc_5
(index K//2 = 5).

Backward compatibility: older shards may have only 2 columns [IPD, PW] or
3 columns [IPD, PW, fraction].  Dataset classes handle all three formats
by zero-padding the missing meth-context columns.

A special metadata key ``"__meta__"`` (string, not a tuple) may be present
in any shard or master .pkl.  It holds provenance information (version,
motifs, timestamp) and is automatically skipped by dataset classes.

Why raw (not log-transformed)?
    The extract/merge pipeline stores raw values so that:
      - Shards can be inspected and plotted without model knowledge
      - Different models can apply their own transforms at load time
      - MLPSignalDataset (data/dataset.py) applies log_transform once

CLI — single-BAM mode (original interface, unchanged):
    kinsim extract reads.bam "m6A,GATC,1" shard.pkl
    kinsim merge   shards/   master_data.pkl

CLI — manifest mode (new, recommended for SLURM array jobs):
    kinsim extract --manifest manifest.csv --task 3 --output-dir shards/
    kinsim merge   shards/  master_data.pkl

Manifest CSV format (see kinsim/config.py):
    sample_id,bam_path,motifs
    strain1,/data/bam1.bam,"m6A,GATC,1"
    strain2,/data/bam2.bam,/data/motifs/strain2.csv
"""

import datetime
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysam

from .utils.encoding import BASE_MAP, K, KMER_MASK, kmer_mask
from .utils.motifs import load_motif_string, parse_motifs, reverse_complement, scan_sequence

try:
    from . import __version__ as _KINSIM_VERSION
except (ImportError, AttributeError):
    try:
        from .__main__ import __version__ as _KINSIM_VERSION
    except ImportError:
        _KINSIM_VERSION = "unknown"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fail-fast BAM validation
# ---------------------------------------------------------------------------

def validate_bam_kinetics(bam_path: str, n_check: int = 10) -> None:
    """Raise ValueError if the BAM has no fi/fp kinetic tags.

    Reads up to *n_check* reads from the BAM and checks for the ``fi`` tag.
    Exits early (fast path) as soon as one read with ``fi`` is found.

    Args:
        bam_path: Path to the BAM file.
        n_check:  Maximum reads to scan before giving up.

    Raises:
        FileNotFoundError: If the BAM does not exist.
        ValueError: If no reads with ``fi`` kinetic tags are found.
    """
    if not os.path.exists(bam_path):
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    log.debug("Validating kinetic tags in: %s", bam_path)
    reads_seen = 0
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if not read.query_sequence:
                continue
            if read.has_tag("fi"):
                log.debug("fi tag confirmed in read %s", read.query_name)
                return   # all good
            reads_seen += 1
            if reads_seen >= n_check:
                break

    raise ValueError(
        f"BAM file has no 'fi' kinetic tags (checked {reads_seen} reads): {bam_path}\n"
        "Kinetic tags are written by the PacBio instrument during primary analysis.\n"
        "Ensure the BAM was produced with --emit-kinetics (or equivalent).\n"
        "Check: samtools view -H '"  + bam_path + "' | grep -i kinetics"
    )


# ---------------------------------------------------------------------------
# Fraction lookup from motif string
# ---------------------------------------------------------------------------

def _build_fraction_lookup(motif_string: str) -> dict[int, float]:
    """Parse the motif string to build a meth_id → fraction lookup.

    The motif string format is "m6A,GATC,1[,nDetected[,fraction]];..."
    When PacBio motifs.csv is the source, parse_motifs_csv preserves the
    fraction as the 5th field.  For plain motif strings without a fraction
    field, defaults to 1.0 (fully methylated).

    If multiple motifs share the same meth_id (e.g., two m6A motifs with
    different fractions), the last one wins.  This is acceptable because
    the kmer embedding already distinguishes different motif contexts.

    Returns:
        dict mapping meth_id → float fraction.  Always includes {0: 0.0}.
    """
    from .utils.encoding import METH_IDS

    fracs: dict[int, float] = {0: 0.0}
    if not motif_string:
        return fracs
    for entry in motif_string.split(';'):
        if not entry or ',' not in entry:
            continue
        parts = entry.split(',')
        if len(parts) < 3:
            continue
        m_id = METH_IDS.get(parts[0], 0)
        frac = float(parts[4]) if len(parts) >= 5 else 1.0
        fracs[m_id] = frac
    return fracs


# ---------------------------------------------------------------------------
# IPD-based binarization helpers
# ---------------------------------------------------------------------------


def _binarize_by_ipd(result: dict) -> dict:
    """Binarize methylated keys using 2D GMM clustering (IPD + PW).

    For each (kmer_id, meth_id > 0) key with enough samples:
      1. Fit a 2-component Gaussian Mixture Model on (IPD, PW).
      2. The component with the higher IPD centroid = methylated cluster.
      3. Each sample is assigned to the cluster with highest posterior
         probability.  Only samples assigned to the methylated cluster
         are kept as methylated; the rest are moved to unmethylated.
      4. Validation: the methylated centroid's IPD must be > 1.5× the
         unmethylated centroid's IPD.  If not, no clear bimodality exists
         and all samples are moved to unmethylated.

    Advantages:
      - 2D: uses both IPD and PW jointly (important for m4C)
      - Per-kmer: each sequence context gets its own GMM fit
      - Data-driven: all parameters learned from the data, no hardcoded
        thresholds or percentiles
      - Handles asymmetric clusters (different sizes, different variances)
      - Probabilistic assignment (not a hard threshold)

    For keys with < 20 samples: kept as methylated (too few for GMM).
    For keys where GMM doesn't converge: kept as methylated.
    """
    from sklearn.mixture import GaussianMixture

    MIN_SAMPLES_FOR_GMM = 20
    MIN_METH_SURVIVORS = 3
    MIN_CENTROID_RATIO = 1.5  # only hard constraint: bimodality check

    new_result: dict = {}
    none_extras: dict = {}
    n_split      = 0
    n_kept_all   = 0
    n_all_false  = 0
    n_too_few    = 0
    n_no_bimodal = 0
    n_gmm_fail   = 0

    for key, arr in result.items():
        if not isinstance(key, tuple):
            new_result[key] = arr
            continue
        kmer_id, meth_id = key
        if meth_id == 0:
            new_result[key] = arr
            continue

        n_samples = len(arr)

        # Too few samples for GMM — keep as methylated
        if n_samples < MIN_SAMPLES_FOR_GMM:
            high_arr = arr.copy()
            high_arr[:, 2] = 1.0
            new_result[(kmer_id, meth_id)] = high_arr
            n_too_few += 1
            continue

        # Fit 2-component GMM on (IPD, PW)
        ipd_pw = arr[:, :2].astype(np.float64)
        try:
            gmm = GaussianMixture(
                n_components=2,
                covariance_type='full',
                n_init=3,
                random_state=42,
                max_iter=100,
            )
            gmm.fit(ipd_pw)
        except Exception:
            # GMM failed to converge — keep as methylated
            high_arr = arr.copy()
            high_arr[:, 2] = 1.0
            new_result[(kmer_id, meth_id)] = high_arr
            n_gmm_fail += 1
            continue

        # Identify which component is "methylated" (higher IPD centroid)
        centroids_ipd = gmm.means_[:, 0]  # IPD of each centroid
        meth_comp = int(np.argmax(centroids_ipd))
        none_comp = 1 - meth_comp

        # Bimodality check: methylated centroid must have meaningfully
        # higher IPD than unmethylated centroid
        ratio = centroids_ipd[meth_comp] / max(centroids_ipd[none_comp], 1e-9)
        if ratio < MIN_CENTROID_RATIO:
            # No clear bimodality — likely all unmethylated
            low_arr = arr.copy()
            low_arr[:, 2] = 0.0
            if low_arr.shape[1] >= 14:
                low_arr[:, 3 + K // 2] = 0
            none_key = (kmer_id, 0)
            none_extras.setdefault(none_key, []).append(low_arr)
            n_no_bimodal += 1
            continue

        # Assign each sample to the most probable component
        labels = gmm.predict(ipd_pw)
        high_mask = labels == meth_comp
        n_high = int(high_mask.sum())

        # All methylated
        if n_high == n_samples:
            high_arr = arr.copy()
            high_arr[:, 2] = 1.0
            new_result[(kmer_id, meth_id)] = high_arr
            n_kept_all += 1
            continue

        # Too few survivors
        if n_high < MIN_METH_SURVIVORS:
            low_arr = arr.copy()
            low_arr[:, 2] = 0.0
            if low_arr.shape[1] >= 14:
                low_arr[:, 3 + K // 2] = 0
            none_key = (kmer_id, 0)
            none_extras.setdefault(none_key, []).append(low_arr)
            n_all_false += 1
            continue

        # Split
        high_arr = arr[high_mask].copy()
        high_arr[:, 2] = 1.0
        new_result[(kmer_id, meth_id)] = high_arr

        low_arr = arr[~high_mask].copy()
        low_arr[:, 2] = 0.0
        if low_arr.shape[1] >= 14:
            low_arr[:, 3 + K // 2] = 0
        none_key = (kmer_id, 0)
        none_extras.setdefault(none_key, []).append(low_arr)

        log.debug(
            "binarize kmer=%d meth=%d  centroids_ipd=(%.1f, %.1f) ratio=%.2f  "
            "n=%d → %d meth  %d unmeth",
            kmer_id, meth_id,
            centroids_ipd[none_comp], centroids_ipd[meth_comp], ratio,
            n_samples, n_high, int((~high_mask).sum()),
        )
        n_split += 1

    # Merge reclassified samples into none keys
    for none_key, arrays in none_extras.items():
        extra = np.concatenate(arrays, axis=0)
        if none_key in new_result:
            new_result[none_key] = np.concatenate([new_result[none_key], extra], axis=0)
        else:
            new_result[none_key] = extra

    log.info(
        "Binarization (2D GMM, per-kmer): "
        "%d keys split, %d all-methylated, %d all-false-positive, "
        "%d no-bimodality, %d too-few (<20), %d GMM-failed",
        n_split, n_kept_all, n_all_false, n_no_bimodal, n_too_few, n_gmm_fail,
    )

    # --- Global post-GMM validation ---
    # Compute the none IPD distribution and reject meth keys whose mean IPD
    # falls within the unmethylated range.  This catches:
    #   - too-few keys that were kept without GMM verification
    #   - GMM splits where the "high" cluster is still close to none
    # Threshold: none_mean + 2*none_std  (data-driven, no hardcoded value)
    none_ipds = []
    for key, arr in new_result.items():
        if isinstance(key, tuple) and key[1] == 0:
            none_ipds.append(arr[:, 0])
    if none_ipds:
        all_none_ipd = np.concatenate(none_ipds)
        none_mean = float(np.mean(all_none_ipd))
        none_std  = float(np.std(all_none_ipd))
        ipd_floor = none_mean + 2.0 * none_std
        log.info(
            "Global validation: none IPD mean=%.1f std=%.1f → "
            "meth keys must have mean IPD > %.1f",
            none_mean, none_std, ipd_floor,
        )

        n_global_reject = 0
        meth_keys_to_check = [
            k for k in new_result
            if isinstance(k, tuple) and k[1] > 0
        ]
        for key in meth_keys_to_check:
            arr = new_result[key]
            key_mean_ipd = float(np.mean(arr[:, 0]))
            if key_mean_ipd < ipd_floor:
                # Reclassify to none
                low_arr = arr.copy()
                low_arr[:, 2] = 0.0
                if low_arr.shape[1] >= 14:
                    low_arr[:, 3 + K // 2] = 0
                none_key = (key[0], 0)
                if none_key in new_result:
                    new_result[none_key] = np.concatenate(
                        [new_result[none_key], low_arr], axis=0,
                    )
                else:
                    new_result[none_key] = low_arr
                del new_result[key]
                n_global_reject += 1

        log.info("Global validation: %d meth keys rejected (mean IPD < %.1f)",
                 n_global_reject, ipd_floor)

    return new_result


# ---------------------------------------------------------------------------
# Extract: raw samples from one BAM file
# ---------------------------------------------------------------------------

def extract_samples_from_bam(
    bam_path: str,
    motif_string: str,
    max_samples_per_key: int = 10_000,
    revcomp: bool = True,
    use_reverse_strand: bool = True,
    max_reads: int = 0,
    kmer_size: int = K,
    unmeth_subsample_rate: float = 0.05,
) -> dict:
    """Extract raw (IPD, PW) pairs from a BAM file for each k-mer context.

    For each read: extract sequence + fi/fp kinetic tags, scan methylation
    motifs, then slide a kmer_size-mer window collecting raw signal values.

    When ``use_reverse_strand=True`` (default) and the BAM contains ``ri``/``rp``
    complementary-strand kinetic tags, a second extraction pass processes the
    reverse strand.  For position *i* in the read, the reverse-strand kinetic
    signal ``ri[i]`` was measured as the polymerase traversed the complementary
    strand in its 5'→3' direction.  The correct sequence context for that signal
    is ``RC(seq[i-mid:i+mid+1])`` — not the forward k-mer.  Using RC kmers with
    the complementary-strand IPD/PW values effectively doubles the training set
    and makes the model strand-invariant.

    Reservoir sampling keeps memory bounded: once a (kmer, meth_id) key
    reaches max_samples_per_key, new samples randomly replace existing ones
    with probability max_samples_per_key / n_seen, giving an unbiased sample.

    Args:
        bam_path:             Path to BAM file with fi/fp kinetic tags.
        motif_string:         Semicolon-delimited motif string (e.g. "m6A,GATC,1").
        max_samples_per_key:  Maximum samples stored per (kmer, meth_id) key.
        revcomp:              Include reverse complement motif patterns (default True).
        use_reverse_strand:   Also extract ri/rp complementary-strand kinetics
                              using RC(k-mer) as the key.  Silently skipped for
                              reads or BAMs that lack ri/rp tags.
        max_reads:            Stop after this many reads (0 = no limit).
                              For smoke-testing only — reservoir sampling is biased
                              when the BAM is not fully read.
        kmer_size:            K-mer window size (default K=11). Must be odd.
        unmeth_subsample_rate: Fraction of unmethylated (meth_id=0) positions to
                              keep (default 0.05 = 5%). Methylated positions are
                              always kept. Controls the unmeth/meth balance in the
                              raw shard and prevents the dict from being dominated
                              by the ~4M possible unmethylated k-mers.

    Returns:
        dict with:
          - tuple keys ``(kmer_id, meth_id)`` → ``np.ndarray(N, 14)``
            columns: [IPD, PW, fraction, mc_0..mc_10]
          - ``"__meta__"``                     → dict with provenance metadata
    """
    validate_bam_kinetics(bam_path)

    _mask  = kmer_mask(kmer_size)
    mid    = kmer_size // 2
    motifs = parse_motifs(motif_string, revcomp=revcomp)
    frac_lookup = _build_fraction_lookup(motif_string)

    samples: dict = defaultdict(list)
    counts:  dict = defaultdict(int)   # total observations seen per key
    n_reads_processed    = 0
    n_reads_with_reverse = 0

    log.info("Extracting from: %s", bam_path)
    log.info("Motifs: %s  |  reverse_strand=%s", motif_string, use_reverse_strand)

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if max_reads > 0 and n_reads_processed >= max_reads:
                log.info("--max-reads %d reached — stopping early (smoke test only)", max_reads)
                break
            seq = read.query_sequence
            if not (seq and len(seq) >= kmer_size and read.has_tag("fi")):
                continue

            ipds    = read.get_tag("fi")
            pws     = read.get_tag("fp")
            min_len = min(len(seq), len(ipds), len(pws))

            # Per-read regex scan for methylation positions (forward strand).
            meth_status = scan_sequence(seq[:min_len], motifs)

            # --- Forward strand: slide kmer_size window, collect fi/fp ---
            current_kmer = 0
            for i in range(min_len):
                base_val     = BASE_MAP.get(seq[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & _mask

                if i >= kmer_size - 1:
                    center  = i - mid
                    meth_id = int(meth_status[center])
                    # Subsample unmethylated positions to control dict size
                    if meth_id == 0 and np.random.random() >= unmeth_subsample_rate:
                        continue
                    key     = (current_kmer, meth_id)
                    ipd_val = float(ipds[center])
                    pw_val  = float(pws[center])
                    frac    = frac_lookup.get(meth_id, 0.0)

                    # 11-position methylation context for the current window
                    mc = meth_status[i - (kmer_size - 1) : i + 1].tolist()

                    counts[key] += 1
                    n = counts[key]
                    if n <= max_samples_per_key:
                        samples[key].append([ipd_val, pw_val, frac] + mc)
                    else:
                        # Reservoir sampling: replace a random existing entry
                        j = np.random.randint(0, n)
                        if j < max_samples_per_key:
                            samples[key][j] = [ipd_val, pw_val, frac] + mc

            # --- Reverse strand: slide RC 11-mer window, collect ri/rp ---
            #
            # ri[i] is the IPD of the polymerase reading the complementary
            # strand at the position paired with seq[i], moving 5'→3' on the
            # complement.  The local 11-mer it sees is RC(seq[i-5:i+6]).
            #
            # Implementation: slide an 11-mer window through rc_seq (the reverse
            # complement of the read).  At window position j in rc_seq, the
            # window centre in rc_seq is rc_center = j - mid, which corresponds
            # to forward position fwd_center = min_rev_len - 1 - rc_center.
            # ri_tags[fwd_center] gives the complementary-strand IPD at that site.
            if use_reverse_strand and read.has_tag("ri") and read.has_tag("rp"):
                ri_tags = read.get_tag("ri")
                rp_tags = read.get_tag("rp")
                min_rev_len = min(min_len, len(ri_tags), len(rp_tags))

                if min_rev_len >= kmer_size:
                    n_reads_with_reverse += 1
                    rc_seq          = reverse_complement(seq[:min_rev_len])
                    rev_meth_status = scan_sequence(rc_seq, motifs)

                    rc_kmer = 0
                    for j in range(min_rev_len):
                        rc_base = BASE_MAP.get(rc_seq[j], 0)
                        rc_kmer = ((rc_kmer << 2) | rc_base) & _mask

                        if j >= kmer_size - 1:
                            rc_center  = j - mid
                            fwd_center = min_rev_len - 1 - rc_center

                            rc_meth_id = int(rev_meth_status[rc_center])
                            if rc_meth_id == 0 and np.random.random() >= unmeth_subsample_rate:
                                continue
                            rc_key = (rc_kmer, rc_meth_id)
                            ri_val = float(ri_tags[fwd_center])
                            rp_val = float(rp_tags[fwd_center])
                            frac   = frac_lookup.get(rc_meth_id, 0.0)

                            # 11-position meth context for the RC window
                            rev_mc = rev_meth_status[j - (kmer_size - 1) : j + 1].tolist()

                            counts[rc_key] += 1
                            n = counts[rc_key]
                            if n <= max_samples_per_key:
                                samples[rc_key].append([ri_val, rp_val, frac] + rev_mc)
                            else:
                                j2 = np.random.randint(0, n)
                                if j2 < max_samples_per_key:
                                    samples[rc_key][j2] = [ri_val, rp_val, frac] + rev_mc

            n_reads_processed += 1

    if use_reverse_strand and n_reads_with_reverse == 0:
        log.warning(
            "No ri/rp tags found in %s — reverse strand extraction skipped. "
            "BAM may not contain complementary-strand kinetics.",
            bam_path,
        )

    n_keys    = len(samples)
    n_samples = sum(len(v) for v in samples.values())
    log.info(
        "Done: %d reads (%d with reverse strand) → %d unique (kmer, meth) keys, "
        "%d total samples",
        n_reads_processed, n_reads_with_reverse, n_keys, n_samples,
    )

    result = {key: np.array(vals, dtype=np.float32) for key, vals in samples.items()}

    # Binarize methylated keys: 2-component GMM on (IPD, PW) separates
    # truly methylated reads from unmethylated reads at motif sites.
    result = _binarize_by_ipd(result)
    b_keys    = sum(1 for k in result if isinstance(k, tuple))
    b_samples = sum(len(v) for k, v in result.items() if isinstance(k, tuple))
    log.info("After IPD binarization: %d unique keys, %d total samples", b_keys, b_samples)

    # Attach provenance metadata so shards can be inspected and traced back.
    result["__meta__"] = {
        "kinsim_version":          _KINSIM_VERSION,
        "source_bam":              str(bam_path),
        "motifs":                  motif_string,
        "kmer_size":               kmer_size,
        "unmeth_subsample_rate":   unmeth_subsample_rate,
        "use_reverse_strand":      use_reverse_strand,
        "max_samples_per_key":     max_samples_per_key,
        "n_reads_processed":       n_reads_processed,
        "n_reads_with_reverse":    n_reads_with_reverse,
        "n_unique_keys":           n_keys,
        "n_total_samples":         n_samples,
        "created":                 datetime.datetime.now().isoformat(timespec="seconds"),
    }

    return result


# ---------------------------------------------------------------------------
# Merge: combine shards from multiple BAMs
# ---------------------------------------------------------------------------

def merge_shards(
    input_dir: str,
    output_file: str,
    max_samples_per_key: int = 50_000,
    glob_pattern: str = "auto",
) -> None:
    """Merge multiple shard pickle files into one master training set.

    Looks for shard files in input_dir using the following precedence:
      1. ``*_shard.pkl`` (produced by ``kinsim extract --manifest``)
      2. ``*_cgan.pkl``  (legacy naming, kept for backward compat)

    Override with ``glob_pattern`` to use a custom pattern.

    After concatenation, keys exceeding max_samples_per_key are randomly
    subsampled to keep the master file manageable.

    The ``"__meta__"`` key (provenance) is merged across all shards and stored
    in the output.

    Args:
        input_dir:           Directory containing shard .pkl files.
        output_file:         Path for the merged output .pkl file.
        max_samples_per_key: Maximum samples to keep per (kmer, meth_id).
        glob_pattern:        Glob pattern for shard files; "auto" tries
                             ``*_shard.pkl`` then ``*_cgan.pkl``.
    """
    import glob as _glob

    if glob_pattern == "auto":
        files = _glob.glob(os.path.join(input_dir, "*_shard.pkl"))
        if not files:
            files = _glob.glob(os.path.join(input_dir, "*_cgan.pkl"))
        if not files:
            files = _glob.glob(os.path.join(input_dir, "*.pkl"))
            # Exclude the output file itself to avoid self-merging
            files = [f for f in files if os.path.abspath(f) != os.path.abspath(output_file)]
    else:
        files = _glob.glob(os.path.join(input_dir, glob_pattern))

    if not files:
        log.error("No shard .pkl files found in %s", input_dir)
        sys.exit(1)

    files = sorted(files)
    log.info("Merging %d shards from: %s", len(files), input_dir)

    master: dict = defaultdict(list)
    shard_metas: list = []

    for f_path in files:
        log.info("  Loading shard: %s", os.path.basename(f_path))
        with open(f_path, "rb") as f:
            shard = pickle.load(f)

        # Collect and skip the metadata key
        if "__meta__" in shard:
            shard_metas.append(shard.pop("__meta__"))

        for key, arr in shard.items():
            if not isinstance(key, tuple):
                continue   # skip any other non-data keys
            master[key].append(arr)

    result = {}
    n_subsampled = 0
    for key, arrays in master.items():
        combined = np.concatenate(arrays, axis=0)
        if len(combined) > max_samples_per_key:
            idx      = np.random.choice(len(combined), max_samples_per_key, replace=False)
            combined = combined[idx]
            n_subsampled += 1
        result[key] = combined

    # Merged metadata
    result["__meta__"] = {
        "kinsim_version":     _KINSIM_VERSION,
        "merged_from":        [m.get("source_bam", "?") for m in shard_metas],
        "n_shards":           len(files),
        "max_samples_per_key": max_samples_per_key,
        "created":            datetime.datetime.now().isoformat(timespec="seconds"),
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(result, f)

    total_keys    = len(result) - 1   # exclude __meta__
    total_samples = sum(len(v) for k, v in result.items() if isinstance(k, tuple))
    log.info("Master dataset saved: %s", output_file)
    log.info(
        "  %d unique contexts, %d total samples (%d keys subsampled to cap=%d)",
        total_keys, total_samples, n_subsampled, max_samples_per_key,
    )


# ---------------------------------------------------------------------------
# Manifest-mode extraction helper
# ---------------------------------------------------------------------------

def extract_from_manifest_task(
    manifest_path: str,
    task_index: int,
    output_dir: str,
    max_samples_per_key: int = 10_000,
    revcomp: bool = True,
    use_reverse_strand: bool = True,
    max_reads: int = 0,
    kmer_size: int = K,
    unmeth_subsample_rate: float = 0.05,
) -> None:
    """Extract one BAM from a manifest CSV (for SLURM array jobs).

    Reads the manifest at ``manifest_path``, picks the row at ``task_index``
    (1-based, matching SLURM_ARRAY_TASK_ID), runs extraction, and writes the
    shard to ``output_dir/<sample_id>_shard.pkl``.

    Args:
        manifest_path:        Path to the manifest CSV.
        task_index:           1-based row index (SLURM_ARRAY_TASK_ID).
        output_dir:           Directory for the output shard .pkl.
        max_samples_per_key:  Reservoir cap per (kmer, meth_id) key.
        revcomp:              Scan reverse complement strand for motifs.
        use_reverse_strand:   Extract ri/rp complementary-strand kinetics.
        max_reads:            Stop after N reads (0 = no limit, smoke test only).
    """
    from .utils.config import load_manifest
    from .utils.motifs import load_motif_string as _load_motif_string

    entries = load_manifest(manifest_path)

    if task_index < 1 or task_index > len(entries):
        log.error(
            "Task index %d is out of range (manifest has %d entries, 1-indexed).",
            task_index, len(entries),
        )
        sys.exit(1)

    entry = entries[task_index - 1]
    log.info("Task %d/%d: %s", task_index, len(entries), entry.sample_id)
    log.info("  BAM:    %s", entry.bam_path)
    log.info("  Motifs: %s", entry.motifs)

    # Auto-detect motif source (CSV, REBASE, or inline string)
    motif_string = _load_motif_string(entry.motifs)
    if not motif_string:
        log.warning("No motifs resolved for sample '%s' -- SKIPPING.", entry.sample_id)
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_pkl = os.path.join(output_dir, f"{entry.sample_id}_shard.pkl")
    log.info("  Output: %s", output_pkl)

    result = extract_samples_from_bam(
        entry.bam_path, motif_string,
        max_samples_per_key=max_samples_per_key,
        revcomp=revcomp,
        use_reverse_strand=use_reverse_strand,
        max_reads=max_reads,
        kmer_size=kmer_size,
        unmeth_subsample_rate=unmeth_subsample_rate,
    )

    with open(output_pkl, "wb") as f:
        pickle.dump(result, f)

    meta = result.get("__meta__", {})
    log.info(
        "Shard saved: %s (%d keys, %d samples)",
        output_pkl,
        meta.get("n_unique_keys", "?"),
        meta.get("n_total_samples", "?"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse
    from .utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="kinsim data",
        description=(
            "Extract raw (IPD, PW) training samples from BAM files, or merge\n"
            "multiple shards into a master training set.\n\n"
            "The output is consumed by:\n"
            "  kinsim train --model mlp  master_data.pkl  checkpoints_mlp/\n\n"
            "Single-BAM extract (simple/testing):\n"
            "  kinsim extract reads.bam \"m6A,GATC,1\" shard.pkl\n\n"
            "Manifest-based extract (recommended for SLURM array jobs):\n"
            "  kinsim extract --manifest manifest.csv --task $SLURM_ARRAY_TASK_ID \\\n"
            "                 --output-dir shards/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- extract subcommand --
    p_extract = sub.add_parser(
        "extract",
        help="Extract raw (IPD, PW) samples from a BAM file (or manifest)",
        description=(
            "Collect individual IPD/PW observations per (11-mer, methylation_state).\n\n"
            "Single-BAM mode:  kinsim extract reads.bam MOTIFS shard.pkl\n"
            "Manifest mode:    kinsim extract --manifest manifest.csv --task N "
            "--output-dir shards/"
        ),
    )
    # Single-BAM positional args (optional — not required when --manifest is used)
    p_extract.add_argument("bam",    nargs="?", default=None,
                           help="Input BAM file with fi/fp kinetic tags")
    p_extract.add_argument("motifs", nargs="?", default=None,
                           help="Motif source: KinSim string ('m6A,GATC,1'), "
                                "PacBio motifs.csv, or REBASE file (auto-detected)")
    p_extract.add_argument("output", nargs="?", default=None,
                           help="Output .pkl shard file (single-BAM mode)")

    # Manifest mode args
    p_extract.add_argument("--manifest",   default=None,
                           help="Manifest CSV with columns: sample_id, bam_path, motifs")
    p_extract.add_argument("--task",       type=int, default=None,
                           help="1-based row index from the manifest (= SLURM_ARRAY_TASK_ID)")
    p_extract.add_argument("--output-dir", default=None,
                           help="Output directory for shard .pkl files (manifest mode)")

    # Common options
    p_extract.add_argument("--max-samples", type=int, default=20_000,
                           help="Max samples per (kmer, meth_id) via reservoir "
                                "sampling (default: 20000)")
    p_extract.add_argument("--no-revcomp", action="store_true",
                           help="Do not scan reverse complement strand for motifs")
    p_extract.add_argument("--no-reverse-strand", action="store_true",
                           help="Do not extract ri/rp complementary-strand kinetics. "
                                "By default, both fi/fp (forward) and ri/rp (reverse) "
                                "are extracted; the reverse-strand samples use "
                                "RC(11-mer) as the key, doubling training data and "
                                "making the model strand-invariant.")
    p_extract.add_argument("--min-fraction", type=float, default=0.40,
                           help="Minimum fraction threshold for PacBio CSV (default: 0.40)")
    p_extract.add_argument("--min-detected", type=int, default=20,
                           help="Minimum nDetected threshold for PacBio CSV (default: 20)")
    p_extract.add_argument("--kmer-size", type=int, default=None,
                           help="K-mer window size (default: from encoding.py K=11). "
                                "Must match the value used during training.")
    p_extract.add_argument("--unmeth-subsample-rate", type=float, default=0.05,
                           help="Fraction of unmethylated positions to keep (default: 0.05). "
                                "Methylated positions are always kept. Prevents the dict "
                                "from being dominated by ~4M unmethylated k-mers.")
    p_extract.add_argument("--max-reads", type=int, default=0,
                           help="Stop after N reads (0 = no limit). "
                                "Smoke-test only — biases reservoir sampling.")
    p_extract.add_argument("--verbose", "-v", action="store_true",
                           help="Enable DEBUG-level logging")

    # -- merge subcommand --
    p_merge = sub.add_parser(
        "merge",
        help="Merge multiple *_shard.pkl shards into one master",
        description=(
            "Concatenate raw sample arrays from all shards in a directory.\n"
            "Detects *_shard.pkl files (also supports legacy *_cgan.pkl naming).\n"
            "Subsamples per key if total exceeds --max-samples."
        ),
    )
    p_merge.add_argument("input_dir",
                         help="Directory containing shard .pkl files")
    p_merge.add_argument("output",
                         help="Output master training set .pkl file")
    p_merge.add_argument("--max-samples", type=int, default=50_000,
                         help="Max samples per (kmer, meth_id) after merging "
                              "(default: 50000)")
    p_merge.add_argument("--glob", default="auto",
                         dest="glob_pattern",
                         help="Glob pattern for shard files (default: auto-detect)")
    p_merge.add_argument("--verbose", "-v", action="store_true",
                         help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=getattr(args, "verbose", False))

    if args.command == "merge":
        merge_shards(
            args.input_dir, args.output,
            max_samples_per_key=args.max_samples,
            glob_pattern=args.glob_pattern,
        )

    else:   # extract
        if args.manifest:
            # ---- Manifest mode ----
            if args.task is None:
                log.error("--task is required when using --manifest")
                sys.exit(1)
            if args.output_dir is None:
                log.error("--output-dir is required when using --manifest")
                sys.exit(1)
            extract_from_manifest_task(
                args.manifest,
                task_index           = args.task,
                output_dir           = args.output_dir,
                max_samples_per_key  = args.max_samples,
                revcomp              = not args.no_revcomp,
                use_reverse_strand   = not args.no_reverse_strand,
                max_reads            = args.max_reads,
                kmer_size            = args.kmer_size or K,
                unmeth_subsample_rate= args.unmeth_subsample_rate,
            )

        else:
            # ---- Single-BAM mode ----
            if not args.bam or not args.motifs or not args.output:
                log.error(
                    "Single-BAM mode requires: kinsim extract <bam> <motifs> <output>\n"
                    "Or use manifest mode: kinsim extract --manifest CSV --task N "
                    "--output-dir DIR"
                )
                sys.exit(1)

            motif_string = load_motif_string(
                args.motifs,
                min_fraction=args.min_fraction,
                min_detected=args.min_detected,
            )
            if not motif_string:
                log.error("No motifs found from the provided source.")
                sys.exit(1)

            log.info("Extracting samples from: %s", os.path.basename(args.bam))
            result = extract_samples_from_bam(
                args.bam, motif_string,
                max_samples_per_key=args.max_samples,
                revcomp=not args.no_revcomp,
                use_reverse_strand=not args.no_reverse_strand,
            )

            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "wb") as f:
                pickle.dump(result, f)

            meta = result.get("__meta__", {})
            log.info(
                "Shard saved: %s (%d contexts, %d samples)",
                args.output,
                meta.get("n_unique_keys", "?"),
                meta.get("n_total_samples", "?"),
            )


if __name__ == "__main__":
    main()
