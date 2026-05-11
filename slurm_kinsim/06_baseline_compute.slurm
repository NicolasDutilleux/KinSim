#!/bin/bash
#SBATCH --job-name=kinsim_baseline_compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_baseline_compute_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# kinsim_baseline compute — biology-aware per-(meth_type, offset)
# IPD/PW distribution model.
#
# Single-pass walk over every BAM in the manifest:
#   For each read, for each meth type T (from kinsim_config.yaml),
#   for each position p where read[p] == modified_base[T],
#   for each k in signal_offsets[T]:  bincount(ipd[p+k]) and bincount(pw[p+k])
#   into the per-(T, k) 256-bin histogram.
#
# Outputs (in OUTPUT_DIR):
#   baseline_hist.tsv      long-form histogram, one row per (T, k, metric, bin)
#   baseline_summary.tsv   per-(T, k) mean / p50 / p95 / p99 + ratio columns
#   baseline.json          full histograms
#   run_info.json          manifest + per-BAM read counts + timing
#
# Usage:
#   sbatch slurm_kinsim/06_baseline_compute.slurm <manifest.csv> <output_dir> [threshold]
#
# Default threshold (used only for the summary's "n_above"/IPD ratio columns): 1.3
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

MANIFEST=${1:?"manifest CSV required (sample_id,bam_path,motifs,ref_path)"}
OUT_DIR=${2:?"output directory required"}
THRESHOLD=${3:-1.3}

mkdir -p "$OUT_DIR"

# Hardcoded — `readlink -f $0` resolves to /var/spool/slurmd inside a SLURM
# job, so we can't recover the actual repo path from $0. Matches the
# pattern used by _v12_orchestrator.sh.
REPO=/data/users/ndutilleux/KinSim

echo "================================================================"
echo "  kinsim_baseline compute"
echo "  Host:      $(hostname)"
echo "  Date:      $(date)"
echo "  Manifest:  $MANIFEST"
echo "  Output:    $OUT_DIR"
echo "  Threshold: $THRESHOLD"
echo "  Repo:      $REPO"
echo "================================================================"

[ -s "$MANIFEST" ] || { echo "ERROR: manifest missing or empty: $MANIFEST"; exit 1; }

START=$(date +%s)

cd "$REPO"
python -m kinsim_baseline compute \
    "$MANIFEST" \
    "$OUT_DIR" \
    --threshold "$THRESHOLD" \
    -v

echo ""
echo "Elapsed: $(( $(date +%s) - START )) s"
echo "Done."
echo ""
echo "── Summary ──"
cat "$OUT_DIR/baseline_summary.tsv"
