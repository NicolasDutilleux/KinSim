#!/bin/bash
#SBATCH --job-name=KinSim_Baselines
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_baselines_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim — Run all 3 baseline models
#
# Trains and saves all baselines for comparison with ConvPredictor:
#   1. Global Gaussian      (4 distributions, no kmer context)
#   2. Per-kmer Gaussian    (kmer-specific + global IPD ratio shift)
#   3. ConvNoFiLM           (ConvPredictor without FiLM, post-hoc ratio)
#
# Usage:
#   sbatch slurm_kinsim/05_baselines.slurm <manifest.csv> <merged.pkl> <output_dir> [epochs]
#
# Arguments:
#   $1  Manifest CSV (sample_id, bam_path, motifs) — for baselines 1 & 2
#   $2  Merged training .pkl (from kinsim merge) — for baseline 3
#   $3  Output directory (subdirs created per baseline)
#   $4  (optional) Epochs for ConvNoFiLM training (default: 50)
# ============================================================

set -euo pipefail

echo "============================================"
echo "  KinSim Baselines"
echo "  Host:  $(hostname)"
echo "  Date:  $(date)"
echo "  GPU:   ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================"

# --- Args ---
MANIFEST=${1:?Usage: sbatch 05_baselines.slurm <manifest.csv> <merged.pkl> <output_dir> [epochs]}
MERGED_PKL=${2:?Missing: merged .pkl path}
OUTDIR=${3:?Missing: output directory}
EPOCHS=${4:-50}

# --- Environment ---
source ~/.bashrc
conda activate kinsim_env

echo ""
echo "Manifest:   $MANIFEST"
echo "Merged PKL: $MERGED_PKL"
echo "Output dir: $OUTDIR"
echo "Epochs:     $EPOCHS"
echo ""

REPODIR=$(dirname "$(dirname "$(readlink -f "$0")")")

# ============================================================
# Baseline 1: Global Gaussian
# ============================================================
echo "=========================================="
echo "  Baseline 1: Global Gaussian"
echo "  $(date)"
echo "=========================================="

GLOBAL_OUT=${OUTDIR}/global_gaussian
mkdir -p "$GLOBAL_OUT"

python -c "
import sys
sys.path.insert(0, '${REPODIR}')
from baseline.global_gaussian import fit_from_manifest, print_model, save_model
model = fit_from_manifest('${MANIFEST}', max_reads_per_bam=5000)
print_model(model)
save_model(model, '${GLOBAL_OUT}/global_gaussian.json')
print('Baseline 1 done.')
"

echo "Baseline 1 saved to: ${GLOBAL_OUT}/global_gaussian.json"
echo ""

# ============================================================
# Baseline 2: Per-kmer Gaussian
# ============================================================
echo "=========================================="
echo "  Baseline 2: Per-kmer Gaussian"
echo "  $(date)"
echo "=========================================="

KMER_OUT=${OUTDIR}/kmer_gaussian
mkdir -p "$KMER_OUT"

python -c "
import sys
sys.path.insert(0, '${REPODIR}')
from baseline.kmer_gaussian import fit_from_manifest, print_model, save_model
model = fit_from_manifest('${MANIFEST}', max_reads_per_bam=5000)
print_model(model)
save_model(model, '${KMER_OUT}')
print('Baseline 2 done.')
"

echo "Baseline 2 saved to: ${KMER_OUT}/"
echo ""

# ============================================================
# Baseline 3: ConvNoFiLM (requires GPU)
# ============================================================
echo "=========================================="
echo "  Baseline 3: ConvNoFiLM (${EPOCHS} epochs)"
echo "  $(date)"
echo "=========================================="

CONV_OUT=${OUTDIR}/conv_no_film

python -c "
import sys, logging
sys.path.insert(0, '${REPODIR}')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s')
from baseline.conv_no_film import fit, evaluate
fit('${MERGED_PKL}', '${CONV_OUT}', epochs=${EPOCHS})
print('')
print('--- Evaluating on training data ---')
evaluate('${CONV_OUT}', '${MERGED_PKL}')
print('Baseline 3 done.')
"

echo "Baseline 3 saved to: ${CONV_OUT}/"
echo ""

# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo "  All baselines complete"
echo "  $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  1. Global Gaussian:  ${GLOBAL_OUT}/global_gaussian.json"
echo "  2. Per-kmer Gaussian: ${KMER_OUT}/"
echo "  3. ConvNoFiLM:       ${CONV_OUT}/"
echo ""
ls -lh "${GLOBAL_OUT}/" "${KMER_OUT}/" "${CONV_OUT}/" 2>/dev/null || true
echo ""
echo "Done."
