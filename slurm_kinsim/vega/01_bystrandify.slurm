#!/bin/bash
#SBATCH --job-name=vega_01_by
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_01_by_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega step 01 — ccs-kinetics-bystrandify (fi/fp → ip/pw per strand)
#
# Usage:
#   sbatch --array=1-16%4 slurm_kinsim/vega/01_bystrandify.slurm
#
# Output:  ${VEGA}/assembly/bcXXXX/bcXXXX_bystrandify.bam
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
OUTBASE=${VEGA}/assembly
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

BARCODES=(2033 2034 2035 2036 2037 2038 2039 2040
          2041 2042 2043 2044 2045 2046 2047 2048)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
BC=${BARCODES[$IDX]}
SAMPLE="bc${BC}"
BAM="${VEGA}/m21026_260313_002345.hifi_reads.bc${BC}.bam"
BYSTRAND="${OUTBASE}/${SAMPLE}/${SAMPLE}_bystrandify.bam"

mkdir -p "$(dirname "$BYSTRAND")"

START=$(date +%s)
echo "========================================================"
echo "  Vega 01/bystrandify — $SAMPLE"
echo "========================================================"

if [ ! -f "$BAM" ]; then
    echo "ERROR: input BAM not found: $BAM"; exit 1
fi

if [ -f "$BYSTRAND" ] && [ -s "$BYSTRAND" ]; then
    echo "SKIP — bystrandify BAM exists"
    exit 0
fi

apptainer exec --bind /data "$SMRT_SIF" \
    ccs-kinetics-bystrandify "$BAM" "$BYSTRAND"

if [ ! -f "$BYSTRAND" ]; then
    echo "FAILED: bystrandify produced no output"; exit 1
fi

echo "Elapsed: $(( $(date +%s) - START ))s"
