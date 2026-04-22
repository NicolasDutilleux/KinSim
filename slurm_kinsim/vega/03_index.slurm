#!/bin/bash
#SBATCH --job-name=vega_03_idx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_03_idx_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega step 03 — samtools index + pbindex (apptainer)
#
# Usage:
#   sbatch --array=1-16%8 slurm_kinsim/vega/03_index.slurm
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
ALIGNED="${OUTBASE}/${SAMPLE}/${SAMPLE}_aligned.bam"

echo "=== Vega 03/index — $SAMPLE ==="

[ -f "$ALIGNED" ] || { echo "ERROR: aligned BAM missing: $ALIGNED"; exit 1; }

[ -f "${ALIGNED}.bai" ] || samtools index "$ALIGNED"
[ -f "${ALIGNED}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$ALIGNED"

echo "OK"
ls -l "${ALIGNED}"* | awk '{print "  "$NF"  "$5}'
