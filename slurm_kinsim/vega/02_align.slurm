#!/bin/bash
#SBATCH --job-name=vega_02_aln
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_02_aln_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega step 02 — pbmm2 align (bystrandified BAM → draft assembly)
#
# Usage:
#   sbatch --array=1-16%4 slurm_kinsim/vega/02_align.slurm
#
# Output:  ${VEGA}/assembly/bcXXXX/bcXXXX_aligned.bam
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
OUTBASE=${VEGA}/assembly

BARCODES=(2033 2034 2035 2036 2037 2038 2039 2040
          2041 2042 2043 2044 2045 2046 2047 2048)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
BC=${BARCODES[$IDX]}
SAMPLE="bc${BC}"
SPECIES_DIR="${OUTBASE}/${SAMPLE}"
ASM_FASTA="${SPECIES_DIR}/${SAMPLE}_assembly.fasta"
BYSTRAND="${SPECIES_DIR}/${SAMPLE}_bystrandify.bam"
ALIGNED="${SPECIES_DIR}/${SAMPLE}_aligned.bam"

START=$(date +%s)
echo "========================================================"
echo "  Vega 02/align — $SAMPLE"
echo "========================================================"

[ -f "$ASM_FASTA" ] || { echo "ERROR: assembly missing: $ASM_FASTA"; exit 1; }
[ -f "$BYSTRAND" ] || { echo "ERROR: bystrandify missing: $BYSTRAND"; exit 1; }

if [ -f "$ALIGNED" ] && [ -s "$ALIGNED" ]; then
    echo "SKIP — aligned BAM exists"
    exit 0
fi

[ -f "${ASM_FASTA}.fai" ] || samtools faidx "$ASM_FASTA"

pbmm2 align "$ASM_FASTA" "$BYSTRAND" "$ALIGNED" \
    --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"

if [ ! -f "$ALIGNED" ]; then
    echo "FAILED: pbmm2 produced no output"; exit 1
fi

samtools flagstat "$ALIGNED" | head -5
echo "Elapsed: $(( $(date +%s) - START ))s"
