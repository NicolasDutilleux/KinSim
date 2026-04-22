#!/bin/bash
#SBATCH --job-name=vega_00_asm
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_00_asm_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega step 00 — hifiasm draft assembly (one per barcode)
#
# Usage:
#   sbatch --array=1-16%4 slurm_kinsim/vega/00_assembly.slurm
#
# Output:  ${VEGA}/assembly/bcXXXX/bcXXXX_assembly.fasta
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
BAM="${VEGA}/m21026_260313_002345.hifi_reads.bc${BC}.bam"

SPECIES_DIR="${OUTBASE}/${SAMPLE}"
ASM_PREFIX="${SPECIES_DIR}/${SAMPLE}"
ASM_FASTA="${SPECIES_DIR}/${SAMPLE}_assembly.fasta"
mkdir -p "$SPECIES_DIR"

START=$(date +%s)
echo "========================================================"
echo "  Vega 00/assembly — $SAMPLE (task $SLURM_ARRAY_TASK_ID)"
echo "  Host: $(hostname)  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

if [ ! -f "$BAM" ]; then
    echo "ERROR: BAM not found: $BAM"; exit 1
fi

if [ -f "$ASM_FASTA" ] && [ -s "$ASM_FASTA" ]; then
    echo "SKIP — assembly exists ($(grep -c '^>' "$ASM_FASTA") contigs)"
    exit 0
fi

FASTQ="${SPECIES_DIR}/${SAMPLE}.fastq"
samtools fastq "$BAM" > "$FASTQ"

hifiasm -o "$ASM_PREFIX" -t "${SLURM_CPUS_PER_TASK}" "$FASTQ"

GFA="${ASM_PREFIX}.bp.p_ctg.gfa"
if [ ! -f "$GFA" ]; then
    echo "ERROR: hifiasm GFA not found: $GFA"; ls -la "${ASM_PREFIX}".*; exit 1
fi
awk '/^S/{print ">"$2; print $3}' "$GFA" > "$ASM_FASTA"

N_CONTIGS=$(grep -c '^>' "$ASM_FASTA")
ASM_SIZE=$(grep -v '^>' "$ASM_FASTA" | wc -c)
echo "Contigs: $N_CONTIGS    Size: $((ASM_SIZE / 1000000)) Mb"

# Cleanup intermediates (keep GFA + FASTA)
rm -f "$FASTQ"
rm -f "${ASM_PREFIX}".bp.hap*.gfa "${ASM_PREFIX}".bp.r_utg.gfa
rm -f "${ASM_PREFIX}"*.bin "${ASM_PREFIX}"*.ec.fa

echo "Elapsed: $(( $(date +%s) - START ))s  Ended: $(date '+%H:%M:%S')"
