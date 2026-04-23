#!/bin/bash
#SBATCH --job-name=prep_assembly
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# prep/assembly_hifiasm — draft de novo assembly from HiFi reads
#
# Usage:
#   sbatch -J prep_asm_<sample> slurm_kinsim/prep/assembly_hifiasm.slurm \
#          <raw_bam> <out_fasta>
#
# Produces: <out_fasta>  (cleaned primary-contig FASTA from hifiasm GFA)
#
# Intermediate files (.bp.*.gfa, *.bin, *.ec.fa) are cleaned up after.
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

RAW_BAM=${1:?"Usage: sbatch $0 <raw_bam> <out_fasta>"}
OUT_FASTA=${2:?"out_fasta missing"}

[ -f "$RAW_BAM" ] || { echo "ERROR: raw BAM missing: $RAW_BAM"; exit 1; }

OUTDIR=$(dirname "$OUT_FASTA")
mkdir -p "$OUTDIR"

ASM_PREFIX="${OUT_FASTA%_assembly.fasta}"
[ "$ASM_PREFIX" = "$OUT_FASTA" ] && ASM_PREFIX="${OUT_FASTA%.fasta}"
FASTQ="${ASM_PREFIX}_tmp.fastq"

T0=$(date +%s)
echo "========================================================"
echo "  prep/assembly_hifiasm — $(basename "$OUT_FASTA")"
echo "========================================================"
echo "Raw BAM:   $RAW_BAM"
echo "Out FASTA: $OUT_FASTA"

if [ -s "$OUT_FASTA" ]; then
    echo "SKIP — assembly exists ($(grep -c '^>' "$OUT_FASTA") contigs)"
    exit 0
fi

echo "=== BAM → FASTQ ==="
samtools fastq "$RAW_BAM" > "$FASTQ"

echo "=== hifiasm (${SLURM_CPUS_PER_TASK} threads) ==="
hifiasm -o "$ASM_PREFIX" -t "${SLURM_CPUS_PER_TASK}" "$FASTQ"

GFA="${ASM_PREFIX}.bp.p_ctg.gfa"
[ -f "$GFA" ] || { echo "ERROR: hifiasm GFA not found: $GFA"; exit 1; }

echo "=== GFA → FASTA ==="
awk '/^S/{print ">"$2; print $3}' "$GFA" > "$OUT_FASTA"

N_CONTIGS=$(grep -c '^>' "$OUT_FASTA")
ASM_SIZE=$(grep -v '^>' "$OUT_FASTA" | wc -c)
echo "Contigs: $N_CONTIGS    Size: $((ASM_SIZE / 1000000)) Mb"

# Cleanup intermediates (keep GFA + FASTA)
rm -f "$FASTQ"
rm -f "${ASM_PREFIX}".bp.hap*.gfa "${ASM_PREFIX}".bp.r_utg.gfa
rm -f "${ASM_PREFIX}"*.bin "${ASM_PREFIX}"*.ec.fa

echo "Elapsed: $(( $(date +%s) - T0 ))s"
