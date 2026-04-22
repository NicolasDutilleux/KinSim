#!/bin/bash
#SBATCH --job-name=sequel_02_aln
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_02_aln_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel step 02 — pbmm2 align against downloaded per-barcode assemblies
# Assembly path: ${SEQUEL}/assemblies/bcXXXX_assembly.fsa
#
# Usage:
#   sbatch --array=1-48%4 slurm_kinsim/sequel/02_align.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
OUTBASE=${SEQUEL}/ipdsummary_ccs
ASM_DIR=${SEQUEL}/assemblies

mapfile -t GZ_BAMS < <(ls "${SEQUEL}"/lima.bc*--bc*.bam.gz 2>/dev/null | sort)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
[ $IDX -lt ${#GZ_BAMS[@]} ] || exit 0

BC=$(basename "${GZ_BAMS[$IDX]}" .bam.gz | grep -oP 'bc\d+' | head -1)
ASM_FASTA="${ASM_DIR}/${BC}_assembly.fsa"
BYSTRAND="${OUTBASE}/${BC}/${BC}_bystrandify.bam"
ALIGNED="${OUTBASE}/${BC}/${BC}_aligned.bam"

echo "=== Sequel 02/align — $BC ==="
[ -f "$ASM_FASTA" ] || { echo "ERROR: assembly missing: $ASM_FASTA"; exit 1; }
[ -f "$BYSTRAND" ] || { echo "ERROR: bystrandify missing: $BYSTRAND"; exit 1; }

if [ -f "$ALIGNED" ] && [ -s "$ALIGNED" ]; then
    echo "SKIP — aligned BAM exists"; exit 0
fi

[ -f "${ASM_FASTA}.fai" ] || samtools faidx "$ASM_FASTA"

pbmm2 align "$ASM_FASTA" "$BYSTRAND" "$ALIGNED" \
    --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"

[ -s "$ALIGNED" ] || { echo "FAILED: pbmm2 produced no output"; exit 1; }
samtools flagstat "$ALIGNED" | head -5
