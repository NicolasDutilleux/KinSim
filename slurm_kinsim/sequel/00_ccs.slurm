#!/bin/bash
#SBATCH --job-name=sequel_00_ccs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_00_ccs_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel step 00 — subreads → HiFi (ccs --hifi-kinetics)
#
# Decompress lima.bc*--bc*.bam.gz, run ccs with per-strand kinetic tags,
# remove the decompressed .bam on success (keep .gz + HiFi).
#
# Usage:
#   sbatch --array=1-48%4 slurm_kinsim/sequel/00_ccs.slurm
#
# Output: ${SEQUEL}/ipdsummary_ccs/bcXXXX/bcXXXX_hifi.bam
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
OUTBASE=${SEQUEL}/ipdsummary_ccs
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

CCS_MIN_RQ=0.99
CCS_MIN_PASSES=3

mapfile -t GZ_BAMS < <(ls "${SEQUEL}"/lima.bc*--bc*.bam.gz 2>/dev/null | sort)
[ ${#GZ_BAMS[@]} -gt 0 ] || { echo "ERROR: no subread .bam.gz found in $SEQUEL"; exit 1; }

IDX=$((SLURM_ARRAY_TASK_ID - 1))
[ $IDX -lt ${#GZ_BAMS[@]} ] || { echo "Task $SLURM_ARRAY_TASK_ID > ${#GZ_BAMS[@]} BAMs — nothing to do"; exit 0; }

GZ_BAM="${GZ_BAMS[$IDX]}"
FNAME=$(basename "$GZ_BAM" .bam.gz)
BC=$(echo "$FNAME" | grep -oP 'bc\d+' | head -1)
SUBREADS="${SEQUEL}/${FNAME}.bam"
SPECIES_DIR="${OUTBASE}/${BC}"
HIFI="${SPECIES_DIR}/${BC}_hifi.bam"

mkdir -p "$SPECIES_DIR"
START=$(date +%s)

echo "=== Sequel 00/ccs — $BC ==="

if [ -f "$HIFI" ] && [ -s "$HIFI" ]; then
    echo "SKIP — HiFi BAM exists ($(samtools view -c "$HIFI" 2>/dev/null || echo '?') reads)"
    exit 0
fi

[ -f "$SUBREADS" ] || gunzip -k "$GZ_BAM"

apptainer exec --bind /data "$SMRT_SIF" \
    ccs "$SUBREADS" "$HIFI" \
        --min-rq "$CCS_MIN_RQ" \
        --min-passes "$CCS_MIN_PASSES" \
        --hifi-kinetics \
        -j "${SLURM_CPUS_PER_TASK}"

N_HIFI=$(samtools view -c "$HIFI" 2>/dev/null || echo 0)
[ "$N_HIFI" -gt 0 ] || { echo "FAILED: ccs produced 0 reads"; exit 1; }

echo "HiFi reads: $N_HIFI"
# keep .gz, drop decompressed intermediate
rm -f "$SUBREADS"

echo "Elapsed: $(( $(date +%s) - START ))s"
