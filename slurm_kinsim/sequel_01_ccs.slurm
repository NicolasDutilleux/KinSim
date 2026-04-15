#!/bin/bash
#SBATCH --job-name=sequel_ccs
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_ccs_%A_%a.log

# Convert Sequel II demultiplexed subreads to HiFi reads with kinetics
# Input:  lima.bcXXXX--bcXXXX.bam.gz  (compressed subreads)
# Output: hifi/lima.bcXXXX--bcXXXX.hifi.bam (HiFi with fi/fp/ri/rp tags)
#
# Usage:
#   sbatch --array=1-48%5 slurm_kinsim/sequel_01_ccs.slurm

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

# Try to find ccs
for mod in "SMRT-Link/13.0.0" "SMRT-Link/12.0.0" "smrtlink/13.0" "smrtlink/12.0" \
           "pb-ccs/7.0.0" "pb-ccs/6.4.0" "ccs/7.0.0" "ccs/6.4.0"; do
    if module load "$mod" 2>/dev/null; then
        echo "Loaded module: $mod"
        break
    fi
done

if ! command -v ccs &>/dev/null; then
    echo "ERROR: ccs not found. Install via: conda install -c bioconda pbccs"
    echo "Available modules:"
    module avail 2>&1 | grep -iE "smrt|ccs" || echo "  (none)"
    exit 1
fi
echo "ccs version: $(ccs --version 2>&1 | head -1)"

SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
HIFI_DIR=${SEQUEL}/hifi
mkdir -p "$HIFI_DIR"

echo "========================================================"
echo "  Sequel II CCS conversion — task ${SLURM_ARRAY_TASK_ID}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""

# Find the Nth .bam.gz file
mapfile -t GZBAMS < <(ls "${SEQUEL}"/lima.bc*--bc*.bam.gz 2>/dev/null | sort)
N_BAMS=${#GZBAMS[@]}

if [ "$N_BAMS" -eq 0 ]; then
    echo "ERROR: No lima.bc*--bc*.bam.gz files found in $SEQUEL"
    exit 1
fi

IDX=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$IDX" -ge "$N_BAMS" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID > $N_BAMS BAMs. Nothing to do."
    exit 0
fi

GZ_BAM="${GZBAMS[$IDX]}"
BASENAME=$(basename "$GZ_BAM" .bam.gz)
SUBREADS_BAM="${SEQUEL}/${BASENAME}.bam"
HIFI_BAM="${HIFI_DIR}/${BASENAME}.hifi.bam"

echo "Input:  $GZ_BAM"
echo "Output: $HIFI_BAM"
echo ""

if [ -f "$HIFI_BAM" ]; then
    echo "Output already exists, skipping."
    exit 0
fi

# Decompress
echo "Decompressing $(basename "$GZ_BAM") ..."
gunzip -k "$GZ_BAM"
echo "  -> $(ls -lh "$SUBREADS_BAM" | awk '{print $5}')"

# Run CCS
echo ""
echo "Running CCS ($(date '+%H:%M:%S')) ..."
ccs "$SUBREADS_BAM" "$HIFI_BAM" \
    --all-kinetics \
    --min-rq 0.99 \
    --num-threads "${SLURM_CPUS_PER_TASK}" \
    --log-level INFO \
    --report-file "${HIFI_DIR}/${BASENAME}.ccs_report.txt"

echo ""
echo "CCS done: $(date '+%H:%M:%S')"

# Index
if command -v pbindex &>/dev/null; then
    pbindex "$HIFI_BAM" 2>/dev/null || true
fi

# Clean up decompressed BAM (keep the .gz)
if [ -f "$HIFI_BAM" ]; then
    echo "Removing decompressed subreads (keeping .gz)"
    rm -f "$SUBREADS_BAM"
fi

# Quick verification
echo ""
echo "=== Verification ==="
ls -lh "$HIFI_BAM"
python -c "
import pysam
bam = pysam.AlignmentFile('${HIFI_BAM}', check_sq=False)
read = next(iter(bam))
tags = {t[0] for t in read.get_tags()}
kinetic = tags & {'fi', 'fp', 'ri', 'rp', 'ip', 'pw'}
print(f'Read length: {read.query_length}bp')
print(f'Kinetic tags: {sorted(kinetic)}')
if kinetic:
    print('OK: Kinetic data preserved')
else:
    print('WARNING: No kinetic tags found!')
bam.close()
"

echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
