#!/bin/bash
#SBATCH --job-name=sequel_ccs
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_ccs_%A_%a.log

# Convert Sequel II subreads to HiFi reads with kinetics (fi/fp/ri/rp tags)
# Requires: ccs (from SMRT-Link or pb-ccs conda package)
#
# Usage:
#   # First count BAMs:
#   N=$(ls /data/projects/p774_MARSD/NDutilleux/training/Sequel/*.subreads.bam 2>/dev/null | wc -l)
#   sbatch --array=1-${N}%5 slurm_kinsim/sequel_01_ccs.slurm

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

# Try to load ccs — multiple possible module names
CCS_FOUND=false
for mod in "SMRT-Link/13.0.0" "SMRT-Link/12.0.0" "smrtlink/13.0" "smrtlink/12.0" \
           "pb-ccs/7.0.0" "pb-ccs/6.4.0" "ccs/7.0.0" "ccs/6.4.0"; do
    if module load "$mod" 2>/dev/null; then
        echo "Loaded module: $mod"
        CCS_FOUND=true
        break
    fi
done

# Check if ccs is available (maybe via conda or PATH)
if ! $CCS_FOUND; then
    if command -v ccs &>/dev/null; then
        echo "ccs found in PATH: $(which ccs)"
        CCS_FOUND=true
    fi
fi

if ! $CCS_FOUND; then
    echo "ERROR: ccs not found. Install via:"
    echo "  conda install -c bioconda pbccs"
    echo "  OR load a SMRT-Link module"
    echo ""
    echo "Available modules matching 'smrt' or 'ccs':"
    module avail 2>&1 | grep -iE "smrt|ccs" || echo "  (none found)"
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

# Find the Nth subread BAM
mapfile -t BAMS < <(ls "${SEQUEL}"/*.subreads.bam 2>/dev/null | sort)
N_BAMS=${#BAMS[@]}

if [ "$N_BAMS" -eq 0 ]; then
    echo "ERROR: No .subreads.bam files found in $SEQUEL"
    exit 1
fi

IDX=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$IDX" -ge "$N_BAMS" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID exceeds number of BAMs ($N_BAMS). Exiting."
    exit 0
fi

SUBREADS_BAM="${BAMS[$IDX]}"
BASENAME=$(basename "$SUBREADS_BAM" .subreads.bam)
HIFI_BAM="${HIFI_DIR}/${BASENAME}.hifi.bam"

echo "Input:  $SUBREADS_BAM"
echo "Output: $HIFI_BAM"
echo ""

if [ -f "$HIFI_BAM" ]; then
    echo "Output already exists, skipping."
    exit 0
fi

# Check if input is compressed (.bam.gz) — decompress if needed
if [ -f "${SUBREADS_BAM}.gz" ] && [ ! -f "$SUBREADS_BAM" ]; then
    echo "Decompressing ${SUBREADS_BAM}.gz ..."
    gunzip -k "${SUBREADS_BAM}.gz"
    DECOMPRESSED=true
else
    DECOMPRESSED=false
fi

# Run CCS with all kinetics preserved
echo "Running CCS ($(date '+%H:%M:%S')) ..."
ccs "$SUBREADS_BAM" "$HIFI_BAM" \
    --all-kinetics \
    --min-rq 0.99 \
    --num-threads "${SLURM_CPUS_PER_TASK}" \
    --log-level INFO \
    --report-file "${HIFI_DIR}/${BASENAME}.ccs_report.txt"

echo ""
echo "CCS done: $(date '+%H:%M:%S')"

# Index the output
if command -v pbindex &>/dev/null; then
    pbindex "$HIFI_BAM" 2>/dev/null || true
fi

# Clean up decompressed BAM (keep the .gz)
if $DECOMPRESSED && [ -f "$HIFI_BAM" ]; then
    echo "Removing decompressed subreads BAM (keeping .gz)"
    rm -f "$SUBREADS_BAM"
fi

# Quick stats
echo ""
echo "=== Stats ==="
echo "Input subreads:"
ls -lh "$SUBREADS_BAM" 2>/dev/null || ls -lh "${SUBREADS_BAM}.gz" 2>/dev/null || true
echo "Output HiFi:"
ls -lh "$HIFI_BAM"

# Verify kinetic tags
python -c "
import pysam
bam = pysam.AlignmentFile('${HIFI_BAM}', check_sq=False)
read = next(iter(bam))
tags = {t[0] for t in read.get_tags()}
kinetic_tags = tags & {'fi', 'fp', 'ri', 'rp', 'ip', 'pw'}
print(f'Read length: {read.query_length}bp')
print(f'Kinetic tags: {sorted(kinetic_tags)}')
if 'fi' in tags or 'ip' in tags:
    print('OK: Kinetic data preserved')
else:
    print('WARNING: No kinetic tags found!')
bam.close()
"

echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
