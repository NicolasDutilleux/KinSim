#!/bin/bash
#SBATCH --job-name=sequel_align
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=08:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_align_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel II 48-plex — Align HiFi BAMs to assemblies for extraction
#
# The Sequel HiFi BAMs carry fi/fp/ri/rp tags (from CCS --all-kinetics).
# For KinSim extraction we need aligned BAMs with ip/pw tags:
#
#   1. bystrandify (fi/fp -> ip/pw, unaligned only)
#   2. pbmm2 align --preset CCS to PacBio assembly
#   3. samtools index + pbindex
#
# GFF annotations from sequel_02_ipdsummary (subreads) are reused —
# same reference, so coordinates are valid.
#
# Usage:
#   sbatch --array=1-48 slurm_kinsim/sequel_03_align_hifi.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

# ---- Config ----
SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
HIFI_DIR=${SEQUEL}/hifi
ASM_DIR=${SEQUEL}/assemblies
OUTBASE=${SEQUEL}/aligned_hifi
SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"

mkdir -p "$OUTBASE"

# ---- Select HiFi BAM for this task ----
mapfile -t HIFI_BAMS < <(ls "${HIFI_DIR}"/*.hifi.bam 2>/dev/null | sort)
N_BAMS=${#HIFI_BAMS[@]}

if [ "$N_BAMS" -eq 0 ]; then
    echo "ERROR: No .hifi.bam files found in $HIFI_DIR"
    exit 1
fi

IDX=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$IDX" -ge "$N_BAMS" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID > $N_BAMS BAMs. Nothing to do."
    exit 0
fi

HIFI_BAM="${HIFI_BAMS[$IDX]}"
SAMPLE=$(basename "$HIFI_BAM" .hifi.bam)
# Extract barcode from filename (e.g., lima.bc1018--bc1018.hifi → bc1018)
BC=$(echo "$SAMPLE" | grep -oP 'bc\d+' | head -1)

ASM_FASTA="${ASM_DIR}/${BC}_assembly.fsa"
SPECIES_DIR="${OUTBASE}/${BC}"
mkdir -p "$SPECIES_DIR"

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  Sequel HiFi Align — $BC (task $SLURM_ARRAY_TASK_ID)"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "HiFi BAM:   $HIFI_BAM"
echo "Assembly:   $ASM_FASTA"
echo "Output:     $SPECIES_DIR/"
echo "========================================================"

if [ ! -f "$ASM_FASTA" ]; then
    echo "ERROR: Assembly not found: $ASM_FASTA"
    exit 1
fi

# ---- Step 1: bystrandify (fi/fp -> ip/pw, unaligned only) ----
BYSTRAND="${SPECIES_DIR}/${BC}_hifi_bystrandify.bam"

if [ -f "$BYSTRAND" ] && [ -s "$BYSTRAND" ]; then
    echo "=== Step 1: SKIP — bystrandify BAM exists ==="
else
    echo "=== Step 1: ccs-kinetics-bystrandify ==="
    echo "Started: $(date '+%H:%M:%S')"

    apptainer exec --bind /data "$SMRT_SIF" \
        ccs-kinetics-bystrandify "$HIFI_BAM" "$BYSTRAND"

    if [ ! -s "$BYSTRAND" ]; then
        echo "FAILED: bystrandify produced empty file"
        exit 1
    fi
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 2: pbmm2 align (bystrandified HiFi to assembly) ----
ALIGNED="${SPECIES_DIR}/${BC}_hifi_aligned.bam"

if [ -f "$ALIGNED" ] && [ -s "$ALIGNED" ]; then
    echo "=== Step 2: SKIP — aligned HiFi BAM exists ==="
else
    echo "=== Step 2: pbmm2 align --preset CCS ==="
    echo "Started: $(date '+%H:%M:%S')"

    if [ ! -f "${ASM_FASTA}.fai" ]; then
        samtools faidx "$ASM_FASTA"
    fi

    pbmm2 align "$ASM_FASTA" "$BYSTRAND" "$ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"

    if [ ! -s "$ALIGNED" ]; then
        echo "FAILED: pbmm2 produced empty file"
        exit 1
    fi
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 3: Index ----
echo "=== Step 3: Index BAM ==="

if [ ! -f "${ALIGNED}.bai" ]; then
    samtools index "$ALIGNED"
fi

if [ ! -f "${ALIGNED}.pbi" ]; then
    apptainer exec --bind /data "$SMRT_SIF" \
        pbindex "$ALIGNED"
fi

echo "Alignment stats:"
samtools flagstat "$ALIGNED" | head -5
echo ""

# ---- Cleanup intermediate bystrandified BAM ----
if [ -s "$ALIGNED" ]; then
    rm -f "$BYSTRAND"
    echo "Cleaned up intermediate: $BYSTRAND"
fi

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================================"
echo "  COMPLETE: $BC"
echo "========================================================"
echo "  Aligned HiFi: $ALIGNED"
echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
