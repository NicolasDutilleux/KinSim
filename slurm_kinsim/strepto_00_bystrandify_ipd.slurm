#!/bin/bash
#SBATCH --job-name=strepto_ipd
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --exclude=binfservas18
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/strepto_ipd_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Streptomyces GFF pipeline — Per-species bystrandify + align + ipdSummary
#
# Array job: one task per species from the training manifest.
# Each task does:
#   1. ccs-kinetics-bystrandify  (recover kinetic signal from CCS)
#   2. pbmm2 align               (align to per-species reference)
#   3. samtools + pbindex         (create indices)
#   4. ipdSummary SP3-C3          (call m6A + m4C with methylFraction)
#
# Usage:
#   N=$(kinsim-prep manifest count <manifest.csv>)
#   sbatch --array=1-$N%4 strepto_00_bystrandify_ipd.slurm <manifest.csv> <output_dir>
#
# Arguments:
#   $1  Existing training manifest CSV (sample_id, bam_path, motifs)
#       bam_path must point to UNALIGNED HiFi BAMs with fi/fp tags.
#   $2  Output directory for bystrandified/aligned BAMs + GFF files.
#       Creates <output_dir>/<sample_id>/ per species.
#
# Output per species:
#   <output_dir>/<sample_id>/
#       <sample_id>_bystrandify.bam
#       <sample_id>_aligned.bam        (.bai + .pbi indexed)
#       <sample_id>_ipdSummary.gff
#       <sample_id>_ipdSummary.csv
#
# Prerequisites:
#   - apptainer with pacbio-smrt-tools-25.3.sif (for bystrandify + pbindex)
#   - SMRT-Link/12.0.0 module (for ipdSummary with SP3-C3 model)
#   - pbmm2 in conda env (kinsim_env)
#   - Per-species reference at: .../Strepto/<sample_id>/final_assembly.fasta
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch --array=1-N%4 strepto_00_bystrandify_ipd.slurm <manifest.csv> <output_dir>"
    echo ""
    echo "  manifest.csv   Training manifest with sample_id, bam_path, motifs columns"
    echo "  output_dir     Base directory for per-species bystrandify + GFF outputs"
    echo ""
    echo "Count N:  kinsim-prep manifest count manifest.csv"
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: Must be submitted as an array job (--array=1-N)"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# Load SMRT-Link for ipdSummary (SP3-C3 model)
source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true

# ---- Config ----
MANIFEST="$1"
OUTBASE="$2"

SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"
STREPTO_BASE="/data/projects/p774_MARSD/NDutilleux/training/Strepto"

# SP3-C3 model — closest available null model to Revio chemistry
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

# ---- Read manifest row ----
TASK=$SLURM_ARRAY_TASK_ID

# Extract the Nth data row (skip header + comment lines)
ROW=$(python -c "
from kinsim.utils.config import load_manifest
entries = load_manifest('${MANIFEST}')
idx = ${TASK} - 1
if idx >= len(entries):
    raise SystemExit(f'Task {${TASK}} exceeds manifest rows ({len(entries)})')
e = entries[idx]
print(f'{e.sample_id}|{e.bam_path}')
")
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to read manifest row $TASK"
    echo "$ROW"
    exit 1
fi

SAMPLE_ID=$(echo "$ROW" | cut -d'|' -f1)
BAM_PATH=$(echo "$ROW" | cut -d'|' -f2)

# Derive reference path from sample_id
REF="${STREPTO_BASE}/${SAMPLE_ID}/final_assembly.fasta"

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  Streptomyces GFF Pipeline — Task $TASK"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Sample:     $SAMPLE_ID"
echo "BAM:        $BAM_PATH"
echo "Reference:  $REF"
echo "Output:     $OUTBASE/$SAMPLE_ID/"
echo "========================================================"
echo ""

# ---- Validate inputs ----
if [ ! -f "$BAM_PATH" ]; then
    echo "ERROR: BAM not found: $BAM_PATH"
    exit 1
fi
if [ ! -f "$REF" ]; then
    echo "ERROR: Reference not found: $REF"
    exit 1
fi

# ---- Output paths ----
SPECIES_DIR="${OUTBASE}/${SAMPLE_ID}"
mkdir -p "$SPECIES_DIR"

BYSTRAND="${SPECIES_DIR}/${SAMPLE_ID}_bystrandify.bam"
ALIGNED="${SPECIES_DIR}/${SAMPLE_ID}_aligned.bam"
GFF="${SPECIES_DIR}/${SAMPLE_ID}_ipdSummary.gff"
CSV="${SPECIES_DIR}/${SAMPLE_ID}_ipdSummary.csv"

# ---- Step 1: ccs-kinetics-bystrandify ----
if [ -f "$BYSTRAND" ] && [ -s "$BYSTRAND" ]; then
    echo "=== Step 1: SKIP — bystrandify BAM already exists ==="
else
    echo "=== Step 1: ccs-kinetics-bystrandify ==="
    echo "Started: $(date '+%H:%M:%S')"

    apptainer exec --bind /data "$SMRT_SIF" \
        ccs-kinetics-bystrandify "$BAM_PATH" "$BYSTRAND"

    RC=$?
    if [ $RC -ne 0 ] || [ ! -f "$BYSTRAND" ]; then
        echo "FAILED: bystrandify exit=$RC"
        exit 1
    fi
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 2: pbmm2 align ----
if [ -f "$ALIGNED" ] && [ -s "$ALIGNED" ]; then
    echo "=== Step 2: SKIP — aligned BAM already exists ==="
else
    echo "=== Step 2: pbmm2 align ==="
    echo "Started: $(date '+%H:%M:%S')"

    # Index reference if needed
    if [ ! -f "${REF}.fai" ]; then
        samtools faidx "$REF"
    fi

    pbmm2 align "$REF" "$BYSTRAND" "$ALIGNED" \
        --preset CCS --sort --num-threads 8

    RC=$?
    if [ $RC -ne 0 ] || [ ! -f "$ALIGNED" ]; then
        echo "FAILED: pbmm2 align exit=$RC"
        exit 1
    fi
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 3: Index (samtools + pbindex) ----
echo "=== Step 3: Index BAM ==="

if [ ! -f "${ALIGNED}.bai" ]; then
    samtools index "$ALIGNED"
fi

if [ ! -f "${ALIGNED}.pbi" ]; then
    apptainer exec --bind /data "$SMRT_SIF" \
        pbindex "$ALIGNED"
fi

# Alignment stats
echo "Alignment stats:"
samtools flagstat "$ALIGNED" | head -5
echo ""

# ---- Step 4: ipdSummary (SP3-C3 model) ----
if [ -f "$GFF" ] && [ -s "$GFF" ]; then
    echo "=== Step 4: SKIP — GFF already exists ==="
    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  Existing GFF: $GFF_LINES data lines"
else
    echo "=== Step 4: ipdSummary (SP3-C3) ==="
    echo "Started: $(date '+%H:%M:%S')"

    ipdSummary "$ALIGNED" \
        --reference "$REF" \
        --identify m6A,m4C \
        --methylFraction \
        --csv "$CSV" \
        --gff "$GFF" \
        --numWorkers 8 \
        --ipdModel "$SP3_MODEL"

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "FAILED: ipdSummary exit=$RC"
        exit 1
    fi

    if [ ! -f "$GFF" ] || [ ! -s "$GFF" ]; then
        echo "WARNING: GFF file empty or missing"
        exit 1
    fi

    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  GFF records: $GFF_LINES"
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================================"
echo "  COMPLETE: $SAMPLE_ID"
echo "========================================================"
echo "  Bystrandify: $BYSTRAND"
echo "  Aligned:     $ALIGNED"
echo "  GFF:         $GFF"
echo "  CSV:         $CSV"
echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
