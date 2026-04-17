#!/bin/bash
#SBATCH --job-name=sequel_ipd
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_ipd_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel II 48-plex — ipdSummary on raw SUBREADS (not CCS)
#
# Subreads have native per-pass ip/pw tags → ipdSummary gets
# maximum statistical power for methylation detection.
#
# Pipeline per barcode:
#   1. Decompress subreads (.bam.gz → .bam)
#   2. pbmm2 align subreads to PacBio-provided assembly
#   3. pbindex + samtools index
#   4. ipdSummary → GFF methylation calls
#   5. Cleanup decompressed .bam (keep .gz)
#
# Assemblies: pre-built by PacBio, downloaded by sequel_00b_download_meta.slurm
# Subreads:   downloaded by sequel_00_download.slurm (lima.bcXXXX--bcXXXX.bam.gz)
#
# Usage:
#   sbatch --array=1-48%5 slurm_kinsim/sequel_02_ipdsummary.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

# ---- Config ----
SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
ASM_DIR=${SEQUEL}/assemblies
OUTBASE=${SEQUEL}/ipdsummary
SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"

mkdir -p "$OUTBASE"

# ---- Build barcode list from available .bam.gz files ----
mapfile -t GZ_BAMS < <(ls "${SEQUEL}"/lima.bc*--bc*.bam.gz 2>/dev/null | sort)
N_BAMS=${#GZ_BAMS[@]}

if [ "$N_BAMS" -eq 0 ]; then
    echo "ERROR: No lima.bc*--bc*.bam.gz files found in $SEQUEL"
    exit 1
fi

IDX=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$IDX" -ge "$N_BAMS" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID > $N_BAMS BAMs. Nothing to do."
    exit 0
fi

GZ_BAM="${GZ_BAMS[$IDX]}"
# Extract barcode: lima.bc1018--bc1018.bam.gz → bc1018
FNAME=$(basename "$GZ_BAM" .bam.gz)
BC=$(echo "$FNAME" | grep -oP 'bc\d+' | head -1)
SUBREADS_BAM="${SEQUEL}/${FNAME}.bam"

# Assembly from PacBio download
ASM_FASTA="${ASM_DIR}/${BC}_assembly.fsa"

SPECIES_DIR="${OUTBASE}/${BC}"
mkdir -p "$SPECIES_DIR"

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  Sequel II ipdSummary — $BC (task $SLURM_ARRAY_TASK_ID)"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "Subreads:   $GZ_BAM"
echo "Assembly:   $ASM_FASTA"
echo "Output:     $SPECIES_DIR/"
echo "========================================================"
echo ""

if [ ! -f "$ASM_FASTA" ]; then
    echo "ERROR: Assembly not found: $ASM_FASTA"
    echo "Run sequel_00b_download_meta.slurm first."
    exit 1
fi

# ---- Step 1: Decompress subreads ----
if [ -f "$SUBREADS_BAM" ]; then
    echo "=== Step 1: SKIP — subreads BAM already decompressed ==="
else
    echo "=== Step 1: Decompress subreads ==="
    echo "Started: $(date '+%H:%M:%S')"
    gunzip -k "$GZ_BAM"
    echo "  $(ls -lh "$SUBREADS_BAM" | awk '{print $5}')"
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 2: pbmm2 align subreads to assembly ----
ALIGNED="${SPECIES_DIR}/${BC}_subreads_aligned.bam"

if [ -f "$ALIGNED" ] && [ -s "$ALIGNED" ]; then
    echo "=== Step 2: SKIP — aligned subreads already exist ==="
else
    echo "=== Step 2: pbmm2 align (subreads) ==="
    echo "Started: $(date '+%H:%M:%S')"

    if [ ! -f "${ASM_FASTA}.fai" ]; then
        samtools faidx "$ASM_FASTA"
    fi

    pbmm2 align "$ASM_FASTA" "$SUBREADS_BAM" "$ALIGNED" \
        --preset SUBREAD --sort --num-threads "${SLURM_CPUS_PER_TASK}"

    RC=$?
    if [ $RC -ne 0 ] || [ ! -f "$ALIGNED" ]; then
        echo "FAILED: pbmm2 align exit=$RC"
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

# ---- Step 4: ipdSummary ----
GFF="${SPECIES_DIR}/${BC}_ipdSummary.gff"
CSV="${SPECIES_DIR}/${BC}_ipdSummary.csv"

if [ -f "$GFF" ] && [ -s "$GFF" ]; then
    echo "=== Step 4: SKIP — GFF already exists ==="
    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  Existing GFF: $GFF_LINES records"
else
    echo "=== Step 4: ipdSummary (on subreads) ==="
    echo "Started: $(date '+%H:%M:%S')"

    apptainer exec --bind /data "$SMRT_SIF" \
        ipdSummary "$ALIGNED" \
            --reference "$ASM_FASTA" \
            --identify m6A,m4C \
            --methylFraction \
            --csv "$CSV" \
            --gff "$GFF" \
            --numWorkers "${SLURM_CPUS_PER_TASK}"

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "FAILED: ipdSummary exit=$RC"
        exit 1
    fi

    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  GFF records: $GFF_LINES"
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 5: Cleanup decompressed subreads (keep .gz) ----
if [ -f "$GFF" ] && [ -s "$GFF" ] && [ -f "$GZ_BAM" ]; then
    echo "=== Step 5: Cleanup ==="
    rm -f "$SUBREADS_BAM"
    echo "Removed decompressed subreads (keeping .gz)"
fi
echo ""

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================================"
echo "  COMPLETE: $BC"
echo "========================================================"
echo "  Aligned subreads: $ALIGNED"
echo "  GFF:              $GFF"
echo "  CSV:              $CSV"
echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
