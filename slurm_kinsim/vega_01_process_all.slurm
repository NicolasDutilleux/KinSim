#!/bin/bash
#SBATCH --job-name=vega_ipd
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --exclude=binfservas18
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_ipd_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega HMB pipeline — Per-species: assemble ref + bystrandify + align + ipdSummary
#
# Array job: one task per barcode (1-16, mapping to bc2033-bc2048).
#
# Since these are unaligned HiFi reads without a reference, we first
# need a reference. Two options:
#   A) De novo assembly with hifiasm (if no reference available)
#   B) Download reference genomes (if species are known from BLAST)
#
# This script uses approach A (hifiasm) as default, with fallback to
# a provided reference directory.
#
# Steps per species:
#   1. hifiasm de novo assembly (or use existing ref)
#   2. ccs-kinetics-bystrandify
#   3. pbmm2 align to assembly
#   4. samtools + pbindex
#   5. ipdSummary SP3-C3
#
# Usage:
#   sbatch --array=1-16%4 vega_01_process_all.slurm [ref_dir]
#
# Arguments:
#   $1  (optional) Directory with per-species references:
#       <ref_dir>/bc20XX/reference.fasta
#       If provided, skips assembly.
# ============================================================

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: Must be submitted as array job (--array=1-16)"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env
source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true

# ---- Config ----
VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
OUTDIR=${VEGA}/processed
SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"
REF_DIR="${1:-}"

# Map task ID to barcode
BARCODES=(bc2033 bc2034 bc2035 bc2036 bc2037 bc2038 bc2039 bc2040 bc2041 bc2042 bc2043 bc2044 bc2045 bc2046 bc2047 bc2048)
IDX=$((SLURM_ARRAY_TASK_ID - 1))

if [ $IDX -ge ${#BARCODES[@]} ]; then
    echo "Task $SLURM_ARRAY_TASK_ID exceeds barcode count (${#BARCODES[@]})"
    exit 0
fi

BC=${BARCODES[$IDX]}
BAM=${VEGA}/m21026_260313_002345.hifi_reads.${BC}.bam
SPECIES_DIR=${OUTDIR}/${BC}

mkdir -p "$SPECIES_DIR"

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  Vega HMB Pipeline — Task $SLURM_ARRAY_TASK_ID ($BC)"
echo "========================================================"
echo "Date:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:    $(hostname)"
echo "BAM:     $BAM"
echo "Output:  $SPECIES_DIR"
echo "========================================================"
echo ""

if [ ! -f "$BAM" ]; then
    echo "ERROR: BAM not found: $BAM"
    exit 1
fi

# ---- Step 1: Reference assembly ----
REF=${SPECIES_DIR}/${BC}_assembly.fasta

if [ -n "$REF_DIR" ] && [ -f "${REF_DIR}/${BC}/reference.fasta" ]; then
    # Use provided reference
    REF="${REF_DIR}/${BC}/reference.fasta"
    echo "=== Step 1: Using provided reference ==="
    echo "  $REF"
elif [ -f "$REF" ] && [ -s "$REF" ]; then
    echo "=== Step 1: SKIP — assembly already exists ==="
else
    echo "=== Step 1: hifiasm de novo assembly ==="
    echo "Started: $(date '+%H:%M:%S')"

    # Extract FASTQ from BAM for hifiasm
    FQ=${SPECIES_DIR}/${BC}.fastq.gz
    if [ ! -f "$FQ" ]; then
        samtools fastq "$BAM" | gzip > "$FQ"
    fi

    # Run hifiasm
    if command -v hifiasm &>/dev/null; then
        hifiasm -o "${SPECIES_DIR}/${BC}_asm" -t 8 "$FQ"

        # Convert GFA to FASTA (primary contigs)
        GFA="${SPECIES_DIR}/${BC}_asm.bp.p_ctg.gfa"
        if [ -f "$GFA" ]; then
            awk '/^S/{print ">"$2; print $3}' "$GFA" > "$REF"
            echo "  Assembly: $(grep -c '^>' "$REF") contigs"
        else
            echo "FAILED: hifiasm output not found"
            exit 1
        fi

        # Cleanup intermediate files
        rm -f "${SPECIES_DIR}/${BC}_asm".*.bin
    else
        echo "ERROR: hifiasm not found. Install with: conda install -c bioconda hifiasm"
        echo "Or provide references via: sbatch ... vega_01_process_all.slurm /path/to/refs/"
        exit 1
    fi
    echo "Done: $(date '+%H:%M:%S')"
fi

# Index reference
if [ ! -f "${REF}.fai" ]; then
    samtools faidx "$REF"
fi
echo ""

# ---- Step 2: Bystrandify ----
BYSTRAND=${SPECIES_DIR}/${BC}_bystrandify.bam

if [ -f "$BYSTRAND" ] && [ -s "$BYSTRAND" ]; then
    echo "=== Step 2: SKIP — bystrandify BAM already exists ==="
else
    echo "=== Step 2: ccs-kinetics-bystrandify ==="
    echo "Started: $(date '+%H:%M:%S')"

    apptainer exec --bind /data "$SMRT_SIF" \
        ccs-kinetics-bystrandify "$BAM" "$BYSTRAND"

    RC=$?
    if [ $RC -ne 0 ] || [ ! -f "$BYSTRAND" ]; then
        echo "FAILED: bystrandify exit=$RC"
        exit 1
    fi
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 3: Align ----
ALIGNED=${SPECIES_DIR}/${BC}_aligned.bam

if [ -f "$ALIGNED" ] && [ -s "$ALIGNED" ]; then
    echo "=== Step 3: SKIP — aligned BAM already exists ==="
else
    echo "=== Step 3: pbmm2 align ==="
    echo "Started: $(date '+%H:%M:%S')"

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

# ---- Step 4: Index ----
echo "=== Step 4: Index BAM ==="
if [ ! -f "${ALIGNED}.bai" ]; then
    samtools index "$ALIGNED"
fi
if [ ! -f "${ALIGNED}.pbi" ]; then
    apptainer exec --bind /data "$SMRT_SIF" pbindex "$ALIGNED"
fi

samtools flagstat "$ALIGNED" | head -5
echo ""

# ---- Step 5: ipdSummary ----
GFF=${SPECIES_DIR}/${BC}_ipdSummary.gff
CSV=${SPECIES_DIR}/${BC}_ipdSummary.csv

if [ -f "$GFF" ] && [ -s "$GFF" ]; then
    echo "=== Step 5: SKIP — GFF already exists ==="
    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  GFF records: $GFF_LINES"
else
    echo "=== Step 5: ipdSummary (SP3-C3) ==="
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

    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  GFF records: $GFF_LINES"
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================================"
echo "  COMPLETE: $BC"
echo "========================================================"
echo "  Reference: $REF"
echo "  Bystrand:  $BYSTRAND"
echo "  Aligned:   $ALIGNED"
echo "  GFF:       $GFF"
echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
