#!/bin/bash
#SBATCH --job-name=vega_asm
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_assembly_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega HMB — Per-barcode assembly + align + ipdSummary
#
# Full pipeline per species:
#   1. hifiasm              → draft assembly (.fasta)
#   2. pbmm2 align          → aligned BAM
#   3. samtools + pbindex    → index
#   4. ipdSummary SP3-C3    → GFF methylation calls
#
# Usage:
#   sbatch --array=1-16%5 slurm_kinsim/vega_01_assembly_pipeline.slurm
#
# Prereqs:
#   - hifiasm in conda (kinsim_env)
#   - pbmm2, samtools in conda
#   - apptainer with pacbio-smrt-tools-25.3.sif (for pbindex)
#   - SMRT-Link/12.0.0 module (for ipdSummary)
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true
set -euo pipefail

# ---- Config ----
VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
OUTBASE=${VEGA}/assembly
SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

# All 16 Vega barcodes (bc2033-bc2048)
BARCODES=(2033 2034 2035 2036 2037 2038 2039 2040
          2041 2042 2043 2044 2045 2046 2047 2048)

# ---- Select barcode for this task ----
IDX=$((SLURM_ARRAY_TASK_ID - 1))
BC=${BARCODES[$IDX]}
BAM="${VEGA}/m21026_260313_002345.hifi_reads.bc${BC}.bam"
SAMPLE="bc${BC}"

SPECIES_DIR="${OUTBASE}/${SAMPLE}"
mkdir -p "$SPECIES_DIR"

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  Vega Assembly Pipeline — $SAMPLE (task $SLURM_ARRAY_TASK_ID)"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "BAM:        $BAM"
echo "Output:     $SPECIES_DIR/"
echo "========================================================"
echo ""

if [ ! -f "$BAM" ]; then
    echo "ERROR: BAM not found: $BAM"
    exit 1
fi

# ---- Step 1: hifiasm assembly ----
ASM_PREFIX="${SPECIES_DIR}/${SAMPLE}"
ASM_FASTA="${SPECIES_DIR}/${SAMPLE}_assembly.fasta"

if [ -f "$ASM_FASTA" ] && [ -s "$ASM_FASTA" ]; then
    echo "=== Step 1: SKIP — assembly already exists ==="
    echo "  $(grep -c '^>' "$ASM_FASTA") contigs"
else
    echo "=== Step 1: hifiasm ==="
    echo "Started: $(date '+%H:%M:%S')"

    # Extract FASTQ for hifiasm
    FASTQ="${SPECIES_DIR}/${SAMPLE}.fastq"
    samtools fastq "$BAM" > "$FASTQ"

    hifiasm -o "$ASM_PREFIX" -t "${SLURM_CPUS_PER_TASK}" "$FASTQ"

    # Convert primary GFA to FASTA
    GFA="${ASM_PREFIX}.bp.p_ctg.gfa"
    if [ ! -f "$GFA" ]; then
        echo "ERROR: hifiasm GFA not found: $GFA"
        ls -la "${ASM_PREFIX}".*
        exit 1
    fi

    awk '/^S/{print ">"$2; print $3}' "$GFA" > "$ASM_FASTA"

    N_CONTIGS=$(grep -c '^>' "$ASM_FASTA")
    ASM_SIZE=$(grep -v '^>' "$ASM_FASTA" | wc -c)
    echo "  Contigs: $N_CONTIGS"
    echo "  Total size: $((ASM_SIZE / 1000000)) Mb"

    # Cleanup intermediate files (keep GFA + FASTA)
    rm -f "$FASTQ"
    rm -f "${ASM_PREFIX}".bp.hap*.gfa
    rm -f "${ASM_PREFIX}".bp.r_utg.gfa
    rm -f "${ASM_PREFIX}"*.bin
    rm -f "${ASM_PREFIX}"*.ec.fa

    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 2: ccs-kinetics-bystrandify ----
BYSTRAND="${SPECIES_DIR}/${SAMPLE}_bystrandify.bam"

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

# ---- Step 3: pbmm2 align ----
ALIGNED="${SPECIES_DIR}/${SAMPLE}_aligned.bam"

if [ -f "$ALIGNED" ] && [ -s "$ALIGNED" ]; then
    echo "=== Step 3: SKIP — aligned BAM already exists ==="
else
    echo "=== Step 3: pbmm2 align ==="
    echo "Started: $(date '+%H:%M:%S')"

    # Index reference
    if [ ! -f "${ASM_FASTA}.fai" ]; then
        samtools faidx "$ASM_FASTA"
    fi

    pbmm2 align "$ASM_FASTA" "$BYSTRAND" "$ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"

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
    apptainer exec --bind /data "$SMRT_SIF" \
        pbindex "$ALIGNED"
fi

echo "Alignment stats:"
samtools flagstat "$ALIGNED" | head -5
echo ""

# ---- Step 5: ipdSummary ----
GFF="${SPECIES_DIR}/${SAMPLE}_ipdSummary.gff"
CSV="${SPECIES_DIR}/${SAMPLE}_ipdSummary.csv"

if [ -f "$GFF" ] && [ -s "$GFF" ]; then
    echo "=== Step 5: SKIP — GFF already exists ==="
    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  Existing GFF: $GFF_LINES records"
else
    echo "=== Step 5: ipdSummary (SP3-C3) ==="
    echo "Started: $(date '+%H:%M:%S')"

    ipdSummary "$ALIGNED" \
        --reference "$ASM_FASTA" \
        --identify m6A,m4C \
        --methylFraction \
        --csv "$CSV" \
        --gff "$GFF" \
        --numWorkers "${SLURM_CPUS_PER_TASK}" \
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
echo "  COMPLETE: $SAMPLE"
echo "========================================================"
echo "  Assembly:    $ASM_FASTA"
echo "  Aligned:     $ALIGNED"
echo "  GFF:         $GFF"
echo "  CSV:         $CSV"
echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
