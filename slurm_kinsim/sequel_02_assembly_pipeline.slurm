#!/bin/bash
#SBATCH --job-name=sequel_asm
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_assembly_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel II 48-plex — Per-barcode assembly + align + ipdSummary
#
# Full pipeline per species:
#   1. hifiasm              → draft assembly (.fasta)
#   2. pbmm2 align          → aligned BAM (HiFi to own assembly)
#   3. samtools + pbindex    → index
#   4. ipdSummary            → GFF methylation calls
#
# NOTE: ipdSummary on HiFi reads (averaged kinetics) is less
#       sensitive than on subreads. We also run ipdSummary on the
#       raw subreads separately (sequel_03_subreads_ipd.slurm)
#       for better methylation calling. The HiFi GFF serves as
#       a baseline / cross-check.
#
# Usage:
#   # Count HiFi BAMs:
#   N=$(ls /data/.../Sequel/hifi/*.hifi.bam | wc -l)
#   sbatch --array=1-${N}%5 slurm_kinsim/sequel_02_assembly_pipeline.slurm
#
# Prereqs:
#   - HiFi BAMs from sequel_01_ccs.slurm in Sequel/hifi/
#   - hifiasm, pbmm2, samtools in conda (kinsim_env)
#   - apptainer with pacbio-smrt-tools-25.3.sif (for pbindex)
#   - SMRT-Link module (for ipdSummary)
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

# ---- Config ----
SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
HIFI_DIR=${SEQUEL}/hifi
OUTBASE=${SEQUEL}/assembly
SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"
# All PacBio tools (pbindex, ipdSummary) run via apptainer
# hifiasm, pbmm2, samtools run from conda

# ---- Select BAM for this task ----
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

BAM="${HIFI_BAMS[$IDX]}"
SAMPLE=$(basename "$BAM" .hifi.bam)
SPECIES_DIR="${OUTBASE}/${SAMPLE}"
mkdir -p "$SPECIES_DIR"

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  Sequel II Assembly Pipeline — $SAMPLE (task $SLURM_ARRAY_TASK_ID)"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "HiFi BAM:   $BAM"
echo "Output:     $SPECIES_DIR/"
echo "========================================================"
echo ""

# ---- Step 1: hifiasm ----
ASM_PREFIX="${SPECIES_DIR}/${SAMPLE}"
ASM_FASTA="${SPECIES_DIR}/${SAMPLE}_assembly.fasta"

if [ -f "$ASM_FASTA" ] && [ -s "$ASM_FASTA" ]; then
    echo "=== Step 1: SKIP — assembly already exists ==="
    echo "  $(grep -c '^>' "$ASM_FASTA") contigs"
else
    echo "=== Step 1: hifiasm ==="
    echo "Started: $(date '+%H:%M:%S')"

    FASTQ="${SPECIES_DIR}/${SAMPLE}.fastq"
    samtools fastq "$BAM" > "$FASTQ"

    hifiasm -o "$ASM_PREFIX" -t "${SLURM_CPUS_PER_TASK}" "$FASTQ"

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

    # Cleanup
    rm -f "$FASTQ"
    rm -f "${ASM_PREFIX}".bp.hap*.gfa
    rm -f "${ASM_PREFIX}".bp.r_utg.gfa
    rm -f "${ASM_PREFIX}"*.bin
    rm -f "${ASM_PREFIX}"*.ec.fa

    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 2: pbmm2 align (HiFi reads to own assembly) ----
ALIGNED="${SPECIES_DIR}/${SAMPLE}_aligned.bam"

if [ -f "$ALIGNED" ] && [ -s "$ALIGNED" ]; then
    echo "=== Step 2: SKIP — aligned BAM already exists ==="
else
    echo "=== Step 2: pbmm2 align ==="
    echo "Started: $(date '+%H:%M:%S')"

    if [ ! -f "${ASM_FASTA}.fai" ]; then
        samtools faidx "$ASM_FASTA"
    fi

    pbmm2 align "$ASM_FASTA" "$BAM" "$ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"

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
GFF="${SPECIES_DIR}/${SAMPLE}_ipdSummary.gff"
CSV="${SPECIES_DIR}/${SAMPLE}_ipdSummary.csv"

if [ -f "$GFF" ] && [ -s "$GFF" ]; then
    echo "=== Step 4: SKIP — GFF already exists ==="
    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  Existing GFF: $GFF_LINES records"
else
    echo "=== Step 4: ipdSummary ==="
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

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================================"
echo "  COMPLETE: $SAMPLE"
echo "========================================================"
echo "  Assembly:    $ASM_FASTA"
echo "  Aligned:     $ALIGNED"
echo "  GFF:         $GFF"
echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
