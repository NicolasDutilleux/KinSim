#!/bin/bash
#SBATCH --job-name=msa_ipd
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --exclude=binfservas18
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/01_ipdsummary_%j.log
#SBATCH --error=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/01_ipdsummary_%j.err

# STEP 01: ipdSummary per species on prepared BAMs
# Uses apptainer pacbio-smrt-tools for ipdSummary + pbindex
#
# Usage:
#   sbatch 01_ipdsummary.slurm [SPECIES_DIR]
#   Default: species_bystrand (from 00_bystrandify.slurm)
#
# Prerequisites:
#   - <contig>_ipd.bam with ip/pw tags + single-contig @SQ header + .pbi
#   - <contig>.fasta + .fai per species

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kinsim_env

OUTDIR="${1:-/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/species_bystrand}"
SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"

echo "========================================"
echo "=== STEP 01: ipdSummary PER SPECIES ==="
echo "=== $(date) ==="
echo "========================================"
echo ""

TOTAL_OK=0
TOTAL_FAIL=0
TOTAL_SKIP=0

for SPECIES_DIR in "${OUTDIR}"/*/; do
    [ -d "$SPECIES_DIR" ] || continue
    CONTIG=$(basename "$SPECIES_DIR")
    BAM="${SPECIES_DIR}/${CONTIG}_ipd.bam"
    REF="${SPECIES_DIR}/${CONTIG}.fasta"
    CSV="${SPECIES_DIR}/${CONTIG}_ipdSummary.csv"
    GFF="${SPECIES_DIR}/${CONTIG}_ipdSummary.gff"

    # Verify prerequisites from step 00b
    if [ ! -f "$BAM" ]; then
        echo "FAIL $CONTIG: _ipd.bam not found (run step 00b first)"
        TOTAL_FAIL=$((TOTAL_FAIL+1))
        continue
    fi
    if [ ! -f "$REF" ]; then
        echo "FAIL $CONTIG: per-species reference not found (run step 00b first)"
        TOTAL_FAIL=$((TOTAL_FAIL+1))
        continue
    fi

    # Skip if already done
    if [ -f "$GFF" ] && [ -s "$GFF" ]; then
        echo "SKIP $CONTIG: ipdSummary already done"
        TOTAL_SKIP=$((TOTAL_SKIP+1))
        continue
    fi

    READS=$(samtools view -c "$BAM")
    echo "========================================"
    echo "=== $CONTIG ($READS reads) ==="
    echo "=== $(date +%H:%M:%S) ==="

    # pbindex the BAM (required by ipdSummary)
    if [ ! -f "${BAM}.pbi" ]; then
        echo "  Creating .pbi index..."
        apptainer exec --bind /data "$SMRT_SIF" pbindex "$BAM"
    fi

    # Run ipdSummary
    echo "  Running ipdSummary..."
    apptainer exec --bind /data "$SMRT_SIF" \
        ipdSummary "$BAM" \
            --reference "$REF" \
            --identify m6A,m4C \
            --methylFraction \
            --csv "$CSV" \
            --gff "$GFF" \
            --numWorkers 8

    RC=$?
    if [ $RC -eq 0 ] && [ -f "$GFF" ] && [ -s "$GFF" ]; then
        GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
        echo "  OK: $GFF_LINES modifications detected"
        TOTAL_OK=$((TOTAL_OK+1))
    else
        echo "  FAILED: ipdSummary exit=$RC"
        TOTAL_FAIL=$((TOTAL_FAIL+1))
    fi
    echo ""
done

echo "========================================"
echo "=== SUMMARY ==="
echo "========================================"
echo "OK     : $TOTAL_OK"
echo "Failed : $TOTAL_FAIL"
echo "Skipped: $TOTAL_SKIP"
echo ""
echo "=== DONE - $(date) ==="
