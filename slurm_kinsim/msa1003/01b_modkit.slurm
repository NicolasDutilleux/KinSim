#!/bin/bash
#SBATCH --job-name=msa_modkit
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --exclude=binfservas18
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/01b_modkit_%j.log
#SBATCH --error=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/01b_modkit_%j.err

# STEP 01b: modkit pileup per species — m5C detection from Jasmine MM/ML tags
# Runs on the same aligned split BAMs as step 01 (ipdSummary)
# No TET control needed — uses per-read ML probabilities from Jasmine

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kinsim_env

OUTDIR="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/species_aligned"
REF="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/Genomes/MSA1003_competitive_ref.fasta"

echo "========================================"
echo "=== STEP 01b: modkit pileup PER SPECIES ==="
echo "=== $(date) ==="
echo "=== Host: $(hostname) ==="
echo "========================================"
echo ""

# Check modkit is available
if ! command -v modkit &> /dev/null; then
    echo "ERROR: modkit not found. Install with: conda install -c nanoporetech modkit"
    echo "   or: pip install modkit"
    echo "   or download from: https://github.com/nanoporetech/modkit/releases"
    exit 1
fi
echo "modkit version: $(modkit --version)"
echo ""

TOTAL_OK=0
TOTAL_FAIL=0
TOTAL_SKIP=0

for SPECIES_DIR in "${OUTDIR}"/*/; do
    [ -d "$SPECIES_DIR" ] || continue
    CONTIG=$(basename "$SPECIES_DIR")

    BAM="${SPECIES_DIR}/${CONTIG}.bam"
    BED_OUT="${SPECIES_DIR}/${CONTIG}_modkit_pileup.bed"
    SUMMARY_OUT="${SPECIES_DIR}/${CONTIG}_modkit_summary.txt"

    if [ ! -f "$BAM" ]; then
        echo "SKIP $CONTIG — no BAM found"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi

    # Check if BAM has MM/ML tags
    HAS_MM=$(samtools view "$BAM" | head -5 | tr '\t' '\n' | grep -c '^MM:' || true)
    if [ "$HAS_MM" -eq 0 ]; then
        echo "SKIP $CONTIG — no MM/ML tags in BAM"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi

    # Ensure index exists
    if [ ! -f "${BAM}.bai" ]; then
        echo "  Indexing $CONTIG..."
        samtools index "$BAM"
    fi

    echo "--- $CONTIG ---"
    echo "  BAM:    $BAM"
    echo "  Output: $BED_OUT"

    START=$(date +%s)

    # modkit pileup: aggregate per-read MM/ML into per-site methylation
    modkit pileup "$BAM" "$BED_OUT" \
        --ref "$REF" \
        --threads 8 \
        --filter-threshold 0.5 \
        --log-filepath "${SPECIES_DIR}/${CONTIG}_modkit.log" \
        2>&1

    RC=$?
    END=$(date +%s)
    ELAPSED=$((END - START))

    if [ $RC -ne 0 ]; then
        echo "  FAILED (exit $RC) after ${ELAPSED}s"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        continue
    fi

    # Count sites
    if [ -f "$BED_OUT" ]; then
        N_SITES=$(wc -l < "$BED_OUT")
        N_5MC=$(grep -c "m" "$BED_OUT" || true)
        echo "  OK: ${N_SITES} sites total, ${ELAPSED}s"
    else
        echo "  WARNING: no output BED produced"
        N_SITES=0
    fi

    # modkit summary for quick overview
    modkit summary "$BAM" \
        --threads 8 \
        > "$SUMMARY_OUT" 2>&1 || true

    TOTAL_OK=$((TOTAL_OK + 1))
    echo ""
done

echo "========================================"
echo "=== STEP 01b COMPLETE ==="
echo "=== $(date) ==="
echo "=== OK: $TOTAL_OK  FAIL: $TOTAL_FAIL  SKIP: $TOTAL_SKIP ==="
echo "========================================"
echo ""

# Quick summary across species
echo "=== Per-species m5C site counts ==="
for SPECIES_DIR in "${OUTDIR}"/*/; do
    [ -d "$SPECIES_DIR" ] || continue
    CONTIG=$(basename "$SPECIES_DIR")
    BED="${SPECIES_DIR}/${CONTIG}_modkit_pileup.bed"
    if [ -f "$BED" ]; then
        N=$(wc -l < "$BED")
        printf "  %-40s %8d sites\n" "$CONTIG" "$N"
    fi
done
echo ""
echo "Done."
