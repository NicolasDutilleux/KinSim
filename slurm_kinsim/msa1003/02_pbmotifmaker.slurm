#!/bin/bash
#SBATCH --job-name=msa_motifs
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --exclude=binfservas18
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/02_pbmotifmaker_%j.log
#SBATCH --error=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/02_pbmotifmaker_%j.err

# STEP 02: pbmotifmaker per species on ipdSummary GFF
# Discovers methylation motifs from kinetic data

OUTDIR="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/species_aligned"
REF="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/Genomes/MSA1003_competitive_ref.fasta"
SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"

echo "========================================"
echo "=== STEP 02: pbmotifmaker PER SPECIES ==="
echo "=== $(date) ==="
echo "========================================"
echo ""

TOTAL_OK=0
TOTAL_FAIL=0
TOTAL_SKIP=0

for SPECIES_DIR in "${OUTDIR}"/*/; do
    [ -d "$SPECIES_DIR" ] || continue
    CONTIG=$(basename "$SPECIES_DIR")
    GFF="${SPECIES_DIR}/${CONTIG}_ipdSummary.gff"
    OUT="${SPECIES_DIR}/${CONTIG}_motifs.csv"

    # Skip if no GFF
    if [ ! -f "$GFF" ] || [ ! -s "$GFF" ]; then
        echo "SKIP $CONTIG: no ipdSummary GFF"
        TOTAL_SKIP=$((TOTAL_SKIP+1))
        continue
    fi

    # Skip if already done
    if [ -f "$OUT" ] && [ -s "$OUT" ]; then
        echo "SKIP $CONTIG: motifs already done"
        TOTAL_SKIP=$((TOTAL_SKIP+1))
        continue
    fi

    echo "========================================"
    echo "=== $CONTIG ==="
    echo "=== $(date +%H:%M:%S) ==="

    apptainer exec --bind /data "$SMRT_SIF" \
        pbmotifmaker find "$REF" "$GFF" "$OUT"

    RC=$?
    if [ $RC -eq 0 ] && [ -f "$OUT" ]; then
        N_MOTIFS=$(tail -n +2 "$OUT" | wc -l)
        echo "  OK: $N_MOTIFS motifs discovered"
        echo "  ---"
        cat "$OUT"
        echo "  ---"
        TOTAL_OK=$((TOTAL_OK+1))
    else
        echo "  FAILED: pbmotifmaker exit=$RC"
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

echo "=== ALL MOTIFS ==="
for f in "${OUTDIR}"/*/*_motifs.csv; do
    [ -f "$f" ] || continue
    CONTIG=$(basename "$(dirname "$f")")
    N=$(tail -n +2 "$f" | wc -l)
    printf "  %-20s %d motifs\n" "$CONTIG" "$N"
done

echo ""
echo "=== DONE - $(date) ==="
