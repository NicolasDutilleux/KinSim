#!/bin/bash
#SBATCH --job-name=msa_smrt251
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --exclude=binfservas18
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/01c_smrtlink251_%j.log
#SBATCH --error=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/01c_smrtlink251_%j.err

# STEP 01c: Run ipdSummary + motifMaker via SMRT Link 25.1 (M. Falquet install)
#
# Uses pbcromwell pb_basemods workflow per species.
# This is an alternative to 01_ipdsummary.slurm + 02_pbmotifmaker.slurm
# to verify results are identical.
#
# Prerequisites: step 00b (ip/pw tags + per-species ref)

set -uxo pipefail

export SMRT_ROOT="/data/users/lfalquet/SOFTS/SMRTlink/smrtlink251R"
export TMPDIR="$HOME/DATA/smrttempdir"
mkdir -p "$TMPDIR"

OUTDIR="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/species_aligned"

date && hostname
echo "========================================"
echo "=== STEP 01c: SMRT Link 25.1 per species ==="
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
    SMRT_OUT="${SPECIES_DIR}/smrtlink251"

    if [ ! -f "$BAM" ]; then
        echo "SKIP $CONTIG — no _ipd.bam (run step 00b first)"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi

    if [ ! -f "$REF" ]; then
        echo "SKIP $CONTIG — no per-species reference (run step 00b first)"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi

    # Skip if already done
    if [ -f "${SMRT_OUT}/motifs.csv" ]; then
        echo "SKIP $CONTIG — SMRT Link output already exists"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi

    READS=$(samtools view -c "$BAM")
    echo "========================================"
    echo "=== $CONTIG ($READS reads) ==="
    echo "=== $(date +%H:%M:%S) ==="
    START=$(date +%s)

    mkdir -p "$SMRT_OUT"
    cd "$SMRT_OUT"

    # Create dataset XML
    XML="${CONTIG}.subreadset.xml"
    if [ ! -f "$XML" ]; then
        $SMRT_ROOT/smrtcmds/bin/dataset create \
            --type ConsensusReadSet --name "$CONTIG" \
            "$XML" "$BAM"
    fi

    # Create reference dataset
    REF_XML="${CONTIG}.referenceset.xml"
    if [ ! -f "$REF_XML" ]; then
        $SMRT_ROOT/smrtcmds/bin/dataset create \
            --type ReferenceSet --name "$CONTIG" \
            "$REF_XML" "$REF"
    fi

    # Run base modification detection
    $SMRT_ROOT/smrtcmds/bin/pbcromwell run pb_basemods \
        -e "$XML" \
        -e "$REF_XML" \
        --task-option kineticstools_use_chemistry="SP3-C3" \
        --task-option kineticstools_identify_mods="m4C,m6A" \
        --task-option tmp_dir="$TMPDIR" \
        --overwrite

    RC=$?

    if [ $RC -eq 0 ]; then
        # Find and copy results
        MOTIFS_FOUND=$(find . -name motifs.csv -not -path "./motifs.csv" | head -1)
        GFF_FOUND=$(find . -name basemods.gff | head -1)

        if [ -n "$MOTIFS_FOUND" ]; then
            cp "$MOTIFS_FOUND" ./motifs.csv
        fi
        if [ -n "$GFF_FOUND" ]; then
            cp "$GFF_FOUND" ./basemods.gff
        fi

        END=$(date +%s)
        ELAPSED=$(( (END - START) / 60 ))
        echo "  OK (${ELAPSED}min)"

        if [ -f ./motifs.csv ]; then
            echo "  --- motifs ---"
            cat ./motifs.csv
            echo "  ---"
        fi
        TOTAL_OK=$((TOTAL_OK + 1))
    else
        END=$(date +%s)
        ELAPSED=$(( (END - START) / 60 ))
        echo "  FAILED exit=$RC (${ELAPSED}min)"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi

    cd "$OUTDIR"
    echo ""
done

echo "========================================"
echo "=== SUMMARY ==="
echo "========================================"
echo "OK     : $TOTAL_OK"
echo "Failed : $TOTAL_FAIL"
echo "Skipped: $TOTAL_SKIP"

echo ""
echo "=== ALL MOTIFS (SMRT Link 25.1) ==="
for f in "${OUTDIR}"/*/smrtlink251/motifs.csv; do
    [ -f "$f" ] || continue
    CONTIG=$(basename "$(dirname "$(dirname "$f")")")
    N=$(tail -n +2 "$f" | wc -l)
    printf "  %-20s %d motifs\n" "$CONTIG" "$N"
done

date
echo "=== Done ==="
