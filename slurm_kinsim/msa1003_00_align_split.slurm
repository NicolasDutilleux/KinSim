#!/bin/bash
#SBATCH --job-name=msa_align
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/00_align_split_%j.log
#SBATCH --error=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/00_align_split_%j.err

# STEP 00: Align MSA1003.bam (all 4 SMRT cells) to competitive ref + split by species
# No split by experience — all reads together for maximum coverage per species

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kinsim_env

BAM="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/MSA1003.bam"
REF="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/Genomes/MSA1003_competitive_ref.fasta"
OUTDIR="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/species_aligned"
ALIGNED="${OUTDIR}/MSA1003_aligned.bam"

mkdir -p "$OUTDIR" /data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs

echo "========================================"
echo "=== STEP 00: ALIGN + SPLIT BY SPECIES ==="
echo "=== $(date) ==="
echo "========================================"
echo "BAM: $BAM"
echo "REF: $REF"
echo "OUT: $OUTDIR"
echo ""

# --- Index reference if needed ---
if [ ! -f "${REF}.fai" ]; then
    echo "Indexing reference..."
    samtools faidx "$REF"
fi

# --- Align with pbmm2 ---
if [ ! -f "$ALIGNED" ]; then
    echo "=== Aligning with pbmm2 (all 4 SMRT cells together) ==="
    echo "Started: $(date +%H:%M:%S)"

    pbmm2 align \
        --preset CCS \
        --sort \
        --num-threads 8 \
        --log-level INFO \
        "$REF" "$BAM" "$ALIGNED" \
        2>"${OUTDIR}/pbmm2_align.log"

    if [ $? -ne 0 ] || [ ! -f "$ALIGNED" ]; then
        echo "FAILED: pbmm2 alignment"
        exit 1
    fi

    samtools index "$ALIGNED"
    echo "Alignment done: $(date +%H:%M:%S)"
else
    echo "Alignment already exists, skipping."
fi

echo ""
echo "=== Alignment stats ==="
samtools flagstat "$ALIGNED"
echo ""

# --- Split by species (contig) ---
echo "=== Splitting by species ==="
samtools idxstats "$ALIGNED" | while read CONTIG LENGTH MAPPED UNMAPPED; do
    [ "$CONTIG" = "*" ] && continue
    [ "$MAPPED" -eq 0 ] && continue

    SPECIES_DIR="${OUTDIR}/${CONTIG}"
    mkdir -p "$SPECIES_DIR"
    OUT_BAM="${SPECIES_DIR}/${CONTIG}.bam"

    if [ ! -f "$OUT_BAM" ]; then
        samtools view -b "$ALIGNED" "$CONTIG" > "$OUT_BAM"
        samtools index "$OUT_BAM"
        echo "  $CONTIG: $MAPPED reads"
    else
        echo "  $CONTIG: already exists ($MAPPED reads)"
    fi
done

echo ""
echo "=== SUMMARY: reads per species ==="
echo "----------------------------------------"
printf "%-20s %10s %10s\n" "CONTIG" "MAPPED" "LENGTH"
echo "----------------------------------------"
samtools idxstats "$ALIGNED" | while read CONTIG LENGTH MAPPED UNMAPPED; do
    [ "$CONTIG" = "*" ] && continue
    [ "$MAPPED" -eq 0 ] && continue
    printf "%-20s %10d %10d\n" "$CONTIG" "$MAPPED" "$LENGTH"
done
echo "----------------------------------------"

echo ""
echo "=== DONE - $(date) ==="
