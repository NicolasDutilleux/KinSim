#!/bin/bash
#SBATCH --job-name=msa_bystrand
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --exclude=binfservas18
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/00_bystrandify_%j.log
#SBATCH --error=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/00_bystrandify_%j.err

# STEP 00: Bystrandify + align + split MSA1003 for ipdSummary
#
# Replaces 00_align_split + 00b_add_ippw with proper bystrandify approach.
# Proven to recover m6A on bc2036 Revio data.
#
# Steps:
#   1. ccs-kinetics-bystrandify — split CCS reads into 2 pseudo-subreads
#   2. pbmm2 align              — align pseudo-subreads to competitive ref
#   3. Split by species          — extract per-contig BAMs
#   4. Per-species: reheader + reference FASTA + pbindex

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kinsim_env

set -uxo pipefail

SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"

BAM="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/MSA1003.bam"
REF="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/Genomes/MSA1003_competitive_ref.fasta"
OUTDIR="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/species_bystrand"
BYSTRAND="${OUTDIR}/MSA1003_bystrandify.bam"
ALIGNED="${OUTDIR}/MSA1003_bystrandify_aligned.bam"

mkdir -p "$OUTDIR" /data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs

date && hostname
echo "========================================"
echo "=== STEP 00: BYSTRANDIFY + ALIGN + SPLIT ==="
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

# --- Step 1: Bystrandify ---
if [ ! -f "$BYSTRAND" ]; then
    echo "=== Step 1: ccs-kinetics-bystrandify ==="
    echo "Started: $(date +%H:%M:%S)"

    apptainer exec --bind /data "$SMRT_SIF" \
        ccs-kinetics-bystrandify "$BAM" "$BYSTRAND"

    RC=$?
    echo "bystrandify exit: $RC"
    if [ $RC -ne 0 ] || [ ! -f "$BYSTRAND" ]; then
        echo "FAILED: bystrandify"
        exit 1
    fi
    echo "Done: $(date +%H:%M:%S)"
else
    echo "Bystrandify BAM already exists, skipping."
fi

# --- Step 2: Align with pbmm2 ---
if [ ! -f "$ALIGNED" ]; then
    echo ""
    echo "=== Step 2: pbmm2 align ==="
    echo "Started: $(date +%H:%M:%S)"

    apptainer exec --bind /data "$SMRT_SIF" \
        pbmm2 align "$REF" "$BYSTRAND" "$ALIGNED" \
        --preset CCS --sort --num-threads 8

    RC=$?
    echo "pbmm2 exit: $RC"
    if [ $RC -ne 0 ] || [ ! -f "$ALIGNED" ]; then
        echo "FAILED: pbmm2 align"
        exit 1
    fi

    samtools index "$ALIGNED"
    echo "Done: $(date +%H:%M:%S)"
else
    echo "Aligned BAM already exists, skipping."
fi

echo ""
echo "=== Alignment stats ==="
samtools flagstat "$ALIGNED"
echo ""

# --- Step 3: Split by species ---
echo "=== Step 3: Split by species + prepare for ipdSummary ==="

TOTAL_OK=0
TOTAL_SKIP=0

samtools idxstats "$ALIGNED" | while read CONTIG LENGTH MAPPED UNMAPPED; do
    [ "$CONTIG" = "*" ] && continue
    [ "$MAPPED" -eq 0 ] && continue

    SPECIES_DIR="${OUTDIR}/${CONTIG}"
    mkdir -p "$SPECIES_DIR"

    OUT_BAM="${SPECIES_DIR}/${CONTIG}_ipd.bam"
    SINGLE_REF="${SPECIES_DIR}/${CONTIG}.fasta"

    # Skip if fully processed
    if [ -f "$OUT_BAM" ] && [ -f "${OUT_BAM}.pbi" ] && [ -f "$SINGLE_REF" ]; then
        echo "  $CONTIG: already done"
        continue
    fi

    echo "  $CONTIG: $MAPPED reads"

    # Extract reads for this contig
    samtools view -b "$ALIGNED" "$CONTIG" > "${SPECIES_DIR}/tmp_extract.bam"

    # Reheader to single contig (pbcore requires FASTA contigs = superset of BAM @SQ)
    samtools view -H "$ALIGNED" | grep -v '^@SQ' > "${SPECIES_DIR}/header.sam"
    samtools view -H "$ALIGNED" | grep "^@SQ.*SN:${CONTIG}	" >> "${SPECIES_DIR}/header.sam"
    samtools reheader "${SPECIES_DIR}/header.sam" "${SPECIES_DIR}/tmp_extract.bam" > "$OUT_BAM"
    rm -f "${SPECIES_DIR}/header.sam" "${SPECIES_DIR}/tmp_extract.bam"

    samtools index "$OUT_BAM"

    # Per-species reference FASTA
    samtools faidx "$REF" "$CONTIG" > "$SINGLE_REF"
    samtools faidx "$SINGLE_REF"

    # pbindex (required by ipdSummary)
    apptainer exec --bind /data "$SMRT_SIF" pbindex "$OUT_BAM"

done

echo ""
echo "=== SUMMARY: species prepared ==="
echo "----------------------------------------"
printf "%-20s %10s %10s\n" "CONTIG" "MAPPED" "STATUS"
echo "----------------------------------------"
samtools idxstats "$ALIGNED" | while read CONTIG LENGTH MAPPED UNMAPPED; do
    [ "$CONTIG" = "*" ] && continue
    [ "$MAPPED" -eq 0 ] && continue
    STATUS="MISSING"
    [ -f "${OUTDIR}/${CONTIG}/${CONTIG}_ipd.bam.pbi" ] && STATUS="OK"
    printf "%-20s %10d %10s\n" "$CONTIG" "$MAPPED" "$STATUS"
done
echo "----------------------------------------"

echo ""
date
echo "=== Done ==="
echo ""
echo "Next: sbatch slurm_kinsim/msa1003/01_ipdsummary.slurm $OUTDIR"
