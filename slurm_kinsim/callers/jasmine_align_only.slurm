#!/bin/bash
#SBATCH --job-name=call_jas_align
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# callers/jasmine_align_only — jasmine 5mC + pbmm2 align (NO modkit, NO motif_merge)
#
# Usage:
#   sbatch -J call_jas_<sample> slurm_kinsim/callers/jasmine_align_only.slurm \
#          <raw_bam> <reference_fasta> <out_aligned_bam> [work_dir]
#
# Outputs the aligned bystrandified-equivalent jasmine BAM with C+m MM/ML
# tags preserved. This is exactly what kinsim_NN/labelers/JasmineMMMLLabeler
# consumes (parses MM/ML per-read, aggregates to per-(strand, ref_pos) 5mC).
#
# Skipped steps vs jasmine_modkit.slurm: modkit pileup + modkit find-motifs +
# combined-parser CSV emission (not needed since kinsim_NN reads MM/ML
# directly, no motifs.csv aggregation step).
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

RAW_BAM=${1:?"Usage: sbatch $0 <raw_bam> <ref> <out_aligned_bam> [work_dir]"}
REF=${2:?"reference missing"}
OUT_BAM=${3:?"out_aligned_bam missing"}
WORK=${4:-$(dirname "$OUT_BAM")/jasmine_work}

SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

# SKIP-first
if [ -s "$OUT_BAM" ]; then
    echo "SKIP — $OUT_BAM exists ($(samtools view -c "$OUT_BAM" 2>/dev/null || echo '?') records)"
    exit 0
fi

[ -f "$RAW_BAM" ]   || { echo "ERROR: raw BAM missing: $RAW_BAM"; exit 1; }
[ -f "$REF" ]       || { echo "ERROR: reference missing: $REF"; exit 1; }
[ -f "${REF}.fai" ] || samtools faidx "$REF"
mkdir -p "$WORK" "$(dirname "$OUT_BAM")"

SAMPLE=$(basename "${OUT_BAM%.bam}")
JAS_BAM="${WORK}/${SAMPLE}_jasmine.bam"

echo "========================================================"
echo "  callers/jasmine_align_only — $SAMPLE"
echo "========================================================"
echo "Raw BAM:   $RAW_BAM"
echo "Reference: $REF"
echo "Out BAM:   $OUT_BAM (aligned, with MM/ML)"
echo "Work dir:  $WORK"

T0=$(date +%s)

# --- 1. decide whether jasmine 5mC needs to run ---
needs_jasmine=true
if [ -s "$JAS_BAM" ]; then
    echo "=== jasmine SKIP — pre-computed BAM exists ==="
    needs_jasmine=false
else
    echo "=== probing raw BAM for existing 5mC (C+m) MM tag ==="
    if apptainer exec --bind /data "$SMRT_SIF" samtools view "$RAW_BAM" 2>/dev/null \
         | head -500 | grep -qE $'MM:Z:[^\t]*C\\+m'; then
        echo "    RAW BAM already has C+m -> reuse as JAS_BAM (skip jasmine)"
        JAS_BAM="$RAW_BAM"
        needs_jasmine=false
    else
        echo "    RAW BAM lacks C+m -> running conda jasmine 5mC default"
    fi
fi

if $needs_jasmine; then
    command -v jasmine >/dev/null || { echo "ERROR: jasmine not on PATH"; exit 1; }
    T=$(date +%s)
    jasmine "$RAW_BAM" "$JAS_BAM" --keep-kinetics
    echo "    jasmine elapsed: $(( $(date +%s) - T ))s"
fi
[ -s "$JAS_BAM" ] || { echo "FAILED: jasmine BAM missing"; exit 1; }

# ensure .pbi for pbmm2
[ -f "${JAS_BAM}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$JAS_BAM"

# --- 2. pbmm2 align (preserves MM/ML) ---
echo "=== pbmm2 align (jasmine BAM -> reference) ==="
T=$(date +%s)
apptainer exec --bind /data "$SMRT_SIF" \
    pbmm2 align "$REF" "$JAS_BAM" "$OUT_BAM" \
    --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
echo "    pbmm2 elapsed: $(( $(date +%s) - T ))s"

[ -f "${OUT_BAM}.bai" ] || samtools index "$OUT_BAM"
[ -f "${OUT_BAM}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$OUT_BAM"

echo ""
echo "Total elapsed: $(( $(date +%s) - T0 ))s"
echo "Aligned BAM:   $OUT_BAM"
echo "MM/ML tags:    $(apptainer exec --bind /data "$SMRT_SIF" samtools view "$OUT_BAM" 2>/dev/null | head -1 | grep -oE 'MM:Z:[^\t]+' | head -c 60)"
