#!/bin/bash
#SBATCH --job-name=call_pbmotifmaker
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# callers/pbmotifmaker — motif discovery from ipdSummary GFF
#
# Usage:
#   sbatch -J call_mm_<sample> slurm_kinsim/callers/pbmotifmaker.slurm \
#          <ref_fasta> <ipdsummary_gff> <out_motifs_csv>
#
# Output is the PacBio motifs.csv format (motifString, centerPos,
# modificationType, fraction, nDetected, nGenome, ...). Conversion to the
# combined-parser format happens later in merge_motifs.
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

REF=${1:?"Usage: sbatch $0 <ref_fasta> <ipdsummary_gff> <out_motifs_csv>"}
GFF=${2:?"ipdsummary_gff missing"}
OUT_CSV=${3:?"out_motifs_csv missing"}

# Use L. Falquet's pbmotifmaker 1.1.0 (SMRT-Link 25.1) for compatibility
# with his reference outputs. The SIF 25.3 pbmotifmaker 1.2.0 is significantly
# more stringent at the same --min-score and produces fewer motifs.
FALQUET_PBMM=/data/users/lfalquet/SOFTS/SMRTlink/smrtlink251R/smrtcmds/bin/pbmotifmaker

[ -x "$FALQUET_PBMM" ] || { echo "ERROR: Falquet pbmotifmaker not executable: $FALQUET_PBMM"; exit 1; }
[ -f "$REF" ] || { echo "ERROR: reference missing"; exit 1; }
[ -s "$GFF" ] || { echo "ERROR: GFF missing or empty"; exit 1; }
mkdir -p "$(dirname "$OUT_CSV")"

echo "=== callers/pbmotifmaker — $(basename "$OUT_CSV") ==="
echo "Reference: $REF"
echo "GFF:       $GFF"
echo "Out CSV:   $OUT_CSV"
echo "Binary:    $FALQUET_PBMM (pbmotifmaker 1.1.0 SMRT-Link 25.1)"

if [ -s "$OUT_CSV" ]; then
    echo "SKIP — motifs.csv exists ($(tail -n +2 "$OUT_CSV" | wc -l) motifs)"
    exit 0
fi

T=$(date +%s)
# Falquet invokes pbmotifmaker with default params (no -m flag) in his
# pb_microbial_analysis cromwell pipeline. The -m 100.0 shown commented
# in his manual script is the default anyway.
"$FALQUET_PBMM" find "$REF" "$GFF" "$OUT_CSV"
echo "Elapsed: $(( $(date +%s) - T ))s"

[ -s "$OUT_CSV" ] || { echo "WARNING: motifs.csv empty"; exit 1; }
echo "Motifs discovered: $(tail -n +2 "$OUT_CSV" | wc -l)"
echo "--- head ---"
head -6 "$OUT_CSV"
