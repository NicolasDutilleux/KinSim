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

SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
[ -f "$REF" ] || { echo "ERROR: reference missing"; exit 1; }
[ -s "$GFF" ] || { echo "ERROR: GFF missing or empty"; exit 1; }
mkdir -p "$(dirname "$OUT_CSV")"

echo "=== callers/pbmotifmaker — $(basename "$OUT_CSV") ==="
echo "Reference: $REF"
echo "GFF:       $GFF"
echo "Out CSV:   $OUT_CSV"

if [ -s "$OUT_CSV" ]; then
    echo "SKIP — motifs.csv exists ($(tail -n +2 "$OUT_CSV" | wc -l) motifs)"
    exit 0
fi

T=$(date +%s)
# -m 100.0 : minimum motif score threshold.  Matches L. Falquet's production
# pipeline (SMRT-Link 25.1 motifMaker find -m 100.0).  The SIF 25.3 default is
# more stringent and returned only 1 motif on our test; -m 100.0 recovers the
# typical 5-10 motifs per bacterial sample.
MOTIFMAKER_MIN_SCORE=${MOTIFMAKER_MIN_SCORE:-100.0}
apptainer exec --bind /data "$SMRT_SIF" \
    pbmotifmaker find -m "$MOTIFMAKER_MIN_SCORE" "$REF" "$GFF" "$OUT_CSV"
echo "Elapsed: $(( $(date +%s) - T ))s"
echo "  min-score threshold (MOTIFMAKER_MIN_SCORE): $MOTIFMAKER_MIN_SCORE"

[ -s "$OUT_CSV" ] || { echo "WARNING: motifs.csv empty"; exit 1; }
echo "Motifs discovered: $(tail -n +2 "$OUT_CSV" | wc -l)"
echo "--- head ---"
head -6 "$OUT_CSV"
