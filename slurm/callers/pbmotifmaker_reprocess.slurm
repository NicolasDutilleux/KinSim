#!/bin/bash
#SBATCH --job-name=call_mm_reprocess
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# callers/pbmotifmaker_reprocess — annotate ipdSummary GFF with motif= attribute
#
# Usage:
#   sbatch -J call_mm_rp_<sample> slurm/callers/pbmotifmaker_reprocess.slurm \
#          <ref_fasta> <ipdsummary_gff> <motifs_csv> <out_annotated_gff>
#
# Takes the raw ipdSummary GFF + the motifs.csv produced by `pbmotifmaker find`
# and writes a GFF where each `modified_base` line gains a `motif=<MOTIF>;
# id=<MOTIF>` attribute in column 9 (matching the Strepto convention).
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

REF=${1:?"Usage: sbatch $0 <ref_fasta> <ipdsummary_gff> <motifs_csv> <out_annotated_gff>"}
IPD_GFF=${2:?"ipdsummary_gff missing"}
MOTIFS_CSV=${3:?"motifs_csv missing"}
OUT_GFF=${4:?"out_annotated_gff missing"}

FALQUET_PBMM=/data/users/lfalquet/SOFTS/SMRTlink/smrtlink251R/smrtcmds/bin/pbmotifmaker

[ -x "$FALQUET_PBMM" ] || { echo "ERROR: Falquet pbmotifmaker not executable: $FALQUET_PBMM"; exit 1; }
[ -f "$REF" ]       || { echo "ERROR: reference missing"; exit 1; }
[ -s "$IPD_GFF" ]   || { echo "ERROR: ipdSummary GFF missing or empty"; exit 1; }
[ -s "$MOTIFS_CSV" ] || { echo "ERROR: motifs.csv missing or empty"; exit 1; }
mkdir -p "$(dirname "$OUT_GFF")"

echo "=== callers/pbmotifmaker reprocess — $(basename "$OUT_GFF") ==="
echo "Reference:    $REF"
echo "ipdSummary:   $IPD_GFF"
echo "Motifs CSV:   $MOTIFS_CSV"
echo "Out GFF:      $OUT_GFF"

if [ -s "$OUT_GFF" ]; then
    echo "SKIP — annotated GFF exists ($(grep -c '^[^#]' "$OUT_GFF" || true) records)"
    exit 0
fi

T=$(date +%s)
"$FALQUET_PBMM" reprocess \
    "$REF" "$IPD_GFF" "$MOTIFS_CSV" "$OUT_GFF"
echo "Elapsed: $(( $(date +%s) - T ))s"

[ -s "$OUT_GFF" ] || { echo "WARNING: output GFF empty"; exit 1; }
echo "Annotated GFF records: $(grep -c '^[^#]' "$OUT_GFF" || true)"
echo "Records with motif= attribute: $(grep -c 'motif=' "$OUT_GFF" || true)"
