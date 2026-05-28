#!/bin/bash
#SBATCH --job-name=call_ipdsummary
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# callers/ipdsummary — ipdSummary SP3-C3 (m6A + m4C with methylFraction)
#
# Usage:
#   sbatch -J call_ipd_<sample> slurm_kinsim/callers/ipdsummary.slurm \
#          <aligned_bam> <reference_fasta> <out_gff> <out_csv>
#
# Uses L. Falquet's SMRT-Link 25.1 install (kineticsTools/ipdSummary 3.0)
# for compatibility with his reference outputs (bc2071 TCGCGA m4C 94%).
# Our cluster module SMRT-Link 12.0 was 13 versions older and produced
# lower detection rates (e.g. TCGCGA 73% instead of 94%).
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

IN_BAM=${1:?"Usage: sbatch $0 <aligned_bam> <reference> <out_gff> <out_csv>"}
REF=${2:?"reference missing"}
OUT_GFF=${3:?"out_gff missing"}
OUT_CSV=${4:?"out_csv missing"}

# L. Falquet's SMRT-Link 25.1 install — matches his reference pipeline.
# Run ipdSummary directly via its path (no module load needed).
FALQUET_SMRT=/data/users/lfalquet/SOFTS/SMRTlink/smrtlink251R
IPDSUMMARY_BIN="$FALQUET_SMRT/smrtcmds/bin/ipdSummary"
SP3_MODEL="$FALQUET_SMRT/install/smrtlink-release_25.1.0.257715/bundles/smrttools/install/smrttools-release_25.1.0.257715/private/pacbio/python3pkgs/kineticstools-py3/lib/python3.9/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

[ -x "$IPDSUMMARY_BIN" ] || { echo "ERROR: Falquet ipdSummary not executable: $IPDSUMMARY_BIN"; exit 1; }
[ -f "$SP3_MODEL" ]      || { echo "ERROR: SP3-C3 model not found: $SP3_MODEL"; exit 1; }

[ -f "$IN_BAM" ]        || { echo "ERROR: input BAM missing"; exit 1; }
[ -f "${IN_BAM}.pbi" ]  || { echo "ERROR: .pbi missing (run prep/index_bam first)"; exit 1; }
[ -f "$REF" ]           || { echo "ERROR: reference missing"; exit 1; }

mkdir -p "$(dirname "$OUT_GFF")" "$(dirname "$OUT_CSV")"

echo "=== callers/ipdsummary — $(basename "$OUT_GFF") ==="
echo "Input BAM: $IN_BAM"
echo "Reference: $REF"
echo "Out GFF:   $OUT_GFF"
echo "Out CSV:   $OUT_CSV"

if [ -s "$OUT_GFF" ]; then
    echo "SKIP — GFF exists ($(grep -c '^[^#]' "$OUT_GFF" || true) records)"
    exit 0
fi

T=$(date +%s)
# Permissive thresholds — required to reproduce Falquet legacy detection rates
# (e.g. bc2071 TCGCGA 73% → 95%+, matches his 94%). Default ipdSummary is too
# strict, missing ~800 marginal sites at well-known methylated motifs. Brutal
# config (pvalue 0.3, etc) produced TCGCGA 97%; we tune slightly stricter to
# reduce noise candidates. Merge step downstream applies 0.7 fraction
# threshold which filters anything still noisy.
"$IPDSUMMARY_BIN" "$IN_BAM" \
    --reference "$REF" \
    --identify m6A,m4C \
    --methylFraction \
    --pvalue 0.2 \
    --methylMinCov 5 \
    --identifyMinCov 3 \
    --minCoverage 2 \
    --mapQvThreshold 10 \
    --csv "$OUT_CSV" \
    --gff "$OUT_GFF" \
    --numWorkers "${SLURM_CPUS_PER_TASK}" \
    --ipdModel "$SP3_MODEL"
echo "Elapsed: $(( $(date +%s) - T ))s"

[ -s "$OUT_GFF" ] || { echo "WARNING: GFF empty"; exit 1; }
echo "GFF records: $(grep -c '^[^#]' "$OUT_GFF" || true)"
