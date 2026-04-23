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
# Input BAM must be pbmm2-aligned + pbindex'd. SP3-C3 model resolved via
# SMRT-Link module (ipdSummary is NOT in the apptainer SIF).
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

IN_BAM=${1:?"Usage: sbatch $0 <aligned_bam> <reference> <out_gff> <out_csv>"}
REF=${2:?"reference missing"}
OUT_GFF=${3:?"out_gff missing"}
OUT_CSV=${4:?"out_csv missing"}

source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true
command -v ipdSummary >/dev/null || { echo "ERROR: ipdSummary not on PATH"; exit 1; }

SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"
[ -f "$SP3_MODEL" ] || { echo "ERROR: SP3-C3 model not found: $SP3_MODEL"; exit 1; }

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
ipdSummary "$IN_BAM" \
    --reference "$REF" \
    --identify m6A,m4C \
    --methylFraction \
    --csv "$OUT_CSV" \
    --gff "$OUT_GFF" \
    --numWorkers "${SLURM_CPUS_PER_TASK}" \
    --ipdModel "$SP3_MODEL"
echo "Elapsed: $(( $(date +%s) - T ))s"

[ -s "$OUT_GFF" ] || { echo "WARNING: GFF empty"; exit 1; }
echo "GFF records: $(grep -c '^[^#]' "$OUT_GFF" || true)"
