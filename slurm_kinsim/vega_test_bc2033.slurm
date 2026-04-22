#!/bin/bash
#SBATCH --job-name=vega_test_bc2033
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_test_bc2033_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega bc2033 — strict ipdSummary + extract + analyze + ceiling
#
# Re-runs ipdSummary with stricter stats so the GFF itself
# contains only high-confidence methylation calls:
#   --pvalue      1e-8   (much stricter than default 0.01)
#   --minCoverage 10     (vs default 3)
#   --identifyMinCov 15
#   --methylMinCov 15
#
# Then extracts + analyzes using that cleaner GFF.
#
# Usage:
#   sbatch slurm_kinsim/vega_test_bc2033.slurm
#   sbatch slurm_kinsim/vega_test_bc2033.slurm 1e-10 15   # stricter
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

# Load SMRT-Link for ipdSummary
source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true

PVALUE=${1:-1e-8}
MINCOV=${2:-10}

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
BC=bc2033
TAG="strict_p${PVALUE}_c${MINCOV}"
TEST=${VEGA}/test_${BC}_${TAG}

ASM_FASTA=${VEGA}/assembly/${BC}/${BC}_assembly.fasta
ALIGNED=${VEGA}/assembly/${BC}/${BC}_aligned.bam
GFF=${TEST}/${BC}_ipdSummary_${TAG}.gff
CSV=${TEST}/${BC}_ipdSummary_${TAG}.csv
SHARD=${TEST}/shards/${BC}_${TAG}_shard.pkl

SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

mkdir -p "$TEST/shards" "$TEST/analyze_report"

echo "========================================================"
echo "  Vega bc2033 strict test  (pvalue=${PVALUE} minCov=${MINCOV})"
echo "========================================================"
echo "Date:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:      $(hostname)"
echo "Test dir:  $TEST"
echo "========================================================"
echo ""

# ---- Step 1: ipdSummary with strict thresholds ----
echo "=== Step 1: ipdSummary (strict) ==="
START=$(date +%s)
ipdSummary "$ALIGNED" \
    --reference "$ASM_FASTA" \
    --identify m6A,m4C \
    --methylFraction \
    --pvalue "$PVALUE" \
    --minCoverage "$MINCOV" \
    --identifyMinCov $((MINCOV + 5)) \
    --methylMinCov   $((MINCOV + 5)) \
    --csv "$CSV" \
    --gff "$GFF" \
    --numWorkers "${SLURM_CPUS_PER_TASK}" \
    --ipdModel "$SP3_MODEL"

echo "ipdSummary done in $(( $(date +%s) - START ))s"
echo "  Total GFF records: $(grep -c '^[^#]' "$GFF" || true)"
echo "  m6A:               $(awk '!/^#/ && $3=="m6A"' "$GFF" | wc -l)"
echo "  m4C:               $(awk '!/^#/ && $3=="m4C"' "$GFF" | wc -l)"
echo ""

# ---- Step 2: extract ----
cat > "$TEST/manifest.csv" <<EOF
sample_id,bam_path,motifs,gff
${BC}_${TAG},${ALIGNED},,${GFF}
EOF

echo "=== Step 2: extract ==="
START=$(date +%s)
kinsim extract \
    --manifest "$TEST/manifest.csv" \
    --task 1 \
    --output-dir "$TEST/shards"
echo "extract done in $(( $(date +%s) - START ))s"
echo ""

# ---- Step 3: analyze ----
echo "=== Step 3: analyze ==="
START=$(date +%s)
kinsim analyze "$SHARD" --output-dir "$TEST/analyze_report"
echo "analyze done in $(( $(date +%s) - START ))s"
echo ""

# ---- Step 4: ceiling ----
echo "=== Step 4: compute_ceiling --per-meth ==="
START=$(date +%s)
python slurm_kinsim/compute_ceiling.py "$SHARD" --per-meth
echo "ceiling done in $(( $(date +%s) - START ))s"
echo ""

echo "========================================================"
echo "  COMPLETE"
echo "========================================================"
echo "  GFF:     $GFF"
echo "  Shard:   $SHARD"
echo "  Report:  $TEST/analyze_report"
echo "  Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
