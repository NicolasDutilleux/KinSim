#!/bin/bash
#SBATCH --job-name=vega_04_ipd
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=08:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_04_ipd_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega step 04 — ipdSummary SP3-C3 (m6A + m4C with methylFraction)
#
# Usage:
#   sbatch --array=1-16%4 slurm_kinsim/vega/04_ipdsummary.slurm
#
# Output:  ${VEGA}/assembly/bcXXXX/bcXXXX_ipdSummary.{gff,csv}
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true
set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
OUTBASE=${VEGA}/assembly
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

BARCODES=(2033 2034 2035 2036 2037 2038 2039 2040
          2041 2042 2043 2044 2045 2046 2047 2048)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
BC=${BARCODES[$IDX]}
SAMPLE="bc${BC}"
SPECIES_DIR="${OUTBASE}/${SAMPLE}"
ASM_FASTA="${SPECIES_DIR}/${SAMPLE}_assembly.fasta"
ALIGNED="${SPECIES_DIR}/${SAMPLE}_aligned.bam"
GFF="${SPECIES_DIR}/${SAMPLE}_ipdSummary.gff"
CSV="${SPECIES_DIR}/${SAMPLE}_ipdSummary.csv"

START=$(date +%s)
echo "=== Vega 04/ipdSummary — $SAMPLE ==="
echo "Started: $(date '+%H:%M:%S')"

[ -f "$ASM_FASTA" ] || { echo "ERROR: assembly missing: $ASM_FASTA"; exit 1; }
[ -f "$ALIGNED" ]   || { echo "ERROR: aligned BAM missing: $ALIGNED"; exit 1; }
[ -f "${ALIGNED}.pbi" ] || { echo "ERROR: pbindex missing ( run 03_index.slurm )"; exit 1; }

if [ -f "$GFF" ] && [ -s "$GFF" ]; then
    echo "SKIP — GFF exists ($(grep -c '^[^#]' "$GFF" || true) records)"
    exit 0
fi

ipdSummary "$ALIGNED" \
    --reference "$ASM_FASTA" \
    --identify m6A,m4C \
    --methylFraction \
    --csv "$CSV" \
    --gff "$GFF" \
    --numWorkers "${SLURM_CPUS_PER_TASK}" \
    --ipdModel "$SP3_MODEL"

if [ ! -s "$GFF" ]; then
    echo "WARNING: GFF empty or missing"; exit 1
fi

echo "GFF records: $(grep -c '^[^#]' "$GFF" || true)"
echo "Elapsed: $(( $(date +%s) - START ))s"
