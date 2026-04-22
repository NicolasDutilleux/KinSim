#!/bin/bash
#SBATCH --job-name=sequel_04_ipd
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=08:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_04_ipd_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel step 04 — ipdSummary SP3-C3 (m6A + m4C with methylFraction)
#   sbatch --array=1-48%4 slurm_kinsim/sequel/04_ipdsummary.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true
set -euo pipefail

SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
OUTBASE=${SEQUEL}/ipdsummary_ccs
ASM_DIR=${SEQUEL}/assemblies
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

mapfile -t GZ_BAMS < <(ls "${SEQUEL}"/lima.bc*--bc*.bam.gz 2>/dev/null | sort)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
[ $IDX -lt ${#GZ_BAMS[@]} ] || exit 0

BC=$(basename "${GZ_BAMS[$IDX]}" .bam.gz | grep -oP 'bc\d+' | head -1)
ASM_FASTA="${ASM_DIR}/${BC}_assembly.fsa"
ALIGNED="${OUTBASE}/${BC}/${BC}_aligned.bam"
GFF="${OUTBASE}/${BC}/${BC}_ipdSummary.gff"
CSV="${OUTBASE}/${BC}/${BC}_ipdSummary.csv"

echo "=== Sequel 04/ipdSummary — $BC ==="
START=$(date +%s)

[ -f "$ASM_FASTA" ] || { echo "ERROR: assembly missing"; exit 1; }
[ -f "${ALIGNED}.pbi" ] || { echo "ERROR: pbindex missing ( run 03_index first )"; exit 1; }

if [ -s "$GFF" ]; then
    echo "SKIP — GFF exists ($(grep -c '^[^#]' "$GFF" || true) records)"; exit 0
fi

ipdSummary "$ALIGNED" \
    --reference "$ASM_FASTA" \
    --identify m6A,m4C \
    --methylFraction \
    --csv "$CSV" \
    --gff "$GFF" \
    --numWorkers "${SLURM_CPUS_PER_TASK}" \
    --ipdModel "$SP3_MODEL"

[ -s "$GFF" ] || { echo "WARNING: GFF empty"; exit 1; }
echo "GFF records: $(grep -c '^[^#]' "$GFF" || true)"
echo "Elapsed: $(( $(date +%s) - START ))s"
