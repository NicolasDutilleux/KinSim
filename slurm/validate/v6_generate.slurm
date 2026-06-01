#!/bin/bash
#SBATCH --job-name=v6_gen
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# validate/v6_generate — kinsim_nn generate on a stripped HiFi-like BAM
#
# Usage:
#   sbatch -J v6_gen_<strain> slurm/validate/v6_generate.slurm \
#          <stripped_bam> <ref_fasta> <motifs_csv> <ckpt_dir> <out_dir>
#
# Output:
#   <out_dir>/<bc>_kinsim.bam   (1 record/ZMW with synthetic fi/fp/ri/rp)
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

STRIPPED=${1:?"Usage: sbatch $0 <stripped.bam> <ref.fa> <motifs.csv> <ckpt_dir> <out_dir>"}
REF=${2:?"ref missing"}
MOTIFS=${3:?"motifs.csv missing"}
CKPT=${4:?"ckpt_dir missing"}
OUT=${5:?"out_dir missing"}

mkdir -p "$OUT"
BC=$(basename "$STRIPPED" | sed -E 's/_hifi_stripped\.bam$//')
KINSIM="$OUT/${BC}_kinsim.bam"

[ -f "$STRIPPED" ] || { echo "ERROR: stripped BAM missing: $STRIPPED"; exit 1; }
[ -f "$REF" ]      || { echo "ERROR: ref missing: $REF"; exit 1; }
[ -f "$MOTIFS" ]   || { echo "ERROR: motifs.csv missing: $MOTIFS"; exit 1; }
[ -d "$CKPT" ]     || { echo "ERROR: ckpt_dir missing: $CKPT"; exit 1; }
[ -f "$CKPT/best_G.pt" ]         || { echo "ERROR: best_G.pt missing in $CKPT"; exit 1; }
[ -f "$CKPT/model_config.json" ] || { echo "ERROR: model_config.json missing in $CKPT"; exit 1; }

echo "========================================================"
echo "  validate/v6_generate — $BC"
echo "========================================================"
echo "Stripped:   $STRIPPED"
echo "Reference:  $REF"
echo "Motifs:     $MOTIFS"
echo "Ckpt dir:   $CKPT"
echo "Output dir: $OUT"
echo
echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"

if [ -s "$KINSIM" ]; then
    echo "SKIP — kinsim BAM exists: $KINSIM"
else
    T=$(date +%s)
    kinsim_nn generate "$STRIPPED" "$REF" "$CKPT" "$MOTIFS" "$KINSIM" -v \
        --n-workers "${SLURM_CPUS_PER_TASK}"
    echo "Elapsed: $(( $(date +%s) - T ))s"
    [ -s "$KINSIM" ] || { echo "FAILED: kinsim_nn generate produced empty BAM"; exit 1; }
fi
echo "Kinsim BAM size:   $(stat -c %s "$KINSIM") B"
echo "Kinsim reads:      $(samtools view -c "$KINSIM")"
ls -la "$OUT"
