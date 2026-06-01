#!/bin/bash
#SBATCH --job-name=v6_gen
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# validate/v6_generate — strip + kinsim_nn generate (GPU-bound)
#
# Usage (single strain):
#   sbatch -J v6_gen_<strain> slurm/validate/v6_generate.slurm \
#          <input_bystrandified_bam> <ref_fasta> <motifs_csv> \
#          <ckpt_dir> <out_dir>
#
# Produces:
#   <out_dir>/<bc>_hifi_stripped.bam   (1 record/ZMW, unaligned, no kinetics)
#   <out_dir>/<bc>_kinsim.bam          (1 record/ZMW with synthetic fi/fp/ri/rp)
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

IN_BAM=${1:?"Usage: sbatch $0 <input.bam> <ref.fa> <motifs.csv> <ckpt_dir> <out_dir>"}
REF=${2:?"ref missing"}
MOTIFS=${3:?"motifs.csv missing"}
CKPT=${4:?"ckpt_dir missing"}
OUT=${5:?"out_dir missing"}

mkdir -p "$OUT"
BC=$(basename "$IN_BAM" | sed -E 's/(_aligned)?\.bam$//')
STRIPPED="$OUT/${BC}_hifi_stripped.bam"
KINSIM="$OUT/${BC}_kinsim.bam"

[ -f "$IN_BAM" ]  || { echo "ERROR: input BAM missing: $IN_BAM"; exit 1; }
[ -f "$REF" ]     || { echo "ERROR: ref missing: $REF"; exit 1; }
[ -f "$MOTIFS" ]  || { echo "ERROR: motifs.csv missing: $MOTIFS"; exit 1; }
[ -d "$CKPT" ]    || { echo "ERROR: ckpt_dir missing: $CKPT"; exit 1; }
[ -f "$CKPT/best_G.pt" ]          || { echo "ERROR: best_G.pt missing in $CKPT"; exit 1; }
[ -f "$CKPT/model_config.json" ]  || { echo "ERROR: model_config.json missing in $CKPT"; exit 1; }

SCRIPT_DIR=$(dirname "$(realpath "$0")")/../../scripts
STRIP_PY="$SCRIPT_DIR/strip_bystrandified_to_hifi.py"
[ -f "$STRIP_PY" ] || { echo "ERROR: strip script not found: $STRIP_PY"; exit 1; }

echo "========================================================"
echo "  validate/v6_generate — $BC"
echo "========================================================"
echo "Input BAM:  $IN_BAM"
echo "Reference:  $REF"
echo "Motifs:     $MOTIFS"
echo "Ckpt dir:   $CKPT"
echo "Output dir: $OUT"

# --- STEP 1: strip bystrandified → HiFi-like unaligned ---
echo
echo "=== STEP 1: strip → $(basename "$STRIPPED") ==="
if [ -s "$STRIPPED" ]; then
    echo "SKIP — stripped BAM exists"
else
    T=$(date +%s)
    python "$STRIP_PY" "$IN_BAM" "$STRIPPED"
    echo "Elapsed: $(( $(date +%s) - T ))s"
    [ -s "$STRIPPED" ] || { echo "FAILED: strip produced empty BAM"; exit 1; }
fi
echo "Stripped BAM size: $(stat -c %s "$STRIPPED") B"
echo "Stripped reads:    $(samtools view -c "$STRIPPED")"

# --- STEP 2: kinsim_nn generate ---
echo
echo "=== STEP 2: kinsim_nn generate → $(basename "$KINSIM") ==="
if [ -s "$KINSIM" ]; then
    echo "SKIP — kinsim BAM exists"
else
    T=$(date +%s)
    kinsim_nn generate "$STRIPPED" "$REF" "$CKPT" "$MOTIFS" "$KINSIM" -v \
        --n-workers "${SLURM_CPUS_PER_TASK}"
    echo "Elapsed: $(( $(date +%s) - T ))s"
    [ -s "$KINSIM" ] || { echo "FAILED: kinsim_nn generate produced empty BAM"; exit 1; }
fi
echo "Kinsim BAM size:   $(stat -c %s "$KINSIM") B"
echo "Kinsim reads:      $(samtools view -c "$KINSIM")"

echo
echo "========================================================"
echo "  v6_generate DONE for $BC"
echo "========================================================"
ls -la "$OUT"
