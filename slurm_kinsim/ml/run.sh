#!/bin/bash
# ============================================================
# ML pipeline orchestrator — generic across Vega / Sequel / Strepto.
#
# Sharded end-to-end: extract → refine → train → evaluate, each step
# consumes a directory (or file) from the previous one. No merge step.
#
# Per-step submit:
#   bash slurm_kinsim/ml/run.sh extract  <manifest.csv> <prefix>
#   bash slurm_kinsim/ml/run.sh refine   <prefix>
#   bash slurm_kinsim/ml/run.sh train    <prefix>  [train_flags...]
#   bash slurm_kinsim/ml/run.sh generate <prefix>  <pbsim3_dir> <motifs> [ckpt_epoch]
#   bash slurm_kinsim/ml/run.sh evaluate <prefix>
#   bash slurm_kinsim/ml/run.sh verify   <manifest.csv> <prefix> [gen_dir]
#   bash slurm_kinsim/ml/run.sh analyze  <prefix>
#   bash slurm_kinsim/ml/run.sh all      <manifest.csv> <prefix>
#
# <prefix> is a working directory laid out as:
#   <prefix>/shards/             per-sample *_shard.pkl    (extract output)
#   <prefix>/refined/            per-sample *_clean.pkl    (refine output)
#   <prefix>/checkpoints/        model_config.json + checkpoint_epoch*.pt
#   <prefix>/generated/          generated BAMs
#   <prefix>/verify/             per-sample verify tsvs
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
# Cap concurrent extract tasks to avoid hammering shared I/O. 16 fits
# comfortably under typical pibu_el8 fairshare; bump for short bursts.
EXTRACT_CONCURRENT=${EXTRACT_CONCURRENT:-16}

paths() {
    local prefix=$1
    SHARDS="${prefix}/shards"
    REFINED="${prefix}/refined"
    CKPT_DIR="${prefix}/checkpoints"
    GEN_DIR="${prefix}/generated"
    VERIFY_DIR="${prefix}/verify"
}

submit_extract() {
    local manifest=$1; local prefix=$2; local dep=${3:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    local n; n=$(python -c "import csv; print(sum(1 for _ in csv.DictReader(open('$manifest'))))")
    mkdir -p "$SHARDS"
    sbatch --parsable $d --array=1-${n}%${EXTRACT_CONCURRENT} \
        "${HERE}/00_extract.slurm" "$manifest" "$SHARDS"
}

submit_refine() {
    local prefix=$1; local dep=${2:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterany:${dep}"
    paths "$prefix"
    mkdir -p "$REFINED"
    sbatch --parsable $d "${HERE}/02_refine.slurm" "$SHARDS" "$REFINED"
}

submit_train() {
    local prefix=$1; local dep=${2:-}; shift 2 || true
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    mkdir -p "$CKPT_DIR"
    sbatch --parsable $d "${HERE}/03_train.slurm" "$REFINED" "$CKPT_DIR" "$@"
}

submit_generate() {
    local prefix=$1; local pbsim=$2; local motifs=$3; local epoch=${4:-}; local dep=${5:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    mkdir -p "$GEN_DIR"
    local ckpt
    if [ -n "$epoch" ]; then
        ckpt="${CKPT_DIR}/checkpoint_epoch${epoch}.pt"
    else
        ckpt=$(ls -1t "${CKPT_DIR}"/checkpoint_epoch*.pt 2>/dev/null | head -1)
    fi
    [ -f "$ckpt" ] || { echo "ERROR: no checkpoint in ${CKPT_DIR}" >&2; exit 1; }
    local n
    if ls "${pbsim}"/*.fq.gz 1>/dev/null 2>&1; then
        n=$(ls "${pbsim}"/*.fq.gz | wc -l)
    else
        n=$(ls -d "${pbsim}"/*/ 2>/dev/null | wc -l)
    fi
    sbatch --parsable $d --array=1-${n} \
        "${HERE}/04_generate.slurm" "$pbsim" "$ckpt" "$motifs" "$GEN_DIR"
}

submit_evaluate() {
    local prefix=$1; local dep=${2:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    sbatch --parsable $d "${HERE}/05_evaluate.slurm" "$CKPT_DIR" "$REFINED"
}

submit_verify() {
    local manifest=$1; local prefix=$2; local gen_override=${3:-}; local dep=${4:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    local gen=${gen_override:-$GEN_DIR}
    mkdir -p "$VERIFY_DIR"
    local n; n=$(python -c "import csv; print(sum(1 for _ in csv.DictReader(open('$manifest'))))")
    sbatch --parsable $d --array=1-${n}%${EXTRACT_CONCURRENT} \
        "${HERE}/06_verify_generate.slurm" "$manifest" "$gen" "$VERIFY_DIR"
}

submit_analyze() {
    local prefix=$1; local dep=${2:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    local pkl="${REFINED}"
    [ -d "$pkl" ] || pkl="${SHARDS}"   # fall back to pre-refine if refined not ready
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 \
        --mem=32G --cpus-per-task=2 --time=02:00:00 \
        --job-name=ml_analyze \
        --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_analyze_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && \
                kinsim analyze '$pkl' --output-dir '$prefix/analyze'"
}

STEP=${1:-}
case "$STEP" in
    extract)
        MANIFEST=${2:?"manifest required"}; PREFIX=${3:?"prefix required"}
        J=$(submit_extract "$MANIFEST" "$PREFIX"); echo "ml.00 extract:  $J"
        ;;
    refine)
        PREFIX=${2:?"prefix required"}
        J=$(submit_refine "$PREFIX"); echo "ml.02 refine:   $J"
        ;;
    train)
        PREFIX=${2:?"prefix required"}; shift 2
        J=$(submit_train "$PREFIX" "" "$@"); echo "ml.03 train:    $J"
        ;;
    generate)
        PREFIX=${2:?"prefix required"}
        PBSIM=${3:?"pbsim3 dir required"}
        MOTIFS=${4:?"motifs arg required"}
        EPOCH=${5:-}
        J=$(submit_generate "$PREFIX" "$PBSIM" "$MOTIFS" "$EPOCH")
        echo "ml.04 generate: $J"
        ;;
    evaluate)
        PREFIX=${2:?"prefix required"}
        J=$(submit_evaluate "$PREFIX"); echo "ml.05 evaluate: $J"
        ;;
    verify)
        MANIFEST=${2:?"manifest required"}
        PREFIX=${3:?"prefix required"}
        GEN=${4:-}
        J=$(submit_verify "$MANIFEST" "$PREFIX" "$GEN"); echo "ml.06 verify:   $J"
        ;;
    analyze)
        PREFIX=${2:?"prefix required"}
        J=$(submit_analyze "$PREFIX"); echo "ml.07 analyze:  $J"
        ;;
    all)
        MANIFEST=${2:?"manifest required"}; PREFIX=${3:?"prefix required"}
        J0=$(submit_extract "$MANIFEST" "$PREFIX"); echo "ml.00 extract:  $J0"
        J1=$(submit_refine  "$PREFIX" "$J0");       echo "ml.02 refine:   $J1 (after $J0)"
        J2=$(submit_train   "$PREFIX" "$J1");       echo "ml.03 train:    $J2 (after $J1)"
        J3=$(submit_evaluate "$PREFIX" "$J2");      echo "ml.05 evaluate: $J3 (after $J2)"
        echo ""
        echo "generate + verify require a pbsim3 dir and motifs — submit manually:"
        echo "  bash $0 generate $PREFIX <pbsim3_dir> <motifs>"
        echo "  bash $0 verify   $MANIFEST $PREFIX"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/ml/run.sh <step> [args...]

Steps:
  extract  <manifest.csv> <prefix>                       array per manifest row → shards/
  refine   <prefix>                                       shards/ → refined/   (per-(meth, offset) GMM)
  train    <prefix> [flags...]                            train ConvPredictor on refined/
  generate <prefix> <pbsim3_dir> <motifs> [epoch]         array per PBSIM3 species
  evaluate <prefix>                                       calibration report
  verify   <manifest.csv> <prefix> [gen_dir]              kinsim verify-generate per sample
  analyze  <prefix>                                       kinsim analyze on refined/

Chains:
  all      <manifest.csv> <prefix>                        extract → refine → train → evaluate

Prefix layout:
  <prefix>/shards/           per-sample *_shard.pkl
  <prefix>/refined/          per-sample *_clean.pkl  — input to train
  <prefix>/checkpoints/      model_config.json + checkpoint_epoch*.pt
  <prefix>/generated/        generated BAMs
  <prefix>/verify/           per-sample verify tsvs
EOF
        exit 1
        ;;
esac
