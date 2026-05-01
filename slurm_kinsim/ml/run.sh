#!/bin/bash
# ============================================================
# ML pipeline orchestrator — generic across Vega / Sequel / Strepto
#
# Per-step submit:
#   bash slurm_kinsim/ml/run.sh extract  <manifest.csv> <prefix>
#   bash slurm_kinsim/ml/run.sh merge    <prefix>
#   bash slurm_kinsim/ml/run.sh refine   <prefix>
#   bash slurm_kinsim/ml/run.sh train    <prefix>  [train_flags...]
#   bash slurm_kinsim/ml/run.sh generate <prefix>  <pbsim3_dir> <motifs> [ckpt_epoch]
#   bash slurm_kinsim/ml/run.sh evaluate <prefix>
#   bash slurm_kinsim/ml/run.sh verify   <manifest.csv> <prefix> [gen_dir]
#   bash slurm_kinsim/ml/run.sh all      <manifest.csv> <prefix>
#
# <prefix> is a working directory. The pipeline lays out:
#   <prefix>/shards/             per-sample *_shard.pkl
#   <prefix>/master.pkl
#   <prefix>/master_clean.pkl    (after refine)
#   <prefix>/refine_report.tsv
#   <prefix>/checkpoints/        model_config.json + checkpoint_epoch*.pt
#   <prefix>/generated/          <sample>_mlp.bam (from generate step)
#   <prefix>/verify/             per-sample verify tsvs
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
EXTRACT_CONCURRENT=4

paths() {
    local prefix=$1
    SHARDS="${prefix}/shards"
    SHARDS_V4="${prefix}/shards_v4"
    MASTER="${prefix}/master.pkl"
    MASTER_CLEAN="${prefix}/master_clean.pkl"          # v3 GMM-refined (legacy chain)
    MASTER_V4="${prefix}/master_v4.pkl"                # v4 merged (single-pass extract)
    MASTER_V4_CLEAN="${prefix}/master_v4_clean.pkl"    # v4 after GMM + p95
    REFINE_REPORT="${prefix}/refine_report.tsv"
    CKPT_DIR="${prefix}/checkpoints"
    GEN_DIR="${prefix}/generated"
    VERIFY_DIR="${prefix}/verify"
}

submit_extract() {
    local manifest=$1; local prefix=$2; local dep=${3:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    local n; n=$(kinsim-prep manifest count "$manifest")
    mkdir -p "$SHARDS"
    sbatch --parsable $d --array=1-${n}%${EXTRACT_CONCURRENT} \
        "${HERE}/00_extract.slurm" "$manifest" "$SHARDS"
}

submit_merge() {
    local prefix=$1; local dep=${2:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterany:${dep}"
    paths "$prefix"
    sbatch --parsable $d "${HERE}/01_merge.slurm" "$SHARDS" "$MASTER"
}

submit_refine() {
    local prefix=$1; local dep=${2:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    sbatch --parsable $d \
        "${HERE}/02_refine.slurm" "$MASTER" "$MASTER_CLEAN" "$REFINE_REPORT"
}

submit_extract_v4() {
    # v4 standalone extract: emits meth+slowed+baseline samples in one pass.
    # No bootstrap required — false-positive motifs are dropped downstream by
    # `kinsim refine` (GMM pass-1 on meth pool + p95 pass-2 on slowed).
    local manifest=$1; local prefix=$2; local dep=${3:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    local n; n=$(kinsim-prep manifest count "$manifest")
    mkdir -p "$SHARDS_V4"
    sbatch --parsable $d --array=1-${n}%${EXTRACT_CONCURRENT} \
        "${HERE}/00b_extract_v4.slurm" "$manifest" "$SHARDS_V4"
}

submit_merge_v4() {
    local prefix=$1; local dep=${2:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterany:${dep}"
    paths "$prefix"
    sbatch --parsable $d "${HERE}/01_merge.slurm" "$SHARDS_V4" "$MASTER_V4"
}

submit_refine_v4() {
    # v4 refine: kinsim refine auto-detects v4 input and runs:
    #   pass-1 GMM on the CATEGORY_METH pool (drops FP motif matches)
    #   pass-2 p95 on the CATEGORY_SLOWED pool (drops FP slowed)
    local prefix=$1; local dep=${2:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    sbatch --parsable $d \
        "${HERE}/02_refine.slurm" "$MASTER_V4" "$MASTER_V4_CLEAN"
}

submit_train() {
    local prefix=$1; local dep=${2:-}; shift 2 || true
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    mkdir -p "$CKPT_DIR"
    # Prefer v4 dataset if present; fall back to v3.
    local pkl
    if [ -f "$MASTER_V4_CLEAN" ]; then
        pkl="$MASTER_V4_CLEAN"
        echo "  train: using v4 master ($pkl)"
    else
        pkl="$MASTER_CLEAN"
        echo "  train: using v3 master ($pkl)"
    fi
    sbatch --parsable $d "${HERE}/03_train.slurm" "$pkl" "$CKPT_DIR" "$@"
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
    sbatch --parsable $d "${HERE}/05_evaluate.slurm" "$CKPT_DIR" "$MASTER_CLEAN"
}

submit_verify() {
    local manifest=$1; local prefix=$2; local gen_override=${3:-}; local dep=${4:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    local gen=${gen_override:-$GEN_DIR}
    mkdir -p "$VERIFY_DIR"
    local n; n=$(kinsim-prep manifest count "$manifest")
    sbatch --parsable $d --array=1-${n}%${EXTRACT_CONCURRENT} \
        "${HERE}/06_verify_generate.slurm" "$manifest" "$gen" "$VERIFY_DIR"
}

submit_analyze() {
    local prefix=$1; local dep=${2:-}
    local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    paths "$prefix"
    local pkl="${MASTER_CLEAN}"
    [ -f "$pkl" ] || pkl="${MASTER}"   # fall back to pre-refine if clean not ready
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
    merge)
        PREFIX=${2:?"prefix required"}
        J=$(submit_merge "$PREFIX"); echo "ml.01 merge:    $J"
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
    extract-v4)
        MANIFEST=${2:?"manifest required"}; PREFIX=${3:?"prefix required"}
        J=$(submit_extract_v4 "$MANIFEST" "$PREFIX")
        echo "ml.00b extract-v4: $J"
        ;;
    merge-v4)
        PREFIX=${2:?"prefix required"}
        J=$(submit_merge_v4 "$PREFIX"); echo "ml.01b merge-v4:  $J"
        ;;
    refine-v4)
        PREFIX=${2:?"prefix required"}
        J=$(submit_refine_v4 "$PREFIX"); echo "ml.02b refine-v4: $J"
        ;;
    all)
        # Bootstrap chain (v3): produces master_clean.pkl needed by v4.
        MANIFEST=${2:?"manifest required"}; PREFIX=${3:?"prefix required"}
        J0=$(submit_extract "$MANIFEST" "$PREFIX");  echo "ml.00 extract:  $J0"
        J1=$(submit_merge   "$PREFIX" "$J0");        echo "ml.01 merge:    $J1 (after $J0)"
        J2=$(submit_refine  "$PREFIX" "$J1");        echo "ml.02 refine:   $J2 (after $J1)"
        J3=$(submit_train   "$PREFIX" "$J2");        echo "ml.03 train:    $J3 (after $J2)"
        J4=$(submit_evaluate "$PREFIX" "$J3");       echo "ml.05 evaluate: $J4 (after $J3)"
        echo ""
        echo "generate + verify require a pbsim3 dir and motifs — submit manually:"
        echo "  bash $0 generate $PREFIX <pbsim3_dir> <motifs>"
        echo "  bash $0 verify   $MANIFEST $PREFIX"
        ;;
    all-v4)
        # Full v4 chain (no bootstrap):
        #   extract-v4  (single pass)  -> shards_v4/
        #   merge-v4                    -> master_v4.pkl
        #   refine-v4   (GMM + p95)     -> master_v4_clean.pkl
        #   train                       -> checkpoints/
        #   evaluate
        MANIFEST=${2:?"manifest required"}; PREFIX=${3:?"prefix required"}
        J3=$(submit_extract_v4 "$MANIFEST" "$PREFIX"); echo "ml.00b extract-v4: $J3"
        J4=$(submit_merge_v4   "$PREFIX" "$J3");        echo "ml.01b merge-v4:   $J4 (after $J3)"
        J5=$(submit_refine_v4  "$PREFIX" "$J4");        echo "ml.02b refine-v4:  $J5 (after $J4)"
        # Train on v4 final (uses MASTER_V4_CLEAN, not MASTER_CLEAN)
        paths "$PREFIX"
        mkdir -p "$CKPT_DIR"
        J6=$(sbatch --parsable --dependency=afterok:${J5} \
                "${HERE}/03_train.slurm" "$MASTER_V4_CLEAN" "$CKPT_DIR")
        echo "ml.03  train-v4:   $J6 (after $J5)"
        J7=$(submit_evaluate "$PREFIX" "$J6"); echo "ml.05  evaluate:   $J7 (after $J6)"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/ml/run.sh <step> [args...]

Steps (v4 — recommended):
  extract-v4 <manifest.csv> <prefix>                     array, single-pass extract, 36-col CATEGORY shards
  merge-v4   <prefix>                                     shards_v4/ → master_v4.pkl
  refine-v4  <prefix>                                     GMM (drops FP meth) + p95 (drops FP slowed) → master_v4_clean.pkl

Steps (v3 — legacy, kept for backward compat):
  extract  <manifest.csv> <prefix>                       array per manifest row, 35-col tuple-keyed shards
  merge    <prefix>                                       shards → master.pkl
  refine   <prefix>                                       GMM only → master_clean.pkl

Common:
  analyze  <prefix>                                       kinsim analyze on master_v4_clean.pkl (or master_clean.pkl)
  train    <prefix> [flags...]                            train ConvPredictor on master_clean.pkl (v3 chain)
  generate <prefix> <pbsim3_dir> <motifs> [epoch]         array per PBSIM3 species
  evaluate <prefix>                                       calibration report
  verify   <manifest.csv> <prefix> [gen_dir]              kinsim verify-generate per sample

Chains:
  all      <manifest.csv> <prefix>                        v3 legacy: extract→merge→refine→train→evaluate
  all-v4   <manifest.csv> <prefix>                        v4 (recommended): extract-v4→merge-v4→refine-v4→train→evaluate

Prefix layout:
  <prefix>/shards_v4/           v4 per-sample *_shard_v4.pkl
  <prefix>/master_v4.pkl        v4 merged
  <prefix>/master_v4_clean.pkl  v4 after GMM + p95 → input to train
  <prefix>/checkpoints/         model_config.json + checkpoint_epoch*.pt
  <prefix>/generated/           generated BAMs
  <prefix>/verify/              per-sample verify tsvs
EOF
        exit 1
        ;;
esac
