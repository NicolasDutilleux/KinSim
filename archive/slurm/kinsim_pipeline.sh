#!/bin/bash
# ============================================================
# KinSim — Full pipeline submission script
#
# Submits all SLURM jobs in dependency order for the Dictionary,
# cGAN, or MLP pipeline.  Run this from the login node;
# do NOT submit it as a SLURM job itself.
#
# Usage:
#   bash kinsim_pipeline.sh <mode> <manifest> \
#       <shards_dir> <master_pkl> \
#       <pbsim3_dir> <motifs> <M> <output_dir> \
#       [checkpoint_dir]   # cgan / mlp mode only
#
# Modes:
#   dictionary   — prepare -> train+merge -> inject -> analyze
#   cgan         — extract+merge -> train -> generate
#   mlp          — extract+merge -> train -> generate  (same data prep as cgan)
#
# Arguments:
#   mode              Pipeline mode: 'dictionary', 'cgan', or 'mlp'
#   manifest          Manifest CSV (sample_id, bam_path, motifs) for extract/train steps
#                     For dictionary mode: pairs file (alternating BAM/motif lines)
#                     N (number of strains) is auto-detected via 'kinsim manifest count'
#   shards_dir        Directory for intermediate .pkl shards
#   master_pkl        Path for the merged master dictionary or training data
#   pbsim3_dir        PBSIM3 output directory (species subdirs or flat layout)
#   motifs            Motif source for generate/inject: KinSim string, CSV, REBASE,
#                     or per-species mapping file ("species|motif_string" lines)
#   M                 Number of genomes to inject/generate (array size)
#   output_dir        Directory for output BAM files
#   checkpoint_dir    [cgan / mlp only] Output directory for checkpoints and logs
#
# Example — Dictionary mode:
#   bash kinsim_pipeline.sh dictionary \
#       pairs.txt \
#       shards/ master_dict.pkl \
#       pbsim3_output/ "m6A,GATC,1;m4C,CCWGG,1" 10 injected/
#
# Example — MLP mode (manifest CSV):
#   bash kinsim_pipeline.sh mlp \
#       manifest.csv \
#       shards/ master_data.pkl \
#       pbsim3_output/ "m6A,GATC,1;m4C,CCWGG,1" 10 generated_mlp/ \
#       checkpoints_mlp/
#
# Example — cGAN mode:
#   bash kinsim_pipeline.sh cgan \
#       manifest.csv \
#       cgan_shards/ master_cgan_data.pkl \
#       pbsim3_output/ "m6A,GATC,1;m4C,CCWGG,1" 10 generated/ \
#       checkpoints/
#
# Manifest CSV format:
#   sample_id,bam_path,motifs
#   strain1,/data/bams/strain1.bam,"m6A,GATC,1"
#   strain2,/data/bams/strain2.bam,/data/motifs/strain2.csv
#
# N is auto-detected:  kinsim manifest count manifest.csv
#
# Expected directory structure:
#   project/
#     manifest.csv            <- BAM manifest for extract (new format)
#     pairs.txt               <- (dictionary mode) alternating BAM/motif pairs
#     shards/                 <- intermediate .pkl shards
#     master_data.pkl         <- merged training data (cgan/mlp modes)
#     master_dict.pkl         <- merged dictionary (dict mode)
#     checkpoints/            <- model checkpoints (cgan/mlp modes)
#     pbsim3_output/          <- PBSIM3 species subdirectories or flat .fq.gz
#     injected/               <- output BAMs (dict mode)
#     generated/              <- output BAMs (cgan/mlp modes)
#     logs/                   <- SLURM logs
#
# All individual SLURM scripts must be in the same directory as this script.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Usage check ----
if [ "$#" -lt 8 ]; then
    echo "Usage: bash kinsim_pipeline.sh <mode> <manifest> \\"
    echo "           <shards_dir> <master_pkl> \\"
    echo "           <pbsim3_dir> <motifs> <M> <output_dir> \\"
    echo "           [checkpoint_dir]   # cgan/mlp mode only"
    echo ""
    echo "Modes:  dictionary | cgan | mlp"
    echo "N (number of strains) is auto-detected from manifest via 'kinsim manifest count'."
    echo ""
    echo "Example (mlp):"
    echo "  bash kinsim_pipeline.sh mlp \\"
    echo "      manifest.csv \\"
    echo "      shards/ master_data.pkl \\"
    echo "      pbsim3_output/ \"m6A,GATC,1\" 10 generated_mlp/ \\"
    echo "      checkpoints_mlp/"
    exit 1
fi

MODE="$1"
MANIFEST="$2"
SHARDS_DIR="$3"
MASTER_PKL="$4"
PBSIM3_DIR="$5"
MOTIFS="$6"
NUM_GENOMES="$7"
OUTPUT_DIR="$8"
CHECKPOINT_DIR="${9:-}"

# Auto-detect number of strains from manifest (skips # comments and blank rows)
NUM_STRAINS=$(kinsim manifest count "$MANIFEST")
echo "  Auto-detected N strains: $NUM_STRAINS  (from: $MANIFEST)"

if [ "$MODE" != "dictionary" ] && [ "$MODE" != "cgan" ] && [ "$MODE" != "mlp" ]; then
    echo "ERROR: mode must be 'dictionary', 'cgan', or 'mlp' (got '$MODE')"
    exit 1
fi

if { [ "$MODE" = "cgan" ] || [ "$MODE" = "mlp" ]; } && [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: $MODE mode requires an 10th argument: checkpoint_dir"
    exit 1
fi

mkdir -p logs

echo "============================================================"
echo "  KinSim pipeline — mode: $MODE"
echo "============================================================"
echo "  Manifest:           $MANIFEST  ($NUM_STRAINS strains)"
echo "  Shards dir:         $SHARDS_DIR"
echo "  Master pkl:         $MASTER_PKL"
echo "  PBSIM3 dir:         $PBSIM3_DIR"
echo "  Motifs:             $MOTIFS"
echo "  M genomes (array):  $NUM_GENOMES"
echo "  Output dir:         $OUTPUT_DIR"
if [ "$MODE" = "cgan" ] || [ "$MODE" = "mlp" ]; then
    echo "  Checkpoint dir:     $CHECKPOINT_DIR"
fi
echo ""

# ============================================================
# DICTIONARY MODE
# ============================================================
if [ "$MODE" = "dictionary" ]; then

    # STEP 1 — Prepare: pairs file -> config
    echo "[1/5] Submitting: kinsim prepare"
    CONFIG_OUT="${SHARDS_DIR%/}/config_strains.txt"
    JOB_PREPARE=$(sbatch --parsable \
        "$SCRIPT_DIR/kinsim_prepare.slurm" \
        "$MANIFEST" "$CONFIG_OUT")
    echo "      Job ID: $JOB_PREPARE  (config → $CONFIG_OUT)"

    # STEP 2 — Train shards (array, depends on prepare)
    # KINSIM_NO_AUTOMERGE=1 prevents the last array task from also auto-submitting
    # a merge, since we submit the merge explicitly to chain further jobs.
    echo "[2/5] Submitting: kinsim dictionary train (array 1-$NUM_STRAINS)"
    JOB_TRAIN=$(sbatch --parsable \
        --array="1-${NUM_STRAINS}" \
        --dependency="afterok:${JOB_PREPARE}" \
        --export="KINSIM_NO_AUTOMERGE=1,ALL" \
        "$SCRIPT_DIR/kinsim_train.slurm" \
        "$CONFIG_OUT" "$SHARDS_DIR" "$MASTER_PKL")
    echo "      Job ID: $JOB_TRAIN"

    # STEP 3 — Merge shards (single, depends on all train tasks)
    echo "[3/5] Submitting: kinsim dictionary merge"
    JOB_MERGE=$(sbatch --parsable \
        --dependency="afterok:${JOB_TRAIN}" \
        "$SCRIPT_DIR/kinsim_train.slurm" \
        "$CONFIG_OUT" "$SHARDS_DIR" "$MASTER_PKL")
    echo "      Job ID: $JOB_MERGE"

    # STEP 4 — Inject signals (array, depends on merge)
    echo "[4/5] Submitting: kinsim dictionary inject (array 1-$NUM_GENOMES)"
    JOB_INJECT=$(sbatch --parsable \
        --array="1-${NUM_GENOMES}" \
        --dependency="afterok:${JOB_MERGE}" \
        "$SCRIPT_DIR/kinsim_inject.slurm" \
        "$PBSIM3_DIR" "$MASTER_PKL" "$MOTIFS" "$OUTPUT_DIR")
    echo "      Job ID: $JOB_INJECT"

    # STEP 5 — Analyze dictionary (single, depends on merge)
    echo "[5/5] Submitting: kinsim dictionary analyze"
    JOB_ANALYZE=$(sbatch --parsable \
        --dependency="afterok:${JOB_MERGE}" \
        "$SCRIPT_DIR/kinsim_analyze.slurm" \
        "$MASTER_PKL")
    echo "      Job ID: $JOB_ANALYZE"

    echo ""
    echo "============================================================"
    echo "  Dictionary pipeline submitted"
    echo "============================================================"
    echo "  prepare:  $JOB_PREPARE"
    echo "  train:    $JOB_TRAIN  (array 1-$NUM_STRAINS)"
    echo "  merge:    $JOB_MERGE"
    echo "  inject:   $JOB_INJECT  (array 1-$NUM_GENOMES)"
    echo "  analyze:  $JOB_ANALYZE"
    echo ""
    echo "  Monitor: squeue -u \$USER"
    echo "  Output BAMs: $OUTPUT_DIR"
    echo "  Dictionary:  $MASTER_PKL"

# ============================================================
# cGAN MODE
# ============================================================
elif [ "$MODE" = "cgan" ]; then

    # STEP 1 — Extract shards from manifest (array, manifest-based)
    echo "[1/5] Submitting: kinsim extract (array 1-$NUM_STRAINS)"
    JOB_EXTRACT=$(sbatch --parsable \
        --array="1-${NUM_STRAINS}" \
        --export="KINSIM_NO_AUTOMERGE=1,ALL" \
        "$SCRIPT_DIR/kinsim_extract.slurm" \
        "$MANIFEST" "$SHARDS_DIR" "$MASTER_PKL")
    echo "      Job ID: $JOB_EXTRACT"

    # STEP 2 — Merge shards (single, depends on all extract tasks)
    echo "[2/5] Submitting: kinsim merge"
    JOB_MERGE=$(sbatch --parsable \
        --dependency="afterok:${JOB_EXTRACT}" \
        "$SCRIPT_DIR/kinsim_extract.slurm" \
        "$MANIFEST" "$SHARDS_DIR" "$MASTER_PKL")
    echo "      Job ID: $JOB_MERGE"

    # STEP 3 — Train cGAN (single GPU, depends on merge)
    echo "[3/5] Submitting: kinsim cgan train"
    JOB_TRAIN=$(sbatch --parsable \
        --dependency="afterok:${JOB_MERGE}" \
        "$SCRIPT_DIR/kinsim_cgan_train.slurm" \
        "$MASTER_PKL" "$CHECKPOINT_DIR")
    echo "      Job ID: $JOB_TRAIN"

    # STEP 4 — Generate signals (array, depends on train)
    # Default 100 training epochs → checkpoint_epoch100.pt
    CKPT_FILE="${CHECKPOINT_DIR}/checkpoint_epoch100.pt"
    echo "[4/5] Submitting: kinsim cgan generate (array 1-$NUM_GENOMES)"
    JOB_GENERATE=$(sbatch --parsable \
        --array="1-${NUM_GENOMES}" \
        --dependency="afterok:${JOB_TRAIN}" \
        "$SCRIPT_DIR/kinsim_cgan_generate.slurm" \
        "$PBSIM3_DIR" "$CKPT_FILE" "$MOTIFS" "$OUTPUT_DIR")
    echo "      Job ID: $JOB_GENERATE"

    echo ""
    echo "============================================================"
    echo "  cGAN pipeline submitted (4 steps)"
    echo "============================================================"
    echo "  extract:   $JOB_EXTRACT  (array 1-$NUM_STRAINS)"
    echo "  merge:     $JOB_MERGE"
    echo "  train:     $JOB_TRAIN"
    echo "  generate:  $JOB_GENERATE  (array 1-$NUM_GENOMES)"
    echo ""
    echo "  Monitor:       squeue -u \$USER"
    echo "  Output BAMs:   $OUTPUT_DIR"
    echo "  Checkpoints:   $CHECKPOINT_DIR"
    echo "  Expected ckpt: $CKPT_FILE"
    echo "  Monitor GAN:   tensorboard --logdir $CHECKPOINT_DIR/runs"

# ============================================================
# MLP MODE
# ============================================================
elif [ "$MODE" = "mlp" ]; then

    # STEP 1 — Extract shards from manifest (array, manifest-based)
    echo "[1/5] Submitting: kinsim extract (array 1-$NUM_STRAINS)"
    JOB_EXTRACT=$(sbatch --parsable \
        --array="1-${NUM_STRAINS}" \
        --export="KINSIM_NO_AUTOMERGE=1,ALL" \
        "$SCRIPT_DIR/kinsim_extract.slurm" \
        "$MANIFEST" "$SHARDS_DIR" "$MASTER_PKL")
    echo "      Job ID: $JOB_EXTRACT"

    # STEP 2 — Merge shards (single, depends on all extract tasks)
    echo "[2/5] Submitting: kinsim merge"
    JOB_MERGE=$(sbatch --parsable \
        --dependency="afterok:${JOB_EXTRACT}" \
        "$SCRIPT_DIR/kinsim_extract.slurm" \
        "$MANIFEST" "$SHARDS_DIR" "$MASTER_PKL")
    echo "      Job ID: $JOB_MERGE"

    # STEP 3 — Train MLP (single GPU, depends on merge)
    echo "[3/5] Submitting: kinsim mlp train"
    JOB_TRAIN=$(sbatch --parsable \
        --dependency="afterok:${JOB_MERGE}" \
        "$SCRIPT_DIR/kinsim_mlp_train.slurm" \
        "$MASTER_PKL" "$CHECKPOINT_DIR")
    echo "      Job ID: $JOB_TRAIN"

    # STEP 4 — Generate signals (array, depends on train)
    # Default 50 training epochs → checkpoint_epoch50.pt
    CKPT_FILE="${CHECKPOINT_DIR}/checkpoint_epoch50.pt"
    echo "[4/5] Submitting: kinsim mlp generate (array 1-$NUM_GENOMES)"
    JOB_GENERATE=$(sbatch --parsable \
        --array="1-${NUM_GENOMES}" \
        --dependency="afterok:${JOB_TRAIN}" \
        "$SCRIPT_DIR/kinsim_mlp_generate.slurm" \
        "$PBSIM3_DIR" "$CKPT_FILE" "$MOTIFS" "$OUTPUT_DIR")
    echo "      Job ID: $JOB_GENERATE"

    echo ""
    echo "============================================================"
    echo "  MLP pipeline submitted (4 steps)"
    echo "============================================================"
    echo "  extract:   $JOB_EXTRACT  (array 1-$NUM_STRAINS)"
    echo "  merge:     $JOB_MERGE"
    echo "  train:     $JOB_TRAIN"
    echo "  generate:  $JOB_GENERATE  (array 1-$NUM_GENOMES)"
    echo ""
    echo "  Monitor:       squeue -u \$USER"
    echo "  Output BAMs:   $OUTPUT_DIR"
    echo "  Checkpoints:   $CHECKPOINT_DIR"
    echo "  Expected ckpt: $CKPT_FILE"
    echo "  Monitor MLP:   tensorboard --logdir $CHECKPOINT_DIR/runs"

fi
