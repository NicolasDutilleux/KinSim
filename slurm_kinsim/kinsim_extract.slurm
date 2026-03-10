#!/bin/bash
#SBATCH --job-name=KinSim_Extract
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_extract_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim — Extract raw kinetic samples + merge (manifest-based, two-phase)
#
# Replaces the legacy kinsim_cgan_extract.slurm.
# Uses a manifest CSV (sample_id, bam_path, motifs) instead of an
# alternating-line config file — cleaner, self-documenting, easier to audit.
#
# Manifest format (manifest.csv):
#   sample_id,bam_path,motifs
#   strain1,/data/bams/strain1.bam,"m6A,GATC,1"
#   strain2,/data/bams/strain2.bam,"m6A,GATC,1;m4C,CCWGG,1"
#   strain3,/data/bams/strain3.bam,/data/motifs/strain3.csv
#
# Generate a manifest from an existing pairs file:
#   kinsim prepare --to-manifest pairs.txt manifest.csv
# (or create manually — CSV with headers sample_id, bam_path, motifs)
#
# Usage — Extract phase (array job, one task per manifest row):
#   sbatch --array=1-N kinsim_extract.slurm <manifest.csv> <shards_dir> <master_output>
#
# Usage — Merge phase (single job, triggered automatically or manually):
#   sbatch kinsim_extract.slurm <manifest.csv> <shards_dir> <master_output>
#
# Arguments:
#   $1  Manifest CSV (sample_id, bam_path, motifs)
#   $2  Directory for *_shard.pkl intermediate files
#   $3  Path for the final merged master training data (.pkl)
#
# Count N for --array:
#   N=$(kinsim manifest count manifest.csv)
#   sbatch --array=1-$N kinsim_extract.slurm manifest.csv shards/ master_data.pkl
#
# Examples:
#   # Submit extract + auto-merge (standalone, no pipeline):
#   N=$(kinsim manifest count manifest.csv)
#   sbatch --array=1-$N kinsim_extract.slurm manifest.csv shards/ master_data.pkl
#
#   # Submit via kinsim_pipeline.sh (handles merge dependency automatically):
#   bash kinsim_pipeline.sh mlp manifest.csv shards/ master_data.pkl ...
#
# Output consumed by:
#   kinsim_mlp_train.slurm     (trains the MLPPredictor)
#   kinsim_cgan_train.slurm    (trains the Generator + Discriminator)
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: sbatch --array=1-N kinsim_extract.slurm <manifest.csv> <shards_dir> <master_output>"
    echo ""
    echo "Arguments:"
    echo "  manifest.csv    CSV with columns: sample_id, bam_path, motifs"
    echo "  shards_dir      Directory for intermediate *_shard.pkl files"
    echo "  master_output   Path for the final merged training data (.pkl)"
    echo ""
    echo "Count N:  kinsim manifest count manifest.csv"
    echo ""
    echo "Example:"
    echo "  N=\$(kinsim manifest count manifest.csv)"
    echo "  sbatch --array=1-\$N kinsim_extract.slurm manifest.csv shards/ master_data.pkl"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  KinSim Extract"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Array ID:   ${SLURM_ARRAY_TASK_ID:-none}"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory:     ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Conda env:  ${CONDA_DEFAULT_ENV:-not set}"
echo "Python:     $(which python)"
echo "Python ver: $(python --version 2>&1)"
echo "KinSim ver: $(kinsim --version 2>&1 || echo 'NOT FOUND')"
echo "========================================================"
echo ""

# ---- Config ----
MANIFEST="$1"
SHARDS_DIR="$2"
MASTER_OUTPUT="$3"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest file '$MANIFEST' not found."
    exit 1
fi

# ---- Phase detection ----
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # ============ PHASE 1: Extract one shard (one manifest row per task) ============
    mkdir -p "$SHARDS_DIR"

    echo "=== PHASE 1: Extract (task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX) ==="
    echo "Manifest: $MANIFEST"
    echo "Shards:   $SHARDS_DIR"
    echo ""

    # kinsim extract reads manifest row $SLURM_ARRAY_TASK_ID (1-indexed)
    # Output: shards/<sample_id>_shard.pkl (name comes from manifest)
    kinsim extract \
        --manifest "$MANIFEST" \
        --task     "$SLURM_ARRAY_TASK_ID" \
        --output-dir "$SHARDS_DIR"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: kinsim extract failed with exit code $EXIT_CODE"
        echo "Ended: $(date '+%Y-%m-%d %H:%M:%S')"
        exit $EXIT_CODE
    fi

    # The last array task auto-submits the merge job.
    # Suppressed when KINSIM_NO_AUTOMERGE=1 (used by kinsim_pipeline.sh, which
    # submits the merge explicitly to chain further dependency jobs).
    if [ "$SLURM_ARRAY_TASK_ID" -eq "$SLURM_ARRAY_TASK_MAX" ] && [ -z "${KINSIM_NO_AUTOMERGE}" ]; then
        echo ""
        echo "Last array task — submitting merge job (dependency: afterok:${SLURM_ARRAY_JOB_ID})..."
        sbatch --dependency=afterok:"${SLURM_ARRAY_JOB_ID}" \
            "$0" "$MANIFEST" "$SHARDS_DIR" "$MASTER_OUTPUT"
    fi

else
    # ============ PHASE 2: Merge all shards ============
    echo "=== PHASE 2: Merge ==="
    echo "Shards dir:    $SHARDS_DIR"
    echo "Master output: $MASTER_OUTPUT"
    echo ""

    # List shards found
    SHARD_COUNT=$(ls "$SHARDS_DIR"/*_shard.pkl 2>/dev/null | wc -l)
    echo "Shards found:  $SHARD_COUNT"
    ls -lh "$SHARDS_DIR"/*_shard.pkl 2>/dev/null || echo "(no *_shard.pkl files)"
    echo ""

    mkdir -p "$(dirname "$MASTER_OUTPUT")"

    # Auto-detects *_shard.pkl (new) then *_cgan.pkl (legacy)
    kinsim merge "$SHARDS_DIR" "$MASTER_OUTPUT"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: kinsim merge failed with exit code $EXIT_CODE"
        echo "Ended: $(date '+%Y-%m-%d %H:%M:%S')"
        exit $EXIT_CODE
    fi

    echo ""
    echo "Master file size: $(ls -lh "$MASTER_OUTPUT" | awk '{print $5}')"
    echo "=== Merge complete: $MASTER_OUTPUT ==="
fi

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
