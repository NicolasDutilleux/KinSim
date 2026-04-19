#!/bin/bash
# ============================================================
# KinSim — Full Sequel II 48-plex GFF pipeline
#
# Sequel data is already processed up through ipdSummary:
#   Sequel/ipdsummary/bcXXXX/bcXXXX_subreads_aligned.bam   (ip/pw tags)
#   Sequel/ipdsummary/bcXXXX/bcXXXX_ipdSummary.gff
#
# This script only runs the KinSim steps:
#   1. Build GFF manifest
#   2. Extract (array, one task per barcode)
#   3. Merge shards into master_sequel_gff.pkl
#
# Usage:
#   bash slurm_kinsim/run_sequel_pipeline.sh
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

BASE=/data/projects/p774_MARSD/NDutilleux/training
SEQUEL=${BASE}/Sequel
MANIFEST=${SEQUEL}/manifest_sequel_gff.csv
SHARDS=${SEQUEL}/shards_gff
MASTER=${SEQUEL}/master_sequel_gff.pkl
LOGS=${BASE}/logs

EXTRACT_CONCURRENT=4

echo "========================================================"
echo "  Sequel II 48-plex GFF pipeline"
echo "========================================================"
echo ""

# ============================================================
# STEP 1: Build GFF manifest (local, fast — no SLURM)
# ============================================================
echo "=== STEP 1: Build GFF manifest ==="
bash slurm_kinsim/sequel_04_build_manifest.sh
echo ""

if [ ! -s "$MANIFEST" ]; then
    echo "ERROR: Manifest not produced or empty: $MANIFEST"
    exit 1
fi

N=$(kinsim-prep manifest count "$MANIFEST")
echo "Sequel manifest: $N species"
echo ""

# ============================================================
# STEP 2: Extract (array job)
# ============================================================
echo "=== STEP 2: Extract ==="

EXTRACT_JOB=$(KINSIM_NO_AUTOMERGE=1 sbatch --parsable \
    --array=1-${N}%${EXTRACT_CONCURRENT} \
    slurm_kinsim/00_extract.slurm \
    "$MANIFEST" "$SHARDS" "$MASTER")
echo "  Job: $EXTRACT_JOB (array 1-$N, max $EXTRACT_CONCURRENT concurrent)"
echo ""

# ============================================================
# STEP 3: Merge shards
# ============================================================
echo "=== STEP 3: Merge ==="

MERGE_JOB=$(sbatch --parsable \
    --dependency=afterany:${EXTRACT_JOB} \
    slurm_kinsim/00_extract.slurm \
    "$MANIFEST" "$SHARDS" "$MASTER")
echo "  Job: $MERGE_JOB (after extract)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo "  Sequel pipeline submitted"
echo "=========================================="
echo "  1. Build manifest: (done, local)"
echo "  2. Extract:        $EXTRACT_JOB (array 1-$N)"
echo "  3. Merge:          $MERGE_JOB"
echo "=========================================="
echo ""
echo "Manifest:   $MANIFEST"
echo "Shards:     $SHARDS"
echo "Master pkl: $MASTER"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ls -lt ${LOGS}/kinsim_extract_*.log | head"
