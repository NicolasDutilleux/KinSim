#!/bin/bash
# ============================================================
# Strepto prep orchestrator — modular pipeline
#
# Uses manifest_strepto.csv for sample→bam mapping. References at
# ${STREPTO}/<sample_id>/final_assembly.fasta (pre-existing, no hifiasm).
#
# Per-sample chain:
#   raw BAM
#     └─► bystrandify ──► align_pbmm2 ──► index_bam
#                                             │
#                               ┌─────────────┴─────────────┐
#                               ▼                           ▼
#                         ipdsummary → pbmotifmaker    jasmine_modkit
#                               │                           │
#                               └──────────┬────────────────┘
#                                          ▼
#                                    merge_motifs (threshold 0.7)
#
# Note: Strepto raw BAMs already carry production jasmine 5mC MM tags
# (Revio GPU default). jasmine_modkit detects this and skips the re-call.
#
# Usage:
#   bash slurm_kinsim/strepto/run.sh all
#   bash slurm_kinsim/strepto/run.sh call <sample_id>
#   bash slurm_kinsim/strepto/run.sh manifest
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
MANIFEST_IN=${STREPTO}/manifest_strepto.csv
HERE=$(dirname "$(readlink -f "$0")")
ROOT=$(dirname "$HERE")
MERGE_THRESHOLD=${MERGE_THRESHOLD:-0.7}

[ -s "$MANIFEST_IN" ] || { echo "ERROR: input manifest missing: $MANIFEST_IN"; exit 1; }

# Resolve raw BAM for a sample via the manifest
lookup_bam() {
    local sample=$1
    python3 -c "
from kinsim.utils.config import load_manifest
for e in load_manifest('${MANIFEST_IN}'):
    if e.sample_id == '${sample}':
        print(e.bam_path); break
"
}

chain_one() {
    local sample=$1
    local raw_bam
    raw_bam=$(lookup_bam "$sample")
    [ -n "$raw_bam" ] || { echo "ERROR: sample $sample not in $MANIFEST_IN" >&2; return 1; }
    [ -f "$raw_bam" ] || { echo "ERROR: raw BAM missing: $raw_bam" >&2; return 1; }

    local ref="${STREPTO}/${sample}/final_assembly.fasta"
    [ -f "$ref" ] || { echo "ERROR: ref missing: $ref" >&2; return 1; }

    local out="${STREPTO}/pipeline/${sample}"
    mkdir -p "$out" "${out}/jasmine_work"

    # prep (no assembly for Strepto)
    local J_BY=$(sbatch --parsable -J ${sample}_by \
        "${ROOT}/prep/bystrandify.slurm" "$raw_bam" "${out}/${sample}_bystrandify.bam")
    local J_AL=$(sbatch --parsable --dependency=afterok:$J_BY -J ${sample}_aln \
        "${ROOT}/prep/align_pbmm2.slurm" "${out}/${sample}_bystrandify.bam" "$ref" "${out}/${sample}_aligned.bam")
    local J_IX=$(sbatch --parsable --dependency=afterok:$J_AL -J ${sample}_idx \
        "${ROOT}/prep/index_bam.slurm" "${out}/${sample}_aligned.bam")

    # callers (parallel)
    local J_IPD=$(sbatch --parsable --dependency=afterok:$J_IX -J ${sample}_ipd \
        "${ROOT}/callers/ipdsummary.slurm" \
        "${out}/${sample}_aligned.bam" "$ref" \
        "${out}/${sample}_ipdSummary.gff" "${out}/${sample}_ipdSummary.csv")
    local J_MM=$(sbatch --parsable --dependency=afterok:$J_IPD -J ${sample}_mm \
        "${ROOT}/callers/pbmotifmaker.slurm" \
        "$ref" "${out}/${sample}_ipdSummary.gff" "${out}/${sample}_motifs_ipdsummary.csv")
    local J_JM=$(sbatch --parsable -J ${sample}_jm \
        "${ROOT}/callers/jasmine_modkit.slurm" \
        "$raw_bam" "$ref" "${out}/${sample}_motifs_jasmine.csv" "${out}/jasmine_work")

    # merge
    local J_MG=$(sbatch --parsable --dependency=afterok:${J_MM}:${J_JM} -J ${sample}_merge \
        "${ROOT}/callers/merge_motifs.slurm" \
        "${out}/${sample}_motifs_merged.csv" "$MERGE_THRESHOLD" \
        "${out}/${sample}_motifs_ipdsummary.csv" "${out}/${sample}_motifs_jasmine.csv")

    printf '  %-12s :  by=%s  aln=%s  idx=%s  ipd=%s  mm=%s  jm=%s  MERGE=%s\n' \
        "$sample" "$J_BY" "$J_AL" "$J_IX" "$J_IPD" "$J_MM" "$J_JM" "$J_MG"
    echo "$J_MG"
}

submit_manifest() {
    local deps=$1
    local d=""
    [ -n "$deps" ] && d="--dependency=afterany:${deps}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:15:00 \
        --job-name=strepto_manifest \
        --output=/data/projects/p774_MARSD/NDutilleux/logs/strepto_manifest_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && bash ${HERE}/build_manifest.sh"
}

STEP=${1:-}
case "$STEP" in
    call)
        SAMPLE=${2:?"Usage: bash $0 call <sample_id>   e.g. bc2071"}
        chain_one "$SAMPLE"
        ;;
    all)
        SAMPLES=$(python3 -c "
from kinsim.utils.config import load_manifest
for e in load_manifest('${MANIFEST_IN}'):
    print(e.sample_id)
")
        N=$(echo "$SAMPLES" | wc -l)
        echo "Launching modular pipeline for $N samples (threshold=$MERGE_THRESHOLD)..."
        echo ""
        MERGE_JIDS=()
        while read -r sample; do
            [ -z "$sample" ] && continue
            jid=$(chain_one "$sample") || { echo "SKIP $sample" >&2; continue; }
            MERGE_JIDS+=("$jid")
        done <<< "$SAMPLES"
        echo ""
        DEP=$(IFS=:; echo "${MERGE_JIDS[*]}")
        JM=$(submit_manifest "$DEP")
        echo "strepto.manifest: $JM  (after ${#MERGE_JIDS[@]} merges)"
        ;;
    manifest)
        J=$(submit_manifest "")
        echo "strepto.manifest (no deps): $J"
        ;;
    *)
        N=$(python3 -c "from kinsim.utils.config import load_manifest; print(len(load_manifest('${MANIFEST_IN}')))")
        cat <<EOF
Usage: bash slurm_kinsim/strepto/run.sh <command>

Manifest: $MANIFEST_IN ($N samples)

Commands:
  call <sample_id>    Chain full pipeline for one sample (e.g. bc2071)
  all                 Chain pipeline for all samples in manifest (parallel)
  manifest            Build manifest_strepto_merged.csv from existing merged CSVs

Env vars:
  MERGE_THRESHOLD     Minimum fraction to keep in merge (default 0.7)

Output per sample: \${STREPTO}/pipeline/<sample>/
  <sample>_bystrandify.bam
  <sample>_aligned.bam + .bai + .pbi
  <sample>_ipdSummary.{gff,csv}
  <sample>_motifs_ipdsummary.csv
  <sample>_motifs_jasmine.csv
  <sample>_motifs_merged.csv     ← what KinSim manifest points to
EOF
        exit 1
        ;;
esac
