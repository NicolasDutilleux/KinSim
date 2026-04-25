#!/bin/bash
# ============================================================
# Vega prep orchestrator — modular pipeline
#
# Per-sample chain using slurm_kinsim/prep/ + slurm_kinsim/callers/:
#
#   raw BAM
#     ├─► assembly_hifiasm ──► reference FASTA
#     │                             │
#     └─► bystrandify ──► align_pbmm2 ──► index_bam
#                                             │
#                               ┌─────────────┴─────────────┐
#                               ▼                           ▼
#                         ipdsummary → pbmotifmaker    jasmine_modkit
#                               │                           │
#                               └──────────┬────────────────┘
#                                          ▼
#                                    merge_motifs (threshold 0.7)
#                                          │
#                                          ▼
#                              <sample>/motifs_merged.csv
#
# Then build_manifest aggregates all merged CSVs → manifest_vega.csv
#
# Usage:
#   bash slurm_kinsim/vega/run.sh all             # full chain, all barcodes
#   bash slurm_kinsim/vega/run.sh call <bc>       # single barcode, full chain
#   bash slurm_kinsim/vega/run.sh manifest        # build final manifest from existing merged CSVs
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
HERE=$(dirname "$(readlink -f "$0")")
ROOT=$(dirname "$HERE")
MERGE_THRESHOLD=${MERGE_THRESHOLD:-0.7}

# Barcodes present (bc2038 excluded — fragmented assembly / contamination)
BARCODES=(2033 2034 2035 2036 2037 2039 2040 2041
          2042 2043 2044 2045 2046 2047 2048)

# Chain one barcode. Returns the final merge job id.
chain_one() {
    local bc=$1
    local sample="bc${bc}"
    local raw_bam="${VEGA}/m21026_260313_002345.hifi_reads.${sample}.bam"
    local out="${VEGA}/pipeline/${sample}"
    local ref="${out}/${sample}_assembly.fasta"
    mkdir -p "$out" "${out}/jasmine_work"

    [ -f "$raw_bam" ] || { echo "MISSING $raw_bam" >&2; return 1; }

    # prep chain
    local J_ASM=$(sbatch --parsable -J ${sample}_asm \
        "${ROOT}/prep/assembly_hifiasm.slurm" "$raw_bam" "$ref")
    local J_BY=$(sbatch --parsable --dependency=afterok:$J_ASM -J ${sample}_by \
        "${ROOT}/prep/bystrandify.slurm" "$raw_bam" "${out}/${sample}_bystrandify.bam")
    local J_AL=$(sbatch --parsable --dependency=afterok:$J_BY -J ${sample}_aln \
        "${ROOT}/prep/align_pbmm2.slurm" "${out}/${sample}_bystrandify.bam" "$ref" "${out}/${sample}_aligned.bam")
    local J_IX=$(sbatch --parsable --dependency=afterok:$J_AL -J ${sample}_idx \
        "${ROOT}/prep/index_bam.slurm" "${out}/${sample}_aligned.bam")

    # caller chains (parallel)
    local J_IPD=$(sbatch --parsable --dependency=afterok:$J_IX -J ${sample}_ipd \
        "${ROOT}/callers/ipdsummary.slurm" \
        "${out}/${sample}_aligned.bam" "$ref" \
        "${out}/${sample}_ipdSummary.gff" "${out}/${sample}_ipdSummary.csv")
    local J_MM=$(sbatch --parsable --dependency=afterok:$J_IPD -J ${sample}_mm \
        "${ROOT}/callers/pbmotifmaker.slurm" \
        "$ref" "${out}/${sample}_ipdSummary.gff" "${out}/${sample}_motifs_ipdsummary.csv")

    local J_JM=$(sbatch --parsable --dependency=afterok:$J_IX -J ${sample}_jm \
        "${ROOT}/callers/jasmine_modkit.slurm" \
        "$raw_bam" "$ref" "${out}/${sample}_motifs_jasmine.csv" "${out}/jasmine_work")

    # merge
    local J_MG=$(sbatch --parsable --dependency=afterok:${J_MM}:${J_JM} -J ${sample}_merge \
        "${ROOT}/callers/merge_motifs.slurm" \
        "${out}/${sample}_motifs_merged.csv" "$MERGE_THRESHOLD" \
        "${out}/${sample}_motifs_ipdsummary.csv" "${out}/${sample}_motifs_jasmine.csv")

    # Display chain summary on stderr so it doesn't pollute the captured stdout.
    # Only the merge job ID goes to stdout (consumed by `jid=$(chain_one ...)`).
    printf '  %s :  asm=%s  by=%s  aln=%s  idx=%s  ipd=%s  mm=%s  jm=%s  MERGE=%s\n' \
        "$sample" "$J_ASM" "$J_BY" "$J_AL" "$J_IX" "$J_IPD" "$J_MM" "$J_JM" "$J_MG" >&2

    echo "$J_MG"   # final dep
}

submit_manifest() {
    local deps=$1  # comma-separated or colon-separated jobids
    local d=""
    [ -n "$deps" ] && d="--dependency=afterany:${deps}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:15:00 \
        --job-name=vega_manifest \
        --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_manifest_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && bash ${HERE}/build_manifest.sh"
}

STEP=${1:-}
case "$STEP" in
    call)
        BC=${2:?"Usage: bash $0 call <barcode>   e.g. 2046 or bc2046"}
        BC=${BC#bc}
        chain_one "$BC"
        ;;
    all)
        echo "Launching modular pipeline for ${#BARCODES[@]} barcodes..."
        echo "Merge threshold: $MERGE_THRESHOLD"
        echo ""
        MERGE_JIDS=()
        for bc in "${BARCODES[@]}"; do
            jid=$(chain_one "$bc") || continue
            MERGE_JIDS+=("$jid")
        done
        echo ""
        # Build comma-separated dep list
        DEP=$(IFS=:; echo "${MERGE_JIDS[*]}")
        JM=$(submit_manifest "$DEP")
        echo "vega.manifest: $JM  (after all merges: $DEP)"
        ;;
    manifest)
        J=$(submit_manifest "")
        echo "vega.manifest (no deps): $J"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/vega/run.sh <command>

Commands:
  call <barcode>    Chain full pipeline for a single barcode (e.g. 2046)
  all               Chain pipeline for all barcodes in parallel
                    (${#BARCODES[@]} barcodes: ${BARCODES[*]})
  manifest          Build manifest_vega.csv from existing merged CSVs

Env vars:
  MERGE_THRESHOLD   Minimum fraction to keep in merge (default 0.7)

Output per sample: \${VEGA}/pipeline/bcXXXX/
  bcXXXX_assembly.fasta
  bcXXXX_bystrandify.bam
  bcXXXX_aligned.bam + .bai + .pbi
  bcXXXX_ipdSummary.{gff,csv}
  bcXXXX_motifs_ipdsummary.csv
  bcXXXX_motifs_jasmine.csv
  bcXXXX_motifs_merged.csv     ← what KinSim manifest points to

Final: \${VEGA}/manifest_vega.csv (built by build_manifest.sh)
EOF
        exit 1
        ;;
esac
