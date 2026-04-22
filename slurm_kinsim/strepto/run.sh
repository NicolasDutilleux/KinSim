#!/bin/bash
# ============================================================
# Strepto prep orchestrator — chain 00_bystrandify → 04_motifmaker → 05_manifest
#
# Manifest-driven: array size comes from ${STREPTO}/manifest_strepto.csv
# (sample_id, bam_path, motifs). References are picked per-sample from
# ${STREPTO}/<sample_id>/final_assembly.fasta.
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
MANIFEST=${STREPTO}/manifest_strepto.csv
HERE=$(dirname "$(readlink -f "$0")")
CONCURRENT=4

[ -s "$MANIFEST" ] || { echo "ERROR: manifest missing: $MANIFEST"; exit 1; }
N=$(kinsim-prep manifest count "$MANIFEST")
[ "$N" -gt 0 ] || { echo "ERROR: manifest has 0 rows"; exit 1; }

submit() {
    local step=$1; local dep=${2:-}; local d=""
    [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d --array=1-${N}%${CONCURRENT} \
        "${HERE}/${step}.slurm"
}

submit_manifest() {
    local dep=${1:-}; local d=""
    [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
        --job-name=strepto_05_manifest \
        --output=/data/projects/p774_MARSD/NDutilleux/logs/strepto_05_manifest_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && bash ${HERE}/05_build_manifest.sh"
}

STEP=${1:-}
case "$STEP" in
    00|bystrandify) J=$(submit 00_bystrandify); echo "strepto.00 bystrandify: $J" ;;
    01|align)       J=$(submit 01_align);       echo "strepto.01 align:       $J" ;;
    02|index)       J=$(submit 02_index);       echo "strepto.02 index:       $J" ;;
    03|ipd)         J=$(submit 03_ipdsummary);  echo "strepto.03 ipdSummary:  $J" ;;
    04|motif)       J=$(submit 04_motifmaker);  echo "strepto.04 motifmaker:  $J" ;;
    manifest)       J=$(submit_manifest);       echo "strepto.05 manifest:    $J" ;;
    all)
        J0=$(submit 00_bystrandify);       echo "strepto.00 bystrandify: $J0"
        J1=$(submit 01_align       "$J0"); echo "strepto.01 align:       $J1 (after $J0)"
        J2=$(submit 02_index       "$J1"); echo "strepto.02 index:       $J2 (after $J1)"
        J3=$(submit 03_ipdsummary  "$J2"); echo "strepto.03 ipdSummary:  $J3 (after $J2)"
        J4=$(submit 04_motifmaker  "$J3"); echo "strepto.04 motifmaker:  $J4 (after $J3)"
        JM=$(submit_manifest       "$J4"); echo "strepto.05 manifest:    $JM (after $J4)"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/strepto/run.sh <step>

Discovered $N samples from ${MANIFEST}

Steps:
  00 | bystrandify    ccs-kinetics-bystrandify         (array 1-${N}%${CONCURRENT})
  01 | align          pbmm2 align                      (array)
  02 | index          samtools index + pbindex         (array)
  03 | ipd            ipdSummary SP3-C3                (array)
  04 | motif          pbmotifmaker                     (array)
  manifest            build manifest_strepto_gff.csv
  all                 chain 00 → 04 → manifest
EOF
        exit 1
        ;;
esac
