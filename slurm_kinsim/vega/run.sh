#!/bin/bash
# ============================================================
# Vega prep orchestrator — chain 00_assembly → 05_motifmaker → 06_manifest
#
# Per-step submit (echoes jobid):
#   bash slurm_kinsim/vega/run.sh 00           # assembly
#   bash slurm_kinsim/vega/run.sh 01           # bystrandify
#   bash slurm_kinsim/vega/run.sh 02           # align
#   bash slurm_kinsim/vega/run.sh 03           # index
#   bash slurm_kinsim/vega/run.sh 04           # ipdSummary
#   bash slurm_kinsim/vega/run.sh 05           # motifmaker
#   bash slurm_kinsim/vega/run.sh manifest     # build GFF manifest
#
# Full chain:
#   bash slurm_kinsim/vega/run.sh all
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

N_BARCODES=16
CONCURRENT=4
HERE=$(dirname "$(readlink -f "$0")")

submit() {
    local step=$1; local dep=${2:-}; local d=""
    [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d --array=1-${N_BARCODES}%${CONCURRENT} \
        "${HERE}/${step}.slurm"
}

submit_manifest() {
    local dep=${1:-}; local d=""
    [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
        --job-name=vega_06_manifest \
        --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_06_manifest_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && bash ${HERE}/06_build_manifest.sh"
}

STEP=${1:-}
case "$STEP" in
    00|assembly)    J=$(submit 00_assembly);    echo "vega.00 assembly:    $J" ;;
    01|bystrandify) J=$(submit 01_bystrandify); echo "vega.01 bystrandify: $J" ;;
    02|align)       J=$(submit 02_align);       echo "vega.02 align:       $J" ;;
    03|index)       J=$(submit 03_index);       echo "vega.03 index:       $J" ;;
    04|ipd)         J=$(submit 04_ipdsummary);  echo "vega.04 ipdSummary:  $J" ;;
    05|motif)       J=$(submit 05_motifmaker);  echo "vega.05 motifmaker:  $J" ;;
    manifest)       J=$(submit_manifest);       echo "vega.06 manifest:    $J" ;;
    all)
        J0=$(submit 00_assembly);           echo "vega.00 assembly:    $J0"
        J1=$(submit 01_bystrandify "$J0");  echo "vega.01 bystrandify: $J1 (after $J0)"
        J2=$(submit 02_align       "$J1");  echo "vega.02 align:       $J2 (after $J1)"
        J3=$(submit 03_index       "$J2");  echo "vega.03 index:       $J3 (after $J2)"
        J4=$(submit 04_ipdsummary  "$J3");  echo "vega.04 ipdSummary:  $J4 (after $J3)"
        J5=$(submit 05_motifmaker  "$J4");  echo "vega.05 motifmaker:  $J5 (after $J4)"
        JM=$(submit_manifest       "$J5");  echo "vega.06 manifest:    $JM (after $J5)"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/vega/run.sh <step>

Steps:
  00 | assembly       hifiasm draft assembly   (array 1-${N_BARCODES}%${CONCURRENT})
  01 | bystrandify    ccs-kinetics-bystrandify (array)
  02 | align          pbmm2 align              (array)
  03 | index          samtools index + pbindex (array)
  04 | ipd            ipdSummary SP3-C3        (array)
  05 | motif          pbmotifmaker             (array)
  manifest            build manifest_vega_gff.csv
  all                 chain 00 → 05 → manifest with --dependency=afterok

After manifest: run ML pipeline via 'bash slurm_kinsim/ml/run.sh all <manifest> <out_prefix>'
EOF
        exit 1
        ;;
esac
