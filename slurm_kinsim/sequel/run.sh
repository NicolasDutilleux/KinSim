#!/bin/bash
# ============================================================
# Sequel prep orchestrator — chain 00_ccs → 05_motifmaker → 06_manifest
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
HERE=$(dirname "$(readlink -f "$0")")
CONCURRENT=4

# Discover array size from available subread BAMs (matches per-step scripts)
N=$(ls "${SEQUEL}"/lima.bc*--bc*.bam.gz 2>/dev/null | wc -l)
[ "$N" -gt 0 ] || { echo "ERROR: no subread .bam.gz files in $SEQUEL"; exit 1; }

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
        --job-name=sequel_06_manifest \
        --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_06_manifest_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && bash ${HERE}/06_build_manifest.sh"
}

STEP=${1:-}
case "$STEP" in
    00|ccs)         J=$(submit 00_ccs);         echo "sequel.00 ccs:         $J" ;;
    01|bystrandify) J=$(submit 01_bystrandify); echo "sequel.01 bystrandify: $J" ;;
    02|align)       J=$(submit 02_align);       echo "sequel.02 align:       $J" ;;
    03|index)       J=$(submit 03_index);       echo "sequel.03 index:       $J" ;;
    04|ipd)         J=$(submit 04_ipdsummary);  echo "sequel.04 ipdSummary:  $J" ;;
    05|motif)       J=$(submit 05_motifmaker);  echo "sequel.05 motifmaker:  $J" ;;
    manifest)       J=$(submit_manifest);       echo "sequel.06 manifest:    $J" ;;
    all)
        J0=$(submit 00_ccs);                echo "sequel.00 ccs:         $J0"
        J1=$(submit 01_bystrandify "$J0");  echo "sequel.01 bystrandify: $J1 (after $J0)"
        J2=$(submit 02_align       "$J1");  echo "sequel.02 align:       $J2 (after $J1)"
        J3=$(submit 03_index       "$J2");  echo "sequel.03 index:       $J3 (after $J2)"
        J4=$(submit 04_ipdsummary  "$J3");  echo "sequel.04 ipdSummary:  $J4 (after $J3)"
        J5=$(submit 05_motifmaker  "$J4");  echo "sequel.05 motifmaker:  $J5 (after $J4)"
        JM=$(submit_manifest       "$J5");  echo "sequel.06 manifest:    $JM (after $J5)"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/sequel/run.sh <step>

Discovered $N barcodes from ${SEQUEL}/lima.bc*--bc*.bam.gz

Steps:
  00 | ccs            subreads → HiFi                  (array 1-${N}%${CONCURRENT})
  01 | bystrandify    ccs-kinetics-bystrandify          (array)
  02 | align          pbmm2 align                       (array)
  03 | index          samtools index + pbindex          (array)
  04 | ipd            ipdSummary SP3-C3                 (array)
  05 | motif          pbmotifmaker                      (array)
  manifest            build manifest_sequel_ccs_gff.csv
  all                 chain 00 → 05 → manifest
EOF
        exit 1
        ;;
esac
