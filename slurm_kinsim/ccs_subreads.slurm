#!/bin/bash
#SBATCH --job-name=ccs_subreads
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ccs_subreads_%J.log
#
# ccs_subreads — multi-pass PBSIM3 subreads → HiFi BAM with kinetics.
#
# Runs `ccs --hifi-kinetics` so the output carries per-strand fi/fp/ri/rp
# tags (the native PacBio HiFi layout). KinSim generate overwrites these
# kinetics with model predictions; ccs's own values are placeholders.
#
# Chain:
#   pbsim3_simulate.slurm  → reads.subreads.bam  (multi-pass, --pass-num 10)
#                          ↓
#   ccs_subreads.slurm     → reads.hifi.bam      (HiFi, fi/fp/ri/rp)
#                          ↓
#   prep/align_pbmm2.slurm → reads.aligned.bam   (ip/pw + alignment)
#                          ↓
#   kinsim generate        → reads_with_kinetics.bam
#
# Usage:
#   sbatch slurm_kinsim/ccs_subreads.slurm <subreads.bam> <hifi.bam>

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SUBREADS=${1:?subreads BAM required}
HIFI=${2:?output HiFi BAM path required}
SMRT_SIF=${SMRT_SIF:-/containers/apptainer/pacbio-smrt-tools-25.3.sif}

[ -f "$SUBREADS" ] || { echo "ERROR: subreads BAM not found: $SUBREADS" >&2; exit 1; }
[ -f "$SMRT_SIF" ] || { echo "ERROR: SMRT-tools SIF not found: $SMRT_SIF" >&2; exit 1; }

mkdir -p "$(dirname "$HIFI")"

if [ -f "$HIFI" ] && [ -s "$HIFI" ]; then
  echo "SKIP — HiFi BAM exists: $HIFI"
  echo "  set FORCE_CCS=1 to re-run (or rm the file)."
  [ "${FORCE_CCS:-0}" = "1" ] || exit 0
fi

echo "── ccs $SUBREADS → $HIFI ──"
START=$(date +%s)

apptainer exec --bind /data "$SMRT_SIF" \
    ccs "$SUBREADS" "$HIFI" \
        --min-rq 0.99 \
        --min-passes 3 \
        --hifi-kinetics \
        -j "$SLURM_CPUS_PER_TASK"

N=$(samtools view -c "$HIFI" 2>/dev/null || echo 0)
[ "$N" -gt 0 ] || { echo "ERROR: ccs produced 0 reads" >&2; exit 1; }

echo "HiFi reads : $N"
echo "Elapsed    : $(( $(date +%s) - START ))s"
echo
echo "  Next: sbatch slurm_kinsim/prep/align_pbmm2.slurm $HIFI <ref.fna> <aligned.bam>"
echo "        then: kinsim generate <aligned.bam> <ref.fna> <ckpt.pt> <motifs> <out.bam>"
