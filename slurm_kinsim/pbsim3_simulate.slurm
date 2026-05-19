#!/bin/bash
#SBATCH --job-name=pbsim3_simulate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/pbsim3_simulate_%A_%a.log
#
# PBSIM3 — simulate PacBio multi-pass subreads from a reference genome.
#
# Default `--pass-num 1` would give single-pass reads (~85% accuracy).
# We use `--pass-num 10` so the output can be fed to `ccs` to produce
# HiFi consensus reads (~99.9% accuracy) with native fi/fp/ri/rp kinetic
# tags. Chain:
#
#   pbsim3_simulate.slurm  → reads.subreads.bam   (multi-pass subreads)
#                         ↓
#   ccs_subreads.slurm     → reads.hifi.bam       (HiFi, fi/fp/ri/rp)
#                         ↓
#   prep/align_pbmm2.slurm → reads.aligned.bam    (ip/pw + alignment)
#                         ↓
#   kinsim generate        → reads_with_kinetics.bam
#                                                 (fi/fp/ri/rp, flag=4)
#
# pbmm2 align is REQUIRED before generate so the mapped path (strand-aware
# mc/rev_meth from the reference) is taken — see kinsim/generate.py.
#
# Usage:
#   sbatch --array=1-N pbsim3_simulate.slurm <genomes_dir> <output_root> [<model>]
#
# Args:
#   $1  Directory with .fna reference genomes (one .fna per species).
#   $2  Output root — one species subdir per genome is created.
#   $3  PBSIM3 quality/error model file.
#       Default: ERRHMM-SEQUEL.model (closest to PacBio Sequel/Revio chemistry).
#       Other options shipped with pbsim3 conda package:
#         - QSHMM-RSII.model       (PacBio RSII)
#         - QSHMM-ONT.model        (ONT)
#         - QSHMM-ONT-HQ.model     (ONT high-quality)
#         - ERRHMM-RSII.model      (PacBio RSII, error-HMM)
#         - ERRHMM-ONT.model       (ONT)
#         - ERRHMM-ONT-HQ.model    (ONT high-quality)
#
# Count N:  ls <genomes_dir>/*.fna | wc -l

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

GENOMES_DIR=${1:?genomes_dir required (directory with .fna files)}
OUTPUT_ROOT=${2:?output_root required}
MODEL_PATH=${3:-$CONDA_PREFIX/data/ERRHMM-SEQUEL.model}

[ -d "$GENOMES_DIR" ] || { echo "ERROR: genomes_dir not found: $GENOMES_DIR" >&2; exit 1; }
[ -f "$MODEL_PATH" ]  || { echo "ERROR: PBSIM3 model not found: $MODEL_PATH" >&2; exit 1; }

FNA_FILE=$(ls "${GENOMES_DIR}"/*.fna | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$FNA_FILE" ] || { echo "ERROR: no .fna at index $SLURM_ARRAY_TASK_ID in $GENOMES_DIR" >&2; exit 1; }

SPECIES=$(basename "$FNA_FILE" .fna)
OUTDIR="${OUTPUT_ROOT}/${SPECIES}"
mkdir -p "$OUTDIR"
cp "$FNA_FILE" "${OUTDIR}/genome.fna"

echo "── PBSIM3 task $SLURM_ARRAY_TASK_ID: $SPECIES ──"
echo "  reference : $FNA_FILE"
echo "  output    : $OUTDIR"
echo "  model     : $MODEL_PATH"

# Pick the right model flag based on filename — errhmm vs qshmm.
case "$(basename "$MODEL_PATH")" in
  ERRHMM-*) METHOD=errhmm; MODEL_FLAG="--errhmm $MODEL_PATH" ;;
  QSHMM-*)  METHOD=qshmm;  MODEL_FLAG="--qshmm  $MODEL_PATH" ;;
  *) echo "ERROR: cannot infer --method from model filename: $MODEL_PATH" >&2; exit 1 ;;
esac

TMPPREFIX="${OUTDIR}/reads"

# --pass-num 10 gives ten subread passes per ZMW. ccs (next step) needs
# >= 3 passes to compute a HiFi consensus; 10 is the sweet spot for
# Sequel/Revio chemistry.
pbsim \
    --strategy wgs \
    --method "$METHOD" \
    $MODEL_FLAG \
    --genome "${OUTDIR}/genome.fna" \
    --prefix "$TMPPREFIX" \
    --depth 30 \
    --length-min 1000 \
    --length-max 25000 \
    --accuracy-mean 0.95 \
    --pass-num 10

# With --pass-num > 1, PBSIM3 emits reads_NNNN.sam (multi-pass subreads),
# reads_NNNN.maf (alignment truth) and reads_NNNN.ref per chromosome.
# Convert all SAMs to a single subread BAM that `ccs` can consume.
SAM_FILES=("${TMPPREFIX}"_*.sam)
[ -f "${SAM_FILES[0]}" ] || { echo "ERROR: PBSIM3 produced no .sam output (check --pass-num >= 2)" >&2; exit 1; }

if [ "${#SAM_FILES[@]}" -eq 1 ]; then
  samtools view -bS "${SAM_FILES[0]}" -o "${OUTDIR}/reads.subreads.bam"
else
  # Concat headers + bodies; samtools merge handles dedup.
  samtools merge -f "${OUTDIR}/reads.subreads.bam" "${SAM_FILES[@]}"
fi
samtools index "${OUTDIR}/reads.subreads.bam"
rm "${TMPPREFIX}"_*.sam "${TMPPREFIX}"_*.ref 2>/dev/null

# Keep the MAF (alignment truth) — useful for downstream validation.
cat "${TMPPREFIX}"_*.maf | gzip > "${OUTDIR}/reads.maf.gz"
rm "${TMPPREFIX}"_*.maf

echo "── PBSIM3 done ──"
echo "  subreads BAM : ${OUTDIR}/reads.subreads.bam ($(du -sh "${OUTDIR}/reads.subreads.bam" | cut -f1))"
echo "  truth MAF    : ${OUTDIR}/reads.maf.gz       ($(du -sh "${OUTDIR}/reads.maf.gz" | cut -f1))"
echo
echo "  Next: sbatch slurm_kinsim/ccs_subreads.slurm ${OUTDIR}/reads.subreads.bam ${OUTDIR}/reads.hifi.bam"
