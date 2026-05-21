#!/bin/bash
# From-scratch generation chain per genome: PBSIM3 → ccs → strip → align
# → kinsim generate → merge → jasmine ‖ bystrandify+align+ipdSummary+motifmaker
# → merge_motifs. Compare output SIM_motifs_merged.csv vs input motifs.csv.
#
# Usage: bash slurm_kinsim/generate.sh <genomes_dir> <output_root> <ckpt_dir> <motifs.csv>
# Env: N_SHARDS (default 10), PBSIM_MODEL (default ERRHMM-SEQUEL)
set -euo pipefail

GENOMES_DIR=${1:?"usage: bash $0 <genomes_dir> <output_root> <ckpt_dir> <motifs.csv>"}
OUTPUT_ROOT=${2:?"output_root required"}
CKPT_DIR=${3:?"ckpt_dir required"}
MOTIFS=${4:?"motifs.csv required"}

[ -d "$GENOMES_DIR" ] || { echo "ERROR: genomes_dir missing: $GENOMES_DIR" >&2; exit 1; }
[ -f "$MOTIFS" ]      || { echo "ERROR: motifs CSV missing: $MOTIFS" >&2; exit 1; }
[ -d "$CKPT_DIR" ]    || { echo "ERROR: CKPT_DIR missing: $CKPT_DIR" >&2; exit 1; }
ls "$CKPT_DIR"/checkpoint_epoch*.pt >/dev/null 2>&1 \
  || { echo "ERROR: no checkpoint_epoch*.pt in $CKPT_DIR" >&2; exit 1; }

N_SHARDS="${N_SHARDS:-10}"
REPO=/data/users/ndutilleux/KinSim
PREP_SLURM="$REPO/slurm_kinsim/prep"
CALLERS="$REPO/slurm_kinsim/callers"
VAL_SLURM="$REPO/slurm_kinsim/validate"
LOG=/data/projects/p774_MARSD/NDutilleux/logs

mkdir -p "$OUTPUT_ROOT"

FNAS=()
while IFS= read -r f; do FNAS+=("$f"); done < <(ls "$GENOMES_DIR"/*.fna 2>/dev/null | sort)
N_GENOMES=${#FNAS[@]}
[ "$N_GENOMES" -gt 0 ] || { echo "ERROR: no .fna in $GENOMES_DIR" >&2; exit 1; }

if [ -n "${PBSIM_MODEL:-}" ]; then
  J_PBSIM=$(sbatch --parsable --array=1-$N_GENOMES \
      "$REPO/slurm_kinsim/pbsim3_simulate.slurm" "$GENOMES_DIR" "$OUTPUT_ROOT" "$PBSIM_MODEL")
else
  J_PBSIM=$(sbatch --parsable --array=1-$N_GENOMES \
      "$REPO/slurm_kinsim/pbsim3_simulate.slurm" "$GENOMES_DIR" "$OUTPUT_ROOT")
fi
echo "pbsim3      $J_PBSIM (array 1-$N_GENOMES)"

declare -a FINAL_JOBS=()
i=0
for FNA in "${FNAS[@]}"; do
  i=$((i + 1))
  SPECIES=$(basename "$FNA" .fna)
  GENOME_DIR="$OUTPUT_ROOT/$SPECIES"
  mkdir -p "$GENOME_DIR/shards"

  SUBREADS="$GENOME_DIR/reads.subreads.bam"
  HIFI_BAM="$GENOME_DIR/reads.hifi.bam"
  STRIPPED_BAM="$GENOME_DIR/reads.stripped.bam"
  STRIPPED_ALIGNED_BAM="$GENOME_DIR/reads.aligned.bam"
  REGIONS_FILE="$GENOME_DIR/shards/regions.txt"
  SHARD_DIR="$GENOME_DIR/shards"
  SIM_BAM="$GENOME_DIR/${SPECIES}_simulated.bam"
  SIM_BYS_BAM="$GENOME_DIR/${SPECIES}_simulated_bystrandified.bam"
  SIM_ALIGNED_BAM="$GENOME_DIR/${SPECIES}_simulated_aligned.bam"
  SIM_GFF="$GENOME_DIR/${SPECIES}_simulated.gff"
  SIM_IPD_CSV="$GENOME_DIR/${SPECIES}_simulated_ipdSummary.csv"
  SIM_MM_CSV="$GENOME_DIR/${SPECIES}_simulated_motifs_ipdsummary.csv"
  SIM_JM_CSV="$GENOME_DIR/${SPECIES}_simulated_motifs_jasmine.csv"
  SIM_MERGED_CSV="$GENOME_DIR/${SPECIES}_simulated_motifs_merged.csv"

  echo "── $SPECIES (task $i)"

  J_CCS=$(sbatch --parsable --dependency=afterok:${J_PBSIM}_${i} \
    --job-name="gen_ccs_$SPECIES" \
    "$REPO/slurm_kinsim/ccs_subreads.slurm" "$SUBREADS" "$HIFI_BAM")

  J_STRIP=$(sbatch --parsable --dependency=afterok:$J_CCS \
    --job-name="gen_strip_$SPECIES" \
    "$VAL_SLURM/prep.slurm" "$HIFI_BAM" "$STRIPPED_BAM")

  J_ALIGN_RAW=$(sbatch --parsable --dependency=afterok:$J_STRIP \
    --job-name="gen_align_raw_$SPECIES" \
    "$PREP_SLURM/align_pbmm2.slurm" "$STRIPPED_BAM" "$FNA" "$STRIPPED_ALIGNED_BAM")

  J_REGIONS=$(sbatch --parsable --dependency=afterok:$J_ALIGN_RAW \
    --job-name="gen_regions_$SPECIES" \
    --partition=pshort_el8 --account=p774 \
    --cpus-per-task=2 --mem=4G --time=00:15:00 \
    --output=$LOG/%x_%J.log \
    --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
            [ -f ${STRIPPED_ALIGNED_BAM}.bai ] || samtools index -@ 2 ${STRIPPED_ALIGNED_BAM}; \
            python3 $VAL_SLURM/write_regions.py ${STRIPPED_ALIGNED_BAM} ${REGIONS_FILE} ${N_SHARDS}")

  J_GEN=$(sbatch --parsable --dependency=afterok:$J_REGIONS \
    --array=0-$((N_SHARDS - 1)) \
    --job-name="gen_kinsim_$SPECIES" \
    "$VAL_SLURM/generate.slurm" \
    "$STRIPPED_ALIGNED_BAM" "$FNA" "$CKPT_DIR" "$MOTIFS" "$REGIONS_FILE" "$SHARD_DIR")

  J_MERGE=$(sbatch --parsable --dependency=afterok:$J_GEN \
    --job-name="gen_merge_$SPECIES" \
    "$VAL_SLURM/merge.slurm" "$SHARD_DIR" "$SIM_BAM")

  J_JM=$(sbatch --parsable --dependency=afterok:$J_MERGE \
    --job-name="gen_jm_$SPECIES" \
    "$CALLERS/jasmine_modkit.slurm" "$SIM_BAM" "$FNA" "$SIM_JM_CSV")

  J_BYS=$(sbatch --parsable --dependency=afterok:$J_MERGE \
    --job-name="gen_bys_$SPECIES" \
    "$PREP_SLURM/bystrandify.slurm" "$SIM_BAM" "$SIM_BYS_BAM")
  J_ALIGN_BYS=$(sbatch --parsable --dependency=afterok:$J_BYS \
    --job-name="gen_align_bys_$SPECIES" \
    "$PREP_SLURM/align_pbmm2.slurm" "$SIM_BYS_BAM" "$FNA" "$SIM_ALIGNED_BAM")
  J_IPD=$(sbatch --parsable --dependency=afterok:$J_ALIGN_BYS \
    --job-name="gen_ipd_$SPECIES" \
    "$CALLERS/ipdsummary.slurm" "$SIM_ALIGNED_BAM" "$FNA" "$SIM_GFF" "$SIM_IPD_CSV")
  J_MM=$(sbatch --parsable --dependency=afterok:$J_IPD \
    --job-name="gen_mm_$SPECIES" \
    "$CALLERS/pbmotifmaker.slurm" "$FNA" "$SIM_GFF" "$SIM_MM_CSV")

  J_FINAL=$(sbatch --parsable --dependency=afterok:${J_MM}:${J_JM} \
    --job-name="gen_final_$SPECIES" \
    "$CALLERS/merge_motifs.slurm" "$SIM_MERGED_CSV" 0.7 "$SIM_MM_CSV" "$SIM_JM_CSV")

  echo "  ccs $J_CCS  strip $J_STRIP  align $J_ALIGN_RAW  regions $J_REGIONS"
  echo "  gen $J_GEN (array)  merge $J_MERGE  jasmine $J_JM"
  echo "  bys $J_BYS  align_bys $J_ALIGN_BYS  ipd $J_IPD  mm $J_MM"
  echo "  final $J_FINAL → $SIM_MERGED_CSV"
  FINAL_JOBS+=("$J_FINAL")
done

echo
echo "Submitted ${#FINAL_JOBS[@]} generation chains across $N_GENOMES genomes."
echo "Final job IDs: ${FINAL_JOBS[*]}"
