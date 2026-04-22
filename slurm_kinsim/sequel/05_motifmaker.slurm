#!/bin/bash
#SBATCH --job-name=sequel_05_mot
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_05_mot_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel step 05 — pbmotifmaker find (GFF → motifs.csv)
#   sbatch --array=1-48%8 slurm_kinsim/sequel/05_motifmaker.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
OUTBASE=${SEQUEL}/ipdsummary_ccs
ASM_DIR=${SEQUEL}/assemblies
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

mapfile -t GZ_BAMS < <(ls "${SEQUEL}"/lima.bc*--bc*.bam.gz 2>/dev/null | sort)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
[ $IDX -lt ${#GZ_BAMS[@]} ] || exit 0

BC=$(basename "${GZ_BAMS[$IDX]}" .bam.gz | grep -oP 'bc\d+' | head -1)
ASM_FASTA="${ASM_DIR}/${BC}_assembly.fsa"
GFF="${OUTBASE}/${BC}/${BC}_ipdSummary.gff"
MOTIFS="${OUTBASE}/${BC}/${BC}_motifs.csv"

echo "=== Sequel 05/motifmaker — $BC ==="

[ -f "$GFF" ] || { echo "ERROR: GFF missing"; exit 1; }
[ -f "$ASM_FASTA" ] || { echo "ERROR: assembly missing"; exit 1; }

if [ -s "$MOTIFS" ]; then
    echo "SKIP — motifs.csv exists ($(tail -n +2 "$MOTIFS" | wc -l) motifs)"; exit 0
fi

apptainer exec --bind /data "$SMRT_SIF" \
    pbmotifmaker find "$ASM_FASTA" "$GFF" "$MOTIFS"

[ -s "$MOTIFS" ] || { echo "WARNING: motifs.csv empty"; exit 1; }
echo "Motifs: $(tail -n +2 "$MOTIFS" | wc -l)"
cat "$MOTIFS"
