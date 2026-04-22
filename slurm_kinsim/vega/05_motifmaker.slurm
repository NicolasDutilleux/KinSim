#!/bin/bash
#SBATCH --job-name=vega_05_mot
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_05_mot_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega step 05 — pbmotifmaker find (GFF → motifs.csv)
#
# Usage:
#   sbatch --array=1-16%8 slurm_kinsim/vega/05_motifmaker.slurm
#
# Output:  ${VEGA}/assembly/bcXXXX/bcXXXX_motifs.csv
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
OUTBASE=${VEGA}/assembly
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

BARCODES=(2033 2034 2035 2036 2037 2038 2039 2040
          2041 2042 2043 2044 2045 2046 2047 2048)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
BC=${BARCODES[$IDX]}
SAMPLE="bc${BC}"
SPECIES_DIR="${OUTBASE}/${SAMPLE}"
ASM_FASTA="${SPECIES_DIR}/${SAMPLE}_assembly.fasta"
GFF="${SPECIES_DIR}/${SAMPLE}_ipdSummary.gff"
MOTIFS="${SPECIES_DIR}/${SAMPLE}_motifs.csv"

echo "=== Vega 05/motifmaker — $SAMPLE ==="

[ -f "$GFF" ] || { echo "ERROR: GFF missing: $GFF"; exit 1; }
[ -f "$ASM_FASTA" ] || { echo "ERROR: assembly missing: $ASM_FASTA"; exit 1; }

if [ -f "$MOTIFS" ] && [ -s "$MOTIFS" ]; then
    echo "SKIP — motifs.csv exists ($(tail -n +2 "$MOTIFS" | wc -l) motifs)"
    exit 0
fi

apptainer exec --bind /data "$SMRT_SIF" \
    pbmotifmaker find "$ASM_FASTA" "$GFF" "$MOTIFS"

if [ ! -s "$MOTIFS" ]; then
    echo "WARNING: motifs.csv empty or missing"; exit 1
fi

echo "Motifs discovered: $(tail -n +2 "$MOTIFS" | wc -l)"
echo "---"
cat "$MOTIFS"
