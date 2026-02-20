#!/bin/bash
#SBATCH --job-name=PBSIM3_Simulate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/pbsim3_simulate_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# PBSIM3 — Simulate PacBio HiFi reads from reference genomes (array job)
#
# Takes a directory of .fna reference genomes and creates one species
# subdirectory per genome, ready for kinsim dictionary metagenome.
#
# Usage:
#   sbatch --array=1-N pbsim3_simulate.slurm <genomes_dir> <output_root> <model>
#
# Arguments:
#   $1  Directory containing reference genomes (.fna files)
#   $2  Output root directory (one species subdir created per genome)
#   $3  PBSIM3 QSHMM model file path (e.g. /data/tools/pbsim3/data/QSHMM-PACBIO-CCS.model)
#
# Count N:
#   ls <genomes_dir>/*.fna | wc -l
#
# Example:
#   sbatch --array=1-57 pbsim3_simulate.slurm genomes/ pbsim3_output/ \
#       /data/tools/pbsim3/data/QSHMM-PACBIO-CCS.model
#
# Output directory structure created per genome:
#   pbsim3_output/
#     Ecoli/
#       genome.fna          <- copy of the reference
#       reads.fq.gz         <- simulated reads (compressed)
#       reads.maf.gz        <- alignment truth file (compressed)
#                              (place your motifs.csv here before running kinsim)
#     Salmonella/
#       ...
#
# After PBSIM3, place motifs.csv in each species subdirectory, then run:
#   kinsim dictionary metagenome pbsim3_output/ master_dict.pkl injected/
#
# Optional PBSIM3 parameters (edit below):
#   --depth        Read depth (default: 30)
#   --length-min   Minimum read length (default: 1000)
#   --length-max   Maximum read length (default: 25000)
#   --accuracy-min Minimum read accuracy (default: 0.9)
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: sbatch --array=1-N pbsim3_simulate.slurm <genomes_dir> <output_root> <model>"
    echo ""
    echo "Arguments:"
    echo "  genomes_dir   Directory with .fna reference genome files"
    echo "  output_root   Root output directory (one species subdir per genome)"
    echo "  model         PBSIM3 QSHMM model file (.model)"
    echo ""
    echo "Count N: ls <genomes_dir>/*.fna | wc -l"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env
# pbsim3 must be on PATH. Install with: conda install -c bioconda pbsim3

# ---- Config ----
GENOMES_DIR="$1"
OUTPUT_ROOT="$2"
MODEL_PATH="$3"

if [ ! -d "$GENOMES_DIR" ]; then
    echo "ERROR: Genomes directory not found: $GENOMES_DIR"
    exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: PBSIM3 model file not found: $MODEL_PATH"
    exit 1
fi

# ---- Pick genome for this task ----
FNA_FILE=$(ls "${GENOMES_DIR}"/*.fna | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$FNA_FILE" ]; then
    echo "ERROR: No .fna file at index $SLURM_ARRAY_TASK_ID in $GENOMES_DIR"
    exit 1
fi

SPECIES=$(basename "$FNA_FILE" .fna)
OUTDIR="${OUTPUT_ROOT}/${SPECIES}"

echo "=== Task $SLURM_ARRAY_TASK_ID: $SPECIES ==="
echo "Reference: $FNA_FILE"
echo "Output:    $OUTDIR"
echo ""

mkdir -p "$OUTDIR"

# Copy reference genome into species subdir
cp "$FNA_FILE" "${OUTDIR}/genome.fna"

# ---- Run PBSIM3 ----
TMPPREFIX="${OUTDIR}/reads"

pbsim \
    --strategy wgs \
    --method qshmm \
    --qshmm "$MODEL_PATH" \
    --genome "${OUTDIR}/genome.fna" \
    --prefix "$TMPPREFIX" \
    --depth 30 \
    --length-min 1000 \
    --length-max 25000 \
    --accuracy-min 0.9

# PBSIM3 outputs: reads_0001.fastq, reads_0001.maf (one set per chromosome)
# For multi-chromosome genomes, merge all fastq/maf files, then compress.

# ---- Collect and compress outputs ----
FASTQ_FILES=("${TMPPREFIX}"_*.fastq)
MAF_FILES=("${TMPPREFIX}"_*.maf)

if [ ${#FASTQ_FILES[@]} -eq 1 ]; then
    # Single chromosome: rename and compress
    gzip -c "${FASTQ_FILES[0]}" > "${OUTDIR}/reads.fq.gz"
    gzip -c "${MAF_FILES[0]}"   > "${OUTDIR}/reads.maf.gz"
    rm "${FASTQ_FILES[0]}" "${MAF_FILES[0]}"
else
    # Multiple chromosomes: concatenate, then compress
    cat "${TMPPREFIX}"_*.fastq | gzip > "${OUTDIR}/reads.fq.gz"
    cat "${TMPPREFIX}"_*.maf   | gzip > "${OUTDIR}/reads.maf.gz"
    rm "${TMPPREFIX}"_*.fastq "${TMPPREFIX}"_*.maf
fi

# Remove PBSIM3 ref files (not needed)
rm -f "${TMPPREFIX}"_*.ref

echo ""
echo "=== Done: $SPECIES ==="
echo "  reads.fq.gz:   $(du -sh "${OUTDIR}/reads.fq.gz" | cut -f1)"
echo "  reads.maf.gz:  $(du -sh "${OUTDIR}/reads.maf.gz" | cut -f1)"
echo ""
echo "  Next: place motifs.csv in ${OUTDIR}/"
echo "  Then: kinsim dictionary metagenome ${OUTPUT_ROOT}/ master_dict.pkl injected/"
