#!/bin/bash
#SBATCH --job-name=KinSim_Prepare
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=p774
#SBATCH --partition=pshort_el8
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_prepare_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim — Prepare pipeline config (single job)
#
# Reads a text file with alternating BAM / motif-source lines,
# resolves all motif sources (PacBio CSV, REBASE, or inline strings),
# and writes a compact config file ready for the train/extract steps.
#
# Usage:
#   sbatch kinsim_prepare.slurm <pairs_file> <config_out>
#
# Arguments:
#   $1  Pairs file (alternating lines: BAM path / motif source path or string)
#   $2  Output config file (alternating lines: BAM path / motif string)
#
# Example:
#   sbatch kinsim_prepare.slurm pairs.txt config_strains.txt
#
# Expected pairs file format:
#   /data/bams/strain1.bam
#   /data/motifs/strain1/motifs.csv
#   /data/bams/strain2.bam
#   /data/motifs/strain2/rebase_motifs.txt
#   /data/bams/strain3.bam
#   m6A,GATC,1;m4C,CCWGG,1
#
# Output config format (one pair per two lines):
#   /data/bams/strain1.bam
#   m6A,GCCGATC,5,3551;m6A,CTGAAG,5,2891
#   /data/bams/strain2.bam
#   m6A,GATC,1
#
# The output config is consumed by:
#   kinsim_train.slurm    (dictionary mode)
#   kinsim_cgan_extract.slurm  (cGAN mode)
#
# Note: Pass --array=1-N to these downstream scripts where N equals
#       the number of BAM/motif pairs in the output config.
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch kinsim_prepare.slurm <pairs_file> <config_out>"
    echo ""
    echo "Arguments:"
    echo "  pairs_file    Text file with alternating BAM path / motif source lines"
    echo "  config_out    Output config file (BAM path / motif string lines)"
    echo ""
    echo "Accepted motif sources (auto-detected per pair):"
    echo "  PacBio motifs.csv      — filtered by --min-fraction / --min-detected"
    echo "  REBASE file            — simplified two-column or Format #19 (withrefm)"
    echo "  KinSim motif string    — used as-is: 'm6A,GATC,1;m4C,CCWGG,1'"
    echo ""
    echo "Example pairs file:"
    echo "  /data/bams/strain1.bam"
    echo "  /data/motifs/strain1/motifs.csv"
    echo "  /data/bams/strain2.bam"
    echo "  m6A,GATC,1"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- Config ----
PAIRS_FILE="$1"
CONFIG_OUT="$2"

if [ ! -f "$PAIRS_FILE" ]; then
    echo "ERROR: Pairs file '$PAIRS_FILE' not found."
    exit 1
fi

mkdir -p "$(dirname "$CONFIG_OUT")"

echo "=== KinSim Prepare ==="
echo "Pairs file: $PAIRS_FILE"
echo "Output config: $CONFIG_OUT"
echo ""

# ---- Run ----
kinsim prepare "$PAIRS_FILE" "$CONFIG_OUT"

echo ""
echo "=== Prepare complete ==="
echo "Config written to: $CONFIG_OUT"
N=$(grep -c '' "$CONFIG_OUT" 2>/dev/null || echo 0)
PAIRS=$(( N / 2 ))
echo "Strain pairs: $PAIRS"
echo "Use --array=1-$PAIRS in downstream SLURM jobs"
