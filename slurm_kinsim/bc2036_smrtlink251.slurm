#!/bin/bash
#SBATCH --job-name=bc2036_smrt251
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --exclude=binfservas18
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/bc2036_smrt251_%J.log

# bc2036 methylation calling using M. Falquet's SMRT Link 25.1 installation
# Uses pbcromwell pb_microbial_analysis workflow with SP3-C3 chemistry override

set -uxo pipefail

export SMRT_ROOT="/data/users/lfalquet/SOFTS/SMRTlink/smrtlink251R"
export TMPDIR="$HOME/DATA/smrttempdir"
mkdir -p "$TMPDIR"

WORKDIR=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_smrtlink251
BAM=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/bc2036.bam

mkdir -p "$WORKDIR"
cd "$WORKDIR"

date && hostname
echo "========================================"
echo "=== bc2036 SMRT Link 25.1 pipeline ==="
echo "========================================"

# Create dataset XML from BAM
if [ ! -f bc2036.subreadset.xml ]; then
    echo "=== Creating ConsensusReadSet ==="
    ln -sf "$BAM" bc2036.bam
    ln -sf "${BAM}.pbi" bc2036.bam.pbi 2>/dev/null || true
    $SMRT_ROOT/smrtcmds/bin/dataset create --type ConsensusReadSet --name bc2036 bc2036.subreadset.xml bc2036.bam
fi

# Run full microbial analysis workflow
echo "=== Running pb_microbial_analysis ==="
$SMRT_ROOT/smrtcmds/bin/pbcromwell run pb_microbial_analysis \
    -e bc2036.subreadset.xml \
    --task-option kineticstools_use_chemistry="SP3-C3" \
    --task-option kineticstools_identify_mods="m4C,m6A" \
    --task-option tmp_dir="$TMPDIR" \
    --overwrite

echo "Exit: $?"

# Extract results
echo ""
echo "=== Searching for output files ==="
find . -name basemods.csv -exec echo "basemods.csv: {}" \;
find . -name basemods.gff -exec echo "basemods.gff: {}" \;
find . -name motifs.csv -exec echo "motifs.csv: {}" \;

# Copy motifs to workdir root
MOTIFS_FOUND=$(find . -name motifs.csv -not -path "./motifs.csv" | head -1)
if [ -n "$MOTIFS_FOUND" ]; then
    cp "$MOTIFS_FOUND" ./motifs.csv
    echo ""
    echo "=== Recovered motifs (SMRT Link 25.1 + SP3-C3) ==="
    cat ./motifs.csv
fi

echo ""
echo "=== Expected motifs ==="
cat /data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/motifs.csv 2>/dev/null \
    || echo "(not found)"

date
echo "=== Done ==="
