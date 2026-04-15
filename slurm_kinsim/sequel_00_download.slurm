#!/bin/bash
#SBATCH --job-name=sequel_dl
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_download_%J.log

# Download PacBio Sequel II Microbial Multiplexing 48-plex dataset
# Source: https://downloads.pacbcloud.com/public/dataset/MicrobialMultiplexing_48plex/

set -euo pipefail

DEST=/data/projects/p774_MARSD/NDutilleux/training/Sequel
BASE_URL=https://downloads.pacbcloud.com/public/dataset/MicrobialMultiplexing_48plex

echo "========================================================"
echo "  Download Sequel II 48-plex Microbial Multiplexing"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""

mkdir -p "$DEST"
cd "$DEST"

# Download metadata files
echo "=== Downloading metadata ==="
for f in sample_list.tsv readme.txt; do
    if [ ! -f "$f" ]; then
        wget -q --show-progress "${BASE_URL}/${f}" -O "$f" 2>&1 || echo "WARN: $f not found"
    else
        echo "  $f already exists, skipping"
    fi
done

# Download all subread BAMs + PBI indexes
echo ""
echo "=== Downloading subread BAMs ==="

# Get file listing from the BAM directory
INDEX_HTML=$(mktemp)
wget -q "${BASE_URL}/48-plex_bam/" -O "$INDEX_HTML" 2>&1 || true

# Extract .bam and .bam.pbi filenames from the directory listing
BAM_FILES=$(grep -oP 'href="\K[^"]+\.subreads\.bam(\.pbi)?"' "$INDEX_HTML" | sort -u)

if [ -z "$BAM_FILES" ]; then
    echo "WARN: Could not parse directory listing. Trying direct pattern..."
    # Fallback: try numbered pattern m54081_*
    # We know there are 48-plex BAMs, try listing
    wget -q "${BASE_URL}/48-plex_bam/" -O - 2>/dev/null | grep -oP 'href="\K[^"]+\.bam[^"]*' | sort -u > "$INDEX_HTML"
    BAM_FILES=$(cat "$INDEX_HTML")
fi

N_FILES=$(echo "$BAM_FILES" | wc -l)
echo "Found $N_FILES files to download"
echo ""

DOWNLOADED=0
SKIPPED=0
for fname in $BAM_FILES; do
    if [ -f "$fname" ]; then
        echo "  SKIP: $fname (exists)"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  GET:  $fname"
        wget -q --show-progress "${BASE_URL}/48-plex_bam/${fname}" -O "$fname" 2>&1
        DOWNLOADED=$((DOWNLOADED + 1))
    fi
done

rm -f "$INDEX_HTML"

echo ""
echo "=== Summary ==="
echo "Downloaded: $DOWNLOADED files"
echo "Skipped:    $SKIPPED files (already existed)"
echo "Location:   $DEST"
du -sh "$DEST"
echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
