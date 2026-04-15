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
# Files: 48 demultiplexed BAMs (lima.bcXXXX--bcXXXX.bam.gz) + PBI indexes

set -euo pipefail

DEST=/data/projects/p774_MARSD/NDutilleux/training/Sequel
BASE_URL=https://downloads.pacbcloud.com/public/dataset/MicrobialMultiplexing_48plex/48-plex_bam

echo "========================================================"
echo "  Download Sequel II 48-plex Microbial Multiplexing"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""

mkdir -p "$DEST"
cd "$DEST"

# All 48 barcodes from the dataset
BARCODES=(
    1018 1019 1031 1034 1037 1040 1043 1044
    1046 1047 1050 1052 1053 1054 1055 1058
    1059 1060 1061 1062 1065 1066 1068 1070
    1074 1076 1077 1078 1079 1081 1082 1083
    1084 1087 1091 1092 1094 1095 1099 1101
    1102 1106 1109 1111 1117 1118 1119 1126
)

echo "Downloading ${#BARCODES[@]} demultiplexed BAMs + PBI indexes"
echo ""

DOWNLOADED=0
SKIPPED=0
FAILED=0

for bc in "${BARCODES[@]}"; do
    for ext in bam.gz bam.pbi.gz; do
        fname="lima.bc${bc}--bc${bc}.${ext}"
        if [ -f "$fname" ]; then
            echo "  SKIP: $fname"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "  GET:  $fname"
            if wget -q "${BASE_URL}/${fname}" -O "$fname" 2>&1; then
                DOWNLOADED=$((DOWNLOADED + 1))
            else
                echo "  FAIL: $fname"
                rm -f "$fname"
                FAILED=$((FAILED + 1))
            fi
        fi
    done
done

echo ""
echo "=== Summary ==="
echo "Downloaded: $DOWNLOADED files"
echo "Skipped:    $SKIPPED files (already existed)"
echo "Failed:     $FAILED files"
echo "Location:   $DEST"
du -sh "$DEST"
echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
