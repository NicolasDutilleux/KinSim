#!/bin/bash
#SBATCH --job-name=sequel_meta
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_meta_%J.log

# Download Sequel II 48-plex metadata + assemblies
# Assemblies already made by PacBio — no need to reassemble

set -euo pipefail

DEST=/data/projects/p774_MARSD/NDutilleux/training/Sequel
BASE_URL=https://downloads.pacbcloud.com/public/dataset/MicrobialMultiplexing_48plex

echo "========================================================"
echo "  Download Sequel II metadata + assemblies"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""

mkdir -p "$DEST/assemblies"
cd "$DEST"

# Metadata files
echo "=== Metadata ==="
for f in 48-plex_sample_list.tsv 48-plex_demux_and_assembly_stats.tsv readme.txt; do
    if [ -f "$f" ]; then
        echo "  SKIP: $f"
    else
        echo "  GET:  $f"
        wget -q "${BASE_URL}/${f}" -O "$f" || echo "  WARN: $f not found"
    fi
done
echo ""

# Assemblies (48 x .fsa.gz)
echo "=== Assemblies ==="
BARCODES=(
    1018 1019 1031 1034 1037 1040 1043 1044
    1046 1047 1050 1052 1053 1054 1055 1058
    1059 1060 1061 1062 1065 1066 1068 1070
    1074 1076 1077 1078 1079 1081 1082 1083
    1084 1087 1091 1092 1094 1095 1099 1101
    1102 1106 1109 1111 1117 1118 1119 1126
)

DOWNLOADED=0
SKIPPED=0
for bc in "${BARCODES[@]}"; do
    fname="bc${bc}_assembly.fsa.gz"
    dest_file="assemblies/${fname}"
    if [ -f "$dest_file" ]; then
        echo "  SKIP: $fname"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  GET:  $fname"
        wget -q "${BASE_URL}/48-plex_assemblies/${fname}" -O "$dest_file" || echo "  FAIL: $fname"
        DOWNLOADED=$((DOWNLOADED + 1))
    fi
done

# Decompress assemblies
echo ""
echo "=== Decompressing assemblies ==="
for gz in assemblies/*.fsa.gz; do
    fsa="${gz%.gz}"
    if [ -f "$fsa" ]; then
        echo "  SKIP: $(basename "$fsa")"
    else
        gunzip -k "$gz"
        echo "  OK:   $(basename "$fsa")"
    fi
done

echo ""
echo "=== Summary ==="
echo "Downloaded: $DOWNLOADED assemblies"
echo "Skipped:    $SKIPPED assemblies"
echo "Sample list:"
cat "$DEST/48-plex_sample_list.tsv" 2>/dev/null | head -5
echo "..."
echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
