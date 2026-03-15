#!/bin/bash
# =============================================================================
# fix_offsets_0to1.sh
#
# Convert all calling motif CSVs from 0-based to 1-based offsets.
# Handles both formats:
#   Combined CSV: mod_type,motif,offset,...   (offset column = 3rd)
#   PacBio CSV:   motifString,centerPos,...   (centerPos column = 2nd)
#
# Usage:
#   bash slurm_kinsim/fix_offsets_0to1.sh <directory>
#
# Finds all *_motifs.csv files recursively and converts in-place.
# Creates a .bak backup of each file before modifying.
# =============================================================================

set -euo pipefail

DIR="${1:-.}"

echo "=== Converting 0-based offsets to 1-based ==="
echo "Directory: $DIR"
echo ""

FIXED=0
SKIPPED=0

find "$DIR" -name "*_motifs.csv" -type f | sort | while read -r CSV; do
    # Detect format from header
    HEADER=$(head -1 "$CSV")

    if echo "$HEADER" | grep -q "^mod_type,motif,offset"; then
        # Combined CSV format: offset is column 3 (1-indexed)
        COL=3
        FMT="combined"
    elif echo "$HEADER" | grep -q "motifString.*centerPos"; then
        # PacBio CSV format: centerPos is column 2 (1-indexed)
        COL=2
        FMT="pacbio"
    else
        echo "  [SKIP] $CSV (unknown format)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Backup
    cp "$CSV" "${CSV}.bak"

    # Add 1 to the offset column (skip header line)
    python3 -c "
import csv, sys

with open('$CSV') as f:
    rows = list(csv.reader(f))

header = rows[0]
col = $COL - 1  # 0-indexed for Python

changed = 0
for row in rows[1:]:
    if col < len(row) and row[col].strip():
        try:
            old = int(row[col])
            row[col] = str(old + 1)
            changed += 1
        except ValueError:
            pass

with open('$CSV', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f'  [FIXED] {\"$CSV\"} ({changed} rows, format={\"$FMT\"})')
"
    FIXED=$((FIXED + 1))
done

echo ""
echo "Done. $FIXED files converted, $SKIPPED skipped."
echo "Backups saved as *.bak"
