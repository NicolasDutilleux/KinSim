#!/bin/bash
# =============================================================================
# trim_species.sh
#
# Create a trimmed copy of species_split_by_experience with only species
# that have >= MIN_READS reads (from species_counts.txt).
# Uses symlinks to avoid duplicating BAM data.
#
# Usage:
#   bash slurm_kinsim/trim_species.sh [MIN_READS]
#
# Default MIN_READS: 1000
# =============================================================================

set -euo pipefail

BASE=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK
SRC=$BASE/species_split_by_experience
DST=$BASE/trimmed_species_by_experience
MIN_READS=${1:-1000}

echo "========================================"
echo "  Trim species by read count"
echo "========================================"
echo "  Source:     $SRC"
echo "  Output:     $DST"
echo "  Min reads:  $MIN_READS"
echo ""

# Use the first experience to get species_counts.txt
# (same species across all experiences, counts may vary slightly)
FIRST_EXP=$(ls -d $SRC/MSA1003.* | head -1)
COUNTS_FILE="$FIRST_EXP/species_counts.txt"

if [ ! -f "$COUNTS_FILE" ]; then
    echo "ERROR: species_counts.txt not found at $COUNTS_FILE"
    exit 1
fi

# Build list of species that pass the threshold
KEEP=()
SKIP=()
while read -r ACC COUNT; do
    if [ "$COUNT" -ge "$MIN_READS" ]; then
        KEEP+=("$ACC")
        echo "  [KEEP]  $ACC  ($COUNT reads)"
    else
        SKIP+=("$ACC")
        echo "  [SKIP]  $ACC  ($COUNT reads < $MIN_READS)"
    fi
done < "$COUNTS_FILE"

echo ""
echo "Keeping ${#KEEP[@]} / $((${#KEEP[@]} + ${#SKIP[@]})) species"
echo ""

# Create trimmed directory structure with symlinks
mkdir -p "$DST"

for EXP_DIR in $SRC/MSA1003.*; do
    EXP=$(basename "$EXP_DIR")
    mkdir -p "$DST/$EXP"

    for ACC in "${KEEP[@]}"; do
        SPECIES_SRC="$EXP_DIR/$ACC"
        SPECIES_DST="$DST/$EXP/$ACC"

        if [ -d "$SPECIES_SRC" ] && [ ! -e "$SPECIES_DST" ]; then
            ln -s "$SPECIES_SRC" "$SPECIES_DST"
        fi
    done

    # Copy species_counts.txt (filtered)
    if [ -f "$EXP_DIR/species_counts.txt" ]; then
        grep -E "$(printf '%s|' "${KEEP[@]}" | sed 's/|$//')" \
            "$EXP_DIR/species_counts.txt" > "$DST/$EXP/species_counts.txt" || true
    fi

    N_LINKED=$(ls -d "$DST/$EXP"/*/ 2>/dev/null | wc -l)
    echo "  $EXP: $N_LINKED species linked"
done

echo ""
echo "Done. Trimmed structure at:"
echo "  $DST"
echo ""
echo "Next: update the SLURM scripts to use this directory, or re-run:"
echo "  SPLIT=$DST bash slurm_kinsim/prep_MSA1003_merge.sh"
