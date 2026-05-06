#!/bin/bash
# =============================================================================
# prep_rebase.sh
#
# Fetch REBASE motifs for species that have an org number assigned.
# Copies rebase_motifs.csv into every experience folder.
#
# Usage:
#   bash slurm_kinsim/msa1003/prep_rebase.sh [SPLIT_DIR]
#
# Default: $BASE/trimmed_species_by_experience
# =============================================================================

set -euo pipefail

BASE=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK
SPLIT="${1:-$BASE/trimmed_species_by_experience}"

echo "========================================"
echo "  REBASE fetch"
echo "========================================"
echo "  Directory: $SPLIT"
echo ""

# Auto-discover experiences
EXPERIENCES=($(ls -d "$SPLIT"/MSA1003.* 2>/dev/null | xargs -n1 basename | sort))
if [ ${#EXPERIENCES[@]} -eq 0 ]; then
    echo "ERROR: No MSA1003.* experience directories found in $SPLIT"
    exit 1
fi
echo "  Experiences: ${#EXPERIENCES[@]}"

# ---------------------------------------------------------------------------
# REBASE organism numbers
# Set to 0 for species with no REBASE match.
# ---------------------------------------------------------------------------
declare -A REBASE_ORG=(
    [AE000511.1]=0         # Helicobacter pylori 26695
    [AE000513.1]=0         # Deinococcus radiodurans R1
    [AE002098.2]=0         # Neisseria meningitidis MC58
    [AE009948.1]=0         # Streptococcus agalactiae 2603V/R
    [AE014133.2]=0         # Streptococcus mutans UA159
    [AE015929.1]=0         # Staphylococcus epidermidis ATCC 12228
    [AE017194.1]=244       # Bacillus cereus ATCC 10987
    [AP009256.1]=0         # Bifidobacterium adolescentis ATCC 15703
    [AP009380.1]=0         # Porphyromonas gingivalis ATCC 33277
    [CP000139.1]=0         # Bacteroides vulgatus ATCC 8482
    [CP000255.1]=0         # Staphylococcus aureus USA300_FPR3757
    [CP000413.1]=0         # Lactobacillus gasseri ATCC 33323
    [CP000521.1]=0         # Acinetobacter baumannii ATCC 17978
    [CP000577.1]=0         # Cereibacter sphaeroides ATCC 17029 (chr 1)
    [CP000744.1]=0         # Pseudomonas paraeruginosa PA7
    [CP003084.1]=0         # Propionibacterium acnes ATCC 11828
    [CP046315.1]=0         # Schaalia odontolytica FDAARGOS_732
    [NC_009050.1]=0        # Cereibacter sphaeroides ATCC 17029 (chr 2)
    [NC_017316.1]=0        # Enterococcus faecalis OG1RF
    [NZ_CP006777.1]=0      # Clostridium beijerinckii ATCC 35702 SA-1
    [U00096.3]=1260        # Escherichia coli K-12 MG1655  (verified)
)

# ---------------------------------------------------------------------------
# Fetch loop: fetch once, copy to all experiences
# ---------------------------------------------------------------------------
for SPECIES_DIR in "$SPLIT/${EXPERIENCES[0]}"/*/; do
    [ -d "$SPECIES_DIR" ] || continue
    ACC=$(basename "$SPECIES_DIR")

    ORG_NUM="${REBASE_ORG[$ACC]:-0}"
    if [[ "$ORG_NUM" == "TODO" || "$ORG_NUM" == "0" ]]; then
        echo "[SKIP] $ACC -- no REBASE org number"
        continue
    fi

    echo ""
    echo "=== $ACC (org $ORG_NUM) ==="

    # Fetch into the first experience
    FIRST_CSV="$SPLIT/${EXPERIENCES[0]}/$ACC/rebase_motifs.csv"
    python -m kinsim.utils.parsers.rebase fetch "$ORG_NUM" --output "$FIRST_CSV"

    # Copy to the other experiences
    for EXP in "${EXPERIENCES[@]:1}"; do
        DEST="$SPLIT/$EXP/$ACC/rebase_motifs.csv"
        if [[ -d "$SPLIT/$EXP/$ACC" ]]; then
            cp "$FIRST_CSV" "$DEST"
            echo "  Copied to $EXP/$ACC/"
        fi
    done
done

echo ""
echo "Done. rebase_motifs.csv written into all experience folders."
echo "Next step: bash slurm_kinsim/msa1003/prep_merge.sh $SPLIT"
