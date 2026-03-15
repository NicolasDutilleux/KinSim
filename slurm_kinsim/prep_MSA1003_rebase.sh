#!/bin/bash
# =============================================================================
# prep_MSA1003_rebase.sh
#
# For each species in the MSA1003 mock community that has a REBASE organism
# number assigned, fetch genuine motifs from REBASE and write rebase_motifs.csv
# into the canonical experience folder.
#
# Run ONCE per species, before prep_MSA1003_merge.sh.
#
# Output per species:
#   $SPLIT/$CANONICAL/$ACC/rebase_motifs.csv
#
# Usage:
#   conda activate kinsim_env
#   bash slurm_kinsim/prep_MSA1003_rebase.sh
#
# Prerequisites:
#   conda activate kinsim_env
#   kinsim-prep --version    # should print kinsim-prep 0.3.0
# =============================================================================

set -euo pipefail

BASE=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK
SPLIT=$BASE/species_split_by_experience

# Motifs prep is done from one canonical experience (same genome => same motifs)
CANONICAL=MSA1003.490fb6ec_6--6

# ---------------------------------------------------------------------------
# REBASE organism numbers
# Find each at: https://rebase.neb.com/cgi-bin/pacbioget?<ORG_NUM>
# Leave as TODO for any species you haven't looked up yet.
# NOTE: CP000577.1 and NC_009050.1 are both chromosomes of the same organism
#       (Cereibacter sphaeroides ATCC 17029) -- give them the same ORG_NUM.
# ---------------------------------------------------------------------------
declare -A REBASE_ORG=(
    [AE000511.1]=0      # Helicobacter pylori 26695
    [AE000513.1]=0    # Deinococcus radiodurans R1
    [AE002098.2]=0     # Neisseria meningitidis MC58
    [AE009948.1]=0     # Streptococcus agalactiae 2603V/R
    [AE014133.2]=0      # Streptococcus mutans UA159
    [AE015929.1]=0      # Staphylococcus epidermidis ATCC 12228
    [AE017194.1]=244     # Bacillus cereus ATCC 10987
    [AP009256.1]=0      # Bifidobacterium adolescentis ATCC 15703
    [AP009380.1]=0     # Porphyromonas gingivalis ATCC 33277
    [CP000139.1]=0      # Bacteroides vulgatus ATCC 8482
    [CP000255.1]=0      # Staphylococcus aureus USA300_FPR3757
    [CP000413.1]=0      # Lactobacillus gasseri ATCC 33323
    [CP000521.1]=0      # Acinetobacter baumannii ATCC 17978
    [CP000577.1]=0      # Cereibacter sphaeroides ATCC 17029 (chr 1)
    [CP000744.1]=0      # Pseudomonas paraeruginosa PA7
    [CP003084.1]=0      # Propionibacterium acnes ATCC 11828
    [CP046315.1]=0     # Schaalia odontolytica FDAARGOS_732
    [NC_009050.1]=0    # Cereibacter sphaeroides ATCC 17029 (chr 2) -- same as CP000577.1
    [NC_017316.1]=0    # Enterococcus faecalis OG1RF
    [NZ_CP006777.1]=0   # Clostridium beijerinckii ATCC 35702 SA-1
    [U00096.3]=1260        # Escherichia coli K-12 MG1655  (verified)
)

# ---------------------------------------------------------------------------
# Fetch loop
# ---------------------------------------------------------------------------
for ACC in "${!REBASE_ORG[@]}"; do
    ORG_NUM="${REBASE_ORG[$ACC]}"
    if [[ "$ORG_NUM" == "TODO" ]]; then
        echo "[SKIP] $ACC -- REBASE org number not set"
        continue
    fi
    if [[ "$ORG_NUM" == "0" ]]; then
        echo "[SKIP] $ACC -- no REBASE match for this species"
        continue
    fi

    REBASE_CSV="$SPLIT/$CANONICAL/$ACC/rebase_motifs.csv"
    echo ""
    echo "=== $ACC (org $ORG_NUM) ==="
    kinsim-prep rebase fetch "$ORG_NUM" --output "$REBASE_CSV"
done

echo ""
echo "Done. rebase_motifs.csv files written into each species folder under:"
echo "  $SPLIT/$CANONICAL/"
echo ""
echo "Next step: run prep_MSA1003_merge.sh to merge with calling motifs."
