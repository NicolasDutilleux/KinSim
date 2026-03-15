#!/bin/bash
# =============================================================================
# prep_MSA1003_rebase.sh
#
# For each species in the MSA1003 mock community that has a REBASE organism
# number assigned, fetch genuine motifs from REBASE and write rebase_motifs.csv
# into EVERY experience folder (all 4 replicates).
#
# REBASE is fetched once per species, then copied to all experiences.
#
# Run ONCE, before prep_MSA1003_merge.sh.
#
# Output per species per experience:
#   $SPLIT/<experience>/$ACC/rebase_motifs.csv
#
# Usage:
#   conda activate kinsim_env
#   bash slurm_kinsim/prep_MSA1003_rebase.sh
# =============================================================================

set -euo pipefail

BASE=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK
SPLIT=$BASE/species_split_by_experience

# All 4 replicates
EXPERIENCES=(
    MSA1003.490fb6ec_6--6
    MSA1003.8f6d4655_6--6
    MSA1003.be664413_6--6
    MSA1003.ee67a0a5_6--6
)

# ---------------------------------------------------------------------------
# REBASE organism numbers
# Find each at: https://rebase.neb.com/cgi-bin/pacbioget?<ORG_NUM>
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

    echo ""
    echo "=== $ACC (org $ORG_NUM) ==="

    # Fetch into the first experience
    FIRST_CSV="$SPLIT/${EXPERIENCES[0]}/$ACC/rebase_motifs.csv"
    kinsim-prep rebase fetch "$ORG_NUM" --output "$FIRST_CSV"

    # Copy to the other 3 experiences
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
echo "Next step: run prep_MSA1003_merge.sh to merge with calling motifs."
