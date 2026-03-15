#!/bin/bash
# =============================================================================
# prep_MSA1003_merge.sh
#
# For each species in each MSA1003 experience:
#   1. Merge calling motifs ({ACC}_motifs.csv) with REBASE motifs
#      (rebase_motifs.csv, if present) into final_motifs.csv
#   2. Write one manifest.csv row per BAM
#
# Run AFTER prep_MSA1003_rebase.sh.
#
# Output per species per experience:
#   $SPLIT/<experience>/$ACC/final_motifs.csv
#
# Manifest:
#   $BASE/manifest.csv
#
# Usage:
#   conda activate kinsim_env
#   bash slurm_kinsim/prep_MSA1003_merge.sh
# =============================================================================

set -euo pipefail

BASE=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK
SPLIT=$BASE/species_split_by_experience
MANIFEST=$BASE/manifest.csv

# All 4 replicates
EXPERIENCES=(
    MSA1003.490fb6ec_6--6
    MSA1003.8f6d4655_6--6
    MSA1003.be664413_6--6
    MSA1003.ee67a0a5_6--6
)

# ---------------------------------------------------------------------------
# Species list (ordered)
# ---------------------------------------------------------------------------
SPECIES=(
    AE000511.1      # Helicobacter pylori 26695
    AE000513.1      # Deinococcus radiodurans R1 chromosome 1
    AE002098.2      # Neisseria meningitidis MC58
    AE009948.1      # Streptococcus agalactiae 2603V/R
    AE014133.2      # Streptococcus mutans UA159
    AE015929.1      # Staphylococcus epidermidis ATCC 12228
    AE017194.1      # Bacillus cereus ATCC 10987
    AP009256.1      # Bifidobacterium adolescentis ATCC 15703
    AP009380.1      # Porphyromonas gingivalis ATCC 33277
    CP000139.1      # Bacteroides vulgatus ATCC 8482
    CP000255.1      # Staphylococcus aureus USA300_FPR3757
    CP000413.1      # Lactobacillus gasseri ATCC 33323
    CP000521.1      # Acinetobacter baumannii ATCC 17978
    CP000577.1      # Cereibacter sphaeroides ATCC 17029 chromosome 1
    CP000744.1      # Pseudomonas paraeruginosa PA7
    CP003084.1      # Propionibacterium acnes ATCC 11828
    CP046315.1      # Schaalia odontolytica FDAARGOS_732
    NC_009050.1     # Cereibacter sphaeroides ATCC 17029 chromosome 2
    NC_017316.1     # Enterococcus faecalis OG1RF
    NZ_CP006777.1   # Clostridium beijerinckii ATCC 35702 SA-1
    U00096.3        # Escherichia coli K-12 MG1655
)

# ---------------------------------------------------------------------------
# Main loop: merge motifs per species per experience, build manifest
# ---------------------------------------------------------------------------
echo "sample_id,bam_path,motifs" > "$MANIFEST"

R=1
for EXP in "${EXPERIENCES[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Experience: $EXP  (replicate $R)"
    echo "=========================================="

    for ACC in "${SPECIES[@]}"; do
        SPECIES_DIR="$SPLIT/$EXP/$ACC"
        CALLING_CSV="$SPECIES_DIR/${ACC}_motifs.csv"
        REBASE_CSV="$SPECIES_DIR/rebase_motifs.csv"
        FINAL_CSV="$SPECIES_DIR/final_motifs.csv"

        echo ""
        echo "=== $ACC (r$R) ==="

        # -- Collect available inputs --
        INPUTS=()
        if [[ -f "$CALLING_CSV" ]]; then
            INPUTS+=("$CALLING_CSV")
        else
            echo "  [WARN] Calling motifs not found: $CALLING_CSV"
        fi
        if [[ -f "$REBASE_CSV" ]]; then
            INPUTS+=("$REBASE_CSV")
        fi

        if [[ ${#INPUTS[@]} -eq 0 ]]; then
            echo "  [ERROR] No motif inputs for $ACC -- skipping"
            continue
        fi

        # -- Merge calling + REBASE -> final_motifs.csv --
        kinsim-prep merge-motifs "${INPUTS[@]}" \
            --output "$FINAL_CSV" \
            --min-frac 0.8 \
            --min-sites 300

        # -- Add manifest row --
        BAM="$SPECIES_DIR/${ACC}.bam"
        if [[ -f "$BAM" ]]; then
            echo "${ACC}_r${R},${BAM},${FINAL_CSV}" >> "$MANIFEST"
        else
            echo "  [WARN] BAM not found: $BAM"
        fi
    done

    R=$((R + 1))
done

echo ""
echo "Done."
echo "  final_motifs.csv written into every species folder under $SPLIT/"
echo "  Manifest: $MANIFEST  ($(grep -c ',' "$MANIFEST" || true) rows)"
echo ""
echo "Next step: validate the manifest with:"
echo "  kinsim-prep manifest validate $MANIFEST"
