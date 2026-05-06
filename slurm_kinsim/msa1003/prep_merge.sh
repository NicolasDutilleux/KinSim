#!/bin/bash
# =============================================================================
# prep_merge.sh
#
# For each species in each experience under SPLIT_DIR:
#   1. Merge calling motifs + REBASE motifs into final_motifs.csv
#   2. Write one manifest row per BAM
#
# Auto-discovers experiences (MSA1003.* subdirs) and species (subdirs within).
#
# Usage:
#   bash slurm_kinsim/msa1003/prep_merge.sh [SPLIT_DIR] [MANIFEST]
#
# Defaults:
#   SPLIT_DIR = $BASE/trimmed_species_by_experience
#   MANIFEST  = $BASE/manifest.csv
# =============================================================================

set -euo pipefail

BASE=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK
SPLIT="${1:-$BASE/trimmed_species_by_experience}"
MANIFEST="${2:-$BASE/manifest.csv}"

echo "========================================"
echo "  Merge motifs + build manifest"
echo "========================================"
echo "  Input:    $SPLIT"
echo "  Manifest: $MANIFEST"
echo ""

# Auto-discover experiences
EXPERIENCES=($(ls -d "$SPLIT"/MSA1003.* 2>/dev/null | xargs -n1 basename | sort))
if [ ${#EXPERIENCES[@]} -eq 0 ]; then
    echo "ERROR: No MSA1003.* experience directories found in $SPLIT"
    exit 1
fi
echo "  Experiences: ${#EXPERIENCES[@]}"

# ---------------------------------------------------------------------------
# Main loop: discover species from directory, merge motifs, build manifest
# ---------------------------------------------------------------------------
echo "sample_id,bam_path,motifs" > "$MANIFEST"

R=1
for EXP in "${EXPERIENCES[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Experience: $EXP  (replicate $R)"
    echo "=========================================="

    for SPECIES_DIR in "$SPLIT/$EXP"/*/; do
        [ -d "$SPECIES_DIR" ] || continue
        ACC=$(basename "$SPECIES_DIR")

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
        python -m kinsim.utils.parsers.motif_merge "${INPUTS[@]}" \
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
echo "  python scripts/manifest.py validate $MANIFEST"
