#!/bin/bash
# ============================================================
# Sequel step 06 — build GFF manifest (sample_id, bam_path, motifs, gff)
#
# Walks all bcXXXX dirs under ipdsummary_ccs/ and keeps any that have
# aligned BAM + ipdSummary GFF. motifs column points to per-barcode
# motifs.csv if produced by step 05, else empty.
#
# If ${SEQUEL}/48-plex_sample_list.tsv exists, sample_id is renamed to
# the species name from that mapping (PacBio 48-plex metadata, barcode
# → organism). Falls back to raw bcXXXX if unavailable.
#
# Usage:  bash slurm_kinsim/sequel/06_build_manifest.sh
# Output: ${SEQUEL}/manifest_sequel_ccs_gff.csv
# ============================================================

set -euo pipefail

SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
OUTBASE=${SEQUEL}/ipdsummary_ccs
MANIFEST=${SEQUEL}/manifest_sequel_ccs_gff.csv
SAMPLE_LIST=${SEQUEL}/48-plex_sample_list.tsv

declare -A SP_NAME
if [ -f "$SAMPLE_LIST" ]; then
    # Tab-separated; expected columns include barcode and organism name.
    # Tolerates header + arbitrary column order by taking col1=barcode, col2=species.
    while IFS=$'\t' read -r bc name rest; do
        [ -z "$bc" ] && continue
        [[ "$bc" =~ ^# ]] && continue
        [[ "$bc" == "barcode" ]] && continue
        # Normalize: strip whitespace, replace spaces with _
        name=$(echo "$name" | tr ' ' '_' | tr -d '\r')
        SP_NAME[$bc]="$name"
    done < "$SAMPLE_LIST"
    echo "Loaded ${#SP_NAME[@]} species names from $SAMPLE_LIST"
fi

mkdir -p "$(dirname "$MANIFEST")"
echo "sample_id,bam_path,motifs,gff" > "$MANIFEST"

kept=0; skipped=0
for SPECIES_DIR in "$OUTBASE"/bc*/; do
    [ -d "$SPECIES_DIR" ] || continue
    BC=$(basename "$SPECIES_DIR")
    ALIGNED="${SPECIES_DIR}/${BC}_aligned.bam"
    GFF="${SPECIES_DIR}/${BC}_ipdSummary.gff"
    MOTIFS="${SPECIES_DIR}/${BC}_motifs.csv"

    if [ ! -s "$ALIGNED" ] || [ ! -s "$GFF" ]; then
        echo "skip $BC — missing aligned BAM or GFF" >&2
        skipped=$((skipped + 1)); continue
    fi
    [ -s "$MOTIFS" ] || MOTIFS=""
    SID=${SP_NAME[$BC]:-$BC}
    echo "${SID},${ALIGNED},${MOTIFS},${GFF}" >> "$MANIFEST"
    kept=$((kept + 1))
done

echo "Wrote $MANIFEST ($kept rows, $skipped skipped)"
python scripts/manifest.py validate "$MANIFEST" || true
