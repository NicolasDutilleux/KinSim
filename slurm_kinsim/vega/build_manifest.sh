#!/bin/bash
# ============================================================
# Vega manifest builder — new modular pipeline output layout
#
# One row per barcode whose merged motifs CSV is present. Points to
# the aligned BAM and the MERGED motifs CSV (ipdSummary + jasmine, filtered).
#
# bc2038 excluded (fragmented assembly / contamination).
# Species names picked from ${VEGA}/species_id.txt when available.
# ============================================================

set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
OUTBASE=${VEGA}/pipeline
MANIFEST=${VEGA}/manifest_vega.csv
SPECIES_ID=${VEGA}/species_id.txt

BARCODES=(2033 2034 2035 2036 2037 2039 2040
          2041 2042 2043 2044 2045 2046 2047 2048)

declare -A SP_NAME
if [ -f "$SPECIES_ID" ]; then
    while IFS=$'\t,' read -r bc name rest; do
        [ -z "$bc" ] && continue
        [[ "$bc" =~ ^# ]] && continue
        SP_NAME[$bc]="$name"
    done < "$SPECIES_ID"
    echo "Loaded ${#SP_NAME[@]} species names from $SPECIES_ID"
fi

mkdir -p "$(dirname "$MANIFEST")"
echo "sample_id,bam_path,motifs,gff" > "$MANIFEST"

kept=0; skipped=0
for BC in "${BARCODES[@]}"; do
    SAMPLE="bc${BC}"
    DIR="${OUTBASE}/${SAMPLE}"
    ALIGNED="${DIR}/${SAMPLE}_aligned.bam"
    MERGED="${DIR}/${SAMPLE}_motifs_merged.csv"
    GFF="${DIR}/${SAMPLE}_ipdSummary.gff"

    if [ ! -s "$ALIGNED" ] || [ ! -s "$MERGED" ]; then
        echo "skip $SAMPLE — missing aligned BAM or merged motifs.csv" >&2
        skipped=$((skipped + 1))
        continue
    fi
    SID=${SP_NAME[$SAMPLE]:-$SAMPLE}
    [ -s "$GFF" ] || GFF=""
    echo "${SID},${ALIGNED},${MERGED},${GFF}" >> "$MANIFEST"
    kept=$((kept + 1))
done

echo "Wrote $MANIFEST ($kept rows kept, $skipped skipped)"
kinsim-prep manifest validate "$MANIFEST" || true
