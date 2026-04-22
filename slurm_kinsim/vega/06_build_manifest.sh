#!/bin/bash
# ============================================================
# Vega step 06 — assemble the GFF manifest CSV (sample_id, bam_path, motifs, gff)
#
# One row per barcode whose 04_ipdsummary output is present. Points the
# ML pipeline at the aligned BAM, motifs.csv (if present) and ipdSummary
# GFF for GFF-based extraction.
#
# bc2038 is excluded: moved to assembly/removed/ — fragmented (146 contigs,
# 12.82 Mb, likely contamination from another organism).
#
# If ${VEGA}/species_id.txt exists (BLAST-based barcode → species mapping,
# produced by vega_00_blast_identify.sh), sample_id is renamed to the
# species name; otherwise the raw bcXXXX id is used.
#
# Usage:
#   bash slurm_kinsim/vega/06_build_manifest.sh
#
# Output: ${VEGA}/manifest_vega_gff.csv
# ============================================================

set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
OUTBASE=${VEGA}/assembly
MANIFEST=${VEGA}/manifest_vega_gff.csv
SPECIES_ID=${VEGA}/species_id.txt

BARCODES=(2033 2034 2035 2036 2037 2039 2040
          2041 2042 2043 2044 2045 2046 2047 2048)
# bc2038 excluded — fragmented assembly, likely contamination

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

kept=0
skipped=0
for BC in "${BARCODES[@]}"; do
    SAMPLE="bc${BC}"
    SPECIES_DIR="${OUTBASE}/${SAMPLE}"
    ALIGNED="${SPECIES_DIR}/${SAMPLE}_aligned.bam"
    GFF="${SPECIES_DIR}/${SAMPLE}_ipdSummary.gff"
    MOTIFS="${SPECIES_DIR}/${SAMPLE}_motifs.csv"

    if [ ! -s "$ALIGNED" ] || [ ! -s "$GFF" ]; then
        echo "skip $SAMPLE — missing aligned BAM or GFF" >&2
        skipped=$((skipped + 1))
        continue
    fi

    SID=${SP_NAME[$SAMPLE]:-$SAMPLE}
    [ -s "$MOTIFS" ] || MOTIFS=""
    echo "${SID},${ALIGNED},${MOTIFS},${GFF}" >> "$MANIFEST"
    kept=$((kept + 1))
done

echo "Wrote $MANIFEST ($kept rows kept, $skipped skipped)"
kinsim-prep manifest validate "$MANIFEST" || true
