#!/bin/bash
# ============================================================
# build_samplesheet.sh — generic samplesheet builder for PREPARE
#
# Walks a directory of BAM (or BAM.GZ) files, optionally pairs each with a
# per-sample reference FASTA, and writes a samplesheet:
#
#   sample_id,reads[,reference]
#
# Usage:
#   bash nextflow/helpers/build_samplesheet.sh \\
#       --reads-dir   <dir of .bam / .bam.gz>           \\
#       --output      <path/to/samplesheet.csv>         \\
#       [--refs-dir   <dir of references, .fna|.fa>]    \\
#       [--reads-glob '*.bam']                          \\
#       [--sample-from name|parent]                     \\
#       [--include 'bc20??']                            \\
#       [--exclude 'bc2038']                            \\
#       [--dry-run]
#
# Options:
#   --reads-dir      directory to scan for read BAMs (required)
#   --output         output CSV path (required)
#   --refs-dir       directory containing one <sample_id>.fna per sample
#                    (optional; unset → reference column is empty and
#                    hifiasm will assemble de novo if --assemble is on)
#   --reads-glob     glob for BAM filenames (default: '*.bam *.bam.gz')
#   --sample-from    'name'   → sample_id = filename minus extensions (default)
#                    'parent' → sample_id = parent directory name
#   --include PAT    include only files/dirs matching shell glob (repeatable)
#   --exclude PAT    exclude files/dirs matching shell glob (repeatable)
#   --dry-run        print what would be written, don't write
#
# Example (Vega — de novo assembly):
#   bash nextflow/helpers/build_samplesheet.sh \\
#       --reads-dir /data/.../Vega/raw \\
#       --reads-glob '*.hifi_reads.bam' \\
#       --exclude   'bc2038*' \\
#       --output    nextflow/samplesheets/vega.csv
#
# Example (Strepto — pre-built refs):
#   bash nextflow/helpers/build_samplesheet.sh \\
#       --reads-dir /data/.../Strepto/hifi \\
#       --refs-dir  /data/.../Strepto/refs \\
#       --output    nextflow/samplesheets/strepto.csv
# ============================================================

set -euo pipefail

READS_DIR=
REFS_DIR=
OUTPUT=
READS_GLOB='*.bam *.bam.gz'
SAMPLE_FROM=name
INCLUDE=()
EXCLUDE=()
DRY_RUN=0

usage() { sed -n '2,37p' "$0"; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --reads-dir)    READS_DIR=$2; shift 2 ;;
        --refs-dir)     REFS_DIR=$2;  shift 2 ;;
        --output)       OUTPUT=$2;    shift 2 ;;
        --reads-glob)   READS_GLOB=$2; shift 2 ;;
        --sample-from)  SAMPLE_FROM=$2; shift 2 ;;
        --include)      INCLUDE+=("$2"); shift 2 ;;
        --exclude)      EXCLUDE+=("$2"); shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      usage ;;
        *)              echo "Unknown arg: $1" >&2; usage ;;
    esac
done

[ -n "$READS_DIR" ] || { echo "ERROR: --reads-dir required" >&2; exit 1; }
[ -n "$OUTPUT"    ] || { echo "ERROR: --output required"    >&2; exit 1; }
[ -d "$READS_DIR" ] || { echo "ERROR: reads dir not found: $READS_DIR" >&2; exit 1; }
[ -z "$REFS_DIR" ] || [ -d "$REFS_DIR" ] || {
    echo "ERROR: refs dir not found: $REFS_DIR" >&2; exit 1;
}

strip_exts() {   # strip .bam.gz or .bam or any other trailing extensions pair
    local f=$1
    f=${f##*/}
    f=${f%.gz}
    f=${f%.bam}
    echo "$f"
}

matches_any() {
    local s=$1; shift
    for p in "$@"; do
        case "$s" in $p) return 0 ;; esac
    done
    return 1
}

# Collect candidate read files
readarray -t CANDIDATES < <(
    for g in $READS_GLOB; do
        find "$READS_DIR" -maxdepth 2 -type f -name "$g" 2>/dev/null || true
    done | sort -u
)

if [ ${#CANDIDATES[@]} -eq 0 ]; then
    echo "ERROR: no files matched under $READS_DIR (glob: $READS_GLOB)" >&2
    exit 1
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "sample_id,reads,reference" > "$TMP"

N_WROTE=0
N_SKIPPED=0

for reads in "${CANDIDATES[@]}"; do
    base=${reads##*/}
    parent=$(basename "$(dirname "$reads")")

    case "$SAMPLE_FROM" in
        name)   sid=$(strip_exts "$reads") ;;
        parent) sid=$parent ;;
        *) echo "ERROR: --sample-from must be name|parent" >&2; exit 1 ;;
    esac

    # Include/exclude filters apply to sample_id and basename
    if [ ${#INCLUDE[@]} -gt 0 ] \
        && ! matches_any "$sid" "${INCLUDE[@]}" \
        && ! matches_any "$base" "${INCLUDE[@]}"; then
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    if [ ${#EXCLUDE[@]} -gt 0 ] \
        && { matches_any "$sid" "${EXCLUDE[@]}" \
            || matches_any "$base" "${EXCLUDE[@]}"; }; then
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi

    # Resolve reference if --refs-dir given
    ref=""
    if [ -n "$REFS_DIR" ]; then
        for ext in fna fa fasta fna.gz fa.gz fasta.gz; do
            candidate="${REFS_DIR}/${sid}.${ext}"
            if [ -f "$candidate" ]; then
                ref=$candidate
                break
            fi
        done
    fi

    echo "${sid},${reads},${ref}" >> "$TMP"
    N_WROTE=$((N_WROTE + 1))
done

if [ "$DRY_RUN" = "1" ]; then
    echo "(dry-run) would write ${N_WROTE} rows to ${OUTPUT} (${N_SKIPPED} filtered)"
    cat "$TMP"
    exit 0
fi

mkdir -p "$(dirname "$OUTPUT")"
mv "$TMP" "$OUTPUT"
trap - EXIT

echo "Wrote ${N_WROTE} rows → ${OUTPUT}  (${N_SKIPPED} filtered out)"
echo
head -n 5 "$OUTPUT"
[ "$N_WROTE" -gt 4 ] && echo "..."
