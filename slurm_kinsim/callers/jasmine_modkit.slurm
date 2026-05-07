#!/bin/bash
#SBATCH --job-name=call_jasmine_modkit
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# callers/jasmine_modkit — 5mC CpG motif discovery via jasmine + modkit
#
# Usage:
#   sbatch -J call_jm_<sample> slurm_kinsim/callers/jasmine_modkit.slurm \
#          <raw_bam> <reference_fasta> <out_motifs_csv> [work_dir]
#
# Strategy:
#   1. If raw BAM already has C+m MM tag (production jasmine): reuse directly
#   2. Else: run conda jasmine (default = 5mC model) to add C+m tags
#   3. pbmm2 align raw/jasmine BAM → reference (preserves MM/ML)
#   4. samtools index
#   5. modkit pileup → bedMethyl
#   6. modkit find-motifs → TSV → combined-parser CSV
#
# Output: combined-parser format
#   mod_type,motif,offset,frac_mod,n_sites,source
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

RAW_BAM=${1:?"Usage: sbatch $0 <raw_bam> <ref> <out_motifs_csv> [work_dir]"}
REF=${2:?"reference missing"}
OUT_CSV=${3:?"out_motifs_csv missing"}
WORK=${4:-$(dirname "$OUT_CSV")/jasmine_work}

SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

# SKIP first — if the per-strain motifs_jasmine.csv already exists we
# don't need raw_bam, ref, or any compute. (Common case after the user
# has run jasmine once and the raw BAMs have been archived.)
if [ -s "$OUT_CSV" ]; then
    echo "SKIP — $OUT_CSV exists ($(tail -n +2 "$OUT_CSV" | wc -l) motif rows)"
    exit 0
fi

[ -f "$RAW_BAM" ]    || { echo "ERROR: raw BAM missing"; exit 1; }
[ -f "$REF" ]        || { echo "ERROR: reference missing"; exit 1; }
[ -f "${REF}.fai" ]  || samtools faidx "$REF"
mkdir -p "$WORK" "$(dirname "$OUT_CSV")"

SAMPLE=$(basename "${OUT_CSV%_motifs_jasmine.csv}")
[ "$SAMPLE" = "$OUT_CSV" ] && SAMPLE=$(basename "${OUT_CSV%.csv}")

JAS_BAM="${WORK}/${SAMPLE}_jasmine.bam"
ALIGNED="${WORK}/${SAMPLE}_jas_aligned.bam"
PILEUP="${WORK}/${SAMPLE}_pileup.bed"
FIND_TSV="${WORK}/${SAMPLE}_find_motifs.tsv"

echo "========================================================"
echo "  callers/jasmine_modkit — $SAMPLE"
echo "========================================================"
echo "Raw BAM:   $RAW_BAM"
echo "Reference: $REF"
echo "Out CSV:   $OUT_CSV"
echo "Work dir:  $WORK"

T0=$(date +%s)

# --- 1. decide whether jasmine 5mC needs to run ---
needs_jasmine=true
if [ -s "$JAS_BAM" ]; then
    echo "=== jasmine SKIP — pre-computed BAM exists ==="
    needs_jasmine=false
else
    echo "=== probing raw BAM for existing 5mC (C+m) MM tag ==="
    if apptainer exec --bind /data "$SMRT_SIF" samtools view "$RAW_BAM" 2>/dev/null \
         | head -500 | grep -qE $'MM:Z:[^\t]*C\\+m'; then
        echo "    RAW BAM already has C+m → reuse as JAS_BAM (skip jasmine)"
        JAS_BAM="$RAW_BAM"
        needs_jasmine=false
    else
        echo "    RAW BAM lacks C+m → running conda jasmine 5mC default"
    fi
fi

if $needs_jasmine; then
    command -v jasmine >/dev/null || { echo "ERROR: jasmine not on PATH"; exit 1; }
    T=$(date +%s)
    jasmine "$RAW_BAM" "$JAS_BAM" --keep-kinetics
    echo "    jasmine elapsed: $(( $(date +%s) - T ))s"
fi
[ -s "$JAS_BAM" ] || { echo "FAILED: jasmine BAM missing"; exit 1; }

# ensure .pbi for pbmm2
[ -f "${JAS_BAM}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$JAS_BAM"

# --- 2. pbmm2 align (preserves MM/ML) ---
if [ -s "$ALIGNED" ]; then
    echo "=== pbmm2 SKIP — aligned BAM exists ==="
else
    echo "=== pbmm2 align (jasmine BAM → reference) ==="
    T=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        pbmm2 align "$REF" "$JAS_BAM" "$ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
    echo "    $(( $(date +%s) - T ))s"
fi
[ -f "${ALIGNED}.bai" ] || samtools index "$ALIGNED"

# --- 3. modkit pileup ---
if [ -s "$PILEUP" ]; then
    echo "=== modkit pileup SKIP — BED exists ($(wc -l < "$PILEUP") sites) ==="
else
    echo "=== modkit pileup ==="
    T=$(date +%s)
    modkit pileup "$ALIGNED" "$PILEUP" \
        --ref "$REF" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        --filter-threshold 0.5 \
        --log-filepath "${WORK}/modkit.log"
    echo "    $(( $(date +%s) - T ))s  ($(wc -l < "$PILEUP") sites)"
fi

# --- 4. modkit find-motifs ---
if [ -s "$FIND_TSV" ]; then
    echo "=== modkit find-motifs SKIP — TSV exists ==="
else
    echo "=== modkit find-motifs ==="
    T=$(date +%s)
    modkit find-motifs \
        --in-bedmethyl "$PILEUP" \
        --ref "$REF" \
        --out-table "$FIND_TSV" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        || { echo "modkit find-motifs failed — writing empty TSV"; : > "$FIND_TSV"; }
    echo "    $(( $(date +%s) - T ))s"
fi

# --- 5. emit motifs.csv in combined-parser format ---
echo "=== emit $OUT_CSV ==="
python3 - <<PY
import csv
from pathlib import Path

tsv = Path('${FIND_TSV}')
out = Path('${OUT_CSV}')

def _cand(d, *names):
    for n in names:
        if n in d: return d[n]
    return None

rows = []
if tsv.exists() and tsv.stat().st_size > 0:
    with tsv.open() as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for r in rdr:
            motif  = _cand(r, 'motif', 'sequence')
            offset = _cand(r, 'mod_offset', 'offset', 'position')
            frac   = _cand(r, 'frac_mod', 'fraction', 'methylated_fraction')
            hi     = _cand(r, 'high_count')
            mid    = _cand(r, 'mid_count')
            lo     = _cand(r, 'low_count')
            nsite  = _cand(r, 'n_sites', 'num_sites', 'total_sites')
            if nsite is None:
                parts = [int(x) for x in (hi, mid, lo) if x not in (None, '')]
                nsite = sum(parts) if parts else 0
            mod    = _cand(r, 'mod_code', 'mod_type') or 'm'
            if not motif or offset is None:
                continue
            mod_type = 'm5C' if mod in ('m', 'm5C', '5mC') else mod
            rows.append((mod_type, motif, int(offset),
                         float(frac or 0.0), int(float(nsite or 0)),
                         'jasmine+modkit'))

with out.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mod_type','motif','offset','frac_mod','n_sites','source'])
    for r in rows:
        w.writerow(r)

print(f'Wrote {out}  ({len(rows)} motifs)')
PY

echo ""
echo "Total elapsed: $(( $(date +%s) - T0 ))s"
[ -f "$OUT_CSV" ] && echo "  output rows: $(tail -n +2 "$OUT_CSV" | wc -l)"
