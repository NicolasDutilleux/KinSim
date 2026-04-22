#!/bin/bash
#SBATCH --job-name=vega_test
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega dual-caller smoke test (any barcode)
#
# Usage:  sbatch slurm_kinsim/vega/test.slurm <barcode>
#
#   <barcode> — bcXXXX (e.g. bc2033, bc2046 for E. coli MG1655)
#
# Runs BOTH calling pipelines on one raw HiFi BAM:
#
#   ipdSummary pipeline (m6A + m4C):
#     bystrandify → pbmm2 → index → ipdSummary SP3-C3 → pbmotifmaker
#     → ${SAMPLE}_motifs_ipdsummary.csv
#
#   jasmine pipeline (5mC):
#     jasmine --keep-kinetics → pbmm2 → index → modkit pileup
#     → modkit find-motifs → combined-parser CSV
#     → ${SAMPLE}_motifs_jasmine.csv
#
# jasmine runs BEFORE the SMRT-Link module load (ipdSummary) so the
# conda tools (jasmine, modkit) see a clean environment.
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SAMPLE=${1:?"Usage: sbatch $0 <barcode>   e.g. bc2046 (E. coli), bc2033 (Listeria)"}
# Accept "2046" or "bc2046"
case "$SAMPLE" in bc*) ;; *) SAMPLE="bc${SAMPLE}" ;; esac

# ---- config ----
VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

RAW_BAM="${VEGA}/m21026_260313_002345.hifi_reads.${SAMPLE}.bam"
ASM_FASTA="${VEGA}/assembly/${SAMPLE}/${SAMPLE}_assembly.fasta"
OUTDIR="${VEGA}/test_dual/${SAMPLE}"
mkdir -p "$OUTDIR"

# ---- ipdSummary pipeline paths ----
BYSTRAND="${OUTDIR}/${SAMPLE}_bystrandify.bam"
IPD_ALIGNED="${OUTDIR}/${SAMPLE}_ipd_aligned.bam"
IPD_GFF="${OUTDIR}/${SAMPLE}_ipdSummary.gff"
IPD_CSV="${OUTDIR}/${SAMPLE}_ipdSummary.csv"
MOTIFS_IPD="${OUTDIR}/${SAMPLE}_motifs_ipdsummary.csv"

# ---- jasmine pipeline paths ----
JAS_BAM="${OUTDIR}/${SAMPLE}_jasmine.bam"
JAS_ALIGNED="${OUTDIR}/${SAMPLE}_jas_aligned.bam"
PILEUP="${OUTDIR}/${SAMPLE}_pileup.bed"
FIND_TSV="${OUTDIR}/${SAMPLE}_find_motifs.tsv"
MOTIFS_JAS="${OUTDIR}/${SAMPLE}_motifs_jasmine.csv"

# ---- migrate legacy 'track A/B' filenames (only affects bc2033 from first run) ----
for pair in \
    "${OUTDIR}/${SAMPLE}_A_aligned.bam:${IPD_ALIGNED}" \
    "${OUTDIR}/${SAMPLE}_A_aligned.bam.bai:${IPD_ALIGNED}.bai" \
    "${OUTDIR}/${SAMPLE}_A_aligned.bam.pbi:${IPD_ALIGNED}.pbi" \
    "${OUTDIR}/${SAMPLE}_B_aligned.bam:${JAS_ALIGNED}" \
    "${OUTDIR}/${SAMPLE}_B_aligned.bam.bai:${JAS_ALIGNED}.bai"; do
    old=${pair%%:*}; new=${pair##*:}
    [ -f "$old" ] && [ ! -f "$new" ] && mv -v "$old" "$new"
done 2>/dev/null || true

# ---- symlink existing outputs from the original Vega pipeline (assembly/<bc>/)
# into test_dual/<bc>/ so we skip redundant compute. Names differ only for the
# bystrandify-aligned BAM which we rename to _ipd_aligned.* in test_dual/.
ASM_DIR="${VEGA}/assembly/${SAMPLE}"
if [ -d "$ASM_DIR" ]; then
    for pair in \
        "${ASM_DIR}/${SAMPLE}_bystrandify.bam:${BYSTRAND}" \
        "${ASM_DIR}/${SAMPLE}_bystrandify.bam.pbi:${BYSTRAND}.pbi" \
        "${ASM_DIR}/${SAMPLE}_aligned.bam:${IPD_ALIGNED}" \
        "${ASM_DIR}/${SAMPLE}_aligned.bam.bai:${IPD_ALIGNED}.bai" \
        "${ASM_DIR}/${SAMPLE}_aligned.bam.pbi:${IPD_ALIGNED}.pbi" \
        "${ASM_DIR}/${SAMPLE}_ipdSummary.gff:${IPD_GFF}" \
        "${ASM_DIR}/${SAMPLE}_ipdSummary.csv:${IPD_CSV}" \
        "${ASM_DIR}/${SAMPLE}_motifs.csv:${MOTIFS_IPD}"; do
        src=${pair%%:*}; dst=${pair##*:}
        if [ -e "$src" ] && [ ! -e "$dst" ]; then
            ln -sv "$src" "$dst"
        fi
    done
fi

echo "========================================================"
echo "  Vega dual-caller test — $SAMPLE"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "Raw BAM:    $RAW_BAM"
echo "Reference:  $ASM_FASTA"
echo "Outdir:     $OUTDIR"
echo "========================================================"

[ -f "$RAW_BAM" ]   || { echo "ERROR: raw HiFi BAM missing: $RAW_BAM"; exit 1; }
[ -f "$ASM_FASTA" ] || { echo "ERROR: assembly missing: $ASM_FASTA"; exit 1; }
[ -f "${ASM_FASTA}.fai" ] || samtools faidx "$ASM_FASTA"

T0=$(date +%s)

# ============================================================
# ipdSummary pipeline — part 1 (pre-alignment)
# The ipdSummary step itself runs later, after the jasmine pipeline.
# ============================================================
echo ""
echo "####################  ipdSummary pipeline  ####################"

if [ -s "$BYSTRAND" ]; then
    echo "=== bystrandify SKIP — BAM exists ==="
else
    echo "=== ccs-kinetics-bystrandify ==="
    START=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        ccs-kinetics-bystrandify "$RAW_BAM" "$BYSTRAND"
    echo "    $(( $(date +%s) - START ))s"
fi
[ -s "$BYSTRAND" ] || { echo "FAILED: bystrandify produced no output"; exit 1; }

[ -f "${BYSTRAND}.pbi" ] || {
    echo "=== pbindex bystrandify ===";
    apptainer exec --bind /data "$SMRT_SIF" pbindex "$BYSTRAND";
}

if [ -s "$IPD_ALIGNED" ]; then
    echo "=== pbmm2 align (bystrandify) SKIP — aligned BAM exists ==="
else
    echo "=== pbmm2 align (bystrandify → assembly) ==="
    START=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        pbmm2 align "$ASM_FASTA" "$BYSTRAND" "$IPD_ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
    echo "    $(( $(date +%s) - START ))s"
fi

[ -f "${IPD_ALIGNED}.bai" ] || samtools index "$IPD_ALIGNED"
[ -f "${IPD_ALIGNED}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$IPD_ALIGNED"

# ============================================================
# jasmine pipeline (5mC) — runs BEFORE the SMRT-Link module load
# so conda jasmine + modkit see a clean environment.
# ============================================================
echo ""
echo "####################  jasmine pipeline  ####################"

# ---- decide whether jasmine 5mC calling is needed ----
# Skip ONLY if the input BAM already carries 5mC MM tag (C+m). An existing
# jasmine @PG is NOT sufficient: PacBio Revio production jasmine is often
# run with --6ma --trt (6mA detection only), not 5mC. We need C+m specifically
# for the CCWGG / 5mC motif discovery downstream.
needs_jasmine=true
if [ -s "$JAS_BAM" ]; then
    # pre-computed jasmine output exists → use it (assumed 5mC from a prior run)
    echo "=== jasmine SKIP — pre-computed jasmine BAM exists: $JAS_BAM ==="
    needs_jasmine=false
else
    echo "=== probing RAW BAM for existing 5mC (C+m) calls ==="
    # look at first 500 reads' MM tags for C+m
    if apptainer exec --bind /data "$SMRT_SIF" samtools view "$RAW_BAM" 2>/dev/null \
         | head -500 | grep -qE $'MM:Z:[^\t]*C\\+m'; then
        echo "    RAW BAM has C+m (5mC) in MM tag → reuse as JAS_BAM"
        JAS_BAM="$RAW_BAM"
        needs_jasmine=false
    else
        echo "    RAW BAM has no C+m tag (jasmine @PG may be for 6mA/other) → jasmine will run"
    fi
fi

if $needs_jasmine; then
    # NOTE: jasmine 2.4.0 has no --model flag. Its default classifier is 5mC,
    # trained on Revio (R/P2-C2). On Vega (R/P1-C1) a chemistry WARN is emitted;
    # calls proceed but are miscalibrated.
    echo "=== jasmine (conda kinsim_env, --keep-kinetics, default=5mC) ==="
    command -v jasmine >/dev/null || { echo "ERROR: jasmine not on PATH"; exit 1; }
    START=$(date +%s)
    jasmine "$RAW_BAM" "$JAS_BAM" --keep-kinetics
    echo "    $(( $(date +%s) - START ))s"
    # Verify the produced BAM actually has 5mC tags
    if ! apptainer exec --bind /data "$SMRT_SIF" samtools view "$JAS_BAM" 2>/dev/null \
         | head -500 | grep -qE $'MM:Z:[^\t]*C\\+m'; then
        echo "WARN: jasmine output BAM has no C+m tag in first 500 reads — 5mC calls may be sparse"
    fi
fi
[ -s "$JAS_BAM" ] || { echo "FAILED: jasmine BAM missing"; exit 1; }

[ -f "${JAS_BAM}.pbi" ] || {
    echo "=== pbindex jasmine ===";
    apptainer exec --bind /data "$SMRT_SIF" pbindex "$JAS_BAM";
}

if [ -s "$JAS_ALIGNED" ]; then
    echo "=== pbmm2 align (jasmine) SKIP — aligned BAM exists ==="
else
    echo "=== pbmm2 align (jasmine → assembly) ==="
    START=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        pbmm2 align "$ASM_FASTA" "$JAS_BAM" "$JAS_ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
    echo "    $(( $(date +%s) - START ))s"
fi

[ -f "${JAS_ALIGNED}.bai" ] || samtools index "$JAS_ALIGNED"

if [ -s "$PILEUP" ]; then
    echo "=== modkit pileup SKIP — BED exists ($(wc -l < "$PILEUP") sites) ==="
else
    echo "=== modkit pileup ==="
    START=$(date +%s)
    modkit pileup "$JAS_ALIGNED" "$PILEUP" \
        --ref "$ASM_FASTA" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        --filter-threshold 0.5 \
        --log-filepath "${OUTDIR}/${SAMPLE}_modkit.log"
    echo "    $(( $(date +%s) - START ))s  ($(wc -l < "$PILEUP") sites)"
fi

if [ -s "$FIND_TSV" ]; then
    echo "=== modkit find-motifs SKIP — TSV exists ==="
else
    echo "=== modkit find-motifs ==="
    START=$(date +%s)
    modkit find-motifs \
        --in-bedmethyl "$PILEUP" \
        --ref "$ASM_FASTA" \
        --out-table "$FIND_TSV" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        || { echo "modkit find-motifs failed — writing empty TSV"; : > "$FIND_TSV"; }
    echo "    $(( $(date +%s) - START ))s"
fi

echo "=== emit jasmine motifs.csv (combined-parser format) ==="
python - <<PY
import csv
from pathlib import Path

tsv = Path('${FIND_TSV}')
out = Path('${MOTIFS_JAS}')

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
            nsite  = _cand(r, 'n_sites', 'num_sites', 'total_sites')
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

# ============================================================
# ipdSummary pipeline — part 2 (kinetic calls + motif discovery)
# SMRT-Link module is loaded HERE only; jasmine pipeline above
# already ran with a clean environment.
# ============================================================
echo ""
echo "####################  ipdSummary pipeline (continued)  ####################"

if [ -s "$IPD_GFF" ]; then
    echo "=== ipdSummary SKIP — GFF exists ($(grep -c '^[^#]' "$IPD_GFF" || true) records) ==="
else
    echo "=== ipdSummary SP3-C3 (--identify m6A,m4C --methylFraction) ==="
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true
    command -v ipdSummary >/dev/null || { echo "ERROR: ipdSummary not on PATH after module load"; exit 1; }
    START=$(date +%s)
    ipdSummary "$IPD_ALIGNED" \
        --reference "$ASM_FASTA" \
        --identify m6A,m4C \
        --methylFraction \
        --csv "$IPD_CSV" \
        --gff "$IPD_GFF" \
        --numWorkers "${SLURM_CPUS_PER_TASK}" \
        --ipdModel "$SP3_MODEL"
    echo "    $(( $(date +%s) - START ))s"
    echo "    GFF records: $(grep -c '^[^#]' "$IPD_GFF" || true)"
fi

if [ -s "$MOTIFS_IPD" ]; then
    echo "=== pbmotifmaker SKIP — motifs.csv exists ==="
else
    echo "=== pbmotifmaker find ==="
    START=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        pbmotifmaker find "$ASM_FASTA" "$IPD_GFF" "$MOTIFS_IPD"
    echo "    $(( $(date +%s) - START ))s"
fi

# ============================================================
# Summary — both motifs.csv must exist at this point
# ============================================================
echo ""
echo "========================================================"
echo "  DUAL-CALLER TEST COMPLETE — $SAMPLE"
echo "========================================================"
echo "  Elapsed total: $(( $(date +%s) - T0 ))s"
echo ""

fail=0
for path in "$MOTIFS_IPD" "$MOTIFS_JAS"; do
    if [ ! -e "$path" ]; then
        echo "  MISSING output: $path"; fail=1
    fi
done

echo "  ipdSummary → pbmotifmaker:"
echo "    $MOTIFS_IPD"
if [ -s "$MOTIFS_IPD" ]; then
    echo "    rows: $(tail -n +2 "$MOTIFS_IPD" | wc -l)"
    echo "    --- head ---"
    head -5 "$MOTIFS_IPD" | sed 's/^/      /'
else
    echo "    (header-only or empty)"
fi
echo ""
echo "  jasmine → modkit find-motifs:"
echo "    $MOTIFS_JAS"
if [ -s "$MOTIFS_JAS" ]; then
    echo "    rows: $(tail -n +2 "$MOTIFS_JAS" | wc -l)"
    echo "    --- head ---"
    head -5 "$MOTIFS_JAS" | sed 's/^/      /'
else
    echo "    (header-only or empty — 0 motifs found)"
fi
echo "========================================================"
exit $fail
