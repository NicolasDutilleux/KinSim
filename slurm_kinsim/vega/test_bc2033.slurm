#!/bin/bash
#SBATCH --job-name=vega_test_bc2033_dual
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_test_bc2033_dual_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega bc2033 DUAL-CALLING smoke test — single script, both tracks
#
# Goal: from ONE raw HiFi BAM, produce TWO motifs.csv files via two
# independent calling pipelines. This is the gate before rolling the
# dual pipeline out to all Vega / Sequel / Strepto samples.
#
# Input:
#   HiFi BAM  = ${VEGA}/m21026_260313_002345.hifi_reads.bc2033.bam
#               (carries per-strand fi/fp/ri/rp kinetics)
#   Reference = ${VEGA}/assembly/bc2033/bc2033_assembly.fasta
#               (hifiasm draft produced by vega/00_assembly.slurm)
#
# ─── Track A — ipdSummary SP3-C3 → pbmotifmaker ───────────────
#   1. ccs-kinetics-bystrandify         fi/fp → ip/pw per strand
#   2. pbmm2 align --preset CCS --sort
#   3. samtools index + pbindex
#   4. ipdSummary SP3-C3 --identify m6A,m4C --methylFraction  → GFF
#   5. pbmotifmaker find                                      → motifs_ipdsummary.csv
#
# ─── Track B — jasmine 5mC → modkit find-motifs ───────────────
#   1. jasmine predict --keep-kinetics   on ORIGINAL fi/fp/ri/rp BAM
#   2. pbmm2 align --preset CCS --sort
#   3. samtools index
#   4. modkit pileup --filter-threshold 0.5                   → bedMethyl
#   5. modkit find-motifs                                     → TSV
#   6. python: modkit TSV → combined-parser CSV              → motifs_jasmine.csv
#
# Output (${VEGA}/test_dual/bc2033/):
#   bc2033_bystrandify.bam
#   bc2033_A_aligned.bam{,.bai,.pbi}
#   bc2033_ipdSummary.{gff,csv}
#   bc2033_motifs_ipdsummary.csv       ← Track A final motif table
#   bc2033_jasmine.bam
#   bc2033_B_aligned.bam{,.bai}
#   bc2033_pileup.bed
#   bc2033_find_motifs.tsv
#   bc2033_motifs_jasmine.csv          ← Track B final motif table
#
# Usage:
#   sbatch slurm_kinsim/vega/test_bc2033.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true
set -euo pipefail

# ---- config ----
VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

BC=2033
SAMPLE="bc${BC}"
RAW_BAM="${VEGA}/m21026_260313_002345.hifi_reads.${SAMPLE}.bam"
ASM_FASTA="${VEGA}/assembly/${SAMPLE}/${SAMPLE}_assembly.fasta"

OUTDIR="${VEGA}/test_dual/${SAMPLE}"
mkdir -p "$OUTDIR"

# Track A outputs
BYSTRAND="${OUTDIR}/${SAMPLE}_bystrandify.bam"
A_ALIGNED="${OUTDIR}/${SAMPLE}_A_aligned.bam"
IPD_GFF="${OUTDIR}/${SAMPLE}_ipdSummary.gff"
IPD_CSV="${OUTDIR}/${SAMPLE}_ipdSummary.csv"
MOTIFS_A="${OUTDIR}/${SAMPLE}_motifs_ipdsummary.csv"

# Track B outputs
JASMINE_BAM="${OUTDIR}/${SAMPLE}_jasmine.bam"
B_ALIGNED="${OUTDIR}/${SAMPLE}_B_aligned.bam"
PILEUP="${OUTDIR}/${SAMPLE}_pileup.bed"
FIND_TSV="${OUTDIR}/${SAMPLE}_find_motifs.tsv"
MOTIFS_B="${OUTDIR}/${SAMPLE}_motifs_jasmine.csv"

echo "========================================================"
echo "  Vega dual-calling test — $SAMPLE"
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
# Track A — ipdSummary SP3-C3 → pbmotifmaker
# ============================================================
echo ""
echo "############  Track A: ipdSummary → pbmotifmaker  ############"

# --- A1 bystrandify ---
if [ -s "$BYSTRAND" ]; then
    echo "=== A1 SKIP — bystrandify BAM exists ==="
else
    echo "=== A1 ccs-kinetics-bystrandify ==="
    START=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        ccs-kinetics-bystrandify "$RAW_BAM" "$BYSTRAND"
    echo "    $(( $(date +%s) - START ))s"
fi
[ -s "$BYSTRAND" ] || { echo "FAILED: bystrandify produced no output"; exit 1; }

# --- A2 pbmm2 align ---
if [ -s "$A_ALIGNED" ]; then
    echo "=== A2 SKIP — aligned BAM exists ==="
else
    echo "=== A2 pbmm2 align (bystrandify → assembly) ==="
    START=$(date +%s)
    pbmm2 align "$ASM_FASTA" "$BYSTRAND" "$A_ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
    echo "    $(( $(date +%s) - START ))s"
fi

# --- A3 samtools index + pbindex ---
[ -f "${A_ALIGNED}.bai" ] || samtools index "$A_ALIGNED"
[ -f "${A_ALIGNED}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$A_ALIGNED"

# --- A4 ipdSummary SP3-C3 ---
if [ -s "$IPD_GFF" ]; then
    echo "=== A4 SKIP — ipdSummary GFF exists ($(grep -c '^[^#]' "$IPD_GFF" || true) records) ==="
else
    echo "=== A4 ipdSummary SP3-C3 ==="
    START=$(date +%s)
    ipdSummary "$A_ALIGNED" \
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

# --- A5 pbmotifmaker ---
if [ -s "$MOTIFS_A" ]; then
    echo "=== A5 SKIP — Track A motifs.csv exists ==="
else
    echo "=== A5 pbmotifmaker find ==="
    START=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        pbmotifmaker find "$ASM_FASTA" "$IPD_GFF" "$MOTIFS_A"
    echo "    $(( $(date +%s) - START ))s"
fi

# ============================================================
# Track B — jasmine 5mC → modkit find-motifs
# ============================================================
echo ""
echo "############  Track B: jasmine 5mC → modkit  ############"

# --- B1 jasmine predict (on the ORIGINAL raw BAM, not bystrandified) ---
if [ -s "$JASMINE_BAM" ]; then
    echo "=== B1 SKIP — jasmine BAM exists ==="
else
    echo "=== B1 jasmine predict (keeps kinetics) ==="
    START=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        jasmine "$RAW_BAM" "$JASMINE_BAM" --keep-kinetics
    echo "    $(( $(date +%s) - START ))s"
fi
[ -s "$JASMINE_BAM" ] || { echo "FAILED: jasmine produced no output"; exit 1; }

# --- B2 pbmm2 align ---
if [ -s "$B_ALIGNED" ]; then
    echo "=== B2 SKIP — jasmine aligned BAM exists ==="
else
    echo "=== B2 pbmm2 align (jasmine → assembly) ==="
    START=$(date +%s)
    pbmm2 align "$ASM_FASTA" "$JASMINE_BAM" "$B_ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
    echo "    $(( $(date +%s) - START ))s"
fi

# --- B3 samtools index ---
[ -f "${B_ALIGNED}.bai" ] || samtools index "$B_ALIGNED"

# --- B4 modkit pileup ---
if [ -s "$PILEUP" ]; then
    echo "=== B4 SKIP — pileup BED exists ($(wc -l < "$PILEUP") sites) ==="
else
    echo "=== B4 modkit pileup ==="
    START=$(date +%s)
    modkit pileup "$B_ALIGNED" "$PILEUP" \
        --ref "$ASM_FASTA" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        --filter-threshold 0.5 \
        --log-filepath "${OUTDIR}/${SAMPLE}_modkit.log"
    echo "    $(( $(date +%s) - START ))s  ($(wc -l < "$PILEUP") sites)"
fi

# --- B5 modkit find-motifs ---
if [ -s "$FIND_TSV" ]; then
    echo "=== B5 SKIP — find-motifs TSV exists ==="
else
    echo "=== B5 modkit find-motifs ==="
    START=$(date +%s)
    modkit find-motifs \
        --in-bedmethyl "$PILEUP" \
        --ref "$ASM_FASTA" \
        --out-table "$FIND_TSV" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        || { echo "modkit find-motifs failed — writing empty TSV"; : > "$FIND_TSV"; }
    echo "    $(( $(date +%s) - START ))s"
fi

# --- B6 TSV → combined-parser CSV ---
echo "=== B6 emit jasmine motifs.csv (combined-parser format) ==="
python - <<PY
import csv
from pathlib import Path

tsv = Path('${FIND_TSV}')
out = Path('${MOTIFS_B}')

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
# Summary
# ============================================================
echo ""
echo "========================================================"
echo "  DUAL-CALLING TEST COMPLETE — $SAMPLE"
echo "========================================================"
echo "  Elapsed total: $(( $(date +%s) - T0 ))s"
echo ""
echo "  Track A (ipdSummary+pbmotifmaker):"
echo "    $MOTIFS_A"
if [ -s "$MOTIFS_A" ]; then
    echo "    rows: $(tail -n +2 "$MOTIFS_A" | wc -l)"
    echo "    --- head ---"
    head -5 "$MOTIFS_A" | sed 's/^/      /'
fi
echo ""
echo "  Track B (jasmine+modkit):"
echo "    $MOTIFS_B"
if [ -s "$MOTIFS_B" ]; then
    echo "    rows: $(tail -n +2 "$MOTIFS_B" | wc -l)"
    echo "    --- head ---"
    head -5 "$MOTIFS_B" | sed 's/^/      /'
fi
echo "========================================================"
