#!/bin/bash
#SBATCH --job-name=msa_ippw
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/00b_add_ippw_%j.log
#SBATCH --error=/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/logs/00b_add_ippw_%j.err

# STEP 00b: Prepare aligned split BAMs for ipdSummary
#
# Three operations per species:
#   1. Add ip/pw tags from fi/fp/ri/rp (ipdSummary requires ip/pw)
#      - Forward reads: ip = fi, pw = fp
#      - Reverse reads: ip = rev(ri), pw = rev(rp)
#   2. Reheader BAM to only list the single species contig in @SQ
#      (pbcore requires FASTA contigs to be superset of BAM @SQ)
#   3. Create per-species single-contig reference FASTA + .fai
#      (pbcore rejects competitive ref .fai as "malformatted")
#
# Run AFTER step 00 (align+split) and BEFORE step 01 (ipdSummary).

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kinsim_env

OUTDIR="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/species_aligned"
BIGREF="/data/projects/p774_MARSD/NDutilleux/training/PB_MOCK/Genomes/MSA1003_competitive_ref.fasta"

echo "========================================"
echo "=== STEP 00b: Prepare BAMs for ipdSummary ==="
echo "=== $(date) ==="
echo "=== Host: $(hostname) ==="
echo "========================================"
echo ""

# Ensure competitive ref is indexed
if [ ! -f "${BIGREF}.fai" ]; then
    echo "Indexing competitive reference..."
    samtools faidx "$BIGREF"
fi

TOTAL_OK=0
TOTAL_FAIL=0
TOTAL_SKIP=0

for SPECIES_DIR in "${OUTDIR}"/*/; do
    [ -d "$SPECIES_DIR" ] || continue
    CONTIG=$(basename "$SPECIES_DIR")

    BAM="${SPECIES_DIR}/${CONTIG}.bam"
    IPD_BAM="${SPECIES_DIR}/${CONTIG}_ipd.bam"
    SINGLE_REF="${SPECIES_DIR}/${CONTIG}.fasta"

    if [ ! -f "$BAM" ]; then
        echo "SKIP $CONTIG — no BAM found"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi

    # Check if output already exists (skip if already processed)
    if [ -f "$IPD_BAM" ] && [ -f "$SINGLE_REF" ]; then
        echo "SKIP $CONTIG — already processed (_ipd.bam + reference exist)"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi

    # Check if fi/fp/ri/rp tags exist
    HAS_FI=$(samtools view "$BAM" | head -3 | tr '\t' '\n' | grep -c '^fi:' || true)
    if [ "$HAS_FI" -eq 0 ]; then
        echo "SKIP $CONTIG — no fi/fp/ri/rp tags in BAM"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi

    READS=$(samtools view -c "$BAM")
    echo "--- $CONTIG ($READS reads) ---"

    START=$(date +%s)

    # --- (A) Create single-contig reference ---
    echo "  Creating per-species reference..."
    samtools faidx "$BIGREF" "$CONTIG" | sed "s/>.*/>${CONTIG}/" > "$SINGLE_REF"
    samtools faidx "$SINGLE_REF"

    # --- (B) Add ip/pw tags + reheader to single contig ---
    # Writes to a SEPARATE _ipd.bam — original BAM is preserved for modkit (MM/ML)
    echo "  Adding ip/pw tags + reheadering -> _ipd.bam..."

    python -c "
import pysam
import array
import numpy as np

bam_in = pysam.AlignmentFile('${BAM}')

# Build new header with only this contig's @SQ line
old_header = bam_in.header.to_dict()
old_header['SQ'] = [sq for sq in old_header['SQ'] if sq['SN'] == '${CONTIG}']

bam_out = pysam.AlignmentFile('${IPD_BAM}', 'wb', header=old_header)

n_converted = 0
n_skipped = 0
for read in bam_in:
    if read.is_unmapped or read.query_length == 0:
        bam_out.write(read)
        n_skipped += 1
        continue

    try:
        fi = np.array(read.get_tag('fi'), dtype=np.uint8)
        fp = np.array(read.get_tag('fp'), dtype=np.uint8)
    except KeyError:
        bam_out.write(read)
        n_skipped += 1
        continue

    if len(fi) != read.query_length:
        bam_out.write(read)
        n_skipped += 1
        continue

    if read.is_reverse:
        # Real PacBio BAMs have ri/rp; KinSim BAMs only have fi/fp — fall back
        try:
            ri = np.array(read.get_tag('ri'), dtype=np.uint8)
            rp = np.array(read.get_tag('rp'), dtype=np.uint8)
            if len(ri) == read.query_length:
                ip_vals = ri[::-1].copy()
                pw_vals = rp[::-1].copy()
            else:
                # ri/rp present but wrong length (e.g. empty tag) — use fi/fp
                ip_vals = fi.copy()
                pw_vals = fp.copy()
        except KeyError:
            ip_vals = fi.copy()
            pw_vals = fp.copy()
    else:
        ip_vals = fi.copy()
        pw_vals = fp.copy()

    # Final safety check — should never fire, but guards against edge cases
    if len(ip_vals) != read.query_length:
        bam_out.write(read)
        n_skipped += 1
        continue

    read.set_tag('ip', array.array('B', ip_vals.tolist()))
    read.set_tag('pw', array.array('B', pw_vals.tolist()))
    bam_out.write(read)
    n_converted += 1

bam_in.close()
bam_out.close()
print(f'  Converted: {n_converted} reads, skipped: {n_skipped}')
"

    RC=$?
    END=$(date +%s)
    ELAPSED=$((END - START))

    if [ $RC -ne 0 ]; then
        echo "  FAILED (exit $RC) after ${ELAPSED}s"
        rm -f "$IPD_BAM"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        continue
    fi

    # Index the new _ipd.bam (original BAM untouched)
    samtools index "$IPD_BAM"

    echo "  OK (${ELAPSED}s)"
    TOTAL_OK=$((TOTAL_OK + 1))
    echo ""
done

echo "========================================"
echo "=== STEP 00b COMPLETE ==="
echo "=== $(date) ==="
echo "=== OK: $TOTAL_OK  FAIL: $TOTAL_FAIL  SKIP: $TOTAL_SKIP ==="
echo "========================================"

# Quick verification on first species
FIRST_DIR=$(ls -d "${OUTDIR}"/*/ | head -1)
if [ -n "$FIRST_DIR" ]; then
    FC=$(basename "$FIRST_DIR")
    FB="${FIRST_DIR}/${FC}_ipd.bam"
    echo ""
    echo "=== Verification: $FC ==="
    echo "Original BAM: ${FIRST_DIR}/${FC}.bam (preserved for modkit)"
    echo "ipdSummary BAM: $FB"
    echo "SQ lines in _ipd.bam: $(samtools view -H "$FB" | grep -c '^@SQ')"
    echo "ip/pw tags:"
    samtools view "$FB" | head -1 | tr '\t' '\n' | grep -E '^(ip|pw):' | cut -c1-20
    echo "Per-species ref: ${FIRST_DIR}/${FC}.fasta"
    [ -f "${FIRST_DIR}/${FC}.fasta.fai" ] && echo "  .fai OK" || echo "  .fai MISSING"
fi
echo ""
echo "Done."
echo "  Original BAMs preserved: <contig>.bam       (for modkit — has MM/ML tags)"
echo "  ipdSummary BAMs created: <contig>_ipd.bam   (has ip/pw + single-contig header)"
echo "  Per-species references:  <contig>.fasta      (for ipdSummary)"
