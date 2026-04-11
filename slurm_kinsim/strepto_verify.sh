#!/bin/bash
# ============================================================
# Streptomyces GFF pipeline — Pre-flight verification
#
# Run with: srun --partition=pibu_el8 --account=p774 --mem=4G --time=00:10:00 bash slurm_kinsim/strepto_verify.sh
# ============================================================

set -uo pipefail

BASEDIR=/data/projects/p774_MARSD/NDutilleux/training
MANIFEST=${BASEDIR}/manifest_combined_train.csv
STREPTO=${BASEDIR}/Strepto
SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

echo "========================================================"
echo "  Streptomyces GFF Pipeline — Pre-flight Check"
echo "  $(date '+%Y-%m-%d %H:%M:%S') on $(hostname)"
echo "========================================================"
echo ""

ERRORS=0
WARNINGS=0

ok()   { echo "  [OK]      $1"; }
warn() { echo "  [WARN]    $1"; WARNINGS=$((WARNINGS+1)); }
fail() { echo "  [FAIL]    $1"; ERRORS=$((ERRORS+1)); }

# ---- 1. Conda + kinsim ----
echo "=== 1. Environment ==="
source ~/.bashrc
conda activate kinsim_env 2>/dev/null

if command -v kinsim &>/dev/null; then
    ok "kinsim $(kinsim --version 2>&1)"
else
    fail "kinsim not found — pip install -e /path/to/KinSim"
fi

if command -v kinsim-prep &>/dev/null; then
    ok "kinsim-prep available"
else
    fail "kinsim-prep not found"
fi

if command -v pbmm2 &>/dev/null; then
    ok "pbmm2 $(pbmm2 --version 2>&1 | head -1)"
else
    fail "pbmm2 not found in kinsim_env"
fi

if command -v samtools &>/dev/null; then
    ok "samtools $(samtools --version | head -1)"
else
    fail "samtools not found"
fi

python -c "import pysam; print(pysam.__version__)" 2>/dev/null \
    && ok "pysam $(python -c 'import pysam; print(pysam.__version__)')" \
    || fail "pysam not importable"

echo ""

# ---- 2. Apptainer + bystrandify ----
echo "=== 2. Apptainer (bystrandify + pbindex) ==="

if command -v apptainer &>/dev/null; then
    ok "apptainer available"
else
    fail "apptainer not found"
fi

if [ -f "$SMRT_SIF" ]; then
    ok "SIF exists: $SMRT_SIF"

    if apptainer exec --bind /data "$SMRT_SIF" ccs-kinetics-bystrandify --version 2>/dev/null; then
        ok "ccs-kinetics-bystrandify accessible"
    else
        # Some versions don't have --version, try --help
        if apptainer exec --bind /data "$SMRT_SIF" ccs-kinetics-bystrandify --help &>/dev/null; then
            ok "ccs-kinetics-bystrandify accessible (no --version)"
        else
            fail "ccs-kinetics-bystrandify not working in SIF"
        fi
    fi

    if apptainer exec --bind /data "$SMRT_SIF" pbindex --version 2>/dev/null; then
        ok "pbindex accessible"
    else
        warn "pbindex --version failed (may still work)"
    fi
else
    fail "SIF not found: $SMRT_SIF"
fi

echo ""

# ---- 3. SMRT-Link module (ipdSummary) ----
echo "=== 3. SMRT-Link / ipdSummary ==="

source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true

if command -v ipdSummary &>/dev/null; then
    ok "ipdSummary available (module loaded)"
else
    fail "ipdSummary not found — module load SMRT-Link/12.0.0.177059-cli-tools-only failed?"
fi

if [ -f "$SP3_MODEL" ]; then
    ok "SP3-C3 model: $(basename $SP3_MODEL)"
else
    fail "SP3-C3 model not found: $SP3_MODEL"
fi

echo ""

# ---- 4. Manifest ----
echo "=== 4. Training manifest ==="

if [ -f "$MANIFEST" ]; then
    N=$(kinsim-prep manifest count "$MANIFEST" 2>/dev/null)
    ok "Manifest exists: $N species"
    echo ""
    echo "  Manifest preview:"
    head -3 "$MANIFEST" | sed 's/^/    /'
    echo "    ..."
else
    fail "Manifest not found: $MANIFEST"
    N=0
fi

echo ""

# ---- 5. Per-species files ----
echo "=== 5. Per-species BAMs + references ==="

if [ "$N" -gt 0 ]; then
    MISSING_BAM=0
    MISSING_REF=0
    MISSING_TAGS=0
    CHECKED=0

    python -c "
from kinsim.utils.config import load_manifest
entries = load_manifest('${MANIFEST}')
for e in entries:
    print(f'{e.sample_id}|{e.bam_path}')
" 2>/dev/null | while IFS='|' read SAMPLE BAM; do
        CHECKED=$((CHECKED+1))
        REF="${STREPTO}/${SAMPLE}/final_assembly.fasta"

        # BAM exists?
        if [ ! -f "$BAM" ]; then
            fail "$SAMPLE: BAM missing: $BAM"
            MISSING_BAM=$((MISSING_BAM+1))
            continue
        fi

        # Reference exists?
        if [ ! -f "$REF" ]; then
            fail "$SAMPLE: Reference missing: $REF"
            MISSING_REF=$((MISSING_REF+1))
            continue
        fi

        # Check fi/fp tags on first read (only for first 3 species as spot check)
        if [ "$CHECKED" -le 3 ]; then
            HAS_FI=$(samtools view "$BAM" 2>/dev/null | head -1 | tr '\t' '\n' | grep -c '^fi:' || true)
            if [ "$HAS_FI" -eq 0 ]; then
                warn "$SAMPLE: No fi tag in first read"
                MISSING_TAGS=$((MISSING_TAGS+1))
            else
                ok "$SAMPLE: BAM + REF + fi/fp tags"
            fi
        fi
    done

    # Summary counts
    echo ""
    python -c "
from kinsim.utils.config import load_manifest
from pathlib import Path
entries = load_manifest('${MANIFEST}')
strepto = '${STREPTO}'
ok_count = 0
bam_missing = []
ref_missing = []
for e in entries:
    ref = Path(strepto) / e.sample_id / 'final_assembly.fasta'
    bam_ok = Path(e.bam_path).exists()
    ref_ok = ref.exists()
    if bam_ok and ref_ok:
        ok_count += 1
    elif not bam_ok:
        bam_missing.append(e.sample_id)
    elif not ref_ok:
        ref_missing.append(e.sample_id)
print(f'  Summary: {ok_count}/{len(entries)} species have both BAM + reference')
if bam_missing:
    print(f'  BAM missing ({len(bam_missing)}): {bam_missing[:5]}...' if len(bam_missing)>5 else f'  BAM missing: {bam_missing}')
if ref_missing:
    print(f'  REF missing ({len(ref_missing)}): {ref_missing[:5]}...' if len(ref_missing)>5 else f'  REF missing: {ref_missing}')
" 2>/dev/null
fi

echo ""

# ---- 6. Output directories ----
echo "=== 6. Output paths ==="

IPD_DIR=${BASEDIR}/Strepto/gff_pipeline
SHARDS=${BASEDIR}/shards_gff_train
CKPT=${BASEDIR}/checkpoints_gff
LOGS=/data/projects/p774_MARSD/NDutilleux/logs

for DIR in "$IPD_DIR" "$SHARDS" "$CKPT" "$LOGS"; do
    if [ -d "$DIR" ]; then
        ok "Exists: $DIR"
    else
        warn "Will be created: $DIR"
    fi
done

# Check disk space
echo ""
echo "  Disk space on $BASEDIR:"
df -h "$BASEDIR" 2>/dev/null | tail -1 | awk '{print "    Total:", $2, " Used:", $3, " Avail:", $4, " Use%:", $5}'

echo ""

# ---- 7. Quick smoke test: bystrandify on 1 read ----
echo "=== 7. Smoke test (optional — first species) ==="

FIRST_SAMPLE=$(python -c "
from kinsim.utils.config import load_manifest
e = load_manifest('${MANIFEST}')[0]
print(e.sample_id)
" 2>/dev/null)

FIRST_BAM=$(python -c "
from kinsim.utils.config import load_manifest
e = load_manifest('${MANIFEST}')[0]
print(e.bam_path)
" 2>/dev/null)

FIRST_REF="${STREPTO}/${FIRST_SAMPLE}/final_assembly.fasta"

echo "  Sample: $FIRST_SAMPLE"
echo "  BAM:    $FIRST_BAM"
echo "  REF:    $FIRST_REF"

if [ -f "$FIRST_BAM" ] && [ -f "$FIRST_REF" ]; then
    # Count reads
    READ_COUNT=$(samtools view -c "$FIRST_BAM" 2>/dev/null || echo "?")
    echo "  Reads:  $READ_COUNT"

    # Check tags
    echo "  Tags in first read:"
    samtools view "$FIRST_BAM" 2>/dev/null | head -1 | tr '\t' '\n' | grep -E '^(fi|fp|ri|rp):' | cut -c1-30 | sed 's/^/    /'

    # Reference contigs
    echo "  Reference contigs:"
    grep "^>" "$FIRST_REF" 2>/dev/null | head -5 | sed 's/^/    /'
    NCONTIGS=$(grep -c "^>" "$FIRST_REF" 2>/dev/null || echo "?")
    echo "    (total: $NCONTIGS contigs)"
else
    warn "Cannot run smoke test — files missing"
fi

echo ""

# ---- Final verdict ----
echo "========================================================"
if [ $ERRORS -eq 0 ]; then
    echo "  READY — $ERRORS errors, $WARNINGS warnings"
    echo "  Run: bash slurm_kinsim/run_strepto_gff_pipeline.sh"
else
    echo "  NOT READY — $ERRORS errors, $WARNINGS warnings"
    echo "  Fix the [FAIL] items above before running the pipeline."
fi
echo "========================================================"
