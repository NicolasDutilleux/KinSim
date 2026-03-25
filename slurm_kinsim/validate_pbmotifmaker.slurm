#!/bin/bash
#SBATCH --job-name=kinsim_validate_pbmotifmaker
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/validate_pbmotifmaker_%J.log

# Run pbmotifmaker on ipdSummary GFF to recover motifs from simulated data.
# This is the ultimate validation: do the same motifs come back?

VALDIR=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_validation
REF=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/final_assembly.fasta
GFF=${VALDIR}/bc2036_generated_v4_ipdSummary.gff
OUT=${VALDIR}/bc2036_generated_v4_motifs.csv

echo "=== pbmotifmaker validation ==="
echo "GFF: $GFF"
echo "REF: $REF"
echo "OUT: $OUT"
date

# Check inputs exist
if [ ! -f "$GFF" ]; then
    echo "ERROR: GFF file not found: $GFF"
    echo "Run validate_ipdsummary.slurm first."
    exit 1
fi

if [ ! -s "$GFF" ]; then
    echo "ERROR: GFF file is empty: $GFF"
    echo "ipdSummary may have failed — check validate_ipdsummary log."
    exit 1
fi

# Run pbmotifmaker with 'find' subcommand
apptainer exec --bind /data \
    /containers/apptainer/pacbio-smrt-tools-25.1.sif \
    pbmotifmaker find "$REF" "$GFF" "$OUT"

RC=$?
echo "pbmotifmaker exit code: $RC"

if [ $RC -eq 0 ] && [ -f "$OUT" ]; then
    echo ""
    echo "=== Recovered motifs ==="
    cat "$OUT"
    echo ""
    echo "=== Comparison with real motifs ==="
    REAL=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/motifs.csv
    if [ -f "$REAL" ]; then
        echo "Real motifs (from SMRT Link):"
        cat "$REAL"
    fi
fi

date
echo "=== Done ==="
