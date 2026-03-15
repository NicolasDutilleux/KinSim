#!/bin/bash
# Validate motif CSV: check that every offset points to the right base
# Convention: 1-based offsets (KinSim standard)
#
# Usage: bash slurm_kinsim/validate_motifs.sh <motifs.csv>

FILE="$1"
if [ -z "$FILE" ]; then
    echo "Usage: bash validate_motifs.sh <motifs.csv>"
    exit 1
fi

python3 -c "
import csv, sys

valid_base = {'m6A': 'A', 'm4C': 'C', 'm5C': 'C'}
norm = {'6mA':'m6A','5mC':'m5C','4mC':'m4C','m6A':'m6A','m5C':'m5C','m4C':'m4C'}
errors = 0
ok = 0

with open('$FILE') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames

    if 'motifString' in cols:
        motif_col, offset_col, mod_col = 'motifString', 'centerPos', 'modificationType'
    elif 'motif' in cols:
        motif_col, offset_col, mod_col = 'motif', 'offset', 'mod_type'
    else:
        print(f'ERROR: unrecognized CSV format: {cols}')
        sys.exit(1)

    for i, row in enumerate(reader, 2):
        motif = row[motif_col].strip().upper()
        mod   = norm.get(row[mod_col].strip(), row[mod_col].strip())
        if mod not in valid_base:
            continue

        try:
            offset_1b = int(row[offset_col])
        except ValueError:
            continue

        idx = offset_1b - 1  # 1-based → 0-based for Python indexing
        if idx < 0 or idx >= len(motif):
            print(f'  LINE {i}: INVALID offset={offset_1b} for {motif} (len={len(motif)})')
            errors += 1
            continue

        base = motif[idx]
        expected = valid_base[mod]
        if base != expected:
            print(f'  LINE {i}: WRONG  {mod} {motif} offset={offset_1b} -> base={base} (expected {expected})')
            errors += 1
        else:
            ok += 1

print(f'Checked: {ok + errors} motifs, {ok} OK, {errors} ERRORS')
if errors > 0:
    print('FIX NEEDED: offsets do not point to the expected base (1-based)')
    sys.exit(1)
else:
    print('All offsets correct (1-based)')
"
