#!/bin/bash
# Validate motif CSV: check that every offset points to the right base
# Convention: 1-based offsets (KinSim standard)
#
# Handles all known formats:
#   PacBio CSV:   motifString,centerPos,modificationType,...
#   Combined CSV: mod_type,motif,offset,...
#   Modkit CSV:   mod_code,motif,offset,...
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
norm = {
    '6mA':'m6A', '5mC':'m5C', '4mC':'m4C',
    'm6A':'m6A', 'm5C':'m5C', 'm4C':'m4C',
    'a':'m6A', 'h':'m5C', 'm':'m5C',   # modkit single-char codes
    'A+a':'m6A',                        # fibertools code
}
errors = 0
ok = 0
skipped = 0

with open('$FILE') as f:
    reader = csv.DictReader(f)
    cols = set(reader.fieldnames or [])

    # Detect format
    if 'motifString' in cols and 'centerPos' in cols:
        motif_col, offset_col = 'motifString', 'centerPos'
        mod_col = 'modificationType' if 'modificationType' in cols else None
    elif 'motif' in cols and 'offset' in cols:
        if 'mod_type' in cols:
            motif_col, offset_col, mod_col = 'motif', 'offset', 'mod_type'
        elif 'mod_code' in cols:
            motif_col, offset_col, mod_col = 'motif', 'offset', 'mod_code'
        else:
            motif_col, offset_col, mod_col = 'motif', 'offset', None
    else:
        print(f'SKIP: unrecognized CSV format in \"{sys.argv[0]}\"')
        sys.exit(0)

    for i, row in enumerate(reader, 2):
        motif = row[motif_col].strip().upper()
        if not motif:
            continue

        raw_mod = row.get(mod_col, '').strip() if mod_col else ''
        mod = norm.get(raw_mod, raw_mod)
        if mod not in valid_base:
            skipped += 1
            continue

        try:
            offset_1b = int(row[offset_col])
        except ValueError:
            continue

        idx = offset_1b - 1  # 1-based -> 0-based for Python indexing
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

total = ok + errors
msg = f'Checked: {total} motifs, {ok} OK, {errors} ERRORS'
if skipped:
    msg += f' ({skipped} skipped: unknown mod type)'
print(msg)
if errors > 0:
    print('FIX NEEDED: offsets do not point to the expected base (1-based)')
    sys.exit(1)
elif total > 0:
    print('All offsets correct (1-based)')
else:
    print('No checkable motifs found')
"
