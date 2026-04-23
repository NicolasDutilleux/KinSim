#!/bin/bash
#SBATCH --job-name=call_merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=00:15:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# callers/merge_motifs — union de motifs.csv filtrés par threshold
#
# Usage:
#   sbatch -J call_merge_<sample> slurm_kinsim/callers/merge_motifs.slurm \
#          <out_merged_csv> <threshold> <input_csv1> [<input_csv2> ...]
#
# Policy: no precedence rules. Concat all inputs, filter frac >= threshold,
# dedup by (motif, offset) keeping the highest-fraction entry.
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

OUT_CSV=${1:?"Usage: sbatch $0 <out_csv> <threshold> <input_csv1> [<input_csv2> ...]"}
THRESH=${2:?"threshold missing (0-1, e.g. 0.7)"}
shift 2

[ $# -ge 1 ] || { echo "ERROR: need at least one input CSV"; exit 1; }

mkdir -p "$(dirname "$OUT_CSV")"

echo "=== callers/merge_motifs — $(basename "$OUT_CSV") ==="
echo "Threshold: $THRESH"
echo "Inputs:"
for p in "$@"; do echo "  $p"; done

python3 - "$OUT_CSV" "$THRESH" "$@" <<'PY'
"""
Merge one or more motifs.csv files into a single combined-parser CSV.

Accepts both formats:
  - PacBio motifs.csv (motifString, centerPos, modificationType, fraction, nDetected, nGenome, ...)
  - combined-parser CSV (mod_type, motif, offset, frac_mod, n_sites, source)

No precedence rules. Just:
  1. Parse each input, normalize to a common row shape
  2. Filter fraction >= threshold
  3. Dedup by (motif, offset), keep highest fraction
  4. Write combined-parser CSV
"""
import csv, sys
from pathlib import Path

out_csv  = Path(sys.argv[1])
thresh   = float(sys.argv[2])
in_paths = [Path(p) for p in sys.argv[3:]]

def _f(d, *names, default=None):
    for n in names:
        if n in d and d[n] not in ('', None):
            return d[n]
    return default

def parse_row(r, source_hint):
    """Return (mod_type, motif, offset, frac, n_sites, source) or None."""
    motif = _f(r, 'motifString', 'motif', 'sequence')
    if not motif:
        return None
    offset = _f(r, 'centerPos', 'mod_offset', 'offset', 'position', 'mod_loc')
    if offset is None:
        return None
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        return None
    frac = _f(r, 'fraction', 'frac_mod', 'methylated_fraction', default=0.0)
    try:
        frac = float(frac)
    except (TypeError, ValueError):
        frac = 0.0
    mod = _f(r, 'modificationType', 'mod_type', 'mod_code', default='')
    # Normalize mod_type labels
    if mod in ('m', '5mC'): mod = 'm5C'
    elif mod in ('a', '6mA'): mod = 'm6A'
    elif mod == '4mC': mod = 'm4C'
    # n_sites: try several sources
    ns = _f(r, 'n_sites', 'nDetected', 'num_sites', 'total_sites', default=0)
    try:
        ns = int(float(ns))
    except (TypeError, ValueError):
        ns = 0
    # source: use explicit column or hint
    source = _f(r, 'source', default=source_hint)
    return (mod, motif, offset, frac, ns, source)

# Collect rows
all_rows = []
for p in in_paths:
    if not p.exists() or p.stat().st_size == 0:
        print(f"  skip (missing/empty): {p}")
        continue
    # Guess source from filename
    name = p.name.lower()
    if 'jasmine' in name:       source_hint = 'jasmine+modkit'
    elif 'kinmethyl' in name:   source_hint = 'kinmethyl'
    elif 'ipdsum' in name or 'ipd' in name: source_hint = 'ipdsummary'
    else:                       source_hint = p.stem
    with p.open() as f:
        rdr = csv.DictReader(f)
        n_parsed = n_kept = 0
        for r in rdr:
            row = parse_row(r, source_hint)
            if row is None:
                continue
            n_parsed += 1
            if row[3] >= thresh:
                all_rows.append(row)
                n_kept += 1
    print(f"  {p.name}: {n_parsed} parsed, {n_kept} passed threshold ({thresh:.2f})")

# Dedup by (motif, offset) keeping highest fraction
best = {}
for row in all_rows:
    key = (row[1], row[2])
    if key not in best or row[3] > best[key][3]:
        best[key] = row

# Sort by fraction desc for readability
final = sorted(best.values(), key=lambda x: -x[3])

with out_csv.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mod_type', 'motif', 'offset', 'frac_mod', 'n_sites', 'source'])
    for row in final:
        w.writerow(row)

print(f"\nWrote {out_csv}  ({len(final)} motifs after dedup from {len(all_rows)} passed rows)")
if final:
    print("Top 10:")
    for r in final[:10]:
        print(f"  {r[0]:<6} {r[1]:<18} offset={r[2]:<3} frac={r[3]:.3f}  n={r[4]:>6}  {r[5]}")
PY
