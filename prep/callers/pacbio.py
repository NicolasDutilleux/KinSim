"""PacBio motifs.csv output parser.

Handles PacBio SMRT Link motifs.csv with variable columns.
Required columns: motifString, centerPos.
Optional columns: modificationType, fraction, nDetected, ...

When optional columns are missing:
  - modificationType absent -> infer from the base at centerPos (A->m6A, C->m4C)
  - fraction absent         -> default 1.0
  - nDetected absent        -> default 0 (bypasses min_detected filter)
"""

from __future__ import annotations

import csv
import logging

from kinsim.utils.encoding import METH_IDS
from kinsim.utils.motifs import reverse_complement
from .base import BaseOutputParser
from .registry import register

log = logging.getLogger(__name__)

# Resolve ambiguous "modified_base" by the base at centerPos (forward strand)
_BASE_TO_METH = {'A': 'm6A', 'C': 'm4C'}

# When the base at centerPos is on the complement strand (G or T),
# the actual modified base is the complement: C→m4C, A→m6A
_COMP_BASE_TO_METH = {'G': 'm4C', 'T': 'm6A'}


@register
class PacBioParser(BaseOutputParser):
    """Parser for PacBio SMRT Link motifs.csv output."""

    name = "pacbio"
    supported_mods = ["m6A", "m4C", "m5C"]

    def parse(
        self,
        filepath: str,
        min_fraction: float = 0.40,
        min_detected: int = 20,
    ) -> str:
        entries: list[str] = []

        with open(filepath) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                log.warning("PacBio CSV: empty or headerless file '%s'", filepath)
                return ""

            has_fraction = 'fraction' in reader.fieldnames
            has_ndetected = 'nDetected' in reader.fieldnames
            has_mod_type = 'modificationType' in reader.fieldnames

            for lineno, row in enumerate(reader, 2):
                # -- motifString and centerPos are required --
                motif_seq = row.get('motifString', '').strip()
                center_str = row.get('centerPos', '').strip()
                if not motif_seq or not center_str:
                    log.warning("PacBio CSV line %d: missing motifString or "
                                "centerPos -- skipped", lineno)
                    continue

                try:
                    center_pos = int(center_str)
                except ValueError:
                    log.warning("PacBio CSV line %d: invalid centerPos '%s' "
                                "-- skipped", lineno, center_str)
                    continue

                # -- fraction (optional, default 1.0) --
                if has_fraction:
                    frac_str = row.get('fraction', '').strip()
                    try:
                        fraction = float(frac_str) if frac_str else 1.0
                    except ValueError:
                        log.warning("PacBio CSV line %d: invalid fraction '%s' "
                                    "-- using 1.0", lineno, frac_str)
                        fraction = 1.0
                else:
                    fraction = 1.0

                # -- nDetected (optional; blank -> bypass min_detected filter) --
                # A blank nDetected means the data is absent (e.g. REBASE-derived
                # entries).  Do not treat blank as 0 -- that would incorrectly
                # filter out valid high-confidence entries that have no count data.
                n_detected: int | None = None
                if has_ndetected:
                    nd_str = row.get('nDetected', '').strip()
                    if nd_str:
                        try:
                            n_detected = int(nd_str)
                        except ValueError:
                            log.warning("PacBio CSV line %d: invalid nDetected '%s' "
                                        "-- bypassing filter", lineno, nd_str)

                # -- Apply thresholds --
                if fraction < min_fraction:
                    continue
                if n_detected is not None and n_detected < min_detected:
                    continue

                # -- modificationType (optional, infer from base if absent) --
                if has_mod_type:
                    mod_type = row.get('modificationType', '').strip()
                else:
                    mod_type = ''

                if not mod_type or mod_type == 'modified_base':
                    # centerPos is 1-based; convert to 0-based for indexing
                    idx = center_pos - 1
                    if idx < 0 or idx >= len(motif_seq):
                        log.warning("PacBio CSV line %d: centerPos %d out of "
                                    "bounds for '%s' -- skipped",
                                    lineno, center_pos, motif_seq)
                        continue
                    base = motif_seq[idx].upper()
                    resolved = _BASE_TO_METH.get(base)
                    if resolved is None:
                        # Try complement strand: G→C→m4C, T→A→m6A
                        comp_resolved = _COMP_BASE_TO_METH.get(base)
                        if comp_resolved is not None:
                            rc_motif = reverse_complement(motif_seq)
                            rc_idx = len(motif_seq) - 1 - idx
                            mod_type = comp_resolved
                            motif_seq = rc_motif
                            center_pos = rc_idx + 1  # back to 1-based
                            log.info("PacBio CSV line %d: base '%s' on complement strand "
                                     "-> RC motif %s centerPos %d (%s)",
                                     lineno, base, motif_seq, center_pos, mod_type)
                        else:
                            log.warning("PacBio CSV line %d: cannot resolve mod "
                                        "type at %s[%d]='%s' -- skipped",
                                        lineno, motif_seq, center_pos, base)
                            continue
                    else:
                        mod_type = resolved

                if mod_type not in METH_IDS:
                    log.warning("PacBio CSV line %d: unknown mod type '%s' "
                                "for %s -- skipped", lineno, mod_type, motif_seq)
                    continue

                nd_out = n_detected if n_detected is not None else 0
                entries.append(
                    f"{mod_type},{motif_seq},{center_pos},{nd_out},{fraction:.6g}"
                )

        return ";".join(entries)

    def is_file_for_this_parser(self, filepath: str) -> bool:
        """Match .csv files that contain motifString in the header."""
        if not filepath.lower().endswith('.csv'):
            return False
        try:
            with open(filepath) as f:
                header = f.readline()
                return 'motifString' in header
        except OSError:
            return False
