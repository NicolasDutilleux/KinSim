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
import functools
import logging

from kinsim2.utils.encoding import get_meth_ids
from kinsim2.utils.motifs import reverse_complement

from .base import BaseOutputParser
from .registry import register

log = logging.getLogger(__name__)


_DNA_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


@functools.lru_cache(maxsize=1)
def _base_to_meth() -> dict[str, str]:
    """``{forward-strand base: meth_type}`` from YAML. Cached per process.

    If two meth types share the same base (m4C + m5C both on C), the
    mapping leaves that base out — user MUST set modificationType in
    their motifs.csv. Cache size 1 means a YAML reload (e.g. tests)
    needs ``_base_to_meth.cache_clear()``.
    """
    from kinsim2.utils.config import get_modified_base_map

    by_base: dict[str, list[str]] = {}
    for mod_type, base in get_modified_base_map().items():
        by_base.setdefault(base, []).append(mod_type)
    return {b: mods[0] for b, mods in by_base.items() if len(mods) == 1}


@functools.lru_cache(maxsize=1)
def _comp_base_to_meth() -> dict[str, str]:
    """``{complement-strand base: meth_type}`` from YAML. Cached per process.

    Used when ``motifs.csv`` reports a centerPos pointing at a non-modifiable
    base (G or T on the forward strand): the modified base of the reverse-
    complement strand maps via the standard ACGT complement.
    """
    fwd = _base_to_meth()
    return {base: fwd[comp] for base, comp in _DNA_COMPLEMENT.items() if comp in fwd}


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

            has_fraction = "fraction" in reader.fieldnames
            has_ndetected = "nDetected" in reader.fieldnames
            has_mod_type = "modificationType" in reader.fieldnames

            for lineno, row in enumerate(reader, 2):
                # -- motifString and centerPos are required --
                motif_seq = row.get("motifString", "").strip()
                center_str = row.get("centerPos", "").strip()
                if not motif_seq or not center_str:
                    log.warning(
                        "PacBio CSV line %d: missing motifString or centerPos -- skipped", lineno
                    )
                    continue

                try:
                    center_pos = int(center_str)
                except ValueError:
                    log.warning(
                        "PacBio CSV line %d: invalid centerPos '%s' -- skipped", lineno, center_str
                    )
                    continue

                # -- fraction (optional, default 1.0) --
                if has_fraction:
                    frac_str = row.get("fraction", "").strip()
                    try:
                        fraction = float(frac_str) if frac_str else 1.0
                    except ValueError:
                        log.warning(
                            "PacBio CSV line %d: invalid fraction '%s' -- using 1.0",
                            lineno,
                            frac_str,
                        )
                        fraction = 1.0
                else:
                    fraction = 1.0

                # -- nDetected (optional; blank -> bypass min_detected filter) --
                # A blank nDetected means the data is absent (e.g. REBASE-derived
                # entries).  Do not treat blank as 0 -- that would incorrectly
                # filter out valid high-confidence entries that have no count data.
                n_detected: int | None = None
                if has_ndetected:
                    nd_str = row.get("nDetected", "").strip()
                    if nd_str:
                        try:
                            n_detected = int(nd_str)
                        except ValueError:
                            log.warning(
                                "PacBio CSV line %d: invalid nDetected '%s' -- bypassing filter",
                                lineno,
                                nd_str,
                            )

                # -- Apply thresholds --
                if fraction < min_fraction:
                    continue
                if n_detected is not None and n_detected < min_detected:
                    continue

                # -- modificationType (optional, infer from base if absent) --
                if has_mod_type:
                    mod_type = row.get("modificationType", "").strip()
                else:
                    mod_type = ""

                if not mod_type or mod_type == "modified_base":
                    # centerPos is 1-based; convert to 0-based for indexing
                    idx = center_pos - 1
                    if idx < 0 or idx >= len(motif_seq):
                        log.warning(
                            "PacBio CSV line %d: centerPos %d out of bounds for '%s' -- skipped",
                            lineno,
                            center_pos,
                            motif_seq,
                        )
                        continue
                    base = motif_seq[idx].upper()
                    resolved = _base_to_meth().get(base)
                    if resolved is None:
                        # Forward base wasn't a known meth target. Try the
                        # complement strand (G→C, T→A — base on the rev
                        # strand IS what's actually modified).
                        comp_resolved = _comp_base_to_meth().get(base)
                        if comp_resolved is not None:
                            rc_motif = reverse_complement(motif_seq)
                            rc_idx = len(motif_seq) - 1 - idx
                            mod_type = comp_resolved
                            motif_seq = rc_motif
                            center_pos = rc_idx + 1  # back to 1-based
                            log.info(
                                "PacBio CSV line %d: base '%s' on complement strand "
                                "-> RC motif %s centerPos %d (%s)",
                                lineno,
                                base,
                                motif_seq,
                                center_pos,
                                mod_type,
                            )
                        else:
                            log.warning(
                                "PacBio CSV line %d: cannot resolve mod "
                                "type at %s[%d]='%s' -- skipped",
                                lineno,
                                motif_seq,
                                center_pos,
                                base,
                            )
                            continue
                    else:
                        mod_type = resolved

                if mod_type not in get_meth_ids():
                    log.warning(
                        "PacBio CSV line %d: unknown mod type '%s' for %s -- skipped",
                        lineno,
                        mod_type,
                        motif_seq,
                    )
                    continue

                nd_out = n_detected if n_detected is not None else 0
                entries.append(f"{mod_type},{motif_seq},{center_pos},{nd_out},{fraction:.6g}")

        return ";".join(entries)

    def is_file_for_this_parser(self, filepath: str) -> bool:
        """Match .csv files that contain motifString in the header."""
        if not filepath.lower().endswith(".csv"):
            return False
        try:
            with open(filepath) as f:
                header = f.readline()
                return "motifString" in header
        except OSError:
            return False
