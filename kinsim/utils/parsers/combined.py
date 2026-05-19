"""Combined methylation motif CSV parser.

Handles the user's combined motifs.csv format produced by merging modkit 5mC
and fibertools 6mA results into a single CSV:

    mod_type,motif,offset,frac_mod,n_sites,source
    5mC,TCNATCY,1,0.964939,633,modkit
    6mA,GATC,1,0.891234,1205,fibertools

Column mapping to KinSim:
    mod_type  -> KinSim mod type (5mC->m5C, 6mA->m6A, 4mC->m4C)
    motif     -> IUPAC recognition sequence
    offset    -> 1-based position of modified base within motif
                 (PacBio convention; matches parse_motifs() which subtracts 1)
    n_sites   -> nDetected (for filtering)
    frac_mod  -> fraction (for filtering)
    source    -> informational only (modkit, fibertools, etc.)

This format is NOT compatible with PacBioParser (which expects motifString,
centerPos columns).  Auto-detection checks for the presence of 'mod_type'
and 'frac_mod' in the CSV header.
"""

import csv
import logging

from kinsim.utils.encoding import METH_IDS, get_meth_ids

from .base import BaseOutputParser
from .registry import register

log = logging.getLogger(__name__)


# Combined CSVs use various conventions for mod names ('5mC' vs 'm5C',
# '6mA' vs 'm6A'). The alias-to-canonical mapping is derived from
# kinsim_config.yaml's ``aliases`` field per meth type, so adding a new
# methylation (or a new alias for an existing one) is a YAML edit only.
def _combined_mod_map() -> dict[str, str]:
    from kinsim.utils.config import get_meth_alias_map

    return get_meth_alias_map()


@register
class CombinedParser(BaseOutputParser):
    """Parser for combined methylation motif CSV (modkit + fibertools merge)."""

    name = "combined"
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
                log.warning("Combined CSV: empty or headerless file '%s'", filepath)
                return ""

            for lineno, row in enumerate(reader, 2):
                # -- mod_type (required) --
                raw_mod = row.get("mod_type", "").strip()
                mod_type = _combined_mod_map().get(raw_mod)
                if mod_type is None:
                    log.warning(
                        "Combined CSV line %d: unknown mod_type '%s' -- skipped", lineno, raw_mod
                    )
                    continue

                # -- motif (required) --
                motif_seq = row.get("motif", "").strip()
                if not motif_seq:
                    log.warning("Combined CSV line %d: missing motif -- skipped", lineno)
                    continue

                # -- offset (required) --
                offset_str = row.get("offset", "").strip()
                try:
                    offset = int(offset_str)
                except ValueError:
                    log.warning(
                        "Combined CSV line %d: invalid offset '%s' -- skipped", lineno, offset_str
                    )
                    continue

                # -- frac_mod (optional, default 1.0) --
                frac_str = row.get("frac_mod", "").strip()
                try:
                    fraction = float(frac_str) if frac_str else 1.0
                except ValueError:
                    log.warning(
                        "Combined CSV line %d: invalid frac_mod '%s' -- using 1.0", lineno, frac_str
                    )
                    fraction = 1.0

                # -- n_sites (optional, default 0) --
                ns_str = row.get("n_sites", "").strip()
                try:
                    n_sites = int(ns_str) if ns_str else 0
                except ValueError:
                    log.warning(
                        "Combined CSV line %d: invalid n_sites '%s' -- using 0", lineno, ns_str
                    )
                    n_sites = 0

                # -- Apply thresholds --
                if fraction < min_fraction or n_sites < min_detected:
                    continue

                if mod_type not in get_meth_ids():
                    log.warning(
                        "Combined CSV line %d: mod_type '%s' not declared in "
                        "kinsim_config.yaml kinetic_signatures -- skipped",
                        lineno,
                        mod_type,
                    )
                    continue

                entries.append(f"{mod_type},{motif_seq},{offset},{n_sites},{fraction:.6g}")

        return ";".join(entries)

    def is_file_for_this_parser(self, filepath: str) -> bool:
        """Match .csv files with mod_type and frac_mod in the header."""
        if not filepath.lower().endswith(".csv"):
            return False
        try:
            with open(filepath) as f:
                header = f.readline().lower()
                return "mod_type" in header and "frac_mod" in header
        except OSError:
            return False
