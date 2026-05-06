"""Oxford Nanopore modkit pileup TSV output parser.

Handles modkit pileup --bedMethyl output format (BED-like TSV).
Columns (0-indexed):
  0: chrom
  1: start (0-based)
  2: end
  3: mod_code (e.g. "m" for 5mC, "a" for 6mA, "21839" for m4C)
  4: score
  5: strand
  6: start (thick)
  7: end (thick)
  8: color
  9: Nvalid
  10: fraction (percent, 0-100)

modkit mod codes:
  "a" or "6mA"   -> m6A
  "m" or "5mC"   -> m5C
  "21839"         -> m4C (SAM spec numeric code)
  "h" or "5hmC"  -> m5C (treated as m5C for KinSim purposes)

This parser produces per-site entries. Since modkit outputs per-position data
(not per-motif), the motif string entries use single-base "motifs" with
position 0. Downstream KinSim tools can still use these for training.
"""

import logging

from .base import BaseOutputParser
from .registry import register

log = logging.getLogger(__name__)

# modkit mod codes -> KinSim mod types
_MODKIT_CODE_MAP = {
    "a": "m6A",
    "6mA": "m6A",
    "m": "m5C",
    "5mC": "m5C",
    "h": "m5C",  # 5hmC -> treat as m5C
    "5hmC": "m5C",
    "21839": "m4C",  # SAM spec numeric code for m4C
}


@register
class ModkitParser(BaseOutputParser):
    """Parser for modkit pileup --bedMethyl TSV output."""

    name = "modkit"
    supported_mods = ["m6A", "m4C", "m5C"]

    def parse(
        self,
        filepath: str,
        min_fraction: float = 0.40,
        min_detected: int = 20,
    ) -> str:
        entries: list[str] = []
        seen: set[str] = set()  # deduplicate identical entries

        with open(filepath) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) < 11:
                    log.warning(
                        "modkit line %d: expected >=11 columns, got %d -- skipped",
                        lineno,
                        len(fields),
                    )
                    continue

                mod_code = fields[3].strip()
                mod_type = _MODKIT_CODE_MAP.get(mod_code)
                if mod_type is None:
                    log.warning(
                        "modkit line %d: unknown mod code '%s' -- skipped", lineno, mod_code
                    )
                    continue

                try:
                    n_valid = int(fields[9])
                    frac_pct = float(fields[10])
                except (ValueError, IndexError):
                    log.warning("modkit line %d: invalid Nvalid/fraction -- skipped", lineno)
                    continue

                fraction = frac_pct / 100.0  # percent -> [0,1]

                if fraction < min_fraction or n_valid < min_detected:
                    continue

                chrom = fields[0]
                start = fields[1]
                strand = fields[5]

                # Per-site entry: use chrom:pos:strand as "motif" context
                entry = f"{mod_type},{chrom}:{start}:{strand},0,{n_valid},{fraction:.6g}"
                if entry not in seen:
                    seen.add(entry)
                    entries.append(entry)

        return ";".join(entries)

    def is_file_for_this_parser(self, filepath: str) -> bool:
        """Match .bed or .tsv files with modkit-like content."""
        lower = filepath.lower()
        if not (
            lower.endswith(".bed")
            or lower.endswith(".tsv")
            or "modkit" in lower
            or "bedmethyl" in lower
        ):
            return False
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split("\t")
                    # modkit bedMethyl has 11+ columns, col 3 is mod code
                    return bool(len(fields) >= 11 and fields[3].strip() in _MODKIT_CODE_MAP)
        except OSError:
            return False
        return False
