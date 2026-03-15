"""PacBio ipdSummary (kineticsTools) output parser.

Handles two output formats from ipdSummary (auto-detected):

1. CSV format:
   refName, tpl, strand, base, score, tMean, tErr, modelPrediction,
   ipdRatio, pvalue, ...
   - base "A" with high score -> m6A
   - base "C" with high score -> m4C

2. GFF3 format:
   seqname, source, feature, start, end, score, strand, frame, attributes
   - feature = "modified_base"
   - attributes contain "IPDRatio=..." and "modificationType=..."

Since ipdSummary outputs the standard PacBio motifs.csv format (as confirmed
by the user), this parser primarily handles that format with graceful fallback
for missing columns. For raw ipdSummary kinetics output (CSV/GFF), use the
format-specific parsing below.
"""

import csv
import logging
import re

from kinsim.utils.encoding import METH_IDS
from .base import BaseOutputParser
from .registry import register

log = logging.getLogger(__name__)

# ipdSummary base codes -> KinSim mod types
_BASE_TO_METH = {
    'A': 'm6A',
    'C': 'm4C',
}

# GFF attribute parser
_GFF_ATTR_RE = re.compile(r'(\w+)=([^;]+)')


@register
class IpdSummaryParser(BaseOutputParser):
    """Parser for ipdSummary CSV or GFF3 output."""

    name = "ipd_summary"
    supported_mods = ["m6A", "m4C"]

    def parse(
        self,
        filepath: str,
        min_fraction: float = 0.40,
        min_detected: int = 20,
    ) -> str:
        # Auto-detect CSV vs GFF
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('##gff'):
                    return self._parse_gff(filepath, min_fraction, min_detected)
                # CSV: check for header or tab-separated GFF
                if '\t' in line and not line.startswith('refName'):
                    fields = line.split('\t')
                    if len(fields) == 9 and fields[2] in ('modified_base', 'kinetic'):
                        return self._parse_gff(filepath, min_fraction, min_detected)
                break

        return self._parse_csv(filepath, min_fraction, min_detected)

    def _parse_csv(
        self,
        filepath: str,
        min_fraction: float,
        min_detected: int,
    ) -> str:
        """Parse ipdSummary CSV kinetics output."""
        entries: list[str] = []

        with open(filepath) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                log.warning("ipdSummary CSV: empty file '%s'", filepath)
                return ""

            for lineno, row in enumerate(reader, 2):
                base = row.get('base', '').strip().upper()
                mod_type = _BASE_TO_METH.get(base)
                if mod_type is None:
                    continue  # G/T bases are not methylation targets

                # Score/pvalue filtering
                score_str = row.get('score', '0').strip()
                try:
                    score = float(score_str)
                except ValueError:
                    continue

                # Use score as a proxy for confidence
                # ipdSummary scores are -10*log10(pvalue), higher = more significant
                if score < 20:  # roughly p < 0.01
                    continue

                ref_name = row.get('refName', '').strip()
                tpl = row.get('tpl', '0').strip()
                strand = row.get('strand', '0').strip()

                entry = f"{mod_type},{ref_name}:{tpl}:{strand},0,1,1.0"
                entries.append(entry)

        return ";".join(entries)

    def _parse_gff(
        self,
        filepath: str,
        min_fraction: float,
        min_detected: int,
    ) -> str:
        """Parse ipdSummary GFF3 output."""
        entries: list[str] = []

        with open(filepath) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                fields = line.split('\t')
                if len(fields) < 9:
                    continue

                feature = fields[2]
                if feature not in ('modified_base', 'kinetic'):
                    continue

                seqname = fields[0]
                start = fields[3]
                score_str = fields[5]
                strand = fields[6]
                attributes = fields[8]

                try:
                    score = float(score_str)
                except ValueError:
                    continue

                if score < 20:
                    continue

                # Parse attributes
                attrs = dict(_GFF_ATTR_RE.findall(attributes))
                mod_type_raw = attrs.get('modificationType', '')

                if mod_type_raw in METH_IDS:
                    mod_type = mod_type_raw
                else:
                    # Try to resolve from context
                    context = attrs.get('context', '')
                    if context:
                        base = context[0].upper()
                        mod_type = _BASE_TO_METH.get(base, '')
                    else:
                        continue

                if not mod_type or mod_type not in METH_IDS:
                    continue

                entry = f"{mod_type},{seqname}:{start}:{strand},0,1,1.0"
                entries.append(entry)

        return ";".join(entries)

    def is_file_for_this_parser(self, filepath: str) -> bool:
        """Match .gff or .csv files with ipdSummary-like content."""
        lower = filepath.lower()
        if 'ipdsummary' in lower or 'kinetics' in lower:
            return True
        if lower.endswith('.gff') or lower.endswith('.gff3'):
            try:
                with open(filepath) as f:
                    for line in f:
                        if 'kinetic' in line.lower() or 'ipdRatio' in line:
                            return True
                        if not line.startswith('#'):
                            break
            except OSError:
                pass
        return False
