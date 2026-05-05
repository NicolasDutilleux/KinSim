"""Methylation caller output parsers for KinSim.

Read-only parsing library that converts output files from different
methylation callers into KinSim motif strings.

Supported parsers:
  - pacbio:      PacBio SMRT Link motifs.csv (variable columns)
  - modkit:      Oxford Nanopore modkit pileup --bedMethyl TSV
  - ipd_summary: PacBio ipdSummary CSV or GFF3 output
  - combined:    Combined CSV (mod_type, motif, offset, frac_mod, n_sites, source)

Usage:
    from prep.callers import create_parser, list_parsers, auto_detect_parser

    # Explicit parser
    parser = create_parser("pacbio")
    motif_string = parser.parse("/data/motifs.csv")

    # Auto-detect from file content
    parser = auto_detect_parser("/data/output.csv")
    if parser:
        motif_string = parser.parse("/data/output.csv")
"""

# Import parsers to trigger @register decorators
from . import combined as _combined  # noqa: F401
from . import ipd_summary as _ipd_summary  # noqa: F401
from . import modkit as _modkit  # noqa: F401
from . import pacbio as _pacbio  # noqa: F401
from .base import BaseOutputParser
from .registry import auto_detect_parser, create_parser, list_parsers

__all__ = [
    "BaseOutputParser",
    "auto_detect_parser",
    "create_parser",
    "list_parsers",
]
