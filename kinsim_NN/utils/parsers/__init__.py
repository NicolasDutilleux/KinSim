"""Methylation caller output parsers for KinSim.

Read-only parsing library that converts output files from different
methylation callers into KinSim motif strings.

Supported parsers (auto-detected by ``load_motif_string``):
  - pacbio:    PacBio SMRT Link motifs.csv (variable columns)
  - modkit:    Oxford Nanopore modkit pileup --bedMethyl TSV
  - combined:  Combined CSV (mod_type, motif, offset, frac_mod, n_sites, source)
  - rebase:    REBASE simplified or Format #19 (withrefm) — via
               ``kinsim_NN.utils.parsers.rebase`` (kept separate because
               rebase parsing has its own CLI and helpers)

Usage:
    from kinsim_NN.utils.parsers import create_parser, list_parsers, auto_detect_parser

    parser = create_parser("pacbio")
    motif_string = parser.parse("/data/motifs.csv")

    parser = auto_detect_parser("/data/output.csv")
    if parser:
        motif_string = parser.parse("/data/output.csv")
"""

# Import parsers to trigger @register decorators
from . import combined as _combined  # noqa: F401
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
