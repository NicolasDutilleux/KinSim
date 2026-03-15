"""KinSim data preparation tools.

This subpackage contains everything needed to prepare input data for KinSim's
core pipeline (extract → merge → train → generate → evaluate):

Modules:
    callers/    Methylation caller output parsers (PacBio, modkit, ipdSummary,
                combined CSV).  Plugin registry with @register decorator.
    rebase      REBASE file parsing and fuzznuc pattern file generation.
    prepare     Legacy BAM + motif-source pair validation (alternating-line format).
    manifest    Manifest CSV inspection CLI (count / validate / list).
    filter      General Dictionary → Training Dictionary filtering with
                configurable thresholds.

CLI commands (via kinsim prep <subcommand>):
    kinsim prep parse        Parse any motif source → KinSim motif string
    kinsim prep rebase       Parse REBASE files / generate fuzznuc patterns
    kinsim prep manifest     Inspect and validate manifest CSVs
    kinsim prep prepare      Validate BAM + motif-source pairs (legacy format)
    kinsim prep filter       Filter a General Dictionary .pkl into a Training set
"""
