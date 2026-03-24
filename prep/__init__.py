"""KinSim data preparation tools.

This package contains everything needed to prepare input data for KinSim's
core pipeline (extract -> merge -> train -> generate -> evaluate):

Modules:
    callers/    Methylation caller output parsers (PacBio, modkit, ipdSummary,
                combined CSV).  Plugin registry with @register decorator.
    rebase      REBASE web fetch and file parsing; fuzznuc pattern file generation.
    motif_merge Merge, filter, and deduplicate motifs from multiple CSV sources.
    manifest    Manifest CSV inspection CLI (count / validate / list).
    prepare     Legacy BAM + motif-source pair validation (alternating-line format).
    filter      Filter .pkl by coverage, mod type, or key count.

CLI (kinsim-prep <subcommand>):
    kinsim-prep parse        Parse any motif source to a KinSim motif string
    kinsim-prep rebase       Fetch/parse REBASE data, generate fuzznuc patterns
    kinsim-prep merge-motifs Merge, filter, and deduplicate motifs
    kinsim-prep manifest     Inspect and validate manifest CSVs
    kinsim-prep prepare      Validate BAM + motif-source pairs (legacy format)
    kinsim-prep filter       Filter a .pkl by coverage, mod type, or key count
"""
