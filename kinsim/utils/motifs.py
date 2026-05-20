"""IUPAC motif parsing, methylation scanning, and reference pre-scanning.

Supports three motif input sources (auto-detected by load_motif_string):
  1. KinSim motif string  — "m6A,GATC,1;m4C,CCWGG,1"
  2. PacBio motifs.csv    — output of SMRT Link basecall pipeline
  3. REBASE file          — simplified two-column or Format #19 (withrefm)
                            Delegated to kinsim.rebase_parser

Motif scanning backends:
  - EMBOSS fuzznuc (primary): used for reference-level genome pre-scanning.
    A single subprocess call with a named-pattern file (@patterns.txt) covers
    all motifs at once.  Falls back to regex automatically if fuzznuc is not
    installed (no error, just a warning).
  - Python regex (in-memory): retained for per-read scanning during BAM
    training and for unmapped-read fallback paths in inject/generate.
    Running fuzznuc via subprocess inside a per-read BAM loop would be
    prohibitively slow; the regex backend handles these cases efficiently.

Motif string format:
  "m6A,GATC,2;m4C,CCWGG,2;m5C,RGATCY,4"
  Each entry: MOD_TYPE,IUPAC_MOTIF,1-based_MOD_POS[,nDetected[,fraction]]
  ``MOD_POS`` is 1-based — matches PacBio motifs.csv's ``centerPos`` column.
  ``parse_motifs`` internally subtracts 1 to convert to a Python index.
  Fields 4 (nDetected) and 5 (fraction) are optional metadata from PacBio CSV.
  They are ignored by train/inject/generate logic but preserved for traceability.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

from .encoding import METH_IDS, get_meth_ids

log = logging.getLogger(__name__)

IUPAC_TO_REGEX = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "N": ".",
    "R": "[AG]",
    "Y": "[CT]",
    "S": "[GC]",
    "W": "[AT]",
    "K": "[GT]",
    "M": "[AC]",
    "B": "[CGT]",
    "D": "[AGT]",
    "H": "[ACT]",
    "V": "[ACG]",
}

# Concrete bases that an IUPAC code can match.  Used by `_validate_mod_pos`
# to check whether a declared mod_pos lands on the right base for a given
# methylation type. The mapping (mod_type → expected base) is itself read
# from kinsim_config.yaml so adding a new modification type is a YAML edit
# only — no code change.
_IUPAC_EXPANSIONS = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "N": "ACGT",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
}

COMPLEMENT = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "N": "N",
    "Y": "R",
    "R": "Y",
    "S": "S",
    "W": "W",
    "K": "M",
    "M": "K",
    "B": "V",
    "V": "B",
    "D": "H",
    "H": "D",
}


# PacBio CSV "modified_base" resolver: when modificationType is blank or
# generic, infer the meth type from the base at centerPos. Built lazily
# from kinsim_config.yaml's ``modified_base`` fields so the mapping is
# generalisable — adding a new modification (e.g. m4mC at C) is a YAML
# edit only.
def _build_base_to_meth() -> dict[str, str]:
    """Build {base: mod_type} from YAML.

    If multiple meth types modify the same base (e.g. m4C and m5C both
    modify C), the resolver cannot disambiguate and the user MUST set
    the ``modificationType`` column explicitly in their motifs.csv.
    Returns the mapping with such ambiguous bases mapped to ``None``.
    """
    from .config import get_modified_base_map

    by_base: dict[str, list[str]] = {}
    for mod_type, base in get_modified_base_map().items():
        by_base.setdefault(base, []).append(mod_type)
    out: dict[str, str] = {}
    for base, mods in by_base.items():
        if len(mods) == 1:
            out[base] = mods[0]
        # else: ambiguous → caller must look at modificationType, not base
    return out


# GFF attribute parser: extracts pattern name from fuzznuc GFF output.
# Matches "Pattern_name=...", "Name=...", or "pattern=..." (case-insensitive).
_GFF_ATTR_NAME_RE = re.compile(r"(?:Pattern_name|Name|pattern)=([^;]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# IUPAC helpers
# ---------------------------------------------------------------------------


def iupac_to_re(motif):
    """Convert an IUPAC motif string to a regex pattern string."""
    return "".join(IUPAC_TO_REGEX.get(b, b) for b in motif)


def _iupac_includes(iupac_char: str, base: str) -> bool:
    """Return True if the IUPAC code ``iupac_char`` can match concrete ``base``.

    e.g. ``_iupac_includes("R", "A") == True`` (R = A/G).
    """
    if not iupac_char or not base:
        return False
    return base.upper() in _IUPAC_EXPANSIONS.get(iupac_char.upper(), "")


def _validate_mod_pos(seq: str, mod_pos: int, meth_type: str) -> None:
    """Hard-validate that ``seq[mod_pos]`` matches ``meth_type``'s modified base.

    The motif format ``mod_type,pattern,pos`` is unambiguous by design:
    ``pos`` is the 1-based position of the modified base in ``pattern``,
    and the modified base is determined by ``mod_type`` via the
    ``modified_base`` field of ``kinetic_signatures.<mod_type>`` in
    kinsim_config.yaml. If the trio is internally inconsistent the
    motif spec is wrong — silently auto-correcting would hide bugs in
    the upstream caller / motif file and corrupt training data with no
    visible failure.

    This function therefore RAISES ``ValueError`` on any mismatch with
    a message naming the offending entry. The caller (``parse_motifs``)
    aggregates these and reports all bad rows at once before failing,
    so the user fixes the source motifs.csv instead of re-running and
    re-failing one row at a time.

    IUPAC codes that *include* the expected base (e.g. R for m6A — R
    can be A or G) pass: the motif designer knowingly used an
    ambiguous code, accepting that some forward-strand matches will
    not actually carry the modification.

    Methylation types that aren't declared in kinsim_config.yaml will
    fail upstream (parse_motifs raises before reaching here), so this
    function never sees an unknown type in practice. As a final guard,
    a missing-from-YAML lookup raises with the same helpful message as
    every other config-gap path.
    """
    from .config import get_modified_base

    if not seq:
        return
    expected = get_modified_base(meth_type)  # raises if meth_type undeclared
    n = len(seq)
    if 0 <= mod_pos < n and _iupac_includes(seq[mod_pos], expected):
        return
    actual = seq[mod_pos] if 0 <= mod_pos < n else "<out-of-range>"
    raise ValueError(
        f"Motif '{seq}' ({meth_type}): position {mod_pos} (0-based) points to "
        f"'{actual}', but {meth_type} modifies '{expected}'. Fix the source "
        f"motifs.csv (the 'pos' column for this entry is wrong: it should "
        f"point to a base that is — or includes via IUPAC — '{expected}')."
    )


def reverse_complement(seq):
    """Reverse complement supporting IUPAC ambiguity codes."""
    return "".join(COMPLEMENT.get(base, base) for base in reversed(seq))


# ---------------------------------------------------------------------------
# Modification-type filter
# ---------------------------------------------------------------------------


def parse_meth_types_arg(arg: str | None) -> set[str] | None:
    """Parse a ``--meth-types`` CLI value into a set of mod type strings.

    Accepts ``"m6A,m4C"``, ``"m6A"``, ``"all"`` (synonym for None / no filter),
    or None.  Empty string → None.

    Returns:
        A set like ``{"m6A", "m4C"}``, or ``None`` to mean "no filter".
    """
    if arg is None:
        return None
    s = arg.strip()
    if not s or s.lower() == "all":
        return None
    return {tok.strip() for tok in s.split(",") if tok.strip()}


def filter_motif_string_by_types(motif_string: str, allowed_mods: set[str] | None) -> str:
    """Keep only motif entries whose mod type is in ``allowed_mods``.

    Used at both extract time (upstream of motif-based scanning) and generate
    time (upstream of reference pre-scan) to enforce the active mod alphabet.
    A skipped entry simply never appears in the scan output — positions are
    NOT relabelled as unmethylated.

    Args:
        motif_string:  Semicolon-delimited motif entries.
        allowed_mods:  ``None`` → no filter.  Otherwise a set of mod type
                       strings (e.g. ``{"m6A", "m4C"}``).

    Returns:
        Filtered motif string.  Empty string if all entries were filtered out.
    """
    if not motif_string or allowed_mods is None:
        return motif_string
    kept = []
    for entry in motif_string.split(";"):
        if not entry or "," not in entry:
            continue
        mod_type = entry.split(",", 1)[0].strip()
        if mod_type in allowed_mods:
            kept.append(entry)
    return ";".join(kept)


# ---------------------------------------------------------------------------
# KinSim motif string: parse and scan (in-memory regex backend)
# ---------------------------------------------------------------------------


def parse_motifs(motif_string, revcomp=True):
    """Parse a motif string and compile regex for forward + reverse complement.

    IN-MEMORY REGEX BACKEND — used for per-read scanning during BAM training
    and unmapped-read fallback in generate.  This function must remain
    regex-based because fuzznuc
    subprocess calls per read are prohibitively slow.

    For reference-level scanning (done once per genome), use
    build_reference_meth_map() instead, which uses EMBOSS fuzznuc as the
    primary backend.

    Input format: "m6A,GATC,1;m4C,CCWGG,1;m5C,RGATCY,4"
    Each entry: MOD_TYPE,IUPAC_MOTIF,MOD_POS[,nDetected[,fraction]] — semicolon-delimited.
    Fields beyond index 2 are optional metadata (ignored here, preserved for traceability).

    Args:
        motif_string: Semicolon-delimited motif entries.
        revcomp: If True (default), generate both forward and reverse complement
            patterns.  Set to False when motif_string already contains both
            orientations (e.g., from PacBio CSV with partner motifs).

    Returns list of dicts with keys: 'pattern' (compiled regex with lookahead),
    'id' (methylation type int), 'pos' (modified base offset within match).
    """
    motifs = []
    if not motif_string:
        return motifs

    # Pre-collect user-provided sequences. If the user has provided BOTH a
    # motif M and its reverse complement (often the case for palindromic
    # methylation systems like Type II R-M, where each strand carries its own
    # methylation event at different positions), we must NOT auto-expand the
    # revcomp again — doing so would tag positions on the read that are NOT
    # the actual methylated base on either strand (the user's two motifs
    # already cover both strand views explicitly).
    user_entries = []
    for entry in motif_string.split(";"):
        if not entry or "," not in entry:
            continue
        parts = entry.split(",")
        if len(parts) < 3:
            continue
        user_entries.append(parts)
    user_seqs = {parts[1] for parts in user_entries}

    # First pass: parse + validate every entry. Collect ALL errors so the user
    # sees every bad row at once, not one-per-rerun. Bail before building any
    # regexes if validation fails — partial state would confuse downstream
    # callers that try/except around the failure.
    parsed: list[tuple] = []
    errors: list[str] = []
    for parts in user_entries:
        m_type, seq, pos = parts[0], parts[1], parts[2]
        try:
            mod_pos = int(pos) - 1
        except ValueError:
            errors.append(f"Motif '{seq}' ({m_type}): pos='{pos}' is not an integer.")
            continue
        try:
            _validate_mod_pos(seq, mod_pos, m_type)
        except ValueError as e:
            errors.append(str(e))
            continue
        parsed.append((m_type, seq, mod_pos, parts))

    if errors:
        bullet = "\n  - ".join(errors)
        raise ValueError(
            f"parse_motifs: {len(errors)} invalid motif "
            f"{'entry' if len(errors) == 1 else 'entries'} "
            f"(out of {len(user_entries)}). The motif format is "
            f"'mod_type,pattern,1-based_pos[,nDetected,fraction]'. "
            f"Fix the source motifs.csv and re-run.\n  - {bullet}"
        )

    for m_type, seq, mod_pos, parts in parsed:
        m_id = get_meth_ids().get(m_type, 0)
        pairs = [(seq, mod_pos)]
        if revcomp:
            rc = reverse_complement(seq)
            # Skip auto-revcomp expansion if the user already provided the
            # revcomp sequence as a separate motif entry (palindromic system).
            if rc not in user_seqs:
                rc_mod_pos = len(seq) - 1 - mod_pos
                # Validate the auto-rc'd entry too. The naive
                # ``len(seq) - 1 - mod_pos`` formula points to the
                # complement-base position on the rc string, which is NOT the
                # methylated base of the same chemistry — e.g. CTGAAG (m6A at
                # the second A, idx 4) auto-rc'd to CTTCAG with idx 1 lands
                # on a 'T'. Catching this at parse time prevents us from
                # silently flagging T positions as m6A in scan_sequence.
                try:
                    _validate_mod_pos(rc, rc_mod_pos, m_type)
                    pairs.append((rc, rc_mod_pos))
                except ValueError as exc:
                    log.warning(
                        "parse_motifs: auto-revcomp of '%s' (%s, mod_pos=%d) → "
                        "(%s, mod_pos=%d) is invalid (%s). Provide the revcomp "
                        "explicitly in your motif file with the correct mod_pos.",
                        seq,
                        m_type,
                        mod_pos,
                        rc,
                        rc_mod_pos,
                        exc,
                    )

        frac = float(parts[4]) if len(parts) >= 5 else 1.0

        for s, offset in pairs:
            regex_pattern = re.compile(f"(?=({iupac_to_re(s)}))")
            motifs.append({"pattern": regex_pattern, "id": m_id, "pos": offset, "frac": frac})
    return motifs


# ---------------------------------------------------------------------------
# Strand-tagged motif parsing — for orientation-aware aligned-BAM extraction
# ---------------------------------------------------------------------------


def parse_motifs_per_strand(motif_string: str) -> tuple[list, list]:
    """Parse motifs into TWO lists: forward-strand and reverse-strand.

    Used by the aligned-BAM extract path to track which reference strand
    each motif lives on. The two resulting lists are scanned against the
    forward reference sequence (forward_motifs scan ``ref``) and against
    the reverse-complement (reverse_motifs scan ``rc_ref``) respectively.
    The methylation positions returned by each scan are then in REFERENCE
    forward-strand coordinates (after rc → ref position conversion).

    Why this matters: in HiFi BAMs, ``ip[read_pos]`` is the IPD when the
    polymerase synthesised position ``read_pos`` of the read sequence. The
    polymerase was reading the OPPOSITE strand of the read as template. So
    ``ip`` carries the kinetic effect of methylations on the strand opposite
    to the read sequence:
        - read.is_reverse=False  → read sequence = + strand of reference
                                  → ip reads − strand template
                                  → captures methylation on − strand at ref_pos
        - read.is_reverse=True   → read sequence = rev-comp of + strand
                                  → ip reads + strand template
                                  → captures methylation on + strand at ref_pos

    Pairing this with strand-resolved motif maps gives signal at the right
    reads. With raw unaligned HiFi we can't do this routing — every read is
    50/50 likely to need fi vs ri at any given position, so the signal
    averages out.

    Returns:
        (fwd_motifs, rev_motifs) — each a list of dicts identical in shape
        to ``parse_motifs`` output (``pattern``, ``id``, ``pos``, ``frac``).
        ``fwd_motifs`` are the user-provided sequences (matching the +
        reference strand). ``rev_motifs`` are the rev-comps with rc-coord
        ``pos``; positions found in rc need to be mapped back to forward
        ref coords via ``ref_pos = ref_len - 1 - rc_pos`` by the caller.
    """
    if not motif_string:
        return [], []

    user_entries = []
    for entry in motif_string.split(";"):
        if not entry or "," not in entry:
            continue
        parts = entry.split(",")
        if len(parts) < 3:
            continue
        user_entries.append(parts)
    user_seqs = {parts[1] for parts in user_entries}

    fwd_motifs: list = []
    rev_motifs: list = []
    errors: list[str] = []
    for parts in user_entries:
        m_type, seq, pos = parts[0], parts[1], parts[2]
        try:
            mod_pos = int(pos) - 1
        except ValueError:
            errors.append(f"Motif '{seq}' ({m_type}): pos='{pos}' is not an integer.")
            continue
        try:
            _validate_mod_pos(seq, mod_pos, m_type)
        except ValueError as e:
            errors.append(str(e))
            continue
        m_id = get_meth_ids().get(m_type, 0)
        frac = float(parts[4]) if len(parts) >= 5 else 1.0
        # Forward: matches the user-provided sequence on the + strand
        fwd_pattern = re.compile(f"(?=({iupac_to_re(seq)}))")
        fwd_motifs.append({"pattern": fwd_pattern, "id": m_id, "pos": mod_pos, "frac": frac})
        # Reverse: matches the same modification on the − strand. Use the
        # rev-comp sequence with ``rc_mod_pos = len(seq) - 1 - mod_pos`` and
        # validate that the rc base is a target for this meth type.
        # NOTE: even when the user provides the rc as a separate motif entry
        # (e.g. asymmetric pair CTGAAG + CTTCAG), we STILL add this rev_motif.
        # Otherwise rev_motifs ends up empty for bilaterally-provided motifs,
        # and both partners land in fwd_motifs — mis-routing − strand
        # methylations into fwd_meth_map. The duplicate that comes from the
        # partner iteration scans the same + strand positions and is fine.
        rc = reverse_complement(seq)
        rc_mod_pos = len(seq) - 1 - mod_pos
        try:
            _validate_mod_pos(rc, rc_mod_pos, m_type)
        except ValueError as exc:
            log.warning(
                "parse_motifs_per_strand: auto-rc of '%s' (%s, mod_pos=%d) → "
                "(%s, mod_pos=%d) invalid (%s). Skipping − strand entry.",
                seq,
                m_type,
                mod_pos,
                rc,
                rc_mod_pos,
                exc,
            )
            continue
        rev_pattern = re.compile(f"(?=({iupac_to_re(rc)}))")
        rev_motifs.append({"pattern": rev_pattern, "id": m_id, "pos": rc_mod_pos, "frac": frac})

    if errors:
        bullet = "\n  - ".join(errors)
        raise ValueError(
            f"parse_motifs_per_strand: {len(errors)} invalid motif "
            f"{'entry' if len(errors) == 1 else 'entries'}.\n  - {bullet}"
        )

    # Strand bookkeeping summary:
    #   Each user-provided entry contributes one fwd_pattern (scans + strand
    #   for the user's exact sequence) and one rev_pattern (scans + strand
    #   for the rc of the user's sequence — which corresponds to − strand
    #   occurrences of the original motif).
    #   - Palindromic motifs (e.g. GATC): fwd and rev patterns scan the same
    #     sites; rev_meth_map tags the partner-strand methylation position.
    #   - Asymmetric bilaterally-provided pairs (CTGAAG + CTTCAG): each entry
    #     adds its own fwd + rev. The "duplicate" rev from one partner equals
    #     the fwd of the other, but their meth_pos differ — each captures the
    #     methylation on its own strand at the correct base.
    return fwd_motifs, rev_motifs


def scan_sequence(seq, motifs):
    """Scan a DNA sequence for methylation motifs (in-memory regex backend).

    Tags each position OF THE MODIFIED BASE with the meth_id. The model
    learns the kinetic signature (e.g. m6A at +5, m5C at +2/+6) from the
    surrounding methylation context fed to FiLM, rather than us
    hard-coding signature offsets here.

    Returns an int8 numpy array of length len(seq), where each position
    holds the methylation type ID (0 = unmethylated).
    """
    status = np.zeros(len(seq), dtype=np.int8)
    for motif in motifs:
        for match in motif["pattern"].finditer(seq):
            target_pos = match.start() + motif["pos"]
            if 0 <= target_pos < len(seq):
                status[target_pos] = motif["id"]
    return status


# ---------------------------------------------------------------------------
# PacBio motifs.csv parser
# ---------------------------------------------------------------------------


def parse_motifs_csv(csv_path, min_fraction=0.40, min_detected=20):
    """Thin wrapper kept for back-compat — forwards to PacBioParser.

    The historical body of this function duplicated PacBioParser.parse() and
    drifted from it over time. Retired in favour of the registered parser
    (kinsim.utils.parsers.pacbio.PacBioParser).
    """
    from .parsers import create_parser
    return create_parser("pacbio").parse(
        csv_path, min_fraction=min_fraction, min_detected=min_detected
    )


# ---------------------------------------------------------------------------
# Unified motif-string loader (auto-detect source)
# ---------------------------------------------------------------------------


def load_motif_string(motifs_arg, min_fraction=0.40, min_detected=20, parser_name=None):
    """Load a KinSim motif string from a file path or return the argument as-is.

    Auto-detection (when parser_name is None):
        1. If motifs_arg is an existing file path ending in '.csv'
           -> parse as PacBio motifs.csv (applies min_fraction / min_detected)
        2. Try auto_detect_parser() from the callers registry
        3. Fall through to REBASE file parser
        4. Otherwise -> treat as a literal KinSim motif string

    Args:
        motifs_arg:    File path or motif string.
        min_fraction:  Minimum fraction threshold (PacBio CSV only).
        min_detected:  Minimum nDetected threshold (PacBio CSV only).
        parser_name:   Explicit parser name ("pacbio", "modkit", "ipd_summary").
                       When provided, bypasses auto-detection.

    Returns:
        A semicolon-delimited KinSim motif string.
    """
    # Explicit parser requested
    if parser_name is not None:
        from .parsers import create_parser

        parser = create_parser(parser_name)
        return parser.parse(motifs_arg, min_fraction=min_fraction, min_detected=min_detected)

    if os.path.isfile(motifs_arg):
        # Try the parser registry first (covers PacBio motifs.csv, modkit
        # bedMethyl, combined CSV).
        from .parsers import auto_detect_parser

        parser = auto_detect_parser(motifs_arg)
        if parser is not None:
            return parser.parse(motifs_arg, min_fraction=min_fraction, min_detected=min_detected)

        # No parser auto-matched → REBASE fallback. The legacy
        # ``parse_motifs_csv`` was retired in favour of PacBioParser: it
        # required the same ``motifString`` + ``centerPos`` columns that
        # PacBioParser checks, so any CSV that fell through here was not
        # a PacBio motifs.csv anyway.
        from .parsers.rebase import parse_rebase_file

        return parse_rebase_file(motifs_arg)

    return motifs_arg


# ---------------------------------------------------------------------------
# Reference-level methylation map (pre-scan entire genome once)
# ---------------------------------------------------------------------------


def build_reference_meth_map(ref_seqs, motif_string, revcomp=True, no_fuzznuc=False):
    """Pre-scan a reference genome for methylation sites.

    PRIMARY BACKEND: EMBOSS fuzznuc — tried first unless no_fuzznuc=True.
    Uses a single subprocess call with a named-pattern file (@patterns.txt),
    covering all motifs at once for efficiency and scientific reproducibility.

    FALLBACK: Python regex — used automatically if fuzznuc is not installed
    (prints a warning) or if no_fuzznuc=True.

    Scanning the reference once and caching results in a per-position array
    enables O(1) methylation lookup during read injection, regardless of
    whether fuzznuc or regex is used.

    Args:
        ref_seqs:     dict[name] -> sequence string (from load_reference).
        motif_string: KinSim motif string ("m6A,GATC,1;m4C,CCWGG,1").
        revcomp:      Also scan the reverse complement strand (default True).
        no_fuzznuc:   Force Python regex mode; skip fuzznuc entirely.

    Returns:
        dict[ref_name] -> np.int8 array of shape (ref_len,)
        Each position holds the methylation type ID (0 = unmethylated).
        For circular-genome lookups, index with pos % ref_len.
    """
    if not no_fuzznuc:
        try:
            meth_map = _build_meth_map_fuzznuc(ref_seqs, motif_string, revcomp)
            # Validate: if motifs were provided but fuzznuc found nothing,
            # fall back to regex (fuzznuc can silently produce empty results)
            total_hits = sum(int(np.count_nonzero(arr)) for arr in meth_map.values())
            if total_hits == 0 and motif_string:
                log.warning(
                    "fuzznuc returned 0 methylation sites — falling back to Python regex scanner"
                )
                return _build_meth_map_regex(ref_seqs, motif_string, revcomp)
            return meth_map
        except FileNotFoundError:
            log.warning("fuzznuc not found on PATH — falling back to Python regex scanner")
    return _build_meth_map_regex(ref_seqs, motif_string, revcomp)


def _build_meth_map_regex(ref_seqs, motif_string, revcomp=True):
    """Build reference methylation map using Python regex (fallback backend)."""
    motifs = parse_motifs(motif_string, revcomp=revcomp)
    return {name: scan_sequence(seq, motifs) for name, seq in ref_seqs.items()}


def build_reference_meth_map_per_strand(ref_seqs, motif_string):
    """Return ``(fwd_map, rev_map)`` — per-strand meth_id maps in forward coords.

    ``fwd_map[contig][p]`` is the meth_id at forward-strand position ``p`` if
    a forward-strand motif methylates there, else 0. ``rev_map[contig][p]``
    is the meth_id of the reverse-strand methylation at the same locus
    (forward-strand coordinates), else 0. The union of these two equals
    what :func:`build_reference_meth_map` returns with ``revcomp=True``.

    Needed by :mod:`kinsim.generate` to populate the ``rev_meth`` block of
    ``meth_full`` (complementary-strand methylation at the active-site
    neighbours) the same way :mod:`kinsim.extract` does at training time.
    Without this, palindromic motifs (e.g. m6A on both strands of GATC)
    lose their partner-strand signal at inference.
    """
    fwd_motifs, rev_motifs = parse_motifs_per_strand(motif_string)
    fwd_map: dict[str, np.ndarray] = {}
    rev_map: dict[str, np.ndarray] = {}
    for name, seq in ref_seqs.items():
        fwd_map[name] = scan_sequence(seq, fwd_motifs)
        rc_hits = scan_sequence(reverse_complement(seq), rev_motifs)
        # rc_hits is in rev-complement coordinates; flip to forward coords.
        rev_map[name] = rc_hits[::-1].copy()
    return fwd_map, rev_map


def build_reference_frac_map(ref_seqs, motif_string, revcomp=True):
    """Build per-position stoichiometric fraction map for the reference genome.

    Tags the fraction at the modification position (matches scan_sequence).

    Returns:
        dict[ref_name] -> np.float32 array of shape (ref_len,)
        Each position holds the stoichiometric fraction (0.0 = unmethylated).
    """
    motifs = parse_motifs(motif_string, revcomp=revcomp)
    frac_map = {}
    for name, seq in ref_seqs.items():
        fmap = np.zeros(len(seq), dtype=np.float32)
        for motif in motifs:
            for match in motif["pattern"].finditer(seq):
                target_pos = match.start() + motif["pos"]
                if 0 <= target_pos < len(seq):
                    fmap[target_pos] = motif["frac"]
        frac_map[name] = fmap
    return frac_map


def _build_meth_map_fuzznuc(ref_seqs, motif_string, revcomp=True):
    """Build reference methylation map using EMBOSS fuzznuc (primary backend).

    A single fuzznuc subprocess call scans all motifs at once using a named
    pattern file.  GFF output is parsed, and each hit's pattern name (from
    the attributes column) is decoded to retrieve meth_id and mod_pos.

    Strand-position arithmetic:
        + strand match at [Start, End] (1-based), modified pos p (0-based):
            meth_pos = (Start - 1) + p
        - strand match at [Start, End] (1-based), modified pos p (0-based):
            meth_pos = (End - 1) - p
        (End is 1-based inclusive; the - strand 5' corresponds to End on +)
    """
    from .parsers.rebase import write_fuzznuc_pattern_file

    if not motif_string:
        return {name: np.zeros(len(seq), dtype=np.int8) for name, seq in ref_seqs.items()}

    meth_map = {name: np.zeros(len(seq), dtype=np.int8) for name, seq in ref_seqs.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write reference FASTA
        ref_fa = os.path.join(tmpdir, "ref.fa")
        with open(ref_fa, "w") as fh:
            for name, seq in ref_seqs.items():
                fh.write(f">{name}\n{seq}\n")

        # Write named-pattern file and get lookup dict
        pattern_file = os.path.join(tmpdir, "patterns.txt")
        pattern_lookup = write_fuzznuc_pattern_file(motif_string, pattern_file)

        if not pattern_lookup:
            return meth_map

        out_gff = os.path.join(tmpdir, "hits.gff")
        cmd = [
            "fuzznuc",
            "-sequence",
            ref_fa,
            "-pattern",
            f"@{pattern_file}",
            "-pmismatch",
            "0",
            "-complement",
            "Y" if revcomp else "N",
            "-rformat",
            "gff",
            "-outfile",
            out_gff,
            "-auto",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.warning(
                "fuzznuc failed (exit %d): %s — falling back to Python regex scanner",
                result.returncode,
                result.stderr.strip(),
            )
            return _build_meth_map_regex(ref_seqs, motif_string, revcomp)

        if not os.path.exists(out_gff):
            log.warning("fuzznuc produced no output file — falling back to Python regex scanner")
            return _build_meth_map_regex(ref_seqs, motif_string, revcomp)

        # Parse GFF output: extract pattern name from attributes to identify motif
        with open(out_gff) as gff:
            for line in gff:
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.split("\t")
                if len(cols) < 7:
                    continue
                ref_name = cols[0]
                start_1b = int(cols[3])
                end_1b = int(cols[4])
                strand = cols[6]
                attrs = cols[8].strip() if len(cols) > 8 else ""

                if ref_name not in meth_map:
                    continue

                # Decode which motif this hit corresponds to
                meth_id, mod_pos = 0, 0
                attr_match = _GFF_ATTR_NAME_RE.search(attrs)
                if attr_match:
                    pname = attr_match.group(1).strip()
                    if pname in pattern_lookup:
                        meth_id, mod_pos = pattern_lookup[pname]
                    else:
                        # Try decode from name convention directly
                        from .parsers.rebase import decode_fuzznuc_pattern_name

                        meth_id, mod_pos = decode_fuzznuc_pattern_name(pname)

                if strand == "+":
                    meth_pos = (start_1b - 1) + mod_pos
                else:
                    meth_pos = (end_1b - 1) - mod_pos

                ref_len = len(ref_seqs[ref_name])
                if 0 <= meth_pos < ref_len:
                    meth_map[ref_name][meth_pos] = meth_id

    return meth_map


# ---------------------------------------------------------------------------
# CLI: kinsim motifs
# ---------------------------------------------------------------------------


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim motifs",
        description=(
            "Parse a motif source and print the KinSim motif string.\n\n"
            "Accepted inputs:\n"
            "  PacBio motifs.csv  — filtered by --min-fraction / --min-detected\n"
            "  REBASE file        — simplified two-column or Format #19 (withrefm)\n"
            "  Motif string       — pass directly as the 'input' argument\n\n"
            "Auto-detection: if the argument is a file ending in '.csv' it is\n"
            "treated as PacBio CSV; any other existing file is treated as REBASE;\n"
            "otherwise it is printed as-is after basic validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="PacBio motifs.csv, REBASE file, or KinSim motif string")
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.40,
        help="Minimum fraction threshold for PacBio CSV (default: 0.40)",
    )
    parser.add_argument(
        "--min-detected",
        type=int,
        default=20,
        help="Minimum nDetected threshold for PacBio CSV (default: 20)",
    )
    args = parser.parse_args(argv)

    result = load_motif_string(
        args.input, min_fraction=args.min_fraction, min_detected=args.min_detected
    )
    if result:
        print(result)
    else:
        log.error("No motifs found / passed the filter.")
        sys.exit(1)


if __name__ == "__main__":
    main()
