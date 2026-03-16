"""REBASE methylation file parsing and fuzznuc pattern file generation.

Supports two REBASE input formats:
  1. Simplified two-column format:
         GATC    2(6)
         CCWGG   2(5),-1(5)
  2. REBASE Format #19 (withrefm / allenz-style tagged records):
         ID   M.TaqI
         RS   TCGA, ?;
         MS   3(6mA);
         //

Also provides write_fuzznuc_pattern_file() which converts a KinSim motif
string into a named-pattern file for fuzznuc's '@file' syntax, enabling a
single subprocess call across all motifs.

REBASE X(Y) position notation:
    X = 1-based position (positive = forward strand,
                          negative = complementary strand, from its 5' end)
    Y = 6 (m6A), 5 (m5C), 4 (m4C)

REBASE Format #19 MS field uses position(type) where type is 6mA, N4mC, or 5mC.
"""

from __future__ import annotations

import logging
import os
import re
import sys

from kinsim.utils.encoding import METH_IDS

log = logging.getLogger(__name__)

# Regex to validate IUPAC-only recognition sequences
_IUPAC_RE = re.compile(r'^[ACGTRYSWKMBDHVN]+$')

# REBASE Y-code -> KinSim mod type (used in simple X(Y) notation)
_REBASE_CODE_TO_METH = {'6': 'm6A', '5': 'm5C', '4': 'm4C'}

# REBASE Format #19 MS type strings -> KinSim mod type
_REBASE_TYPE_TO_METH = {
    '6mA': 'm6A',
    '5mC': 'm5C',
    'N4mC': 'm4C',
}

# Regex for simplified two-column REBASE annotations: "2(6)" or "-1(4)"
_SITE_RE = re.compile(r'(-?\d+)\((\d)\)')

# Regex for Format #19 MS field annotations: "3(6mA)" or "-1(N4mC)"
_MS_SITE_RE = re.compile(r'(-?\d+)\((\w+)\)')


# ---------------------------------------------------------------------------
# X(Y) notation parser (shared between simple and Format #19)
# ---------------------------------------------------------------------------

def parse_rebase_annotation(recognition_seq, meth_annotation):
    """Parse a REBASE X(Y) methylation annotation into KinSim motif entries.

    REBASE methylation site notation:
        X(Y)  or  X1(Y1),X2(Y2)
    Where:
        X  = 1-based position within the recognition sequence (positive =
             forward strand; negative = complementary strand counted from
             its 5' end, i.e. from the 3' end of the top strand)
        Y  = modification type: 6 = m6A, 5 = m5C, 4 = m4C

    Conversion to KinSim 0-based position:
        positive X  ->  pos = X - 1
        negative X  ->  pos = len(recognition_seq) - abs(X)  (0-based, top strand)

    Returns:
        List of "MOD_TYPE,RECOGNITION_SEQ,POS" strings (no nDetected field).
    """
    recognition_seq = recognition_seq.strip().upper()
    seq_len = len(recognition_seq)
    entries = []

    for site_match in _SITE_RE.finditer(meth_annotation):
        x = int(site_match.group(1))
        y = site_match.group(2)

        meth_type = _REBASE_CODE_TO_METH.get(y)
        if meth_type is None:
            log.warning("REBASE: unknown methylation code (%s) -- skipped", y)
            continue

        if x > 0:
            pos_0 = x - 1            # 1-based -> 0-based
        else:
            pos_0 = seq_len - abs(x)  # from 5' of complementary strand

        if not (0 <= pos_0 < seq_len):
            log.warning("REBASE: position %d out of range for '%s' (len=%d) -- skipped",
                        x, recognition_seq, seq_len)
            continue

        entries.append(f"{meth_type},{recognition_seq},{pos_0}")
    return entries


# ---------------------------------------------------------------------------
# Simplified two-column REBASE format
# ---------------------------------------------------------------------------

def parse_rebase_simple(filepath):
    """Parse a simplified two-column REBASE tab-delimited file.

    Expects lines of the form:
        RECOGNITION_SEQUENCE    METHYLATION_SITES

    Example:
        GATC    2(6)
        CCWGG   2(5)
        GCWGC   2(6),-1(6)

    Lines beginning with '#' are comments and are skipped.
    Blank lines are skipped.

    Returns:
        A semicolon-delimited KinSim motif string.
    """
    all_entries = []
    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.split(r'\s+', line, maxsplit=1)
            if len(parts) < 2:
                log.warning("REBASE line %d: expected 2 columns, got %d -- skipped",
                            lineno, len(parts))
                continue
            rec_seq = parts[0].upper()
            meth_ann = parts[1]
            if not _IUPAC_RE.match(rec_seq):
                log.warning("REBASE line %d: invalid IUPAC sequence '%s' -- skipped",
                            lineno, rec_seq)
                continue
            entries = parse_rebase_annotation(rec_seq, meth_ann)
            all_entries.extend(entries)
    return ';'.join(all_entries)


# ---------------------------------------------------------------------------
# REBASE Format #19 (withrefm / allenz-style tagged records)
# ---------------------------------------------------------------------------

def parse_rebase_withrefm(filepath):
    """Parse a REBASE Format #19 file (withrefm or allenz-style).

    Record format:
        ID   enzyme_name
        ET   enzyme_type
        OS   source_organism
        RS   recognition_sequence, cut_site;
        MS   position(type)[,position(type)];
        //   end of record

    MS field position notation:
        position  = 1-based (positive = forward strand, negative = complementary)
        type      = 6mA | N4mC | 5mC

    Records are skipped when:
        - RS is '?' (unknown recognition sequence)
        - MS is absent or '?' (no methylation info)
        - RS contains characters other than IUPAC codes

    Returns:
        A semicolon-delimited KinSim motif string.
    """
    all_entries = []

    with open(filepath) as f:
        current = {}
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('//'):
                # End of record: process if we have RS and MS
                rec_seq = current.get('RS', '').strip()
                ms_raw  = current.get('MS', '').strip()

                if rec_seq and rec_seq != '?' and ms_raw and ms_raw != '?':
                    # Clean recognition sequence: remove cleavage site indicators
                    rec_clean = re.sub(r'[^ACGTRYMKSWHBVDN]', '', rec_seq.upper())
                    if rec_clean and not _IUPAC_RE.match(rec_clean):
                        log.warning("REBASE: invalid IUPAC in RS '%s' (enzyme %s) -- skipped",
                                    rec_seq, current.get('ID', '?'))
                        rec_clean = ''
                if rec_clean:
                        for site_match in _MS_SITE_RE.finditer(ms_raw):
                            x   = int(site_match.group(1))
                            typ = site_match.group(2)
                            meth_type = _REBASE_TYPE_TO_METH.get(typ)
                            if meth_type is None:
                                log.warning("REBASE: unknown MS type '%s' (enzyme %s) -- skipped",
                                            typ, current.get('ID', '?'))
                                continue

                            seq_len = len(rec_clean)
                            if x > 0:
                                pos_0 = x - 1
                            else:
                                pos_0 = seq_len - abs(x)

                            if 0 <= pos_0 < seq_len:
                                all_entries.append(
                                    f"{meth_type},{rec_clean},{pos_0}")

                current = {}
                continue

            # Tagged-field line: "ID   name" or "RS\tGATC, 2;"
            m = re.match(r'^(\w{2})\s{2,}(.+)', line)
            if not m and '\t' in line:
                m = re.match(r'^(\w{2})\t(.+)', line)
            if m:
                tag = m.group(1).strip()
                val = m.group(2).strip()
                if tag in ('ID', 'RS', 'MS', 'ET'):
                    current[tag] = val

    return ';'.join(all_entries)


# ---------------------------------------------------------------------------
# Auto-detecting REBASE file parser
# ---------------------------------------------------------------------------

def parse_rebase_file(filepath):
    """Auto-detect REBASE format and parse accordingly.

    Detection heuristic:
        - If the file contains lines starting with 'ID   ' or 'RS   '
          (three spaces after the two-char tag) -> Format #19 (withrefm)
        - Otherwise -> simplified two-column format

    Returns:
        A semicolon-delimited KinSim motif string.
    """
    with open(filepath) as f:
        for line in f:
            if line.startswith(('ID   ', 'RS   ', 'MS   ')):
                return parse_rebase_withrefm(filepath)
    return parse_rebase_simple(filepath)


# ---------------------------------------------------------------------------
# Fuzznuc pattern file generator
# ---------------------------------------------------------------------------

def write_fuzznuc_pattern_file(motif_string, filepath):
    """Convert a KinSim motif string to a fuzznuc named-pattern file.

    Writes a file in PROSITE/fuzznuc format with named patterns:
        >m6A_GATC_1
        GATC
        >m4C_CCWGG_1
        CCWGG

    Pattern names encode the motif type, sequence, and 0-based modified
    position, allowing GFF output to be linked back to meth_id and mod_pos
    without re-parsing the pattern file.

    Args:
        motif_string: KinSim semicolon-delimited motif string.
        filepath:     Path to write the pattern file.

    Returns:
        dict mapping pattern_name -> (meth_id, mod_pos)
        so the caller can look up each fuzznuc GFF hit.
    """
    pattern_lookup = {}  # name -> (meth_id, mod_pos)

    lines = []
    for entry in motif_string.split(';'):
        if not entry or ',' not in entry:
            continue
        parts = entry.split(',')
        m_type, seq, pos_str = parts[0], parts[1], parts[2]
        meth_id  = METH_IDS.get(m_type, 0)
        mod_pos  = int(pos_str)

        # Unique name encodes all fields needed for GFF lookup
        name = f"{m_type}_{seq}_{mod_pos}"

        # Deduplicate: if two entries produce the same name, keep first
        if name not in pattern_lookup:
            pattern_lookup[name] = (meth_id, mod_pos)
            lines.append(f">{name}\n{seq}")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return pattern_lookup


def decode_fuzznuc_pattern_name(name):
    """Decode a pattern name generated by write_fuzznuc_pattern_file.

    Inverse of the naming scheme: "{m_type}_{seq}_{mod_pos}"

    Returns (meth_id, mod_pos) or (0, 0) if decoding fails.
    """
    parts = name.split('_')
    if len(parts) < 3:
        return 0, 0
    try:
        mod_pos = int(parts[-1])
        m_type  = parts[0]
        meth_id = METH_IDS.get(m_type, 0)
        return meth_id, mod_pos
    except (ValueError, IndexError):
        return 0, 0


# ---------------------------------------------------------------------------
# Isoschizomer mapping (Format #19 only)
# ---------------------------------------------------------------------------

def parse_rebase_isoschizomers(filepath):
    """Parse a REBASE Format #19 file and group enzymes by recognition sequence.

    Returns a dict mapping each unique recognition sequence (cleaned, uppercase
    IUPAC) to a deduplicated list of enzyme names (ID fields) that recognise it.

    Only processes Format #19 files.  Returns an empty dict for simplified
    two-column files.

    Args:
        filepath: Path to a REBASE Format #19 file.

    Returns:
        dict[str, list[str]]: recognition_seq -> [enzyme_name, ...], no duplicates.
    """
    iso_map: dict[str, list[str]] = {}

    with open(filepath) as f:
        current: dict[str, str] = {}
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('//'):
                rec_seq = current.get('RS', '').strip()
                enz_id  = current.get('ID', '').strip()

                if rec_seq and rec_seq != '?' and enz_id:
                    rec_clean = re.sub(r'[^ACGTRYMKSWHBVDN]', '', rec_seq.upper())
                    if rec_clean and _IUPAC_RE.match(rec_clean):
                        if rec_clean not in iso_map:
                            iso_map[rec_clean] = []
                        if enz_id not in iso_map[rec_clean]:
                            iso_map[rec_clean].append(enz_id)

                current = {}
                continue

            m = re.match(r'^(\w{2})\s{2,}(.+)', line)
            if not m and '\t' in line:
                m = re.match(r'^(\w{2})\t(.+)', line)
            if m:
                tag = m.group(1).strip()
                val = m.group(2).strip()
                if tag in ('ID', 'RS', 'MS', 'ET'):
                    current[tag] = val

    return iso_map


# ---------------------------------------------------------------------------
# REBASE web fetch (kinsim-prep rebase fetch <org_num>)
# ---------------------------------------------------------------------------
#
# URL: https://rebase.neb.com/cgi-bin/pacbioget?<org_num>
#
# The page "Showing only Genuine Motifs" has:
#   - A color legend:  "Meth type colors: m4C  m5C  m6A  unknown"
#     Each meth-type label is wrapped in <font color="#RRGGBB">...</font>.
#   - A table "MTases active in the genome" with columns:
#     Enzymes | DNA | Locus | Type | Length | Motif | Count | Unique | Genuine
#             | % Detected | Coverage
#
# In the Motif cell, the modified base is a single letter inside
# <font color="#RRGGBB">X</font>; all other bases are plain text.
# The color matches the legend color for the corresponding methylation type.
#
# % Detected:
#   Palindromic motif  -> single value, e.g. "88.5"
#   Non-palindromic    -> two values,   e.g. "86.4/85.0"  (top / complementary)
#   We store the mean as `fraction`.
#
# nGenome is estimated as round(nDetected / fraction) when both are available.
# ---------------------------------------------------------------------------

# Fallback color map in case the page legend cannot be parsed.
# These are REBASE's historic colors as of 2024.
_REBASE_FALLBACK_COLORS: dict[str, str] = {
    # m6A -- blue variants
    '#1e90ff': 'm6A', '1e90ff': 'm6A',
    '#0000ff': 'm6A', '0000ff': 'm6A',
    '#4169e1': 'm6A', '4169e1': 'm6A',
    # m5C -- green variants
    '#008000': 'm5C', '008000': 'm5C',
    '#228b22': 'm5C', '228b22': 'm5C',
    '#006400': 'm5C', '006400': 'm5C',
    # m4C -- orange variants
    '#ff8c00': 'm4C', 'ff8c00': 'm4C',
    '#ffa500': 'm4C', 'ffa500': 'm4C',
    '#ff7f00': 'm4C', 'ff7f00': 'm4C',
}

# IUPAC validation for motif sequences extracted from HTML
_IUPAC_MOTIF_RE = re.compile(r'^[ACGTNRYSWKMBDHV]+$')


def _fetch_rebase_html(org_num: int) -> str:
    """Fetch the REBASE PacBio genome analysis page for *org_num*.

    URL: https://rebase.neb.com/cgi-bin/pacbioget?<org_num>

    Raises RuntimeError on network or HTTP errors.
    """
    import urllib.request
    import urllib.error

    url = f"https://rebase.neb.com/cgi-bin/pacbioget?{org_num}"
    log.info("Fetching REBASE page: %s", url)
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'KinSim/0.3.0 (github.com/NicolasDutilleux/KinSim)'},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"REBASE returned HTTP {e.code} for organism {org_num}. "
            "Check that the organism number is valid."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch REBASE page for organism {org_num}: {e}"
        ) from e

    if 'MTases active in the genome' not in html:
        raise RuntimeError(
            f"REBASE page for organism {org_num} does not contain expected "
            "'MTases active in the genome' section. "
            "The organism number may not have PacBio data."
        )
    return html


def _parse_color_legend(html: str) -> dict[str, str]:
    """Extract {color_string: meth_type} from the REBASE page legend.

    The legend looks like:
        Meth type colors: <font color="#FF8C00">m4C</font>
                          <font color="#008000">m5C</font>
                          <font color="#1E90FF">m6A</font> unknown
    """
    color_to_meth: dict[str, str] = {}
    # Scan only the first 4 KB -- the legend is always near the top
    snippet = html[:4096]
    pattern = re.compile(
        r'<font[^>]+color\s*=\s*["\']?([^"\'>\s]+)["\']?[^>]*>\s*(m6A|m5C|m4C)\s*</font>',
        re.IGNORECASE,
    )
    for m in pattern.finditer(snippet):
        color = m.group(1).strip()
        meth  = m.group(2)
        # Store both with and without leading '#' so lookup is forgiving
        color_to_meth[color.lower()] = meth
        stripped = color.lower().lstrip('#')
        color_to_meth[stripped] = meth
        color_to_meth['#' + stripped] = meth
    return color_to_meth


def _resolve_color(color: str, color_to_meth: dict[str, str]) -> str | None:
    """Resolve a raw color attribute string to a methylation type.

    Tries the color as-is, then lowercased, then with/without '#', then
    falls back to _REBASE_FALLBACK_COLORS.
    """
    for key in (color, color.lower(), color.lower().lstrip('#'),
                '#' + color.lower().lstrip('#')):
        result = color_to_meth.get(key) or _REBASE_FALLBACK_COLORS.get(key)
        if result:
            return result
    return None


def _parse_motif_cell(cell_html: str,
                      color_to_meth: dict[str, str]) -> list[tuple[str, int, str, str]]:
    """Parse a REBASE motif table cell.

    REBASE colors bases to indicate methylation:
      - A colored with m6A color → direct: m6A on this A (top strand)
      - T colored with m6A color → complement: m6A on the A of the RC strand
      - C colored with m5C color → direct: m5C on this C (top strand)
      - G colored with m5C color → complement: m5C on the C of the RC strand

    Returns:
        List of (motif_string, center_pos_0based, mod_type, strand) tuples.
        strand is 'top' (base matches mod type directly) or 'rc' (complement).
        Empty list if nothing parseable.
    """
    tag_re   = re.compile(r'<[^>]+>')
    font_re  = re.compile(
        r'<font[^>]+color\s*=\s*["\']?([^"\'>\s]+)["\']?[^>]*>([A-Za-z]+)</font>',
        re.IGNORECASE,
    )

    # Direct base and complement base for each mod type
    _DIRECT = {'m6A': 'A', 'm4C': 'C', 'm5C': 'C'}
    _COMPL  = {'m6A': 'T', 'm4C': 'G', 'm5C': 'G'}

    full_motif = tag_re.sub('', cell_html).strip().upper()
    if not full_motif or not _IUPAC_MOTIF_RE.match(full_motif):
        return []

    results: list[tuple[str, int, str, str]] = []
    for m in font_re.finditer(cell_html):
        color    = m.group(1).strip()
        base_str = m.group(2).strip().upper()
        mod_type = _resolve_color(color, color_to_meth)
        if mod_type is None:
            continue

        before_tag  = cell_html[:m.start()]
        text_before = tag_re.sub('', before_tag).strip()
        pos         = len(text_before)

        for char_offset, base_char in enumerate(base_str):
            actual_pos = pos + char_offset
            if actual_pos >= len(full_motif):
                continue

            if base_char == _DIRECT.get(mod_type):
                results.append((full_motif, actual_pos, mod_type, 'top'))
            elif base_char == _COMPL.get(mod_type):
                results.append((full_motif, actual_pos, mod_type, 'rc'))

    return results


def _find_table_end(html: str, table_start: int) -> int:
    """Return the index just past the </table> that closes the <table> at table_start."""
    depth = 0
    pos   = table_start
    lo    = html.lower()
    while pos < len(lo):
        next_open  = lo.find('<table',  pos)
        next_close = lo.find('</table>', pos)
        if next_close == -1:
            break
        if next_open != -1 and next_open < next_close:
            depth += 1
            pos = next_open + len('<table')
        else:
            depth -= 1
            pos = next_close + len('</table>')
            if depth == 0:
                return pos
    return len(html)


def _is_palindromic(motif: str) -> bool:
    """True if the IUPAC motif is its own reverse complement."""
    from kinsim.utils.motifs import reverse_complement
    return reverse_complement(motif.upper()) == motif.upper()


def _rc_offset(motif: str, offset: int) -> int:
    """Compute the 0-based offset of the methylated base on the RC strand.

    For a motif of length L with methylated base at position P:
    the complementary base is at position (L - 1 - P) in the RC string.
    """
    return len(motif) - 1 - offset


def _parse_active_mtases_table(html: str,
                                color_to_meth: dict[str, str]) -> list[dict]:
    """Parse the 'MTases active in the genome' table and return entry dicts.

    For non-palindromic motifs (% Detected shows two values like "86.4/85.0"),
    generates TWO entries: one for the top strand and one for the reverse
    complement, each with its own fraction and offset.

    For palindromic motifs (% Detected shows one value), generates one entry.
    """
    from .motif_merge import _make_entry
    from kinsim.utils.motifs import reverse_complement

    marker = 'MTases active in the genome'
    idx = html.find(marker)
    if idx == -1:
        return []

    # Walk back to the nearest <table> that contains the marker
    table_start = html.lower().rfind('<table', 0, idx)
    if table_start == -1:
        log.warning("REBASE HTML: <table> not found before active MTases section")
        return []

    table_end  = _find_table_end(html, table_start)
    table_html = html[table_start:table_end]

    tag_re  = re.compile(r'<[^>]+>')
    row_re  = re.compile(r'<tr[^>]*>(.*?)</tr>',       re.IGNORECASE | re.DOTALL)
    cell_re = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.IGNORECASE | re.DOTALL)

    entries: list[dict] = []
    header_found = False

    # Default column indices (guard against header parsing failure)
    col_motif    = 5
    col_count    = 6
    col_pct      = 9
    col_coverage = 10

    for row_m in row_re.finditer(table_html):
        row_html = row_m.group(1)
        cells = [c.group(1) for c in cell_re.finditer(row_html)]
        if not cells:
            continue

        texts = [tag_re.sub('', c).strip() for c in cells]

        # Identify header row by presence of 'Motif' column
        if not header_found and 'Motif' in texts:
            for i, t in enumerate(texts):
                if t == 'Motif':         col_motif    = i
                elif t == 'Count':       col_count    = i
                elif 'Detected' in t:    col_pct      = i
                elif t == 'Coverage':    col_coverage = i
            header_found = True
            continue

        if not header_found or col_motif >= len(cells):
            continue

        motif_plain = texts[col_motif]
        if not motif_plain or not _IUPAC_MOTIF_RE.match(motif_plain.upper()):
            continue

        hits = _parse_motif_cell(cells[col_motif], color_to_meth)
        if not hits:
            log.warning("REBASE HTML: could not parse motif cell '%s'", motif_plain)
            continue

        # Use the first hit to get motif_str and mod_type
        motif_str = hits[0][0]
        mod_type  = hits[0][2]

        # Count -> nDetected
        n_detected: int | str = ''
        if col_count < len(cells):
            try:
                n_detected = int(texts[col_count].replace(',', ''))
            except (ValueError, AttributeError):
                pass

        # % Detected -> parse top/bottom fractions separately
        pct_parts: list[float] = []
        if col_pct < len(cells):
            raw = texts[col_pct].split('/')
            try:
                pct_parts = [float(p.strip()) / 100.0 for p in raw if p.strip()]
            except ValueError:
                pass

        is_palindrome = _is_palindromic(motif_str)

        # Coverage -> meanCoverage
        mean_coverage: float | str = ''
        if col_coverage < len(cells):
            cov_parts = texts[col_coverage].split('/')
            try:
                mean_coverage = round(
                    sum(float(p.strip()) for p in cov_parts if p.strip()) / len(cov_parts),
                    1,
                )
            except (ValueError, ZeroDivisionError):
                pass

        rc_motif = reverse_complement(motif_str)

        # Separate hits by strand
        top_hits = [h for h in hits if h[3] == 'top']
        rc_hits  = [h for h in hits if h[3] == 'rc']

        # ---- Top strand entry ----
        if top_hits:
            center_pos = top_hits[0][1]   # 0-based from HTML
            center_pos_1b = center_pos + 1
            frac_top = round(pct_parts[0], 7) if pct_parts else ''
            n_genome_top: int | str = ''
            if isinstance(n_detected, int) and isinstance(frac_top, float) and frac_top > 0:
                n_genome_top = round(n_detected / frac_top)

            entries.append(_make_entry(
                motif_str=motif_str,
                offset=center_pos_1b,
                mod_type=mod_type,
                fraction=frac_top,
                n_detected=n_detected,
                n_genome=n_genome_top,
                mean_coverage=mean_coverage,
                source='rebase',
            ))
            log.info("  [TOP]  %s offset=%d %s  frac=%.3f  palindromic=%s",
                     motif_str, center_pos_1b, mod_type,
                     frac_top if isinstance(frac_top, float) else 0,
                     is_palindrome)

        # ---- RC strand entry (from complement-colored bases) ----
        if rc_hits and not is_palindrome:
            # A complement-colored base (T for m6A, G for m5C) at position P
            # in the top motif means the RC motif has the methylated base at
            # position (len - 1 - P), which IS the correct base (A or C).
            rc_pos_0b = len(motif_str) - 1 - rc_hits[0][1]
            rc_pos_1b = rc_pos_0b + 1
            frac_bot = round(pct_parts[1], 7) if len(pct_parts) >= 2 else ''
            n_genome_bot: int | str = ''
            if isinstance(n_detected, int) and isinstance(frac_bot, float) and frac_bot > 0:
                n_genome_bot = round(n_detected / frac_bot)

            entries.append(_make_entry(
                motif_str=rc_motif,
                offset=rc_pos_1b,
                mod_type=mod_type,
                fraction=frac_bot,
                n_detected=n_detected,
                n_genome=n_genome_bot,
                mean_coverage=mean_coverage,
                source='rebase',
            ))
            log.info("  [RC]   %s offset=%d %s  frac=%.3f  (complement of %s)",
                     rc_motif, rc_pos_1b, mod_type,
                     frac_bot if isinstance(frac_bot, float) else 0,
                     motif_str)

        # ---- Fallback: no top hit but we have a single colored base ----
        # (shouldn't happen normally, but handles edge cases)
        if not top_hits and not rc_hits:
            log.warning("REBASE HTML: no valid direct or complement hits for '%s'",
                        motif_plain)

    # Deduplicate: same (motif, offset, mod_type) from multiple table rows
    seen: dict[tuple, dict] = {}
    for e in entries:
        key = (e['motif'], e['offset'], e['mod_type'])
        if key in seen:
            log.info("  [DEDUP] duplicate entry %s %s offset=%d -- keeping first",
                     e['mod_type'], e['motif'], e['offset'])
        else:
            seen[key] = e
    entries = list(seen.values())

    return entries


def fetch_rebase_org(org_num: int, output_path: str) -> list[dict]:
    """Fetch REBASE PacBio data for *org_num* and write standard PacBio CSV.

    Steps:
        1. Fetch https://rebase.neb.com/cgi-bin/pacbioget?<org_num>
        2. Extract methylation type color legend from the page header.
        3. Parse the 'MTases active in the genome' table.
        4. Write a standard 12-column PacBio motifs.csv to *output_path*.

    Args:
        org_num:     REBASE organism number (integer shown as 'Org#' on page).
        output_path: Destination CSV file path.

    Returns:
        List of entry dicts written (useful for logging / testing).
    """
    from .motif_merge import write_pacbio_motifs_csv

    html          = _fetch_rebase_html(org_num)
    color_to_meth = _parse_color_legend(html)

    if not color_to_meth:
        log.warning(
            "Could not extract color legend from REBASE page -- "
            "using fallback color map.  Colors may be wrong if REBASE changed."
        )
        color_to_meth = {}   # _resolve_color() will hit the fallback dict

    entries = _parse_active_mtases_table(html, color_to_meth)

    if not entries:
        raise RuntimeError(
            f"No active MTases with parseable motifs found for organism {org_num}. "
            "Check that the organism has genuine PacBio motifs on REBASE."
        )

    write_pacbio_motifs_csv(entries, output_path)
    return entries


# ---------------------------------------------------------------------------
# CLI: kinsim-prep rebase
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim-prep rebase",
        description=(
            "Parse REBASE files and generate fuzznuc pattern files.\n\n"
            "Accepted REBASE formats:\n"
            "  Simplified two-column   : RECOGNITION_SEQ  X(Y)[,X2(Y2)]\n"
            "  Format #19 (withrefm)   : tagged records (ID/RS/MS fields)\n"
            "                           Auto-detected from file content.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- parse subcommand --
    p_parse = sub.add_parser(
        "parse",
        help="Parse a REBASE file to KinSim string or standard PacBio motifs.csv",
        description=(
            "Parse a REBASE file (auto-detects simplified or Format #19).\n\n"
            "Default: print the KinSim motif string to stdout.\n\n"
            "With --output-csv: write a standard PacBio motifs.csv instead.\n"
            "This file is named 'rebase_motifs.csv' by default and can be\n"
            "merged with calling-derived motifs via:\n\n"
            "    kinsim-prep merge-motifs species_motifs.csv rebase_motifs.csv \\\n"
            "        --output final_motifs.csv\n"
        ),
    )
    p_parse.add_argument("input",
                         help="REBASE file (simplified two-column or Format #19)")
    p_parse.add_argument(
        "--output-csv", metavar="FILE", nargs="?", const="rebase_motifs.csv",
        default=None,
        help=(
            "Write standard PacBio motifs.csv instead of printing to stdout. "
            "FILE defaults to 'rebase_motifs.csv' when the flag is given "
            "without a value."
        ),
    )

    # -- fetch subcommand --
    p_fetch = sub.add_parser(
        "fetch",
        help="Fetch motifs for an organism directly from REBASE by organism number",
        description=(
            "Fetch the REBASE PacBio Genome Analysis page for an organism and\n"
            "write a standard PacBio motifs.csv (rebase_motifs.csv by default).\n\n"
            "URL used: https://rebase.neb.com/cgi-bin/pacbioget?<org_num>\n\n"
            "The organism number (Org#) is shown on REBASE organism pages.\n"
            "Parses only 'MTases active in the genome' (genuine motifs).\n\n"
            "Example:\n"
            "  kinsim-prep rebase fetch 1260 --output Ecoli_rebase.csv\n"
            "  kinsim-prep rebase fetch 1260   # writes rebase_motifs.csv\n"
        ),
    )
    p_fetch.add_argument(
        "org_num", type=int,
        help="REBASE organism number (the 'Org#' integer on REBASE pages)",
    )
    p_fetch.add_argument(
        "--output", "-o", metavar="FILE", default="rebase_motifs.csv",
        help="Output CSV file (default: rebase_motifs.csv)",
    )

    # -- patterns subcommand --
    p_patt = sub.add_parser(
        "patterns",
        help="Convert a motif source to a fuzznuc pattern file",
        description=(
            "Convert a motif source (KinSim string, PacBio CSV, or REBASE file)\n"
            "to a fuzznuc-compatible named pattern file.\n\n"
            "The output can be used with: fuzznuc -pattern @<output> ..."
        ),
    )
    p_patt.add_argument("motifs",
                        help="Motif source: KinSim string, REBASE file, or PacBio CSV")
    p_patt.add_argument("output",
                        help="Output fuzznuc pattern file")
    p_patt.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum fraction threshold (PacBio CSV only, default: 0.40)")
    p_patt.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold (PacBio CSV only, default: 20)")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        from kinsim.utils.config import setup_logging
        setup_logging(verbose=getattr(args, 'verbose', False))
        try:
            entries = fetch_rebase_org(args.org_num, args.output)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Wrote {len(entries)} motifs to {args.output}")

    elif args.command == "parse":
        result = parse_rebase_file(args.input)
        if not result:
            print("No motifs found in the REBASE file.", file=sys.stderr)
            sys.exit(1)

        if getattr(args, 'output_csv', None) is not None:
            # Write standard PacBio motifs.csv (rebase_motifs.csv)
            from .motif_merge import write_pacbio_motifs_csv, _make_entry
            entries = []
            for part in result.split(';'):
                if not part:
                    continue
                fields = part.split(',')
                if len(fields) < 3:
                    continue
                mod_type, motif, pos_str = fields[0], fields[1], fields[2]
                try:
                    offset = int(pos_str)
                except ValueError:
                    continue
                entries.append(_make_entry(
                    motif_str=motif,
                    offset=offset,
                    mod_type=mod_type,
                    fraction=1.0,   # Restriction enzymes: 100% methylation
                    source='rebase',
                ))
            output_file = args.output_csv
            write_pacbio_motifs_csv(entries, output_file)
            print(f"Wrote {len(entries)} REBASE motifs to {output_file}")
        else:
            print(result)

    elif args.command == "patterns":
        from kinsim.utils.motifs import load_motif_string
        motif_string = load_motif_string(args.motifs,
                                         min_fraction=args.min_fraction,
                                         min_detected=args.min_detected)
        if not motif_string:
            print("ERROR: no motifs found from the provided source.", file=sys.stderr)
            sys.exit(1)
        lookup = write_fuzznuc_pattern_file(motif_string, args.output)
        print(f"Pattern file written to {args.output} ({len(lookup)} patterns)")
        for name, (mid, pos) in lookup.items():
            print(f"  {name}  ->  meth_id={mid} mod_pos={pos}")


if __name__ == "__main__":
    main()
