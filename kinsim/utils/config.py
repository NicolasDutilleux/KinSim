"""KinSim configuration helpers: manifest CSV loading and YAML config support.

Manifest CSV
------------
A manifest CSV describes the set of real PacBio BAMs to extract kinetic
training data from.  It replaces the alternating-line text file previously
used by ``kinsim prepare``.

Format (CSV, comma-separated, with header):

    sample_id,bam_path,motifs
    strain1,/data/bams/strain1.bam,"m6A,GATC,1"
    strain2,/data/bams/strain2.bam,"m6A,GATC,1;m4C,CCWGG,1"
    strain3,/data/bams/strain3.bam,/data/motifs/strain3.csv

Column semantics:

    sample_id  — Unique identifier for this BAM.  Used as the output shard
                 filename prefix: ``shards/<sample_id>_shard.pkl``.
    bam_path   — Absolute path to a BAM file with fi/fp kinetic tags.
    motifs     — Resolved KinSim motif string OR path to a PacBio motifs.csv
                 or REBASE file (auto-detected by ``load_motif_string``).

The ``motifs`` field may contain commas (e.g. ``m6A,GATC,1``) and is handled
correctly by standard CSV quoting.

YAML Training Config
--------------------
A YAML config file can pin all training hyperparameters for reproducibility.
Pass it with ``kinsim train --model mlp --config config.yaml``.

    # config_mlp.yaml
    pkl:           /data/master.pkl
    output_dir:    /data/checkpoints_mlp/
    epochs:        50
    batch_size:    4096
    lr:            0.001
    loss:          gnll
    kmer_embed_dim: 64
    hidden_dim:    128
    meth_proj_dim: 8

Logging
-------
KinSim uses Python's standard ``logging`` module throughout.  Call
``setup_logging()`` once in the entry point (``__main__.py`` or the CLI
``main()`` of each module).  This sets up a timestamp format appropriate
for SLURM log files.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manifest CSV
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {"sample_id", "bam_path", "motifs"}
_OPTIONAL_COLUMNS = {"ref_path"}  # added: reference FASTA for orientation-aware extract


@dataclass
class SampleEntry:
    """One BAM + motifs + reference entry from a manifest CSV.

    ``ref_path`` is required by :mod:`kinsim.extract` — the orientation-aware
    pipeline needs the reference FASTA the aligned BAM was mapped against.
    """

    sample_id: str
    bam_path: str
    motifs: str  # KinSim motif string or path (resolved later by load_motif_string)
    ref_path: str = ""  # required — reference FASTA for the aligned extract path


def load_manifest(manifest_path: str) -> list[SampleEntry]:
    """Load a manifest CSV and return a list of sample entries.

    The manifest must be a comma-separated file with the header:

        sample_id,bam_path,motifs,ref_path

    ``ref_path`` is the reference FASTA the aligned BAM was mapped against.
    :mod:`kinsim.extract` requires it to build per-strand methylation maps
    and route ``read.is_reverse`` lookups correctly.

    Any column order is accepted.  Empty rows and rows starting with ``#``
    are silently skipped.

    Args:
        manifest_path: Path to the manifest CSV file.

    Returns:
        List of :class:`SampleEntry` objects (one per non-empty data row).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or any field is empty.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    entries: list[SampleEntry] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate header
        fieldnames = set(reader.fieldnames or [])
        missing = _REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"Manifest CSV is missing required columns: {missing}\n"
                f"Required: {_REQUIRED_COLUMNS}\n"
                f"Optional: {_OPTIONAL_COLUMNS}\n"
                f"Found:    {fieldnames}\n"
                f"File:     {manifest_path}"
            )

        has_ref = "ref_path" in fieldnames

        for row_num, row in enumerate(reader, start=2):
            # Skip comment rows
            raw_first = next(iter(row.values())).strip()
            if raw_first.startswith("#"):
                continue
            # Skip entirely empty rows
            if not any(v.strip() for v in row.values()):
                continue

            sample_id = row["sample_id"].strip()
            bam_path = row["bam_path"].strip()
            motifs = row["motifs"].strip()
            ref_path = row["ref_path"].strip() if has_ref else ""

            if not sample_id:
                raise ValueError(f"Empty 'sample_id' at row {row_num} in {manifest_path}")
            if not bam_path:
                raise ValueError(f"Empty 'bam_path'  at row {row_num} in {manifest_path}")
            if not motifs:
                raise ValueError(f"Empty 'motifs' at row {row_num} in {manifest_path}.")

            entries.append(
                SampleEntry(
                    sample_id=sample_id,
                    bam_path=bam_path,
                    motifs=motifs,
                    ref_path=ref_path,
                )
            )

    if not entries:
        raise ValueError(f"Manifest is empty (no data rows): {manifest_path}")

    n_aligned = sum(1 for e in entries if e.ref_path)
    log.info(
        "Loaded %d samples from manifest: %s (%d with ref_path → aligned extract path)",
        len(entries), manifest_path, n_aligned,
    )
    return entries


def validate_manifest(
    entries: list[SampleEntry],
    check_files: bool = True,
) -> list[str]:
    """Validate a loaded manifest and return a list of error strings.

    Checks performed:
    1. Duplicate ``sample_id`` values (always checked).
    2. ``bam_path`` file existence (when ``check_files=True``).
    3. ``motifs`` file existence when the field looks like a path (starts with
       ``/``, ``~``, or ends with ``.csv``/``.tsv``/``.txt``).

    Args:
        entries:     List of :class:`SampleEntry` from :func:`load_manifest`.
        check_files: If True (default), verify that BAM and motif files exist
                     on disk.  Set to False for a quick structural check only.

    Returns:
        List of error message strings.  Empty list means the manifest is valid.
    """
    errors: list[str] = []

    # 1. Duplicate sample_ids
    seen_ids: dict[str, int] = {}
    for idx, entry in enumerate(entries, start=1):
        if entry.sample_id in seen_ids:
            errors.append(
                f"Duplicate sample_id '{entry.sample_id}' at rows "
                f"{seen_ids[entry.sample_id]} and {idx}"
            )
        else:
            seen_ids[entry.sample_id] = idx

    if check_files:
        for idx, entry in enumerate(entries, start=1):
            # 2. BAM existence
            if not Path(entry.bam_path).exists():
                errors.append(
                    f"Row {idx} ({entry.sample_id}): bam_path does not exist: {entry.bam_path}"
                )
            # 3. Motif file existence (only when field looks like a path)
            motif_looks_like_path = (
                entry.motifs.startswith("/")
                or entry.motifs.startswith("~")
                or entry.motifs.startswith("./")
                or any(entry.motifs.endswith(ext) for ext in (".csv", ".tsv", ".txt"))
            )
            if motif_looks_like_path and not Path(entry.motifs).expanduser().exists():
                errors.append(
                    f"Row {idx} ({entry.sample_id}): motifs file does not exist: {entry.motifs}"
                )

    return errors


# ---------------------------------------------------------------------------
# YAML training config
# ---------------------------------------------------------------------------


def load_yaml_config(path: str) -> dict:
    """Load a YAML training config and return it as a plain dict.

    PyYAML must be installed (``pip install pyyaml``).  The config file must
    be a YAML mapping at the top level.

    Args:
        path: Path to the ``.yaml`` config file.

    Returns:
        Dict of config key-value pairs (strings, ints, floats as parsed by
        PyYAML).

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If the file does not exist.
        ValueError: If the top-level YAML document is not a mapping.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for --config support.\n"
            "Install with:  pip install pyyaml\n"
            "Or:            conda install pyyaml"
        ) from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config file must be a YAML mapping (dict) at the top level, "
            f"got {type(cfg).__name__}: {path}"
        )

    log.info("Loaded %d config keys from %s", len(cfg), path)
    return cfg


# ---------------------------------------------------------------------------
# Project-wide config (kinsim_config.yaml at repo root)
# ---------------------------------------------------------------------------

# Default config — used if kinsim_config.yaml is missing or unreadable.
# Numbers match the YAML at the repo root; keep in sync.
_DEFAULT_KINSIM_CONFIG = {
    "kinetic_signatures": {
        "m6A": {"modified_base": "A", "signal_offsets": [0, 5]},
        "m4C": {"modified_base": "C", "signal_offsets": [0]},
        "m5C": {"modified_base": "C", "signal_offsets": [2, 6]},
    },
    "meth_context": {"left": 7, "right": 3},
    "kinetic_profile": {"start": 0, "end": 8},
    "extract": {
        "n_baseline_per_kmer": 50,
        "baseline_min_dist_to_meth": 11,
        "baseline_sample_rate": 0.10,
        "near_meth_max_dist": 7,
    },
    "refine": {
        "slowed_split": {"secondary_percentile": 95},
    },
}

_CACHED_KINSIM_CONFIG: dict | None = None


def load_kinsim_config(explicit_path: str | None = None) -> dict:
    """Load the project-wide kinsim_config.yaml.

    Search order:
      1. ``explicit_path`` if given.
      2. ``$KINSIM_CONFIG`` environment variable.
      3. ``kinsim_config.yaml`` at the repo root (parent of the kinsim package).
      4. Built-in defaults (returned with a warning).

    Returns the parsed dict; caches the result so repeated calls are cheap.
    """
    global _CACHED_KINSIM_CONFIG
    if explicit_path is None and _CACHED_KINSIM_CONFIG is not None:
        return _CACHED_KINSIM_CONFIG

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    env_path = os.environ.get("KINSIM_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    pkg_root = Path(__file__).resolve().parents[2]
    candidates.append(pkg_root / "kinsim_config.yaml")

    for cand in candidates:
        if cand.exists():
            try:
                cfg = load_yaml_config(str(cand))
                _CACHED_KINSIM_CONFIG = cfg
                return cfg
            except Exception as exc:
                log.warning("Failed to load kinsim config %s: %s", cand, exc)

    log.warning("kinsim_config.yaml not found — using built-in defaults")
    _CACHED_KINSIM_CONFIG = _DEFAULT_KINSIM_CONFIG
    return _DEFAULT_KINSIM_CONFIG


def _get_meth_entry(meth_name: str) -> dict:
    """Return the YAML entry for ``meth_name`` or raise with a helpful message.

    Single source of truth: every helper that asks the config for a
    per-type field (signal_offsets, modified_base, …) goes through here
    so the error message is consistent and lists what IS declared.
    """
    cfg = load_kinsim_config()
    sigs = (cfg.get("kinetic_signatures") or {}).get(meth_name)
    if sigs is None:
        declared = sorted((cfg.get("kinetic_signatures") or {}).keys())
        raise ValueError(
            f"kinsim_config.yaml is missing 'kinetic_signatures.{meth_name}'. "
            f"Declared types: {declared}. Add an entry like:\n"
            f"  kinetic_signatures:\n"
            f"    {meth_name}:\n"
            f"      modified_base:  A      # one of A/C/G/T — base this mod sits on\n"
            f"      signal_offsets: [0]    # adjust for the actual biology of this mod\n"
            f"and re-run."
        )
    return sigs


def get_signature_offsets(meth_name: str) -> list[int]:
    """Return the signature offsets for a methylation type, from kinsim_config.yaml.

    Raises ValueError if ``meth_name`` is not declared in the config.
    The previous behaviour was to silently return ``[0]`` — correct only
    for m4C, silently wrong for m5C (real signal is +2/+6) and incomplete
    for m6A (misses the +5 footprint).
    """
    return list(_get_meth_entry(meth_name).get("signal_offsets", [0]))


def get_modified_base(meth_name: str) -> str:
    """Return the concrete base (A/C/G/T) that ``meth_name`` modifies.

    Reads ``kinetic_signatures.<meth_name>.modified_base`` from
    kinsim_config.yaml. This is the single, code-free way to declare
    the chemistry of a new modification: any module that needs to know
    "what base does m6A sit on" goes through here, no hardcoding.

    Raises ValueError on missing entry, missing field, or an invalid
    base (anything outside A/C/G/T).
    """
    entry = _get_meth_entry(meth_name)
    base = entry.get("modified_base")
    if base is None:
        raise ValueError(
            f"kinsim_config.yaml: kinetic_signatures.{meth_name} is missing "
            f"the required 'modified_base' field. Set it to the concrete base "
            f"(A/C/G/T) this modification sits on, e.g. 'modified_base: A' for m6A."
        )
    base = str(base).upper()
    if base not in ("A", "C", "G", "T"):
        raise ValueError(
            f"kinsim_config.yaml: kinetic_signatures.{meth_name}.modified_base "
            f"must be one of A/C/G/T, got '{base}'."
        )
    return base


def get_modified_base_map() -> dict[str, str]:
    """Return ``{meth_name: modified_base}`` for every declared meth type.

    Used by parsers (PacBioParser, validators) that need to know the
    chemistry of every modification at once, without hardcoding
    ``{"m6A": "A", "m4C": "C", "m5C": "C"}`` in source.
    """
    cfg = load_kinsim_config()
    out: dict[str, str] = {}
    for mname in (cfg.get("kinetic_signatures") or {}):
        out[mname] = get_modified_base(mname)
    return out


def get_meth_alias_map() -> dict[str, str]:
    """Return ``{alias: canonical_meth_name}`` for every declared meth type.

    Built from kinsim_config.yaml's ``aliases`` field per meth-type
    entry. Each canonical name maps to itself, plus any alternative
    names declared. Lets upstream callers (modkit, fibertools, hand-
    edited combined CSVs) write whichever convention they use ('5mC',
    '5-mC', 'm5C', etc.) and KinSim normalises to the canonical
    ``meth_name`` (the dict key in ``kinetic_signatures``).

    Example output for the default YAML::

        {'m6A': 'm6A', '6mA': 'm6A', '6-mA': 'm6A',
         'm4C': 'm4C', '4mC': 'm4C', '4-mC': 'm4C',
         'm5C': 'm5C', '5mC': 'm5C', '5-mC': 'm5C', '5hmC': 'm5C'}

    Adding a new alias requires only a YAML edit — never a code change.
    """
    cfg = load_kinsim_config()
    sigs = cfg.get("kinetic_signatures") or {}
    out: dict[str, str] = {}
    for canonical, entry in sigs.items():
        out[canonical] = canonical
        for alias in entry.get("aliases") or []:
            out[str(alias)] = canonical
    return out


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger for KinSim CLI runs.

    Should be called once, early in ``main()``.  Uses a timestamp format
    suited to SLURM log files so each line is independently parseable.

    Format example::

        2026-03-03 14:32:01 [INFO]    kinsim.common.extract: Extracting strain1.bam
        2026-03-03 14:32:05 [WARNING] kinsim.prepare: No motifs found for strain7

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
