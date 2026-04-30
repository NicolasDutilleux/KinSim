# `prep/` — Data preparation

Read-only data preparation utilities used to produce the inputs of the
`kinsim` ML pipeline:

1. Parse methylation caller outputs into a unified motif format
2. Merge / dedup motifs across multiple callers
3. Build / validate / inspect manifest CSVs
4. Filter and balance `.pkl` libraries before training

This package exposes the `kinsim-prep` CLI (entry point in
`pyproject.toml`).

For project-level overview see the [top-level README](../README.md).

---

## CLI subcommands

| Command | Module | What it does |
|---|---|---|
| `kinsim-prep parse` | `kinsim/utils/motifs.py` | Unified motif parser (auto-detects format) |
| `kinsim-prep rebase` | `prep/rebase.py` | REBASE web fetch + file parsing + fuzznuc patterns |
| `kinsim-prep merge-motifs` | `prep/motif_merge.py` | Merge / dedup motifs across callers, threshold-filtered |
| `kinsim-prep manifest` | `prep/manifest.py` | Manifest CSV count / validate / list |
| `kinsim-prep filter` | `prep/filter.py` | Filter `.pkl` by coverage / mod type / max keys |
| `kinsim-prep balance` | `prep/balance.py` | Per-meth-type balanced subset for fair training |

---

## Caller parser plugin registry

Every methylation caller emits its own output format. The `prep/callers/`
package provides a plugin registry so that any consumer (extract,
generate, etc.) can call the same `parse()` interface and get a unified
KinSim motif string back.

```python
from prep.callers import create_parser, list_parsers, auto_detect_parser

parser = create_parser("pacbio")        # explicit
parser = auto_detect_parser("file.csv") # auto-detect from content
motif_string = parser.parse("file.csv", min_fraction=0.40, min_detected=20)

list_parsers()  # ['combined', 'ipd_summary', 'modkit', 'pacbio']
```

| Parser | Format | Required columns |
|---|---|---|
| `pacbio` | PacBio motifs.csv (motifmaker output) | `motifString`, `centerPos` (+ optional `modificationType`, `fraction`, `nDetected`) |
| `modkit` | modkit pileup `--bedMethyl` TSV | 11+ columns, score = methylation fraction |
| `ipd_summary` | ipdSummary CSV / GFF3 | auto-detect by extension |
| `combined` | merged-callers CSV | `mod_type, motif, offset, frac_mod, n_sites, source` |

### Adding a new parser

Drop a file in `prep/callers/<name>.py`:

```python
from .base import BaseOutputParser
from .registry import register

@register
class MyCallerParser(BaseOutputParser):
    name: str = "my_caller"
    supported_mods: list[str] = ["m6A", "m5C"]

    def parse(self, filepath: str, min_fraction: float = 0.40,
              min_detected: int = 20) -> str:
        ...

    @classmethod
    def is_file_for_this_parser(cls, filepath: str) -> bool:
        ...
```

Add `from . import my_caller` to `prep/callers/__init__.py` to trigger
registration. The parser is then available via `create_parser("my_caller")`
and joins the auto-detection chain.

---

## Motif merging

`prep/motif_merge.py` combines motifs from multiple callers into one
PacBio-style motifs.csv suitable for `kinsim extract`. Filters at a
configurable threshold (default `frac_mod >= 0.7`) and deduplicates by
`(motif, offset)`.

```bash
kinsim-prep merge-motifs species_motifs.csv rebase_motifs.csv \
    --output final_motifs.csv \
    --threshold 0.7
```

---

## Manifest tools

```bash
kinsim-prep manifest count <csv>       # prints integer for SLURM --array
kinsim-prep manifest validate <csv>    # checks duplicates, file existence
kinsim-prep manifest list <csv>        # tabular display
```

The manifest schema:

```csv
sample_id,bam_path,motifs
HMB-10,/data/pacbio/HMB-10_bystrandify.bam,/data/calls/HMB-10_motifs_merged.csv
```

Three columns. `sample_id` is also used as the shard filename
(`shards/<sample_id>_shard.pkl`).

---

## Library filtering & balancing

### `kinsim-prep filter`

Filter a `.pkl` by coverage / mod type / max keys:

```bash
kinsim-prep filter master_clean.pkl training_data.pkl \
    --min-coverage 50 \
    --mod-type m6A,m4C \
    --max-keys 200000
```

### `kinsim-prep balance`

Methylated keys are far less numerous than unmethylated. Balance forces a
fair mix for training:

```bash
kinsim-prep balance master_clean.pkl balanced.pkl \
    --meth-fraction 0.5 \
    --samples-per-key 200
```

`--samples-per-key` uses IPD-quantile diversity selection (not random
subsampling) so the kept samples span the full distribution.

---

## REBASE integration

REBASE is the public database of restriction-modification systems. Useful
when motifmaker / ipdSummary disagree or when you have a reference
methylase known from a related strain.

```bash
kinsim-prep rebase fetch <org_num>            # fetch from REBASE website
kinsim-prep rebase parse <file>               # parse local REBASE file
kinsim-prep rebase patterns <motifs> <out>    # write fuzznuc pattern file
```

---

## Module layout

```
prep/
├── __init__.py
├── __main__.py              — CLI router for kinsim-prep
│
├── rebase.py                — REBASE fetch / parse / fuzznuc patterns
├── motif_merge.py           — merge / filter / dedup motifs across callers
├── manifest.py              — manifest CSV count / validate / list
├── balance.py               — per-meth-type balanced subset
├── filter.py                — filter .pkl by coverage / type / max keys
│
└── callers/                 — methylation caller output parsers
    ├── __init__.py          — exports BaseOutputParser, create_parser, list_parsers, auto_detect_parser
    ├── base.py              — BaseOutputParser ABC
    ├── registry.py          — @register decorator, factory functions
    ├── pacbio.py            — PacBio motifs.csv
    ├── modkit.py            — modkit pileup --bedMethyl TSV
    ├── ipd_summary.py       — ipdSummary CSV / GFF3
    └── combined.py          — combined CSV (mod_type, motif, offset, frac_mod, n_sites, source)
```
