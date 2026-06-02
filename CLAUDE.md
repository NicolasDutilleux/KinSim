# KinSim — developer reference for Claude

`kinsim_NN` is the only package shipped by this repository. It trains a
conditional WGAN-GP with a transformer generator and critic to inject
biologically realistic PacBio HiFi kinetics (IPD and PW) into a target
BAM. See [`README.md`](README.md) for the user-facing entry point and
[`DECISIONS.md`](DECISIONS.md) for the architectural rationale.

## Package layout

```
kinsim_NN/
├── __init__.py
├── __main__.py           CLI dispatcher: extract|train|generate|evaluate|analyze
├── extract.py            aligned bystrandified BAM + labels → shard.pkl
├── train.py              WGAN-GP loop
├── generate.py           ckpt + input BAM → output BAM with fi/fp/ri/rp
├── evaluate.py           distribution-level W1 on held-out shards
├── analyze.py            shard QC dashboard (HTML)
├── models/
│   ├── blocks.py         AdaLNZeroBlock, TransformerBlock, MultiHeadSelfAttention, FFN
│   ├── generator.py      TransformerGenerator
│   └── discriminator.py  TransformerDiscriminator
├── data/
│   ├── shard.py          ShardData dataclass, read/write/finalize helpers
│   └── dataset.py        ShardedDataset, MultiShardDataset
├── labelers/
│   ├── base.py           BaseLabeler ABC
│   ├── registry.py       @register decorator, create_labeler factory
│   ├── gff.py            GFFLabeler (motifs.gff via pbmotifmaker)
│   └── jasmine_mm_ml.py  JasmineMMMLLabeler (5mC via MM/ML tags)
└── utils/
    ├── config.py         YAML loader, KinsimNNConfig dataclass
    ├── encoding.py       BASE_MAP, METH_IDS, encode_seq, get_meth_ids
    ├── pacbio_codec.py   uint8 ↔ frames lookup tables
    ├── bam_io.py         detect_bam_format, iter_window_samples, iter_chunk_samples
    ├── losses.py         wgan_gp_d_loss, wgan_g_loss, gradient_penalty
    ├── metrics.py        wasserstein_1d
    ├── motifs.py         IUPAC motif handling (vendored, self-contained)
    └── parsers/          PacBio / REBASE / modkit / combined motif parsers
```

## YAML: single source of truth

`kinsim_nn_config.yaml` at the repository root is read by every stage
of the pipeline. The model checkpoint (`model_config.json`) freezes the
relevant subset at training start: `k`, `n_meth_types`,
`meth_id_by_name`, the generator and discriminator architecture, and
the training hyperparameters. The frozen copy is preferred over the
YAML at inference time, so editing the YAML between training and
generation cannot silently break a checkpoint.

## Data flow

```
Aligned bystrandified BAM (ip/pw, 2 records per ZMW)
  + reference FASTA
  + GFF / jasmine MM-ML labels
                                       │
                          kinsim_nn extract
                                       │
                       shards/<strain>_shard.pkl
                                       │
                            kinsim_nn train
                                       │
                  ckpts/{G.pt, D.pt, best_G.pt,
                         model_config.json, metrics.csv, tb/}
                                       │
              ┌────────────────────────┴────────────────────────┐
              │                                                 │
       kinsim_nn evaluate                              kinsim_nn generate
       W1 on held-out shards                  BAM(stripped) → BAM(fi/fp/ri/rp)
```

## Shard layout (`kinsim_NN/data/shard.py`)

Per row of length `K = 21`:

| field | dtype | shape |
|---|---|---|
| `base_fwd` | uint8 | (N, K) — A/C/G/T = 0..3, forward strand |
| `meth_fwd`, `meth_rev` | uint8 | (N, K) — methylation ID per position per strand |
| `signal` | uint8 | (N, K, 4) — IPD_fwd, PW_fwd, IPD_rev, PW_rev (PacBio uint8 codec) |
| `category` | uint8 | (N,) — 0 = BASELINE, 1 = SLOWED, 2 = NEAR_METH |
| `parent_meth`, `parent_offset` | uint8 / int8 | (N,) — parent methylation ID and offset |
| `ref_id`, `ref_pos`, `strand`, `zmw` | — | (N,) — traceability |

## CLI

```
kinsim_nn extract  --manifest <csv> --output-dir <dir> [--task <i>] [--config <yaml>]
kinsim_nn train    <shards_dir> <ckpt_dir> [--resume] [--config <yaml>]
kinsim_nn generate <input.bam> <ref.fa> <ckpt_dir> <motifs.csv> <out.bam>
kinsim_nn evaluate <ckpt_dir> <shards_dir> --output-prefix <prefix>
kinsim_nn analyze  <shards_dir_or_file> [--output-dir <dir>] [--no-html]
```

Note: `extract` takes named flags (`--manifest`, `--output-dir`); every
other subcommand is positional.

## SLURM building blocks (kept for the validation chain)

```
slurm/
├── prep/
│   ├── bystrandify.slurm
│   ├── align_pbmm2.slurm
│   ├── index_bam.slurm
│   └── assembly_hifiasm.slurm
└── callers/
    ├── ipdsummary.slurm
    ├── pbmotifmaker.slurm
    ├── pbmotifmaker_reprocess.slurm
    ├── jasmine_modkit.slurm
    ├── jasmine_align_only.slurm
    └── merge_motifs.slurm
```

The validation chain is `kinsim_nn generate` → `bystrandify.slurm` →
`align_pbmm2.slurm` → `ipdsummary.slurm` → `pbmotifmaker.slurm`.
Comparison against real ipdSummary output uses
`scripts/plot_perread_ipd_at_gff_sites.py` with the GFF produced by the
real-data ipdSummary as the position reference.

## Coding conventions

- Python 3.9 or above (the production cluster runs 3.9.25 inside the
  `kinsim_env` conda environment). Type hints on public function
  signatures in `kinsim_NN/models/` and `kinsim_NN/data/`. Every module
  starts with ``from __future__ import annotations`` so that PEP 604
  union syntax (``int | None``) and PEP 585 generics (``dict[str, int]``)
  are stringified at runtime and remain valid on 3.9.
- `Path` objects for file I/O.
- `sys.exit(1)` with a stderr message on fatal errors.
- Never catch `Exception` broadly; catch specific exceptions.
- `model.eval()` and `@torch.no_grad()` on inference paths.
- Always save `model_config.json` before the first epoch.
- Use `MultiShardDataset` for training, `ShardedDataset` for evaluation
  on a known shard.
- Logging: `log = logging.getLogger(__name__)` per module;
  `setup_logging()` from `kinsim_NN.utils.config` called once per CLI
  entry point.

## What not to do

- Do not edit `kinsim_nn_config.yaml` between training and generation
  unless you understand the consequence for `model_config.json`.
- Do not skip the strand-aware kinetic-tag reversal in
  `kinsim_NN/utils/bam_io.py` when handling bystrandified BAMs.
- Do not emit a BAM with a SAM flag other than `4` on the unaligned
  output, do not leave `SO:coordinate` on the `@HD` line, do not leave
  stale `ip` and `pw` tags on the output, do not allow `0` in the
  `fi`, `fp`, `ri`, `rp` arrays, and do not omit `fn:i:1` / `rn:i:1`
  (without these, `ccs-kinetics-bystrandify` silently drops every
  record). See [`BUGS_FOUND.md`](BUGS_FOUND.md) for the full case
  history and [`PACBIO_COMPATIBILITY.md`](PACBIO_COMPATIBILITY.md) for
  the forward-looking rules.
- Do not mix PacBio tool sources inside one chain (SIF `pbmm2` 1.18 vs
  conda `pbmm2` 26.x produce mutually unreadable BAMs). See
  [`PACBIO_COMPATIBILITY.md`](PACBIO_COMPATIBILITY.md).
