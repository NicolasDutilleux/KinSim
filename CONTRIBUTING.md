# Contributing to KinSim

Thanks for taking the time to contribute. This document covers the
development workflow.

## Development setup

We recommend [uv](https://docs.astral.sh/uv/) for the fastest install,
but plain `pip` also works.

```bash
git clone https://github.com/NicolasDutilleux/KinSim.git
cd KinSim
uv venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
pre-commit install
```

This installs:
- `kinsim_nn` in editable mode
- the `[plot]` extras for the analysis dashboards
- `ruff`, `pre-commit`, `pytest`, `pytest-cov` for development

## Running the checks

```bash
# fast lint (autofix)
ruff check --fix .
ruff format .

# tests
pytest

# coverage
pytest --cov=kinsim_NN --cov-report=term-missing
```

`pre-commit` runs the same lints automatically on every `git commit`.
CI runs the matrix on Python 3.9 / 3.10 / 3.11 / 3.12. The production
cluster's `kinsim_env` conda environment is Python 3.9.25.

## Project layout

- `kinsim_NN/`            core package (extract / train / generate / evaluate / analyze)
- `kinsim_nn_config.yaml` single source of truth for all stages
- `slurm/`                HPC SLURM scripts (prep + callers)
- `scripts/`              auxiliary one-off tools (run with `python`, not via the CLI)
- `tests/`                pytest suite

See [`CLAUDE.md`](CLAUDE.md) for the in-depth developer reference (data
flow, file formats, import rules, conventions) and
[`DECISIONS.md`](DECISIONS.md) for the architectural rationale.

## Coding conventions

- Python 3.9 or above; type hints on public APIs,
  `from __future__ import annotations` mandatory on every module so
  PEP 604 union and PEP 585 generic syntax stay 3.9-compatible.
- `Path` for I/O, never `os.path.join`.
- Every module: `log = logging.getLogger(__name__)`; never bare
  `print()` for operational output.
- Catch specific exceptions, never bare `except Exception`.
- Don't add features, refactors, or abstractions beyond what the task
  needs. A bug fix doesn't need surrounding cleanup.

## Commit style

- Conventional commit subjects when natural (`feat:`, `fix:`,
  `docs:`, `refactor:`, `test:`, `chore:`).
- Keep the subject under 70 characters; use the body for the rationale.
- Avoid mass reformatting commits — keep behavioural changes separate
  from style changes so review is easier.

## Pull requests

1. Open one PR per change.
2. Make sure CI is green before requesting review.
3. Update `CHANGELOG.md` under `## [Unreleased]`.
4. Update `CLAUDE.md` if the developer-reference statements change.
5. The PR template is auto-populated; fill in *Summary* and *Test plan*.

## Reporting bugs

Use the GitHub issue templates (`Bug report` / `Feature request`).
Include the exact command, full traceback, KinSim version, Python
version, and OS — these are the four things that turn a vague report
into a fixable one.
