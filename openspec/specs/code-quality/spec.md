# code-quality Specification

## Purpose

Hold eight independently-transcribed implementations to one mechanical standard.

`model1`-`model8` come from eight different tutorials with eight different house styles.
Unifying the *code* is explicitly not the goal — the value of this repo is in comparing the
implementations — so `ruff` and `mypy --strict` are what keep them comparable instead: the
same lint rules and the same type discipline everywhere, enforced in CI, so a difference
between two models is a real design difference and not an artifact of whoever wrote it.

## Requirements

### Requirement: Lint and format with ruff

The repository SHALL configure `ruff` for both linting and formatting via `pyproject.toml`, and all first-party modules SHALL pass `ruff check` with no errors. Formatting SHALL be stable under `ruff format` (or an explicitly configured line length compatible with the existing `black -l 159` history).

#### Scenario: Clean lint run

- **WHEN** a developer runs `ruff check .` from the repository root
- **THEN** the command exits 0 with no reported violations for first-party modules (`config.py`, `utils.py`, `train.py`, `translate.py`, `test.py`, `model*.py`, `dataset*.py`, `tutorial*.py`)

#### Scenario: Formatting is idempotent

- **WHEN** a developer runs `ruff format --check .`
- **THEN** the command reports that no files would be reformatted

#### Scenario: Vendored and generated paths are excluded

- **WHEN** ruff runs
- **THEN** `.venv/`, `__pycache__/`, checkpoint folders, and notebook checkpoints are excluded from linting via configuration

### Requirement: Static type checking with mypy

The repository SHALL configure `mypy` via `pyproject.toml`, and the first-party modules SHALL type-check cleanly. Public functions SHALL carry parameter and return-type annotations; the `config` dictionary contract SHALL be expressed with a typed structure (e.g. `TypedDict`) or documented `dict[str, ...]` aliases so checkpoint and training helpers are checkable.

#### Scenario: Clean type-check run

- **WHEN** a developer runs `mypy .` (respecting the configured module include/exclude set)
- **THEN** the command exits 0 with no type errors in first-party modules

#### Scenario: Return annotations present

- **WHEN** mypy runs with the configured strictness for untyped definitions
- **THEN** no first-party public function is reported as missing a return-type annotation
