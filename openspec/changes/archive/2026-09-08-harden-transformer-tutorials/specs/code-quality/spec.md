## ADDED Requirements

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
