## Why

This repository collects eight transformer implementations (`model1`–`model8`) gathered from public tutorials, videos, and blog posts. It was written and run exclusively on an NVIDIA CUDA box under WSL, accumulated copy-paste bugs along the way (dozens of `# JEB:` markers flag open correctness questions), and carries **zero** type or lint enforcement — `mypy` and `ruff` are not configured, no function is return-annotated, and at least two call sites raise at runtime before doing any useful work. Recent commits began migrating to Apple Silicon (MPS) but the move is unfinished and undocumented. The goal is to make the tutorials *runnable, typed, and linted* on a developer laptop while preserving the train-then-infer workflow whose flagship demo is generating Shakespeare-style text from `model8`.

## What Changes

- Add and enforce **ruff** (lint + format) and **mypy** tooling: a `pyproject.toml` config, dependency pins, and a clean baseline across the first-party modules (`config.py`, `utils.py`, `train.py`, `translate.py`, `test.py`, and the `model*/dataset*/tutorial*` families).
- Formalize **cross-platform device selection** so the code runs on CUDA, Apple Silicon (MPS), or CPU — replacing the implicit CUDA/WSL-only assumption. **BREAKING** for any caller that hard-coded `"cuda"`.
- Document and lock down the **train → checkpoint → infer** workflow: `train.py` dispatches to `train_model{1..8}`, checkpoints land under `<datasource>_<src>_<tgt>_<model>/`, and `translate.py` reloads them for inference/generation.
- Fix the **known runtime bugs** surfaced by typing: the missing `train_model7` import in `train.py`, the wrong-arity `get_best_model_params_path(...)` call in `utils.py`, and `test.py` not exercising `model8`.
- Add a repository **`architecture.md`** describing the model/dataset/tutorial triad layout and the two entrypoint flows.

## Capabilities

### New Capabilities
- `code-quality`: ruff + mypy are configured and the first-party tree passes both; CI-checkable formatting and typing baseline.
- `device-runtime`: a single device-selection contract that picks CUDA, MPS, or CPU at runtime and reports the choice.
- `model-training`: the `train.py` entrypoint trains a selected model from config and writes per-epoch checkpoints.
- `model-inference`: the `translate.py` entrypoint reloads a trained checkpoint and produces output (translation, or Shakespeare-style generation for `model8`).
- `checkpoint-management`: save / reload / load-for-inference of model + optimizer state under a deterministic folder layout.

### Modified Capabilities
<!-- None: openspec/specs/ is currently empty; every capability above is new. -->

## Impact

- **Tooling/deps:** new `pyproject.toml` (ruff + mypy config); `mypy`, `ruff` added to the requirements set (`mps-requirements.txt` already pins `mypy==1.14.1`).
- **Code:** type annotations and lint fixes across all first-party `.py` files; bug fixes in `train.py` and `utils.py`; device handling routed through `config.get_device()`.
- **Runtime:** behavior unchanged for CUDA users; newly functional on Apple Silicon (MPS) and CPU.
- **Docs:** new `architecture.md`; `README.md` and `CLAUDE.md` project-purpose section updated to reflect the workflow.
- **Out of scope:** changing model math, retraining for accuracy, or unifying the eight implementations into one — they intentionally stay as parallel tutorials.
