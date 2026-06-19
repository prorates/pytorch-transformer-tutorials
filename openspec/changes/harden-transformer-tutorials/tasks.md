## 1. Tooling baseline

- [ ] 1.1 Add `pyproject.toml` with `[tool.ruff]` (lint + format, `line-length = 159`, exclude `.venv/`, `__pycache__/`, checkpoint folders, `*.ipynb_checkpoints`)
- [ ] 1.2 Add `[tool.mypy]` to `pyproject.toml`: target first-party modules, `ignore_missing_imports = true` for torch/torchtext/datasets/torchmetrics, require return annotations
- [ ] 1.3 Add `ruff` to the requirements set; confirm `mypy==1.14.1` pin in `mps-requirements.txt` (already present) and mirror into `cuda-requirements.txt`
- [ ] 1.4 Capture the initial `ruff check .` and `mypy .` error counts as the baseline to drive down

## 2. Fix known runtime bugs

- [ ] 2.1 `train.py`: import `train_model7` from `tutorial7` (or remove its `case`) so no dispatch target is undefined
- [ ] 2.2 `utils.py`: fix `get_best_model_params_path(config, f"{epoch:02d}")` call to match the single-arg signature in `config.py`
- [ ] 2.3 `test.py`: import and call `debug_code_model8`; ensure every `debug_code_model*` it calls is imported
- [ ] 2.4 Re-run `train.py`/`translate.py`/`test.py` import smoke to confirm no `NameError`/`TypeError` on dispatch

## 3. Device runtime

- [ ] 3.1 Confirm `config.get_device()` returns CUDA→MPS→CPU and reports the choice; annotate its return type
- [ ] 3.2 Audit `model*/dataset*/tutorial*` for hard-coded `"cuda"` / `.cuda()` and route through the selected device
- [ ] 3.3 Guard backend-specific calls (e.g. `torch.cuda.empty_cache()`) behind `if device == "cuda"`

## 4. Type the cross-cutting modules

- [ ] 4.1 Add a `Config` `TypedDict` (or documented `dict[str, ...]` alias) and apply it in `config.py`
- [ ] 4.2 Annotate `config.py` helpers (`get_config`, `get_device`, `get_model_folder`, `get_weights_file_path`, `get_best_model_params_path`, `latest_weights_file_path`)
- [ ] 4.3 Annotate `utils.py` (`collect_training_metrics`, `reload_model`, `save_model`, `load_trained_model`)
- [ ] 4.4 Annotate `train.py`, `translate.py`, `test.py` entrypoints

## 5. Lint + type the model/dataset/tutorial families

- [ ] 5.1 Run `ruff check --fix` and `ruff format` across first-party tree; hand-resolve remaining lint errors
- [ ] 5.2 Add return/param annotations to `tutorial1`–`tutorial8` public functions (`train_model*`, `translate*`, `debug_code_model*`, `build_model*`)
- [ ] 5.3 Add annotations to `model1`–`model8` and `dataset1`–`dataset8` public surfaces until `mypy .` is green
- [ ] 5.4 Triage `# JEB:` markers: fix the ones that are real bugs, leave research notes as-is

## 6. Verify the workflow

- [ ] 6.1 Train `model8` for 1 epoch on CPU/MPS against `tinyshakespeare`; confirm a `tmodel_*.pt` checkpoint is written
- [ ] 6.2 Run `translate8` against that checkpoint; confirm Shakespeare-style text is generated
- [ ] 6.3 Confirm `ruff check .` and `mypy .` both exit 0 on the first-party tree

## 7. Documentation

- [ ] 7.1 Add `architecture.md` (triad layout, dispatch, train→checkpoint→infer flows, device runtime, gotchas)
- [ ] 7.2 Update `CLAUDE.md` "Project purpose" and `README.md` to describe the train-then-infer workflow and Apple Silicon support
