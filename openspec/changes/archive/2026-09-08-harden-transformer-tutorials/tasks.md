## 1. Tooling baseline

- [x] 1.1 Add `pyproject.toml` with `[tool.ruff]` (lint + format, `line-length = 159`, exclude `.venv/`, `__pycache__/`, checkpoint folders, `*.ipynb_checkpoints`)
- [x] 1.2 Add `[tool.mypy]` to `pyproject.toml`: target first-party modules, `ignore_missing_imports = true` for torch/torchtext/datasets/torchmetrics, require return annotations
- [x] 1.3 Add `ruff` to the requirements set — landed differently than written: `ruff`/`mypy`/`pytest` are a `[dependency-groups] dev` group in `pyproject.toml`, and the two requirements files this task wanted to mirror into were retired (uv.lock replaces them)
- [x] 1.4 Capture the initial `ruff check .` and `mypy .` error counts as the baseline to drive down

## 2. Fix known runtime bugs

- [x] 2.1 `train.py`: import `train_model7` from `tutorial7` (or remove its `case`) so no dispatch target is undefined
- [x] 2.2 `utils.py`: fix `get_best_model_params_path(config, f"{epoch:02d}")` call to match the single-arg signature in `config.py`
- [x] 2.3 `test.py`: import and call `debug_code_model8`; ensure every `debug_code_model*` it calls is imported — and wiring it up surfaced a second bug this task existed to catch: `debug_code_model8` overrode `datasource` to `"translate"`, but `dataset8.load_custom_dataset` reads the unsuffixed `custom_datasets/<datasource>/<lang>.txt`, so it raised `FileNotFoundError` on `custom_datasets/translate/en.txt`. Repointed at tinyshakespeare / en→en, which is what model8 actually models.
- [x] 2.4 Re-run `train.py`/`translate.py`/`test.py` import smoke to confirm no `NameError`/`TypeError` on dispatch

## 3. Device runtime

- [x] 3.1 Confirm `config.get_device()` returns CUDA→MPS→CPU and reports the choice; annotate its return type
- [x] 3.2 Audit `model*/dataset*/tutorial*` for hard-coded `"cuda"` / `.cuda()` and route through the selected device — one real find: `model3.PositionalEncoder.forward` had `if x.is_cuda: pe.cuda()`, which was both CUDA-only and a no-op (`.cuda()` returns a new tensor rather than moving in place). `pe` is a registered buffer and is already on the module's device, so the block was removed. `model5.py` is vendored torch and out of scope.
- [x] 3.3 Guard backend-specific calls (e.g. `torch.cuda.empty_cache()`) behind `if device == "cuda"`

## 4. Type the cross-cutting modules

- [x] 4.1 Add a `Config` `TypedDict` (or documented `dict[str, ...]` alias) and apply it in `config.py` — shipped as `ConfigDict`
- [x] 4.2 Annotate `config.py` helpers (`get_config`, `get_device`, `get_model_folder`, `get_weights_file_path`, `get_best_model_params_path`, `latest_weights_file_path`)
- [x] 4.3 Annotate `utils.py` (`collect_training_metrics`, `reload_model`, `save_model`, `load_trained_model`)
- [x] 4.4 Annotate `train.py`, `translate.py`, `test.py` entrypoints

## 5. Lint + type the model/dataset/tutorial families

- [x] 5.1 Run `ruff check --fix` and `ruff format` across first-party tree; hand-resolve remaining lint errors
- [x] 5.2 Add return/param annotations to `tutorial1`–`tutorial8` public functions (`train_model*`, `translate*`, `debug_code_model*`, `build_model*`)
- [x] 5.3 Add annotations to `model1`–`model8` and `dataset1`–`dataset8` public surfaces until `mypy .` is green

**Moved out of this change:** triaging the `# JEB:` markers. There are 59 of them across 14
files (`model1` and `model2` carry 10 each), and separating inherited real bugs from study
notes is a semantic review, not the lint/type pass this change scoped. Tracked in
`openspec/ideas.md` as its own entry.

## 6. Verify the workflow

- [x] 6.1 Train `model8` for 1 epoch on CPU/MPS against `tinyshakespeare`; confirm a `tmodel_*.pt` checkpoint is written — re-verified 2026-09-08 on MPS at reduced dims (d_model 128, N 4, h 4, block 64, batch 32, 300 iters): loss 2.57 → 2.30, checkpoint written. The committed config (d_model 384, block 256, batch 64) needs ~15 h for its 5000-iter loop; that cost is tracked separately as the model8 batched-attention idea.
- [x] 6.2 Run `translate8` against that checkpoint; confirm Shakespeare-style text is generated — generated Shakespeare-shaped text, character-spaced on decode, which is the separately-tracked `get_or_build_tokenizer8` whitespace-pre-tokenizer idea, not a regression here.
- [x] 6.3 Confirm `ruff check .` and `mypy .` both exit 0 on the first-party tree — and both now run in CI on every PR.

## 7. Documentation

- [x] 7.1 Add `architecture.md` (triad layout, dispatch, train→checkpoint→infer flows, device runtime, gotchas)
- [x] 7.2 Update `CLAUDE.md` "Project purpose" and `README.md` to describe the train-then-infer workflow and Apple Silicon support
