## Context

The repo is a *teaching museum*: eight independent transformer implementations, each lifted from a different source (Umar Jamil's `pytorch-transformer`, a DataCamp build-from-scratch, Sam Lynn-Evans' translator, Karpathy's nanoGPT/`tinyshakespeare`, the official PyTorch `nn.Transformer`, Ajay Halthor's series, the PyTorch language-model tutorial, and a GPT colab). Each model `N` is realized as a triad:

- `modelN.py` — the network (blocks, attention, generation/`forward`)
- `datasetN.py` — dataset construction, tokenizer, batching, masks
- `tutorialN.py` — orchestration: `train_modelN`, `translateN`/inference, `debug_code_modelN`

Three cross-cutting modules glue them: `config.py` (config dict + device + path helpers), `utils.py` (checkpoint save/reload/load + metrics), and the two CLI entrypoints `train.py` / `translate.py` plus a `test.py` smoke driver. Dispatch is a `match config["alt_model"]`.

Current state: written for CUDA-on-WSL; no `ruff`/`mypy` config; `black -l 159` was the only formatter; no return annotations anywhere; `mps-requirements.txt` already pins `torch==2.5.1` + `mypy==1.14.1` and `config.get_device()` already prefers CUDA→MPS→CPU (the migration started). Several real bugs hide behind the missing type checker.

## Goals / Non-Goals

**Goals:**
- One `pyproject.toml` carrying `ruff` + `mypy` config; first-party tree passes both.
- Return + parameter annotations on first-party public functions; a typed shape for the `config` dict.
- All `model{1..8}` importable and dispatchable; no `NameError`/`TypeError` on selection.
- Device obtained only via `config.get_device()`; backend-specific calls guarded.
- `architecture.md` documenting the triad layout and the train→infer flows.

**Non-Goals:**
- Changing model mathematics or chasing accuracy/BLEU improvements.
- Merging the eight implementations or deleting any tutorial.
- Re-training or shipping checkpoints; resolving every `# JEB:` research note (only the ones that are real bugs).
- Packaging into `src/<package>/` layout — the flat tutorial layout is intentional and stays.

## Decisions

- **Tooling lives in `pyproject.toml`, not `ruff.toml`/`mypy.ini`.** Single file, matches the repo's Python convention in CLAUDE.md. Keep `line-length = 159` to avoid a churning reformat of the entire history; alternative (re-format to 88) was rejected as needless diff noise.
- **`mypy` runs in "lenient-but-annotated" mode, not `--strict`.** Heavy `torch`/`torchtext`/`datasets` dynamism makes `--strict` impractical across 16k lines of tutorial code. Require return annotations and check first-party modules; allow `ignore_missing_imports` for un-stubbed ML libs. `--strict` was considered and deferred to a follow-up change.
- **Config shape via `TypedDict` alias (`Config`), applied incrementally.** Lets `get_weights_file_path`, `save_model`, etc. be checkable without rewriting every reader. Alternative (a dataclass) would ripple through YAML load/dump and all eight tutorials — too invasive for this pass.
- **Device selection stays centralized in `config.get_device()`** and returns a `str`; call sites pass it through `.to(device)`. Guard `torch.cuda.empty_cache()` behind `if device == "cuda"`.
- **Bug fixes are scoped to what typing/imports reveal**, not a behavioral rewrite: (1) import `train_model7` (or remove its `case`) in `train.py`; (2) fix `get_best_model_params_path(config, ...)` arity in `utils.py`; (3) make `test.py` exercise `model8` and import what it calls.

## Risks / Trade-offs

- **No checkpoints in-repo to validate inference end-to-end** → Smoke-test via `debug_code_model*` and a 1-epoch CPU/MPS train of `model8` (tinyshakespeare is tiny) before trusting `translate8`.
- **`torchtext==0.18.0` is end-of-life and import-fragile on new torch** → Keep it pinned; isolate its use to the dataset modules so a future swap is contained; document in `architecture.md` gotchas.
- **Lenient mypy can mask real type errors in ML glue** → Accept for this pass; the goal is a green baseline to build on, recorded as an open follow-up.
- **MPS kernels differ from CUDA (dtype/op coverage)** → Some ops may fall back to CPU or error on MPS; guard known cases and document, don't chase full parity here.

## Open Questions

- Should `ruff format` replace `black -l 159` outright, or run alongside it during transition? (Lean: replace, single formatter.)
- Is `model7` (PyTorch LM tutorial) worth wiring fully into `train.py`/`test.py`, or should its `case` be dropped until it's fixed? (Lean: import it and mark the dataset hack, keep the case.)
- Do we add a minimal CI workflow (`ruff` + `mypy`) in this change or a follow-up? (Lean: follow-up; this change establishes the green baseline locally first.)
