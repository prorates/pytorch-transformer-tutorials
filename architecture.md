# Architecture

This repository is a **collection of eight independent transformer implementations**
(`model1`–`model8`), each transcribed from a different public tutorial, video, or blog
(see `README.md` for sources). They are kept deliberately parallel — the value is in
*comparing* the implementations, not in unifying them.

Originally built and run on an NVIDIA **CUDA** box under **WSL**, the code is being
brought up to run on **Apple Silicon (MPS)** and CPU, and up to **`ruff` + `mypy`**
standards. The flagship demo is training `model8` on `tinyshakespeare` and then
generating **Shakespeare-style text**.

## The model triad

Every model `N` is implemented as three files with a consistent contract:

| File           | Responsibility                                                        |
| -------------- | --------------------------------------------------------------------- |
| `modelN.py`    | The network: embeddings, attention, blocks, `forward` / `generate`.   |
| `datasetN.py`  | Dataset loading, tokenizer, batching, attention masks.                |
| `tutorialN.py` | Orchestration: `train_modelN`, `translateN` (inference), `debug_code_modelN`. |

```
                 ┌────────────────────────────────────────────┐
                 │              config.py                      │
                 │  get_config · get_device · path helpers     │
                 └───────────────┬────────────────────────────┘
                                 │ config dict (alt_model, datasource, …)
        ┌────────────────────────┼────────────────────────┐
        │                        │                         │
   train.py                 translate.py                test.py
   (training CLI)           (inference CLI)         (debug smoke driver)
        │                        │                         │
        │ match config["alt_model"] → dispatch             │
        ▼                        ▼                         ▼
  train_modelN(config)   translateN(config, sentence)  debug_code_modelN(config, device)
        │                        │
        ▼                        ▼
   tutorialN.py ── builds ──▶ modelN.py  ── reads data from ──▶ datasetN.py
        │                                                          │
        ▼                                                          │
   utils.save_model ──▶ <model_folder>/tmodel_<epoch>.pt           │
        ▲                                                          │
        └──────────── utils.reload_model / load_trained_model ◀────┘
```

## Cross-cutting modules

- **`config.py`** — the single source of runtime truth.
  - `get_config(filename?, modelfolder?)` builds/loads a config `dict` (YAML on disk,
    defaults in `get_default_config()`). Key field: `alt_model` selects the model.
  - `get_device()` selects the compute backend in priority **CUDA → MPS → CPU** and
    prints the choice. All training/inference should obtain the device here rather
    than hard-coding `"cuda"`.
  - Path helpers derive a deterministic checkpoint layout (see below).
- **`utils.py`** — checkpoint lifecycle and metrics.
  - `save_model` persists `{epoch, model_state_dict, optimizer_state_dict, global_step}`.
  - `reload_model` restores all of that to **resume training**.
  - `load_trained_model` restores weights only, for **inference** (raises if none exist).
  - `collect_training_metrics` logs CER/WER/BLEU to TensorBoard.

## Entrypoints and dispatch

All three drivers parse CLI flags, call `get_config()`, then `match config["alt_model"]`
to route to the right model:

- **`train.py`** `-c <config> -m <model_folder>` → `train_model{1..8}(config)`.
- **`translate.py`** `-c -m -s <sentence>` → `translate{1..8}(config, sentence)`.
  For `model8` this is autoregressive **generation**, not translation.
- **`test.py`** → `debug_code_model{1..8}(config, device)` — a quick build/shape smoke check.

## The two workflows: train → checkpoint → infer

1. **Train.** `train.py` builds the model, runs the epoch loop, and calls
   `save_model` at the end of every epoch. With `config["preload"] == "latest"`,
   `reload_model` resumes from the most recent checkpoint.
2. **Infer.** `translate.py` rebuilds the model, calls `load_trained_model` to restore
   weights, then translates a sentence (`model1`–`model7`) or **generates Shakespeare**
   (`model8`, seeded from a zero context and decoded char-by-char).

**A model must be trained before it can be used for inference** — there are no
checkpoints committed to the repo.

## Checkpoint layout

Paths are derived entirely from config by `config.py`:

```
<datasource>_<lang_src>_<lang_tgt>_<alt_model>/   e.g. tinyshakespeare_en_en_model8/
├── config.yaml                 # snapshot of the config used
├── tmodel_00.pt                # per-epoch weights  (<model_basename><epoch>.pt)
├── tmodel_01.pt
└── best_model_params.pt        # best-so-far weights
```

`latest_weights_file_path()` finds the newest `tmodel_*` for resume/inference.

## Runtime & platform

- **Device:** chosen by `get_device()`; CUDA-only calls (e.g. `torch.cuda.empty_cache()`)
  must be guarded by `if device == "cuda"` so MPS/CPU runs don't raise.
- **Dependencies:** `mps-requirements.txt` (Apple Silicon) and `cuda-requirements.txt`
  (NVIDIA). Pinned to `torch==2.5.1`; `mypy==1.14.1` is already present.
- **Default config:** `model8` / `tinyshakespeare` / `en`→`en` — i.e. the Shakespeare demo.

## Known gotchas

- `torchtext==0.18.0` is end-of-life and import-fragile on newer torch; its use is
  confined to the dataset modules.
- `# JEB:` comments throughout flag open correctness questions inherited from the
  source tutorials — some are real bugs, some are study notes.
- Backend coverage differs on MPS; a few ops may fall back to CPU or error.

## Where to make changes

- New model behavior → the relevant `modelN.py` / `datasetN.py` / `tutorialN.py` triad.
- Runtime/device/paths → `config.py`.
- Checkpoint format → `utils.py`.
- Wiring a model into the CLIs → the `match` blocks in `train.py` / `translate.py` / `test.py`.

> The hardening effort (lint, types, device portability, bug fixes) is tracked as the
> OpenSpec change `openspec/changes/harden-transformer-tutorials/`.
