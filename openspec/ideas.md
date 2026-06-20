# Ideas — pytorch-transformer-tutorials

Lightweight backlog for this repo. Captures ideas before they become OpenSpec changes.

- **Raw ideas** — unfiltered `[ ]` entries; check `[x]` when a change ships for it.
- **Suggested next-up** — pointers curated by `/alemax:reprioritise-ideas` before planning.
- **Archived ideas — by capability** — condensed by `/alemax:archive-ideas` as changes archive.

## Suggested next-up

<!-- pointer-only entries, curated from Raw ideas before a planning session -->

## Raw ideas

- [x] **Decide torchtext's fate (blocks `translate.py`)** — RESOLVED via path 3: `dataset7.py` now loads WikiText-2 through HuggingFace `datasets` (`load_dataset("wikitext", "wikitext-2-raw-v1")`), dropping the EOL `torchtext` dependency entirely. The earlier interim lazy-guard is superseded; `torchtext` is gone from the codebase and model7 keeps its dataset. Original context: `torchtext==0.18.0` failed to import on torch 2.5.1 (ABI mismatch `libtorchtext.so` vs `libc10.dylib`), and since `translate.py` imports `tutorial7` at module top, the whole inference CLI died on import for every model.
  - Still open for model7 (separate ideas): it is not wired into `train.py`'s dispatch (commented-out `train_model7` import) and carries several `# JEB:` batching hacks in `dataset7.py`.
- [ ] **Add an MLX backend track (`model9`)** — Apple Silicon native (unified memory, lazy eval). Not a torch drop-in; add as a *new parallel implementation* — an MLX Shakespeare GPT mirroring `model8`, wired into the existing `match config["alt_model"]` dispatch. Gives an apples-to-apples MLX-vs-torch-MPS comparison on tinyshakespeare. Candidate for its own change `adopt-mlx-backend`.
- [ ] **Bump & unify PyTorch pins** — `mps-requirements.txt` is `torch==2.5.1`, `cuda-requirements.txt` is `torch==2.3.1` (drifted). Unify on one current-stable torch (verify latest on pytorch.org); newer torch closes MPS operator gaps (fewer silent CPU fallbacks / unimplemented-op errors). Tied to the torchtext decision above.
- [ ] **Migrate env to `uv` + `pyproject.toml`** — CLAUDE.md mandates `uv`/`pyproject`, but the repo still uses plain `requirements.txt` + venv. The venv was rebuilt with `uv venv` (2026-06-19) but deps still come from `mps-requirements.txt`. Consolidate into `pyproject.toml` with a lockfile; decide whether to keep classic `pip` in the venv (uv omits it by default).
- [ ] **Add CI for ruff + mypy** — once `harden-transformer-tutorials` lands a green baseline, add a GitHub Actions workflow running `ruff check`, `ruff format --check`, and `mypy` so the standard is enforced on every PR.
- [ ] **Reconcile auto-generated config defaults with the committed `config.yaml`** — `get_config()` with no `-c`/`-m` auto-writes `<datasource>_<src>_<tgt>_<model>/config.yaml` from `get_default_config()` on first run (`config.py:44-50`), then reads it back as source of truth. The in-code defaults and the committed `tinyshakespeare_en_en_model8/config.yaml` disagree: defaults `d_model=256, N=6, h=8, d_ff=1024, batch_size=8, num_epochs=20, lr=1e-4` vs committed (retuned 2026-06-19 for a 24 GB Apple Silicon machine, nanoGPT-shakespeare) `d_model=384, N=6, h=6, d_ff=1536, batch_size=64, block_size=256, num_epochs=1, lr=1e-3`. Footgun: deleting/regenerating the config silently swaps the committed model for the smaller in-code defaults. Options: (a) align `get_default_config()` to the committed values, (b) have training log which config it generated vs loaded and warn on mismatch, (c) document the intended defaults. Also note editing `get_default_config()` has no effect once a folder's `config.yaml` exists. Related: `d_ff` and `seq_len` are dead keys for model8 (the builder ignores `d_ff`; FFN is hardwired to 4×`d_model`, and model8 uses `block_size` not `seq_len`).
- [ ] **model8 attention is unbatched (slow on MPS)** — `MultiHeadAttention` is a `ModuleList` of separate `Head` modules looped in Python, so each forward redoes 36 (`N=6`×`h=6`) tiny attention ops. Measured on this 24 GB Apple Silicon machine (2026-06-19), 10.8M params: at the committed `block_size=256, batch=64` it is **~11 s/step → ~15 h for the hardcoded 5000-iter loop** (genuine MPS compute, CPU ~40%, no fallback — memory is fine at ~5 GB). Dropping to `block_size=64, batch=32` for a quick run gave **~0.2 s/step** (≈60× faster: attention is O(block²), 256²/64²=16, plus the 2× batch), finishing 1500 iters + 2000-token generation in a few minutes (loss 1.89→1.56, recognizable Shakespeare). A batched-QKV / `scaled_dot_product_attention` rewrite of `model8.py` would make the committed `block_size=256` config train in ~1–2 h instead of ~15 h. This is the single biggest lever for making the quality config usable.
- [ ] **model8 tokenizer inserts spaces on decode** — `get_or_build_tokenizer8` builds a `BPE(char_level=True)` tokenizer with a `Whitespace` pre-tokenizer (and logs `Ignored unknown kwarg option char_level`). Generation is coherent but decodes character-spaced (`r e e - y o u c a s t`), because the whitespace pre-tokenizer round-trips each char as its own token. Fix options: use a plain char-level vocab (the `local_tokenizer` in `dataset8.py` already does stoi/itos with no spacing) instead of `tokenizers.BPE`, or post-process the decoded stream. Cosmetic only — does not affect training/loss.

- [ ] **model4 & model5 training is unimplemented** — both `train_model4`/`train_model5` build the model + dataloaders (via `get_ds3`, translate en-fr CSV) and then `raise RuntimeError("Training for modelN not implemented")` (`tutorial4.py:35`, `tutorial5.py:34`). They dispatch and start cleanly (verified 2026-06-19: data + model build succeed on MPS, then the deliberate raise), but there is no actual training loop. model1/2/3/6 all train (loss descends on MPS); model4/5 are stubs. Decide whether to implement their training loops or document them as intentionally inference-only / WIP. Config folders `translate_en_fr_model4` / `_model5` now exist so they're dispatchable.

## Archived ideas — by capability

<!-- condensed bullets grouped under capability headings; filled by /alemax:archive-ideas -->

### Code quality

- [x] **ruff + mypy baseline + bug sweep** (PR #6, 2026-06-19) — brought common code + model1–8 to a clean ruff + pragmatic-strict mypy baseline (`pyproject.toml` tooling config; `model5.py` vendored copy and `custom_datasets/`/notebooks excluded). The sweep surfaced ~15 real bugs:
  - `model1.py`: non-in-place `masked_fill` → attention mask silently never applied
  - `tutorial1.py`: `sos_idx`/`eos_idx` passed swapped into `greedy_decode`
  - `tutorial3.py`: `np.cos`/`np.pi` used but numpy never imported (NameError)
  - `tutorial5.py`: typo `to5enizer_src` left `tokenizer_src` undefined
  - `tutorial8.py`: unpacked 6 values from a 5-tuple; `config["model"]="model7"` copy-paste
  - `model6.py`: vocab dict + SOS/EOS/PAD mis-annotated `int` (str keys / `dict[str,int]`)
  - `model7.py`: implicit-Optional `src_mask`; wrong `IterableDataset` type; float `StepLR.step_size`
  - `tutorial2.py`/`tutorial3.py`: `evaluate_*` referenced undefined `model_out` (decode commented out)
  - assorted: `argmax(axis=)`→`dim=`, loss accumulators typed `int` while holding floats
  - side effects: `train.py` now wires model7 into dispatch (was an un-imported function → undefined name); `reload_model`/`load_trained_model` are now PEP 695 generic so callers keep concrete types
