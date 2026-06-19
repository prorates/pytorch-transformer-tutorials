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

## Archived ideas — by capability

<!-- condensed bullets grouped under capability headings; filled by /alemax:archive-ideas -->
