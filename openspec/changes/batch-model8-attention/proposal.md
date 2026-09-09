## Why

`model8`'s `MultiHeadAttention` holds an `nn.ModuleList` of independent `Head` modules and
concatenates them in a Python loop. Each `Head` runs its own `key`/`query`/`value` projections,
its own softmax, and registers its own `(block_size, block_size)` causal mask — so at the
committed `N=6, h=6` a forward pass dispatches **36 small attention computations** and the model
carries 36 identical mask buffers.

The cost that matters is **memory, not speed**. Peak memory during training, measured at batch
64 on a GTX 1660:

| `h` | per-head | fused | ratio |
|-----|----------|-------|-------|
| 1 | 2.33 GiB | 2.11 GiB | 1.10× |
| 3 | 2.75 GiB | 2.11 GiB | 1.30× |
| **6** (committed) | **3.39 GiB** | **2.11 GiB** | **1.61×** |
| 12 | 4.66 GiB | 2.12 GiB | 2.20× |

Per-head memory **doubles** from `h=1` to `h=12`; fused moves 1.005×. Per-head materialises the
full `B×h×T×T` attention matrix and pays for every head, so its memory ceiling *tightens as head
count rises*. `scaled_dot_product_attention` dispatches to a memory-efficient kernel that does
not materialise it, so **head count is essentially free in memory**.

That matters because of how running out fails. On WSL2 — one of this repo's two documented
environments — exceeding VRAM does not raise `CUDA out of memory`. The shared-memory fallback
pages into host RAM over PCIe and training degrades **9–26×** with no error: model8 goes from
0.809 s/step to 7.289 s/step for a 1.25× batch increase, and 21 s/step further out. The user
experiences "training is inexplicably slow" and has nothing to search for. Per-head reaches that
regime at 1.5–2× smaller batch than fused, and the gap widens with `h`.

Speed is a real but secondary benefit: **22% faster** at the committed config on CUDA
(0.585 → 0.456 s/step), and 25–30% estimated on MPS from the head-count sweep. Dispatch cost is
~32 ms per extra head-module on CUDA and ~24 ms on MPS, against ~6 ms fused.

Finally, this is a teaching repo. The fused-QKV + SDPA formulation is the one a reader will meet
in every production GPT implementation, and `model8` is the file that should show it.

### What this change is NOT justified by

Two earlier arguments were tested and did not survive. They are recorded here so they are not
revived:

- **Not a large speedup.** `ideas.md` once claimed ~11 s/step → ~15 h for the committed config.
  It re-measures at 0.438 s/step (0.61 h) on MPS and 0.585 s/step on a GTX 1660. The committed
  config was already trainable in well under an hour; the ~60× figure was never real.
- **Not an OOM rescue.** Both implementations complete at batch 64 on a 6 GB card (per-head peaks
  3.39 GiB). "Fused enables a run per-head cannot do" is false at the committed config. Fused is
  also **not immune** to spilling — it spills at `h=6` batch 192 (6.05 GiB). The honest claim is
  a headroom multiplier, not immunity.

## What Changes

- Replace the per-head `Head` modules with a fused QKV projection
  (`Linear(n_embd, 3 * n_embd, bias=False)`) and one `F.scaled_dot_product_attention` call.
- Delete the `Head` class and its per-head `tril` buffer; causal masking moves to `is_causal=True`.
- Keep `MultiHeadAttention`'s name, constructor signature, and output shape, so `Block`,
  `Transformer8`, `build_transformer8`, and every caller are untouched.
- Preserve the existing attention scale explicitly (below), so the change is numerically
  equivalent and the memory and speed results are attributable.
- **BREAKING (checkpoints only):** the parameter layout changes and the per-head `tril` buffers
  leave `state_dict`, so existing `model8` `.pt` files will not load. None are committed and none
  exist in `tinyshakespeare_en_en_model8/`, so nothing is stranded.

### Deliberately not changed: the attention scale

The current code scales scores by `C**-0.5` where `C` is `x.shape[-1]` — the full embedding dim,
not the head size. That differs from SDPA's `head_size**-0.5` default by exactly **√h**, so a
naive swap would silently retune the model while claiming to be a performance change.

This change passes `scale=n_embd**-0.5` to reproduce current behaviour exactly. Whether that
scale is *correct* is a separate question tracked in `openspec/ideas.md` — the error vanishes at
`h=1`, which is the evidence it is a single-head transcription slip — and answering it needs a
training comparison, not a refactor.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-training`: adds a requirement that the GPT-style model's attention is computed batched
  across heads, with peak memory independent of head count, and pins the behavioural contract the
  rewrite must not break (identical outputs for identical weights, unchanged causal masking,
  unchanged dropout semantics).

## Impact

- **Code:** `model8.py` only — `Head` removed, `MultiHeadAttention` rewritten.
- **Tests:** `tests/test_models.py` already covers model8's logit shape and loss; this adds an
  equivalence test driving both implementations from one set of weights.
- **Checkpoints:** existing model8 weights become unloadable (BREAKING above).
- **Other models:** none. `model1`–`model7` carry their own attention implementations and the
  repo deliberately keeps the families parallel.
- **Dependencies:** none. `F.scaled_dot_product_attention` and `scale=` are in the pinned
  `torch==2.5.1`.
- **Measurement note:** peak memory is the metric, not OOM. On WSL2 an OOM-detecting harness is
  structurally blind — that branch is never reached — so a config "surviving" proves nothing.
- **Out of scope:** the pre-norm `# JEB:` note in `Block`, the tokenizer's character-spaced
  decode, and the `d_ff` key model8 ignores. All tracked separately.
