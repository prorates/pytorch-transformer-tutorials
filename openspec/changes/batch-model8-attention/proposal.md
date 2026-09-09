## Why

`model8` is the repo's flagship demo — the nanoGPT-style char-level Shakespeare model — and
its committed config cannot realistically be run: **~11 s/step, ~15 h** for the hardcoded
5000-iteration loop, measured on Apple Silicon at `block_size=256, batch_size=64`. It is
genuine MPS compute (CPU ~40%, ~5 GB resident, no fallback), so the cost is not a device
problem. `MultiHeadAttention` holds an `nn.ModuleList` of independent `Head` modules and
concatenates them in a Python loop, each `Head` running its own `key`/`query`/`value`
projections and its own softmax — so at the committed `N=6, h=6` one forward pass dispatches
**36 tiny attention computations** where the batched formulation dispatches two.

The result is that the committed quality config is aspirational: every run to date has been
at reduced dimensions.

## What Changes

- Replace the per-head `Head` modules with a single fused QKV projection
  (`Linear(n_embd, 3 * n_embd, bias=False)`) and one
  `F.scaled_dot_product_attention` call inside `MultiHeadAttention`.
- Delete the now-unused `Head` class.
- Drop the per-head `tril` causal-mask buffer in favour of SDPA's `is_causal=True`.
- Keep `MultiHeadAttention`'s name, constructor signature, and output shape, so `Block`,
  `Transformer8`, `build_transformer8`, and every caller are untouched.
- Preserve the existing attention scale explicitly (see below) so the change is
  **numerically equivalent** and the speedup is the only variable.
- **BREAKING (checkpoints only):** the parameter layout changes, so existing `model8` `.pt`
  files will not load. No model8 checkpoints are committed and none exist in
  `tinyshakespeare_en_en_model8/`, so nothing is stranded.

### Deliberately not changed: the attention scale

The current code scales scores by `C**-0.5` where `C` is `x.shape[-1]` — the **full embedding
dim**, not the head size:

```python
_B, T, C = x.shape                        # C = n_embd = 384
wei = q @ k.transpose(-2, -1) * C**-0.5   # 384**-0.5 ≈ 0.051
```

`scaled_dot_product_attention` defaults to `head_size**-0.5` (`64**-0.5` = 0.125) — **2.45×
larger** at the committed config. A naive swap would therefore change the model's maths while
claiming to be a performance refactor, and confound the two results.

This change passes `scale=n_embd**-0.5` to SDPA to reproduce current behaviour exactly.
Whether that scale is *correct* is a separate question — Karpathy's nanoGPT uses
`k.shape[-1]**-0.5`, so this looks like a transcription slip — but answering it needs a
training comparison, not a refactor. It gets its own change; recorded in `openspec/ideas.md`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-training`: adds a requirement that model8's attention is computed batched across
  heads, and pins the behavioural contract the rewrite must not break (identical outputs for
  identical weights, unchanged causal masking, unchanged dropout semantics).

## Impact

- **Code:** `model8.py` only — `Head` (removed) and `MultiHeadAttention` (rewritten).
  `Block`, `Transformer8`, `build_transformer8` unchanged.
- **Tests:** `tests/test_models.py` already asserts model8's logit shape and loss; add an
  equivalence test that drives old and new implementations from the same weights.
- **Checkpoints:** existing model8 weights become unloadable (see BREAKING above).
- **Other models:** none. `model1`–`model7` carry their own attention implementations, and
  the repo deliberately keeps the families parallel rather than factoring out shared code.
- **Dependencies:** none. `F.scaled_dot_product_attention` and its `scale=` argument are
  available in the pinned `torch==2.5.1`.
- **Out of scope:** the pre-norm `# JEB:` note in `Block`, the tokenizer's character-spaced
  decode, and the `d_ff` config key model8 ignores. All tracked separately in `ideas.md`.
