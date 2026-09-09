## Context

See `proposal.md — Why` for motivation. The constraints that shape the approach:

- `MultiHeadAttention.__init__` already takes `num_heads` and `head_size` separately, but the
  existing code only works when `num_heads * head_size == n_embd`: it concatenates the head
  outputs and feeds them to `self.proj = Linear(n_embd, n_embd)`. `Block` always passes
  `head_size = n_embd // n_head`, so this holds today and the fused form makes it explicit.
- `torch==2.5.1` is pinned, so `F.scaled_dot_product_attention` and its `scale=` keyword are
  both available.
- The repo deliberately keeps `model1`–`model8` parallel rather than sharing code, so this
  change stays inside `model8.py` and does not factor anything out.
- `model8.py` is a teaching file: its comments explain what each tensor shape means, and that
  explanatory value has to survive the rewrite.

## Goals / Non-Goals

**Goals:**

- One QKV projection and one attention call per layer, replacing `h` of each.
- Provable equivalence to the current implementation in eval mode, so the speedup is
  attributable and nothing about the model's behaviour is silently in flight.
- Keep the file readable as a tutorial — the batched formulation is explained, not just
  applied.

**Non-Goals:**

- Bit-identical *training* runs. Dropout RNG consumption necessarily differs (one fused draw
  instead of `h` separate ones), so equivalence is asserted in eval mode only.
- Flash-attention or any backend selection. SDPA picks its own kernel; we do not pin one.
- Touching `Block`, `Transformer8`, `build_transformer8`, or any other model family.
- Changing the attention scale — see the decision below.

## Decisions

### Fuse Q/K/V into one `Linear(n_embd, 3 * n_embd)` rather than three separate ones

Three separate `Linear(n_embd, n_embd)` projections would already collapse 18 per-head
matmuls into 3, which is most of the win. One fused projection makes it 1, and it is the
shape nanoGPT settled on. The cost is a slightly less obvious mapping from weights to roles,
which the equivalence test pins down anyway.

**Alternative considered:** keep three projections for readability. Rejected — the fused form
is the one a reader will meet in every production GPT implementation, and this file's job is
to teach that.

### Preserve the existing `n_embd**-0.5` scale via SDPA's `scale=` argument

The current code scales by `x.shape[-1]**-0.5` — the embedding dim, not the head size — which
is 2.45× smaller than SDPA's default at the committed config. Passing `scale=` reproduces it
exactly.

**Alternative considered:** take SDPA's default and treat the scale correction as part of
this change. Rejected: it would mean a performance PR that also silently retunes the model,
and any loss-curve difference afterwards would be unattributable. Karpathy's nanoGPT uses
`k.shape[-1]**-0.5`, so the current value is probably a transcription slip — but that is a
claim to be settled with a training comparison, in its own change.

### Assert `num_heads * head_size == n_embd` in the constructor

The invariant is already load-bearing but implicit; today a bad pairing fails deep inside
`self.proj` with a shape error. The fused projection makes the invariant structural, so state
it where it can produce a readable message.

### Replace the per-head `tril` buffer with `is_causal=True`

Each `Head` currently registers its own `(block_size, block_size)` causal mask, so the model
carries `N * h` identical buffers. SDPA's `is_causal` applies the same lower-triangular mask
without materialising one. Query and key lengths are always equal here (`generate` crops to
`block_size` before calling forward), which is the condition `is_causal` requires.

**Note:** those buffers are non-persistent in effect but currently appear in `state_dict`,
which is a second reason old checkpoints will not load — beyond the QKV layout change already
called out in the proposal.

### Prove equivalence with a test that drives both implementations from one set of weights

Keep the old per-head code in the test (not in `model8.py`) as a reference implementation,
copy weights into the fused layout, and compare outputs under `torch.no_grad()` in eval mode.
This is what makes "numerically equivalent" a checked claim rather than an assertion in a
commit message.

### The negative control is degenerate at `h=1` — document it or it reads as a bug

The equivalence test is only trustworthy if it *fails* when `scale=` is dropped. But the
model8 scale and SDPA's default differ by exactly `sqrt(h)` (`sqrt(n_embd / head_size)`), so
at `h=1` — where `head_size == n_embd` — the two are numerically identical and the negative
control **cannot** fail. That is arithmetic, not a defect.

A test asserting "dropping `scale=` breaks equivalence" uniformly across head counts would
therefore fail at `h=1` for a reason unrelated to any bug, and whoever runs it first will
think they found something. Run the negative control at `h=6` and `h=12`, and at `h=1` either
skip it with a reason or assert the degeneracy explicitly.

Identified by the WSL2/CUDA session while amending the shared benchmark harness, 2026-09-09.

## Risks / Trade-offs

- **The rewrite changes behaviour without anyone noticing** → the equivalence test is written
  before the rewrite lands and runs in CI on every PR.
- **`scale=` is forgotten, and the model silently retunes** → the equivalence test fails
  immediately if the scale differs; it is the single most likely regression.
- **`is_causal=True` is wrong for some call path** → covered by an explicit causality
  scenario: perturb a later token, assert earlier outputs are unchanged.
- **Existing model8 checkpoints become unloadable** → none are committed and none exist
  locally; stated in the proposal as BREAKING. If one turns up, retraining is cheap once this
  change lands, which is the point of the change.
- **The measured speedup disappoints on other hardware** → the throughput scenario is written
  against the reference Apple Silicon machine and phrased as "materially less than ~11 s/step"
  rather than a fixed target, so it stays honest on a different device.
- **Readability regresses for a tutorial file** → the shape-annotation comment style is kept,
  rewritten for the batched shapes rather than deleted.

## Migration Plan

No deployment surface — this is a local training repo. The only migration concern is
checkpoints, and there are none for model8. `tinyshakespeare_en_en_model8/` holds
`config.yaml` and `tokenizer_en.json` only, both unaffected.

Rollback is `git revert`; nothing persists state in the new layout until someone trains.
