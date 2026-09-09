## 1. Pin the current behaviour before changing it

- [x] 1.1 Record a baseline step time for the committed config on this machine — run
      `train.py` with `max_iters` set low at `d_model 384, N 6, h 6, block_size 256,
      batch_size 64` and note seconds/step, so the "materially faster" claim has a
      before-number measured today rather than one quoted from June
      → **measured 0.438 s/step** (median of 10, after 3 warmup, MPS-synchronised), against
      the real `train_model8` inner step: `train_ds.get_batch()` + `.to(device)` +
      forward/backward/`optimizer.step()`, config loaded from
      `tinyshakespeare_en_en_model8/config.yaml`, 10.79M params, vocab 68. `get_batch` alone
      is 0.3 ms. **5000 iters = 0.61 h, not ~15 h.** Instrumented the step directly rather
      than via `train.py` because `train_model8` ends with a 2000-token generation and a
      checkpoint write, which would swamp the measurement; same instrument will be reused
      for task 4.1. **This killed the original premise** (~11 s/step / ~15 h): the committed config already trained in ~36 min. `proposal.md` has since been rebuilt around memory scaling. Peak memory was NOT captured on MPS — `torch.mps` exposes no equivalent of `max_memory_allocated`, which is why the memory evidence is CUDA-only and task 4.1 measures time here and memory there.
- [ ] 1.2 Add `tests/test_model8_attention.py` carrying a `_ReferenceHead` /
      `_ReferenceMultiHeadAttention` copy of the current per-head implementation, verbatim
      including the `C**-0.5` scale; verify the file imports and the reference runs a forward
      pass at toy dims
- [ ] 1.3 Add a weight-copy helper mapping per-head `query`/`key`/`value` weights into the
      fused `c_attn` layout (head `i` occupies rows `[i*hs:(i+1)*hs]` within each of the q, k,
      v thirds); verify it round-trips by asserting shapes and a slice-by-slice comparison
- [ ] 1.4 Add the equivalence test: same weights, same input, both in eval mode, outputs
      agree under `torch.testing.assert_close`. Verify it **fails** against a deliberately
      wrong scale (drop the `scale=` argument) before the rewrite exists, so it is known to
      have teeth — but run that negative control at `h=6` and `h=12` only: at `h=1`
      `head_size == n_embd`, the two scales are numerically identical (the `sqrt(h)` result),
      and the control cannot fail. Skip it there with a stated reason or assert the
      degeneracy, so nobody later reads it as a defect

## 2. Rewrite the attention

- [ ] 2.1 Rewrite `MultiHeadAttention` in `model8.py` with a fused
      `c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)`, reshape to `(B, h, T, hs)`, one
      `F.scaled_dot_product_attention(..., is_causal=True, scale=n_embd**-0.5,
      dropout_p=... if self.training else 0.0)`, then reshape back and apply the existing
      `proj` + residual dropout; verify the equivalence test from 1.4 passes
- [ ] 2.2 Add the `num_heads * head_size == n_embd` assertion in `__init__` with a readable
      message; verify a deliberately mismatched pairing raises it rather than failing deep in
      `proj`
- [ ] 2.3 Delete the `Head` class and its per-head `tril` buffer; verify nothing references
      `Head` (`grep -n "Head" model8.py`) and `uv run pytest` stays green
- [ ] 2.4 Rewrite the shape-annotation comments for the batched tensors — this file is a
      tutorial, so the `(B, T, C)` running commentary has to describe the new shapes rather
      than be dropped; verify by reading the diff

## 3. Verify the behavioural contract

- [ ] 3.1 Add a causality test: perturb the last token of the input, assert the outputs at
      every earlier position are unchanged; verify it passes
- [ ] 3.2 Add an eval-mode determinism test: two forward passes over the same input in eval
      mode return identical outputs (no dropout applied); verify it passes
- [ ] 3.3 Confirm the existing `tests/test_models.py` model8 cases (logit shape, loss present
      with targets, loss `None` without) still pass unchanged
- [ ] 3.4 Run `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` and
      `uv run mypy .`; verify all four exit 0

## 4. Measure and record the result

- [ ] 4.1 Re-measure at the committed config with the same instrument as 1.1 and record
      before/after. Time on this machine (expect no regression; ~25-30% is the estimate, not
      the requirement). **Peak memory is the load-bearing number and needs the CUDA box** —
      ask the paired WSL2 session to re-run its h-sweep against the rewritten `model8.py`,
      since `torch.mps` has no `max_memory_allocated` equivalent. The claim to verify is that
      fused peak memory is flat across `h`, matching the 2.11/2.11/2.11/2.12 GiB it measured
      from the prototype
- [ ] 4.2 Train a short run end to end (`train.py` at reduced `max_iters`) and confirm loss
      descends and a checkpoint is written, then delete the toy checkpoint so a later
      `preload: latest` cannot pick it up
- [ ] 4.3 Mark the model8 batched-attention entry `[x]` in `openspec/ideas.md`, confirming the
      shipped numbers match the prototype's. The entry already carries the measurements, the
      corrected ~11 s/step history, the disproved OOM hypothesis, and the separate
      "which attention scale is correct" entry (all landed in PR #15) — so this is a
      confirm-and-close, not a write-up
- [ ] 4.4 Note the model8 checkpoint-compatibility break in `architecture.md`'s known gotchas
      (existing `.pt` files will not load: QKV layout changed and the per-head `tril` buffers
      left `state_dict`); verify by reading the rendered section. The WSL2 spill behaviour is
      already documented in README's limited-VRAM notes — do not duplicate it
