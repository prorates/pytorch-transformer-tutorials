## ADDED Requirements

### Requirement: Attention is computed batched across heads

Multi-head self-attention in the GPT-style model SHALL be computed for all heads in a single
batched operation rather than by iterating per-head modules in Python, and SHALL NOT
materialise a per-head attention matrix. Consequently peak training memory SHALL be
independent of head count, where the per-head implementation's grows roughly in proportion
to it.

The rewrite SHALL NOT alter the model's mathematics: for identical weights and inputs, the
batched implementation SHALL produce the same outputs as the per-head implementation it
replaces, including the existing attention score scale, causal masking, and dropout
semantics.

#### Scenario: One dispatch per attention layer

- **WHEN** a forward pass runs through an attention layer configured with `h` heads
- **THEN** the query, key, and value projections are computed as one operation covering all
  `h` heads, not `h` separate operations

#### Scenario: Numerically equivalent to the per-head implementation

- **WHEN** the batched implementation and the per-head implementation are given the same
  weights and the same input tensor, both in eval mode
- **THEN** their outputs agree to floating-point tolerance

#### Scenario: Peak memory does not grow with head count

- **WHEN** the model is trained at a fixed batch size and embedding width, with head count
  varied across the supported range
- **THEN** peak memory is substantially flat across head counts, rather than rising roughly in
  proportion to them as the per-head implementation does

#### Scenario: More headroom before the memory-pressure regime

- **WHEN** batch size is raised until training degrades from memory pressure — which on some
  platforms means silent slowdown rather than an out-of-memory error
- **THEN** the batched implementation reaches that point at a materially larger batch than the
  per-head implementation it replaces, at every supported head count

#### Scenario: A training step is no slower than before

- **WHEN** the model trains at the committed configuration (`d_model` 384, `N` 6, `h` 6,
  `block_size` 256, `batch_size` 64)
- **THEN** a training step completes in no more time than the per-head implementation took on
  the same machine, measured with the same instrument

#### Scenario: Attention remains causal

- **WHEN** the model attends over a sequence of length `T`
- **THEN** position `i` attends only to positions `j <= i`, so a change to a later token
  cannot alter the output at an earlier one

#### Scenario: Dropout applies only while training

- **WHEN** the model is in eval mode
- **THEN** no attention dropout is applied and repeated forward passes over the same input
  return identical outputs
