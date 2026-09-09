## ADDED Requirements

### Requirement: Attention is computed batched across heads

Multi-head self-attention in the GPT-style model SHALL be computed for all heads in a single
batched operation rather than by iterating per-head modules in Python. Training throughput at
the committed configuration SHALL be dominated by tensor arithmetic rather than by per-head
kernel dispatch.

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

#### Scenario: Committed configuration is trainable in practice

- **WHEN** the GPT-style model trains at the committed configuration (`d_model` 384, `N` 6,
  `h` 6, `block_size` 256, `batch_size` 64) on the reference Apple Silicon machine
- **THEN** a training step completes in materially less time than the ~11 s/step measured
  before this change, bringing the 5000-iteration loop within a few hours rather than ~15 h

#### Scenario: Attention remains causal

- **WHEN** the model attends over a sequence of length `T`
- **THEN** position `i` attends only to positions `j <= i`, so a change to a later token
  cannot alter the output at an earlier one

#### Scenario: Dropout applies only while training

- **WHEN** the model is in eval mode
- **THEN** no attention dropout is applied and repeated forward passes over the same input
  return identical outputs
