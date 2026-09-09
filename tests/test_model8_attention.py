"""model8's batched attention must compute the same function as the per-head version it replaced.

`_ReferenceHead` / `_ReferenceMultiHeadAttention` below are a verbatim copy of the
`nn.ModuleList`-of-`Head` implementation that `model8.MultiHeadAttention` used before it was
rewritten around a fused QKV projection and `F.scaled_dot_product_attention`. Keeping the old
code here rather than in `model8.py` is what makes "numerically equivalent" a checked claim
instead of an assertion in a commit message.

Three caveats on what these tests do and do not establish:

- **Eval mode only.** Dropout consumes RNG differently in the two implementations — one fused
  draw against `h` separate ones — so training runs will not match step for step even from
  identical seeds. Equivalence is asserted with dropout disabled.
- **fp32, no autocast.** Nothing here says anything about mixed-precision behaviour.
- The `scale` argument is load-bearing. model8 scales attention scores by the **full embedding
  dim** (`n_embd**-0.5`), not the head size that `scaled_dot_product_attention` defaults to.
  `test_negative_control_*` exists to prove these tests would notice if that were dropped.
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from model8 import MultiHeadAttention

N_EMBD = 96
BLOCK = 16
BATCH = 4
ATOL = 1e-5
RTOL = 1e-5


class _ReferenceHead(nn.Module):
    """One head of self-attention — the pre-rewrite `model8.Head`, copied verbatim."""

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        _B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # C is the FULL embedding dim here, not head_size — the scale model8 has always used.
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class _ReferenceMultiHeadAttention(nn.Module):
    """The pre-rewrite `model8.MultiHeadAttention`, copied verbatim."""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([_ReferenceHead(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


def _copy_weights(ref: _ReferenceMultiHeadAttention, fused: nn.Module, n_embd: int, head_size: int) -> None:
    """Head i's q/k/v rows -> the q/k/v blocks of c_attn, the layout
    view(B, T, nh, hs).transpose(1, 2) reassembles."""
    with torch.no_grad():
        for i, head in enumerate(ref.heads):
            lo, hi = i * head_size, (i + 1) * head_size
            fused.c_attn.weight[lo:hi] = head.query.weight
            fused.c_attn.weight[n_embd + lo : n_embd + hi] = head.key.weight
            fused.c_attn.weight[2 * n_embd + lo : 2 * n_embd + hi] = head.value.weight
        fused.proj.weight.copy_(ref.proj.weight)
        fused.proj.bias.copy_(ref.proj.bias)


def _pair(num_heads: int) -> tuple[_ReferenceMultiHeadAttention, nn.Module, Tensor]:
    """A weight-matched (reference, fused) pair in eval mode, plus an input."""
    head_size = N_EMBD // num_heads
    torch.manual_seed(0)
    ref = _ReferenceMultiHeadAttention(num_heads, head_size, N_EMBD, BLOCK, 0.0).eval()
    fused = MultiHeadAttention(num_heads, head_size, N_EMBD, BLOCK, 0.0).eval()
    _copy_weights(ref, fused, N_EMBD, head_size)
    return ref, fused, torch.randn(BATCH, BLOCK, N_EMBD)


# --- 1.2: the reference implementation itself runs -----------------------------------------


@pytest.mark.parametrize("num_heads", [1, 4, 6])
def test_reference_implementation_runs(num_heads: int) -> None:
    ref, _fused, x = _pair(num_heads)
    with torch.no_grad():
        out = ref(x)
    assert out.shape == (BATCH, BLOCK, N_EMBD)
    assert torch.isfinite(out).all()


# --- 1.3: the weight mapping ----------------------------------------------------------------


@pytest.mark.parametrize("num_heads", [1, 4, 6])
def test_weight_copy_places_every_head_in_its_slice(num_heads: int) -> None:
    ref, fused, _x = _pair(num_heads)
    head_size = N_EMBD // num_heads
    assert fused.c_attn.weight.shape == (3 * N_EMBD, N_EMBD)
    for i, head in enumerate(ref.heads):
        lo, hi = i * head_size, (i + 1) * head_size
        torch.testing.assert_close(fused.c_attn.weight[lo:hi], head.query.weight)
        torch.testing.assert_close(fused.c_attn.weight[N_EMBD + lo : N_EMBD + hi], head.key.weight)
        torch.testing.assert_close(fused.c_attn.weight[2 * N_EMBD + lo : 2 * N_EMBD + hi], head.value.weight)


# --- 1.4: equivalence, and proof the check has teeth ----------------------------------------


@pytest.mark.parametrize("num_heads", [1, 4, 6])
def test_fused_matches_per_head_reference(num_heads: int) -> None:
    ref, fused, x = _pair(num_heads)
    with torch.no_grad():
        torch.testing.assert_close(fused(x), ref(x), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("num_heads", [4, 6])
def test_negative_control_wrong_scale_breaks_equivalence(num_heads: int) -> None:
    """Dropping model8's scale must be detected — otherwise the test above proves nothing.

    h=1 is excluded deliberately: model8's `n_embd**-0.5` and SDPA's `head_size**-0.5` differ
    by exactly sqrt(h), so at h=1 (`head_size == n_embd`) they are the same number and this
    control cannot fail. That is arithmetic, not a defect — see `test_scales_coincide_at_one_head`.
    """
    ref, fused, x = _pair(num_heads)
    head_size = N_EMBD // num_heads
    with torch.no_grad():
        expected = ref(x)
        # Recompute exactly as `fused.forward` does, but at SDPA's default scale.
        B, T, C = x.shape
        q, k, v = fused.c_attn(x).split(C, dim=2)
        reshape = lambda t: t.view(B, T, num_heads, head_size).transpose(1, 2)  # noqa: E731
        y = F.scaled_dot_product_attention(reshape(q), reshape(k), reshape(v), is_causal=True)
        wrong = fused.proj(y.transpose(1, 2).contiguous().view(B, T, C))
    assert not torch.allclose(wrong, expected, atol=ATOL, rtol=RTOL)


def test_scales_coincide_at_one_head() -> None:
    """Why the negative control skips h=1: the two scales are the same number there."""
    assert pytest.approx(N_EMBD**-0.5) == (N_EMBD // 1) ** -0.5
    assert pytest.approx(N_EMBD**-0.5) != (N_EMBD // 6) ** -0.5


# --- 3.1 / 3.2: the behavioural contract ----------------------------------------------------


def test_attention_is_causal() -> None:
    """Changing a later token must not move any earlier position's output."""
    _ref, fused, x = _pair(6)
    perturbed = x.clone()
    perturbed[:, -1, :] += 1.0
    with torch.no_grad():
        torch.testing.assert_close(fused(perturbed)[:, :-1, :], fused(x)[:, :-1, :], atol=ATOL, rtol=RTOL)


def test_eval_mode_is_deterministic() -> None:
    """No attention dropout in eval mode, so repeated passes agree exactly."""
    _ref, _fused, x = _pair(6)
    # dropout 0.5, so a leak would show: eval() must suppress it entirely.
    fused_dropout = MultiHeadAttention(6, N_EMBD // 6, N_EMBD, BLOCK, 0.5).eval()
    with torch.no_grad():
        torch.testing.assert_close(fused_dropout(x), fused_dropout(x), rtol=0, atol=0)


def test_head_size_must_tile_the_embedding() -> None:
    """The invariant was implicit before; a bad pairing should say so, not fail inside proj."""
    with pytest.raises(AssertionError):
        MultiHeadAttention(num_heads=5, head_size=7, n_embd=N_EMBD, block_size=BLOCK, dropout=0.0)
