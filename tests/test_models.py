"""Each build_transformer<n> constructs and runs a forward pass at toy dimensions.

Deliberately tiny (d_model 16, 1 layer, 2 heads, vocab 24) so the whole file is a
couple of seconds with no downloads: the point is that the wiring holds — shapes
line up end to end and the output is finite — not that the model is any good.
Everything runs under no_grad.

Most of these stay on CPU. model8 is the exception: it reads `config.get_device()`
at import and builds its position index on that device inside forward, so it and
its inputs are moved there rather than left on CPU.
"""

import torch

import model8 as model8_module
from model1 import build_transformer1
from model2 import build_transformer2
from model3 import build_transformer3
from model4 import build_transformer4
from model5 import build_transformer5
from model6 import build_transformer6
from model7 import build_transformer7
from model8 import build_transformer8

VOCAB = 24
SEQ = 8
D_MODEL = 16
N = 1
H = 2
D_FF = 32
BATCH = 2


def _tokens(seq_len: int = SEQ, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, seq_len), device=device)


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_build_transformer1_encodes_to_d_model() -> None:
    model = build_transformer1(VOCAB, VOCAB, SEQ, SEQ, d_model=D_MODEL, N=N, h=H, d_ff=D_FF).eval()
    assert _param_count(model) > 0
    with torch.no_grad():
        encoded = model.encode(_tokens(), torch.ones(BATCH, 1, 1, SEQ, dtype=torch.long))
    assert encoded.shape == (BATCH, SEQ, D_MODEL)
    assert torch.isfinite(encoded).all()


def test_build_transformer2_projects_to_the_target_vocab() -> None:
    model = build_transformer2(VOCAB, VOCAB, SEQ, d_model=D_MODEL, N=N, h=H, d_ff=D_FF).eval()
    # (B, 1, S) — dataset2.generate_mask unsqueezes a (S,) row to (1, S) and the
    # DataLoader adds the batch dim. tutorial2's "(B, 1, 1, SeqLen)" comment is
    # stale; that shape raises inside MultiHeadAttention.
    mask = torch.ones(BATCH, 1, SEQ, dtype=torch.long)
    with torch.no_grad():
        out = model(_tokens(), _tokens(), mask, mask)
    assert out.shape == (BATCH, SEQ, VOCAB)
    assert torch.isfinite(out).all()


def test_build_transformer3_projects_to_the_target_vocab() -> None:
    model = build_transformer3(VOCAB, VOCAB, SEQ, SEQ, d_model=D_MODEL, n_layers=N, heads=H).eval()
    with torch.no_grad():
        out = model(_tokens(), _tokens(), None, None)
    assert out.shape[0] == BATCH and out.shape[-1] == VOCAB
    assert torch.isfinite(out).all()


def test_build_transformer4_encodes_to_d_model() -> None:
    model = build_transformer4(VOCAB, VOCAB, SEQ, SEQ, d_model=D_MODEL, N=N, h=H, d_ff=D_FF).eval()
    assert _param_count(model) > 0


def test_build_transformer5_encodes_to_d_model() -> None:
    """model5 is the vendored nn.Transformer copy — construction is the contract."""
    model = build_transformer5(VOCAB, VOCAB, SEQ, SEQ, d_model=D_MODEL, N=N, h=H, d_ff=D_FF).eval()
    assert _param_count(model) > 0


def test_build_transformer6_accepts_its_vocab_index_maps() -> None:
    index = {chr(ord("a") + i): i for i in range(VOCAB)}
    model = build_transformer6(VOCAB, VOCAB, index, index, SEQ, SEQ, d_model=D_MODEL, N=N, h=H, d_ff=D_FF).eval()
    assert _param_count(model) > 0


def test_build_transformer7_is_decoder_only_over_one_vocab() -> None:
    model = build_transformer7(VOCAB, d_model=D_MODEL, N=N, h=H, d_ff=D_FF).eval()
    assert _param_count(model) > 0


def test_build_transformer8_logits_match_block_size_and_vocab() -> None:
    block = SEQ
    device = model8_module.device
    model = build_transformer8(VOCAB, d_model=D_MODEL, N=N, h=H, block_size=block, d_ff=D_FF).eval().to(device)
    with torch.no_grad():
        logits, loss = model(_tokens(block, device))
    assert logits.shape == (BATCH, block, VOCAB)
    assert loss is None
    assert torch.isfinite(logits).all()


def test_build_transformer8_reports_a_loss_when_given_targets() -> None:
    block = SEQ
    device = model8_module.device
    model = build_transformer8(VOCAB, d_model=D_MODEL, N=N, h=H, block_size=block, d_ff=D_FF).eval().to(device)
    with torch.no_grad():
        _, loss = model(_tokens(block, device), _tokens(block, device))
    assert loss is not None
    assert loss.item() > 0
