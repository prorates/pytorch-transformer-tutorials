# See [video](https://youtu.be/kCc8FmEb1nY)
# The colab repo is [here](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

# JEB: Ugly but will do for right now
from config import get_device

device = get_device()


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention, computed as one batched operation"""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float) -> None:
        super().__init__()
        # The heads tile the embedding: concatenating their outputs has to land back on
        # n_embd for self.proj. That was always required — a bad pairing used to fail deep
        # inside proj with a shape error — and the fused projection makes it structural.
        assert num_heads * head_size == n_embd, f"num_heads * head_size ({num_heads} * {head_size}) must equal n_embd ({n_embd})"
        self.num_heads = num_heads
        self.head_size = head_size
        # One projection for all heads and all three roles: (B,T,C) -> (B,T,3C). This replaces
        # 3 * num_heads separate nn.Linear calls, which is the whole point — the arithmetic was
        # never the cost, the per-head dispatch was.
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = dropout
        self.dropout = nn.Dropout(dropout)
        # No `tril` buffer: F.scaled_dot_product_attention applies the causal mask itself via
        # is_causal, so the model no longer carries num_layers * num_heads identical copies.

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        # Split the fused projection back into the three roles, each still (B,T,C).
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs): heads become a batch dimension, so one
        # attention call covers all of them. Head i keeps columns [i*hs:(i+1)*hs], the same
        # layout the old torch.cat over per-head outputs produced.
        q, k, v = (t.view(B, T, self.num_heads, self.head_size).transpose(1, 2) for t in (q, k, v))
        # scale= is not decoration: model8 has always divided by sqrt of the FULL embedding
        # dim, where this function defaults to sqrt of the head size. They differ by sqrt(h).
        # Passing it keeps the rewrite numerically equivalent; dropping it silently retunes
        # the model. tests/test_model8_attention.py has a negative control for exactly this.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=C**-0.5,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        # (B,nh,T,hs) -> (B,T,nh,hs) -> (B,T,C), reassembling the heads side by side.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # DFF is 4 time n_embd
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size, n_embd=n_embd, block_size=block_size, dropout=dropout)
        self.ffwd = FeedFoward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        # JEB: This is one of the only that changed compared to the original
        # paper. The normalization is made first in this model.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model


class Transformer8(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        # JEB: Broadcasting. pos_emb gets right-aligned, a new dimension is added
        # and it gets added accross batch.
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # JEB: Interesting. This model computes the loss
            # in the forward method.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions. (We invoke forward here with a target)
            logits, _loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def build_transformer8(
    tgt_vocab_size: int, d_model: int = 64, N: int = 4, h: int = 4, block_size: int = 32, dropout: float = 0.0, d_ff: int = 256
) -> Transformer8:

    # Create the transformer
    transformer = Transformer8(vocab_size=tgt_vocab_size, n_embd=d_model, n_head=h, n_layer=N, block_size=block_size, dropout=dropout)

    # When computing the loss, we are ignoring cases when the label is the padding token
    # for params in transformer.parameters():
    #     if params.dim() > 1:
    #         nn.init.xavier_uniform_(params)

    return transformer
