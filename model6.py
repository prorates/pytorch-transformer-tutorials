import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from config import SOS, EOS, PAD, UNK


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def scaled_dot_product(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int):
        super().__init__()
        self.max_sequence_length: int = max_sequence_length
        self.d_model: int = d_model

    def forward(self) -> Tensor:
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                    .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE


class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"

    def __init__(self, max_sequence_length: int, d_model: int, language_to_index: int, START_TOKEN: int, END_TOKEN: int, PADDING_TOKEN: int):
        super().__init__()
        self.vocab_size: int = len(language_to_index)
        self.max_sequence_length: int = max_sequence_length
        self.embedding: nn.Embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index: int = language_to_index
        self.position_encoder: PositionalEncoding = PositionalEncoding(d_model, max_sequence_length)
        self.dropout: nn.Dropout = nn.Dropout(p=0.1)
        self.START_TOKEN: int = START_TOKEN
        self.END_TOKEN: int = END_TOKEN
        self.PADDING_TOKEN: int = PADDING_TOKEN

    def batch_tokenize(self, batched_sentences: tuple[str], start_token: bool, end_token: bool):

        def tokenize(sentence: str, start_token: bool, end_token: bool) -> Tensor:
            # JEB: This tokenize function assumes that each caractere is a token.
            # The tokens are only 1 character long.
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized_sentences = []
        for sentence_num in range(len(batched_sentences)):
            tokenized_sentences.append(tokenize(batched_sentences[sentence_num], start_token, end_token))
        tokenized = torch.stack(tokenized_sentences)
        return tokenized.to(get_device())

    # fowards returns a (bs, SeqLen, d_model) tensor
    def forward(self, batched_sentences: tuple[str], start_token: bool, end_token: bool) -> Tensor:  # sentence
        x = self.batch_tokenize(batched_sentences, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    # fowards returns a (bs, SeqLen, d_model) tensor
    def forward(self, x, mask) -> Tensor:
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)  # (bs, SeqLen, d_model)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        # JEB: Neeed to study. Look like a list with value[d_model]
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    # fowards returns a (bs, SeqLen, d_model) tensor
    def forward(self, inputs: Tensor) -> Tensor:
        # inputs has shape (bs, SeqLen, d_model)
        # JEB: Need to come back to that dims computation
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    # fowards returns a (bs, SeqLen, d_model) tensor
    def forward(self, x: Tensor) -> Tensor:
        # the shape of input x is (bs, SqeqLen, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    # fowards returns a (bs, SeqLen, d_model) Tensor
    def forward(self, x: Tensor, self_attention_mask: Tensor) -> Tensor:
        # x input Tensor is a (bs, SeqLen, d_model) Tensor
        # self_attention_mask is a (bs, SeqLen, SeqLen) Tensor
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x


class SequentialEncoder(nn.Sequential):
    # SequentialEncoder is instantiate with a list of EncoderLayer

    # foward returns a (bs, SeqLen, d_model) tensor
    def forward(self, *inputs) -> Tensor:
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob, num_layers: int, max_sequence_length: int, language_to_index: dict,
                 START_TOKEN: int, END_TOKEN: int, PADDING_TOKEN: int):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(
            max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                          for _ in range(num_layers)])

    # forward returns a (bs, SeqLen, d_model) Tensor
    def forward(self, encoder_batched_sentences: tuple[str], self_attention_mask: Tensor, start_token: bool, end_token: bool) -> Tensor:
        # attention_mask shape is (bs, SeqLen, SeqLen)
        x = self.sentence_embedding(encoder_batched_sentences, start_token, end_token)  # (bs, SeqLen, d_model)
        x = self.layers(x, self_attention_mask)  # (bs, SeqLen, d_model)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    # forward returns a (bs, SeqLen, d_model) Tensor
    def forward(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        # The x shape is (bs, SeqLen, d_model)
        # The y shape is (bs, SeqLen, d_model)
        # The mask shape is (bs, SeqLen, SeqLen)
        # in practice, this is the same for both languages...so we can technically combine with normal attention
        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        # We don't need the mask for cross attention, removing in outer function!
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    # forward returns a (bs, SeqLen, d_model) tensor
    def forward(self, encoder_out: Tensor, y: Tensor, self_attention_mask: Tensor, cross_attention_mask: Tensor) -> Tensor:
        # encoder_out shape is (bs,SeqLen, d_model)
        # y shape is (bs_SeqLen, d_model)
        # self_attention_mask shape is (bs, SeqLen, SeqLen)
        # cross_attention_mask shape is (bs, SeqLen, SeqLen)
        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(encoder_out, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y


class SequentialDecoder(nn.Sequential):
    # SequentialDecoder is instantiate with a list of DecoderLayer

    # foward returns a (bs, SeqLen, d_model) tensor
    def forward(self, *inputs) -> Tensor:
        encoder_out, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(encoder_out, y, self_attention_mask, cross_attention_mask)
        return y


class Decoder(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob, num_layers: int, max_sequence_length: int, language_to_index: dict,
                 START_TOKEN: int, END_TOKEN: int, PADDING_TOKEN: int):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(
            max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(
            *[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    # forward returns a (bs, SeqLen, d_model) Tensor
    def forward(self, encoder_out: Tensor, decoder_batched_sentences: tuple[str], self_attention_mask: Tensor, cross_attention_mask: Tensor, start_token: bool, end_token: bool) -> Tensor:
        # The input of the decoder will include the start_token but not the end_token
        # The ouput of the decoder will not include the start_token but will include the end_token
        # encoder_out shape is (bs, SeqLen, d_model)
        # self_attention_mask shape is (bs, SeqLen, SeqLen)
        # cross_attention_mask shape is (bs, SeqLen, SeqLen)
        y = self.sentence_embedding(decoder_batched_sentences, start_token, end_token)  # (bs, SeqLen, d_model)
        y = self.layers(encoder_out, y, self_attention_mask, cross_attention_mask)  # (bs, SeqLen, d_model)
        return y


class Transformer6(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob, num_layers: int, max_sequence_length: int, kn_vocab_size: int,
                 english_to_index: dict, kannada_to_index: dict,
                 START_TOKEN: int, END_TOKEN: int, PADDING_TOKEN: int):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                               max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                               max_sequence_length, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, kn_vocab_size)
        self.device = get_device()  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # forward returns a (bs, SeqLen, vocab_size) Tensor
    def forward(self,
                encoder_batched_sentences: tuple[str],
                decoder_batched_sentences: tuple[str],
                encoder_self_attention_mask: Tensor = None,
                decoder_self_attention_mask: Tensor = None,
                decoder_cross_attention_mask: Tensor = None,
                enc_start_token: bool = False,
                enc_end_token: bool = False,
                dec_start_token: bool = False,  # JEB: ? We should make this true
                dec_end_token: bool = False) -> Tensor:
        encoder_out = self.encoder(encoder_batched_sentences, encoder_self_attention_mask,
                                   start_token=enc_start_token, end_token=enc_end_token)  # (bs, SeqLen, d_model)
        # The input of the decoder will include the start_token but not the end_token
        # The ouput of the decoder will not include the start_token but will include the end_token
        decoder_out = self.decoder(encoder_out, decoder_batched_sentences, decoder_self_attention_mask, decoder_cross_attention_mask,
                                   start_token=dec_start_token, end_token=dec_end_token)  # (bs, SeqLen, d_model)
        decoder_out = self.linear(decoder_out)  # (bs, SeqLen, vocab_size)
        return decoder_out


def build_transformer6(src_vocab_size: int, tgt_vocab_size: int, src_to_index: dict, tgt_to_index: dict,
                       src_seq_len: int, tgt_seq_len: int,
                       d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer6:

    # Create the transformer
    # JEB: TODO. Need to use the tokenizer instead
    transformer = Transformer6(d_model=d_model, ffn_hidden=d_ff, num_heads=h, drop_prob=dropout, num_layers=N, max_sequence_length=tgt_seq_len,
                               kn_vocab_size=tgt_vocab_size, english_to_index=src_to_index, kannada_to_index=tgt_to_index, START_TOKEN=SOS, END_TOKEN=EOS, PADDING_TOKEN=PAD)

    # When computing the loss, we are ignoring cases when the label is the padding token
    for params in transformer.parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)

    return transformer
