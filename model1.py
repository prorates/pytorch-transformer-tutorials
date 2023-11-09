import torch
import torch.nn as nn
import math
from torch import Tensor

# Layer normalization. Minute 14:00. Each sentence is made of many words
# For each sentence compute the mean and variance for each item/sentence
# Parameters alpha/beta....gamma(multiplicative)/beta(additive)....alpha/bias
# the network will learn to tune these two parameters to introduce fluctations when necessary


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        # If sigma is getting to 0, x will be very big. Also want to avoid to devide by 0
        self.eps = eps
        # Parameter is learnable. Multiplied
        self.alpha = nn.Parameter(torch.ones(features))
        # Parameter is learnable. Multiplied
        # JEB: Need to undertand why bias is set to 0 and alpha to 1
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x) -> Tensor:
        # usually the mean cancels the dimension to which it is applied
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# Feed foward: Fully connected layer. Two matrices which are multiplied with the relu in between
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x) -> Tensor:
        # (Batch, Seq_Len, d_model) --> (Batch, Seql_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# InputEmbeddings converts each word/token to vector of size 512
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # 512 in the paper
        self.vocab_size = vocab_size
        # pytorch already provide with a layer mapping between a number and vector of size 512 (3:50)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x) -> Tensor:
        # kind of a dictionary. embedding maps number to the same vector every time.
        # The vector is learned by the model.
        # see 3.4 of the paper
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


# PositionalEncodding encodes the postion of the words.
# Needs to have the same size has the InputEmbedding, so vector of size 512
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len  # we need a vector for each position.
        # we need to get the model less overfit.
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        # seq_len is the maximum length of the sentence
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Compute in the log space for better mathematical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))

        # we will have a batch of sentences
        pe = pe.unsqueeze(0)  # 1 (1, Seq_Len, d_model)

        # we want the tensor to be saved with the model but not as parameter
        self.register_buffer('pe', pe)

    def forward(self, x) -> Tensor:
        # we need to the positional encoding to every word in the sentence
        # that self.pe is fixed and does not need to be learned, hence requires_grad set to False
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer) -> Tensor:
        # Defintion of add and norm
        # We apply the norm and then the sublyaer.
        # The paper seems to apply the sublayer and then the normalization
        return x + self.dropout(sublayer(self.norm(x)))


# We have the input model (seq,d_model)
# We copy the input model in Q K and V (seq,d_model)
# We multiply Q by WQ , K by WK and V by QV (d_model, d_model)
# We obtain Q', K' and V'
# We split those matrices along the d_model dimension, not along the sequence dimension.
# Each head has access to the full sequence but a difference part of the embedding
# We apply attention to the the smaller matrcices Q1, K1 and V1...to obtain H1
# We combine back teh HEAD matrices and multtiple by W0.
# We obtain a matrix with the same dimension as the input matrix.
# We also have to account for the batch dimension if you have more than one sentence.

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        assert d_model % h == 0, "d_model must be multiple of h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # JEB: Original video left bias to True
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)  # (h * dv * dk)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        # we transpose the last two dimension. key is ...Seq_Len, d_k so it becomes...d_k, Seq_Len
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Replace all the value for which mask == 0 with -1e9 (minus infinity)
            # Some words will not be able to see future words...or padding values
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, Seq_Len, Seq_Len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # We return a tuple....attention_scores is mainly used for visualizing
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask) -> Tensor:
        # mask avoid that some word interact with some other workds
        # we need to put a value very small to the matrix before we apply
        # the soft max. e to the power of infinity will be very small.
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Batch dimension is preserved, Seuence dimension is preserved,
        # We want the h dimension to be the second dimension hence we invoke the transpose method.
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_Len, d_K) --> (Batch, Sql_Len, h, d_K) -->  (Batch, Seq_Len, d_model
        # Pytorch needs the memory to be continguouos to create a view
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, Sql_Len, d_model)
        return self.w_o(x)


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    # src_mask is hiding the padding words
    def forward(self, x, src_mask) -> Tensor:
        # query, key, value are the same
        # this invokes the forward function of the MultiHeadAttention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # second connection is the feedfoward. No Lambda needed. Why ?
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask) -> Tensor:
        for layer in self.layers:
            # The ouput of the previous layer is the input for the next layer
            # See the forward method of the EncoderBlock
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    # src_mask is hiding the padding words
    def forward(self, x, encoder_ouput, src_mask, tgt_mask) -> Tensor:
        # query, key, value are the same
        # this invokes the forward function of the MultiHeadAttention
        # x (decoder side) is used for q, k and v
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # x (decoder side) is used for q. k and v are coming from encoder side
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,
                                         encoder_ouput, encoder_ouput, src_mask))
        # third connection is the feedfoward. No Lambda needed. Why ?
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask) -> Tensor:
        for layer in self.layers:
            # The ouput of the previous layer is the input for the next layer
            # See the forward method of the DecoderBlock
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    # Need to convert the embedding back into a position in the vocabulary
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> Tensor:
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, Vocab_Size)
        # JEB: For some reasons log_softmax has been removed. This needs to be studied
        # return torch.log_softmax(self.proj(x), dim=-1)
        return self.proj(x)


class Transformer1(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # during inference we can reuse the output of the decoder.
    def encode(self, src, src_mask: Tensor) -> Tensor:
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: Tensor, src_mask: Tensor, tgt: Tensor, tgt_mask: Tensor) -> Tensor:
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x) -> Tensor:
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer1(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                       d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer1:
    # Create the embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # In theory no replicated needed. but provide better understand
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        # JEB: d_model has been added has feature
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            decoder_feed_forward_block,
            dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer1(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
