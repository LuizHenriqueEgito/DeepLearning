import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from sinusoidal_positional_encoding import PositionalEncoding
from embedding import EmbeddingModel

class EncoderBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_heads: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, hidden_size, dropout)

        self.norm_mha = nn.LayerNorm(normalized_shape=d_model)
        self.norm_feed_forward = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):
        x = self.norm_mha(x + self.mha(x, x, x))  # add & norm
        x = self.norm_feed_forward(x + self.feed_forward(x))  # add & norm
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        nx: int,
        n_heads: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            EmbeddingModel(d_model, vocab_size),
            PositionalEncoding(d_model, seq_len)
        )
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    n_heads,
                    hidden_size,
                    dropout=dropout,
                )
                for _ in range(nx)
            ]
        )

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x

