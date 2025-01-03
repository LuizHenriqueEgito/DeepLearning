import math

import torch
import torch.nn as nn
import numpy as np


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class PositionalEncoding(nn.Module):
    def __init__(
        self, model_dimension, dropout_probability, expected_max_sequence_length=5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(
            10000.0,
            -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension,
        )

        positional_encodings_table = torch.zeros(
            expected_max_sequence_length, model_dimension
        )
        positional_encodings_table[:, 0::2] = torch.sin(
            position_id * frequencies
        )  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(
            position_id * frequencies
        )  # cosine on odd positions

        self.register_buffer("positional_encodings_table", positional_encodings_table)

    def forward(self, embeddings_batch):
        assert (
            embeddings_batch.ndim == 3
            and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1]
        ), f"Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}"

        positional_encodings = self.positional_encodings_table[
            : embeddings_batch.shape[1]
        ]

        return self.dropout(embeddings_batch + positional_encodings)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int = 8, embed_dim: int = 512, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert (
            embed_dim % num_heads == 0
        ), "The number of dimensions must be divible by the number of heads"

        self.head_dim = embed_dim // num_heads
        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_projection = nn.Dropout(dropout)

    def reshape_for_attention(self, x):
        B, L, E = x.shape
        # shape x = [batch, len, embed_dim]
        # precisa virar [batch * heads, len, head_dim]
        x = x.contiguous().view((B, L, self.num_heads, self.head_dim)).transpose(1, 2)
        # virou [batch, heads, len, head_dim]
        # x = x.contiguous().view((B * self.num_heads, L, self.head_dim))
        return x

    def reshape_from_attention(self, x):
        B, H, L, HD = x.shape
        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x = x.transpose(1, 2)
        # virou [batch, len, heads, head_dim]
        x = x.contiguous().view((B, L, self.embed_dim))
        # virou [batch, len, embed_dim]
        return x

    def QKVattention(self, q, k, v, mask=None):
        b, heads, len_tokens, embed_dim = q.shape
        k_t = torch.transpose(k, -1, -2)
        # shapes for q, k, v are [B, HEADS, SEQ, HEAD_DIM]
        # for K_t we have [B, HEADS, HEAD_DIM, SEQ]
        qk = torch.einsum("bhsd, bhde -> bhse", q, k_t)
        # qk = torch.bmm(q, k_t)
        # shape of qk is [B, SEQ, SEQ]
        if mask is not None:
            qk = qk.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(qk / np.sqrt(embed_dim), dim=-1)
        attention = self.dropout_attention(attention)
        # [batch, heads, decoder_len, head_dim] * [batch, heasd, encoder_len, head_dim]
        full_attention = torch.einsum("bhde, bher -> bhdr", attention, v)
        return self.dropout_projection(full_attention)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        q = self.reshape_for_attention(self.proj_q(x))
        k = self.reshape_for_attention(self.proj_k(x))
        v = self.reshape_for_attention(self.proj_v(x))

        x_att = self.QKVattention(q, k, v, mask)

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att = self.reshape_from_attention(x_att)

        # projecao final
        x_att = self.out_projection(x_att)
        return x_att


class FeedFowardBlock(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout: int = 0.1):
        super().__init__()

        self.ff_1 = nn.Linear(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ff_2 = nn.Linear(hidden_size, embed_dim)
        self.activation = NewGELU()

    def forward(self, x):
        x = self.ff_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff_2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, dropout: int = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x, mask):
        x = x + self.attention(self.norm_1(x), mask)
        x = x + self.feedforward(self.norm_2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim,
        num_layers,
        num_heads,
        hidden_size,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(
            model_dimension=embed_dim,
            dropout_probability=dropout,
            expected_max_sequence_length=max_len,
        )

        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_blocks.append(
                DecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
            )

        self.last_norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x):
        token_len = x.shape[1]
        causal_mask = (
            torch.tril(torch.ones((token_len, token_len)))
            .view((1, 1, token_len, token_len))
            .to(x.device)
        )

        x = self.embedding(x)
        x = self.pos_encoding(x)

        for block in self.decoder_blocks:
            x = block(x, causal_mask)

        return self.last_norm(x)
