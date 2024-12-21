import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        seq_len: int,
        dropout: float
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_len, d_model)
        k = torch.arange(0, seq_len).unsqueeze(1)
        div_term = (
            torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10_000.0)) / d_model))
        )
        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x += self.pe[:, :x.size(1)].requires_grad_(False)
        x = self.dropout(x)
        return x