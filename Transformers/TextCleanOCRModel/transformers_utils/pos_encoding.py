import yaml
import math
import torch
import torch.nn as nn

with open('settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int = settings['EMBED_DIM'],
        max_seq_len: int = settings['MAX_SEQ_LEN']
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        pos = torch.arange(self.max_seq_len).unsqueeze(1)  # cria um vetor que vai do 0 até o max_seq_len - 1 de shape (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
        pe = torch.zeros(self.max_seq_len, 1, self.embed_dim)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)  # preenche os valores pares
        pe[:, 0, 1::2] = torch.cos(pos * div_term)  # preenche os valores impares
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)  # faz esse tensor não ser treinavel

    def forward(self, x):
        x = x.clone()
        x += self.pe
        return x
