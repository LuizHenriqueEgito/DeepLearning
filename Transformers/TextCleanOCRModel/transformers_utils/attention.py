import yaml
import torch
import torch.nn as nn


with open('settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int = settings['N_HEADS'],
        embed_dim: int = settings['EMBED_DIM']
    ):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        assert (embed_dim % n_heads == 0), "embed_dim % n_heads != 0"

        # Querys
        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
        # Keys
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
        # Values
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)

        # MultiHeadAttention
        self.multihead_att = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=self.n_heads, 
            batch_first=True
        )

    def forward(self, x):
        querys = self.proj_q(x)
        keys = self.proj_k(x)
        values = self.proj_v(x)

        attn_output, attn_output_weights = self.multihead_att(querys, keys, values)
        return attn_output
