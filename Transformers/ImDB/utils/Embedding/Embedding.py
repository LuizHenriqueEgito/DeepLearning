import torch
import torch.nn as nn
from torch import Tensor

class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=30_000,  # vocab_size
            embedding_dim=8,
            padding_idx=0
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        return x