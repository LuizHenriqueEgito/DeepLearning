import torch
import torch.nn as nn
from torch import Tensor

class EmbeddingModel(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        return x