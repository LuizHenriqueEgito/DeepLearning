import yaml
import torch
import torch.nn as nn

with open('settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)

class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int = settings['VOCAB_SIZE'],
        embed_dim: int = settings['EMBED_DIM']
    ):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=0
        )

    def forward(self, x):
        x_emb = self.embedding_layer(x)
        return x_emb