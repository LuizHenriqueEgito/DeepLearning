import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        hidden_size: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, d_model)
        )

    def forward(self, x):
        return self.feed_forward(x)