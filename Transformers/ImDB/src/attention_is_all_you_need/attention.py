import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module): 
    
    def __init__(self, d_model=2):
        super().__init__()
        
        self.d_model=d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)
        sims = torch.matmul(q, k.transpose(-1, -2))

        scaled_sims = sims / torch.tensor(k.size(-1)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
        attention_percents = F.softmax(scaled_sims, dim=-2)
        attention_scores = torch.matmul(attention_percents, v)
        
        return attention_scores