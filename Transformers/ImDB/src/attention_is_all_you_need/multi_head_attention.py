import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module): 
    
    def __init__(self, d_model: int =2, n_heads: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_z = nn.Linear(d_model, d_model, bias=False)

    def pre_attention_reshape(self, x):
        # [B, L, E] -> [B, H, L, HD]
        B, L, E = x.shape  # B: batch size, L: SEQ_LEN, E: D_MODEL
        x = x.contiguous().view(B, L, self.n_heads, self.d_heads)
        x = x.transpose(1, 2)
        return x

    def post_attention_reshape(self, x):
        # [B, H, L, HD] -> [B, L, E]
        B, H, L, HD = x.shape  # B: batch size, H: N_HEADS, L: SEQ_LEN, HD: D_HEADS
        x = x.transpose(2, 1)
        x = x.contiguous().view((B, L, self.d_model))
        return x
        
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.pre_attention_reshape(self.W_q(encodings_for_q))
        k = self.pre_attention_reshape(self.W_k(encodings_for_k))
        v = self.pre_attention_reshape(self.W_v(encodings_for_v))
        sims = torch.matmul(q, k.transpose(-1, -2))

        scaled_sims = sims / torch.tensor(k.size(-1)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
        attention_percents = F.softmax(scaled_sims, dim=-2)
        attention_scores = torch.matmul(attention_percents, v)
        
        attention_scores = self.post_attention_reshape(attention_scores)
        z = self.W_z(attention_scores)
        
        return z