import torch
import torch.nn as nn
import math

# 1. Learnable Positional Embedding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, T, D] 
        T = x.size(1)
        # slice to match your actual T
        return x + self.pos_embed[:T]

# 2. Sinusoidal Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, T, D]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 3. Relative Positional Encoding
class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.rel_pos = nn.Embedding(2 * max_len - 1, dim)
        self.max_len = max_len

    def forward(self, q):
        # q: [B, T, D]
        B, T, D = q.shape
        pos_indices = torch.arange(T, device=q.device).unsqueeze(1) - torch.arange(T, device=q.device).unsqueeze(0)
        pos_indices += self.max_len - 1  # shift to be >= 0
        rel_embedding = self.rel_pos(pos_indices)  # [T, T, D]
        q = q.transpose(0, 1)  # [T, B, D]
        output = torch.einsum('tbd,tsd->bsd', q, rel_embedding)  # sum over t
        #output = output.transpose(0, 1)  # [B, T, D]
        return output
