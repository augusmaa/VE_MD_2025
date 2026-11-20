"""Temporal encodings, PersonQuery decoding and optional ST-GCN."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from encoders import ResDown


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        if x.size(1) > self.pos_embed.size(0):
            raise ValueError("Sequence exceeds the configured positional encoding length.")
        return x + self.pos_embed[:x.size(1)]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale[:dim // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence exceeds the configured positional encoding length.")
        return x + self.pe[:, :x.size(1)]


class RelativePositionalEncoding(nn.Module):
    """Relative embedding transform, distinct from an attention position bias."""

    def __init__(self, max_len, dim):
        super().__init__()
        self.rel_pos = nn.Embedding(2 * max_len - 1, dim)
        self.max_len = max_len

    def forward(self, x):
        length = x.size(1)
        if length > self.max_len:
            raise ValueError("Sequence exceeds the configured positional encoding length.")
        positions = torch.arange(length, device=x.device)
        indices = positions[:, None] - positions[None, :] + self.max_len - 1
        return torch.einsum("btd,tsd->bsd", x, self.rel_pos(indices))


class AuxiliaryConvolutions(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.featmap2_1 = nn.Conv2d(latent_dim, latent_dim, 3)
        self.featmap2_2 = ResDown(latent_dim, latent_dim)

    def forward(self, x):
        middle = F.elu(self.featmap2_1(x))
        return x, middle, self.featmap2_2(middle)


class SkeletonDETR(nn.Module):
    def __init__(self, 
                 num_queries, 
                 num_limbs, 
                 latent_dim=512, 
                 nheads=8,
                 num_encoder_layers=3, 
                 num_decoder_layers=3, 
                 feedforward_dim=2048):
        super().__init__()
        self.multiscale = AuxiliaryConvolutions(latent_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(5000, latent_dim)
        self.query_embed = nn.Embedding(num_queries, latent_dim)
        self.transformer = nn.Transformer(
                                        d_model=latent_dim, 
                                        nhead=nheads, 
                                        batch_first=True,
                                        num_encoder_layers=num_encoder_layers, 
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=feedforward_dim,
                                    )
        self.limb_head = nn.Sequential(nn.Linear(latent_dim, 256), 
                                       nn.ReLU(), nn.Linear(256, num_limbs * 4), 
                                       nn.Sigmoid()
                                       )
        self.adjacency_head = nn.Sequential(nn.Linear(latent_dim, num_limbs**2), nn.Sigmoid())

    def forward(self, features):
        source = torch.cat([x.flatten(2) for x in self.multiscale(features)], dim=2)
        source = self.pos_encoder(source.transpose(1, 2))
        queries = self.query_embed.weight.unsqueeze(0).expand(features.size(0), -1, -1)
        decoded = self.transformer(source, queries)
        return self.limb_head(decoded), self.adjacency_head(decoded)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / math.sqrt(out_channels), 1 / math.sqrt(out_channels))

    def forward(self, x, adjacency):
        # x: B,T,Q,C,K; adjacency: B,T,Q,K,K.
        x = torch.einsum("btqvw,btqcw->btqcv", adjacency, x)
        return torch.einsum("btqcv,co->btqov", x, self.weight) + self.bias[None, None, None, :, None]


class STGCN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.graph = GraphConv(in_channels, in_channels)
        # Flatten query/limb axes for a Conv2d with the same temporal operation
        # as kernel=(3,1,1) Conv3d; this also avoids a Conv3d requirement on MPS.
        self.temporal = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x, adjacency):
        batch, frames, queries, channels, limbs = x.shape
        x = self.graph(x, adjacency).permute(0, 3, 1, 2, 4).contiguous()
        x = x.reshape(batch, channels, frames, queries * limbs)
        x = torch.relu(self.bn(self.temporal(x)))
        return x.reshape(batch, channels, frames, queries, limbs).permute(0, 2, 3, 1, 4)
