"""Dense heatmap decoders and temporal positional encodings."""

import math

import torch
from torch import nn


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


class ResUp(nn.Module):
    """Residual upsampling block: double the spatial resolution."""

    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.dconv1 = nn.ConvTranspose2d(channel_in, channel_in // 2, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.dconv2 = nn.ConvTranspose2d(channel_in // 2, channel_out, 3, 2, 1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.dconv3 = nn.ConvTranspose2d(channel_in, channel_out, 3, 2, 1, output_padding=1)
        self.activation = nn.ELU()

    def forward(self, x):
        skip = self.dconv3(x)
        x = self.activation(self.bn1(self.dconv1(x)))
        return self.activation(self.bn2(self.dconv2(x) + skip))


def convolution_block(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
    )


class UNetResDecoderFeature(nn.Module):
    """Decode a 7 x 7 latent map into 56 x 56 features."""

    def __init__(self, latent_dim=512, channels=256):
        super().__init__()
        self.up5 = nn.Conv2d(latent_dim, channels * 8, 1)
        self.up4 = ResUp(channels * 8, channels * 2)
        self.dec4 = convolution_block(channels * 2)
        self.up3 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.dec3 = convolution_block(channels)
        self.up2 = nn.ConvTranspose2d(channels, channels // 2, 2, stride=2)
        self.dec2 = convolution_block(channels // 2)
        self.up1 = nn.ConvTranspose2d(channels // 2, channels // 2, 1)
        self.dec1 = convolution_block(channels // 2)
        self.final_conv = nn.Conv2d(channels // 2, channels, 1)

    def forward(self, latent):
        x = self.dec4(self.up4(self.up5(latent)))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        return self.final_conv(self.dec1(self.up1(x)))


class HeatmapDecoder(nn.Module):
    """Stage heads share one decoded feature map; each returns heatmap logits."""

    def __init__(self, latent_dim, out_channels, num_stages=6, channels=256):
        super().__init__()
        if channels < 2 or channels % 2 or num_stages < 1:
            raise ValueError("Decoder channels must be positive and even; at least one stage is required.")
        self.unet_decoder = UNetResDecoderFeature(latent_dim, channels)
        self.limb_branch = nn.ModuleList()
        for _ in range(num_stages):
            self.limb_branch.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, out_channels, 1),
            ))

    def forward(self, latent):
        features = self.unet_decoder(latent)
        return [stage(features) for stage in self.limb_branch]


