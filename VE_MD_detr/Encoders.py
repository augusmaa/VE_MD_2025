"""Variational visual encoders; optional dependencies are imported on demand."""

import torch
from torch import nn
from torch.nn import functional as F



class ResDown(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)
        self.activation = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.activation(self.bn1(self.conv1(x)))
        return self.activation(self.bn2(self.conv2(x) + skip))


def reparameterize(mu, logvar, sample=True):
    """Sample during training; use the mean during deterministic evaluation."""
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu


class EncoderResidual(nn.Module):
    def __init__(self, ch, latent_channels, num_blocks, sample_in_eval=False):
        super().__init__()
        self.sample_in_eval = sample_in_eval
        self.conv_in = nn.Conv2d(3, ch, 7, 1, 3)
        self.res_blocks = nn.ModuleList(
            ResDown(ch * 2**i, ch * 2**(i + 1)) for i in range(num_blocks)
        )
        self.conv_mu = nn.Conv2d(ch * 2**num_blocks, latent_channels, 1)
        self.conv_logvar = nn.Conv2d(ch * 2**num_blocks, latent_channels, 1)

    def forward(self, x):
        x = F.elu(self.conv_in(x))
        for block in self.res_blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        return reparameterize(self.conv_mu(x), self.conv_logvar(x),
                              self.training or self.sample_in_eval)


class EncoderResnet(nn.Module):
    def __init__(self, latent_channels, resnet_name="resnet50", pretrained=False, sample_in_eval=False):
        super().__init__()
        from torchvision import models

        constructors = {
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
            "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
            "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
        }
        constructor, weights = constructors[resnet_name]
        backbone = constructor(weights=weights if pretrained else None)
        self.resnet = nn.Sequential(*list(backbone.children())[:-2])
        self.conv_mu = nn.Conv2d(2048, latent_channels, 1)
        self.conv_logvar = nn.Conv2d(2048, latent_channels, 1)
        self.sample_in_eval = sample_in_eval

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.resnet(x), (7, 7))
        return reparameterize(self.conv_mu(x), self.conv_logvar(x),
                              self.training or self.sample_in_eval)


class EncoderViTBase(nn.Module):
    """Optional ViT-Base ablation, separate from the cached 1024-D ViT branch."""

    def __init__(self, latent_dim=512, pretrained=False, sample_in_eval=False):
        super().__init__()
        try:
            from transformers import ViTConfig, ViTModel
        except ImportError as exc:
            raise ImportError("For the vitB encoder: pip install transformers==4.46.3") from exc
        self.vit_backbone = (
            ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False)
            if pretrained else ViTModel(ViTConfig(), add_pooling_layer=False)
        )
        self.vit_backbone.requires_grad_(False)
        self.latent = nn.Linear(768, latent_dim)
        self.conv_mu = nn.Conv2d(latent_dim, latent_dim, 8)
        self.conv_logvar = nn.Conv2d(latent_dim, latent_dim, 8)
        self.sample_in_eval = sample_in_eval

    def forward(self, images):
        if images.shape[-2:] != (224, 224):
            raise ValueError("The vitB encoder requires 224 x 224 images.")
        self.vit_backbone.eval()
        with torch.no_grad():
            tokens = self.vit_backbone(pixel_values=images).last_hidden_state[:, 1:]
        # Exclude CLS; transpose channels before reshaping spatial patch tokens.
        features = self.latent(tokens).transpose(1, 2).reshape(images.size(0), -1, 14, 14)
        return reparameterize(self.conv_mu(features), self.conv_logvar(features),
                              self.training or self.sample_in_eval)
