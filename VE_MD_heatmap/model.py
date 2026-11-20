"""VE-MD-Heatmap with cached ViT features and one classification head."""

import torch
from torch import nn

from encoders import EncoderResidual, EncoderResnet, reparameterize
from decoders import (
    HeatmapDecoder, LearnablePositionalEncoding,
    RelativePositionalEncoding, SinusoidalPositionalEncoding,
)


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Linear(dim, 1)

    def forward(self, x):
        return (self.attention(x).softmax(dim=1) * x).sum(dim=1)


class Classification(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 latent_branches, 
                 num_classes, 
                 structural_branches,
                 projection_dim=None, 
                 positional_encoding="sinusoidal", 
                 dropout=0.0,
                 feedforward_dim=2048):
        super().__init__()
        self.linear_reduction = nn.Linear(latent_branches * latent_dim * 7 * 7, latent_dim)
        self.structural_projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(56 * 56), nn.Linear(56 * 56, projection_dim), nn.ReLU())
            if projection_dim is not None else nn.Identity() for _ in range(structural_branches)
        )
        dim = latent_dim + structural_branches * (projection_dim or 56 * 56)
        encodings = {"sinusoidal": SinusoidalPositionalEncoding,
                     "learnable": LearnablePositionalEncoding, "relative": RelativePositionalEncoding}
        self.pos_enc = encodings[positional_encoding](500, dim) if positional_encoding != "none" else nn.Identity()
        layer = nn.TransformerEncoderLayer(dim, 1, dim_feedforward=feedforward_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.pool = AttentionPooling(dim)
        self.classifier = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(512, num_classes))

    def forward(self, latents, heatmaps):
        reduced = self.linear_reduction(torch.cat(latents, dim=-1))
        features = [reduced] + [projection(h) for projection, h in zip(self.structural_projections, heatmaps)]
        sequence = self.pos_enc(torch.cat(features, dim=-1))
        return self.classifier(self.pool(self.transformer(sequence)))


class VE_MultiDecoder(nn.Module):
    def __init__(self, 
                 num_classes=3, 
                 encoder_name="resnet50", 
                 latent_dim=512,
                 channel_factor=64, 
                 blocks_res=5, 
                 pretrained=False,
                 use_person_pose=True, 
                 use_face_pose=False, 
                 two_encoders=False,
                 use_vit=True, 
                 vit_dim=1024, 
                 add_heatmaps=True, 
                 projection_dim=None,
                 num_heatmaps_person=18, 
                 num_heatmaps_face=83, 
                 num_stages=6,
                 decoder_channels=256, 
                 positional_encoding="sinusoidal", 
                 dropout=0.0,
                 feedforward_dim=2048, 
                 sample_in_eval=False):
        super().__init__()
        if latent_dim < 2 or num_classes < 2 or vit_dim < 1:
            raise ValueError("Invalid latent dimension, class count or feature dimension.")
        if two_encoders and not (use_person_pose and use_face_pose):
            raise ValueError("Two encoders require both body and face supervision.")
        if add_heatmaps and not (use_person_pose or use_face_pose):
            raise ValueError("Enable a structural decoder when adding heatmaps to classification.")
        if projection_dim is not None and projection_dim < 1:
            raise ValueError("projection_dim must be a positive integer or null.")
        self.num_classes = num_classes
        self.use_vit = use_vit
        self.vit_dim = vit_dim
        self.latent_dim = latent_dim
        self.add_heatmaps = add_heatmaps
        self.sample_in_eval = sample_in_eval

        def make_encoder():
            if encoder_name == "residual":
                return EncoderResidual(channel_factor, latent_dim, blocks_res, sample_in_eval)
            if encoder_name in ("resnet50", "resnet101", "resnet152"):
                return EncoderResnet(latent_dim, encoder_name, pretrained, sample_in_eval)
            raise ValueError(f"Unknown visual encoder: {encoder_name}")

        self.encoders = nn.ModuleList(make_encoder() for _ in range(2 if two_encoders else 1))
        if use_vit:
            # Same cached-feature projection and variational mapping as PersonQuery.
            self.vit_projection = nn.Linear(vit_dim, latent_dim * 7 * 7)
            self.vit_mu = nn.Conv2d(latent_dim, latent_dim, 1)
            self.vit_logvar = nn.Conv2d(latent_dim, latent_dim, 1)
        self.heatmap_decoders = nn.ModuleDict()
        for name, enabled, channels in (("person", use_person_pose, num_heatmaps_person),
                                        ("face", use_face_pose, num_heatmaps_face)):
            if enabled:
                self.heatmap_decoders[name] = HeatmapDecoder(latent_dim, channels, num_stages, decoder_channels)
                
        self.emotion_decoder = Classification(
            latent_dim, len(self.encoders) + int(use_vit), num_classes,
            len(self.heatmap_decoders) if add_heatmaps else 0,
            projection_dim, positional_encoding, dropout, feedforward_dim,
        )

    def forward(self, images, feat_vit=None):
        if images.ndim != 5 or images.size(2) != 3:
            raise ValueError("images must have shape (B,T,3,H,W).")
        batch, frames, channels, height, width = images.shape
        images = images.reshape(batch * frames, channels, height, width)
        latents = [encoder(images) for encoder in self.encoders]
        outputs, structures = {}, []
        for name, decoder in self.heatmap_decoders.items():
            latent = latents[-1] if name == "face" else latents[0]
            stage_logits = decoder(latent)
            heatmaps = stage_logits[-1].sigmoid()
            outputs[f"heatmap_{name}_logits"] = stage_logits
            outputs[f"heatmap_{name}"] = heatmaps.reshape(batch, frames, -1, 56, 56)
            if self.add_heatmaps:
                # Retain the sum over structural channels before raw/projected fusion.
                structures.append(heatmaps.sum(dim=1).reshape(batch, frames, 56 * 56))
        if self.use_vit:
            if feat_vit is None or feat_vit.shape != (batch, frames, self.vit_dim):
                raise ValueError(f"feat_vit must have shape ({batch},{frames},{self.vit_dim}).")
            features = self.vit_projection(feat_vit).reshape(batch * frames, self.latent_dim, 7, 7)
            latents.append(reparameterize(self.vit_mu(features), self.vit_logvar(features),
                                          self.training or self.sample_in_eval))
        outputs["z"] = torch.cat(latents, dim=1)
        outputs["preds_emotion"] = self.emotion_decoder(
            [z.reshape(batch, frames, -1) for z in latents], structures
        )
        return outputs
