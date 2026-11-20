"""VE-MD-PersonQuery for one classification task per training run.

Ground-truth skeletons are auxiliary training targets, never model inputs.
The emotion decoder may consume the model's predicted structural outputs.
"""

import torch
from torch import nn

from encoders import EncoderResidual, EncoderResnet, EncoderViTBase, reparameterize
from decoders import (
    SkeletonDETR, STGCN,
    LearnablePositionalEncoding, RelativePositionalEncoding, SinusoidalPositionalEncoding,
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
                 structural_dims,
                 projection_dim=None, 
                 positional_encoding="sinusoidal", 
                 dropout=0.0,
                 feedforward_dim=2048):
        super().__init__()
        self.linear_reduction = nn.Linear(latent_branches * latent_dim * 7 * 7, latent_dim)
        self.structural_projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, projection_dim), nn.ReLU())
            if projection_dim is not None else nn.Identity() for dim in structural_dims
        )
        dim = latent_dim + sum(projection_dim or size for size in structural_dims)
        encodings = {
            "sinusoidal": SinusoidalPositionalEncoding,
            "learnable": LearnablePositionalEncoding,
            "relative": RelativePositionalEncoding,
        }
        self.pos_enc = encodings[positional_encoding](500, dim) if positional_encoding != "none" else nn.Identity()
        layer = nn.TransformerEncoderLayer(dim, 1, dim_feedforward=feedforward_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.pool = AttentionPooling(dim)
        self.classifier = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(512, num_classes))

    def forward(self, latents, structures):
        reduced = self.linear_reduction(torch.cat(latents, dim=-1))
        features = [reduced] + [proj(x) for proj, x in zip(self.structural_projections, structures)]
        return self.classifier(self.pool(self.transformer(self.pos_enc(torch.cat(features, dim=-1)))))


class VE_MultiDecoder(nn.Module):
    def __init__(self, 
                 num_classes=3, 
                 encoder_name="resnet50", 
                 latent_dim=512,
                 channel_factor=64, 
                 blocks_res=5, 
                 num_queries=100, 
                 nheads=8,
                 num_encoder_layers=3, 
                 num_decoder_layers=3, 
                 feedforward_dim=2048,
                 use_person_pose=True, 
                 use_face_pose=False, 
                 two_encoders=False,
                 use_vit=True, 
                 vit_dim=1024, 
                 add_keypoints=True, 
                 projection_dim=None,
                 stgcn_active=False, 
                 num_limbs_person=18, 
                 num_limbs_face=20,
                 positional_encoding="sinusoidal", 
                 dropout=0.0, pretrained=False,
                 sample_in_eval=False, 
                 coordinate_layout="channels_first"):
        super().__init__()
        if latent_dim < 2 or latent_dim % nheads:
            raise ValueError("latent_dim must be >= 2 and divisible by nheads.")
        if two_encoders and not (use_person_pose and use_face_pose):
            raise ValueError("Two encoders require both body and face supervision.")
        if (add_keypoints or stgcn_active) and not (use_person_pose or use_face_pose):
            raise ValueError("Structural classification/ST-GCN requires body or face decoding.")
        if stgcn_active and not add_keypoints:
            raise ValueError("ST-GCN is used before classification; enable add_keypoints.")
        self.num_classes = num_classes
        self.use_person = use_person_pose
        self.use_face = use_face_pose
        self.use_vit = use_vit
        self.vit_dim = vit_dim
        self.latent_dim = latent_dim
        self.num_queries = num_queries
        self.num_limbs_person = num_limbs_person
        self.num_limbs_face = num_limbs_face
        self.add_keypoints = add_keypoints
        self.sample_in_eval = sample_in_eval
        if coordinate_layout not in ("channels_first", "limb_first"):
            raise ValueError("Unknown coordinate layout.")
        self.coordinate_layout = coordinate_layout

        def make_encoder():
            if encoder_name == "residual":
                return EncoderResidual(channel_factor, latent_dim, blocks_res, sample_in_eval)
            if encoder_name.startswith("resnet"):
                return EncoderResnet(latent_dim, encoder_name, pretrained, sample_in_eval)
            if encoder_name == "vitB":
                return EncoderViTBase(latent_dim, pretrained, sample_in_eval)
            raise ValueError(f"Unknown encoder: {encoder_name}")

        self.encoders = nn.ModuleList(make_encoder() for _ in range(2 if two_encoders else 1))
        if use_vit:
            self.vit_projection = nn.Linear(vit_dim, latent_dim * 7 * 7)
            self.vit_mu = nn.Conv2d(latent_dim, latent_dim, 1)
            self.vit_logvar = nn.Conv2d(latent_dim, latent_dim, 1)
        decoder_args = dict(num_queries=num_queries, latent_dim=latent_dim, nheads=nheads,
                            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                            feedforward_dim=feedforward_dim)
        self.pose_decoders = nn.ModuleDict()
        self.graphs = nn.ModuleDict()
        structural_dims = []
        for name, enabled, limbs in (("person", use_person_pose, num_limbs_person),
                                     ("face", use_face_pose, num_limbs_face)):
            if enabled:
                self.pose_decoders[name] = SkeletonDETR(num_limbs=limbs, **decoder_args)
                if stgcn_active:
                    self.graphs[name] = STGCN(4)
                if add_keypoints:
                    structural_dims.append(num_queries * 4 * limbs)
        self.emotion_decoder = Classification(
            latent_dim, len(self.encoders) + int(use_vit), num_classes, structural_dims,
            projection_dim, positional_encoding, dropout, feedforward_dim,
        )

    def forward(self, images, feat_vit=None):
        if images.ndim != 5 or images.size(2) != 3:
            raise ValueError("images must have shape (B,T,3,H,W).")
        batch, frames, channels, height, width = images.shape
        flattened = images.reshape(batch * frames, channels, height, width)
        latents = [encoder(flattened) for encoder in self.encoders]
        outputs, structures = {}, []
        for name, decoder in self.pose_decoders.items():
            latent = latents[-1] if name == "face" else latents[0]
            coords, adjacency = decoder(latent)
            outputs[f"skeleton_{name}_pred"] = coords
            outputs[f"adjacency_{name}_pred"] = adjacency
            if self.add_keypoints:
                if name in self.graphs:
                    limbs = self.num_limbs_person if name == "person" else self.num_limbs_face
                    if self.coordinate_layout == "limb_first":
                        coords = coords.reshape(batch, frames, self.num_queries, limbs, 4).permute(0, 1, 2, 4, 3)
                    else:
                        # Coordinates are stored as four channels over K limbs.
                        coords = coords.reshape(batch, frames, self.num_queries, 4, limbs)
                    adjacency = adjacency.reshape(batch, frames, self.num_queries, limbs, limbs)
                    coords = self.graphs[name](coords, adjacency)
                    if self.coordinate_layout == "limb_first":
                        coords = coords.permute(0, 1, 2, 4, 3)
                structures.append(coords.reshape(batch, frames, -1))
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
