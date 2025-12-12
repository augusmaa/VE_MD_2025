import torch
import math
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from sampleUpDown import *


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:length, :].unsqueeze(1)  # (S, 1, d_model)


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self,
                 latent_dim,
                ):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the Encoder#        
        # from fmp2
        self.featmap2_1 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=0) 
        # from featmap2_1
        self.featmap2_2 = ResDown(latent_dim, latent_dim)
                                    
    def forward(self, latent_space):# fmp2
        """
        Forward propagation.
        :param Latent-space as input:(the last feature map of the encoder)
        :return: higher-level feature maps fmp2_1, fmp2_2, fmp2_3,
        """
        fmp2_1 = F.elu(self.featmap2_1(latent_space)) 
        # from fmp2_1  
        fmp2_2 = self.featmap2_2(fmp2_1)        
        # Higher-level feature maps
        return latent_space,  fmp2_1, fmp2_2
 

class SkeletonDETR(nn.Module):
    def __init__(self, 
                 num_queries: int, 
                 num_limbs: int,
                 latent_dim: int = 512,
                 nheads: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 pe_fix: bool = True):
        super().__init__()
        self.num_queries = num_queries
        self.num_limbs = num_limbs
        self.latent_dim = latent_dim
        self.pe_fix = pe_fix
        
        # Transformer (encoder-decoder)
        self.transformer = nn.Transformer(
                            d_model=latent_dim,
                            nhead=nheads,
                            num_encoder_layers=num_encoder_layers,   
                            num_decoder_layers=num_decoder_layers
                        )

        # Learned queries for decoder
        self.query_embed = nn.Embedding(num_queries, latent_dim)

        # Heads for limb and adjacency predictions
        self.limb_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_limbs*4),
            nn.Sigmoid(),
        )

        self.adjacency_head = nn.Sequential(
            nn.Linear(latent_dim, num_limbs*num_limbs),
            nn.Sigmoid(),
        )

        # Auxiliary convolutions for multiscale features
        self.multiscalce_feat = AuxiliaryConvolutions(latent_dim=latent_dim)
        if pe_fix:
            # Sinusoidal positional encoding explicitly
            self.pos_encoder = SinusoidalPositionEncoding(d_model=latent_dim)
        else:
             # Learned positional embeddings
            self.max_len = 10000
            self.pos_embed = nn.Parameter(torch.zeros(self.max_len, latent_dim))
                

    def forward(self, features: torch.Tensor):
        B = features.size(0)

        # Extract multiscale features
        F1, F2, F3 = self.multiscalce_feat(features)  
        F1, F2, F3 = F1.flatten(2), F2.flatten(2), F3.flatten(2)
        src_seq = torch.cat((F1, F2, F3), dim=2)  # (B, C, H'*W')
        src_seq = src_seq.permute(2, 0, 1)  # (S, B, C), S=H'*W'

        S = src_seq.size(0)
        if self.pe_fix:
            # Add sinusoidal positional encoding to encoder input
            pos_embed = self.pos_encoder(S).to(src_seq.device)  # (S, 1, latent_dim)
            src_seq = src_seq + pos_embed  # (S, B, latent_dim)
        else:
            pe = self.pos_embed[:S]                   # (S, latent_dim)
            pe = pe.unsqueeze(1)                      # (S, 1, latent_dim)
            src_seq = src_seq + pe  

        # Learned queries for decoder (one per skeleton)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  
        tgt = query_pos  # decoder input explicitly

        # Transformer forward pass
        out = self.transformer(src=src_seq, tgt=tgt)
        out = out.permute(1, 0, 2)  # (B, num_queries, latent_dim)

        # Heads predictions
        limb_preds = self.limb_head(out)  
        adjacency_preds = self.adjacency_head(out)  

        return limb_preds, adjacency_preds





    
