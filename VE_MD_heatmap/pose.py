import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from sampleUpDown import *


class UNetResDecoderFeature(nn.Module):
    def __init__(self, latent_dim=512, out_feat_channels=512):
        super().__init__()
        self.up5 = nn.Conv2d(latent_dim, 2048, 1)
        self.up4 = ResUp(2048, 512)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(128, 128, 1, stride=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(128, out_feat_channels, kernel_size=1)
        
    def forward(self, fmp):
        d5 = self.up5(fmp)
        d4 = self.up4(d5)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = self.dec1(d1)
        return self.final_conv(d1)  # (B, C, 56, 56)


class DecoderOpenPose(nn.Module):
    def __init__(self, latent_dim, out_channels, num_stages):
        super().__init__()
        self.num_stages = num_stages
        self.out_channels = out_channels
        
        # 1. Use UNet-style decoder for initial feature extraction
        self.unet_decoder = UNetResDecoderFeature(latent_dim, out_feat_channels=256)

        # 2. Create limb prediction branches
        self.limb_branch = nn.ModuleList()
        
        # Initial input channels
        self.base_channels = 256
        
        for i in range(num_stages):
            # Main prediction branch
            self.limb_branch.append(nn.Sequential(
                nn.Conv2d(self.base_channels, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, out_channels, 1)
            ))

    def forward(self, features):
        x = self.unet_decoder(features)  # (B, 256, 56, 56)
        heatmaps = []        
        for stage_idx in range(self.num_stages):
            # Predict heatmap
            heatmap = self.limb_branch[stage_idx](x)
            heatmaps.append(heatmap)
        return heatmaps if self.training else torch.sigmoid(heatmaps[-1])