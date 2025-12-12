
import numpy  as  np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim as optim
import torchvision
import torchvision.models as models



## ResUp and Down
class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2) 
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4) 
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2) 
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4) 
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2) 
        self.act_fnc = nn.ELU() 

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResUp, self).__init__()
        
        self.dconv1 = nn.ConvTranspose2d(channel_in, channel_in//2, kernel_size, 1, kernel_size // 2) 
        self.bn1 = nn.BatchNorm2d(channel_in//2 , eps=1e-4) 
        self.dconv2 = nn.ConvTranspose2d(channel_in//2, channel_out, kernel_size, 2, kernel_size // 2, output_padding=1) 
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4) 
        self.dconv3 = nn.ConvTranspose2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2, output_padding=1) 
        self.act_fnc = nn.ELU() 

    def forward(self, x):
        skip = self.dconv3(x)
        x = self.act_fnc(self.bn1(self.dconv1(x)))
        x = self.dconv2(x)

        return self.act_fnc(self.bn2(x + skip))
