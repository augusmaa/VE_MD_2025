import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from einops.layers.torch import Rearrange, Reduce
from collections import OrderedDict
#from torchvision.models import resnet50, resnet101,resnet152
from torchvision.models.resnet import resnet50, resnet101,resnet152
from transformers import ViTModel


## Dependencies
"""To import dependance functions for Upsample and ResDown class"""
from sampleUpDown import *
from pose import *

     
class EncoderResidual(nn.Module):
    def __init__(self,
                 ch,
                 latent_channels,
                 num_blocks):
        super(EncoderResidual, self).__init__()
        self.conv_in = nn.Conv2d(3, ch, 7, 1, 3) # when there is No Mask
        self.res_blocks = nn.ModuleList([ResDown(ch * 2**i, ch * 2**(i+1)) for i in range(num_blocks)])
        # Additional 1x1 convs to adjust channels to latent_channels
        self.latent = nn.Conv2d(ch*2**num_blocks, latent_channels, 1)
        self.act_fnc = nn.ELU()
    def forward(self, x):
        x = self.act_fnc(self.conv_in(x))
        for block in self.res_blocks:
            x = block(x)
        x = self.latent(x)
        return x

class EncoderResnet(nn.Module):
    def __init__(self,
                 latent_channels,
                 resnet_name='resnet50'):
        super(EncoderResnet, self).__init__()
        self.latent_channels = latent_channels
        self.resnet_name = resnet_name

        # Use ResNet50 or ResNet152d
        if self.resnet_name=='resnet50':
            ##self.resnet = resnet50(pretrained=True) #resnet50(pretrained=True)
            backbone = resnet50(pretrained=True)
            self.resnet = nn.Sequential(*list(backbone.children())[:-2]) 
        elif self.resnet_name=='resnet101':
            ##self.resnet = resnet101(pretrained=True)
            backbone = resnet101(pretrained=True)
            self.resnet = nn.Sequential(*list(backbone.children())[:-2]) 
        elif self.resnet_name=='resnet152':
            ##self.resnet = resnet152(pretrained=True)
            backbone = resnet152(pretrained=True)
            self.resnet = nn.Sequential(*list(backbone.children())[:-2]) 
        else:
            raise ValueError('This resnet not defined')
        # Remove the global pooling and fully connected layers
        #self.resnet.fc = torch.nn.Identity()
        #self.resnet.avgpool = torch.nn.Identity()

        self.latent = nn.Conv2d(2048, latent_channels, 1)

    def forward(self, x):
        bs =x.size(0)
        x = self.resnet(x).view(bs, 2048, 7, 7)
        x = self.latent(x)
        return x
    
    
    
class EncoderViTLarge(nn.Module):
    def __init__(self,
                latent_dim=512,
                ):
        super(EncoderViTLarge, self).__init__()   
        self.latent_dim = latent_dim 
        # ViT Backbone: Use a pretrained ViT-Large model
        ##self.vit_backbone = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
        self.vit_backbone = timm.create_model('vit_large_patch14_224_clip_laion2b', pretrained=True)
        # Freeze ViT Backbone parameters if desired
        for param in self.vit_backbone.parameters():
            param.requires_grad = False
         
        # for i in range(23,25):
        #     for param in self.vit_backbone.blocks[i].parameters(): # unfreeze transformer encoder
        #         param.requires_grad = True
            
        self.lantent = nn.Linear(1024, latent_dim)
        self.latent_resized = nn.Conv2d(latent_dim, latent_dim, 10)
    def forward(self, images):
        bs = images.size(0)
        vit_outputs = self.vit_backbone.forward_features(images)
        encoded_features = self.lantent(vit_outputs)
        encoded_features = encoded_features[:,: 256,:].view(bs, self.latent_dim, 16, 16)
        encoded_features = self.latent_resized(encoded_features)
        return encoded_features
    
    

class EncoderViTBase(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 ):
        super(EncoderViTBase, self).__init__()   
        self.latent_dim = latent_dim 
        
        # ViT Backbone: Use a pretrained ViT-Base model
        self.vit_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # # Freeze ViT Backbone parameters if desired
        # for param in self.vit_backbone.parameters():
        #     param.requires_grad = False
        
        # for param in self.vit_backbone.encoder.parameters():
        #     param.requires_grad = True
        
        
        # Latent representation transformation
        self.lantent = nn.Linear(768, latent_dim)  # ViT-Base output size is 768
        self.latent_resized = nn.Conv2d(latent_dim, latent_dim, 8)
    
    def forward(self, images):
        bs = images.size(0)
        
        # Get the outputs from ViT backbone
        vit_outputs = self.vit_backbone(pixel_values=images).last_hidden_state  # Shape: [batch_size, 197, 768]
        
        # Linear transformation to latent dimension
        encoded_features = self.lantent(vit_outputs)  # Shape: [batch_size, 197, latent_dim]
        
        # Reshape and resize latent features
        encoded_features = encoded_features[:, :196, :].view(bs, self.latent_dim, 14, 14)  # Shape: [batch_size, latent_dim, 14, 14]
        encoded_features = self.latent_resized(encoded_features)  # Shape: [batch_size, latent_dim, 7, 7]
        
        return encoded_features
