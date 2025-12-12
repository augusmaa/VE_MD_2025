import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy  as  np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
from random import choice
import collections
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


## Dependencies
from dataset import *
from pad_collate import *

# transforms
general_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


def Train_DataLoaders(rank,
                    world_size,
                    Dir_vgaf_train,
                    Dir_vgaf_train_vit,
                    vgaf_train_csvfile,
                    Dir_gaf3_train,
                    Dir_gaf3_train_heatmap,
                    Dir_gaf3_train_vit,
                    gaf3_train_csvfile,
                    Dir_dfew_train,
                    Dir_dfew_train_vit,
                    dfew_train_csvfile,
                    Dir_mer_train,
                    Dir_mer_train_vit,
                    mer_train_csvfile,
                    Dir_engage_train,
                    Dir_engage_train_vit,
                    engage_train_csvfile,
                    Dir_samsemo_train,
                    Dir_samsemo_train_vit,
                    samsemo_train_csvfile,
                    Dir_mersemi_train,
                    mersemi_train_csvfile,
                    batch_vgaf,
                    batch_gaf3,
                    batch_dfew,
                    batch_mer,
                    batch_engage,
                    batch_samsemo,
                    batch_mersemi,   
                    ):
    # DataLoaders     

    dataset_vgaf = VGAFVideoDataset(Dir_vgaf_train,Dir_vgaf_train_vit, vgaf_train_csvfile, general_transforms['train']) 
    sampler_vgaf = DistributedSampler(dataset_vgaf, num_replicas=world_size, rank=rank, shuffle=True)
    loader_vgaf = torch.utils.data.DataLoader(dataset_vgaf,
                                                batch_size=batch_vgaf,
                                                sampler=sampler_vgaf,
                                                )
    if dist.get_rank() == 0:   
        print(f'Number of vgaf data Train:{len(dataset_vgaf)}')
        
    dataset_gaf3 = GAFDataset(Dir_gaf3_train,Dir_gaf3_train_heatmap, Dir_gaf3_train_vit, gaf3_train_csvfile, general_transforms['train']) # for .h5 file
    sampler_gaf3 = DistributedSampler(dataset_gaf3, num_replicas=world_size, rank=rank, shuffle=True)
    loader_gaf3 = torch.utils.data.DataLoader(dataset_gaf3,
                                                batch_size=batch_gaf3,
                                                sampler=sampler_gaf3,
                                                )
    if dist.get_rank() == 0:   
        print(f'Number of gaf3 data Train:{len(dataset_gaf3)}')
        
    dataset_samsemo = SamSemoVideoDataset(Dir_samsemo_train, Dir_samsemo_train_vit, samsemo_train_csvfile, general_transforms['train'])
    sampler_samsemo = DistributedSampler(dataset_samsemo, num_replicas=world_size, rank=rank, shuffle=True)
    loader_samsemo = torch.utils.data.DataLoader(dataset_samsemo,
                                                batch_size=batch_samsemo,
                                                sampler=sampler_samsemo,
                                                )

    if dist.get_rank() == 0:   
        print(f'Number of samsemo data Train:{len(dataset_samsemo)}')

    # DFEW 
    dataset_dfew = DFEWVideoDataset(Dir_dfew_train, Dir_dfew_train_vit, dfew_train_csvfile, general_transforms['train'])
    sampler_dfew = DistributedSampler(dataset_dfew, num_replicas=world_size, rank=rank, shuffle=True)
    loader_dfew = torch.utils.data.DataLoader(dataset_dfew,
                                                batch_size=batch_dfew,
                                                sampler=sampler_dfew,
                                                )
    if dist.get_rank() == 0:   
        print(f'Number of DFEW data Train:{len(dataset_dfew)}')
        

    # MER2023 
    dataset_mer = MER2023VideoDataset(Dir_mer_train, Dir_mer_train_vit, mer_train_csvfile, general_transforms['train'])
    sampler_mer = DistributedSampler(dataset_mer, num_replicas=world_size, rank=rank, shuffle=True)
    loader_mer = torch.utils.data.DataLoader(dataset_mer,
                                                batch_size=batch_mer,
                                                sampler=sampler_mer,
                                                )
    if dist.get_rank() == 0:   
        print(f'Number of MER data Train:{len(dataset_mer)}')

    dataset_engage = EngageNetVideoDataset(Dir_engage_train,Dir_engage_train_vit, engage_train_csvfile, general_transforms['train'])
    sampler_engage = DistributedSampler(dataset_engage, num_replicas=world_size, rank=rank, shuffle=True)
    loader_engage = torch.utils.data.DataLoader(dataset_engage,
                                                batch_size=batch_engage,
                                                sampler=sampler_engage,
                                                )
    if dist.get_rank() == 0: 
        print(f'Number of engage data Train: {len(dataset_engage)}')

    dataset_mersemi = MER2023Semi(Dir_mersemi_train, mersemi_train_csvfile, general_transforms['train'])
    sampler_mersemi = DistributedSampler(dataset_mersemi, num_replicas=world_size, rank=rank, shuffle=True)
    loader_mersemi = torch.utils.data.DataLoader(dataset_mersemi,
                                                batch_size=batch_mersemi,
                                                sampler=sampler_mersemi,
                                                )
    if dist.get_rank() == 0: 
        print(f'Number of mersemi data Train:{len(dataset_mersemi)}')


    loader_data= {'vgaf': loader_vgaf,
                'sampler_vgaf': sampler_vgaf,
                'gaf3': loader_gaf3,
                'sampler_gaf3': sampler_gaf3,
                'samsemo': loader_samsemo,
                'sampler_samsemo': sampler_samsemo,
                'dfew': loader_dfew,
                'sampler_dfew': sampler_dfew,
                'mer': loader_mer,
                'sampler_mer': sampler_mer,
                'engagenet':loader_engage,
                'sampler_engagenet': sampler_engage,
                'mersemi':loader_mersemi,
                'sampler_mersemi': sampler_mersemi,
                }
    
    return loader_data

def Val_DataLoaders(rank,
                    world_size,
                    Dir_vgaf_val,
                    Dir_vgaf_val_vit,
                    vgaf_val_csvfile,
                    Dir_gaf3_val,
                    Dir_gaf3_val_vit,
                    gaf3_val_csvfile,
                    Dir_dfew_val,
                    Dir_dfew_val_vit,
                    dfew_val_csvfile,
                    Dir_mer_val,
                    Dir_mer_val_vit,
                    mer_val_csvfile,
                    Dir_engage_val,
                    Dir_engage_val_vit,
                    engage_val_csvfile,
                    Dir_samsemo_val,
                    Dir_samsemo_val_vit,
                    samsemo_val_csvfile,
                    batch_vgaf,
                    batch_gaf3,
                    batch_dfew,
                    batch_mer,
                    batch_engage,
                    batch_samsemo,
                    ):
    # DataLoaders 
    dataset_vgaf = MyDatasetVGAF(Dir_vgaf_val, Dir_vgaf_val_vit, vgaf_val_csvfile, general_transforms['val'])
    sampler_vgaf = DistributedSampler(dataset_vgaf, num_replicas=world_size, rank=rank, shuffle=False)
    loader_vgaf = torch.utils.data.DataLoader(dataset_vgaf,
                                                batch_size=batch_vgaf,
                                                sampler=sampler_vgaf
                                                )
    if dist.get_rank() == 0:  #
        print(f'Number of vgaf data Val:{len(dataset_vgaf)}')
        
     # DataLoaders 
    dataset_gaf3 = GAFDatasetTest(Dir_gaf3_val,Dir_gaf3_val_vit, gaf3_val_csvfile, general_transforms['val'])
    sampler_gaf3 = DistributedSampler(dataset_gaf3, num_replicas=world_size, rank=rank, shuffle=False)
    loader_gaf3 = torch.utils.data.DataLoader(dataset_gaf3,
                                                batch_size=batch_gaf3,
                                                sampler=sampler_gaf3
                                                )
    if dist.get_rank() == 0:  #
        print(f'Number of gaf3 data Val:{len(dataset_gaf3)}')
    
    dataset_samsemo = SamSemoDataset(Dir_samsemo_val,Dir_samsemo_val_vit, samsemo_val_csvfile, general_transforms['val'])
    sampler_samsemo = DistributedSampler(dataset_samsemo, num_replicas=world_size, rank=rank, shuffle=False)
    loader_samsemo = torch.utils.data.DataLoader(dataset_samsemo,
                                                batch_size=batch_samsemo,
                                                sampler=sampler_samsemo
                                                )
    if dist.get_rank() == 0:  #
        print(f'Number of samsemo data Val:{len(dataset_samsemo)}')

     # DFEW 
    dataset_dfew = DFEWDataset(Dir_dfew_val, Dir_dfew_val_vit, dfew_val_csvfile, general_transforms['val'])
    sampler_dfew = DistributedSampler(dataset_dfew, num_replicas=world_size, rank=rank, shuffle=False)
    loader_dfew = torch.utils.data.DataLoader(dataset_dfew,
                                                batch_size=batch_dfew,
                                                sampler=sampler_dfew,
                                                )
    if dist.get_rank() == 0:   
        print(f'Number of DFEW data Val:{len(dataset_dfew)}')

    # MER2023 
    dataset_mer = MER23Dataset(Dir_mer_val, Dir_mer_val_vit, mer_val_csvfile, general_transforms['val'])
    sampler_mer = DistributedSampler(dataset_mer, num_replicas=world_size, rank=rank, shuffle=False)
    loader_mer = torch.utils.data.DataLoader(dataset_mer,
                                                batch_size=batch_mer,
                                                sampler=sampler_mer
                                                )
    if dist.get_rank() == 0:   
        print(f'Number of MER2023 data Val:{len(dataset_mer)}')
   

    dataset_engage = EngageDataset(Dir_engage_val, Dir_engage_val_vit, engage_val_csvfile, general_transforms['val'])
    sampler_engage = DistributedSampler(dataset_engage, num_replicas=world_size, rank=rank, shuffle=False)
    loader_engage = torch.utils.data.DataLoader(dataset_engage,
                                                batch_size=batch_engage,
                                                sampler=sampler_engage
                                                )
    if dist.get_rank() == 0:  #                                            
        print(f'Number of engage data Val:{len(dataset_engage)}')

    loader_data= {'vgaf': loader_vgaf,
                  'gaf3': loader_gaf3,
                'samsemo': loader_samsemo,
                'dfew': loader_dfew,
                'mer': loader_mer,
                'engagenet':loader_engage,
            }
    return loader_data


