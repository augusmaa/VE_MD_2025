#!/usr/bin/env python

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
from torch.optim import lr_scheduler
from PIL import Image
import glob
from einops.layers.torch import Rearrange, Reduce
from random import choice
import collections
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


## Dependencies
from dataset import *
from loss_functions import *

class DataLoaderWrapper:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iterator = iter(data_loader)
        self.exhausted = False

    def next_batch(self):
        if not self.exhausted:
            try:
                return next(self.iterator)
            except StopIteration:
                self.exhausted = True
                return None
        else:
            return None

def random_loader_iterator(loaders_pairs):
    loaders_pairs = [(DataLoaderWrapper(loader), name) for loader, name in loaders_pairs]
    non_exhausted_loaders = [(loader,name) for loader, name in loaders_pairs if not loader.exhausted]
    if dist.get_rank() == 0:  #
        for _, name in non_exhausted_loaders:
            print(f'\n Non exhausted data RUNNING: {name}')

    
    while non_exhausted_loaders:
        selected_loader,name = random.choice(non_exhausted_loaders)
        batch, batch_name = selected_loader.next_batch(), name
        if batch is not None:
            yield (batch, batch_name) # Return a list for the batch(data, bboc, label ....), and the batch name
        non_exhausted_loaders = [(loader,name) for loader, name in loaders_pairs if not loader.exhausted]
        
       
def get_loader_data_pairs(Loader, selected_datasets):
    available_datasets = {
        'vgaf': (Loader['vgaf'], 'vgaf'),
        'gaf3': (Loader['gaf3'], 'gaf3'),
        'dfew': (Loader['dfew'], 'dfew'),
        'mer': (Loader['mer'], 'mer'),
        'engagenet': (Loader['engagenet'], 'engagenet'),
        'samsemo': (Loader['samsemo'], 'samsemo'),
       }
    # Filter based on selected datasets
    #loader_data_pairs = [available_datasets[dataset] for dataset in selected_datasets if dataset in available_datasets else raise ValueError('dataset not available')]
    loader_data_pairs = []
    for dataset in selected_datasets:
        if dataset in available_datasets:
            loader_data_pairs.append(available_datasets[dataset])
        # else:
        #     raise ValueError(f"Dataset '{dataset}' not available. Choose from {list(available_datasets.keys())}")
    return loader_data_pairs


# Common config val
DATA_CFG = {
    'vgaf': {
        'opt_key':    'vgaf',
        'vit_key':    'vgaf',
        'has_emotion':True,
        'has_pose':   True,
        'has_mmd':    True,
    },
    'gaf3': {
        'opt_key':    'gaf3',
        'vit_key':    'gaf3',
        'has_emotion':True,
        'has_pose':   True,
        'has_mmd':    True,
    },
    'mer': {
        'opt_key':    'mer',
        'vit_key':    'mer',
        'has_emotion':True,
        'has_pose':   True,
        'has_mmd':    True,
    },
    'dfew': {
        'opt_key':    'dfew',
        'vit_key':    'dfew',
        'has_emotion':True,
        'has_pose':   True,
        'has_mmd':    True,
    },
    'samsemo': {
        'opt_key':    'samsemo',
        'vit_key':    'samsemo',
        'has_emotion':True,
        'has_pose':   True,
        'has_mmd':    True,
    },
    'engagenet': {
        'opt_key':    'engagenet',
        'vit_key':    'engagenet',
        'has_emotion':True,
        'has_pose':   True,
        'has_mmd':    True,
    },
    'coco': {
        'opt_key':    'coco',
        'vit_key':    None,
        'has_emotion':False,
        'has_person': True,
        'has_face':   True,
        'has_mmd':    True,
    },
}


def Val_epoch(model,
              selected_datasets,
              LoaderVal,
              CLS_Loss,
              device):
    model.to(device).eval()

    # Only datasets with emotion
    metrics = {
        ds: {'count':0, 
             'cls_loss':0.,
             'acc':0
             }
        for ds in selected_datasets if DATA_CFG[ds]['has_emotion'] }

    with torch.no_grad():
        for batch_data, ds in random_loader_iterator(get_loader_data_pairs(LoaderVal, selected_datasets)):

            cfg = DATA_CFG[ds]
            # skip if no emotion head
            if not cfg['has_emotion']:
                continue

            imgs   = batch_data['images'].to(device)
            labels = batch_data['labels'].to(device)

            # build forward kwargs
            fwd = {'dataset': ds}
            if cfg['vit_key'] and 'feat_vit' in batch_data:
                fwd[cfg['vit_key']] = batch_data['feat_vit'].to(device)

            outputs = model(imgs, **fwd)
            logits  = outputs['preds_emotion']

            loss    = CLS_Loss(logits, labels)
            
            probs = F.softmax(logits, dim=1)
            preds   = probs.argmax(dim=1)
            correct = (preds==labels).sum().item()

            bs = imgs.size(0)
            metrics[ds]['count']    += bs
            metrics[ds]['cls_loss'] += loss.item() * bs
            metrics[ds]['acc']      += correct

    # all‐reduce & finalize
    for ds, m in metrics.items():
        cnt = torch.tensor(m['count'], device=device)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
        m['count'] = cnt.item() or 1

        tot_loss = torch.tensor(m['cls_loss'], device=device)
        tot_acc  = torch.tensor(m['acc'],      device=device)
        dist.all_reduce(tot_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(tot_acc,  op=dist.ReduceOp.SUM)

        m['cls_loss'] = tot_loss.item() / m['count']
        m['acc']      = tot_acc.item()  / m['count']

    return {
        ds: {
            'count':    m['count'],
            'cls_loss': m['cls_loss'],
            'acc':      m['acc'],
        }
        for ds, m in metrics.items()
    }














