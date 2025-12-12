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
from PIL import Image
import glob
from random import choice
import collections
import torch.distributed as dist


## Dependencies
from dataset import *
from loss_functions import *


class DataLoaderWrapper:
    def __init__(self, data_loader):
        """
        The purpose of this dataloader is to be able to train two or many dataset in the same time (at will)
        by switching batches in the same epoch. At then of the epoch all data from all datasets will have already used.
        That allow the latent space (encoder) is get improved from all dataset in one epoch.
        """
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
        'coco': (Loader['coco'], 'coco'),
       }
    # Filter based on selected datasets
    loader_data_pairs = []
    
    for dataset in selected_datasets:
        if dataset in available_datasets:
            loader_data_pairs.append(available_datasets[dataset])
        else:
            raise ValueError(f"Dataset '{dataset}' not available. Choose from {list(available_datasets.keys())}")
    return loader_data_pairs


mmd_loss_fn = MMDLoss() # from loss_functions.py

def Train_epoch(model,
                selected_datasets,
                LoaderTrain,
                epoch,
                optimizer,
                CLS_Loss,
                beta_weight_mmd,
                beta_weight_pose,
                use_mmd,
                device):
    model.to(device).train()
    
    
    DATA_CFG = {
    'vgaf': {
        'opt_key':    'vgaf',
        'vit_key':    'vgaf',
        'has_emotion':True,
        'has_person': True,
        'has_face':   True,
        'has_mmd':    use_mmd,
    },
    'gaf3': {
        'opt_key':    'gaf3',
        'vit_key':    'gaf3',
        'has_emotion':True,
        'has_person': True,
        'has_face':   True,
        'has_mmd':    True,
    },
    'mer': {
        'opt_key':    'mer',
        'vit_key':    'mer',
        'has_emotion':True,
        'has_person': True,
        'has_face':   True,
        'has_mmd':    use_mmd,
    },
    'dfew': {
        'opt_key':    'dfew',
        'vit_key':    'dfew',
        'has_emotion':True,
        'has_person': False,  # <-- no person keypoints
        'has_face':   True,
        'has_mmd':    use_mmd,
    },
    'samsemo': {
        'opt_key':    'samsemo',
        'vit_key':    'samsemo',
        'has_emotion':True,
        'has_person': True,
        'has_face':   True,
        'has_mmd':    use_mmd,
    },
    'engagenet': {
        'opt_key':    'engagenet',
        'vit_key':    'engagenet',
        'has_emotion':True,
        'has_person': True,
        'has_face':   True,
        'has_mmd':    use_mmd,
    },
    'coco': {
        'opt_key':    'coco',
        'vit_key':    None,
        'has_emotion':False,
        'has_person': True,
        'has_face':   True,
        'has_mmd':    use_mmd,
    },
    }

    # Initialize metrics
    metrics = {
        ds: {
            'count':       0,
            'cls_loss':    0.0,
            'pose_person': 0.0,
            'pose_face':   0.0,
            'mmd_loss':    0.0,
            'acc':         0.0,
        }
        for ds in selected_datasets
    }

    # Epoch‐aware samplers
    for samp in LoaderTrain.values():
        if hasattr(samp, 'set_epoch'):
            samp.set_epoch(epoch)

    for batch_data, ds in random_loader_iterator(get_loader_data_pairs(LoaderTrain, selected_datasets)):
        cfg = DATA_CFG[ds]
        opt = optimizer[cfg['opt_key']]
        opt.zero_grad()

        imgs = batch_data['images'].to(device)
        fwd = {'dataset': ds}
        if cfg['vit_key'] and 'feat_vit' in batch_data:
            fwd[cfg['vit_key']] = batch_data['feat_vit'].to(device)
            
        outputs = model(imgs, **fwd)
        loss_terms = []

        # — emotion head
        if cfg['has_emotion']:
            labels   = batch_data['label_emotion'].to(device)
            logits   = outputs['preds_emotion']
            l_cls    = CLS_Loss(logits, labels)
            loss_terms.append(l_cls)

            probs = F.softmax(logits, dim=1)
            preds    = probs.argmax(dim=1)
            correct  = (preds == labels).sum().item()
            metrics[ds]['acc']      += correct
            metrics[ds]['cls_loss'] += l_cls.item() * imgs.size(0)

        # — person pose
        if cfg['has_person'] and  getattr(model, 'use_person', True):
            kp_p  = outputs.get('skeleton_person_pred')
            adj_p = outputs.get('adjacency_person_pred')
            if (kp_p is not None) and (adj_p is not None):
                kp_gt  = batch_data['skeleton_person'].to(device)
                adj_gt = batch_data['adjacency_person'].to(device)
                l_p, _ = loss_skeleton_adjacency_with_mask(
                    kp_p, adj_p, kp_gt, adj_gt,
                    lambda_l1=1.0, lambda_adj=0.5
                )
                l_p   = beta_weight_pose * l_p
                loss_terms.append(l_p)
                metrics[ds]['pose_person'] += l_p.item() * imgs.size(0)

        # — face pose
        if cfg['has_face'] and getattr(model, 'use_face', True):
            kp_f  = outputs.get('skeleton_face_pred')
            adj_f = outputs.get('adjacency_face_pred')
            if (kp_f is not None) and (adj_f is not None):
                kp_gtf = batch_data['skeleton_face'].to(device)
                adj_gtf= batch_data['adjacency_face'].to(device)
                l_f, _ = loss_skeleton_adjacency_with_mask(
                    kp_f, adj_f, kp_gtf, adj_gtf,
                    lambda_l1=1.0, lambda_adj=0.5
                )
                l_f   = beta_weight_pose * l_f
                loss_terms.append(l_f)
                metrics[ds]['pose_face'] += l_f.item() * imgs.size(0)


        # — MMD head
        if cfg['has_mmd']:
            z    = outputs['z']
            prior= torch.randn_like(z)
            l_m  = mmd_loss_fn(z, prior) * beta_weight_mmd
            loss_terms.append(l_m)
            metrics[ds]['mmd_loss'] += l_m.item() * imgs.size(0)

        # backward + step
        total_loss = sum(loss_terms)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        metrics[ds]['count'] += imgs.size(0)

    # All‐reduce & normalize
    for ds, m in metrics.items():
        cnt = torch.tensor(m['count'], device=device)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
        m['count'] = cnt.item() or 1

        for key in ('cls_loss','pose_person','pose_face','mmd_loss','acc'):
            val = torch.tensor(m[key], device=device)
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            m[key] = val.item() / m['count']

    # Return for TensorBoard
    return {
        ds: {
            'count':       m['count'],
            'cls_loss':    m['cls_loss'],
            'acc':         m['acc'],
            'pose_person': m['pose_person'],
            'pose_face':   m['pose_face'],
            'mmd_loss':    m['mmd_loss'],
        }
        for ds, m in metrics.items()
    }



