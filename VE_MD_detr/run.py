#!/usr/bin/env python

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import argparse
import copy
import math
import numpy  as  np
import pandas as pd
import random
import shutil
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler

from torch.utils.data.dataset import Dataset

from torch.utils.tensorboard import SummaryWriter
##from torchviz import make_dot

import glob
import torch.distributed as dist


# Dependencies 
from ve_multi_decoders import *
from train_loop import *
from val_loop import *
from Mydataloders import *

RS = 2025
random.seed(RS)
np.random.seed(RS)
torch.manual_seed(RS)
torch.cuda.manual_seed(RS)
torch.backends.cudnn.deterministic = True


def train_model(model,
                LoaderTrain,
                LoaderVal,
                selected_datasets,
                add_keypoints,
                optimizer,
                CLS_Loss,
                beta_weight_mmd,
                beta_weight_pose,
                save_name_model,
                num_epochs,
                start_epoch,
                notifications,
                encoder_name,
                resume,
                writer,
                use_mmd,
                device):
    # Notify start
    since = time.time()

    # Resume if requested
    save_base = os.path.join('./Save_models/VE_MD', f'save_{save_name_model}')
    save_base_best = os.path.join('./Save_models/VE_MD_best', f'save_{save_name_model}')

    if resume and os.path.isdir(save_base):
        # load latest checkpoint
        ckpts = sorted(glob.glob(os.path.join(save_base, '*.tar')))
        if ckpts:
            print('Resuming from', ckpts[-1])
            checkpoint = torch.load(ckpts[-1])
            model.load_state_dict(checkpoint['model_state_dict'])
            # optionally load optimizers:
            for k, opt in optimizer.items():
                key = f'opt_{k}_state_dict'
                if key in checkpoint:
                    opt.load_state_dict(checkpoint[key])
        else:
            raise ValueError("No checkpoint found; training from scratch")

    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}\n' + '-'*20)
        
        # --- TRAIN ---
        train_stats = Train_epoch(
                            model,
                            selected_datasets,
                            LoaderTrain,
                            epoch,
                            optimizer,
                            CLS_Loss,
                            beta_weight_mmd,
                            beta_weight_pose,
                            use_mmd,
                            device
                        )
        
        if dist.get_rank() == 0:
            for ds, metrics in train_stats.items():
                print('\n')
                for name, value in metrics.items():
                    writer.add_scalar(f"{ds}/train_{name}", value, epoch)
                    print(f"Train: {ds}/{name} = {value:.4f}")

        # --- VALIDATE ---
        val_stats = Val_epoch(
                        model,
                        selected_datasets,
                        LoaderVal,
                        CLS_Loss,
                        device
                    )

        if dist.get_rank() == 0:
            for ds, metrics in val_stats.items():
                print('\n')
                for name, value in metrics.items():
                    writer.add_scalar(f"{ds}/val_{name}", value, epoch)
                    print(f"Val::   {ds}/{name} = {value:.4f}")

        # --- CHECKPOINT on best avg acc ---
        vgaf_acc = val_stats.get('vgaf', {}).get('acc', 0.0)
        gaf3_acc = val_stats.get('gaf3', {}).get('acc', 0.0)
        mer_acc = val_stats.get('mer', {}).get('acc', 0.0)
        dfew_acc = val_stats.get('dfew', {}).get('acc', 0.0)
        samsemo_acc = val_stats.get('samsemo', {}).get('acc', 0.0)
        engagenet_acc = val_stats.get('engagenet', {}).get('acc', 0.0)
        
        avg_acc = (vgaf_acc + gaf3_acc + mer_acc +dfew_acc+ samsemo_acc + engagenet_acc)/6
        
        if dist.get_rank() == 0:
            # Clean+recreate
            if os.path.exists(save_base):
                shutil.rmtree(save_base, ignore_errors=True)
            os.makedirs(save_base, exist_ok=True)

            if avg_acc > best_acc:
                best_acc = avg_acc
                
                 # Clean+recreate
                if os.path.exists(save_base_best):
                    shutil.rmtree(save_base_best, ignore_errors=True)
                os.makedirs(save_base_best, exist_ok=True)
                # Build checkpoint dict
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }
                for k, opt in optimizer.items():
                    ckpt[f'opt_{k}_state_dict'] = opt.state_dict()
                fname = f"epoch_{epoch}_vgaf_acc_{vgaf_acc:.4f}_gaf3_acc_{gaf3_acc:.4f}_mer_acc_{mer_acc:.4f}_samsemo_acc_{samsemo_acc:.4f}_engagenet_acc_{engagenet_acc:.4f}_dfew_acc_{dfew_acc:.4f}.tar"
                torch.save(ckpt, os.path.join(save_base_best, fname))
                print(f"Saved best performance to {fname}")

            # Save for all others 
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            for k, opt in optimizer.items():
                ckpt[f'opt_{k}_state_dict'] = opt.state_dict()
            fname = f"epoch_{epoch}_vgaf_acc_{vgaf_acc:.4f}_gaf3_acc_{gaf3_acc:.4f}_mer_acc_{mer_acc:.4f}_samsemo_acc_{samsemo_acc:.4f}_engagenet_acc_{engagenet_acc:.4f}_dfew_acc_{dfew_acc:.4f}.tar"
            torch.save(ckpt, os.path.join(save_base, fname))
            print(f"Saved best model to {fname}")


        # Flush TensorBoard writer
        if dist.get_rank() == 0:
            writer.flush()

    # --- FINISH ---
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    return model


# manage GPUs fro DataParallel
def set_gpus_parall(gpu_ids):
        """
        Set the GPUs to be used in PyTorch training.

        Args:
        gpu_ids (list of int): List of GPU IDs to be used for training. 
                            Pass an empty list to use ALL GPUs available
        """
        if not gpu_ids:
            print("Using all available GPUs for training.")
        else:
            gpu_ids_str = ','.join(map(str, gpu_ids))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
            print(f"GPUs {gpu_ids_str} for training.")


## Init DDP
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '2541'+str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    parser = argparse.ArgumentParser(description='parameters to train VE-MD for emotion recognition')
    #Training model settings
    parser.add_argument('--resume', action='store_true',
                       help="If you want to resume the training from the last saved epoch (default:False)")
    parser.add_argument('--num_epochs',type=int,default=100, metavar='epoch',
                        help='Number of epochs for the training (default: 100)')
    parser.add_argument('--start_epoch',type=int,default=0, metavar='epoch',
                        help='Start epoch for training if resume==True, in this case provide the start epoch (default: 0)')
    parser.add_argument('--num_workers',type=int,default=120, metavar='workers',
                        help='num workers (default: 120)')
    
    # Encoders 
    parser.add_argument('--latent_dim',type=int, default=512,
                       help='Dimension of the latent space (default:512)')
    parser.add_argument('--encoder_name',type=str, default='residual',metavar='name',
                       help='Encoder to use: residual,resnet50, resnet152; (default:"resnet50")')
    parser.add_argument('--channel_factor',type=int, default=64,
                       help='input factor of channel of the network CNN (default:64)')
    parser.add_argument("--resblocks",type=int, default=5,
                        help="number of ResBlocks in the VAE (default:5)")
    parser.add_argument('--two_encoders', action='store_true',
                       help="If you want to use two Encoder Multi-taskh (default:False)")
    parser.add_argument('--use_person_pose', action='store_true',
                       help="to use person pose in the training (default:False)") 
    parser.add_argument('--use_face_pose', action='store_true',
                       help="to use face pose in the training (default:False)")       
    
    # Classifiacion Head
    parser.add_argument('--dropout_classif', type=float, default=0.0,  
                        help='dropout classification  (default:0.0)')
    parser.add_argument('--add_keypoints', action='store_true',
                       help=" Using keypoints in the classification head (default:False)")
    parser.add_argument('--classif_projection', action='store_true',
                       help=" Using classif_projection in the classification head (default:False)")
    parser.add_argument("--proj_dim",type=int, default=512,
                        help=" projection dimension (default:512)")
    parser.add_argument('--stgcn_active', action='store_true',
                       help=" Using stgcn in the classification skeleton (default:False)")
    parser.add_argument('--pos_enc_type', type=str, default='sinusoidal', 
                        help="positional enc type: learnable, sinusoidal, relative, None")


    
     
    # PETR parameters
    parser.add_argument('--nun_heads_petr',type=int, default=8,
                       help='Number of Heads for transformers in DETR (default:8)')
    parser.add_argument('--num_layers_petr',type=int, default=3,
                       help='Number of Encoder Layers for transformers in DETR (default:3)')
    parser.add_argument('--num_decoder_layers_petr',type=int, default=3,
                       help='Number of Decoder Layers for transformers in DETR (default:3)')
    parser.add_argument("--num_queries",type=int, default=100,
                        help="number of object class to detect if person--> class=100: (default:100)")

                     
    # Devices
    parser.add_argument('--cuda_device', type=int, default=0, 
                        help='GPU to use if not Dataparallel, if cuda = -1 ==> cpu, (default:0)')
    parser.add_argument('--gpus', nargs='+', type=int, default=[],
                        help='List of GPU IDs to use in case of DataParallel. Example: --gpus 0 1')
    parser.add_argument('--dataparall',  action='store_true',
                       help="If you want to train with DataParallel (default:False)")
    parser.add_argument('--portvalue', type=int, default=0, 
                        help='port values to listen ofr DDP')
    

    # Data parameters
    parser.add_argument('--datasets', type=str, nargs='+', default=['vgaf', 'gaf3'], 
                        help="List of datasets to use in training. Options: 'vgaf','gaf3', 'dfew', 'mer', 'engagenet', 'coco")
    parser.add_argument('--dataset_emotion',type=str, default='vgaf',metavar='name',
                       help='data name to use: vgaf,mer, dfew ,samsemo, engagenet (default:"vgaf")')
    parser.add_argument('--nb_frames_vgaf',type=int,default=5, metavar='frames',
                        help='number of frames emotion (default: 5)')
    parser.add_argument('--nb_frames_engage',type=int,default=1, metavar='frames',
                        help='number of frames enagegenet (default: 10)')
    parser.add_argument('--synt_percent',type=int,default=0, metavar='object',
                        help='Percentage of synthetic dataset (default: 0)')
    parser.add_argument('--nb_person_max',type=int, default=10, metavar='object',
                        help='number max of alls to detect from images (default: 10)')
    parser.add_argument('--score_person',type=float, default=0.1, metavar='object',
                        help='score of bbox to be considered from annotation (default: 0.1)')
    parser.add_argument('--dir_data_parent',type=str,
                        default='/Corpora/...',metavar='Dir',
                       help='Parent directory path for dataset; (default:"/Corpora/")')
    parser.add_argument('--dir_data_coco_parent',type=str,
                        default='/Corpora/COCO/',metavar='Dir',
                       help='Parent directory path for COCO dataset; (default:"/Corpora/COCO/")')
    
    parser.add_argument('--batch_vgaf',type=int,default=8, metavar='batch',
                        help='batch size for training emotion (default: 8)')
    parser.add_argument('--batch_gaf3',type=int,default=32, metavar='batch',
                        help='batch size for training emotion (default: 32)')
    parser.add_argument('--batch_mer',type=int,default=10, metavar='batch',
                        help='batch size for training (default: 10)')
    parser.add_argument('--batch_samsemo',type=int,default=10, metavar='batch',
                        help='batch size for training emotion (default: 10)')
    parser.add_argument('--batch_engage',type=int,default=10, metavar='batch',
                        help='batch size for training emotion (default: 10)')
    parser.add_argument('--batch_dfew',type=int,default=10, metavar='batch',
                        help='batch size for training emotion (default: 10)')
    parser.add_argument('--batch_coco', type=int, default=32, metavar='batch',
                        help='batch size for training coco  (default: 10)')
    
    # Hyper parameters
    parser.add_argument('--LR_emotion', type=float, default=0.000001, metavar='LR',
                        help='Learning rate for the optimizer (default:0.0000001)')
    parser.add_argument('--LR_mer', type=float, default=0.000001, metavar='LR',
                        help='Learning rate for the optimizer (default:0.0000001)')
    parser.add_argument('--LR_coco', type=float, default=0.0000001, metavar='LR',
                        help='Learning rate for the optimizer (default:0.0000001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for the optimizer (default:0.9)')
    parser.add_argument('--beta_weight_pose',type=float,default=0.1, metavar='beta',
                        help='value for giving weight pose loss   (default:1.0)')
    parser.add_argument('--beta_weight_mmd',type=float,default=1.0, metavar='beta',
                        help='value for giveing weight to MMD  (default:0.1)')
    parser.add_argument('--beta_weight_cls',type=int,default=1, metavar='beta',
                        help='beta value for giveing weight to CLS loss classification (default:1)')
    parser.add_argument('--use_mmd', action='store_true',
                       help="To use mmd loss or not. if Not no VE ) (default:False)")  
    parser.add_argument('--notifications', action='store_true',
                       help="If you want to receive notifications with pushover on your device (cellphone or others) (default:False)")  
                   
    args = parser.parse_args()  
    setup(rank, world_size, args.portvalue)  # for DDP
    
    # some restrcitions
    if (args.use_face_pose and not args.use_person_pose) or (not args.use_face_pose and args.use_person_pose):
        two_encoders=False
    else:
        two_encoders=args.two_encoders
    
    # 1) master config for every supported dataset
    dataset_configs = {
            'vgaf':      {'nb_frames': args.nb_frames_vgaf,  'num_class': 3},
            'gaf3':      {'nb_frames': 1,  'num_class': 3},
            'dfew':      {'nb_frames': 16, 'num_class': 7},
            'mer':       {'nb_frames': 10, 'num_class': 6},
            'engagenet': {'nb_frames': 10, 'num_class': 4},
            'samsemo':   {'nb_frames': 10, 'num_class': 5},
            'coco':   {'nb_frames': None,'num_class': None},  # no emotion head
            }

    # 2) pick only those the user asked for (and that actually have classes)
    cls_counts = {
        ds: dataset_configs[ds]['num_class']
        for ds in args.datasets
        if dataset_configs.get(ds, {}).get('num_class') is not None
            }

    # 3) also read nb_frames_video from the same map if you need it
    nb_frames_video = list([dataset_configs[ds]['nb_frames'] 
                       for ds in args.datasets
                       if dataset_configs.get(ds, {}).get('nb_frames') is not None
                       ])      
    # device     
    device = rank
    
    ## Load the model 
    model = VE_MultiDecoder(device=device,
                            encoder_name=args.encoder_name,
                            two_encoders= two_encoders,
                            channel_factor=args.channel_factor,
                            blocks_res=args.resblocks,
                            latent_dim=args.latent_dim,
                            nun_heads_petr = args.nun_heads_petr,
                            num_encoder_layers_petr = args.num_layers_petr,
                            num_decoder_layers_petr= args.num_layers_petr,
                            dropout_classif=args.dropout_classif,
                            stgcn_active=args.stgcn_active,
                            num_queries = args.num_queries,
                            cls_counts=cls_counts, 
                            use_person_pose=args.use_person_pose,
                            use_face_pose=args.use_face_pose,
                            add_keypoints = args.add_keypoints,
                            pos_enc_type=args.pos_enc_type,
                            classif_projection=args.classif_projection,
                            proj_dim=args.proj_dim,
                            )
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    if dist.get_rank() == 0: 
        print('model loaded ...')
    
   # parameters 
    params = sum(p.numel() for p in model.parameters())
    if dist.get_rank() == 0: 
        print('Number of parameters is:',params)
                
    # Optimizer  
    optimizer = {
                "vgaf": optim.Adam(model.parameters(), lr=args.LR_emotion),
                "gaf3": optim.Adam(model.parameters(), lr=args.LR_emotion),
                "mer": optim.Adam(model.parameters(), lr=args.LR_emotion),
                "dfew": optim.Adam(model.parameters(), lr=args.LR_emotion),
                "samsemo": optim.Adam(model.parameters(), lr=args.LR_emotion),
                "engagenet": optim.Adam(model.parameters(), lr=args.LR_emotion),
                "coco": optim.Adam(model.parameters(), lr=args.LR_coco),
                }
    
    CLS_Loss = nn.CrossEntropyLoss()

    
    #Loading DATA
    if  args.nb_frames_vgaf == 5:
        Dir_vgaf_train = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/Train_skeleton_fa_tensor')
        Dir_vgaf_train_vit = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/ViT_Feature_Nosynt/Train')
        Dir_vgaf_val = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/Val.h5')
        Dir_vgaf_val_vit = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/ViT_Feature_Nosynt/Val')

        vgaf_train_csvfile = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/train_video_annotation_all_vitpose_kp_fa.json')
        vgaf_val_csvfile = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/Val_labels.txt')
        
    else:
        Dir_vgaf_train = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/Train_skeleton_fa_tensor')
        Dir_vgaf_train_vit = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/ViT_Feature/Train')
        Dir_vgaf_val = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/Val.h5')
        Dir_vgaf_val_vit = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/ViT_Feature/Val')
        
        vgaf_train_csvfile = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/train_video_annotation_all_vitpose_kp_fa.json')
        vgaf_val_csvfile = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/Val_labels.txt')
        
        
    Dir_gaf3_train = os.path.join(str(args.dir_data_parent),'GAF_3.0/Train')
    Dir_gaf3_train_heatmap = os.path.join(str(args.dir_data_parent),'GAF_3.0/Train_skeleton_fa_tensor')
    Dir_gaf3_train_vit = os.path.join(str(args.dir_data_parent),'GAF_3.0/ViT_Feature/Train')
    Dir_gaf3_val = os.path.join(str(args.dir_data_parent),'GAF_3.0/Val')
    Dir_gaf3_val_vit = os.path.join(str(args.dir_data_parent),'GAF_3.0/ViT_Feature/Val')
    
    gaf3_train_csvfile = os.path.join(str(args.dir_data_parent),'GAF_3.0/train_annotation_all_vitpose_fa.json')
    gaf3_val_csvfile = os.path.join(str(args.dir_data_parent),'GAF_3.0/Val_labels.txt')
    
    Dir_dfew_train = os.path.join(str(args.dir_data_parent),'DFEW/Train_skeleton_tensor')
    Dir_dfew_train_vit = os.path.join(str(args.dir_data_parent),'DFEW/ViT_Feature/Train')
    Dir_dfew_val = os.path.join(str(args.dir_data_parent),'DFEW/dfew.h5')
    Dir_dfew_val_vit = os.path.join(str(args.dir_data_parent),'DFEW/ViT_Feature/Val')
    dfew_train_csvfile = os.path.join(str(args.dir_data_parent),'DFEW/train_video_annotation_fa_kp_set_1.json') # addition to  set_2,set_3 ...set_5
    dfew_val_csvfile = os.path.join(str(args.dir_data_parent),'DFEW/test_label/set_1.csv') # addition to set_2,set_3 ...set_5

    Dir_mer_train = os.path.join(str(args.dir_data_parent),'MER2023_10/Train_skeleton_fa_tensor')
    Dir_mer_train_vit = os.path.join(str(args.dir_data_parent),'MER2023_10/ViT_Feature/Train')
    Dir_mer_val = os.path.join(str(args.dir_data_parent),'MER2023_10/val.h5') # test also
    Dir_mer_val_vit = os.path.join(str(args.dir_data_parent),'MER2023_10/ViT_Feature/Val')
    
    mer_train_csvfile = os.path.join(str(args.dir_data_parent),'MER2023_10/train_video_annotation_all_vitpose_kp_fa.json')
    mer_val_csvfile =  os.path.join(str(args.dir_data_parent),'MER2023_10/test1-label.csv') # 'test2-label.csv
    
    Dir_samsemo_train = os.path.join(str(args.dir_data_parent),'SamSemo/Train_skeleton_fa_tensor')
    Dir_samsemo_train_vit = os.path.join(str(args.dir_data_parent),'SamSemo/ViT_Feature/Train')
    Dir_samsemo_val = os.path.join(str(args.dir_data_parent),'SamSemo/val.h5')
    Dir_samsemo_val_vit = os.path.join(str(args.dir_data_parent),'SamSemo/ViT_Feature/Val')
    
    samsemo_train_csvfile = os.path.join(str(args.dir_data_parent),'SamSemo/train_video_annotation_all_vitpose_kp_fa.json')
    samsemo_val_csvfile = os.path.join(str(args.dir_data_parent),'SamSemo/val_labels.tsv')
    
    Dir_engage_train = os.path.join(str(args.dir_data_parent),'Engage_images_10/Train_skeleton_fa_tensor')
    Dir_engage_train_vit = os.path.join(str(args.dir_data_parent),'Engage_images_10/ViT_Feature/Train')
    Dir_engage_val = os.path.join(str(args.dir_data_parent),'Engage_images_10/Val.h5')
    Dir_engage_val_vit = os.path.join(str(args.dir_data_parent),'Engage_images_10/ViT_Feature/Val')
    
    engage_train_csvfile = os.path.join(str(args.dir_data_parent),'Engage_images_10/train_video_annotation_all_vitpose_kp_fa.json')
    engage_val_csvfile = os.path.join(str(args.dir_data_parent),'Engage_images_10/val_labels.txt')
    


    Dir_coco_train = os.path.join(str(args.dir_data_parent),'COCO_bbox_person/train_224x224') 
    Dir_coco_train_kp = os.path.join(str(args.dir_data_parent),'COCO_bbox_person/Train_skeleton_fa_tensor')
    coco_train_csvfile = os.path.join(str(args.dir_data_parent),'COCO_bbox_person/train_annotation_all_vitpose_kp_fa.json')

    

    LoaderTrain = Train_DataLoaders(rank,
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
                                    Dir_coco_train,
                                    Dir_coco_train_kp,  
                                    coco_train_csvfile,
                                    batch_vgaf=args.batch_vgaf,
                                    batch_gaf3=args.batch_gaf3,
                                    batch_dfew=args.batch_dfew,
                                    batch_mer=args.batch_mer,
                                    batch_engage=args.batch_engage,
                                    batch_samsemo=args.batch_samsemo,
                                    batch_coco=args.batch_coco,  
                                    )
    
    LoaderVal =  Val_DataLoaders(rank,
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
                                batch_vgaf=args.batch_vgaf,
                                batch_gaf3=args.batch_gaf3,
                                batch_dfew=args.batch_dfew,
                                batch_mer=args.batch_mer,
                                batch_engage=args.batch_engage,
                                batch_samsemo=args.batch_samsemo,
                                )


    # save model name 
    save_name_model = 'VE_MD_detr_'+str(args.encoder_name)+'_num_query_'+str(args.num_queries)+'_use_pers_'+\
        str(args.use_person_pose)+'_use_face_'+str(args.use_face_pose)+'_two_enc_'+str(args.two_encoders)+'_add_kp_'+\
            str(args.add_keypoints)+'pos_enc_'+str(args.pos_enc_type)+'_stgcn_active_'+str(args.stgcn_active)+'_latent_dim_'+str(args.latent_dim)+'_use_mmd_'+str(args.use_mmd)+ \
            '_'+str("_".join(args.datasets))+'_nbfr_'+str("_".join(str(nb_frames_video[i]) for i in range(len(nb_frames_video))))+'_classif_proj_'+str(args.classif_projection)
            
    writer = SummaryWriter('TB/TB_'+str(save_name_model))

   
    model = train_model(model,
                        LoaderTrain,
                        LoaderVal,
                        selected_datasets = args.datasets,  # This will be the list of datasets passed from the command line
                        add_keypoints=args.add_keypoints,
                        optimizer=optimizer,
                        CLS_Loss=CLS_Loss,
                        beta_weight_mmd=args.beta_weight_mmd,
                        beta_weight_pose = args.beta_weight_pose, 
                        save_name_model=save_name_model,
                        num_epochs=args.num_epochs,
                        start_epoch=args.start_epoch,
                        notifications=args.notifications,
                        encoder_name=args.encoder_name,
                        resume=args.resume,
                        writer=writer,
                        use_mmd=args.use_mmd,
                        device=device)
    

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)










