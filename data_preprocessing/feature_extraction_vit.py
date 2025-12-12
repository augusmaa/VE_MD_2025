
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
import matplotlib.pyplot as plt
import numpy as np
import os
import timm
from PIL import Image
from torchvision import transforms


import glob

RS = 2025
random.seed(RS)
np.random.seed(RS)
torch.manual_seed(RS)
torch.cuda.manual_seed(RS)
torch.backends.cudnn.deterministic = True


# VGAF transforms
general_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


# Decoder Emotion VGAF
class FramesAttentionPooling(nn.Module):
    def __init__(self, d_model, device):
        super(FramesAttentionPooling, self).__init__()
        
        self.d_model = d_model
        self.attention = nn.Sequential(
                        nn.Linear(d_model, 1),
                        nn.Softmax(dim=1)
                        ).to(device)

    def forward(self, x):
        # x (batch_size, sequence_length, d_model)
        attention_scores = self.attention(x)# shape (batch_size, sequence_length, 1)
        # Use the attention scores to create a weighted sum of the input vectors
        output = attention_scores * x # (batch_size, sequence_length, d_model)
        # Sum over the sequence dimension to get the final vector
        output = output.sum(dim=1)
        return output # output has shape (batch_size, d_model)
    

# VitL model
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class VideoVitNet(nn.Module):
    def __init__(self, device, nb_class=3):
        super(VideoVitNet, self).__init__()        

        self.vit = timm.create_model('vit_large_patch14_224_clip_laion2b', pretrained=True).to(device)
        for param in self.vit.parameters(): # Not compute gradient
            param.requires_grad = False
        self.vit.head = Identity()
        self.att_pooling = FramesAttentionPooling(1024, device).to(device)
        self.linear_out =  nn.Sequential(nn.Linear(1024, nb_class),
                                         nn.Softmax(dim=1),
                                        ).to(device)
        
    def forward(self, videos):
        #input img (bs, frames, c, h, w)
        bs,frames,C,H,W= videos.size()
        videos = videos.view(bs*frames,C, H, W)
        X_videos = self.vit(videos)#.to(device)
        X_videos = X_videos.view(bs, frames, -1)
        return self.linear_out(self.att_pooling(X_videos))


import h5py

def SaveFeatureSamSemo(model,
                    device,
                    Dir_save_folder,
                    Dir_data_train,
                    train_file_label,
                    Dir_data_val, 
                    val_file_label,
                    nb_of_frames, 
                    Train_flag = True):
    
    if not os.path.exists(Dir_save_folder):
        os.mkdir(Dir_save_folder)
    
    if Train_flag:
        dataset_type='Train'
        Dir_data = Dir_data_train
        csv_file = train_file_label
    else:
        dataset_type='Val'
        Dir_data = Dir_data_val
        csv_file = val_file_label
        
    path_save = os.path.join(Dir_save_folder, dataset_type)
    
    # if os.path.exists(path_save):
    #     shutil.rmtree(path_save)
    os.makedirs(path_save, exist_ok=True)

     
    file_labels = pd.read_csv(csv_file, engine='python', sep='\t')
    dir_data = h5py.File(Dir_data, 'r')
    
    model.eval()    
    with torch.no_grad():
         for i, row in file_labels.iterrows():
            video_name = row['utterance_id']
            frames = dir_data[video_name][:]
            
            # -- Load and process the image
            images=[]
            for frame in frames:
                images.append(general_transforms['val'](frame))
            
            images = torch.stack(images).unsqueeze(0).to(device)
            feature_emotion = model(images)
        
            output = path_save+'/'+ str(video_name)+'.vit_feat.pt'
            torch.save(feature_emotion, output)
            print(output)
              #break      
    

def SaveFeatureMER(model,
                device,
                path_save,
                Dir_data,
                csv_file,
               ):
    
    # if os.path.exists(path_save):
    #     shutil.rmtree(path_save)
    os.makedirs(path_save, exist_ok=True)

    
    file_labels = pd.read_csv(csv_file)
    
    dir_data = h5py.File(Dir_data, 'r')

    
    model.eval()
    with torch.no_grad():
         for i, row in file_labels.iterrows():
            video_name = row['name']
            frames = dir_data[video_name+'.img'][:10]
            
            # -- Load and process the image
            images=[]
            for frame in frames:
                images.append(general_transforms['val'](frame))
            
            images = torch.stack(images).unsqueeze(0).to(device)
            feature_emotion = model(images)        
            output = path_save+'/'+ str(video_name)+'.vit_feat.pt'
            torch.save(feature_emotion, output)
            print(output)
              #break  


def SaveFeatureVGAF(model,
                device,
                Dir_save_folder,
                Dir_data_train,
                train_file_label,
                Dir_data_val, 
                val_file_label,
                nb_of_frames, 
                Train_flag = True):
    
    if not os.path.exists(Dir_save_folder):
        os.mkdir(Dir_save_folder)
    
    if Train_flag:
        dataset_type='Train'
        Dir_data = Dir_data_train
        csv_file = train_file_label
    else:
        dataset_type='Val'
        Dir_data = Dir_data_val
        csv_file = val_file_label
        
    path_save = os.path.join(Dir_save_folder, dataset_type)
    
    
    if os.path.exists(path_save):
        shutil.rmtree(path_save)
    os.makedirs(path_save, exist_ok=True)

    hf = h5py.File(Dir_data, 'r')    
    file_labels = pd.read_csv(csv_file, engine='python', sep=' ')
    
    model.eval()
    with torch.no_grad():
        for index in range(len(file_labels)):
            video_name = file_labels['Vid_name'][index]
            # -- Load and process the image
            frames = hf[video_name+'.mp4.img'][:]
            # Apply transformation to each frame
            frames = [general_transforms['val'](frame) for frame in frames]
            images = torch.stack(frames).unsqueeze(0).to(device)
            feature_emotion = model(images)
            output = path_save+'/'+ str(video_name)+'.vit_feat.pt'
            torch.save(feature_emotion, output)
            print(output)
              #break      
    

def SaveFeatureGAF3(model,
                device,
                Dir_save_folder,
                Dir_data_train,
                train_file_label,
                Dir_data_val, 
                val_file_label,
                nb_of_frames, 
                Train_flag = True):
    
    if not os.path.exists(Dir_save_folder):
        os.mkdir(Dir_save_folder)
    
    if Train_flag:
        dataset_type='Train'
        Dir_data = Dir_data_train
        csv_file = train_file_label
    else:
        dataset_type='Val'
        Dir_data = Dir_data_val
        csv_file = val_file_label
        
    path_save = os.path.join(Dir_save_folder, dataset_type)
    
    
    if os.path.exists(path_save):
        shutil.rmtree(path_save)
    os.makedirs(path_save, exist_ok=True)

    file_labels = pd.read_csv(csv_file, engine='python', sep=',')
    
    model.eval()
    with torch.no_grad():
        for index in range(len(file_labels)):
           # Load and process the image
            video_name = file_labels['Img_name'][index]
            image_context = Image.open(os.path.join(Dir_data, video_name))
            if image_context.mode != 'RGB':
                image_context = image_context.convert('RGB')
            image_context = image_context.resize((224, 224), Image.LANCZOS)
            images =  general_transforms['val'](image_context).unsqueeze(0).to(device)
            feature_emotion = model(images.unsqueeze(0))
            output = path_save+'/'+ str(video_name.split('.')[0])+'.vit_feat.pt'
            torch.save(feature_emotion, output)
            print(output)
              #break      
    

def SaveFeatureDFEW(model,
                device,
                Dir_save_folder,
                Dir_data_train,
                train_file_label,
                Dir_data_val, 
                val_file_label,
                nb_of_frames, 
                Train_flag = True):
    
    if not os.path.exists(Dir_save_folder):
        os.mkdir(Dir_save_folder)
    
    if Train_flag:
        dataset_type='Train'
        Dir_data = Dir_data_train
        csv_file = train_file_label
    else:
        dataset_type='Val'
        Dir_data = Dir_data_val
        csv_file = val_file_label
        
    path_save = os.path.join(Dir_save_folder, dataset_type)
    
    if os.path.exists(path_save):
        shutil.rmtree(path_save)
    os.makedirs(path_save, exist_ok=True)

    hf = h5py.File(Dir_data, 'r')    
    file_labels = pd.read_csv(csv_file)
    
    model.eval()
    with torch.no_grad():
        for index in range(len(file_labels)):
           # Convert the video_name from CSV to zero-padded format
            video_name = str(file_labels['video_name'][index]).zfill(5)
            # Load and transform images from HDF5 file
            frames = hf[video_name][:]
            # Apply transformation to each frame
            frames = [general_transforms['val'](frame) for frame in frames]
            images = torch.stack(frames).unsqueeze(0).to(device)
            feature_emotion = model(images)
            output = path_save+'/'+ str(video_name)+'.vit_feat.pt'
            torch.save(feature_emotion, output)
            print(output)
    
    

def SaveFeatureEngageNet(model,
                        device,
                        Dir_save_folder,
                        Dir_data_train,
                        train_file_label,
                        Dir_data_val, 
                        val_file_label,
                        nb_of_frames, 
                        Train_flag = True):
            
    if not os.path.exists(Dir_save_folder):
        os.mkdir(Dir_save_folder)
    
    if Train_flag:
        dataset_type='Train'
        Dir_data = Dir_data_train
        csv_file = train_file_label
    else:
        dataset_type='Val'
        Dir_data = Dir_data_val
        csv_file = val_file_label
        
    path_save = os.path.join(Dir_save_folder, dataset_type)
    
    if os.path.exists(path_save):
        shutil.rmtree(path_save)
    os.makedirs(path_save, exist_ok=True)

    hf = h5py.File(Dir_data, 'r')    
    file_labels = pd.read_csv(csv_file, engine='python', sep=',')
    
    model.eval()
    with torch.no_grad():
        for index in range(len(file_labels)):
           # Convert the video_name from CSV to zero-padded format
            video_name = (file_labels['chunk']+'.img')[index]
            # Load and transform images from HDF5 file
            frames = hf[video_name][:]
            # Apply transformation to each frame
            frames = [general_transforms['val'](frame) for frame in frames]
            images = torch.stack(frames).unsqueeze(0).to(device)
            feature_emotion = model(images)
            output = path_save+'/'+ str(video_name)+'.vit_feat.pt'
            torch.save(feature_emotion, output)
            print(output)
              #break  


def ExtractViTFeaT(pretrained_vit_path,
                    num_class_emotion, 
                    device, 
                    ):
            #vit 
        vit_model = VideoVitNet(device, num_class_emotion)#.to(device)
        vit_model = nn.DataParallel(vit_model)
        
        # Load saved video model
        video_saved = glob.glob(pretrained_vit_path +'*.tar')
        if len(video_saved)>0:
            model_file_video = video_saved[0]
            checkpoint = torch.load(model_file_video)
            vit_model.load_state_dict(checkpoint['model_state_dict'])
            print('Pretrained vit from', model_file_video)
        else:
            raise ValueError('No saved video model found for ViT')
        
        # Create a new state dict to remove 'module' prefic from keys of state_dict' dataparallel
        new_state_dict_video = {}
        for key , value in vit_model.state_dict().items():
            new_key = key.replace('module.','') # remove 'module' prefix
            new_state_dict_video[new_key] = value
        # Recall the model 
        vit_model = VideoVitNet(device, num_class_emotion).to(device)
        vit_model.load_state_dict(new_state_dict_video)
                   
        vit_model.att_pooling=Identity()
        vit_model.linear_out = Identity()
        for param in vit_model.parameters(): 
            param.requires_grad = False
            
        return vit_model

def main():
    #Training model settings
    parser = argparse.ArgumentParser(description='ViT Feat Extraction')
    # Devices
    parser.add_argument('--cuda_device', type=int, default=0, 
                        help='GPU to use if not Dataparallel, if cuda = -1 ==> cpu, (default:0)')
    parser.add_argument('--dataset_emotion',type=str, default='vgaf',metavar='name',
                       help='data name to use: vgaf,mer, dfew ,samsemo, engagenet (default:"vgaf")')
    parser.add_argument('--dataset_type',type=str, default='train',metavar='name',
                       help='data type: train, test, val, test2')
    parser.add_argument('--nb_frames_vgaf',type=int,default=5, metavar='frames',
                        help='number of frames emotion (default: 5)')
    parser.add_argument('--dir_data_parent',type=str,
                        default='/Corpora/VGAF/',metavar='Dir',
                       help='Parent directory path for dataset; (default:"/Corpora/VGAF/")')
    parser.add_argument('--train_flag', action='store_true',
                       help=" Train of Val feature dataset (default:False)")
    
    args = parser.parse_args()    
    
     # pretrained vit path for video emotion
    if (args.dataset_emotion=='vgaf'):
        print("Dataset Emotion:", args.dataset_emotion)
        nb_frames_video = 5
        num_class =3
        pretrained_vit_path = '/Corpora/VGAF/Save_models/pretrained_ViT_attp/save_model_video_5fr_flatten_False_synt_rate_0.0%_Nb_block_RW_24/'
        #pretrained_vit_path = '/Corpora/VGAF/Save_models/pretrained_ViT_attp/save_model_video_5fr_flatten_False_synt_rate_30.0%_Nb_block_RW_24/'
     # pretrained vit path for video emotion
    elif (args.dataset_emotion=='gaf3'):
        print("Dataset Emotion:", args.dataset_emotion)
        nb_frames_video=1
        num_class =3
        pretrained_vit_path = '/Corpora/VGAF/Save_models/pretrained_ViT_attp/save_model__dataset_name_gaf3_Nb_blocks_released_24/'
    elif args.dataset_emotion=='dfew':
        print("Dataset Emotion:", args.dataset_emotion)
        nb_frames_video = 16
        num_class =7
        pretrained_vit_path ='/Corpora/VGAF/Save_models/pretrained_ViT_attp/save_model__dataset_name_dfew_Nb_blocks_released_24/'
    elif args.dataset_emotion=='mer':
        print("Dataset Emotion:", args.dataset_emotion)
        nb_frames_video = 10
        num_class =6
        pretrained_vit_path = '/Corpora/VGAF/Save_models/pretrained_ViT_attp/save_model__dataset_name_mer2023_Nb_blocks_released_24/'
    elif args.dataset_emotion=='engagenet':
        print("Dataset Emotion:", args.dataset_emotion)
        nb_frames_video = 10
        num_class =4
        pretrained_vit_path = '/Corpora/VGAF/Save_models/pretrained_ViT_attp/save_model_engagement__Nb_blocks_released_24/'
    elif args.dataset_emotion=='samsemo':
        print("Dataset Emotion:", args.dataset_emotion)
        nb_frames_video = 10
        num_class =5
        pretrained_vit_path ='/Corpora/VGAF/Save_models/pretrained_ViT_attp/save_model__dataset_name_samsemo_Nb_blocks_released_24/'
    else:
        raise ValueError('Dataset Emotion name does not exist')

    # device        
    device = torch.device('cuda:'+str(args.cuda_device) if torch.cuda.is_available() else "cpu")      
    model = ExtractViTFeaT(pretrained_vit_path, num_class, device)
    # parameters 
    params = sum(p.numel() for p in model.parameters())
    print('Number of parameters is:',params)
            
   
    if  args.dataset_emotion=='samsemo':
        Dir_save = os.path.join(str(args.dir_data_parent),'SamSemo/ViT_Feature')
        Dir_train = os.path.join(str(args.dir_data_parent),'SamSemo/train.h5')
        Dir_val = os.path.join(str(args.dir_data_parent),'SamSemo/val.h5')
        train_csvfile = os.path.join(str(args.dir_data_parent),'SamSemo/train_labels.tsv')
        val_csvfile = os.path.join(str(args.dir_data_parent),'SamSemo/val_labels.tsv')
        
        SaveFeatureSamSemo(model,
                        device,
                        Dir_save,
                        Dir_train,
                        train_csvfile,
                        Dir_val, 
                        val_csvfile,
                        nb_frames_video, 
                        Train_flag=args.train_flag,)
        
    
    elif (args.dataset_emotion=='mer'):
        Dir_save_folder = os.path.join(str(args.dir_data_parent),'MER2023_10/ViT_Feature')
        Dir_data_train = os.path.join(str(args.dir_data_parent),'MER2023_16/train.h5')
        Dir_data_val = os.path.join(str(args.dir_data_parent),'MER2023_10/val_10.h5')
        Dir_data_test2 = os.path.join(str(args.dir_data_parent),'MER2023_16/test2.h5')
        Dir_data_test3 = os.path.join(str(args.dir_data_parent),'MER2023_16/test3.h5')

        
        train_file_label = os.path.join(str(args.dir_data_parent),'MER2023_16/train-label.csv')
        val_file_label = os.path.join(str(args.dir_data_parent),'MER2023_16/test1-label.csv')
        test2_file_label = os.path.join(str(args.dir_data_parent),'MER2023_16/test2-label.csv')
        test3_file_label = os.path.join(str(args.dir_data_parent),'MER2023_16/test3-label.csv')
        
        if not os.path.exists(Dir_save_folder):
            os.mkdir(Dir_save_folder)
    
        if args.dataset_type=='Train':
            Dir_data = Dir_data_train
            csv_file = train_file_label
        elif args.dataset_type=='Val':
            Dir_data = Dir_data_val
            csv_file = val_file_label
        elif args.dataset_type=='test2':
            Dir_data = Dir_data_test2
            csv_file = test2_file_label
        elif args.dataset_type=='test3':
            Dir_data = Dir_data_test3
            csv_file = test3_file_label
        else:
            raise ValueError("Dataset Type does not exist")

        path_save = os.path.join(Dir_save_folder, args.dataset_type)
    
        SaveFeatureMER(model,
                        device,
                        path_save,
                        Dir_data,
                        csv_file,
                        )
        
    elif (args.dataset_emotion=='vgaf'):
        if args.nb_frames_vgaf==5:
            Dir_save = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/ViT_Feature_Nosynt')
            Dir_train = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/Train.h5')
            Dir_val = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/Val.h5')
            train_csvfile = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/Train_labels.txt')
            val_csvfile = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_5/Val_labels.txt')
            
        else:
            Dir_save = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/ViT_Feature')
            Dir_train = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/Train.h5')
            Dir_val = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/Val.h5')
            train_csvfile = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/Train_labels.txt')
            val_csvfile = os.path.join(str(args.dir_data_parent),'VGAF_Video_images_25/Val_labels.txt')
            
        SaveFeatureVGAF(model,
                        device,
                        Dir_save,
                        Dir_train,
                        train_csvfile,
                        Dir_val, 
                        val_csvfile,
                        nb_frames_video, 
                        Train_flag=args.train_flag,)
        
    elif (args.dataset_emotion=='gaf3'):
        Dir_save = os.path.join(str(args.dir_data_parent),'GAF_3.0/ViT_Feature')
        Dir_train = os.path.join(str(args.dir_data_parent),'GAF_3.0/Train')
        Dir_val = os.path.join(str(args.dir_data_parent),'GAF_3.0/Val')
        train_csvfile = os.path.join(str(args.dir_data_parent),'GAF_3.0/Train_labels.txt')
        val_csvfile = os.path.join(str(args.dir_data_parent),'GAF_3.0/Val_labels.txt')
        
        SaveFeatureGAF3(model,
                        device,
                        Dir_save,
                        Dir_train,
                        train_csvfile,
                        Dir_val, 
                        val_csvfile,
                        nb_frames_video, 
                        Train_flag=args.train_flag,)
        
    elif (args.dataset_emotion=='engagenet'):
        Dir_save = os.path.join(str(args.dir_data_parent),'Engage_images_10/ViT_Feature')
        Dir_train = os.path.join(str(args.dir_data_parent),'Engage_images_10/Train.h5')
        Dir_val = os.path.join(str(args.dir_data_parent),'Engage_images_10/Val.h5')
        train_csvfile = os.path.join(str(args.dir_data_parent),'Engage_images_10/train_labels.txt')
        val_csvfile = os.path.join(str(args.dir_data_parent),'Engage_images_10/val_labels.txt')
        
        SaveFeatureEngageNet(model,
                        device,
                        Dir_save,
                        Dir_train,
                        train_csvfile,
                        Dir_val, 
                        val_csvfile,
                        nb_frames_video, 
                        Train_flag=args.train_flag,)
        
    elif (args.dataset_emotion=='dfew'):
        Dir_save = os.path.join(str(args.dir_data_parent),'DFEW/ViT_Feature')
        Dir_train = os.path.join(str(args.dir_data_parent),'DFEW/dfew.h5')
        Dir_val = os.path.join(str(args.dir_data_parent),'DFEW/dfew.h5')
        train_csvfile = os.path.join(str(args.dir_data_parent),'DFEW/train_label/set_1.csv')
        val_csvfile = os.path.join(str(args.dir_data_parent),'DFEW/test_label/set_1.csv')
        
        SaveFeatureDFEW(model,
                        device,
                        Dir_save,
                        Dir_train,
                        train_csvfile,
                        Dir_val, 
                        val_csvfile,
                        nb_frames_video, 
                        Train_flag=args.train_flag,)
        
        
    else:
        raise ValueError("Dataset name does not exist")
        

if __name__ == "__main__":
    main()
