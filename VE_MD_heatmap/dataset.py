import numpy as np 
import os 
from PIL import Image
import pandas as pd 
import json
import h5py

import torch 
from torch.utils.data import Dataset
from torchvision import transforms


class GAFDataset(Dataset):
    def __init__(self, data_folder, data_folder_kp, data_folder_vit, csv_file, transform=None):
        with open(os.path.join(csv_file), 'r') as file:
            self.annotations = json.load(file)
        self.data_folder = data_folder
        self.data_folder_kp = data_folder_kp
        self.data_folder_vit = data_folder_vit

        self.transform = transform
       
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        label_emotion = annotation['label_emotion']
        file_name =  annotation['file_name']
         # Load and process the image
        image_context = Image.open(os.path.join(self.data_folder, annotation['file_name']))
        if image_context.mode != 'RGB':
            image_context = image_context.convert('RGB')
        original_size = image_context.size
        new_size = (224, 224)
        image_context = image_context.resize(new_size, Image.LANCZOS)
        if self.transform:
            image_context = self.transform(image_context)
        #  #load heatmap
        sekeleton_person_gt = torch.load(os.path.join( self.data_folder_kp, str(file_name)+'_person_limb.pt'))
        sekeleton_face_gt = torch.load(os.path.join( self.data_folder_kp, str(file_name)+'_face_limb.pt'))
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(file_name.split('.')[0])+'.vit_feat.pt'))
       
        # Prepare the data to return
        data_return = {}
        data_return['images'] = image_context.unsqueeze(0)
        data_return['label_emotion'] = label_emotion
        data_return['limb_person'] = (sekeleton_person_gt).unsqueeze(0)
        data_return['limb_face'] = (sekeleton_face_gt).unsqueeze(0) 
        data_return['feat_vit'] = feat_vit
        return data_return


class VGAFVideoDataset(Dataset):
    def __init__(self, data_folder, data_folder_vit, csv_file, transform=None,):
        with open(os.path.join(csv_file), 'r') as file:
            self.annotations = json.load(file)
        self.data_folder = data_folder
        self.data_folder_vit = data_folder_vit
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        file_name = annotation['file_name']
        label_emotion = annotation['label_emotion']-1 # from 0, 1, 2
        #load images
        images_tensor = torch.load(os.path.join(self.data_folder, str(file_name)+'.pt'))
        #load heatmap
        sekeleton_person_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_person_limb.pt'))
        sekeleton_face_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_face_limb.pt'))
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(file_name.split('.')[0])+'.vit_feat.pt'))
        data_return = {}
        data_return['images'] = images_tensor # (nb_frames, 3, 224, 224)
        data_return['label_emotion'] = label_emotion
        data_return['limb_person'] = sekeleton_person_gt
        data_return['limb_face'] = sekeleton_face_gt
        data_return['feat_vit'] = feat_vit
        return data_return
    
   
class EngageNetVideoDataset(Dataset):
    def __init__(self, data_folder,data_folder_vit, csv_file, transform=None,):
        with open(os.path.join(csv_file), 'r') as file:
            self.annotations = json.load(file)
        self.data_folder = data_folder
        self.data_folder_vit = data_folder_vit
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        file_name = annotation['file_name']
        label_emotion = annotation['label_emotion']
        #load images
        images_tensor = torch.load(os.path.join(self.data_folder, str(file_name)+'.pt'))
        #load heatmap
        sekeleton_person_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_person_limb.pt'))
        sekeleton_face_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_face_limb.pt'))
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(file_name)+'.vit_feat.pt'))
       
        data_return = {}
        data_return['images'] = images_tensor # (nb_frames, 3, 224, 224)
        data_return['label_emotion'] = label_emotion
        data_return['limb_person'] =sekeleton_person_gt
        data_return['limb_face'] =sekeleton_face_gt
        data_return['feat_vit'] = feat_vit

        return data_return

    
class MER2023VideoDataset(Dataset):
    def __init__(self, data_folder,data_folder_vit,csv_file, transform=None,nb_frames=16):
        with open(os.path.join(csv_file), 'r') as file:
            self.annotations = json.load(file)
        self.data_folder = data_folder
        self.data_folder_vit= data_folder_vit
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]    
        #file_name = annotation['file_name']
        file_name = annotation['file_name'].split('.')[0]+'.img'
        label_emotion = annotation['label_emotion'] # from 0, 1, 2,3,4
        
        #load images
        images_tensor = torch.load(os.path.join(self.data_folder, str(file_name)+'.pt'))
        sekeleton_person_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_person_limb.pt'))
        sekeleton_face_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_face_limb.pt'))
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(file_name.split('.')[0])+'.vit_feat.pt'))
       
        data_return = {}
        data_return['images'] = images_tensor # (nb_frames, 3, 224, 224)
        data_return['label_emotion'] = label_emotion
        data_return['limb_person'] =sekeleton_person_gt
        data_return['limb_face'] =sekeleton_face_gt
        data_return['feat_vit'] = feat_vit

        return data_return
    
    
class SamSemoVideoDataset(Dataset):
    def __init__(self, data_folder, data_folder_vit, csv_file, transform=None,nb_frames=10):
        with open(os.path.join(csv_file), 'r') as file:
            self.annotations = json.load(file)
        self.data_folder = data_folder
        self.data_folder_vit = data_folder_vit
        self.transform = transform
        self.nb_frames = nb_frames

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]    
        file_name = annotation['file_name']
        label_emotion = annotation['label_emotion'] 
        #load images
        images_tensor = torch.load(os.path.join(self.data_folder, str(file_name)+'.pt'))
        sekeleton_face_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_face_limb.pt'))
        sekeleton_person_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_person_limb.pt'))
          # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(file_name)+'.vit_feat.pt'))
       
        data_return = {}
        data_return['images'] = images_tensor # (nb_frames, 3, 224, 224)
        data_return['label_emotion'] = label_emotion
        data_return['limb_person'] =sekeleton_person_gt
        data_return['limb_face'] =sekeleton_face_gt
        data_return['feat_vit'] = feat_vit

        return data_return
    
   
class DFEWVideoDataset(Dataset):
    def __init__(self, data_folder,data_folder_vit,  csv_file, transform=None):
        with open(os.path.join(csv_file), 'r') as file:
            self.annotations = json.load(file)
        self.data_folder = data_folder
        self.transform = transform
        self.data_folder_vit = data_folder_vit

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]    
        file_name = annotation['file_name']
        label_emotion = annotation['label_emotion'] 
        #load images
        images_tensor = torch.load(os.path.join(self.data_folder, str(file_name)+'.pt'))
        sekeleton_face_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_face_limb.pt'))
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(file_name)+'.vit_feat.pt'))        # Stack frames into a single tensor
        
        data_return = {}
        data_return['images'] = images_tensor # (nb_frames, 3, 224, 224)
        data_return['label_emotion'] = label_emotion
        data_return['limb_face'] =sekeleton_face_gt
        data_return['feat_vit'] = feat_vit

        return data_return

    
    
class MER2023Semi(Dataset):
    def __init__(self, data_folder,csv_file, transform=None):
        with open(os.path.join(csv_file), 'r') as file:
            self.annotations = json.load(file)
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]    
        file_name = annotation['file_name']
        
        #load images
        images_tensor = torch.load(os.path.join(self.data_folder, str(file_name)+'.pt'))
        sekeleton_person_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_person_limb.pt'))
        adjacency_person_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_person_adj.pt'))
        sekeleton_face_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_face_limb.pt'))
        adjacency_face_gt = torch.load(os.path.join( self.data_folder, str(file_name)+'_face_adj.pt'))
       
        data_return = {}
        data_return['images'] = images_tensor # (nb_frames, 3, 224, 224)
        data_return['limb_person'] =sekeleton_person_gt
        data_return['adjacency_person'] = adjacency_person_gt
        data_return['limb_face'] =sekeleton_face_gt
        data_return['adjacency_face'] = adjacency_face_gt

        return data_return

    
############################################
###########################################
# For Test and Validation Emotion dataset loader
###########################################
###########################################


class GAFDatasetTest(Dataset):
    def __init__(self, data_folder,data_folder_vit,  csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file, engine='python', sep=',')
        self.data_folder = data_folder
        self.data_folder_vit = data_folder_vit
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_emotion = self.annotations['Labels'][index]
        file_name = self.annotations['Img_name'][index]
        # Load and process the image
        image_context = Image.open(os.path.join(self.data_folder, self.annotations['Img_name'][index]))
        if image_context.mode != 'RGB':
            image_context = image_context.convert('RGB')

        image_context = image_context.resize((224, 224), Image.LANCZOS)

        if self.transform:
            image_context = self.transform(image_context)
            
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(file_name.split('.')[0])+'.vit_feat.pt'))

        # Prepare the data to return
        data_return = {}
        data_return['images'] = image_context.unsqueeze(0)
        data_return['labels'] = label_emotion
        data_return['feat_vit'] = feat_vit
        return data_return   


class MyDatasetVGAF(Dataset):
    def __init__(self, hdf5_file,data_folder_vit, csv_file, img_transform=None):
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(self.hdf5_file, 'r')
        self.file_labels = pd.read_csv(csv_file, engine='python', sep=' ')
        self.labels = self.file_labels['Label'] - 1  # Adjust labels to start from 0
        self.img_transform = img_transform
        self.data_folder_vit = data_folder_vit

    def __getitem__(self, index):
        # Retrieve video name and corresponding frames from HDF5 file
        video_name = self.file_labels['Vid_name'][index]
        frames = self.hf[video_name+'.mp4.img'][:]
        
        # Apply transformation to each frame
        if self.img_transform:
            frames = [self.img_transform(Image.fromarray(frame)) for frame in frames]
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(video_name)+'.vit_feat.pt'))        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames)
        label = self.labels[index]

        data_return = {}
        data_return['images'] = frames_tensor
        data_return['labels'] = label
        data_return['feat_vit'] = feat_vit
        return data_return
    
    def __len__(self):
        return len(self.labels)

    def close(self):
        self.hf.close()

   
    
class EngageDataset(Dataset):
    def __init__(self, hdf5_file,data_folder_vit,  csv_file, img_transform=None):
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(hdf5_file, 'r')
        self.file_labels = pd.read_csv(csv_file, engine='python', sep=',')
        self.img_transform = img_transform
        self.data_folder_vit = data_folder_vit

    def __getitem__(self, index):
        # Read the label
        label = self.file_labels['label'][index]

        # Get the video chunk name from the CSV file
        chunk_name = (self.file_labels['chunk']+'.img')[index]

        # Load frames from the HDF5 file
        frames = self.hf[chunk_name][:]
        
        # Apply transformation to each frame
        transformed_frames = []
        for frame in frames:
            frame = Image.fromarray(frame)
            if self.img_transform:
                frame = self.img_transform(frame)
            transformed_frames.append(frame)
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(chunk_name)+'.vit_feat.pt'))        # Stack frames into a single tensor

        # Stack the images into a single tensor
        data_return = {}
        data_return['images'] = torch.stack(transformed_frames)
        data_return['labels'] = int(label)
        data_return['feat_vit'] = feat_vit
        return data_return

    def __len__(self):
        return len(self.file_labels)

    def close(self):
        self.hf.close()

    
class MER23Dataset(Dataset):
    def __init__(self, hdf5_file, data_folder_vit, csv_file, img_transform=None):
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(hdf5_file, 'r')
        self.img_transform = img_transform
        self.data_folder_vit = data_folder_vit
        self.df_csv = pd.read_csv(csv_file)
        self.labels_dict = {'worried': 0, 'happy': 1, 'neutral': 2, 'sad': 3, 'angry': 4, 'surprise': 5}

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, idx):
        # Get the video name from the CSV file
        video_name = self.df_csv['name'][idx]
        label_str = self.df_csv['discrete'][idx]
        label = self.labels_dict[label_str]

        # Retrieve frames from the HDF5 file
        frames = self.hf[video_name+'.img'][:]
        
        # Apply transformation to each frame
        if self.img_transform:
            frames = [self.img_transform(Image.fromarray(frame)) for frame in frames]
            
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(video_name)+'.vit_feat.pt'))        # Stack frames into a single tensor

        # Stack frames into a single tensor
        frames = torch.stack(frames)
        data_return = {}
        data_return['images'] = frames
        data_return['labels'] = label
        data_return['feat_vit'] = feat_vit
        return data_return
    
    def close(self):
        self.hf.close()

   
class DFEWDataset(Dataset):
    def __init__(self, hdf5_file,data_folder_vit, csv_file, img_transform=None, zero_pad_length=5):
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(hdf5_file, 'r')
        self.df_csv = pd.read_csv(csv_file)
        self.img_transform = img_transform
        self.zero_pad_length = zero_pad_length
        self.data_folder_vit = data_folder_vit

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, index):
        # Get the label
        label = self.df_csv['label'][index] - 1  # Adjusting label to start from 0: 7 classes

        # Convert the video_name from CSV to zero-padded format
        video_name = str(self.df_csv['video_name'][index]).zfill(self.zero_pad_length)

        # Load and transform images from HDF5 file
        frames = self.hf[video_name][:]
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(video_name)+'.vit_feat.pt'))   
        
        # Apply transformation to each frame
        transformed_frames = []
        for frame in frames:
            frame = Image.fromarray(frame)
            if self.img_transform:
                frame = self.img_transform(frame)
            transformed_frames.append(frame)
        # Stack the images into a single tensor
        data_return = {}
        data_return['images'] = torch.stack(transformed_frames)
        data_return['labels'] = int(label)
        data_return['feat_vit'] = feat_vit
        
        return data_return

    def close(self):
        self.hf.close()



class SamSemoDataset(Dataset):
    def __init__(self, hdf5_file, data_folder_vit, csv_file, img_transform=None):
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(self.hdf5_file, 'r')
        self.file_labels = pd.read_csv(csv_file, engine='python', sep='\t')
        # Mapping of simple emotions; adjust for compound emotions if necessary
        self.labels_dict = {'happiness': 0, 'neutral': 1, 'sadness': 2, 'disgust': 3, 'anger': 3, 'surprise': 4}
        self.img_transform = img_transform
        self.data_folder_vit = data_folder_vit
        
    def __len__(self):
        return len(self.file_labels)

    def __getitem__(self, index):
        # Retrieve video name and corresponding frames from HDF5 file
        video_name = self.file_labels['utterance_id'][index]
        frames = self.hf[video_name][:]
        
        # Handle missing or unexpected labels
        label_str = self.file_labels['aggregated_emotions'][index]
        if label_str in self.labels_dict:
            label = self.labels_dict[label_str]
        else:
            raise ValueError(f"Unexpected label: {label_str}")
        
        # Apply transformations to each frame (resize and transformations)
        if self.img_transform:
            frames = [self.img_transform((Image.fromarray(frame)).resize((224, 224), Image.LANCZOS)) for frame in frames]
        # load vit feat
        feat_vit = torch.load(os.path.join( self.data_folder_vit, str(video_name)+'.vit_feat.pt'))        # Stack frames into a single tensor

        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames)
        
        # Returning a dictionary with images and labels
        data_return = {}
        data_return['images'] = frames_tensor
        data_return['labels'] = label
        data_return['feat_vit'] = feat_vit
        return data_return

    def close(self):
        self.hf.close()