import torch
import torch.nn as nn
import glob

from Encoders import *

# Decoder Emotion VGAF
class AttentionPooling(nn.Module):
    def __init__(self, d_model, device):
        super(AttentionPooling, self).__init__()
        
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
    
    

class VideoVitNet(nn.Module):
    def __init__(self, device, nb_class=3):
        super(VideoVitNet, self).__init__()        

        self.vit = timm.create_model('vit_large_patch14_224_clip_laion2b', pretrained=True).to(device)
        for param in self.vit.parameters(): # Not compute gradient
            param.requires_grad = False
        self.vit.head = nn.Identity()
        self.att_pooling = AttentionPooling(1024, device).to(device)
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

class VE_MultiDecoder(nn.Module):
    def __init__(
        self,
        device,
        encoder_name,
        two_encoders,
        channel_factor,
        blocks_res,
        latent_dim,
        dropout_classif,
        cls_counts: dict,
        # ablation flags
        use_emotion,
        decoder_sktname, 
        use_person_pose,
        use_face_pose,
        add_keypoints,
        pretrained_vit_path, 
        classif_projection,
        proj_dim,
        heatmap_dim=(56, 56),
        # per-dataset class counts
        num_limbs_person=18,
        num_limbs_face=83,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.two_encoders = two_encoders
        self.use_emotion = use_emotion
        self.use_person = use_person_pose
        self.use_face = use_face_pose
        self.latent_dim = latent_dim
        self.num_limbs_person = num_limbs_person
        self.num_limbs_face = num_limbs_face
        self.add_keypoints = add_keypoints
        self.device = device
        self.heatmap_dim = heatmap_dim

        # 1) Dynamic encoders
        encs = 2 if two_encoders else 1
        self.encoders = nn.ModuleList([
            self._make_encoder(encoder_name, device, channel_factor, blocks_res, latent_dim)
            for _ in range(encs)])

        if self.use_emotion:
            #vit 
            for ds, nclass in cls_counts.items():
                vit_model = VideoVitNet(device, nclass)#.to(device)
                self.vit_model = nn.DataParallel(vit_model)
                
                # Load saved video model
                video_saved = glob.glob(pretrained_vit_path +'*.tar')
                if len(video_saved)>0:
                    model_file_video = video_saved[0]
                    checkpoint = torch.load(model_file_video)
                    self.vit_model.load_state_dict(checkpoint['model_state_dict'])
                    print('Pretrained vit from', model_file_video)
                else:
                    print('No saved video model found for ViT')
                
                # Create a new state dict to remove 'module' prefic from keys of state_dict' dataparallel
                new_state_dict_video = {}
                for key , value in self.vit_model.state_dict().items():
                    new_key = key.replace('module.','') # remove 'module' prefix
                    new_state_dict_video[new_key] = value
                # Recall the model 
                self.vit_model = VideoVitNet(device, nclass).to(device)
                self.vit_model.load_state_dict(new_state_dict_video)
                        
                self.vit_model.att_pooling=nn.Identity()
                self.vit_model.linear_out = nn.Identity()
                for param in self.vit_model.parameters(): 
                    param.requires_grad = False
                
            # 2) ViT projection heads
            ds_names = [d for d in cls_counts.keys()]
            self.vit_heads = nn.ModuleDict({
                ds: nn.Linear(1024, latent_dim * 7 * 7).to(device)
                for ds in ds_names
            })

            # 4) Per-dataset emotion heads
            self.decoder_emotions = nn.ModuleDict()
            for ds, nclass in cls_counts.items():
                if classif_projection:
                    self.decoder_emotions[ds] = Classification_proj(
                                                device,
                                                latent_dim,
                                                dropout_classif,
                                                nclass,
                                                two_encoders,
                                                use_person_pose,
                                                use_face_pose,
                                                add_keypoints,
                                                heatmap_dim,
                                                proj_dim,
                                                pos_enc_type='sinusoidal'
                                            ).to(device)
                else:
                    self.decoder_emotions[ds] = Classification(
                                                device,
                                                latent_dim,
                                                dropout_classif,
                                                nclass,
                                                two_encoders,
                                                use_person_pose,
                                                use_face_pose,
                                                add_keypoints,
                                                heatmap_dim,
                                                pos_enc_type='sinusoidal'
                                            ).to(device)

        # 3) Skeleton / landmark decoders (conditionally)
        if self.use_person:
            self.decoder_skeleton = self._make_decoder_skeleton(decoder_sktname, latent_dim, num_limbs_person)
                                   
        if self.use_face:
            self.decoder_landmark = self._make_decoder_skeleton(decoder_sktname, latent_dim, num_limbs_face)
                              

        
    def _make_decoder_skeleton(self, name, latent_dim,  out_channels):
        if name =='openpose':
            return DecoderOpenPose(latent_dim, out_channels, num_stages=6).to(self.device) 
        raise ValueError(f"Unsupported Skt Decoder: {name}")

    def _make_encoder(self, name, device, channel_factor, blocks_res, latent_dim):
        if name == 'residual':
            return EncoderResidual(channel_factor, latent_dim, blocks_res).to(device)
        if name.startswith('resnet'):
            return EncoderResnet(latent_dim, resnet_name=name).to(device)
        if name == 'vitL':
            return EncoderViTLarge(latent_dim).to(device)
        if name == 'vitB':
            return EncoderViTBase(latent_dim).to(device)
        raise ValueError(f"Unsupported encoder: {name}")

    def forward(self, inputs, **featvits):
        bs, fr, c, h, w = inputs.shape
        imgs = inputs.view(bs * fr, c, h, w)

        # 1) Encode
        zs = [enc(imgs) for enc in self.encoders]

        # 2) Vit if emotion is used
        if self.use_emotion:
            dataset = featvits.pop('dataset', 'vgaf')
            ##vit_feat = featvits.get(dataset, None) # we use the vit branch directly
            vit_feat = self.vit_model(inputs) # inputs
            if vit_feat is not None and dataset in self.vit_heads:
                z_vit = self.vit_heads[dataset](vit_feat).view(-1, self.latent_dim, 7, 7)
                zs.append(z_vit)

        # 3) Concat & split dynamically
        z = torch.cat(zs, dim=1)
        parts = torch.split(z, self.latent_dim, dim=1)
        if len(parts) == 1:
            z1 = z2 = parts[0]
            z3 = None
        elif len(parts) == 2:
            z1, z2 = parts
            z3 = None
        else:
            z1, z2, z3 = parts[0], parts[1], parts[2]

        # 4) Skeleton & landmark decode
        sp_pred = None
        sf_pred = None

        if self.use_person:
            sp_pred = self.decoder_skeleton(z1)

        if self.use_face:
            # always give the face‐decoder *something* to decode
            sf_input = z2 if self.two_encoders else z1
            sf_pred = self.decoder_landmark(sf_input)

        

       # 7) Emotion classification
        if self.use_emotion:
            # 6) Flatten for classification
            flat = lambda x: x.view(bs, fr, -1) if x is not None else None
            fz1, fz2, fz3 = flat(z1), flat(z2), flat(z3)
            if isinstance(sp_pred, list):
                fsp = flat(torch.sigmoid(sp_pred[-1]).sum(1))
            elif sp_pred is None:
                fsp = flat(sp_pred)
            else:
                fsp=flat(sp_pred.sum(1)) 

            if isinstance(sf_pred, list):
                fsf = flat(torch.sigmoid(sf_pred[-1]).sum(1))
            elif sf_pred is None:
                fsf = flat(sf_pred) 
            else:
                fsf=flat(sf_pred.sum(1))       

            head = self.decoder_emotions[dataset]
            class_out = head(
                fz1, fz2, fz3,
                kp_pose=fsp if self.use_person else None,
                kp_face=fsf if self.use_face else None
            )
            return {
                'preds_emotion':        class_out,
                'skeleton_person_pred': sp_pred,
                'skeleton_face_pred':   sf_pred,
                'z':                     z,
            }
        else:
            return {
                'skeleton_person_pred': sp_pred,
                'skeleton_face_pred':   sf_pred,
                'z':                     z,
            }


from positional_encodings import *


class Classification_proj(nn.Module):
    def __init__(
        self,
        device,
        latent_dim,
        dropout,
        nb_class,
        two_encoders,
        use_person_pose,
        use_face_pose,
        add_keypoints,
        heatmap_dim,
        proj_dim,
        pos_enc_type="sinusoidal",  # NEW
    ):
        super().__init__()
        self.device = device
        self.two_encoders = two_encoders
        self.add_keypoints = add_keypoints
        self.use_person_pose = use_person_pose
        self.use_face_pose = use_face_pose
        self.heatmap_dim = heatmap_dim
        self.pos_enc_type = pos_enc_type  # save for debugging
        
        # 3) Normalize+project keypoints
        if self.add_keypoints:
            self.kp_face_proj = nn.Sequential(
                nn.LayerNorm(heatmap_dim[0]*heatmap_dim[1]),
                nn.Linear(heatmap_dim[0]*heatmap_dim[1], proj_dim),
                nn.ReLU(),
            ).to(device)
            self.kp_pose_proj = nn.Sequential(
                nn.LayerNorm(heatmap_dim[0]*heatmap_dim[1]),
                nn.Linear(heatmap_dim[0]*heatmap_dim[1], proj_dim),
                nn.ReLU(),
            ).to(device)
    
        #if add_keypoints:
            if self.use_face_pose and not self.use_person_pose:
                self.last_dim = latent_dim + proj_dim 
            elif not self.use_face_pose and self.use_person_pose:
                self.last_dim =latent_dim + proj_dim  
            else:
                self.last_dim =latent_dim + 2*proj_dim  
        else:
            self.last_dim = latent_dim 

        factor = 3 if two_encoders else 2
        self.linear_reduction = nn.Linear(factor * latent_dim * 7 * 7, latent_dim).to(device)

        # Add positional encoding module
        self._init_positional_encoding(pos_enc_type, max_len=500)  # assuming max 500 frames

        # Transformer + pooling
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.last_dim, nhead=1, 
                                                   batch_first=True,
                                                   ).to(device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1).to(device)
        self.att_pool = AttentionPooling(self.last_dim, device).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(self.last_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, nb_class),
        ).to(device)

    def _init_positional_encoding(self, pos_enc_type, max_len):
        if pos_enc_type == "learnable":
            self.pos_enc = LearnablePositionalEncoding(seq_len=max_len, dim=self.last_dim).to(self.device)
        elif pos_enc_type == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(seq_len=max_len, dim=self.last_dim).to(self.device)
        elif pos_enc_type == "relative":
            self.pos_enc = RelativePositionalEncoding(max_len=max_len, dim=self.last_dim).to(self.device)
        elif pos_enc_type is None:
            self.pos_enc = None # no pos enc.
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_enc_type}")
    
    def forward(self, z1, z2, z3=None, kp_pose=None, kp_face=None):
        # z*: (bs, fr, latent_dim)
        if self.two_encoders:
            concat_z = torch.cat((z1, z2, z3), dim=2)
        else:
            concat_z = torch.cat((z1, z2), dim=2)
        reduced = self.linear_reduction(concat_z)

        if self.add_keypoints:
            parts = [reduced]
            if self.use_person_pose: parts.append(self.kp_pose_proj(kp_pose))
            if self.use_face_pose:   parts.append(self.kp_face_proj(kp_face))
            reduced = torch.cat(parts, dim=2)  # (B,T,last_dim)
        if self.pos_enc_type is not None:
            reduced = self.pos_enc(reduced)    
        emb = self.transformer(reduced)        # (B,T,C)
        emb = self.att_pool(emb)               # (B,C)
        logits = self.classifier(emb)          # (B,nb_class)
        return logits
    
class Classification(nn.Module):
    def __init__(
        self,
        device,
        latent_dim,
        dropout,
        nb_class,
        two_encoders,
        use_person_pose,
        use_face_pose,
        add_keypoints,
        heatmap_dim,
        pos_enc_type="sinusoidal",  # NEW
    ):
        super().__init__()
        self.device = device
        self.two_encoders = two_encoders
        self.add_keypoints = add_keypoints
        self.use_person_pose = use_person_pose
        self.use_face_pose = use_face_pose
        self.heatmap_dim = heatmap_dim
        self.pos_enc_type = pos_enc_type  # save for debugging
    
        if add_keypoints:
            if self.use_face_pose and not self.use_person_pose:
                self.last_dim = latent_dim + heatmap_dim[0]*heatmap_dim[1]
            elif not self.use_face_pose and self.use_person_pose:
                self.last_dim = latent_dim + heatmap_dim[0]*heatmap_dim[1]
            else:
                self.last_dim = latent_dim + (heatmap_dim[0]*heatmap_dim[1])*2
        else:
            self.last_dim = latent_dim 

        factor = 3 if two_encoders else 2
        self.linear_reduction = nn.Linear(factor * latent_dim * 7 * 7, latent_dim).to(device)

        # Add positional encoding module
        self._init_positional_encoding(pos_enc_type, max_len=1000)  # assuming max 500 frames

        # Transformer + pooling
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.last_dim, nhead=1).to(device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1).to(device)
        self.att_pool = AttentionPooling(self.last_dim, device).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(self.last_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, nb_class),
        ).to(device)

    def _init_positional_encoding(self, pos_enc_type, max_len):
        if pos_enc_type == "learnable":
            self.pos_enc = LearnablePositionalEncoding(seq_len=max_len, dim=self.last_dim).to(self.device)
        elif pos_enc_type == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(seq_len=max_len, dim=self.last_dim).to(self.device)
        elif pos_enc_type == "relative":
            self.pos_enc = RelativePositionalEncoding(max_len=max_len, dim=self.last_dim).to(self.device)
        elif pos_enc_type is None:
            self.pos_enc = None # no pos enc.
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_enc_type}")
    
    def forward(self, z1, z2, z3=None, kp_pose=None, kp_face=None):
        # z*: (bs, fr, latent_dim)
        if self.two_encoders:
            concat_z = torch.cat((z1, z2, z3), dim=2)
        else:
            concat_z = torch.cat((z1, z2), dim=2)
        reduced = self.linear_reduction(concat_z)

        if self.add_keypoints:
            if self.use_face_pose and not self.use_person_pose:
                reduced = torch.cat((reduced, kp_face), dim=2)
            elif not self.use_face_pose and self.use_person_pose:
                reduced = torch.cat((reduced, kp_pose), dim=2)
            else:
                reduced = torch.cat((reduced, kp_pose, kp_face), dim=2)

        # Inject positional encoding
        if self.pos_enc_type != None:
            reduced = self.pos_enc(reduced)
            
        emb = self.transformer(reduced)
        emb = self.att_pool(emb)
        out = self.classifier(emb)
        return out



