    
import torch
import torch.nn as nn

from Encoders import *
from st_gcn import *


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

class VE_MultiDecoder(nn.Module):
    def __init__(
        self,
        device,
        encoder_name,
        two_encoders,
        channel_factor,
        blocks_res,
        latent_dim,
        nun_heads_petr,
        num_encoder_layers_petr,
        num_decoder_layers_petr,
        dropout_classif,
        stgcn_active,
        num_queries,
        cls_counts: dict,
        # ablation flags
        use_person_pose,
        use_face_pose,
        add_keypoints,
        pos_enc_type,
        classif_projection,
        proj_dim,
        # per-dataset class counts
        num_limbs_person=18,
        num_limbs_face=20,
    ):
        super().__init__()
        self.two_encoders = two_encoders
        self.use_person = use_person_pose
        self.use_face = use_face_pose
        self.latent_dim = latent_dim
        self.num_queries = num_queries
        self.num_limbs_person = num_limbs_person
        self.num_limbs_face = num_limbs_face
        self.add_keypoints = add_keypoints

        # 1) Dynamic encoders
        encs = 2 if two_encoders else 1
        self.encoders = nn.ModuleList([
            self._make_encoder(encoder_name, device, channel_factor, blocks_res, latent_dim)
            for _ in range(encs)
        ])

        # 2) ViT projection heads
        ds_names = [d for d in cls_counts.keys() if d != 'coco']
        self.vit_heads = nn.ModuleDict({
            ds: nn.Linear(1024, latent_dim * 7 * 7).to(device)
            for ds in ds_names
        })

        # 3) Skeleton / landmark decoders (conditionally)
        if self.use_person:
            self.decoder_skeleton = SkeletonDETR(
                                    num_queries,
                                    num_limbs_person,
                                    latent_dim,
                                    nun_heads_petr,
                                    num_encoder_layers_petr,
                                    num_decoder_layers_petr,
                                    pe_fix=True,
                                    ).to(device)
        if self.use_face:
            self.decoder_landmark = SkeletonDETR(
                                    num_queries, 
                                    num_limbs_face,
                                    latent_dim,
                                    nun_heads_petr,
                                    num_encoder_layers_petr,
                                    num_decoder_layers_petr,
                                    pe_fix=True,
                                ).to(device)

        if stgcn_active:
            if self.use_person:
                self.stgcn_person = STGCN(in_channels=4).to(device)
            if self.use_face:
                self.stgcn_face = STGCN(in_channels=4).to(device)

        # 4) Per-dataset emotion heads
        self.decoder_emotions = nn.ModuleDict()
        for ds, nclass in cls_counts.items():
            if ds == 'mersemi':
                continue
            if classif_projection:
                self.decoder_emotions[ds] = Classification_proj(
                                            device,
                                            latent_dim,
                                            dropout_classif,
                                            num_queries,
                                            nclass,
                                            two_encoders,
                                            use_person_pose,
                                            use_face_pose,
                                            add_keypoints,
                                            proj_dim,
                                            pos_enc_type,
                                        ).to(device)
            else:
                self.decoder_emotions[ds] = Classification(
                                            device,
                                            latent_dim,
                                            dropout_classif,
                                            num_queries,
                                            nclass,
                                            two_encoders,
                                            use_person_pose,
                                            use_face_pose,
                                            add_keypoints,
                                            pos_enc_type,
                                        ).to(device)

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

        # 2) Vit if provided
        dataset = featvits.pop('dataset', 'vgaf')
        vit_feat = featvits.get(dataset, None)
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
        sp_pred, ap_pred = None, None
        sf_pred, af_pred = None, None

        if self.use_person:
            sp_pred, ap_pred = self.decoder_skeleton(z1)

        if self.use_face:
            # always give the face‐decoder *something* to decode
            sf_input = z2 if self.two_encoders else z1
            sf_pred, af_pred = self.decoder_landmark(sf_input)

        # 5) Apply ST-GCN if available
        sp = sp_pred
        sf = sf_pred
        if hasattr(self, 'stgcn_person') and sp_pred is not None:
            sp = self._apply_stgcn(sp_pred, ap_pred, self.stgcn_person, bs, fr, self.num_limbs_person)
        if hasattr(self, 'stgcn_face') and sf_pred is not None:
            sf = self._apply_stgcn(sf_pred, af_pred, self.stgcn_face, bs, fr, self.num_limbs_face)

        # 6) Flatten for classification
        flat = lambda x: x.view(bs, fr, -1) if x is not None else None
        fz1, fz2, fz3 = flat(z1), flat(z2), flat(z3)
        fsp, fsf = flat(sp), flat(sf)

        # 7) Emotion classification or return pose only
        if dataset in self.decoder_emotions:
            head = self.decoder_emotions[dataset]
            class_out = head(
                fz1, fz2, fz3,
                kp_pose=fsp if self.use_person else None,
                kp_face=fsf if self.use_face else None
            )
            return {
                'preds_emotion':        class_out,
                'skeleton_person_pred': sp_pred,
                'adjacency_person_pred': ap_pred,
                'skeleton_face_pred':   sf_pred,
                'adjacency_face_pred':  af_pred,
                'z':                     z,
            }
        else:
            return {
                'skeleton_person_pred': sp_pred,
                'adjacency_person_pred': ap_pred,
                'skeleton_face_pred':   sf_pred,
                'adjacency_face_pred':  af_pred,
                'z':                     z,
            }

    @staticmethod
    def _apply_stgcn(skel, adj, stgcn, bs, fr, num_limbs):
        sk = skel.view(bs, fr, -1, 4, num_limbs)
        adj_mat = adj.view(bs, fr, -1, num_limbs, num_limbs)
        return stgcn(sk, adj_mat).contiguous()



from positional_encodings import *


class Classification_proj(nn.Module):
    def __init__(
        self,
        device,
        latent_dim,
        dropout,
        num_queries,
        nb_class,
        two_encoders,
        use_person_pose,
        use_face_pose,
        add_keypoints,
        proj_dim,
        pos_enc_type="sinusoidal",  # NEW
    ):
        super().__init__()
        self.device = device
        self.two_encoders = two_encoders
        self.add_keypoints = add_keypoints
        self.use_person_pose = use_person_pose
        self.use_face_pose = use_face_pose
        self.pos_enc_type = pos_enc_type  # save for debugging
                

        # Keypoint dimensions
        if use_face_pose: nlf = 20
        else: nlf = 0
        if use_person_pose: nlp = 18
        else: nlp = 0
        
        if add_keypoints:
            kp_dim = 4 * (nlp) * num_queries
            face_dim = 4 * (nlf) * num_queries
            if use_face_pose and not use_person_pose:
                self.last_dim = latent_dim +  proj_dim
            elif not use_face_pose and use_person_pose:
                self.last_dim = latent_dim +  proj_dim
            else:
                self.last_dim = latent_dim +  proj_dim +  proj_dim
            
            self.kp_face_proj = nn.Sequential(
                nn.LayerNorm(int(face_dim), eps=1e-5), 
                nn.Linear(face_dim,  proj_dim),
                nn.ReLU(),
            ).to(device)
            self.kp_pose_proj = nn.Sequential(
                nn.LayerNorm(int(kp_dim), eps=1e-5), 
                nn.Linear(kp_dim,  proj_dim),
                nn.ReLU(),
            ).to(device)
        else:
            self.last_dim = latent_dim
            

        factor = 3 if two_encoders else 2
        self.linear_reduction = nn.Linear(factor * latent_dim * 7 * 7, latent_dim).to(device)

        # Add positional encoding module
        self._init_positional_encoding(pos_enc_type, max_len=5000)  # assuming max 500 frames

        # Transformer + pooling
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.last_dim, nhead=1, batch_first=True).to(device)
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
        elif pos_enc_type==None:
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
                kp_face = self.kp_face_proj(kp_face)
                reduced = torch.cat((reduced, kp_face), dim=2)
            elif not self.use_face_pose and self.use_person_pose:
                kp_pose = self.kp_pose_proj(kp_pose)
                reduced = torch.cat((reduced, kp_pose), dim=2)
            else:
                kp_face = self.kp_face_proj(kp_face)
                kp_pose = self.kp_pose_proj(kp_pose)
                reduced = torch.cat((reduced, kp_pose, kp_face), dim=2)

        # Inject positional encoding
        if self.pos_enc_type != None:
            reduced = self.pos_enc(reduced)
            
        emb = self.transformer(reduced)
        emb = self.att_pool(emb)
        out = self.classifier(emb)
        return out



class Classification(nn.Module):
    def __init__(
        self,
        device,
        latent_dim,
        dropout,
        num_queries,
        nb_class,
        two_encoders,
        use_person_pose,
        use_face_pose,
        add_keypoints,
        pos_enc_type="sinusoidal",  # NEW
    ):
        super().__init__()
        self.device = device
        self.two_encoders = two_encoders
        self.add_keypoints = add_keypoints
        self.use_person_pose = use_person_pose
        self.use_face_pose = use_face_pose
        self.pos_enc_type = pos_enc_type  # save for debugging

        # Keypoint dimensions
        if use_face_pose: nlf = 20
        else: nlf = 0
        if use_person_pose: nlp = 18
        else: nlp = 0
        if add_keypoints:
            kp_dim = 4 * (nlp + nlf) * num_queries
            last_dim = latent_dim + kp_dim
        else:
            last_dim = latent_dim

        self.last_dim = last_dim  # Save for positional enc init
        factor = 3 if two_encoders else 2
        self.linear_reduction = nn.Linear(factor * latent_dim * 7 * 7, latent_dim).to(device)

        # Add positional encoding module
        self._init_positional_encoding(pos_enc_type, max_len=500)  # assuming max 500 frames

        # Transformer + pooling
        encoder_layer = nn.TransformerEncoderLayer(d_model=last_dim, nhead=1, batch_first=True).to(device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1).to(device)
        self.att_pool = AttentionPooling(last_dim, device).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(last_dim, 512),
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
        elif pos_enc_type==None:
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
