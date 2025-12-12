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

 
def pad_collate_fn(batch):
    """
    Custom collate function to handle variable-sized skeletons (num_persons)
    and face landmarks (num_persons_face) in a sequence of frames. Instead of
    zero-padding, we replicate the last skeleton/adjacency for samples that
    have fewer persons than the max number of persons in the batch.

    Expected shapes per sample:
      - images:           (num_frames, C, H, W)
      - skeleton_person:  (num_frames, num_persons_body, body_dim)
      - adjacency_person: (num_frames, num_persons_body, num_body_limbs, num_body_limbs)
      - skeleton_face:    (num_frames, num_persons_face, face_dim)
      - adjacency_face:   (num_frames, num_persons_face, num_face_limbs, num_face_limbs)
      - label_emotion:    (scalar)
    """

    # Prepare lists to hold each field
    images_list = []
    vitfeat_list = []
    labels_emotion_list = []
    skeleton_body_list = []
    adjacency_body_list = []
    skeleton_face_list = []
    adjacency_face_list = []

    # 1) Find the maximum number of BODY persons across all samples in the batch
    max_num_persons_body = max(
        sample["skeleton_person"].shape[1] for sample in batch
    )

    # 2) Find the maximum number of FACE persons across all samples in the batch
    max_num_persons_face = max(
        sample["skeleton_face"].shape[1] for sample in batch
    )

    # 3) Process each sample in the batch
    for sample in batch:
        # --- Images (shape: [num_frames, C, H, W])
        images_list.append(sample["images"])
        vitfeat_list.append(sample["feat_vit"])
        labels_emotion_list.append(sample["label_emotion"])

        # ========== BODY SKELETON ==========
        skeleton_body = sample["skeleton_person"]
        num_frames_body, num_persons_body, body_dim = skeleton_body.shape

        # We will create a padded skeleton of shape:
        #   [num_frames_body, max_num_persons_body, body_dim].
        padded_skeleton_body = torch.zeros(
            (num_frames_body, max_num_persons_body, body_dim),
            dtype=skeleton_body.dtype
        )

        for f_idx in range(num_frames_body):
            frame_skeleton = skeleton_body[f_idx]  # shape: [num_persons_body, body_dim]
            n_persons_in_frame = frame_skeleton.shape[0]

            # Copy the real skeleton data
            padded_skeleton_body[f_idx, :n_persons_in_frame] = frame_skeleton

            # If there's room left, replicate the last person's skeleton
            if n_persons_in_frame < max_num_persons_body:
                last_person = frame_skeleton[-1].unsqueeze(0)  # shape: [1, body_dim]
                reps = max_num_persons_body - n_persons_in_frame
                padded_skeleton_body[f_idx, n_persons_in_frame:] = last_person.expand(
                    reps, body_dim
                )

        skeleton_body_list.append(padded_skeleton_body)

        # ========== BODY ADJACENCY ==========
        adjacency_body = sample["adjacency_person"]
        # shape is [num_frames, num_persons_body, num_body_limbs, num_body_limbs]
        _, _, num_body_limbs, _ = adjacency_body.shape

        padded_adjacency_body = torch.zeros(
            (num_frames_body, max_num_persons_body, num_body_limbs, num_body_limbs),
            dtype=adjacency_body.dtype
        )

        for f_idx in range(num_frames_body):
            frame_adjacency = adjacency_body[f_idx]  # shape: [num_persons_body, num_body_limbs, num_body_limbs]
            n_persons_in_frame = frame_adjacency.shape[0]

            padded_adjacency_body[f_idx, :n_persons_in_frame] = frame_adjacency

            if n_persons_in_frame < max_num_persons_body:
                last_adjacency = frame_adjacency[-1].unsqueeze(0)
                reps_adj = max_num_persons_body - n_persons_in_frame
                padded_adjacency_body[f_idx, n_persons_in_frame:] = last_adjacency.expand(
                    reps_adj, num_body_limbs, num_body_limbs
                )

        adjacency_body_list.append(padded_adjacency_body)

        # ========== FACE SKELETON ==========
        skeleton_face = sample["skeleton_face"]
        num_frames_face, num_persons_face, face_dim = skeleton_face.shape

        padded_skeleton_face = torch.zeros(
            (num_frames_face, max_num_persons_face, face_dim),
            dtype=skeleton_face.dtype
        )

        for f_idx in range(num_frames_face):
            frame_face = skeleton_face[f_idx]  # shape: [num_persons_face, face_dim]
            n_persons_face_in_frame = frame_face.shape[0]

            padded_skeleton_face[f_idx, :n_persons_face_in_frame] = frame_face

            if n_persons_face_in_frame < max_num_persons_face:
                last_face = frame_face[-1].unsqueeze(0)  # shape: [1, face_dim]
                reps_face = max_num_persons_face - n_persons_face_in_frame
                padded_skeleton_face[f_idx, n_persons_face_in_frame:] = last_face.expand(
                    reps_face, face_dim
                )

        skeleton_face_list.append(padded_skeleton_face)

        # ========== FACE ADJACENCY ==========
        adjacency_face = sample["adjacency_face"]
        # shape: [num_frames, num_persons_face, num_face_limbs, num_face_limbs]
        _, _, num_face_limbs, _ = adjacency_face.shape

        padded_adjacency_face = torch.zeros(
            (num_frames_face, max_num_persons_face, num_face_limbs, num_face_limbs),
            dtype=adjacency_face.dtype
        )

        for f_idx in range(num_frames_face):
            frame_adjacency_face = adjacency_face[f_idx]
            n_persons_face_in_frame = frame_adjacency_face.shape[0]

            padded_adjacency_face[f_idx, :n_persons_face_in_frame] = frame_adjacency_face

            if n_persons_face_in_frame < max_num_persons_face:
                last_face_adj = frame_adjacency_face[-1].unsqueeze(0)
                reps_face_adj = max_num_persons_face - n_persons_face_in_frame
                padded_adjacency_face[f_idx, n_persons_face_in_frame:] = last_face_adj.expand(
                    reps_face_adj, num_face_limbs, num_face_limbs
                )

        adjacency_face_list.append(padded_adjacency_face)

    # 4) Stack results into final batch tensors
    # images_list: [B, num_frames, C, H, W]
    images_batch = torch.stack(images_list, dim=0)
    vitfeat_batch = torch.stack(vitfeat_list, dim=0)

    # skeleton_body_list: [B, num_frames, max_num_persons_body, body_dim]
    skeleton_body_batch = torch.stack(skeleton_body_list, dim=0)

    # adjacency_body_list: [B, num_frames, max_num_persons_body, num_body_limbs, num_body_limbs]
    adjacency_body_batch = torch.stack(adjacency_body_list, dim=0)

    # skeleton_face_list: [B, num_frames, max_num_persons_face, face_dim]
    skeleton_face_batch = torch.stack(skeleton_face_list, dim=0)

    # adjacency_face_list: [B, num_frames, max_num_persons_face, num_face_limbs, num_face_limbs]
    adjacency_face_batch = torch.stack(adjacency_face_list, dim=0)

    label_emotion = torch.LongTensor(labels_emotion_list)

    # 5) Return your batched dictionary
    return {
        "images": images_batch,
        "skeleton_person": skeleton_body_batch,
        "adjacency_person": adjacency_body_batch,
        "skeleton_face": skeleton_face_batch,
        "adjacency_face": adjacency_face_batch,
        "label_emotion": label_emotion,
        "feat_vit": vitfeat_batch,
    }

def pad_collate_fn_face(batch):
    """
    Custom collate function to handle variable-sized skeletons (num_persons)
    in a sequence of frames. Instead of zero-padding, we replicate the last
    skeleton/adjacency for samples that have fewer persons than the max
    number of persons in the batch.

    Expected shapes per sample:
      - images:          (num_frames, C, H, W)
      - skeleton_person: (num_frames, num_persons, dim)
      - adjacency_person:(num_frames, num_persons, num_limbs, num_limbs)
      - label_emotion:   (scalar)
    """

    # Prepare lists to hold each field
    images_list = []
    labels_emotion_list = []
    skeleton_list = []
    adjacency_list = []

    # 1) Find the maximum number of persons among all samples in the batch
    #    We look at the shape [num_frames, num_persons, dim]
    #    So we take sample["skeleton_person"].shape[1] for the number of persons
    max_num_persons = max(
        sample["skeleton_face"].shape[1] for sample in batch
    )

    # 2) Process each sample in the batch
    for sample in batch:
        # --- Images (shape: [num_frames, C, H, W])
        images_list.append(sample["images"])
        labels_emotion_list.append(sample["label_emotion"])

        # --- Skeleton (shape: [num_frames, num_persons, dim])
        skeleton = sample["skeleton_face"]
        num_frames, num_persons, dim = skeleton.shape

        # We will create a padded skeleton of shape [num_frames, max_num_persons, dim].
        # For each frame, if there are fewer than max_num_persons, replicate the last one.
        padded_skeleton = torch.zeros(
            (num_frames, max_num_persons, dim),
            dtype=skeleton.dtype
        )

        for f_idx in range(num_frames):
            frame_skeleton = skeleton[f_idx]  # shape: [num_persons, dim]
            n_persons_in_frame = frame_skeleton.shape[0]

            # Copy the real skeleton data
            padded_skeleton[f_idx, :n_persons_in_frame] = frame_skeleton

            # If there's room left, replicate the last person's skeleton
            if n_persons_in_frame < max_num_persons:
                last_person = frame_skeleton[-1].unsqueeze(0)  # shape: [1, dim]
                reps = max_num_persons - n_persons_in_frame
                padded_skeleton[f_idx, n_persons_in_frame:] = last_person.expand(reps, dim)

        skeleton_list.append(padded_skeleton)

        # --- Adjacency (shape: [num_frames, num_persons, num_limbs, num_limbs])
        adjacency = sample["adjacency_face"]
        _, num_persons_adj, num_limbs, _ = adjacency.shape

        padded_adjacency = torch.zeros(
            (num_frames, max_num_persons, num_limbs, num_limbs),
            dtype=adjacency.dtype
        )

        for f_idx in range(num_frames):
            frame_adjacency = adjacency[f_idx]  # shape: [num_persons_adj, num_limbs, num_limbs]
            n_persons_in_frame = frame_adjacency.shape[0]

            padded_adjacency[f_idx, :n_persons_in_frame] = frame_adjacency

            if n_persons_in_frame < max_num_persons:
                last_adjacency = frame_adjacency[-1].unsqueeze(0)  # shape: [1, num_limbs, num_limbs]
                reps_adj = max_num_persons - n_persons_in_frame
                padded_adjacency[f_idx, n_persons_in_frame:] = last_adjacency.expand(
                    reps_adj, num_limbs, num_limbs
                )

        adjacency_list.append(padded_adjacency)

    # 3) Stack results into final batch tensors
    # images_list has len B, each is [num_frames, C, H, W]
    images_batch = torch.stack(images_list, dim=0)  
    # shape: [B, num_frames, C, H, W]

    label_emotion = torch.LongTensor(labels_emotion_list)

    # skeleton_list has len B, each is [num_frames, max_num_persons, dim]
    skeleton_batch = torch.stack(skeleton_list, dim=0)
    # shape: [B, num_frames, max_num_persons, dim]

    # adjacency_list has len B, each is [num_frames, max_num_persons, num_limbs, num_limbs]
    adjacency_batch = torch.stack(adjacency_list, dim=0)
    # shape: [B, num_frames, max_num_persons, num_limbs, num_limbs]

    # 4) Return your batched dictionary
    return {
        "images": images_batch,
        "skeleton_face": skeleton_batch,
        "adjacency_face": adjacency_batch,
        "label_emotion": label_emotion
    }

def pad_collate_fn_no_emotion(batch):
    """
    Custom collate function to handle variable-sized skeletons (num_persons)
    and face landmarks (num_persons_face) in a sequence of frames. Instead of
    zero-padding, we replicate the last skeleton/adjacency for samples that
    have fewer persons than the max number of persons in the batch.

    Expected shapes per sample:
      - images:           (num_frames, C, H, W)
      - skeleton_person:  (num_frames, num_persons_body, body_dim)
      - adjacency_person: (num_frames, num_persons_body, num_body_limbs, num_body_limbs)
      - skeleton_face:    (num_frames, num_persons_face, face_dim)
      - adjacency_face:   (num_frames, num_persons_face, num_face_limbs, num_face_limbs)
      - label_emotion:    (scalar)
    """

    # Prepare lists to hold each field
    images_list = []
    skeleton_body_list = []
    adjacency_body_list = []
    skeleton_face_list = []
    adjacency_face_list = []

    # 1) Find the maximum number of BODY persons across all samples in the batch
    max_num_persons_body = max(
        sample["skeleton_person"].shape[1] for sample in batch
    )

    # 2) Find the maximum number of FACE persons across all samples in the batch
    max_num_persons_face = max(
        sample["skeleton_face"].shape[1] for sample in batch
    )

    # 3) Process each sample in the batch
    for sample in batch:
        # --- Images (shape: [num_frames, C, H, W])
        images_list.append(sample["images"])

        # ========== BODY SKELETON ==========
        skeleton_body = sample["skeleton_person"]
        num_frames_body, num_persons_body, body_dim = skeleton_body.shape

        # We will create a padded skeleton of shape:
        #   [num_frames_body, max_num_persons_body, body_dim].
        padded_skeleton_body = torch.zeros(
            (num_frames_body, max_num_persons_body, body_dim),
            dtype=skeleton_body.dtype
        )

        for f_idx in range(num_frames_body):
            frame_skeleton = skeleton_body[f_idx]  # shape: [num_persons_body, body_dim]
            n_persons_in_frame = frame_skeleton.shape[0]

            # Copy the real skeleton data
            padded_skeleton_body[f_idx, :n_persons_in_frame] = frame_skeleton

            # If there's room left, replicate the last person's skeleton
            if n_persons_in_frame < max_num_persons_body:
                last_person = frame_skeleton[-1].unsqueeze(0)  # shape: [1, body_dim]
                reps = max_num_persons_body - n_persons_in_frame
                padded_skeleton_body[f_idx, n_persons_in_frame:] = last_person.expand(
                    reps, body_dim
                )

        skeleton_body_list.append(padded_skeleton_body)

        # ========== BODY ADJACENCY ==========
        adjacency_body = sample["adjacency_person"]
        # shape is [num_frames, num_persons_body, num_body_limbs, num_body_limbs]
        _, _, num_body_limbs, _ = adjacency_body.shape

        padded_adjacency_body = torch.zeros(
            (num_frames_body, max_num_persons_body, num_body_limbs, num_body_limbs),
            dtype=adjacency_body.dtype
        )

        for f_idx in range(num_frames_body):
            frame_adjacency = adjacency_body[f_idx]  # shape: [num_persons_body, num_body_limbs, num_body_limbs]
            n_persons_in_frame = frame_adjacency.shape[0]

            padded_adjacency_body[f_idx, :n_persons_in_frame] = frame_adjacency

            if n_persons_in_frame < max_num_persons_body:
                last_adjacency = frame_adjacency[-1].unsqueeze(0)
                reps_adj = max_num_persons_body - n_persons_in_frame
                padded_adjacency_body[f_idx, n_persons_in_frame:] = last_adjacency.expand(
                    reps_adj, num_body_limbs, num_body_limbs
                )

        adjacency_body_list.append(padded_adjacency_body)

        # ========== FACE SKELETON ==========
        skeleton_face = sample["skeleton_face"]
        num_frames_face, num_persons_face, face_dim = skeleton_face.shape

        padded_skeleton_face = torch.zeros(
            (num_frames_face, max_num_persons_face, face_dim),
            dtype=skeleton_face.dtype
        )

        for f_idx in range(num_frames_face):
            frame_face = skeleton_face[f_idx]  # shape: [num_persons_face, face_dim]
            n_persons_face_in_frame = frame_face.shape[0]

            padded_skeleton_face[f_idx, :n_persons_face_in_frame] = frame_face

            if n_persons_face_in_frame < max_num_persons_face:
                last_face = frame_face[-1].unsqueeze(0)  # shape: [1, face_dim]
                reps_face = max_num_persons_face - n_persons_face_in_frame
                padded_skeleton_face[f_idx, n_persons_face_in_frame:] = last_face.expand(
                    reps_face, face_dim
                )

        skeleton_face_list.append(padded_skeleton_face)

        # ========== FACE ADJACENCY ==========
        adjacency_face = sample["adjacency_face"]
        # shape: [num_frames, num_persons_face, num_face_limbs, num_face_limbs]
        _, _, num_face_limbs, _ = adjacency_face.shape

        padded_adjacency_face = torch.zeros(
            (num_frames_face, max_num_persons_face, num_face_limbs, num_face_limbs),
            dtype=adjacency_face.dtype
        )

        for f_idx in range(num_frames_face):
            frame_adjacency_face = adjacency_face[f_idx]
            n_persons_face_in_frame = frame_adjacency_face.shape[0]

            padded_adjacency_face[f_idx, :n_persons_face_in_frame] = frame_adjacency_face

            if n_persons_face_in_frame < max_num_persons_face:
                last_face_adj = frame_adjacency_face[-1].unsqueeze(0)
                reps_face_adj = max_num_persons_face - n_persons_face_in_frame
                padded_adjacency_face[f_idx, n_persons_face_in_frame:] = last_face_adj.expand(
                    reps_face_adj, num_face_limbs, num_face_limbs
                )

        adjacency_face_list.append(padded_adjacency_face)

    # 4) Stack results into final batch tensors
    # images_list: [B, num_frames, C, H, W]
    images_batch = torch.stack(images_list, dim=0)

    # skeleton_body_list: [B, num_frames, max_num_persons_body, body_dim]
    skeleton_body_batch = torch.stack(skeleton_body_list, dim=0)

    # adjacency_body_list: [B, num_frames, max_num_persons_body, num_body_limbs, num_body_limbs]
    adjacency_body_batch = torch.stack(adjacency_body_list, dim=0)

    # skeleton_face_list: [B, num_frames, max_num_persons_face, face_dim]
    skeleton_face_batch = torch.stack(skeleton_face_list, dim=0)

    # adjacency_face_list: [B, num_frames, max_num_persons_face, num_face_limbs, num_face_limbs]
    adjacency_face_batch = torch.stack(adjacency_face_list, dim=0)

    # 5) Return your batched dictionary
    return {
        "images": images_batch,
        "skeleton_person": skeleton_body_batch,
        "adjacency_person": adjacency_body_batch,
        "skeleton_face": skeleton_face_batch,
        "adjacency_face": adjacency_face_batch,
    }
