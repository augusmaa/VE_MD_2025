import torch
import numpy as np 
import os 
from PIL import Image
import pandas as pd 
import json
import ast
import cv2
import h5py


import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import collections



def resize_keypoints(keypoints, original_size, new_size):
    """
    Resize keypoints to match the new image size.

    Args:
        keypoints (list of tuple): List of keypoint coordinates [(x, y), ...]. for One person
        original_size (tuple): Original image size (H, W).
        new_size (tuple): Target image size (H, W).

    Returns:     
        list of tuple: Resized keypoint coordinates [(x', y'), ...]. for one person
    """
    orig_h, orig_w = original_size
    new_h, new_w = new_size
    scale_h, scale_w = new_h / orig_h, new_w / orig_w

    resized_keypoints = [(x * scale_w, y * scale_h) for x, y in keypoints]
    return resized_keypoints


def clamp_to_range(landmarks, new_size):
    """
    Clamps (x, y) coordinates to [0, width-1] x [0, height-1].
    If a landmark is out of range, set it to (-1, -1).
    
    Args:
      landmarks (list of tuples): [(x0, y0), (x1, y1), ...]
      width (int):  target image width (e.g., 224)
      height (int): target image height (e.g., 224)
      
    Returns:
      list of tuples: updated landmarks
    """
    width, height = new_size
    clamped = []
    for (x, y) in landmarks:
        # Check if (x, y) is out of [0, width-1] or [0, height-1]
        if x <= 0 or x >= width or y <= 0 or y >= height:
            # Mark invalid by (-1, -1)
            clamped.append((-1, -1))
        else:
            # Keep original coordinate (or you can round if you want integer)
            clamped.append((x, y))
    return clamped


limbs_pairs_person  = [
    (0, 5),   # Nose → Left Shoulder
    (0, 6),   # Nose → Right Shoulder
    (0, 1),   # Nose → Left Eye
    (0, 2),   # Nose → Right Eye
    (1, 3),   # Left Eye → Left Ear
    (2, 4),   # Right Eye → Right Ear
    (5, 6),   # Left Shoulder → Right Shoulder
    (5, 7),   # Left Shoulder → Left Elbow
    (7, 9),   # Left Elbow → Left Wrist
    (6, 8),   # Right Shoulder → Right Elbow
    (8, 10),  # Right Elbow → Right Wrist
    (5, 11),  # Left Shoulder → Left Hip
    (6, 12),  # Right Shoulder → Right Hip
    (11, 12), # Left Hip → Right Hip
    (11, 13), # Left Hip → Left Knee
    (13, 15), # Left Knee → Left Ankle
    (12, 14), # Right Hip → Right Knee
    (14, 16)  # Right Knee → Right Ankle
    ]




# 1) Define the facial edges for 68 landmarks
jaw = [(i, i+1) for i in range(0, 16)]
left_eyebrow_1 = [(i, i+1) for i in range(17, 21)]
right_eyebrow_1 = [(i, i+1) for i in range(22, 26)]
nose_bridge = [(i, i+1) for i in range(27, 30)]
nose_base = [(i, i+1) for i in range(31, 35)]
left_eye = [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)]
right_eye = [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)]
mouth_outer = [
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
    (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48)
]
mouth_inner = [
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65),
    (65, 66), (66, 67), (67, 60)
]


chin = [(7,8),(8,9)]
chin_to_mouth = [(8,65),(8,66),(7, 48),(9, 54)]
nose_to_mouth = [(30,48),(30,54),(30,55),(30,56),(30,57),(30,58),(30,59)]
mouth_entry = [(48, 54), (49, 55), (50, 56), (51, 57), (52, 58), (53, 59)]
eye_left_entry = [(36, 41),(37, 40),(38, 39)]
eye_right_entry = [(42, 47), (43, 46),(44, 45)]
left_eyebrow = [(17, 19), (19, 21)]
right_eyebrow = [(22, 24), (24, 26)]
mouth_open = [(51,56),(50,57)]

limbs_pairs_face_1 = (jaw + left_eyebrow_1 + right_eyebrow_1 +
              nose_bridge + nose_base +
              left_eye + right_eye +
              mouth_outer + mouth_inner)


cunstom_limbs = (eye_left_entry
                 + eye_right_entry
                 +left_eyebrow
                 +right_eyebrow
                 +mouth_inner
                 +mouth_open
                ) # 20 limbs

limbs_pairs_face =  (cunstom_limbs 
                     +
                     limbs_pairs_face_1
                     )



def generate_person_limbs_heatmap(keypoints_list, new_size, heatmap_size, limb_pairs, sigma=2):
    """
    Generate heatmaps for limbs (connections between keypoints) across multiple persons.

    Args:
        keypoints_list (list of list): List of keypoints for all persons, where each person's keypoints
            are a list of (x, y).
        original_size (tuple): Original image size (H, W).
        new_size (tuple): Resized image dimensions (H, W) used before mapping to the heatmap.
        heatmap_size (tuple): The desired heatmap size (H, W).
        limb_pairs (list of tuples): List of limb pairs, each a tuple (start_index, end_index).
        sigma (float): Standard deviation for the Gaussian.

    Returns:
        torch.Tensor: Limb heatmaps of shape (num_limbs, H, W).
    """
    num_limbs = len(limb_pairs)
    H, W = heatmap_size
    img_H, img_W = new_size  # Resized image dimensions

    # Create an empty tensor for all limb heatmaps
    heatmaps = torch.zeros((num_limbs, H, W), dtype=torch.float32)

    # Create a meshgrid for the heatmap coordinates
    x = torch.arange(W, dtype=torch.float32)
    y = torch.arange(H, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # Shapes: (H, W)

    # Process each limb
    for l, (idx1, idx2) in enumerate(limb_pairs):
        # Iterate over each person's keypoints
        for person_kp in keypoints_list:
            # Check that both keypoints exist for this person
            if max(idx1, idx2) >= len(person_kp):
                continue

            # Get the keypoints for the limb
            y1, x1 = person_kp[idx1]
            y2, x2 = person_kp[idx2]

            # Scale keypoint coordinates to fit the heatmap size
            x1 = x1 * (W / img_W)
            y1 = y1 * (H / img_H)
            x2 = x2 * (W / img_W)
            y2 = y2 * (H / img_H)

            # Optionally, skip limbs with invalid coordinates (outside heatmap boundaries)
            if (x1 < 0 or y1 < 0 or x1 >= W or y1 >= H or
                x2 < 0 or y2 < 0 or x2 >= W or y2 >= H):
                continue

            # Compute differences and squared length of the limb vector
            dx = x2 - x1
            dy = y2 - y1
            norm_sq = dx**2 + dy**2

            if norm_sq < 1e-6:
                # If the limb is degenerate, create a heatmap like a keypoint
                limb_heatmap = torch.exp(-((xx - x1)**2 + (yy - y1)**2) / (2 * sigma**2))
            else:
                # Compute the projection factor t for each pixel on the line direction.
                t = ((xx - x1) * dx + (yy - y1) * dy) / norm_sq
                # Clamp t to lie between 0 and 1 (i.e. restrict to the segment)
                t = t.clamp(0, 1)
                # Compute the projection coordinates (closest points on the segment)
                xp = x1 + t * dx
                yp = y1 + t * dy
                # Compute squared distances from each pixel to the limb
                d2 = (xx - xp)**2 + (yy - yp)**2
                # Apply the Gaussian
                limb_heatmap = torch.exp(-d2 / (2 * sigma**2))

            # Accumulate the heatmap (if multiple persons contribute to the same limb)
            heatmaps[l] += limb_heatmap

        # Clamp the accumulated heatmap values to avoid saturation beyond 1
        heatmaps[l] = heatmaps[l].clamp(max=1.0)
    
    return heatmaps



def generate_face_limbs_heatmap(keypoints_list, new_size, heatmap_size, limb_pairs, sigma=2):
    """
    Generate heatmaps for limbs (connections between keypoints) across multiple persons.

    Args:
        keypoints_list (list of list): List of keypoints for all persons, where each person's keypoints
            are a list of (x, y).
        original_size (tuple): Original image size (H, W).
        new_size (tuple): Resized image dimensions (H, W) used before mapping to the heatmap.
        heatmap_size (tuple): The desired heatmap size (H, W).
        limb_pairs (list of tuples): List of limb pairs, each a tuple (start_index, end_index).
        sigma (float): Standard deviation for the Gaussian.

    Returns:
        torch.Tensor: Limb heatmaps of shape (num_limbs, H, W).
    """
    num_limbs = len(limb_pairs)
    H, W = heatmap_size
    img_H, img_W = new_size  # Resized image dimensions

    # Create an empty tensor for all limb heatmaps
    heatmaps = torch.zeros((num_limbs, H, W), dtype=torch.float32)

    # Create a meshgrid for the heatmap coordinates
    x = torch.arange(W, dtype=torch.float32)
    y = torch.arange(H, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # Shapes: (H, W)

    # Process each limb
    for l, (idx1, idx2) in enumerate(limb_pairs):
        # Iterate over each person's keypoints
        for person_kp in keypoints_list:
            # Check that both keypoints exist for this person
            if max(idx1, idx2) >= len(person_kp):
                continue

            # Get the keypoints for the limb
            x1, y1 = person_kp[idx1]
            x2, y2 = person_kp[idx2]

            # Scale keypoint coordinates to fit the heatmap size
            x1 = x1 * (W / img_W)
            y1 = y1 * (H / img_H)
            x2 = x2 * (W / img_W)
            y2 = y2 * (H / img_H)

            # Optionally, skip limbs with invalid coordinates (outside heatmap boundaries)
            if (x1 < 0 or y1 < 0 or x1 >= W or y1 >= H or
                x2 < 0 or y2 < 0 or x2 >= W or y2 >= H):
                continue

            # Compute differences and squared length of the limb vector
            dx = x2 - x1
            dy = y2 - y1
            norm_sq = dx**2 + dy**2

            if norm_sq < 1e-6:
                # If the limb is degenerate, create a heatmap like a keypoint
                limb_heatmap = torch.exp(-((xx - x1)**2 + (yy - y1)**2) / (2 * sigma**2))
            else:
                # Compute the projection factor t for each pixel on the line direction.
                t = ((xx - x1) * dx + (yy - y1) * dy) / norm_sq
                # Clamp t to lie between 0 and 1 (i.e. restrict to the segment)
                t = t.clamp(0, 1)
                # Compute the projection coordinates (closest points on the segment)
                xp = x1 + t * dx
                yp = y1 + t * dy
                # Compute squared distances from each pixel to the limb
                d2 = (xx - xp)**2 + (yy - yp)**2
                # Apply the Gaussian
                limb_heatmap = torch.exp(-d2 / (2 * sigma**2))

            # Accumulate the heatmap (if multiple persons contribute to the same limb)
            heatmaps[l] += limb_heatmap

        # Clamp the accumulated heatmap values to avoid saturation beyond 1
        heatmaps[l] = heatmaps[l].clamp(max=1.0)
    
    return heatmaps

    
    # VGAF transforms
general_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            #transforms.RandomHorizontalFlip(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }



import os
import json
import torch
from PIL import Image

def SaveLimbsVideo(
    data_folder_src,
    csv_file_src,
    data_folder_dst,
    csv_file_dst,
    save_imgs=False,
    heatmap_size=(56, 56),
    sigma=1,
    nb_frames=10,
    score=0.1
    ):
    # create dir to save data
    os.makedirs(data_folder_dst, exist_ok=True)
    
    with open(os.path.join(csv_file_src), 'r') as file:
        annotations = json.load(file)

    keep_files_names = []

    for index in range(len(annotations)):
        annotation = annotations[index]
        label_emotion = annotation['emotion_label']
        file_name = annotation['file_name']
        video_folder_src = os.path.join(data_folder_src, file_name)

        # Storage for frames, skeletons, adjacency
        Frames = []
        One_video_person_limbs = []
        One_video_face_limbs = []
        keep_frames_names = []

        # ---- Loop over frames in this video
        for frame_annotation in annotation.get("frames", []):
            frame_name = frame_annotation["frame_name"]
            frame_path = os.path.join(video_folder_src, frame_name)

            # -- Load and process the image
            image_context = Image.open(frame_path)
            if image_context.mode != 'RGB':
                image_context = image_context.convert('RGB')
            original_size = image_context.size

            new_size = (224, 224)
            image_frame = image_context.resize(new_size, Image.LANCZOS)
            image_context = general_transforms['train'](image_frame)  # your transform

            # -- People in this frame
            persons = frame_annotation["persons"][0]
            faces = frame_annotation.get("faces", [])

            if not persons['pose']:
                keypoints_list = []
            else:
                # Filter out low confidence keypoints
                keypoints_list = [
                    [
                        kp[:2] if kp[-1] >= score else (-1, -1)
                        for kp in keypoints
                    ]
                    for keypoints in persons['pose'][0]
                ]

            # -- Resize keypoints to new image size
            keypoints_list_resized = []
            landmarks_list_resized = []

            if len(keypoints_list) != 0 and len(faces) != 0:
                keypoints_list_resized = [
                    resize_keypoints(person_kp, original_size, new_size)
                    for person_kp in keypoints_list
                ]
                
                landmarks_list_resized = [
                    resize_keypoints(face_kp, original_size, new_size)
                    for face_kp in faces
                ]
                # clamp for landmarks only
                landmarks_list_resized = [
                    clamp_to_range(face_kp, new_size)
                    for face_kp in landmarks_list_resized
                ]  

                limbs_frame_person = generate_person_limbs_heatmap(keypoints_list_resized, new_size, heatmap_size, limbs_pairs_person, sigma)
                limbs_frame_face = generate_face_limbs_heatmap(landmarks_list_resized, new_size, heatmap_size, limbs_pairs_face, sigma)





            # Append to the list of frames for this video
            One_video_person_limbs.append(limbs_frame_person)
            One_video_face_limbs.append(limbs_frame_face)
            Frames.append(image_context)

            frame_info = {
                "frame_name": frame_name,
                "persons": persons,
                "faces": faces,
            }
            keep_frames_names.append(frame_info)

        # --------------------------------------------------------------
        # If after processing all frames, no persons => skip the video
        # --------------------------------------------------------------
        if len(One_video_person_limbs) == 0 or len(One_video_face_limbs) == 0:
            print(f"Skipping video {file_name}: no valid frames with persons.")
            continue

    

        # --------------------------------------------------------------
        # 3) Pad the number of frames if needed (temporal dimension)
        # --------------------------------------------------------------
       
        current_nb_frames = len(Frames)
        if current_nb_frames < nb_frames:
            while len(One_video_person_limbs) < nb_frames:
                One_video_person_limbs.append(One_video_person_limbs[-1])
                One_video_face_limbs.append(One_video_face_limbs[-1])
                Frames.append(Frames[-1])

        person_limbs = torch.stack(One_video_person_limbs)
        face_limbs = torch.stack(One_video_face_limbs)
        #print(limb.shape)
        # --------------------------------------------------------------
        # 4) Save the data
        # --------------------------------------------------------------
        output_person_limbs = os.path.join(data_folder_dst, f"{file_name}_person_limb.pt")
        output_face_limbs = os.path.join(data_folder_dst, f"{file_name}_face_limb.pt")
        output_imgs = os.path.join(data_folder_dst, f"{file_name}.pt")

        # shape: [T, max_num_persons, ...]
        torch.save(torch.stack(One_video_person_limbs), output_person_limbs)
        torch.save(torch.stack(One_video_face_limbs), output_face_limbs)

        if save_imgs:
            torch.save(torch.stack(Frames), output_imgs)
            print("saved:", output_imgs)

        print("saved:", output_person_limbs)

        # # Record info for writing JSON
        # if len(One_video_person_limbs) != 0:
        #     data_info = {
        #         "file_name": file_name,
        #         "label_emotion": label_emotion,
        #         "frames": keep_frames_names,
        #     }
        #     keep_files_names.append(data_info)

        # (Optionally remove the break if you want to process *all* videos)
        #break

    # Finally, write out the new CSV/JSON with the updated data
    with open(csv_file_dst, 'w') as file:
        json.dump(keep_files_names, file, indent=4) 
    return person_limbs, face_limbs,  torch.stack(Frames)

if __name__=="__main__":
    csv_file_dst =  "..../SamSemo/train_video_annotation_all_vitpose_limbs.json"
    data_folder_dst= '..../SamSemo/Train_heatmap_56_2'

    csv_file_src =  "...../SamSemo/train_video_annotation_all_vitpose_fa.json"
    data_folder_src= '...../SamSemo/Train'
    
    person, face, imgs = SaveLimbsVideo(data_folder_src,csv_file_src, data_folder_dst, csv_file_dst,save_imgs=True,heatmap_size=(56, 56), sigma=1, nb_frames=10)


