import os
import cv2
import h5py
import numpy as np
from PIL import Image
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import copy
import cv2
import math
import multiprocessing
import numpy  as  np
import os
import pandas as pd
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

import io


import subprocess, tempfile, os
from PIL import Image


import os
import subprocess
from PIL import Image

def Frames_one_Video_FFmpeg(frames_per_sec, Dir_videos, vid_name, duration, input_size=224):
    """
    Uses ffmpeg to dump frames, handling both .mp4 and .avi extensions.
    Returns list of PIL Images.
    """
    # Check for existing video files with either extension
    found_path = None
    for ext in ['.mp4', '.avi']:
        check_path = os.path.join(Dir_videos, f"{vid_name}{ext}")
        if os.path.exists(check_path):
            found_path = check_path
            break
            
    if not found_path:
        raise FileNotFoundError(f"Video {vid_name} not found with .mp4 or .avi extensions")

    total_frames = int(frames_per_sec * duration)
    cmd = [
        'ffmpeg', '-v', 'error', '-i', found_path,
        '-vf', f"fps={frames_per_sec},scale={input_size}:{input_size}",
        '-frames:v', str(total_frames),
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    images = []
    frame_size = input_size * input_size * 3
    
    try:
        for _ in range(total_frames):
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            img = Image.frombytes('RGB', (input_size, input_size), raw)
            images.append(img)
    finally:
        proc.stdout.close()
        proc.wait()
    
    # Pad if necessary
    if len(images) < total_frames and images:
        images += [images[-1]] * (total_frames - len(images))
        
    return images



VIDEO_EXTENSIONS = ['.mp4', '.avi']


def resolve_video_path(base_name, Dir_videos):
    """
    Given a base filename (with or without extension),
    return the full filename that exists in Dir_videos.
    Tries the name as-is first, then tries common extensions.
    """
    # If name includes extension, check directly
    full_path = os.path.join(Dir_videos, base_name)
    if os.path.exists(full_path):
        return base_name
    # Otherwise try common extensions
    name_no_ext, _ = os.path.splitext(base_name)
    for ext in VIDEO_EXTENSIONS:
        candidate = name_no_ext + ext
        if os.path.exists(os.path.join(Dir_videos, candidate)):
            return candidate
    raise FileNotFoundError(f"No video file found for base '{base_name}' in {Dir_videos}")


def Frames_one_Video_uniform_ffmpeg(nb_of_frames, Dir_videos, vid_name, input_size=224):
    """
    Extract nb_of_frames uniformly sampled frames from a video using ffmpeg.
    Handles both .mp4 and .avi by resolving extension.
    Returns a list of PIL Images (RGB, resized to input_size x input_size).
    """
    # resolve full filename (handles .mp4 and .avi)
    filename = resolve_video_path(vid_name, Dir_videos)
    path_in = os.path.join(Dir_videos, filename)

    # Probe duration via ffprobe
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path_in
    ]
    proc = subprocess.Popen(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        proc.stderr.close()
        raise RuntimeError(f"ffprobe error on '{filename}': {err.decode().strip()}")
    try:
        duration = float(out)
    except ValueError:
        return []
    if duration <= 0:
        return []

    # Calculate FPS filter to get exactly nb_of_frames over full duration
    fps_filter = nb_of_frames / duration
    # Build ffmpeg command to pipe raw RGB frames
    ffmpeg_cmd = [
        'ffmpeg', '-v', 'error', '-i', path_in,
        '-vf', f"fps={fps_filter},scale={input_size}:{input_size}",
        '-frames:v', str(nb_of_frames),
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    images = []
    frame_size = input_size * input_size * 3
    for _ in range(nb_of_frames):
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        img = Image.frombytes('RGB', (input_size, input_size), raw)
        images.append(img)
    proc.stdout.close()
    proc.wait()

    # Pad if fewer frames extracted
    if len(images) < nb_of_frames and images:
        images.extend([images[-1]] * (nb_of_frames - len(images)))
    return images


def SaveVideoFrames_to_H5(nb_of_frames, data_flag='train', output_dir='/Corpora/VGAF'):
    """
    For each video in the specified split, extract uniform frames and save to a single HDF5 file.
    The file will be created at output_dir/datasetname_{nb_of_frames}/{split}.h5
    Each video is stored as a dataset named by its filename (without extension).
    """
    # define input directories
    DIR_map = {
        'Train': "...../VGAF_original/Train/",
        'Val': "....../VGAF_original/Val/",
        'Test': "...../augusmaa/Documents/VGAF/Test/",
    }
    if data_flag not in DIR_map:
        raise ValueError(f"Unknown data_flag: {data_flag}")
    DIR_videos = DIR_map[data_flag]
    
    # prepare output folder and HDF5 path
    save_folder = os.path.join(output_dir, f"VGAF_{nb_of_frames}")
    os.makedirs(save_folder, exist_ok=True)
    h5_path = os.path.join(save_folder, f"{data_flag}.h5")
    
    # open HDF5 file
    with h5py.File(h5_path, 'w') as h5f:
        for vid_name in sorted(os.listdir(DIR_videos)):
            base, ext = os.path.splitext(vid_name)
            dataset_name = base + ext + '.img'          # ← build “video.mp4.img”
            frames = Frames_one_Video_uniform_ffmpeg(nb_of_frames, DIR_videos, vid_name)
            if not frames:
                print(f"Skipping {vid_name}: no frames extracted.")
                continue
            arr = np.stack([np.array(im) for im in frames], axis=0)
            h5f.create_dataset(
                name=dataset_name,                    # ← use the new key
                data=arr,
                shape=arr.shape,
                dtype='uint8',
                compression='gzip'
            )
            print(f"Saved {vid_name} → dataset '{dataset_name}' shape {arr.shape}")
            #break
    print(f"All done! HDF5 file created at: {h5_path}")

if __name__ == "__main__":
    #SaveVideoFrames_to_H5(nb_of_frames=10, data_flag='Train')
    SaveVideoFrames_to_H5(nb_of_frames=10, data_flag='Val')
    #SaveVideoFrames_to_H5(nb_of_frames=10, data_flag='Test')



