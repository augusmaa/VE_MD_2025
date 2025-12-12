import os
import sys
import subprocess
import time
import re
import json
import pandas as pd
import numpy as np
import cv2
import ast


# Helper function to convert numpy types to native Python types
def convert_numpy_types(data):
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(element) for element in data]
    elif isinstance(data, np.generic):  # Convert numpy types to native types
        return data.item()
    else:
        return data

def save_persons_pose_videos(train_flag=True):
    """
    all explanation of using ViTPose is available on the github link bellow.
    https://github.com/Pukei-Pukei/ViTPose-ONNX
    
    """

    path_annotation = '....../VGAF_Video_images_25/'
    
    # Determine paths based on train or validation flag
    if train_flag:
        csv_file = 'Train_labels.txt'
        data_folder = os.path.join(path_annotation, "Train/")
        annotations = pd.read_csv(os.path.join(path_annotation, csv_file), engine='python', sep=' ')
        output_file = os.path.join(path_annotation, 'train_video_annotation_vitpose.json')
    else:
        csv_file = 'Val_labels.txt'
        data_folder = os.path.join(path_annotation, "Val/")
        annotations = pd.read_csv(os.path.join(path_annotation, csv_file), engine='python', sep=' ')
        output_file = os.path.join(path_annotation, 'val_video_annotation_vitpose.json')
        
    final_labels_bbox = []
  
    # Iterate over each video (folder)
    for video_nb in range(len(annotations)):
        emotion_label = annotations['Label'][video_nb]
        video_name =(annotations['Vid_name']+'.mp4.img')[video_nb] # return the video name
        video_folder_path = os.path.join(data_folder, video_name)
        # Only process if it's a directory (video folder)
        if not os.path.isdir(video_folder_path):
            continue
        # Initialize storage for this video’s frames
        video_annotations = []

        # Process each frame (image) in the video folder
        for i in range(50):        
            frame_path = os.path.join(os.path.join(video_folder_path, str(i)+'.png'))
            # Initialize list to hold bbox and scores for persons detected
            person_data = []
            run_py_path = "keypoints.py"
            command = f"python3 {run_py_path} -img \"{frame_path}\" "
            # Run the command and capture output
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            poose_info = result.stdout
            poose_info = ast.literal_eval(poose_info)
            # Add pose info to person_data
            person_data.append(poose_info)
            # Add frame annotations if any faces were detected
            if person_data:
                video_annotations.append({
                    'frame_name':  str(i)+'.png',
                    'persons': person_data
                })
                print(f"Processed {frame_path}")

        # Only add video if it has any annotated frames
        if video_annotations:
            final_labels_bbox.append({
                'file_name':video_name,
                'emotion_label': int(emotion_label),
                'person_annotations':video_annotations,
            })  

    # Convert final data to native Python types
    final_labels_bbox = convert_numpy_types(final_labels_bbox)
    # Save the results to output file
    with open(output_file, 'w') as file:
        json.dump(final_labels_bbox, file, indent=4)


if __name__ == "__main__":
    save_persons_pose_videos()