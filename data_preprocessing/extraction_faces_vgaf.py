import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import os
import json
import pandas as pd
import numpy as np
import face_alignment
from skimage import io

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



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



def save_faces_align_videos(train_flag=True):
    path_annotation = '/....../VGAF_Video_images_25/'
    
    # Determine paths based on train or validation flag
    if train_flag:
        csv_file = 'Train_labels.txt'
        data_folder = os.path.join(path_annotation, "Train/")
        annotations = pd.read_csv(os.path.join(path_annotation, csv_file), engine='python', sep=' ')
        output_file = os.path.join(path_annotation, 'train_video_face_align.json')
    else:
        csv_file = 'Val_labels.txt'
        data_folder = os.path.join(path_annotation, "Val/")
        annotations = pd.read_csv(os.path.join(path_annotation, csv_file), engine='python', sep=' ')
        output_file = os.path.join(path_annotation, 'val_video_face_align.json')

    final_labels_bbox = []
    
     # Initialize Face Alignment once (outside the loop)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')
  
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
        for i in range(25):        
            frame_path = os.path.join(os.path.join(video_folder_path, str(i)+'.png'))
            # Load image
            input_image = io.imread(frame_path)

            # Get landmarks (list of numpy arrays, one per face)
            landmarks_list = fa.get_landmarks(input_image)
            if landmarks_list is None or len(landmarks_list) == 0:
                continue

            # Convert each array to list-of-lists
            landmarks_list_python = [arr.tolist() for arr in landmarks_list]

            # Add frame annotations if any faces were detected
            if landmarks_list_python:
                video_annotations.append({
                    'frame_name':  str(i)+'.png',
                    'faces_fa': landmarks_list_python
                })
            print(f"Processed {frame_path}")

        # Only add video if it has any annotated frames
        if video_annotations:
            final_labels_bbox.append({
                'file_name':video_name,
                'emotion_label': int(emotion_label),
                'face_annotations':video_annotations,
            })  

    # Convert final data to native Python types
    final_labels_bbox = convert_numpy_types(final_labels_bbox)
    # Save the results to output file
    with open(output_file, 'w') as file:
        json.dump(final_labels_bbox, file, indent=4)

    print(f"Saved file to {output_file}")


if __name__ == "__main__":
    save_faces_align_videos()