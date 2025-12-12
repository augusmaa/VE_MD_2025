#!/bin/bash
#ulimit -n 600000 # this the number of files that open in the same time for the OS.
export CUDA_VISIBLE_DEVICES=0,1 # to set devices
#unset CUDA_VISIBLE_DEVICES

nice -n 5 python run.py --use_mmd  --portvalue 1 --datasets gaf3 --use_person_pose  --encoder_name resnet50 --notifications --num_epochs 400 --num_queries 2  --add_keypoints  --LR_emotion 0.00001  #--classif_projection

 
