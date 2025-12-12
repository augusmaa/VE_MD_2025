#!/bin/bash
#ulimit -n 600000 # this the number of files that open in the same time for the OS.
export CUDA_VISIBLE_DEVICES=1,0 # to set devices
#unset CUDA_VISIBLE_DEVICES

nice -n 5 python run.py  --portvalue 1  --use_mmd --datasets gaf3 --decoder_sktname openpose  --use_person_pose --use_emotion --encoder_name resnet50  --add_keypoints --batch_gaf3 2 --latent_dim 16
