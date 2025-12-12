#!/bin/bash
#ulimit -n 600000 # this the number of files that open in the same time for the OS.
#export CUDA_VISIBLE_DEVICES=0 # to set devices
#unset CUDA_VISIBLE_DEVICES
 

#nice -n 5 python  feature_extraction_vit.py  --cuda_device 0  --dataset_emotion vgaf  
#nice -n 5 python  feature_extraction_vit.py --cuda_device 0   --dataset_emotion vgaf  --train_flag

#nice -n 5 python  feature_extraction_vit.py  --cuda_device 0  --dataset_emotion vgaf  --nb_frames_vgaf  25
#nice -n 5 python  feature_extraction_vit.py --cuda_device 0   --dataset_emotion vgaf  --nb_frames_vgaf 25 --train_flag

#nice -n 5 python  feature_extraction_vit.py  --cuda_device 0  --dataset_emotion gaf3  
#nice -n 5 python  feature_extraction_vit.py --cuda_device 0   --dataset_emotion gaf3  --train_flag

#nice -n 5 python  feature_extraction_vit.py  --cuda_device 0  --dataset_emotion samsemo  
#nice -n 5 python  feature_extraction_vit.py --cuda_device 0   --dataset_emotion samsemo --train_flag

#nice -n 5 python  feature_extraction_vit.py  --cuda_device 0  --dataset_emotion engagenet  
#nice -n 5 python  feature_extraction_vit.py --cuda_device 0   --dataset_emotion engagenet  --train_flag

#nice -n 5 python  feature_extraction_vit.py  --cuda_device 0  --dataset_emotion dfew  
#nice -n 5 python  feature_extraction_vit.py --cuda_device 0   --dataset_emotion dfew  --train_flag


# nice -n 5 python  feature_extraction_vit.py --cuda_device 0      --dataset_emotion mer    --dataset_type Train
# nice -n 5 python  feature_extraction_vit.py --cuda_device 0     --dataset_emotion mer     --dataset_type test3
# nice -n 5 python  feature_extraction_vit.py --cuda_device 0     --dataset_emotion mer     --dataset_type test2
# nice -n 5 python  feature_extraction_vit.py --cuda_device 0     --dataset_emotion mer     --dataset_type Val
