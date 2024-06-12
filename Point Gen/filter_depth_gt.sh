#!/usr/bin/env bash

# test on DTU's evaluation set
DTU_TESTING="/home/share/jx/dtu_training/dtu/"

python filter_depth_gt.py --scan_list ../lists/dtu/train.txt --input_folder=$DTU_TESTING --output_folder=$DTU_TESTING \
--num_views 5 --image_max_dim 640 --geo_mask_thres 3 --photo_thres 0.8 "$@"