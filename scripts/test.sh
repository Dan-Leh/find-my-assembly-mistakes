#!/bin/bash

for TESTSET in  Main_test_set Novel_poses_test_set Novel_parts_test_set Random_background
do
    python test.py \
        --config '/shared/nl011006/res_ds_ml_restricted/dlehman/find-my-assembly-mistakes/results/noam/config.yaml' \
        --experiment_name "noam" \
        --test_set_name $TESTSET \
        --bg_img_root /shared/nl011006/res_ds_ml_restricted/dlehman/COCO_Images  # only relevant for Random_background test set
done

