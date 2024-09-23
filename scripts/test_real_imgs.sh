#!/bin/bash

for TESTSET in Random_background #Main_test_set Novel_poses_test_set Novel_parts_test_set 
do
    python test_on_real_imgs.py \
        --config /shared/nl011006/res_ds_ml_restricted/dlehman/find-my-assembly-mistakes/results/noam/config.yaml \
        --test_set_name real_img_test_set 
done

