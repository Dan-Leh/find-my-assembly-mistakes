#!/bin/bash

for TESTSET in Random_background #Main_test_set Novel_poses_test_set Novel_parts_test_set 
do
    python test.py \
        --config '/shared/nl011006/res_ds_ml_restricted/dlehman/find-my-assembly-mistakes/results/gca(2)/config.yaml' \
        --experiment_name "gca(2)" \
        --test_set_name $TESTSET
done

