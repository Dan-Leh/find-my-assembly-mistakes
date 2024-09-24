#!/bin/bash

MODEL=noam # can be noam, gca, msa, or lca
python train.py \
    --config 'config_files/example_train.yaml' \
    --experiment_name $MODEL \
    --cyws/attention $MODEL \
    --data_root /shared/nl011006/res_ds_ml_restricted/dlehman/SyntheticData \
    --checkpoint_root /hpc/scratch/dlehman/CD_checkpoints \
    --output_root /shared/nl011006/res_ds_ml_restricted/dlehman/find-my-assembly-mistakes/results \
    --bg_img_root /shared/nl011006/res_ds_ml_restricted/dlehman/COCO_Images