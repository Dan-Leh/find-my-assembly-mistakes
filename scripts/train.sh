#!/bin/bash

MODEL=gca # can be noam, gca, msa, or lca
python train.py \
    --config 'config_files/example_train.yaml' \
    --experiment_name $MODEL \
