#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N rand_bg_gca
#PBS -M dan.lehman@asml.com
#PBS -m e

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

NAME=gca_rand_bg
python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/gca_w_shearing/config.yaml' \
--experiment_name $NAME \
--img_transforms/random_background true