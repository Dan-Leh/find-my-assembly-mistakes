#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N gca_shear

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

NAME=gca_w_shearing
python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/example_config_train.yaml' \
--experiment_name $NAME \
--T_0 100 \
--T_mult 2 \
--cyws/attention gca \
--img_transforms/shear 10 \
--img_transforms/saturation 0.6