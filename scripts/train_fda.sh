#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N fda_50_mildaug
#PBS -M dan.lehman@asml.com
#PBS -m e
source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

NAME=gca_fda50%_mildaug
python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/example_config_train_fda.yaml' \
--experiment_name $NAME \
--cyws/attention gca \
--img_transforms/brightness 0.1 \
--img_transforms/contrast 0.1 \
--img_transforms/saturation 0.2 \
--img_transforms/hue 0.05 \
--img_transforms/gradually_augment True \
--fda/frac_imgs_w_fda 0.5 \
--fda/beta_min 0.005 \
--fda/beta_max 0.005


