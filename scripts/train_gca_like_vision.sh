#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N gca_like_VISION

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

NAME=gca_like_VISION
python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/example_config_train.yaml' \
--experiment_name $NAME \
--img_transforms/brightness 0.1 \
--img_transforms/contrast 0.1 \
--img_transforms/saturation 0.6 \
--img_transforms/hue 0.1 \
--img_transforms/gradually_augment false \
--T_0 300 

FT_NAME=gca_like_VISION_ft
python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
--resume_ckpt_path '/hpc/scratch/dlehman/CD_checkpoints/'$NAME'/last_ckpt.pt' \
--resume_results_dir '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME \
--max_epochs 100 \
--img_transforms/contrast 0.5 \
--img_transforms/brightness 0.5 \
--lr_policy linear \
--lr 1e-5 \
--warmup_epochs 5 \
--experiment_name $FT_NAME 

python test.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$FT_NAME'/config.yaml' \
--test_sets 'v2 extra inter' 

python test.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
--test_sets 'v2 extra inter' 