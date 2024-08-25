#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N ft_fda50
#PBS -M dan.lehman@asml.com
#PBS -m e

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

NAME=gca_fda50%_moreaug
FT_NAME=ft_fda50%_moreaug_w_shear_randbg
# python train.py \
# --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
# --resume_ckpt_path '/hpc/scratch/dlehman/CD_checkpoints/'$NAME'/last_ckpt.pt' \
# --resume_results_dir '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME \
# --max_epochs 100 \
# --lr_policy linear \
# --lr 1e-5 \
# --warmup_epochs 5 \
# --experiment_name $FT_NAME \
# --img_transforms/gradually_augment false \
# --img_transforms/frac_random_background 0.3 \
# --img_transforms/shear 12

python test.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$FT_NAME'/config.yaml' \
--experiment_name $FT_NAME \
--test_sets 'v2 extra inter' 

python test_dirty_imgs.py \
--experiment_name $FT_NAME \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$FT_NAME'/config.yaml' \
--test_sets 'rand_background' 

# gca_fda50%_moreaug