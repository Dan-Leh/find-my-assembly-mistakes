#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N rand_bg_gca
#PBS -M dan.lehman@asml.com
#PBS -m e

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

NAME=gca_rand_bg_resume
python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/gca_rand_bg/config.yaml' \
--experiment_name $NAME \
--img_transforms/random_background true \
--fda/frac_imgs_w_fda 0 \
--resume_ckpt_path /hpc/scratch/dlehman/CD_checkpoints/gca_rand_bg/last_ckpt.pt \
--resume_results_dir /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/gca_rand_bg \
--max_epochs 67 \
--lr 1e-5 \
--T_0 67