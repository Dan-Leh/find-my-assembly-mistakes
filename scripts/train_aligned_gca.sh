#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N gca_aligned
#PBS -M dan.lehman@asml.com
#PBS -m e

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/cfg_perfect_alignment.yaml' \
--experiment_name aligned_gca \
--cyws/attention gca 

# python train.py \
# --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/cfg_perfect_alignment.yaml' \
# --experiment_name aligned_gca30%randbg \
# --cyws/attention gca \
# --img_transforms/frac_random_background 0.3


for NAME in aligned_gca
do
    python test.py \
    --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
    --test_sets 'v2 extra new_parts' \
    --log_iter 500 \
    --save_fig_iter 20

    python test_dirty_imgs.py \
    --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
    --test_sets 'rand_background' \
    --log_iter 500 \
    --save_fig_iter 20 
done