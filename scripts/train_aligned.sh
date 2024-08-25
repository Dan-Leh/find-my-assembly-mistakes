#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N aligned
#PBS -M dan.lehman@asml.com
#PBS -m e

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net


python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/cfg_perfect_alignment.yaml' \
--experiment_name aligned_noam \
--cyws/attention noam 

python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/cfg_perfect_alignment.yaml' \
--experiment_name aligned_gca \
--cyws/attention noam 

python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/cfg_perfect_alignment.yaml' \
--experiment_name aligned_noam50%randbg \
--cyws/attention noam \
--img_transforms/frac_random_background 0.5

python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/cfg_perfect_alignment.yaml' \
--experiment_name aligned_gca50%randbg \
--cyws/attention noam \
--img_transforms/frac_random_background 0.5


for NAME in  aligned_noam aligned_gca aligned_noam50%randbg aligned_gca50%randbg
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