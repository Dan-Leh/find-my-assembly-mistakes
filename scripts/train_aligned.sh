#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N test_noam_unaligned
#PBS -M dan.lehman@asml.com
#PBS -m e

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

for NAME in noam_final
do
    # python train.py \
    # --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/gca_final/config.yaml' \
    # --experiment_name $NAME \
    # --cyws/attention noam \
    # --resume_ckpt_path /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/aarchive_fails_w_QD0.1/aligned_noam/checkpoints/last_ckpt.pt \
    # --resume_results_dir /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/aarchive_fails_w_QD0.1/aligned_noam

    python test.py \
    --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
    --test_sets 'extra new_parts v2' \
    --log_iter 500 \
    --save_fig_iter 20

    
    python test.py \
    --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
    --test_sets 'extra new_parts v2' \
    --log_iter 500 \
    --save_fig_iter 20 \
    --img_transforms/center_roi True

    python test_dirty_imgs.py \
    --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
    --test_sets 'rand_background' \
    --log_iter 500 \
    --save_fig_iter 20 
done