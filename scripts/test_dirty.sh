#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N test_dirty

# source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

for NAME in gca_w_shearing
do
    python test_dirty_imgs.py \
    --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
    --experiment_name $NAME \
    --test_sets 'rand_background' \
    # --checkpoint_dir /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net-extra/ckpts_from_vision/$NAME \
    --log_iter 500 \
    --save_fig_iter 20 \
    # --output_root /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/VISION_ckpts
done