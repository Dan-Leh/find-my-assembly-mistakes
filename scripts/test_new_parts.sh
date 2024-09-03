#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N test

# source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

for NAME in aligned_gca aligned_noam aligned_gca30%randbg aligned_noam30%randbg # 'aligned_noam(1)' 
do
    python test.py \
    --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
    --test_sets 'new_parts' \
    --log_iter 500 \
    --save_fig_iter 20 \
    --img_transforms/center_roi true

python test.py \
    --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
    --test_sets 'new_parts' \
    --log_iter 500 \
    --save_fig_iter 20 \
    --img_transforms/center_roi false
done

# python test.py \
#     --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/lca_final/config.yaml' \
#     --test_sets 'new_parts' \
#     --log_iter 500 \
#     --save_fig_iter 20 \
#     --img_transforms/center_roi true

# for NAME in aligned_gca aligned_gca50%randbg aligned_noam aligned_noam50%randbg # 'aligned_noam(1)' 
# do
#     python test.py \
#     --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/aarchive_fails_w_QD0.1/'$NAME'/config.yaml' \
#     --test_sets 'v2' \
#     --log_iter 500 \
#     --save_fig_iter 20
# done

# for NAME in aligned_gca aligned_gca50%randbg aligned_noam aligned_noam50%randbg # 'aligned_noam(1)' 
# do
#     python test_dirty_imgs.py \
#     --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/aarchive_fails_w_QD0.1/'$NAME'/config.yaml' \
#     --test_sets 'rand_background' \
#     --log_iter 500 \
#     --save_fig_iter 20 
# done

