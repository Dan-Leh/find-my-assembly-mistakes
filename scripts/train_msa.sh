#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N msa_final

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

MODEL=msa
NAME=$MODEL'_final'
python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/final_experiments_cfg.yaml' \
--experiment_name $NAME \
--resume_ckpt_path /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/$MODEL/checkpoints/last_ckpt.pt \
--resume_results_dir /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/$MODEL \
--cyws/attention $MODEL \

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