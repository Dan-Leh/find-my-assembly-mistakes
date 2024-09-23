#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
<<<<<<< HEAD
#PBS -N roi50_gca_final
=======
#PBS -N gca
>>>>>>> find-my-assembly-mistakes/main

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

<<<<<<< HEAD
MODEL=gca
NAME=$MODEL'_final_roim50'
=======
NAME=gca_less_aug
>>>>>>> find-my-assembly-mistakes/main
python train.py \
--config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/config_files/final_experiments_cfg.yaml' \
--experiment_name $NAME \
<<<<<<< HEAD
--resume_ckpt_path /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/$MODEL/checkpoints/last_ckpt.pt \
--resume_results_dir /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/$MODEL \
--T_0 300 \
--save_ckpt_interval 20 \
--cyws/attention $MODEL \
--img_transforms/roi_margin 50

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
=======
--T_0 100 \
--T_mult 2 \
--cyws/attention gca
>>>>>>> find-my-assembly-mistakes/main
