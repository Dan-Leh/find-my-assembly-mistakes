#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:gpu_type=v100:mem=32gb:dcloc=htc
#PBS -N test

source activate /hpc/data/hpc-smc-internships/dlehman/python_envs/new

cd /shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net

for NAME in lca
do
    python test.py \
    --config '/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/'$NAME'/config.yaml' \
    --test_sets 'v2 extra' \
    --log_iter 500 \
    --save_fig_iter 20
done
