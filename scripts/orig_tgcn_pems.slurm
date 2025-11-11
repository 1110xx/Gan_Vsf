#!/bin/bash

#SBATCH --job-name=pems_tgcn_exp1_batch64_run10_epoch100_seed3407_fdw
#SBATCH --partition=information-gpu-0405
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -w g05
#SBATCH --gres=gpu:1
#SBATCH --output=../results/%j.out
#SBATCH --error=../results/%j.err

source activate vsf_lrc

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_JOB_NODELIST: " $SLURM_JOB_NODELIST
echo "start pems_tgcn_exp1_batch64_run10_epoch100_seed3407_fdw"

# # VSF setting
# echo "VSF setting"
# python ../train_multi_step_original.py \
#     --data ../data/PEMS-BAY \
#     --seed 3407 \
#     --model_name tgcn \
#     --device cuda:0 \
#     --expid 1 \
#     --epochs 100 \
#     --batch_size 64 \
#     --runs 10 \
#     --random_node_idx_split_runs 100 \
#     --lower_limit_random_node_selections 15 \
#     --upper_limit_random_node_selections 15 \
#     --step_size1 2500 >../logs/pems_tgcn_exp1_batch64_run10_epoch100_seed3407_original.out

# # oracle setting
# echo "oracle setting"
# python ../train_multi_step_original.py \
#     --data ../data/PEMS-BAY \
#     --seed 3407 \
#     --model_name tgcn \
#     --device cuda:0 \
#     --expid 1 \
#     --epochs 0 \
#     --batch_size 64 \
#     --runs 10 \
#     --random_node_idx_split_runs 100 \
#     --lower_limit_random_node_selections 100 \
#     --upper_limit_random_node_selections 100 \
#     --do_full_set_oracle true \
#     --full_set_oracle_lower_limit 15 \
#     --full_set_oracle_upper_limit 15 \
#     --step_size1 2500 >../logs/pems_tgcn_exp1_batch64_run10_epoch100_seed3407_oracle.out

# fdw setting
echo "fdw setting"
python ../train_multi_step_original.py \
    --data ../data/PEMS-BAY \
    --seed 3407 \
    --model_name tgcn \
    --device cuda:0 \
    --expid 1 \
    --epochs 0 \
    --batch_size 64 \
    --runs 10 \
    --lower_limit_random_node_selections 15 \
    --upper_limit_random_node_selections 15 \
    --borrow_from_train_data true \
    --num_neighbors_borrow 5 \
    --dist_exp_value 0.5 \
    --neighbor_temp 0.1 \
    --use_ewp True \
    --step_size1 2500 >../logs/pems_tgcn_exp1_batch64_run10_epoch100_seed3407_fdw.out

# # partial setting
# echo "partial setting"
# python ../train_multi_step_original.py \
#     --data ../data/PEMS-BAY \
#     --seed 3407 \
#     --model_name tgcn \
#     --device cuda:0 \
#     --expid 2 \
#     --epochs 100 \
#     --batch_size 64 \
#     --runs 10 \
#     --random_node_idx_split_runs 100 \
#     --lower_limit_random_node_selections 15 \
#     --upper_limit_random_node_selections 15 \
#     --mask_remaining True \
#     --step_size1 2500 >../logs/pems_tgcn_exp2_batch64_run10_epoch100_seed3407_partial.out

echo "end success!"