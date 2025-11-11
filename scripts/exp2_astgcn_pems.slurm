#!/bin/bash

#SBATCH --job-name=pems_astgcn_exp3_batch64_run1_epoch100_seed3407_train2
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
echo "start pems_astgcn_exp3_batch64_run1_epoch100_seed3407_train2"

# VSF setting
echo "VSF setting"
python ../train_multi_step_exp.py \
    --data ../data/expData \
    --seed 3407 \
    --model_name astgcn \
    --device cuda:0 \
    --expid 3 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --lower_limit_random_node_selections 15 \
    --upper_limit_random_node_selections 15 \
    --useTrain1forTraining False \
    --step_size1 2500 >../logs/pems_astgcn_exp3_batch64_run1_epoch100_seed3407_original_train2.out

# oracle setting
echo "oracle setting"
python ../train_multi_step_exp.py \
    --data ../data/expData \
    --seed 3407 \
    --model_name astgcn \
    --device cuda:0 \
    --expid 3 \
    --epochs 0 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --lower_limit_random_node_selections 100 \
    --upper_limit_random_node_selections 100 \
    --do_full_set_oracle true \
    --full_set_oracle_lower_limit 15 \
    --full_set_oracle_upper_limit 15 \
    --useTrain1forTraining False \
    --step_size1 2500 >../logs/pems_astgcn_exp3_batch64_run10_epoch100_seed3407_oracle_train2.out

# fdw setting
echo "fdw setting"
python ../train_multi_step_exp.py \
    --data ../data/expData \
    --seed 3407 \
    --model_name astgcn \
    --device cuda:0 \
    --expid 3 \
    --epochs 0 \
    --batch_size 64 \
    --runs 1 \
    --lower_limit_random_node_selections 15 \
    --upper_limit_random_node_selections 15 \
    --borrow_from_train_data true \
    --num_neighbors_borrow 5 \
    --dist_exp_value 0.5 \
    --neighbor_temp 0.1 \
    --use_ewp True \
    --useTrain1forTraining False \
    --step_size1 2500 >../logs/pems_astgcn_exp3_batch64_run1_epoch100_seed3407_fdw_train2.out