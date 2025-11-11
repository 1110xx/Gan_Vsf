#!/bin/bash

#SBATCH --job-name=traffic_astgcn_exp5_batch64_run1_epoch100_seed3407_TCN
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
echo "traffic_astgcn_exp5_batch64_run1_epoch100_seed3407_TCN"
echo "exp 5-10 use eval_subset for validation; mid_channels: 1024"

echo "--w_imp 0.0 --w_fc 1.0"
python ../train_multi_step.py \
    --data ../data/TRAFFIC \
    --seed 3407 \
    --model_name astgcn \
    --device cuda:0 \
    --expid 5 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.0 \
    --w_fc 1.0  \
    --patience 40 \
    --mid_channels 1024 \
    --useTCN True \
    --useCNN False \
    --step_size1 1000 >../logs/traffic_astgcn_exp5_batch64_run1_epoch100_seed3407_tcn_i0f10.out

echo "--w_imp 0.1 --w_fc 0.9"
python ../train_multi_step.py \
    --data ../data/TRAFFIC \
    --seed 3407 \
    --model_name astgcn \
    --device cuda:0 \
    --expid 6 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.1 \
    --w_fc 0.9  \
    --patience 40 \
    --mid_channels 1024 \
    --useTCN True \
    --useCNN False \
    --step_size1 1000 >../logs/traffic_astgcn_exp6_batch64_run1_epoch100_seed3407_tcn_i1f9.out

echo "--w_imp 0.3 --w_fc 0.7"
python ../train_multi_step.py \
    --data ../data/TRAFFIC \
    --seed 3407 \
    --model_name astgcn \
    --device cuda:0 \
    --expid 7 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.3 \
    --w_fc 0.7  \
    --patience 40 \
    --mid_channels 1024 \
    --useTCN True \
    --useCNN False \
    --step_size1 1000 >../logs/traffic_astgcn_exp7_batch64_run1_epoch100_seed3407_tcn_i3f7.out

echo "--w_imp 0.7 --w_fc 0.3"
python ../train_multi_step.py \
    --data ../data/TRAFFIC \
    --seed 3407 \
    --model_name astgcn \
    --device cuda:0 \
    --expid 8 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.7 \
    --w_fc 0.3  \
    --patience 40 \
    --mid_channels 1024 \
    --useTCN True \
    --useCNN False \
    --step_size1 1000 >../logs/traffic_astgcn_exp8_batch64_run1_epoch100_seed3407_tcn_i7f3.out

echo "--w_imp 0.9 --w_fc 0.1"
python ../train_multi_step.py \
    --data ../data/TRAFFIC \
    --seed 3407 \
    --model_name astgcn \
    --device cuda:0 \
    --expid 9 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.9 \
    --w_fc 0.1  \
    --patience 40 \
    --mid_channels 1024 \
    --useTCN True \
    --useCNN False \
    --step_size1 1000 >../logs/traffic_astgcn_exp9_batch64_run1_epoch100_seed3407_tcn_i9f1.out

echo "--w_imp 1.0 --w_fc 0.0"
python ../train_multi_step.py \
    --data ../data/TRAFFIC \
    --seed 3407 \
    --model_name astgcn \
    --device cuda:0 \
    --expid 10 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 1.0 \
    --w_fc 0.0  \
    --patience 40 \
    --mid_channels 1024 \
    --useTCN True \
    --useCNN False \
    --step_size1 1000 >../logs/traffic_astgcn_exp10_batch64_run1_epoch100_seed3407_tcn_i10f0.out

# echo "--w_imp 0.5 --w_fc 0.5"
# python ../train_multi_step.py \
#     --data ../data/TRAFFIC \
#     --seed 3407 \
#     --model_name astgcn \
#     --device cuda:0 \
#     --expid 4 \
#     --epochs 100 \
#     --batch_size 64 \
#     --runs 2 \
#     --random_node_idx_split_runs 100 \
#     --w_imp 0.5 \
#     --w_fc 0.5  \
#     --patience 40 \
#     --mid_channels 1024 \
#     --useTCN False \
#     --useCNN true \
#     --step_size1 1000 >../logs/traffic_astgcn_exp4_batch64_run2_epoch100_seed3407_CNN.out

echo "end success!"
