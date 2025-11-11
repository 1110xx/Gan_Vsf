#!/bin/bash

#SBATCH --job-name=ecg5000_tgcn_exp4_batch64_run1_epoch100_seed3407_tcn
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
echo "start ecg5000_tgcn_exp4_batch64_run1_epoch100_seed3407_tcn"

echo "--w_imp 0.0 --w_fc 1.0"
python ../train_multi_step.py \
    --data ../data/ECG5000 \
    --seed 3407 \
    --model_name tgcn \
    --device cuda:0 \
    --expid 4 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.0 \
    --w_fc 1.0 \
    --patience 100 \
    --mid_channels 1024 \
    --useTCN True \
    --step_size1 400 >../logs/ecg5000_tgcn_exp4_batch64_run1_epoch100_seed3407_tcn_i0f10.out

echo "--w_imp 0.1 --w_fc 0.9"
python ../train_multi_step.py \
    --data ../data/ECG5000 \
    --seed 3407 \
    --model_name tgcn \
    --device cuda:0 \
    --expid 5 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.1 \
    --w_fc 0.9 \
    --patience 100 \
    --mid_channels 1024 \
    --useTCN True \
    --step_size1 400 >../logs/ecg5000_tgcn_exp4_batch64_run1_epoch100_seed3407_tcn_i1f9.out

echo "--w_imp 0.3 --w_fc 0.7"
python ../train_multi_step.py \
    --data ../data/ECG5000 \
    --seed 3407 \
    --model_name tgcn \
    --device cuda:0 \
    --expid 6 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.3 \
    --w_fc 0.7 \
    --patience 100 \
    --mid_channels 1024 \
    --useTCN True \
    --step_size1 400 >../logs/ecg5000_tgcn_exp6_batch64_run1_epoch100_seed3407_tcn_i3f7.out

echo "--w_imp 0.7 --w_fc 0.3"
python ../train_multi_step.py \
    --data ../data/ECG5000 \
    --seed 3407 \
    --model_name tgcn \
    --device cuda:0 \
    --expid 7 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.7 \
    --w_fc 0.3 \
    --patience 100 \
    --mid_channels 1024 \
    --useTCN True \
    --step_size1 400 >../logs/ecg5000_tgcn_exp7_batch64_run1_epoch100_seed3407_tcn_i7f3.out

echo "--w_imp 0.9 --w_fc 0.1"
python ../train_multi_step.py \
    --data ../data/ECG5000 \
    --seed 3407 \
    --model_name tgcn \
    --device cuda:0 \
    --expid 8 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.9 \
    --w_fc 0.1 \
    --patience 100 \
    --mid_channels 1024 \
    --useTCN True \
    --step_size1 400 >../logs/ecg5000_tgcn_exp8_batch64_run1_epoch100_seed3407_tcn_i9f1.out

echo "--w_imp 1.0 --w_fc 0.0"
python ../train_multi_step.py \
    --data ../data/ECG5000 \
    --seed 3407 \
    --model_name tgcn \
    --device cuda:0 \
    --expid 9 \
    --epochs 100 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 1.0 \
    --w_fc 0.0 \
    --patience 100 \
    --mid_channels 1024 \
    --useTCN True \
    --step_size1 400 >../logs/ecg5000_tgcn_exp9_batch64_run1_epoch100_seed3407_tcn_i10f0.out

echo "end success!"


