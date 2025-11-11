#!/bin/bash

#SBATCH --job-name=ecg5000_mtgnn_exp1_batch64_run2_epoch100_seed3407_tcn
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
echo "start ecg5000_mtgnn_exp1_batch64_run2_epoch100_seed3407_tcn"
echo "对齐完整变量的time attention + tcn 之后直接传入head, 不做cross attention, 取消align"
echo "mid_channels: 1024; add a dropout; early stopping; input original subset to backbone"

python ../train_multi_step.py \
    --data ../data/ECG5000 \
    --seed 3407 \
    --model_name mtgnn \
    --device cuda:0 \
    --expid 1 \
    --epochs 100 \
    --batch_size 64 \
    --runs 2 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.5 \
    --w_fc 0.5  \
    --patience 20 \
    --mid_channels 1024 \
    --useTCN True \
    --step_size1 400 >../logs/ecg5000_mtgnn_exp1_batch64_run2_epoch100_seed3407_tcn.out

echo "end success!"
