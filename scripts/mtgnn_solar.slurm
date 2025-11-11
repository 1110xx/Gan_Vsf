#!/bin/bash

#SBATCH --job-name=solar_mtgnn_exp5_batch64_run1_epoch200_seed3407_tcn
#SBATCH --partition=information-gpu-0405
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -w g04
#SBATCH --gres=gpu:1
#SBATCH --output=../results/%j.out
#SBATCH --error=../results/%j.err

source activate vsf_lrc

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_JOB_NODELIST: " $SLURM_JOB_NODELIST
echo "start solar_mtgnn_exp5_batch64_run1_epoch200_seed3407_tcn"
echo "对齐完整变量的time attention 之后直接传入head, 不做cross attention, 取消align; early stopping; input original subset to backbone"
echo "mid_channels: 1024; add a dropout"

python ../train_multi_step.py \
    --data ../data/SOLAR \
    --seed 3407 \
    --model_name mtgnn \
    --device cuda:0 \
    --expid 5 \
    --epochs 200 \
    --batch_size 64 \
    --runs 1 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.5 \
    --w_fc 0.5 \
    --patience 20 \
    --mid_channels 1024 \
    --step_size1 2500 >../logs/solar_mtgnn_exp5_batch64_run1_epoch200_seed3407_tcn.out

echo "end success!"
