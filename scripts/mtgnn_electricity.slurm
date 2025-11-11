#!/bin/bash

#SBATCH --job-name=electricity_mtgnn_exp5_batch64_run3_epoch200_seed3407_df100_usetcn
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
echo "start electricity_mtgnn_exp5_batch64_run3_epoch200_seed3407_df100_usetcn"
echo "time attention + cnn 之后直接传入head, 不做cross attention, 取消align; df / 100"
echo "use tcn; mid_channels: 1024; add a dropout; input original subset to backbone"

python ../train_multi_step.py \
    --data ../data/electricity \
    --seed 3407 \
    --model_name mtgnn \
    --device cuda:0 \
    --expid 5 \
    --epochs 200 \
    --batch_size 64 \
    --runs 3 \
    --random_node_idx_split_runs 100 \
    --w_imp 0.5 \
    --w_fc 0.5 \
    --patience 20 \
    --mid_channels 1024 \
    --step_size1 1500 >../logs/electricity_mtgnn_exp5_batch64_run3_epoch200_seed3407_df100_usetcn.out

echo "end success!"
