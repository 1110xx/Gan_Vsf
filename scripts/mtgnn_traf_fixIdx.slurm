#!/bin/bash

#SBATCH --job-name=traffic_mtgnn_exp1_batch64_run3_epoch100_seed3407_fixIdx
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
echo "start traffic_mtgnn_exp1_batch64_run3_epoch100_seed3407_fixIdx"

python ../train_fixIdx.py \
    --data ../data/TRAFFIC \
    --seed 3407 \
    --model_name mtgnn \
    --device cuda:0 \
    --expid 1 \
    --epochs 100 \
    --batch_size 64 \
    --runs 3 \
    --random_node_idx_split_runs 100 \
    --pred_weight 0.5 \
    --imp_weight 0.5 \
    --step_size1 1000 >../logs/traffic_mtgnn_exp1_batch64_run3_epoch100_seed3407_fixIdx.out

echo "end success!"
