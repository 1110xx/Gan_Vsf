#!/bin/bash

#SBATCH --job-name=part_traf_tgcn_102_batch64_run10_epochs120
#SBATCH --partition=information-gpu-0405
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -w g05
#SBATCH --gres=gpu:1
#SBATCH --output=../results/%j.out
#SBATCH --error=../results/%j.err
#SBATCH --mail-type=end           # 作业结束时，邮件提醒
#SBATCH --mail-user=962102538@qq.com     # 邮箱地址

source activate vsf_lrc


echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_JOB_NODELIST: " $SLURM_JOB_NODELIST
echo "start partial traffic TGCN expid 102 batch 64 run 10 epochs 120"

python ../partial_run.py --data ../data/TRAFFIC --forecast_model TGCN --device cuda:0 --expid 102 --epochs 120 --batch_size 64 --runs 10 --step_size1 1000 --partial True >../results/part_traf_tgcn_102_bat64_run10_epoch120.out

echo "end success!"

