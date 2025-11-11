#!/bin/bash

#SBATCH --job-name=k_toi_batch64_run10_epoch100_seed3407_tcn
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
echo "start k_toi_batch64_run10_epoch100_seed3407_tcn"

seed=3407
# model_name=tgcn
dataset=ECG5000
runs=1
lb=5
ub=5
expid=21
w_fc=0.9
w_imp=$(echo "scale=2; 1.0 - $w_fc" | bc -l)
device='cuda:0'

# 21-28: toi-ecg5000: S/N ablation exp

step_size1=0
if [ "$dataset" == "ECG5000" ]; then
    step_size1=400
elif [ "$dataset" == "SOLAR" ]; then
    step_size1=2500
elif [ "$dataset" == "TRAFFIC" ]; then
    step_size1=1000
elif [ "$dataset" == "METR-LA" ]; then
    step_size1=2500
elif [ "$dataset" == "PEMS-BAY" ]; then
    step_size1=2500
elif [ "$dataset" == "exchange" ]; then
    step_size1=600
elif [ "$dataset" == "ETTh1" ]; then
    step_size1=1400
elif [ "$dataset" == "ETTh2" ]; then
    step_size1=1400
elif [ "$dataset" == "ETTm1" ]; then
    step_size1=5800
elif [ "$dataset" == "ETTm2" ]; then
    step_size1=5800
fi

for k in 5 10 20 25 30 35 40 45
# for k in 30 35 40 45
# for k in 5 10 20 25
do
    lb=$k
    ub=$k
    echo "lb: $lb; ub: $ub"
    for model_name in mtgnn astgcn mstgcn tgcn Informer
    do
        echo "backbone: $model_name; dataset: $dataset; expid: $expid; step_size1: $step_size1"
        python ../train_multi_step.py \
            --data '../data/'$dataset \
            --seed $seed \
            --model_name $model_name \
            --device $device \
            --expid $expid \
            --epochs 150 \
            --batch_size 64 \
            --runs 1 \
            --random_node_idx_split_runs 100 \
            --w_imp $w_imp \
            --w_fc $w_fc  \
            --patience 20 \
            --mid_channels 1024 \
            --useTCN True \
            --lower_limit_random_node_selections $lb  \
            --upper_limit_random_node_selections $ub  \
            --step_size1 $step_size1 >../logs/ablation/k/'k_'$ub'_'$dataset'_'$model_name'_exp'$expid'_seed'$seed'_toi'.out
    done
    expid=$((expid + 1))
done
echo "end success!"
