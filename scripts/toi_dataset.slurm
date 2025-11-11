#!/bin/bash

#SBATCH --job-name=toi_dataset_batch64_run10_epoch100_seed3407_tcn
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
echo "start toi_dataset_batch64_run10_epoch100_seed3407_tcn"

seed=3407
# model_name=tgcn
dataset=ETTh1
device='cuda:0'

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

for model_name in mtgnn astgcn mstgcn tgcn Informer
do
    expid=11
    echo "backbone: $model_name; dataset: $dataset; expid: $expid; step_size1: $step_size1"
    python ../train_multi_step.py \
        --data '../data/'$dataset \
        --seed $seed \
        --model_name $model_name \
        --device $device \
        --expid $expid \
        --epochs 100 \
        --batch_size 64 \
        --runs 1 \
        --random_node_idx_split_runs 100 \
        --w_imp 0.1 \
        --w_fc 0.9  \
        --patience 20 \
        --mid_channels 1024 \
        --useTCN True \
        --step_size1 $step_size1 >../logs/mainexp/$dataset'_'$model_name'_exp'$expid'_seed'$seed'_toi'.out

# expid=$((expid + 1))
done

echo "end success!"