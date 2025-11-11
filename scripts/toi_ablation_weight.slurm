#!/bin/bash

#SBATCH --job-name=toi_weights_batch64_run10_epoch100_seed3407_tcn
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
echo "start toi_weights_batch64_run10_epoch100_seed3407_tcn"

seed=3407
# model_name=tgcn
dataset=ECG5000
lb=15
ub=15
w_fc=1.0
w_imp=$(echo "scale=2; 1.0 - $w_fc" | bc -l)
device='cuda:0'

expid=31

# 31-36: toi-ecg5000: weights ablation exp

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

for w_1 in 0.0 0.2 0.4 0.6 0.8 1.0
do
    w_fc=$w_1
    w_imp=$(echo "scale=2; 1.0 - $w_1" | bc -l)
    echo "w_fc: $w_fc; w_imp: $w_imp"
    for model_name in mtgnn astgcn mstgcn informer tgcn
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
            --step_size1 $step_size1 >../logs/ablation/weights/'weights_'$dataset'_'$model_name'_exp'$expid'_seed'$seed'_toi'.out
    done
    expid=$((expid + 1))
done

echo "end success!"