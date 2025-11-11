#!/bin/bash

#SBATCH --job-name=newData_batch64_run10_epoch100_seed3407_orig
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
echo "start newData_batch64_run10_epoch100_seed3407_orig"

model_name=mtgnn
seed=3407 

for dataset in exchange ETTh1 ETTh2 ETTm1 ETTm2
do
    expid=1
    step_size1=600
    if [ "$dataset" == "exchange" ]; then
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

    echo "backbone: $model_name; dataset: $dataset; expid: $expid; step_size1: $step_size1"
    # VSF setting
    echo "VSF setting"
    python ../train_multi_step_original.py \
        --data '../data/'$dataset \
        --seed $seed \
        --model_name $model_name \
        --device cuda:0 \
        --expid $expid \
        --epochs 100 \
        --batch_size 64 \
        --runs 10 \
        --random_node_idx_split_runs 100 \
        --lower_limit_random_node_selections 15 \
        --upper_limit_random_node_selections 15 \
        --step_size1 $step_size1 >../logs/newData/$dataset'_'$model_name'_exp'$expid'_seed'$seed'_original'.out

    # oracle setting
    echo "oracle setting"
    python ../train_multi_step_original.py \
        --data '../data/'$dataset \
        --seed $seed \
        --model_name $model_name \
        --device cuda:0 \
        --expid $expid \
        --epochs 0 \
        --batch_size 64 \
        --runs 10 \
        --random_node_idx_split_runs 100 \
        --lower_limit_random_node_selections 100 \
        --upper_limit_random_node_selections 100 \
        --do_full_set_oracle true \
        --full_set_oracle_lower_limit 15 \
        --full_set_oracle_upper_limit 15 \
        --step_size1 $step_size1 >../logs/newData/$dataset'_'$model_name'_exp'$expid'_seed'$seed'_oracle'.out

    # fdw setting
    echo "fdw setting"
    python ../train_multi_step_original.py \
        --data '../data/'$dataset \
        --seed $seed \
        --model_name $model_name \
        --device cuda:0 \
        --expid 1 \
        --epochs 0 \
        --batch_size 64 \
        --runs 10 \
        --lower_limit_random_node_selections 15 \
        --upper_limit_random_node_selections 15 \
        --borrow_from_train_data true \
        --num_neighbors_borrow 5 \
        --dist_exp_value 0.5 \
        --neighbor_temp 0.1 \
        --use_ewp True \
        --step_size1 $step_size1 >../logs/newData/$dataset'_'$model_name'_exp'$expid'_seed'$seed'_fdw'.out

# expid = expid + 1
done

# partial setting
echo "partial setting"
for dataset in exchange ETTh1 ETTh2 ETTm1 ETTm2
do
    expid=1
    step_size1=600
    if [ "$dataset" == "exchange" ]; then
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
    echo "backbone: $model_name; dataset: $dataset; expid: $expid; step_size1: $step_size1"
    python ../train_multi_step_original.py \
        --data '../data/'$dataset \
        --seed $seed \
        --model_name $model_name \
        --device cuda:0 \
        --expid $expid \
        --epochs 100 \
        --batch_size 64 \
        --runs 10 \
        --random_node_idx_split_runs 100 \
        --lower_limit_random_node_selections 15 \
        --upper_limit_random_node_selections 15 \
        --mask_remaining True \
        --step_size1 $step_size1 >../logs/newData/$dataset'_'$model_name'_exp'$expid'_seed'$seed'_partial'.out

# expid = expid + 1
done

echo "end success!"