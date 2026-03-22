#!/usr/bin/env bash
set -euo pipefail

# ====================== 全局配置 ======================
DEVICE="cuda"            # 可选: cuda, mps, cpu
RUNS=10                   # 外层重复训练次数
RANDOM_SPLIT_RUNS=100     # 每个 run 的随机子集测试次数
BASE_SEED=2024

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="../logs"
mkdir -p "${LOG_DIR}"

setup_dataset_config() {
    local dataset=$1

    case "$dataset" in
        etth1|ETTh1)
            DATASET_NAME="ETTh1"
            DATA_PATH="../data/ETTh1"
            IN_DIM=1
            HIDDEN_DIM=32
            N_LAYERS=3
            N_HEADS=4
            BATCH_SIZE=64
            SUBSET_RATIO=0.10
            NUM_EPOCHS_PRETRAIN=100
            NUM_EPOCHS_DOWNSTREAM=200
            LR_G=1e-4
            LR_D=5e-5
            LR_DOWNSTREAM=5e-4
            PATIENCE=10
            ;;
        ettm1|ETTm1)
            DATASET_NAME="ETTm1"
            DATA_PATH="../data/ETTm1"
            IN_DIM=1
            HIDDEN_DIM=32
            N_LAYERS=3
            N_HEADS=4
            BATCH_SIZE=64
            SUBSET_RATIO=0.15
            NUM_EPOCHS_PRETRAIN=100
            NUM_EPOCHS_DOWNSTREAM=200
            LR_G=1e-4
            LR_D=5e-5
            LR_DOWNSTREAM=5e-4
            PATIENCE=10
            ;;
        metr-la|METR-LA)
            DATASET_NAME="METR-LA"
            DATA_PATH="../data/METR-LA"
            IN_DIM=1
            HIDDEN_DIM=128
            N_LAYERS=4
            N_HEADS=8
            BATCH_SIZE=64
            SUBSET_RATIO=0.15
            NUM_EPOCHS_PRETRAIN=100
            NUM_EPOCHS_DOWNSTREAM=400
            LR_G=5e-5
            LR_D=2e-5
            LR_DOWNSTREAM=1e-4
            PATIENCE=20
            ;;
        ecg5000|ECG5000)
            DATASET_NAME="ECG5000"
            DATA_PATH="../data/ECG5000"
            IN_DIM=1
            HIDDEN_DIM=64
            N_LAYERS=4
            N_HEADS=4
            BATCH_SIZE=32
            SUBSET_RATIO=0.15
            NUM_EPOCHS_PRETRAIN=100
            NUM_EPOCHS_DOWNSTREAM=200
            LR_G=1e-4
            LR_D=5e-5
            LR_DOWNSTREAM=5e-4
            PATIENCE=20
        *)
            echo ":x: 未知数据集: $dataset"
            echo "支持的数据集: etth1, ettm1, metr-la，ecg5000"
            exit 1
            ;;
    esac

    SAVE_DIR_PRETRAIN="../history/checkpoints_all_pretrain_${DATASET_NAME}"
    SAVE_DIR_PRETRAIN_RUN="${SAVE_DIR_PRETRAIN}/${TIMESTAMP}"

    SAVE_DIR_DOWNSTREAM_BASE="../history/checkpoints_all_downstream_${DATASET_NAME}_multi/${TIMESTAMP}"
    SAVE_DIR_DOWNSTREAM_FROZEN="${SAVE_DIR_DOWNSTREAM_BASE}/frozen"
    SAVE_DIR_DOWNSTREAM_FINETUNE="${SAVE_DIR_DOWNSTREAM_BASE}/finetune"
    SAVE_DIR_DOWNSTREAM_SCRATCH="${SAVE_DIR_DOWNSTREAM_BASE}/scratch"

    RESULT_DIR_BASE="../multi_results/${DATASET_NAME}/${TIMESTAMP}"
    mkdir -p "${RESULT_DIR_BASE}"

    echo "=============================================="
    echo " 数据集配置: ${DATASET_NAME}"
    echo "=============================================="
    echo "  数据路径: ${DATA_PATH}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  子集比例: ${SUBSET_RATIO}"
    echo "  RUNS: ${RUNS}"
    echo "  RANDOM_SPLIT_RUNS: ${RANDOM_SPLIT_RUNS}"
    echo "=============================================="
}

run_pretrain() {
    local LOG_FILE="${LOG_DIR}/pretrain_${DATASET_NAME}_${TIMESTAMP}.log"

    echo ""
    echo "=============================================="
    echo " GAN 预训练 - ${DATASET_NAME}"
    echo "=============================================="

    python ../Gan.py \
        --data ${DATA_PATH} \
        --device ${DEVICE} \
        --batch_size ${BATCH_SIZE} \
        --in_dim ${IN_DIM} \
        --seq_in_len 12 \
        --seq_out_len 12 \
        --hidden_dim ${HIDDEN_DIM} \
        --n_layers ${N_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout 0.1 \
        --num_epochs ${NUM_EPOCHS_PRETRAIN} \
        --lr_g ${LR_G} \
        --lr_d ${LR_D} \
        --weight_decay 1e-4 \
        --lambda_temporal 1.0 \
        --lambda_spatial 0.5 \
        --lambda_rec 1.0 \
        --lambda_adv 0.1 \
        --subset_ratio ${SUBSET_RATIO} \
        --use_global_recon True \
        --lambda_obs 0.3 \
        --use_latent_dyn True \
        --lambda_latent_dyn 0.1 \
        --latent_dyn_version v2 \
        --disc_alpha 0.7 \
        --max_grad_norm_g 2.0 \
        --max_grad_norm_d 1.0 \
        --save_dir ${SAVE_DIR_PRETRAIN_RUN} \
        --save_interval 10 \
        --print_every 20 \
        --seed ${BASE_SEED} \
        2>&1 | tee "${LOG_FILE}"

    if [ -f "${SAVE_DIR_PRETRAIN_RUN}/best_model.pt" ]; then
        mkdir -p "${SAVE_DIR_PRETRAIN}"
        cp "${SAVE_DIR_PRETRAIN_RUN}/best_model.pt" "${SAVE_DIR_PRETRAIN}/best_model.pt"
    fi

    echo "✓ 预训练完成! ${SAVE_DIR_PRETRAIN}/best_model.pt"
}

run_test_pretrain() {
    local LOG_FILE="${LOG_DIR}/test_pretrain_${DATASET_NAME}_${TIMESTAMP}.log"
    local MODEL_PATH="${SAVE_DIR_PRETRAIN}/best_model.pt"
    local VIZ_DIR="../viz_all_pretrain_${DATASET_NAME}"

    if [ ! -f "${MODEL_PATH}" ]; then
        echo ":x: 模型文件不存在: ${MODEL_PATH}"
        return 1
    fi

    python ../test_gan.py \
        --data ${DATA_PATH} \
        --model_path ${MODEL_PATH} \
        --device ${DEVICE} \
        --batch_size ${BATCH_SIZE} \
        --in_dim ${IN_DIM} \
        --seq_in_len 12 \
        --seq_out_len 12 \
        --hidden_dim ${HIDDEN_DIM} \
        --n_layers ${N_LAYERS} \
        --n_heads ${N_HEADS} \
        --dropout 0.1 \
        --subset_ratio ${SUBSET_RATIO} \
        --num_batches 20 \
        --save_viz True \
        --viz_dir ${VIZ_DIR} \
        2>&1 | tee "${LOG_FILE}"

    echo "✓ 预训练测试完成"
}

run_downstream_once() {
    local mode=$1
    local run_idx=$2
    local save_root=$3

    local FREEZE_ENCODER="False"
    local LR=${LR_DOWNSTREAM}
    local USE_PRETRAIN="True"
    local PRETRAIN_CKPT="${SAVE_DIR_PRETRAIN}/best_model.pt"

    case "$mode" in
        frozen)
            FREEZE_ENCODER="True"
            LR="1e-3"
            ;;
        finetune)
            FREEZE_ENCODER="False"
            ;;
        scratch)
            USE_PRETRAIN="False"
            PRETRAIN_CKPT=""
            ;;
    esac

    local run_seed=$((BASE_SEED + run_idx - 1))
    local run_dir="${save_root}/run_${run_idx}"
    local LOG_FILE="${LOG_DIR}/downstream_${mode}_${DATASET_NAME}_${TIMESTAMP}_run${run_idx}.log"

    mkdir -p "${run_dir}"

    local CMD="python ../Forecast.py \
        --data ${DATA_PATH} \
        --device ${DEVICE} \
        --batch_size ${BATCH_SIZE} \
        --in_dim ${IN_DIM} \
        --seq_in_len 12 \
        --seq_out_len 12 \
        --hidden_dim ${HIDDEN_DIM} \
        --n_layers ${N_LAYERS} \
        --n_heads ${N_HEADS} \
        --pred_head_type v8 \
        --pred_n_layers 4 \
        --tcn_kernel_size 3 \
        --lambda_smooth 0 \
        --use_residual_pred False \
        --num_epochs ${NUM_EPOCHS_DOWNSTREAM} \
        --lr ${LR} \
        --subset_ratio ${SUBSET_RATIO} \
        --freeze_encoder ${FREEZE_ENCODER} \
        --use_clean_obs True \
        --loss_fn mae \
        --print_every 20 \
        --seed ${run_seed} \
        --save_dir ${run_dir} \
        --early_stop_patience ${PATIENCE} "

    if [ "${USE_PRETRAIN}" = "True" ] && [ -f "${PRETRAIN_CKPT}" ]; then
        CMD="${CMD} --pretrain_ckpt ${PRETRAIN_CKPT}"
    fi

    echo "[${mode}] Run ${run_idx}/${RUNS} ..."
    eval "${CMD}" 2>&1 | tee "${LOG_FILE}"
}

run_downstream_multirun_mode() {
    local mode=$1
    local mode_root=""
    local LOG_FILE="${LOG_DIR}/test_downstream_multirun_${mode}_${DATASET_NAME}_${TIMESTAMP}.log"

    case "$mode" in
        frozen)
            mode_root="${SAVE_DIR_DOWNSTREAM_FROZEN}"
            ;;
        finetune)
            mode_root="${SAVE_DIR_DOWNSTREAM_FINETUNE}"
            ;;
        scratch)
            mode_root="${SAVE_DIR_DOWNSTREAM_SCRATCH}"
            ;;
        *)
            echo ":x: 未知模式: ${mode}"
            return 1
            ;;
    esac

    mkdir -p "${mode_root}"

    echo ""
    echo "=============================================="
    echo " 下游训练 + 多次测试 (${mode}) - ${DATASET_NAME}"
    echo "=============================================="

    for run_idx in $(seq 1 ${RUNS}); do
        run_downstream_once "${mode}" "${run_idx}" "${mode_root}"
    done

    python ../test.py \
        --data ${DATA_PATH} \
        --device ${DEVICE} \
        --mode ${mode} \
        --batch_size ${BATCH_SIZE} \
        --seq_in_len 12 \
        --seq_out_len 12 \
        --in_dim ${IN_DIM} \
        --runs ${RUNS} \
        --random_node_idx_split_runs ${RANDOM_SPLIT_RUNS} \
        --seed ${BASE_SEED} \
        --model_glob "${mode_root}/run_*/best_model.pt" \
        --lower_limit_random_node_selections 15 \
        --upper_limit_random_node_selections 15 \
        --output_dir "${RESULT_DIR_BASE}" \
        2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "[${mode}] Final 汇总文件: ${RESULT_DIR_BASE}/${mode}/summary.json"
}

run_all() {
    run_pretrain
    run_test_pretrain

    run_downstream_multirun_mode frozen
    run_downstream_multirun_mode finetune
    run_downstream_multirun_mode scratch

    echo ""
    echo "=============================================="
    echo " 完整流程完成: ${DATASET_NAME}"
    echo " 结果目录: ${RESULT_DIR_BASE}"
    echo "=============================================="
}

run_all_datasets() {
    local action=${1:-"all"}
    for ds in etth1 ettm1 metr-la; do
        setup_dataset_config "$ds"
        case "$action" in
            all)
                run_all
                ;;
            downstream_multi)
                run_downstream_multirun_mode frozen
                run_downstream_multirun_mode finetune
                run_downstream_multirun_mode scratch
                ;;
            pretrain)
                run_pretrain
                ;;
            test_pretrain)
                run_test_pretrain
                ;;
            *)
                run_all
                ;;
        esac
    done
}

print_usage() {
    echo "用法: $0 <dataset> <action> [runs] [random_split_runs]"
    echo ""
    echo "dataset: etth1 | ettm1 | metr-la | all_datasets"
    echo "action:"
    echo "  pretrain"
    echo "  test_pretrain"
    echo "  downstream_multi_frozen"
    echo "  downstream_multi_finetune"
    echo "  downstream_multi_scratch"
    echo "  downstream_multi            # 三模式"
    echo "  all"
    echo ""
    echo "可选覆盖参数:"
    echo "  runs                默认 ${RUNS}"
    echo "  random_split_runs   默认 ${RANDOM_SPLIT_RUNS}"
}

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

DATASET=$1
ACTION=${2:-"all"}
if [ $# -ge 3 ]; then RUNS=$3; fi
if [ $# -ge 4 ]; then RANDOM_SPLIT_RUNS=$4; fi

if [ "$DATASET" = "all_datasets" ]; then
    run_all_datasets "$ACTION"
    exit 0
fi

setup_dataset_config "$DATASET"

case "$ACTION" in
    pretrain)
        run_pretrain
        ;;
    test_pretrain)
        run_test_pretrain
        ;;
    downstream_multi_frozen)
        run_downstream_multirun_mode frozen
        ;;
    downstream_multi_finetune)
        run_downstream_multirun_mode finetune
        ;;
    downstream_multi_scratch)
        run_downstream_multirun_mode scratch
        ;;
    downstream_multi)
        run_downstream_multirun_mode frozen
        run_downstream_multirun_mode finetune
        run_downstream_multirun_mode scratch
        ;;
    all)
        run_all
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo " 执行完成"
echo "=============================================="
