#!/bin/bash
# ==============================================================================
# METR-LA 数据集训练脚本
#
# METR-LA 数据集信息:
# - 207 个传感器节点（交通流量传感器）
# - 时间分辨率: 5 分钟
# - 预测任务: 给定历史 12 步 (1小时)，预测未来 12 步
# ==============================================================================

set -e

DATA_PATH="./data/METR-LA"
DEVICE="cuda"
EPOCHS=300

# METR-LA 参数设置
# 207 节点，15% 子集 = 31 个观测节点
NUM_NODES=207
SUBSET_RATIO=0.15
HIDDEN_DIM=128
NUM_SLOTS=32  # 可以适当增加，因为节点更多

echo "========================================"
echo "METR-LA 数据集训练"
echo "========================================"
echo "节点数: $NUM_NODES"
echo "观测节点: ~31 (15%)"
echo "预测任务: 12 步 → 12 步"
echo "========================================"

# 检查数据是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo ":x: 数据不存在，请先运行数据准备脚本:"
    echo "   python scripts/prepare_metr_la.py --download"
    echo "   或"
    echo "   python scripts/prepare_metr_la.py --input /path/to/metr-la.h5"
    exit 1
fi

mkdir -p logs

# ==============================================================================
# 训练实验
# ==============================================================================

echo ""
echo "===== 开始训练 ====="

python trian_perd.py \
    --data $DATA_PATH \
    --batch_size 32 \
    --device $DEVICE \
    --seq_in_len 12 \
    --seq_out_len 12 \
    --lr 1e-3 \
    --hidden_dim $HIDDEN_DIM \
    --num_slots $NUM_SLOTS \
    --subset_ratio $SUBSET_RATIO \
    --lambda_recon 0.5 \
    --use_decoder True \
    --pred_head_version 2 \
    --pred_n_layers 3 \
    --use_node_attn True \
    --lr_scheduler cosine \
    --num_epochs $EPOCHS \
    --save_dir ./checkpoints_metr_la4 \
    2>&1 | tee logs/metr_la4_train.log

# ==============================================================================
# 结果
# ==============================================================================
echo ""
echo "========================================"
echo "训练完成!"
echo "========================================"
grep -h "Best validation" logs/metr_la4_train.log | tail -1

echo ""
echo "模型保存在: ./checkpoints_metr_la4/"