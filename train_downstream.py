#!/usr/bin/env python3
"""
下游预测任务训练脚本

使用新的 Encoder V4 + 改进的预测头

使用方法：
1. 从头训练（无预训练）：
   python train_downstream.py --data ./data/ETTh1 --num_epochs 50

2. 使用预训练模型：
   python train_downstream.py --data ./data/ETTh1 --pretrain_ckpt ./checkpoints_temporal_pretrain/best_model.pt

3. 比较不同预测头：
   python train_downstream.py --data ./data/ETTh1 --pred_head_type simple
   python train_downstream.py --data ./data/ETTh1 --pred_head_type cross_attn
   python train_downstream.py --data ./data/ETTh1 --pred_head_type v2

4. 使用自回归预测头 V4（类似 LLM 逐步生成）：
   python train_downstream.py --data ./data/ETTh1 --pred_head_type v4 --teacher_forcing_ratio 0.5
"""

import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as Func

from util import load_dataset, StandardScaler, masked_mae

# 新模型
from model.encoder_v4 import NodeAwareTemporalEncoder
# 预测头选项
from model.pred_decoder_v3 import CrossAttentionPredHead, SimplePredHead, compute_subset_pred_loss
from model.pred_decoder_v2 import TemporalPredHeadV2
from model.pred_decoder import TemporalPredHead
from model.pred_decoder_v4 import AutoRegressivePredHeadV4, compute_ar_pred_loss_v4
from model.pred_decoder_v5 import ResidualPredHead, StatefulCrossAttentionPredHead, compute_residual_pred_loss
from model.pred_decoder_v6 import create_pred_head_v6, compute_v6_pred_loss
# 增强损失函数
from model.temporal_variance_loss import compute_enhanced_pred_loss


def create_pred_head(args, device):
    """根据参数创建预测头"""
    if args.pred_head_type == 'cross_attn':
        return CrossAttentionPredHead(
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(device)
    elif args.pred_head_type == 'simple':
        return SimplePredHead(
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            dropout=0.1
        ).to(device)
    elif args.pred_head_type == 'v2':
        return TemporalPredHeadV2(
            hidden_dim=args.hidden_dim,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_layers=3,
            n_heads=4,
            kernel_size=3,
            dropout=0.1,
            use_node_attn=True
        ).to(device)
    elif args.pred_head_type == 'v1':
        return TemporalPredHead(
            hidden_dim=args.hidden_dim,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            kernel_size=3,
            dropout=0.1
        ).to(device)
    elif args.pred_head_type == 'v4' or args.pred_head_type == 'ar':
        return AutoRegressivePredHeadV4(
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(device)
    elif args.pred_head_type == 'v5' or args.pred_head_type == 'residual':
        #v5 预测头（残差连接）+GRU
        return ResidualPredHead(
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            use_residual=getattr(args, 'use_residual_pred', True)
        ).to(device)
    elif args.pred_head_type == 'v5_stateful' or args.pred_head_type == 'stateful':
        #v5 预测头（状态保持版）
        return StatefulCrossAttentionPredHead(
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(device)
    elif args.pred_head_type == 'v6':
        return create_pred_head_v6(
            head_type='hybrid',
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            device=device,
            use_cross_attn=getattr(args, 'use_cross_attn', True)
        ).to(device)
    else:
        raise ValueError(f"Unknown pred_head_type: {args.pred_head_type}")


def train_epoch(encoder, pred_head, dataloader, optimizer, args, epoch, scaler=None):
    """训练一个 epoch"""
    encoder.train()
    pred_head.train()

    dataloader.shuffle()

    pred_losses = []
    num_subset = max(1, int(args.num_nodes * args.subset_ratio))

    start_time = time.time()

    for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
        # x: (B, T, N, F) -> (B, F, N, T)
        x_full = torch.Tensor(x).to(args.device)
        x_full = x_full.transpose(1, 3)

        # y: (B, T, N, F) 或 (B, T, N) - 未来真值
        y_full = torch.Tensor(y).to(args.device)
        if y_full.dim() == 4:
            y_full = y_full[..., 0]  # 取第一个特征

        # 归一化 y
        if scaler is not None:
            y_full = (y_full - scaler['mean']) / scaler['std']
        y_full = y_full.transpose(1, 2)  # (B, N, T)

        B, C, N, T = x_full.shape

        # 随机采样子集
        idx_subset = np.random.choice(args.num_nodes, size=num_subset, replace=False)
        idx_subset = torch.tensor(idx_subset, device=args.device)

        # 提取子集数据
        x_subset = x_full[:, :, idx_subset, :]
        y_subset = y_full[:, idx_subset, :]

        optimizer.zero_grad()

        # ========== 前向传播 ==========
        h_all = encoder(x_subset, idx_subset)  # (B, D, N_all, T)

        # 最后一个时间步的值（v5残差）
        #x_full:(B,F,N,T)-> 最后一个时间步第一个特征
        x_last = x_full[:, 0:1, idx_subset, -1]  # (B, 1, N)
        #归一化
        if scaler is not None:
            x_last = (x_last - scaler['mean']) / scaler['std']

        # V4/AR 预测头需要传入真值用于 Teacher Forcing
        if args.pred_head_type in ['v4', 'ar']:
            pred_all = pred_head(
                h_all,
                y_true=y_full,  # 传入全集真值，pred_head 内部会处理
                teacher_forcing_ratio=args.teacher_forcing_ratio
            )
        elif args.pred_head_type in ['v5', 'residual', 'v5_stateful', 'stateful']:
            #
            pred_all = pred_head(h_all,x_last=x_last)  # (B, 1, N_all, T_out)    
        elif args.pred_head_type == 'v6':
            pred_all = pred_head(h_all,x_last=x_last, node_idx=idx_subset)
        else:
            pred_all = pred_head(h_all)  # (B, 1, N_all, T_out)

        # ========== 计算损失 ==========
        if args.use_enhanced_loss:
            loss_pred, loss_dict = compute_enhanced_pred_loss(
                pred_all, y_subset, idx_subset,
                lambda_base=args.lambda_base,
                lambda_diff=args.lambda_diff,
                lambda_corr=getattr(args, 'lambda_corr', 0.2),
                lambda_range=getattr(args, 'lambda_range', 0.1),
                # lambda_start=getattr(args, 'lambda_start', 0.5),
                lambda_var=args.lambda_var,
                loss_type=args.loss_fn
            )
        else:
            loss_pred = compute_subset_pred_loss(
                pred_all, y_subset, idx_subset,
                loss_fn=args.loss_fn, huber_delta=args.huber_delta
            )
            loss_dict = None

        # ========== 反向传播 ==========
        loss_pred.backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(pred_head.parameters()),
            max_norm=args.max_grad_norm
        )

        optimizer.step()

        pred_losses.append(loss_pred.item())

        if iter_idx % args.print_every == 0:
            if loss_dict:
                corr_str = f" Corr: {loss_dict.get('mean_corr', 0.0):.4f}" if 'mean_corr' in loss_dict else ""
                print(f"  Iter [{iter_idx:3d}/{dataloader.num_batch:3d}] "
                      f"Total: {loss_pred.item():.4f} "
                      f"Base: {loss_dict['base_loss']:.4f} "
                      f"Diff: {loss_dict['diff_loss']:.4f} "
                      f"Var: {loss_dict['var_loss']:.4f}{corr_str}")
            else:
                print(f"  Iter [{iter_idx:3d}/{dataloader.num_batch:3d}] "
                      f"Pred Loss: {loss_pred.item():.6f}")

    epoch_time = time.time() - start_time

    return {
        'loss_pred': np.mean(pred_losses),
        'epoch_time': epoch_time,
    }


def validate(encoder, pred_head, dataloader, args, scaler=None):
    """验证"""
    encoder.eval()
    pred_head.eval()

    val_mae_list = []
    horizon_mae_list = {h: [] for h in range(args.seq_out_len)}

    num_subset = max(1, int(args.num_nodes * args.subset_ratio))

    with torch.no_grad():
        for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
            x_full = torch.Tensor(x).to(args.device)
            x_full = x_full.transpose(1, 3)

            # y_real: 原始尺度的真值
            y_real = torch.Tensor(y).to(args.device)
            if y_real.dim() == 4:
                y_real = y_real[..., 0]

            # 随机采样子集
            idx_subset = np.random.choice(args.num_nodes, size=num_subset, replace=False)
            idx_subset = torch.tensor(idx_subset, device=args.device)

            x_subset = x_full[:, :, idx_subset, :]

            # 前向传播
            h_all = encoder(x_subset, idx_subset)

            # 最后一个时间步的值（v5残差）
            x_last = x_full[:, 0:1, idx_subset, -1]  # (B, 1, N)
            if scaler is not None:
                x_last_norm = (x_last - scaler['mean']) / scaler['std']
            else:
                x_last_norm = x_last
            # V4/AR 预测头推理时不使用 Teacher Forcing
            if args.pred_head_type in ['v4', 'ar']:
                pred_all = pred_head(h_all, y_true=None, teacher_forcing_ratio=0.0)
            elif args.pred_head_type in ['v5', 'residual', 'v5_stateful', 'stateful']:
                pred_all = pred_head(h_all,x_last=x_last_norm)  # (B, 1, N_all, T_out)
            elif args.pred_head_type == 'v6':
                pred_all = pred_head(h_all,x_last=x_last_norm, node_idx=idx_subset)
            else:
                pred_all = pred_head(h_all)

            # 提取子集预测
            pred_subset = pred_all[:, 0, idx_subset, :]  # (B, N_subset, T_out)

            # 逆变换到原始尺度
            if scaler is not None:
                pred_real = pred_subset * scaler['std'] + scaler['mean']
            else:
                pred_real = pred_subset

            # 提取子集真值
            y_subset_real = y_real[:, :, idx_subset]  # (B, T, N_subset)
            y_subset_real = y_subset_real.transpose(1, 2)  # (B, N_subset, T)

            # 计算 MAE
            mae_val, _ = masked_mae(pred_real, y_subset_real, null_val=0.0)
            val_mae_list.append(mae_val.item())

            # 按 horizon 计算
            for h in range(args.seq_out_len):
                pred_h = pred_real[:, :, h]
                y_h = y_subset_real[:, :, h]
                mae_h, _ = masked_mae(pred_h, y_h, null_val=0.0)
                horizon_mae_list[h].append(mae_h.item())

    mean_mae = np.mean(val_mae_list)
    horizon_mae = {h: np.mean(horizon_mae_list[h]) for h in range(args.seq_out_len)}

    return {
        'val_mae': mean_mae,
        'horizon_mae': horizon_mae,
    }


def train_loop(encoder, pred_head, train_loader, val_loader, args, scaler=None):
    """训练循环"""
    os.makedirs(args.save_dir, exist_ok=True)

    # 优化器
    params = list(encoder.parameters()) + list(pred_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    history = {
        'train_loss': [],
        'val_mae': [],
        'best_val_mae': float('inf'),
    }

    print("\n" + "=" * 80)
    print(" " * 20 + "下游预测任务训练")
    print("=" * 80)
    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    print(f"Num nodes: {args.num_nodes}")
    print(f"Subset ratio: {args.subset_ratio}")
    print(f"Pred head type: {args.pred_head_type}")
    print(f"Pretrain checkpoint: {args.pretrain_ckpt or 'None (从头训练)'}")
    print(f"\nModel parameters:")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Pred Head: {sum(p.numel() for p in pred_head.parameters()):,}")
    print("=" * 80 + "\n")

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 80)

        train_metrics = train_epoch(
            encoder, pred_head, train_loader, optimizer, args, epoch, scaler
        )

        val_metrics = validate(encoder, pred_head, val_loader, args, scaler)

        history['train_loss'].append(train_metrics['loss_pred'])
        history['val_mae'].append(val_metrics['val_mae'])

        scheduler.step()

        # 打印结果
        print(f"\n[Epoch {epoch} Summary]")
        print(f"  Train Pred Loss: {train_metrics['loss_pred']:.6f}")
        print(f"  Val MAE (masked, real scale): {val_metrics['val_mae']:.4f}")

        # 打印各 horizon 的 MAE
        horizons_to_show = [0, 2, 5, 11]
        horizon_str = ", ".join([
            f"H{h+1}:{val_metrics['horizon_mae'][h]:.3f}"
            for h in horizons_to_show if h < args.seq_out_len
        ])
        print(f"  Horizon MAE: [{horizon_str}]")
        print(f"  Time: {train_metrics['epoch_time']:.2f}s")

        # 保存最佳模型
        if val_metrics['val_mae'] < history['best_val_mae']:
            history['best_val_mae'] = val_metrics['val_mae']
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'pred_head_state_dict': pred_head.state_dict(),
                'val_mae': history['best_val_mae'],
                'args': vars(args),
            }, best_path)
            print(f"  → Best model saved! Val MAE: {history['best_val_mae']:.4f}")

        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'pred_head_state_dict': pred_head.state_dict(),
                'history': history,
                'args': vars(args),
            }, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    print("\n" + "=" * 80)
    print(" " * 25 + "训练完成!")
    print("=" * 80)
    print(f"Best validation MAE: {history['best_val_mae']:.4f}")

    return history
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main():
    parser = argparse.ArgumentParser(description='下游预测任务训练')

    # 数据参数
    parser.add_argument('--data', type=str, required=True, help='数据路径')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--in_dim', type=int, default=None)
    parser.add_argument('--seq_in_len', type=int, default=12)
    parser.add_argument('--seq_out_len', type=int, default=12)

    # Encoder 参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # 预测头参数
    parser.add_argument('--pred_head_type', type=str, default='cross_attn',
                        choices=['cross_attn', 'simple', 'v2', 'v1', 'v4', 'ar', 'v5', 'residual', 'v5_stateful', 'stateful', 'v6'],
                        help='预测头类型 (v4/ar: 自回归预测头), (v5/residual/v5_stateful/stateful: 带残差连接的预测头)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='Teacher Forcing 概率 (仅 v4/ar 预测头有效)')
    parser.add_argument('--use_residual_pred', type=str_to_bool, default=True,
                        help='是否在 v5 预测头中使用残差连接')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=2.0)

    # 损失函数
    parser.add_argument('--loss_fn', type=str, default='mae', choices=['mae', 'mse', 'huber'])
    parser.add_argument('--huber_delta', type=float, default=0.5)

    # 增强损失函数（解决预测过于平滑的问题）
    parser.add_argument('--use_enhanced_loss', type=str_to_bool, default=True,
                        help='是否使用增强损失函数')
    parser.add_argument('--lambda_base', type=float, default=1.0,
                        help='基础预测损失权重')
    parser.add_argument('--lambda_diff', type=float, default=0.5,
                        help='时序变化损失权重')
    parser.add_argument('--lambda_var', type=float, default=0.3,
                        help='方差匹配损失权重')
    parser.add_argument('--lambda_corr', type=float, default=0.2,
                        help='相关性损失权重')
    parser.add_argument('--lambda_range', type=float, default=0.1,
                        help='范围匹配损失权重')
    parser.add_argument('--lambda_start', type=float, default=0.5,
                        help='起始点匹配损失权重')
    parser.add_argument('--use_cross_attn', type=str_to_bool, default=True,
                        help='v6 预测头中是否使用交叉注意力机制')

    # 子集配置
    parser.add_argument('--subset_ratio', type=float, default=0.3)

    # 预训练权重
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--freeze_encoder', type=str_to_bool, default=False,
                        help='是否冻结 Encoder')

    # load_dataset 需要的参数
    parser.add_argument('--predefined_S', type=str_to_bool, default=False)
    parser.add_argument('--predefined_S_frac', type=int, default=15)

    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_downstream')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=50)

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # 加载数据
    print(f"Loading data from {args.data}...")
    dataloader_dict = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)

    train_loader = dataloader_dict['train_loader']
    val_loader = dataloader_dict['val_loader']
    scaler_obj = dataloader_dict['scaler']

    # 提取 scaler
    if hasattr(scaler_obj, 'mean'):
        scaler = {'mean': scaler_obj.mean, 'std': scaler_obj.std}
    elif isinstance(scaler_obj, dict) and 'mean' in scaler_obj:
        scaler = scaler_obj
    else:
        x_train = train_loader.xs
        scaler = {
            'mean': x_train[..., 0].mean(),
            'std': x_train[..., 0].std()
        }

    print(f"\n✓ Scaler: mean={scaler['mean']:.4f}, std={scaler['std']:.4f}")

    args.num_nodes = train_loader.num_nodes

    if args.in_dim is None:
        sample_x = train_loader.xs[0]
        args.in_dim = sample_x.shape[-1]

    print(f"\n✓ Data loaded:")
    print(f"  Num nodes: {args.num_nodes}")
    print(f"  Input dim: {args.in_dim}")
    print(f"  Train samples: {train_loader.size}")
    print(f"  Val samples: {val_loader.size}")

    # 创建 Encoder V4
    encoder = NodeAwareTemporalEncoder(
        num_nodes=args.num_nodes,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        device=args.device
    ).to(device)

    # 加载预训练权重
    if args.pretrain_ckpt is not None and os.path.isfile(args.pretrain_ckpt):
        ckpt = torch.load(args.pretrain_ckpt, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        print(f"✓ Loaded pretrained encoder from {args.pretrain_ckpt}")

        if args.freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
            print("  → Encoder frozen (will not be updated)")
    elif args.pretrain_ckpt is not None:
        print(f"Warning: pretrain_ckpt not found: {args.pretrain_ckpt}")

    # 创建预测头
    pred_head = create_pred_head(args, device)

    print(f"✓ Models created")

    # 训练
    history = train_loop(encoder, pred_head, train_loader, val_loader, args, scaler)

    return history


if __name__ == "__main__":
    main()