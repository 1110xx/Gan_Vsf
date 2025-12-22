import os
import sys
import time
import argparse
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import load_dataset, StandardScaler

from model.encoder_v3 import SlotBasedEncoder
from model.decoder_v2 import STDecoder
from model.pred_decoder import TemporalPredHead, compute_subset_pred_loss


def train_epoch(encoder, pred_head, decoder,dataloader, optimizer, args, epoch):
    """
    训练一个 epoch

    Args:
        encoder: SlotBasedEncoder
        pred_head: TemporalPredHead
        decoder: STDecoder (用于重构，可选)
        dataloader: 数据加载器
        optimizer: 优化器
        args: 参数
        epoch: 当前 epoch
    """
    encoder.train()
    pred_head.train()
    if decoder is not None:
        decoder.train()

    dataloader.shuffle()

    pred_losses = []
    recon_losses = []
    total_losses = []

    num_subset = int(args.num_nodes * args.subset_ratio)

    # 预测 loss 权重 warmup
    if args.pred_warmup_epochs > 0:
        warmup_ratio = min(1.0, epoch / args.pred_warmup_epochs)
        lambda_pred = args.lambda_pred_min + (args.lambda_pred - args.lambda_pred_min) * warmup_ratio
    else:
        lambda_pred = args.lambda_pred

    start_time = time.time()

    for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
        # x: (B, T, N, F) -> (B, F, N, T)
        x_full = torch.Tensor(x).to(args.device)
        x_full = x_full.transpose(1, 3)  # (B, F, N, T)

        # y: (B, T, N) - 未来真值
        y_full = torch.Tensor(y).to(args.device)
        y_full = y_full.transpose(1, 2)  # (B, N, T)

        B, F, N, T = x_full.shape

        # 随机采样子集
        idx_subset = np.random.choice(args.num_nodes, size=num_subset, replace=False)
        idx_subset = torch.tensor(idx_subset, device=args.device)

        # 提取子集数据
        x_subset = x_full[:, :, idx_subset, :]  # (B, F, N_obs, T)
        y_subset = y_full[:, idx_subset, :]     # (B, N_obs, T_out)

        optimizer.zero_grad()

        # ========== 前向传播 ==========
        # Encoder: 子集 -> 全集 embedding
        h_all = encoder(x_subset, idx_subset)  # (B, D, N_all, T)

        # Pred Head: embedding -> 全集预测
        pred_all = pred_head(h_all)  # (B, 1, N_all, T_out)

        # ========== 计算预测损失 ==========
        # 只在子集上计算预测 loss
        loss_pred = compute_subset_pred_loss(
            pred_all, y_subset, idx_subset, loss_fn='mae'
        )

        # ========== 计算重构损失 (可选) ==========
        if decoder is not None and args.lambda_recon > 0:
            recon_all = decoder(h_all)  # (B, F, N_all, T)
            loss_recon = F.mse_loss(recon_all, x_full)
        else:
            loss_recon = torch.tensor(0.0, device=args.device)

        # ========== 总损失 ==========
        loss_total = lambda_pred * loss_pred + args.lambda_recon * loss_recon

        # ========== 反向传播 ==========
        loss_total.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(pred_head.parameters()),
            max_norm=args.max_grad_norm
        )

        optimizer.step()

        # 记录
        pred_losses.append(loss_pred.item())
        recon_losses.append(loss_recon.item())
        total_losses.append(loss_total.item())

        # 打印
        if iter_idx % args.print_every == 0:
            print(f"  Iter [{iter_idx:3d}/{dataloader.num_batch:3d}] "
                  f"Pred: {loss_pred.item():.4f} "
                  f"Recon: {loss_recon.item():.4f} "
                  f"Total: {loss_total.item():.4f}")

    epoch_time = time.time() - start_time

    return {
        'loss_pred': np.mean(pred_losses),
        'loss_recon': np.mean(recon_losses),
        'loss_total': np.mean(total_losses),
        'lambda_pred': lambda_pred,
        'epoch_time': epoch_time,
    }

def validate(encoder, pred_head, dataloader, args):
    """验证"""
    encoder.eval()
    pred_head.eval()

    val_pred_losses = []
    num_subset = int(args.num_nodes * args.subset_ratio)

    with torch.no_grad():
        for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
            x_full = torch.Tensor(x).to(args.device)
            x_full = x_full.transpose(1, 3)

            y_full = torch.Tensor(y).to(args.device)
            y_full = y_full.transpose(1, 2)

            # 随机采样子集
            idx_subset = np.random.choice(args.num_nodes, size=num_subset, replace=False)
            idx_subset = torch.tensor(idx_subset, device=args.device)

            x_subset = x_full[:, :, idx_subset, :]
            y_subset = y_full[:, idx_subset, :]

            h_all = encoder(x_subset, idx_subset)
            pred_all = pred_head(h_all)

            loss_pred = compute_subset_pred_loss(
                pred_all, y_subset, idx_subset, loss_fn='mae'
            )
            val_pred_losses.append(loss_pred.item())

    return {'val_pred_loss': np.mean(val_pred_losses)}


def train_loop(encoder, pred_head, decoder, train_loader, val_loader, args):
    """训练循环"""

    os.makedirs(args.save_dir, exist_ok=True)

    # 优化器 - 联合优化 Encoder 和 Pred Head
    params = list(encoder.parameters()) + list(pred_head.parameters())
    if decoder is not None and args.lambda_recon > 0:
        params += list(decoder.parameters())

    optimizer = torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    history = {
        'train_pred_loss': [],
        'train_recon_loss': [],
        'val_pred_loss': [],
        'best_val_loss': float('inf'),
    }

    print("\n" + "=" * 80)
    print(" " * 20 + "联合训练: Encoder + TemporalPredHead")
    print("=" * 80)
    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    print(f"Num nodes: {args.num_nodes}")
    print(f"Subset ratio: {args.subset_ratio}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss weights: λ_pred={args.lambda_pred}, λ_recon={args.lambda_recon}")
    print(f"Pred warmup epochs: {args.pred_warmup_epochs}")
    print(f"\nModel parameters:")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Pred Head: {sum(p.numel() for p in pred_head.parameters()):,}")
    if decoder is not None:
        print(f"  Decoder: {sum(p.numel() for p in decoder.parameters()):,}")
    print("=" * 80 + "\n")

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 80)

        # 训练
        train_metrics = train_epoch(
            encoder, pred_head, decoder,
            train_loader, optimizer, args, epoch
        )

        # 验证
        val_metrics = validate(encoder, pred_head, val_loader, args)

        # 记录
        history['train_pred_loss'].append(train_metrics['loss_pred'])
        history['train_recon_loss'].append(train_metrics['loss_recon'])
        history['val_pred_loss'].append(val_metrics['val_pred_loss'])

        # 学习率调度
        scheduler.step(val_metrics['val_pred_loss'])

        # 打印
        print(f"\n[Epoch {epoch} Summary]")
        print(f"  Train Pred Loss: {train_metrics['loss_pred']:.6f}")
        print(f"  Train Recon Loss: {train_metrics['loss_recon']:.6f}")
        print(f"  Val Pred Loss: {val_metrics['val_pred_loss']:.6f}")
        print(f"  λ_pred (warmup): {train_metrics['lambda_pred']:.4f}")
        print(f"  Time: {train_metrics['epoch_time']:.2f}s")

        # 保存最佳模型
        if val_metrics['val_pred_loss'] < history['best_val_loss']:
            history['best_val_loss'] = val_metrics['val_pred_loss']
            best_path = os.path.join(args.save_dir, 'best_joint_model.pt')
            save_dict = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'pred_head_state_dict': pred_head.state_dict(),
                'val_pred_loss': history['best_val_loss'],
                'args': vars(args),
            }
            if decoder is not None:
                save_dict['decoder_state_dict'] = decoder.state_dict()
            torch.save(save_dict, best_path)
            print(f"  → Best model saved! Val loss: {history['best_val_loss']:.6f}")

        # 定期保存
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f'joint_checkpoint_epoch_{epoch}.pt')
            save_dict = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'pred_head_state_dict': pred_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'args': vars(args),
            }
            if decoder is not None:
                save_dict['decoder_state_dict'] = decoder.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    print("\n" + "=" * 80)
    print(" " * 25 + "训练完成!")
    print("=" * 80)
    print(f"Best validation pred loss: {history['best_val_loss']:.6f}")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main():
    parser = argparse.ArgumentParser(description='联合训练: Encoder + TemporalPredHead')

    # 数据参数
    parser.add_argument('--data', type=str, required=True, help='数据路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--in_dim', type=int, default=None, help='输入特征维度')
    parser.add_argument('--seq_in_len', type=int, default=12, help='输入序列长度')
    parser.add_argument('--seq_out_len', type=int, default=12, help='输出序列长度')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_slots', type=int, default=16, help='Slot 数量')
    parser.add_argument('--pred_kernel_size', type=int, default=3, help='预测头卷积核大小')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help='梯度裁剪')

    # 损失权重
    parser.add_argument('--lambda_pred', type=float, default=1.0, help='预测损失权重')
    parser.add_argument('--lambda_pred_min', type=float, default=0.1, help='预测损失最小权重 (warmup)')
    parser.add_argument('--lambda_recon', type=float, default=0.1, help='重构损失权重')
    parser.add_argument('--pred_warmup_epochs', type=int, default=10, help='预测 loss 权重 warmup epochs')

    # 子集配置
    parser.add_argument('--subset_ratio', type=float, default=0.3, help='子集比例')

    # 预训练权重
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--use_decoder', type=str_to_bool, default=True, help='是否使用 decoder 计算重构 loss')

    # load_dataset 需要的参数
    parser.add_argument('--predefined_S', type=str_to_bool, default=False, help='是否使用预定义子集S')
    parser.add_argument('--predefined_S_frac', type=int, default=15, help='预定义子集S的比例')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_joint', help='保存目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--print_every', type=int, default=50, help='打印间隔')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备设置
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # 加载数据
    print(f"Loading data from {args.data}...")
    dataloader_dict = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)

    train_loader = dataloader_dict['train_loader']
    val_loader = dataloader_dict['val_loader']

    args.num_nodes = train_loader.num_nodes

    if args.in_dim is None:
        sample_x = train_loader.xs[0]
        args.in_dim = sample_x.shape[-1]

    print(f"\n✓ Data loaded:")
    print(f"  Num nodes: {args.num_nodes}")
    print(f"  Input dim: {args.in_dim}")
    print(f"  Train samples: {train_loader.size}")
    print(f"  Val samples: {val_loader.size}")

    # 创建 Encoder
    encoder = SlotBasedEncoder(
        num_nodes=args.num_nodes,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_slots=args.num_slots,
        device=args.device
    ).to(device)

    # 创建 TemporalPredHead
    pred_head = TemporalPredHead(
        hidden_dim=args.hidden_dim,
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
        kernel_size=args.pred_kernel_size,
        dropout=0.1
    ).to(device)

    # 创建 Decoder (可选，用于重构 loss)
    decoder = None
    if args.use_decoder and args.lambda_recon > 0:
        from model.decoder_v2 import STDecoder
        decoder = STDecoder(
            in_dim=args.hidden_dim,
            out_dim=args.in_dim
        ).to(device)

    # 加载预训练权重
    if args.pretrain_ckpt is not None and os.path.isfile(args.pretrain_ckpt):
        ckpt = torch.load(args.pretrain_ckpt, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        if decoder is not None and 'decoder_state_dict' in ckpt:
            decoder.load_state_dict(ckpt['decoder_state_dict'])
        print(f"✓ Loaded pretrained weights from {args.pretrain_ckpt}")
    elif args.pretrain_ckpt is not None:
        print(f"Warning: pretrain_ckpt not found: {args.pretrain_ckpt}")

    print(f"✓ Models created")

    # 训练
    train_loop(encoder, pred_head, decoder, train_loader, val_loader, args)


if __name__ == "__main__":
    main()