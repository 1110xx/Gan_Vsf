#!/usr/bin/env python3

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from util import load_dataset, StandardScaler

from model.encoder_v2 import TimeFirstEncoder
from model.decoder_v2 import STDecoder
from model.discriminator_v2 import HybridNodeDiscriminator, create_discriminator


def compute_hybrid_discriminator_loss(
        cond_score_real, cond_score_fake,
        internal_score_real, internal_score_fake,
        alpha=0.7
):
    """混合判别器损失计算"""
    cond_loss = (
            F.binary_cross_entropy_with_logits(cond_score_real, torch.ones_like(cond_score_real) * 0.9) +
            F.binary_cross_entropy_with_logits(cond_score_fake, torch.zeros_like(cond_score_fake))
    )
    internal_loss = (
            F.binary_cross_entropy_with_logits(internal_score_real, torch.ones_like(internal_score_real) * 0.9) +
            F.binary_cross_entropy_with_logits(internal_score_fake, torch.zeros_like(internal_score_fake))
    )
    d_loss = alpha * cond_loss + (1 - alpha) * internal_loss

    return d_loss, {
        'd_loss': d_loss.item(),
        'd_loss_cond': cond_loss.item(),
        'd_loss_internal': internal_loss.item(),
        'cond_score_real': cond_score_real.mean().item(),
        'cond_score_fake': cond_score_fake.mean().item(),
        'internal_score_real': internal_score_real.mean().item(),
        'internal_score_fake': internal_score_fake.mean().item(),
    }


def compute_hybrid_generator_loss(
        cond_score_fake, internal_score_fake,
        x_fake, x_real, missing_indices, lambda_rec=1.0, lambda_adv=0.1, alpha=0.7
):
    """混合判别器的生成器损失，只计算缺失节点的重构损失"""
    # 对抗损失（混合）
    cond_loss = F.binary_cross_entropy_with_logits(cond_score_fake, torch.ones_like(cond_score_fake))
    internal_loss = F.binary_cross_entropy_with_logits(internal_score_fake, torch.ones_like(internal_score_fake))
    loss_adv = alpha * cond_loss + (1 - alpha) * internal_loss

    # 重构损失 - 只计算缺失节点部分
    x_missing_fake = x_fake[:, :, missing_indices, :]
    x_missing_real = x_real[:, :, missing_indices, :]

    # 使用L2损失
    mse = (x_missing_fake - x_missing_real) ** 2
    loss_rec = mse.mean()

    # 可选：也可以计算缺失节点的MAE作为额外指标
    mae = torch.abs(x_missing_fake - x_missing_real).mean()

    g_loss = lambda_rec * loss_rec + lambda_adv * loss_adv

    return g_loss, {
        'g_loss': g_loss.item(),
        'g_loss_adv': loss_adv.item(),
        'g_loss_cond': cond_loss.item(),
        'g_loss_internal': internal_loss.item(),
        'g_loss_rec': loss_rec.item(),
        'g_loss_rec_missing_mae': mae.item(),
        'missing_nodes_count': len(missing_indices),
    }


def train_step(encoder, decoder, discriminator, x_full, idx_subset,
               opt_g, opt_d, scaler, lambda_rec, lambda_adv,
               use_amp, device, args):
    """混合判别器的训练步骤"""

    B, F, N, T = x_full.shape

    # 创建掩码来分离子集节点和缺失节点
    subset_mask = torch.zeros(N, dtype=torch.bool, device=device)
    subset_mask[idx_subset] = True

    missing_mask = ~subset_mask
    missing_indices = torch.where(missing_mask)[0]

    # 提取真实数据的子集和缺失部分
    x_subset_real = x_full[:, :, idx_subset, :]
    x_missing_real = x_full[:, :, missing_indices, :]

    # ========== 判别器训练 ==========
    discriminator.train()
    encoder.eval()
    decoder.eval()

    opt_d.zero_grad()

    with autocast(enabled=use_amp):
        with torch.no_grad():
            h = encoder(x_subset_real, idx_subset)
            x_fake = decoder(h)

        # 提取生成数据的缺失部分
        x_missing_fake = x_fake[:, :, missing_indices, :]

        # 判别器前向传播（真实数据）
        cond_score_real, internal_score_real = discriminator(x_subset_real, x_missing_real)

        # 判别器前向传播（生成数据）
        cond_score_fake, internal_score_fake = discriminator(x_subset_real, x_missing_fake.detach())

        # 计算判别器损失
        d_loss, d_metrics = compute_hybrid_discriminator_loss(
            cond_score_real, cond_score_fake,
            internal_score_real, internal_score_fake,
            alpha=args.disc_alpha
        )

    scaler.scale(d_loss).backward()
    scaler.unscale_(opt_d)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.max_grad_norm_d)
    scaler.step(opt_d)

    # ========== 生成器训练 ==========
    encoder.train()
    decoder.train()
    discriminator.eval()

    opt_g.zero_grad()

    with autocast(enabled=use_amp):
        h = encoder(x_subset_real, idx_subset)
        x_fake = decoder(h)

        # 生成器前向传播
        x_missing_fake = x_fake[:, :, missing_indices, :]
        cond_score_fake, internal_score_fake = discriminator(x_subset_real, x_missing_fake)

        # 计算生成器损失 - 只使用缺失节点的重构损失
        g_loss, g_metrics = compute_hybrid_generator_loss(
            cond_score_fake, internal_score_fake,
            x_fake, x_full, missing_indices,
            lambda_rec, lambda_adv,
            alpha=args.disc_alpha
        )

    scaler.scale(g_loss).backward()
    scaler.unscale_(opt_g)
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()),
        max_norm=args.max_grad_norm_g
    )
    scaler.step(opt_g)
    scaler.update()

    metrics = {**d_metrics, **g_metrics}
    return metrics


def train_epoch_hybrid(encoder, decoder, discriminator, dataloader, opt_g, opt_d,
                       scaler, args, epoch):
    """混合判别器的训练epoch"""

    encoder.train()
    decoder.train()
    discriminator.train()

    d_losses = []
    g_losses = []
    g_rec_losses = []
    d_cond_losses = []
    d_internal_losses = []
    g_rec_mae_losses = []

    num_subset = int(args.num_nodes * args.subset_ratio)

    start_time = time.time()

    # 初始化 idx_subset
    perm = np.random.permutation(args.num_nodes)
    idx_subset = torch.tensor(perm[:num_subset], device=args.device)

    for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
        x_full = torch.Tensor(x).to(args.device)
        x_full = x_full.transpose(1, 3)

        if iter_idx % args.step_size2 == 0:
            perm = np.random.permutation(args.num_nodes)
            idx_subset = torch.tensor(perm[:num_subset], device=args.device)

        metrics = train_step(
            encoder, decoder, discriminator,
            x_full, idx_subset,
            opt_g, opt_d, scaler,
            args.lambda_rec, args.lambda_adv,
            args.use_amp, args.device,
            args
        )

        d_losses.append(metrics['d_loss'])
        g_losses.append(metrics['g_loss'])
        g_rec_losses.append(metrics['g_loss_rec'])
        g_rec_mae_losses.append(metrics.get('g_loss_rec_missing_mae', 0.0))
        d_cond_losses.append(metrics['d_loss_cond'])
        d_internal_losses.append(metrics['d_loss_internal'])

        if iter_idx % args.print_every == 0:
            log_str = f"  Iter [{iter_idx:3d}/{dataloader.num_batch:3d}] "
            log_str += f"D: {metrics['d_loss']:.4f} "
            log_str += f"(cond: {metrics['d_loss_cond']:.4f}, internal: {metrics['d_loss_internal']:.4f}) "
            log_str += f"G: {metrics['g_loss']:.4f} "
            log_str += f"Rec_missing_MSE: {metrics['g_loss_rec']:.6f} "
            log_str += f"Rec_missing_MAE: {metrics.get('g_loss_rec_missing_mae', 0.0):.6f}"
            print(log_str)

    epoch_time = time.time() - start_time

    return {
        'd_loss': np.mean(d_losses),
        'g_loss': np.mean(g_losses),
        'g_loss_rec': np.mean(g_rec_losses),
        'g_loss_rec_missing_mae': np.mean(g_rec_mae_losses),
        'd_loss_cond': np.mean(d_cond_losses),
        'd_loss_internal': np.mean(d_internal_losses),
        'cond_score_real': metrics.get('cond_score_real', 0.0),
        'cond_score_fake': metrics.get('cond_score_fake', 0.0),
        'internal_score_real': metrics.get('internal_score_real', 0.0),
        'internal_score_fake': metrics.get('internal_score_fake', 0.0),
        'epoch_time': epoch_time,
    }


def validate(encoder, decoder, dataloader, args):
    """验证函数，只计算缺失节点的重构损失"""
    encoder.eval()
    decoder.eval()

    val_rec_losses = []
    val_rec_mae_losses = []
    num_subset = int(args.num_nodes * args.subset_ratio)

    with torch.no_grad():
        for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
            x_full = torch.Tensor(x).to(args.device)
            x_full = x_full.transpose(1, 3)

            idx_subset = np.random.choice(args.num_nodes, num_subset, replace=False)
            idx_subset = torch.tensor(idx_subset, device=args.device)

            # 创建掩码
            subset_mask = torch.zeros(args.num_nodes, dtype=torch.bool, device=args.device)
            subset_mask[idx_subset] = True
            missing_mask = ~subset_mask
            missing_indices = torch.where(missing_mask)[0]

            x_subset = x_full[:, :, idx_subset, :]

            h = encoder(x_subset, idx_subset)
            x_fake = decoder(h)

            # 只计算缺失节点的重构损失
            x_missing_fake = x_fake[:, :, missing_indices, :]
            x_missing_real = x_full[:, :, missing_indices, :]

            mse = (x_missing_fake - x_missing_real) ** 2
            loss_rec = mse.mean()
            loss_mae = torch.abs(x_missing_fake - x_missing_real).mean()

            val_rec_losses.append(loss_rec.item())
            val_rec_mae_losses.append(loss_mae.item())

    return {
        'val_rec_loss': np.mean(val_rec_losses),
        'val_rec_mae': np.mean(val_rec_mae_losses)
    }


def train_loop_hybrid(encoder, decoder, discriminator, train_loader, val_loader, args):
    """混合判别器的训练循环"""

    os.makedirs(args.save_dir, exist_ok=True)

    # 优化器
    opt_g = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr_g, betas=(0.5, 0.999), weight_decay=args.weight_decay
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d, betas=(0.5, 0.999), weight_decay=args.weight_decay
    )

    scaler = GradScaler(enabled=args.use_amp)

    history = {
        'train_d_loss': [],
        'train_g_loss': [],
        'train_g_loss_rec': [],
        'train_g_loss_rec_mae': [],
        'val_rec_loss': [],
        'val_rec_mae': [],
        'best_val_loss': float('inf'),
        'train_d_cond_loss': [],
        'train_d_internal_loss': [],
    }

    print("\n" + "=" * 80)
    print(" " * 20 + "Hybrid Discriminator GAN Pretraining")
    print("=" * 80)
    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    print(f"Num nodes: {args.num_nodes}")
    print(f"Subset ratio: {args.subset_ratio}")
    print(f"Discriminator alpha: {args.disc_alpha}")
    print(f"Input dim: {args.in_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rates: G={args.lr_g}, D={args.lr_d}")
    print(f"Loss weights: λ_rec={args.lambda_rec}, λ_adv={args.lambda_adv}")
    print(f"AMP: {args.use_amp}")
    print(f"\nModel parameters:")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Decoder: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"  Discriminator: {sum(p.numel() for p in discriminator.parameters()):,}")
    print("=" * 80 + "\n")

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 80)

        # 训练一个epoch
        train_metrics = train_epoch_hybrid(
            encoder, decoder, discriminator, train_loader,
            opt_g, opt_d, scaler, args, epoch
        )

        # 验证
        val_metrics = validate(encoder, decoder, val_loader, args)

        # 记录历史
        history['train_d_loss'].append(train_metrics['d_loss'])
        history['train_g_loss'].append(train_metrics['g_loss'])
        history['train_g_loss_rec'].append(train_metrics['g_loss_rec'])
        history['train_g_loss_rec_mae'].append(train_metrics.get('g_loss_rec_missing_mae', 0.0))
        history['val_rec_loss'].append(val_metrics['val_rec_loss'])
        history['val_rec_mae'].append(val_metrics['val_rec_mae'])
        history['train_d_cond_loss'].append(train_metrics['d_loss_cond'])
        history['train_d_internal_loss'].append(train_metrics['d_loss_internal'])

        # 打印总结
        print(f"\n[Epoch {epoch} Summary]")
        print(f"  Train D_loss: {train_metrics['d_loss']:.6f}")
        print(f"    - D_cond: {train_metrics['d_loss_cond']:.6f}")
        print(f"    - D_internal: {train_metrics['d_loss_internal']:.6f}")
        print(
            f"  Cond scores - real: {train_metrics['cond_score_real']:.4f}, fake: {train_metrics['cond_score_fake']:.4f}")
        print(
            f"  Internal scores - real: {train_metrics['internal_score_real']:.4f}, fake: {train_metrics['internal_score_fake']:.4f}")
        print(f"  Train G_loss: {train_metrics['g_loss']:.6f}")
        print(f"  Train Rec_loss (missing nodes MSE): {train_metrics['g_loss_rec']:.6f}")
        print(f"  Train Rec_loss (missing nodes MAE): {train_metrics.get('g_loss_rec_missing_mae', 0.0):.6f}")
        print(f"  Val Rec_loss (missing nodes MSE): {val_metrics['val_rec_loss']:.6f}")
        print(f"  Val Rec_loss (missing nodes MAE): {val_metrics['val_rec_mae']:.6f}")
        print(f"  Time: {train_metrics['epoch_time']:.2f}s")

        # 保存最佳模型（基于缺失节点的MSE）
        if val_metrics['val_rec_loss'] < history['best_val_loss']:
            history['best_val_loss'] = val_metrics['val_rec_loss']
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'val_rec_loss': history['best_val_loss'],
                'args': vars(args),
            }, best_path)
            print(f"  → Best model saved! Val MSE loss (missing nodes): {history['best_val_loss']:.6f}")

        # 定期保存检查点
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'opt_d_state_dict': opt_d.state_dict(),
                'history': history,
                'args': vars(args),
            }, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    print("\n" + "=" * 80)
    print(" " * 25 + "Training Completed!")
    print("=" * 80)
    print(f"Best validation loss (missing nodes MSE): {history['best_val_loss']:.6f}")


def str_to_bool(value):
    """字符串转布尔值"""
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main():
    parser = argparse.ArgumentParser(description='通用GAN预训练脚本')

    # 数据参数
    parser.add_argument('--data', type=str, required=True, help='数据路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--in_dim', type=int, default=None, help='输入特征维度')
    parser.add_argument('--seq_in_len', type=int, default=12, help='输入序列长度')
    parser.add_argument('--seq_out_len', type=int, default=12, help='输出序列长度')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_prototypes', type=int, default=32, help='原型数量')
    parser.add_argument('--temporal_dilations', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='时序膨胀率列表')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='生成器学习率')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='判别器学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')

    # 损失权重
    parser.add_argument('--lambda_rec', type=float, default=1.0, help='重构损失权重')
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='对抗损失权重')

    # 子集配置
    parser.add_argument('--subset_ratio', type=float, default=0.15, help='子集比例')
    parser.add_argument('--step_size2', type=int, default=100, help='子集变化步长')

    # 混合判别器参数
    parser.add_argument('--disc_alpha', type=float, default=0.7,
                        help='条件判别损失的权重（内部判别权重为1-alpha）')

    # 梯度截断参数
    parser.add_argument('--max_grad_norm_g', type=float, default=2.0,
                        help='生成器梯度最大范数')
    parser.add_argument('--max_grad_norm_d', type=float, default=1.0,
                        help='判别器梯度最大范数')

    # load_dataset 需要的参数
    parser.add_argument('--predefined_S', type=str_to_bool, default=False, help='是否使用预定义子集S')
    parser.add_argument('--predefined_S_frac', type=int, default=15, help='预定义子集S的比例')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_pretrain_hybrid', help='保存目录')
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

    args.use_amp = (args.device == 'cuda')
    device = torch.device(args.device)

    # 加载数据
    print(f"Loading data from {args.data}...")
    dataloader_dict = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)

    train_loader = dataloader_dict['train_loader']
    val_loader = dataloader_dict['val_loader']
    scaler_data = dataloader_dict['scaler']  # 避免与 GradScaler 命名冲突

    args.num_nodes = train_loader.num_nodes

    if args.in_dim is None:
        sample_x = train_loader.xs[0]
        args.in_dim = sample_x.shape[-1]

    print(f"\n✓ Data loaded:")
    print(f"  Num nodes: {args.num_nodes}")
    print(f"  Input dim: {args.in_dim}")
    print(f"  Train samples: {train_loader.size}")
    print(f"  Val samples: {val_loader.size}")
    print(f"  Subset nodes: {int(args.num_nodes * args.subset_ratio)}")
    print(f"  Missing nodes: {args.num_nodes - int(args.num_nodes * args.subset_ratio)}")

    # 创建模型
    print(f"\nCreating models...")

    encoder = TimeFirstEncoder(
        num_nodes=args.num_nodes,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_prototypes=args.num_prototypes,
        temporal_dilations=args.temporal_dilations,
        device=args.device
    ).to(device)

    decoder = STDecoder(
        in_dim=args.hidden_dim,
        out_dim=args.in_dim
    ).to(device)

    # 使用混合判别器
    discriminator = create_discriminator(
        feature_dim=args.in_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    print(f"✓ Models created")
    print(f"  Using Hybrid Discriminator with alpha={args.disc_alpha}")

    # 训练循环（train_loop函数也需要更新，主要是打印信息的变化）
    train_loop_hybrid(encoder, decoder, discriminator, train_loader, val_loader, args)


if __name__ == "__main__":
    main()