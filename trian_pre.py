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
from model.discriminator_v2 import FullSequenceDiscriminator


def compute_discriminator_loss(score_real, score_fake):
    loss_real = F.relu(1.0 - score_real).mean()
    loss_fake = F.relu(1.0 + score_fake).mean()
    d_loss = loss_real + loss_fake

    return d_loss, {
        'd_loss': d_loss.item(),
        'd_loss_real': loss_real.item(),
        'd_loss_fake': loss_fake.item(),
        'score_real': score_real.mean().item(),
        'score_fake': score_fake.mean().item(),
    }


def compute_generator_loss(score_fake, x_fake, x_real, mask, lambda_rec=1.0, lambda_adv=0.1):
    loss_adv = -score_fake.mean()
    mse = (x_fake - x_real) ** 2
    masked_mse = mse * mask
    loss_rec = masked_mse.sum() / mask.sum().clamp(min=1.0)
    g_loss = lambda_rec * loss_rec + lambda_adv * loss_adv

    return g_loss, {
        'g_loss': g_loss.item(),
        'g_loss_adv': loss_adv.item(),
        'g_loss_rec': loss_rec.item(),
    }


def train_step(encoder, decoder, discriminator, x_full, idx_subset,
              opt_g, opt_d, scaler, lambda_rec, lambda_adv, use_amp, device):
    B, F, N, T = x_full.shape

    x_subset = x_full[:, :, idx_subset, :]

    mask = torch.zeros(B, 1, N, T, device=device)
    mask[:, :, idx_subset, :] = 1.0

    discriminator.train()
    encoder.eval()
    decoder.eval()

    opt_d.zero_grad()

    with autocast(enabled=use_amp):
        with torch.no_grad():
            h = encoder(x_subset, idx_subset)
            x_fake = decoder(h)

        score_real = discriminator(x_full)
        score_fake = discriminator(x_fake.detach())

        d_loss, d_metrics = compute_discriminator_loss(score_real, score_fake)

    scaler.scale(d_loss).backward()
    scaler.step(opt_d)
    scaler.update()

    encoder.train()
    decoder.train()
    discriminator.eval()

    opt_g.zero_grad()

    with autocast(enabled=use_amp):
        h = encoder(x_subset, idx_subset)
        x_fake = decoder(h)
        score_fake = discriminator(x_fake)

        g_loss, g_metrics = compute_generator_loss(
            score_fake, x_fake, x_full, mask, lambda_rec, lambda_adv
        )

    scaler.scale(g_loss).backward()

    scaler.unscale_(opt_g)
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()),
        max_norm=5.0
    )

    scaler.step(opt_g)
    scaler.update()

    metrics = {**d_metrics, **g_metrics}

    return metrics


def train_epoch(encoder, decoder, discriminator, dataloader, opt_g, opt_d,
               scaler, args, epoch):
    encoder.train()
    decoder.train()
    discriminator.train()

    d_losses = []
    g_losses = []
    g_rec_losses = []

    num_subset = int(args.num_nodes * args.subset_ratio)

    start_time = time.time()

    for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
        x_full = torch.Tensor(x).to(args.device)
        x_full = x_full.transpose(1, 3)

        if iter_idx % args.step_size2 == 0:
            perm = np.random.permutation(args.num_nodes)

        idx_subset = perm[:num_subset]
        idx_subset = torch.tensor(idx_subset, device=args.device)

        metrics = train_step(
            encoder, decoder, discriminator,
            x_full, idx_subset,
            opt_g, opt_d, scaler,
            args.lambda_rec, args.lambda_adv,
            args.use_amp, args.device
        )

        d_losses.append(metrics['d_loss'])
        g_losses.append(metrics['g_loss'])
        g_rec_losses.append(metrics['g_loss_rec'])

        if iter_idx % args.print_every == 0:
            print(f"  Iter [{iter_idx:3d}/{dataloader.num_batch:3d}] "
                  f"D: {metrics['d_loss']:.4f} "
                  f"G: {metrics['g_loss']:.4f} "
                  f"Rec: {metrics['g_loss_rec']:.6f}")

    epoch_time = time.time() - start_time

    return {
        'd_loss': np.mean(d_losses),
        'g_loss': np.mean(g_losses),
        'g_loss_rec': np.mean(g_rec_losses),
        'epoch_time': epoch_time,
    }


def validate(encoder, decoder, dataloader, args):
    encoder.eval()
    decoder.eval()

    val_rec_losses = []
    num_subset = int(args.num_nodes * args.subset_ratio)

    with torch.no_grad():
        for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
            x_full = torch.Tensor(x).to(args.device)
            x_full = x_full.transpose(1, 3)

            idx_subset = np.random.choice(args.num_nodes, num_subset, replace=False)
            idx_subset = torch.tensor(idx_subset, device=args.device)

            x_subset = x_full[:, :, idx_subset, :]

            B, F, N, T = x_full.shape
            mask = torch.zeros(B, 1, N, T, device=args.device)
            mask[:, :, idx_subset, :] = 1.0

            h = encoder(x_subset, idx_subset)
            x_fake = decoder(h)

            mse = (x_fake - x_full) ** 2
            masked_mse = mse * mask
            loss_rec = masked_mse.sum() / mask.sum().clamp(min=1.0)
            val_rec_losses.append(loss_rec.item())

    return {'val_rec_loss': np.mean(val_rec_losses)}


def train_loop(encoder, decoder, discriminator, train_loader, val_loader, args):
    os.makedirs(args.save_dir, exist_ok=True)

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
        'val_rec_loss': [],
        'best_val_loss': float('inf'),
    }

    print("\n" + "=" * 80)
    print(" " * 25 + "GAN Pretraining")
    print("=" * 80)
    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    print(f"Num nodes: {args.num_nodes}")
    print(f"Subset ratio: {args.subset_ratio} ({int(args.num_nodes * args.subset_ratio)} nodes)")
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

        train_metrics = train_epoch(
            encoder, decoder, discriminator, train_loader,
            opt_g, opt_d, scaler, args, epoch
        )

        val_metrics = validate(encoder, decoder, val_loader, args)

        history['train_d_loss'].append(train_metrics['d_loss'])
        history['train_g_loss'].append(train_metrics['g_loss'])
        history['val_rec_loss'].append(val_metrics['val_rec_loss'])

        print(f"\n[Epoch {epoch} Summary]")
        print(f"  Train D_loss: {train_metrics['d_loss']:.6f}")
        print(f"  Train G_loss: {train_metrics['g_loss']:.6f}")
        print(f"  Train Rec_loss: {train_metrics['g_loss_rec']:.6f}")
        print(f"  Val Rec_loss: {val_metrics['val_rec_loss']:.6f}")
        print(f"  Time: {train_metrics['epoch_time']:.2f}s")

        if val_metrics['val_rec_loss'] < history['best_val_loss']:
            history['best_val_loss'] = val_metrics['val_rec_loss']
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'val_rec_loss': history['best_val_loss'],
                'args': vars(args),
            }, best_path)
            print(f"  → Best model saved! Val loss: {history['best_val_loss']:.6f}")

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
    print(f"Best validation loss: {history['best_val_loss']:.6f}")


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
    parser.add_argument('--data', type=str, required=True, help='数据路径（如 ./data/ECG5000）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--in_dim', type=int, default=None, help='输入特征维度（自动检测）')
    parser.add_argument('--seq_in_len', type=int, default=12, help='输入序列长度')
    parser.add_argument('--seq_out_len', type=int, default=12, help='输出序列长度')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_prototypes', type=int, default=32, help='原型数量')
    parser.add_argument('--temporal_dilations', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='时序膨胀率列表')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='生成器学习率')
    parser.add_argument('--lr_d', type=float, default=4e-4, help='判别器学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')

    # 损失权重
    parser.add_argument('--lambda_rec', type=float, default=1.0, help='重构损失权重')
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='对抗损失权重')

    # 子集配置
    parser.add_argument('--subset_ratio', type=float, default=0.3, help='子集比例')
    parser.add_argument('--step_size2', type=int, default=100, help='子集变化步长')

    # load_dataset需要的参数（参考main.py）
    parser.add_argument('--predefined_S', type=str_to_bool, default=False, help='是否使用预定义子集S')
    parser.add_argument('--predefined_S_frac', type=int, default=15, help='预定义子集S的比例')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_pretrain', help='保存目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--print_every', type=int, default=50, help='打印间隔')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    args.use_amp = (args.device == 'cuda')
    device = torch.device(args.device)

    print(f"Loading data from {args.data}...")
    dataloader_dict = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)

    train_loader = dataloader_dict['train_loader']
    val_loader = dataloader_dict['val_loader']
    scaler = dataloader_dict['scaler']

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

    discriminator = FullSequenceDiscriminator(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    print(f"✓ Models created")

    train_loop(encoder, decoder, discriminator, train_loader, val_loader, args)


if __name__ == "__main__":
    main()