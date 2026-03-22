#!/usr/bin/env python3
"""
GAN 预训练脚本 - 适配 Encoder V4

结合 LLM 风格的时序因果预训练 + GAN 对抗训练

设计理念：
1. 时序因果预训练（Next Step Prediction）：学习时序依赖
2. 空间重构 + GAN：学习全局空间信息的细节

优势（相比纯 MAE 预训练）：
- GAN 强迫模型学习数据分布，而非仅仅均值
- 判别器提供额外梯度信号，学习真实数据的统计特性
- 条件判别强化子集→全集的映射能力

使用方法：
    python train_gan_pretrain_v4.py \
        --data ./data/METR-LA \
        --num_epochs 50 \
        --lambda_temporal 1.0 \
        --lambda_spatial 0.5 \
        --lambda_adv 0.1 \
        --subset_ratio 0.15 \
        --save_dir ./checkpoints_gan_pretrain
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from util import load_dataset, StandardScaler

# 新模型
from model.encoder import NodeAwareTemporalEncoder
from model.decoder import STDecoder
from model.discriminator import HybridNodeDiscriminator, create_discriminator
# 时序预训练损失
from model.temporal_pretrain import NextStepPredictionLoss, LatentDynamicsLoss


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


# ============================================================
# 损失函数
# ============================================================

def compute_discriminator_loss(
    cond_score_real, cond_score_fake,
    internal_score_real, internal_score_fake,
    alpha=0.7
):
    """混合判别器损失"""
    # 条件判别损失（基于子集判断缺失节点真假）
    cond_loss = (
        F.binary_cross_entropy_with_logits(
            cond_score_real, torch.ones_like(cond_score_real) * 0.9  # label smoothing
        ) +
        F.binary_cross_entropy_with_logits(
            cond_score_fake, torch.zeros_like(cond_score_fake)
        )
    )

    # 内部判别损失（仅看缺失节点内部一致性）
    internal_loss = (
        F.binary_cross_entropy_with_logits(
            internal_score_real, torch.ones_like(internal_score_real) * 0.9
        ) +
        F.binary_cross_entropy_with_logits(
            internal_score_fake, torch.zeros_like(internal_score_fake)
        )
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


def compute_generator_loss(
    cond_score_fake, internal_score_fake,
    x_fake, x_real, missing_indices, subset_indices=None,
    lambda_rec=1.0, lambda_adv=0.1, alpha=0.7,
    use_global_recon=False, lambda_obs=0.3
   ):
    """生成器损失（重构 + 对抗）

    Args:
        cond_score_fake: 条件判别器对 fake 的评分
        internal_score_fake: 内部判别器对 fake 的评分
        x_fake: 重构的全集 (B, F, N, T)
        x_real: 真实的全集 (B, F, N, T)
        missing_indices: 缺失节点索引
        subset_indices: 子集节点索引（全局重构时需要）
        lambda_rec: 重构损失权重
        lambda_adv: 对抗损失权重
        alpha: 条件判别损失权重
        use_global_recon: 是否使用全局重构（子集 + 缺失）
        lambda_obs: 子集重构损失权重（相对于缺失节点）
    """
    # 对抗损失
    cond_loss = F.binary_cross_entropy_with_logits(
        cond_score_fake, torch.ones_like(cond_score_fake)
    )
    internal_loss = F.binary_cross_entropy_with_logits(
        internal_score_fake, torch.ones_like(internal_score_fake)
    )
    loss_adv = alpha * cond_loss + (1 - alpha) * internal_loss

    # 缺失节点重构损失
    x_missing_fake = x_fake[:, :, missing_indices, :]
    x_missing_real = x_real[:, :, missing_indices, :]
    loss_rec_missing = F.mse_loss(x_missing_fake, x_missing_real)

    if use_global_recon and subset_indices is not None:
        # 【全局重构】同时约束子集和缺失节点
        x_subset_fake = x_fake[:, :, subset_indices, :]
        x_subset_real = x_real[:, :, subset_indices, :]
        loss_rec_subset = F.mse_loss(x_subset_fake, x_subset_real)

        # 组合损失：缺失节点权重更大（更难重构）
        loss_rec = loss_rec_missing + lambda_obs * loss_rec_subset

        metrics = {
            'g_loss_rec': loss_rec.item(),
            'g_loss_rec_missing': loss_rec_missing.item(),
            'g_loss_rec_subset': loss_rec_subset.item(),
        }
    else:
        # 【原始模式】只重构缺失节点
        loss_rec = loss_rec_missing
        metrics = {
            'g_loss_rec': loss_rec.item(),
        }

    g_loss = lambda_rec * loss_rec + lambda_adv * loss_adv

    metrics.update({
        'g_loss': g_loss.item(),
        'g_loss_adv': loss_adv.item(),
    })

    return g_loss, loss_rec, loss_adv, metrics

def compute_temporal_loss(
    h_all,x_full, temporal_loss_fn,
    idx_subset=None, temporal_missing_weight=1.0
    ):
    loss = temporal_loss_fn(h_all, x_full)
    return loss

def get_epoch_subset_slices(args, epoch: int, stream: str = "train"):
    """按 epoch 构建节点子集切片：同一 epoch 内覆盖全集，周期性刷新 perm。"""
    num_subset = max(1, int(args.num_nodes * args.subset_ratio))
    refresh_every = max(1, int(getattr(args, 'perm_refresh_epochs', 100)))
    state_key = f"_subset_state_{stream}"
    state = getattr(args, state_key, None)

    should_refresh = (
        state is None or
        state.get('perm', None) is None or
        (epoch - 1) % refresh_every == 0
    )

    if should_refresh:
        perm = np.random.permutation(args.num_nodes)
        state = {'perm': perm, 'epoch': epoch}
        setattr(args, state_key, state)
    else:
        perm = state['perm']

    subset_slices = []
    for start in range(0, args.num_nodes, num_subset):
        subset_slices.append(perm[start:start + num_subset])

    return subset_slices, should_refresh

def compute_real_space_mae(x_fake, x_real, missing_indices, scaler):
    """计算原始尺度的 MAE"""
    x_fake_missing = x_fake[:, :, missing_indices, :]
    x_real_missing = x_real[:, :, missing_indices, :]

    # 逆变换
    x_fake_real = x_fake_missing * scaler.std + scaler.mean
    x_real_real = x_real_missing * scaler.std + scaler.mean

    mae = torch.abs(x_fake_real - x_real_real).mean()
    return mae.item()


# ============================================================
# 训练步骤
# ============================================================

def train_step(
    encoder, decoder, discriminator, temporal_loss_fn,
    x_full, idx_subset,
    opt_g, opt_d, grad_scaler,
    args, device,
    latent_dyn_loss_fn=None,
    epoch : int = 0,
):
    """单步训练"""
    B, F, N, T = x_full.shape

    # 创建掩码
    subset_mask = torch.zeros(N, dtype=torch.bool, device=device)
    subset_mask[idx_subset] = True
    missing_mask = ~subset_mask
    missing_indices = torch.where(missing_mask)[0]

    # 提取子集和缺失部分
    x_subset_real = x_full[:, :, idx_subset, :]
    x_missing_real = x_full[:, :, missing_indices, :]

    # ========== 1. 判别器训练 ==========
    discriminator.train()
    encoder.eval()
    decoder.eval()

    opt_d.zero_grad()

    with autocast(enabled=args.use_amp):
        with torch.no_grad():
            h = encoder(x_subset_real, idx_subset)
            x_fake = decoder(h)

        x_missing_fake = x_fake[:, :, missing_indices, :]

        # 判别器前向
        cond_score_real, internal_score_real = discriminator(x_subset_real, x_missing_real)
        cond_score_fake, internal_score_fake = discriminator(x_subset_real, x_missing_fake.detach())

        d_loss, d_metrics = compute_discriminator_loss(
            cond_score_real, cond_score_fake,
            internal_score_real, internal_score_fake,
            alpha=args.disc_alpha
        )

    grad_scaler.scale(d_loss).backward()
    grad_scaler.unscale_(opt_d)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.max_grad_norm_d)
    grad_scaler.step(opt_d)

    # ========== 2. 生成器训练 ==========
    encoder.train()
    decoder.train()
    discriminator.eval()

    opt_g.zero_grad()

    with autocast(enabled=args.use_amp):
        # Encoder 前向
        h = encoder(x_subset_real, idx_subset)

        # 时序因果损失
        loss_temporal = compute_temporal_loss(h, x_full, temporal_loss_fn)

        # Decoder 重构
        x_fake = decoder(h)
        x_missing_fake = x_fake[:, :, missing_indices, :]

        # 对抗损失
        cond_score_fake, internal_score_fake = discriminator(x_subset_real, x_missing_fake)
        g_loss_gan, loss_rec, loss_adv, g_metrics = compute_generator_loss(
            cond_score_fake, internal_score_fake,
            x_fake, x_full, missing_indices,
            subset_indices=idx_subset,
            lambda_rec = args.lambda_rec, 
            lambda_adv = args.lambda_adv, 
            alpha = args.disc_alpha,
            use_global_recon=getattr(args, 'use_global_recon', True),
            lambda_obs=getattr(args, 'lambda_obs', 0.3)
        )

        # latent dynamics loss
        loss_latent_dyn = torch.tensor(0.0, device=device)
        late_dyn_metrics = {}
        if latent_dyn_loss_fn is not None and args.lambda_latent_dyn > 0:
            loss_latent_dyn ,late_dyn_metrics = latent_dyn_loss_fn(h)

        # 总损失 = 时序损失 + 空间重构损失 + 对抗损失+latent dynamics loss
        total_g_loss = (
            args.lambda_temporal * loss_temporal +
            args.lambda_spatial * loss_rec +
            args.lambda_adv * loss_adv+
            args.lambda_latent_dyn * loss_latent_dyn
        )

    grad_scaler.scale(total_g_loss).backward()
    grad_scaler.unscale_(opt_g)
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()),
        max_norm=args.max_grad_norm_g
    )
    grad_scaler.step(opt_g)
    grad_scaler.update()

    # 汇总指标
    metrics = {
        **d_metrics,
        **g_metrics,
        'loss_temporal': loss_temporal.item(),
        'loss_latent_dyn': loss_latent_dyn.item() if isinstance(loss_latent_dyn, torch.Tensor) else loss_latent_dyn,
        'total_g_loss': total_g_loss.item(),
    }

    if getattr(args, 'scaler', None) is not None:
        metrics['g_mae_real'] = compute_real_space_mae(
            x_fake, x_full, missing_indices, args.scaler
        )

    return metrics


def train_epoch(
    encoder, decoder, discriminator, temporal_loss_fn,
    dataloader, opt_g, opt_d, grad_scaler, args, epoch,
    latent_dyn_loss_fn=None
):
    """训练一个 epoch"""
    encoder.train()
    decoder.train()
    discriminator.train()

    dataloader.shuffle()

    # 指标累积
    all_metrics = {
        'd_loss': [], 'g_loss': [], 'loss_temporal': [], 'g_loss_rec': [],
        'd_loss_cond': [], 'd_loss_internal': [], 'g_mae_real': [],
        'g_loss_rec_missing': [], 'g_loss_rec_subset': [],
        'loss_latent_dyn': [], 'latent_dyn_loss': [], 'latent_cahnge': [], 'latent_pred_error': [],
    }

    num_subset = int(args.num_nodes * args.subset_ratio)
    start_time = time.time()

    subset_slices, perm_refreshed = get_epoch_subset_slices(args, epoch, stream="train")
    num_split = len(subset_slices)


    for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
        x_full = torch.Tensor(x).to(args.device)
        x_full = x_full.transpose(1, 3)  # (B, F, N, T)

        # epoch 内按切片循环。保证覆盖全集
        idx_subset = torch.tensor(subset_slices[iter_idx % num_split], device=args.device)

        metrics = train_step(
            encoder, decoder, discriminator, temporal_loss_fn,
            x_full, idx_subset,
            opt_g, opt_d, grad_scaler,
            args, args.device,
            latent_dyn_loss_fn=latent_dyn_loss_fn,
            epoch=epoch
        )

        # 累积指标
        for key in all_metrics:
            if key in metrics:
                all_metrics[key].append(metrics[key])

        if iter_idx % args.print_every == 0:
            log_str = f"  Iter [{iter_idx:3d}/{dataloader.num_batch:3d}] "
            log_str += f"D: {metrics['d_loss']:.4f} "
            log_str += f"G: {metrics['total_g_loss']:.4f} "
            log_str += f"(Temp: {metrics['loss_temporal']:.4f}, "
            log_str += f"Rec: {metrics['g_loss_rec']:.4f}, "
            log_str += f"Adv: {metrics['g_loss_adv']:.4f}"
            if 'loss_latent_dyn' in metrics and metrics['loss_latent_dyn'] > 0:
                log_str += f", Dyn: {metrics['loss_latent_dyn']:.4f}"
            log_str += ")"
            if 'g_mae_real' in metrics:
                log_str += f" MAE: {metrics['g_mae_real']:.4f}"
            print(log_str)

    epoch_time = time.time() - start_time

    # 平均指标
    avg_metrics = {
        key: np.mean(vals) if vals else float('nan')
        for key, vals in all_metrics.items()
    }
    avg_metrics['epoch_time'] = epoch_time
    avg_metrics['cond_score_real'] = metrics.get('cond_score_real', 0)
    avg_metrics['cond_score_fake'] = metrics.get('cond_score_fake', 0)
    avg_metrics['internal_score_real'] = metrics.get('internal_score_real', 0)
    avg_metrics['internal_score_fake'] = metrics.get('internal_score_fake', 0)
    avg_metrics['num_split'] = num_split
    avg_metrics['perm_refreshed'] = int(perm_refreshed)
    avg_metrics['subset_cover_nodes'] = int(sum(len(s) for s in subset_slices))

    return avg_metrics

def validate(encoder, decoder, temporal_loss_fn, dataloader, args, epoch: int):
    """验证"""
    encoder.eval()
    decoder.eval()

    val_rec_losses = []
    val_rec_missing_losses = []
    val_rec_subset_losses = []
    val_temporal_losses = []
    val_real_maes = []
    val_real_maes_subset = []

    use_global_recon = getattr(args, 'use_global_recon', False)
    lambda_obs = getattr(args, 'lambda_obs', 0.3)
    subset_slices, _ = get_epoch_subset_slices(args, epoch, stream="val")
    num_split = len(subset_slices)  

    with torch.no_grad():
        for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
            x_full = torch.Tensor(x).to(args.device)
            x_full = x_full.transpose(1, 3)

            idx_subset = torch.tensor(subset_slices[iter_idx % num_split], device=args.device)
            x_subset = x_full[:, :, idx_subset, :]

            # 前向
            h = encoder(x_subset, idx_subset)
            x_fake = decoder(h)

            # 时序损失
            loss_temporal = compute_temporal_loss(h, x_full, temporal_loss_fn)
            val_temporal_losses.append(loss_temporal.item())

            # 重构损失
            subset_mask = torch.zeros(args.num_nodes, dtype=torch.bool, device=args.device)
            subset_mask[idx_subset] = True
            missing_indices = torch.where(~subset_mask)[0]

            # 缺失节点重构损失
            x_missing_fake = x_fake[:, :, missing_indices, :]
            x_missing_real = x_full[:, :, missing_indices, :]
            loss_rec_missing = F.mse_loss(x_missing_fake, x_missing_real)
            val_rec_missing_losses.append(loss_rec_missing.item())

            if use_global_recon:
                # 子集重构损失
                x_subset_fake = x_fake[:, :, idx_subset, :]
                x_subset_real = x_full[:, :, idx_subset, :]
                loss_rec_subset = F.mse_loss(x_subset_fake, x_subset_real)
                val_rec_subset_losses.append(loss_rec_subset.item())

                # 组合损失
                loss_rec = loss_rec_missing + lambda_obs * loss_rec_subset
            else:
                loss_rec = loss_rec_missing

            val_rec_losses.append(loss_rec.item())

            if getattr(args, 'scaler', None) is not None:
                val_real_maes.append(
                    compute_real_space_mae(x_fake, x_full, missing_indices, args.scaler)
                )
                if use_global_recon:
                    # 计算子集的 MAE
                    x_subset_fake_real = x_fake[:, :, idx_subset, :] * args.scaler.std + args.scaler.mean
                    x_subset_real_real = x_full[:, :, idx_subset, :] * args.scaler.std + args.scaler.mean
                    mae_subset = torch.abs(x_subset_fake_real - x_subset_real_real).mean()
                    val_real_maes_subset.append(mae_subset.item())

    result = {
        'val_rec_loss': np.mean(val_rec_losses),
        'val_rec_loss_missing': np.mean(val_rec_missing_losses),
        'val_temporal_loss': np.mean(val_temporal_losses),
        'val_mae_real': np.mean(val_real_maes) if val_real_maes else float('nan'),
    }

    if use_global_recon and val_rec_subset_losses:
        result['val_rec_loss_subset'] = np.mean(val_rec_subset_losses)
        result['val_mae_real_subset'] = np.mean(val_real_maes_subset) if val_real_maes_subset else float('nan')

    return result

# ============================================================
# 主训练循环
# ============================================================

def train_loop(encoder, decoder, discriminator, temporal_loss_fn,
               train_loader, val_loader, args,
               latent_dyn_loss_fn=None):
    """主训练循环"""
    os.makedirs(args.save_dir, exist_ok=True)

    # 优化器
    g_params = list(encoder.parameters()) + list(decoder.parameters()) + list(temporal_loss_fn.parameters())
    if latent_dyn_loss_fn is not None:
        g_params += list(latent_dyn_loss_fn.parameters())

    opt_g = torch.optim.AdamW(
        g_params,
        lr=args.lr_g, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.lr_d, betas=(0.5, 0.999), weight_decay=args.weight_decay
    )

    grad_scaler = GradScaler(enabled=args.use_amp)

    history = {
        'train_d_loss': [], 'train_g_loss': [], 'train_temporal_loss': [],
        'val_rec_loss': [], 'val_temporal_loss': [], 'best_val_loss': float('inf'),
    }

    # 打印配置
    print("\n" + "=" * 80)
    print(" " * 15 + "GAN + Temporal Causal Pretraining (Encoder V4)")
    print("=" * 80)
    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    print(f"Num nodes: {args.num_nodes}")
    print(f"Subset ratio: {args.subset_ratio}")
    print(f"Subset perm refresh epochs: {args.perm_refresh_epochs}")
    print(f"Input dim: {args.in_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rates: G={args.lr_g}, D={args.lr_d}")
    print(f"Loss weights: λ_temporal={args.lambda_temporal}, λ_spatial={args.lambda_spatial}, λ_adv={args.lambda_adv}")
    print(f"Latent Dynamics: λ_latent_dyn={args.lambda_latent_dyn}(enabled={latent_dyn_loss_fn is not None and args.lambda_latent_dyn > 0})")
    print(f"Discriminator alpha: {args.disc_alpha}")
    print(f"Global reconstruction: {args.use_global_recon} λ_obs={args.lambda_obs}")
    print(f"\nModel parameters:")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Decoder: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"  Discriminator: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"  Temporal Loss: {sum(p.numel() for p in temporal_loss_fn.parameters()):,}")
    print("=" * 80 + "\n")

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 80)

        # 训练
        train_metrics = train_epoch(
            encoder, decoder, discriminator, temporal_loss_fn,
            train_loader, opt_g, opt_d, grad_scaler, args, epoch,
            latent_dyn_loss_fn=latent_dyn_loss_fn
        )

        # 验证
        val_metrics = validate(encoder, decoder, temporal_loss_fn, val_loader, args, epoch=epoch)

        # 记录历史
        history['train_d_loss'].append(train_metrics['d_loss'])
        history['train_g_loss'].append(train_metrics['g_loss'])
        history['train_temporal_loss'].append(train_metrics['loss_temporal'])
        history['val_rec_loss'].append(val_metrics['val_rec_loss'])
        history['val_temporal_loss'].append(val_metrics['val_temporal_loss'])

        # 打印总结
        print(f"\n[Epoch {epoch} Summary]")
        print(f"  Train D_loss: {train_metrics['d_loss']:.6f}")
        print(f"    - D_cond: {train_metrics['d_loss_cond']:.6f}")
        print(f"    - D_internal: {train_metrics['d_loss_internal']:.6f}")
        print(f"  Scores - cond: real={train_metrics['cond_score_real']:.3f}, fake={train_metrics['cond_score_fake']:.3f}")
        print(f"         - internal: real={train_metrics['internal_score_real']:.3f}, fake={train_metrics['internal_score_fake']:.3f}")
        print(f"  Train Temporal_loss: {train_metrics['loss_temporal']:.6f}")
        print(f"  Train Rec_loss: {train_metrics['g_loss_rec']:.6f}")
        if 'g_loss_rec_missing' in train_metrics and train_metrics['loss_latent_dyn'] > 0:
            print(f" Train Latent Dynamics Loss: {train_metrics['loss_latent_dyn']:.6f}")
            if 'latent_cahnge' in train_metrics:
                print(f"      * Latent Change: {train_metrics['latent_cahnge']:.6f}")   
            if 'latent_pred_error' in train_metrics:
                print(f"      * Latent Prediction Error: {train_metrics['latent_pred_error']:.6f}")
        if args.use_global_recon:
            print(f"    - Missing: {train_metrics['g_loss_rec_missing']:.6f}")
            print(f"    - Subset: {train_metrics['g_loss_rec_subset']:.6f}")
        print(f"  Train MAE (real): {train_metrics.get('g_mae_real', float('nan')):.6f}")
        print(f"  Val Rec_loss: {val_metrics['val_rec_loss']:.6f}")
        if args.use_global_recon:
            print(f"    - Missing: {val_metrics['val_rec_loss_missing']:.6f}")
            print(f"    - Subset: {val_metrics['val_rec_loss_subset']:.6f}")
        print(f"  Val Temporal_loss: {val_metrics['val_temporal_loss']:.6f}")
        print(f"  Val MAE_Missing (real): {val_metrics['val_mae_real']:.6f}")
        if args.use_global_recon:
            print(f"  Val MAE_Subset (real): {val_metrics['val_mae_real_subset']:.6f}")
        print(f"  Time: {train_metrics['epoch_time']:.2f}s")
        print(f"  Subset coverage: num_split={int(train_metrics.get('num_split', 0))}, "
                f"cover_nodes={int(train_metrics.get('subset_cover_nodes', 0))}, "
                f"perm_refreshed={bool(train_metrics.get('perm_refreshed', False))}")

        # 保存最佳模型
        combined_val_loss = val_metrics['val_rec_loss'] + val_metrics['val_temporal_loss']
        if combined_val_loss < history['best_val_loss']:
            history['best_val_loss'] = combined_val_loss
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            save_dict={
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'temporal_loss_state_dict': temporal_loss_fn.state_dict(),
                'val_rec_loss': val_metrics['val_rec_loss'],
                'val_temporal_loss': val_metrics['val_temporal_loss'],
                'args': vars(args),
            }
            if latent_dyn_loss_fn is not None:
                save_dict['latent_dyn_loss_state_dict'] = latent_dyn_loss_fn.state_dict()
            torch.save(save_dict, best_path)
            print(f"  → Best model saved! Val loss: {combined_val_loss:.6f}")

        # 定期保存检查点
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
            ckpt_dict={
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'temporal_loss_state_dict': temporal_loss_fn.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'opt_d_state_dict': opt_d.state_dict(),
                'history': history,
                'args': vars(args),
            }
            if latent_dyn_loss_fn is not None:
                ckpt_dict['latent_dyn_loss_state_dict'] = latent_dyn_loss_fn.state_dict()
            torch.save(ckpt_dict, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    print("\n" + "=" * 80)
    print(" " * 25 + "Training Completed!")
    print("=" * 80)
    print(f"Best validation loss: {history['best_val_loss']:.6f}")


def main():
    parser = argparse.ArgumentParser(description='GAN + Temporal Causal Pretraining (Encoder V4)')

    # 数据参数
    parser.add_argument('--data', type=str, required=True, help='数据路径')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--in_dim', type=int, default=None)
    parser.add_argument('--seq_in_len', type=int, default=12)
    parser.add_argument('--seq_out_len', type=int, default=12)

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr_g', type=float, default=1e-4, help='生成器学习率')
    parser.add_argument('--lr_d', type=float, default=5e-5, help='判别器学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # 损失权重
    parser.add_argument('--lambda_temporal', type=float, default=1.0, help='时序因果损失权重')
    parser.add_argument('--lambda_spatial', type=float, default=0.5, help='空间重构损失权重')
    parser.add_argument('--lambda_rec', type=float, default=1.0, help='重构损失权重（GAN内部）')
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='对抗损失权重')
    parser.add_argument('--use_global_recon', type=str_to_bool, default=True, help='是否使用全局重构（子集 + 缺失）')
    parser.add_argument('--lambda_obs', type=float, default=0.3, help='子集重构损失权重（相对于缺失节点）') 
    parser.add_argument('--lambda_latent_dyn', type=float, default=0.1, help='潜在动态损失权重')
    parser.add_argument('--use_latent_dyn', type=str_to_bool, default=True, help='是否使用潜在动态损失')
    parser.add_argument('--latent_dyn_version', type=str, default='v2', choices=['v1', 'v2'], help='v1=因果卷积，v2=MLP')

    # 子集配置
    parser.add_argument('--subset_ratio', type=float, default=0.15)
    parser.add_argument('--perm_refresh_epochs', type=int, default=100, help='子集 perm 刷新周期（单位：epoch）')


    # 判别器参数
    parser.add_argument('--disc_alpha', type=float, default=0.7,
                        help='条件判别损失权重（内部判别权重为1-alpha）')

    # 梯度截断
    parser.add_argument('--max_grad_norm_g', type=float, default=2.0)
    parser.add_argument('--max_grad_norm_d', type=float, default=1.0)

    # load_dataset 需要的参数
    parser.add_argument('--predefined_S', type=str_to_bool, default=False)
    parser.add_argument('--predefined_S_frac', type=int, default=15)

    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_gan_pretrain_v4')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--resume_ckpt', type=str, default=None)

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
    args.scaler = dataloader_dict['scaler']
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

    # Encoder V4（新架构）
    encoder = NodeAwareTemporalEncoder(
        num_nodes=args.num_nodes,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        device=args.device
    ).to(device)

    # Decoder（空间重构）
    decoder = STDecoder(
        in_dim=args.hidden_dim,
        out_dim=args.in_dim,
    ).to(device)

    # 混合判别器
    discriminator = create_discriminator(
        feature_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # 时序因果预训练损失
    temporal_loss_fn = NextStepPredictionLoss(
        hidden_dim=args.hidden_dim
    ).to(device)

    # 潜在动态损失
    latent_dyn_loss_fn = None
    if args.use_latent_dyn and args.lambda_latent_dyn > 0:
        latent_dyn_loss_fn = LatentDynamicsLoss(
            hidden_dim=args.hidden_dim,
            version=args.latent_dyn_version,
            detach_target=True
        ).to(device)
        print(f"✓ Using Latent Dynamics Loss: version={args.latent_dyn_version}, λ={args.lambda_latent_dyn}")
    print(f"✓ Models created")

    # 加载检查点（如果有）
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        if 'temporal_loss_state_dict' in ckpt:
            temporal_loss_fn.load_state_dict(ckpt['temporal_loss_state_dict'])
        print(f"✓ Loaded checkpoint from {args.resume_ckpt}")

    # 训练
    train_loop(
        encoder, decoder, discriminator, temporal_loss_fn,
        train_loader, val_loader, args,
        latent_dyn_loss_fn=latent_dyn_loss_fn
    )


if __name__ == "__main__":
    main()