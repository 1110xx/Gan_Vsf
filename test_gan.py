#!/usr/bin/env python3
"""
GAN 预训练模型 - 空间补全测试脚本

测试预训练模型的空间重构能力：
1. 输入子集节点数据
2. 重构全局节点数据
3. 评估缺失节点的重构精度

重点检查：
1. 重构精度（MAE/MSE）
2. 子集敏感性（不同子集是否产生不同输出）
3. 节点多样性（不同节点的重构是否有差异）
4. 可视化重构结果
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from util import load_dataset, StandardScaler


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def compute_metrics(x_recon, x_real, missing_indices, scaler=None):
    """
    计算重构精度指标

    Args:
        x_recon: (B, F, N, T) 重构结果
        x_real: (B, F, N, T) 真值
        missing_indices: 缺失节点索引
        scaler: 归一化器

    Returns:
        dict: 各项指标
    """
    # 提取缺失节点
    x_missing_recon = x_recon[:, :, missing_indices, :]
    x_missing_real = x_real[:, :, missing_indices, :]

    # 归一化空间的 MSE/MAE
    mse_norm = F.mse_loss(x_missing_recon, x_missing_real).item()
    mae_norm = torch.abs(x_missing_recon - x_missing_real).mean().item()

    # 原始尺度的 MAE
    if scaler is not None:
        x_recon_real = x_missing_recon * scaler['std'] + scaler['mean']
        x_real_real = x_missing_real * scaler['std'] + scaler['mean']
        mae_real = torch.abs(x_recon_real - x_real_real).mean().item()
    else:
        mae_real = mae_norm

    return {
        'mse_norm': mse_norm,
        'mae_norm': mae_norm,
        'mae_real': mae_real,
    }


def analyze_smoothness(x_recon, x_real, missing_indices):
    """
    分析重构结果的平滑度

    Args:
        x_recon: (B, F, N, T) 重构结果
        x_real: (B, F, N, T) 真值
        missing_indices: 缺失节点索引
    """
    # 提取缺失节点
    recon = x_recon[:, 0, missing_indices, :]  # (B, N_miss, T)
    real = x_real[:, 0, missing_indices, :]

    # 时间步之间的变化（一阶差分）
    recon_diff = torch.diff(recon, dim=-1)
    real_diff = torch.diff(real, dim=-1)

    recon_diff_std = recon_diff.std().item()
    real_diff_std = real_diff.std().item()

    # 节点之间的变化
    recon_node_std = recon.std(dim=1).mean().item()
    real_node_std = real.std(dim=1).mean().item()

    # 整体方差
    recon_overall_std = recon.std().item()
    real_overall_std = real.std().item()

    return {
        'recon_temporal_var': recon_diff_std,
        'real_temporal_var': real_diff_std,
        'temporal_ratio': recon_diff_std / (real_diff_std + 1e-8),
        'recon_node_var': recon_node_std,
        'real_node_var': real_node_std,
        'node_ratio': recon_node_std / (real_node_std + 1e-8),
        'recon_overall_var': recon_overall_std,
        'real_overall_var': real_overall_std,
        'overall_ratio': recon_overall_std / (real_overall_std + 1e-8),
    }


def analyze_subset_sensitivity(encoder, decoder, x_full, args, num_tests=5):
    """
    分析不同子集是否产生不同输出
    """
    num_subset = max(1, int(args.num_nodes * args.subset_ratio))

    all_recons = []

    with torch.no_grad():
        for _ in range(num_tests):
            # 不同的随机子集
            idx_subset = np.random.choice(args.num_nodes, size=num_subset, replace=False)
            idx_subset = torch.tensor(idx_subset, device=args.device)

            x_subset = x_full[:, :, idx_subset, :]

            h = encoder(x_subset, idx_subset)
            x_recon = decoder(h)

            all_recons.append(x_recon)

    # 计算不同子集重构之间的差异
    all_recons = torch.stack(all_recons, dim=0)  # (num_tests, B, F, N, T)

    # 跨子集的方差
    subset_var = all_recons.var(dim=0).mean().item()

    # 跨子集的最大差异
    max_diff = (all_recons.max(dim=0)[0] - all_recons.min(dim=0)[0]).mean().item()

    return {
        'subset_variance': subset_var,
        'max_subset_diff': max_diff,
        'is_sensitive': subset_var > 0.001,
    }


def analyze_node_diversity(x_recon, missing_indices):
    """
    分析不同节点重构的多样性
    """
    # 取缺失节点
    recon = x_recon[:, 0, missing_indices, :]  # (B, N_miss, T)
    B, N, T = recon.shape

    # 节点之间的相关性
    correlations = []
    for b in range(min(B, 5)):
        node_data = recon[b].cpu().numpy()  # (N, T)

        if node_data.shape[0] < 2:
            correlations.append(0.0)
            continue

        corr_matrix = np.corrcoef(node_data)

        if corr_matrix.ndim < 2:
            correlations.append(0.0)
            continue

        n = corr_matrix.shape[0]
        upper_tri = np.triu(corr_matrix, k=1)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_values = upper_tri[mask]
        avg_corr = float(valid_values.mean()) if len(valid_values) > 0 else 0.0
        correlations.append(avg_corr)

    avg_node_correlation = np.mean(correlations)

    # 节点均值的方差
    node_means = recon.mean(dim=-1)  # (B, N)
    node_mean_var = node_means.var(dim=-1).mean().item()

    return {
        'avg_node_correlation': avg_node_correlation,
        'node_mean_variance': node_mean_var,
        'is_diverse': avg_node_correlation < 0.95,
    }


def plot_reconstruction(x_recon, x_real, subset_indices, missing_indices, save_dir, scaler=None):
    """绘制重构结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 转换到原始尺度
    if scaler is not None:
        x_recon = x_recon * scaler['std'] + scaler['mean']
        x_real = x_real * scaler['std'] + scaler['mean']

    B, F, N, T = x_recon.shape

    # 1. 缺失节点重构对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Reconstruction vs Ground Truth (Missing Nodes)', fontsize=14)

    for i, ax in enumerate(axes.flatten()):
        if i >= len(missing_indices):
            break
        node_idx = missing_indices[i].item()
        ax.plot(x_real[0, 0, node_idx, :].cpu().numpy(), 'b-o', label='Ground Truth', markersize=4)
        ax.plot(x_recon[0, 0, node_idx, :].cpu().numpy(), 'r-x', label='Reconstruction', markersize=4)
        ax.set_title(f'Missing Node {node_idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recon_missing_nodes.png'), dpi=150)
    plt.close()

    # 2. 观测节点（应该很接近，因为是条件输入）
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Reconstruction vs Ground Truth (Observed Nodes)', fontsize=14)

    for i, ax in enumerate(axes.flatten()):
        if i >= len(subset_indices):
            break
        node_idx = subset_indices[i].item()
        ax.plot(x_real[0, 0, node_idx, :].cpu().numpy(), 'b-o', label='Ground Truth', markersize=4)
        ax.plot(x_recon[0, 0, node_idx, :].cpu().numpy(), 'g-x', label='Reconstruction', markersize=4)
        ax.set_title(f'Observed Node {node_idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recon_observed_nodes.png'), dpi=150)
    plt.close()

    # 3. 多样本同一节点
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Reconstruction vs Ground Truth (Different Samples)', fontsize=14)

    node_idx = missing_indices[0].item() if len(missing_indices) > 0 else 0
    for i, ax in enumerate(axes):
        if i >= B:
            break
        ax.plot(x_real[i, 0, node_idx, :].cpu().numpy(), 'b-o', label='Ground Truth', markersize=4)
        ax.plot(x_recon[i, 0, node_idx, :].cpu().numpy(), 'r-x', label='Reconstruction', markersize=4)
        ax.set_title(f'Sample {i}, Node {node_idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recon_multi_samples.png'), dpi=150)
    plt.close()

    # 4. 重构分布直方图
    x_missing_recon = x_recon[:, 0, missing_indices, :].cpu().numpy().flatten()
    x_missing_real = x_real[:, 0, missing_indices, :].cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(x_missing_recon, bins=50, alpha=0.7, label='Reconstruction')
    axes[0].hist(x_missing_real, bins=50, alpha=0.7, label='Ground Truth')
    axes[0].set_title('Value Distribution (Missing Nodes)')
    axes[0].legend()
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # 误差分布
    error = x_missing_recon - x_missing_real
    axes[1].hist(error, bins=50, alpha=0.7, color='red')
    axes[1].set_title('Reconstruction Error Distribution')
    axes[1].set_xlabel('Error')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recon_distribution.png'), dpi=150)
    plt.close()

    # 5. 热力图：节点 x 时间步
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 真值
    im0 = axes[0].imshow(x_real[0, 0, :, :].cpu().numpy(), aspect='auto', cmap='viridis')
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Node')
    plt.colorbar(im0, ax=axes[0])

    # 重构
    im1 = axes[1].imshow(x_recon[0, 0, :, :].cpu().numpy(), aspect='auto', cmap='viridis')
    axes[1].set_title('Reconstruction')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Node')
    plt.colorbar(im1, ax=axes[1])

    # 误差
    error_map = torch.abs(x_recon[0, 0, :, :] - x_real[0, 0, :, :]).cpu().numpy()
    im2 = axes[2].imshow(error_map, aspect='auto', cmap='Reds')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Node')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recon_heatmap.png'), dpi=150)
    plt.close()

    print(f"  可视化结果保存到: {save_dir}/")

def evaluate(encoder, decoder, dataloader, args, scaler=None, num_batches=10):
    """评估模型重构能力"""
    encoder.eval()
    decoder.eval()

    all_metrics = []
    all_smoothness = []

    num_subset = max(1, int(args.num_nodes * args.subset_ratio))

    print(f"\n评估配置:")
    print(f"  子集大小: {num_subset}/{args.num_nodes} (比例: {args.subset_ratio})")
    print(f"  缺失节点: {args.num_nodes - num_subset}")
    print(f"  评估批次: {num_batches}")

    first_batch_data = None

    with torch.no_grad():
        for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
            if iter_idx >= num_batches:
                break

            x_full = torch.Tensor(x).to(args.device)
            x_full = x_full.transpose(1, 3)  # (B, F, N, T)

            # 随机子集
            idx_subset = np.random.choice(args.num_nodes, size=num_subset, replace=False)
            idx_subset = torch.tensor(idx_subset, device=args.device)

            # 缺失节点索引
            subset_mask = torch.zeros(args.num_nodes, dtype=torch.bool, device=args.device)
            subset_mask[idx_subset] = True
            missing_indices = torch.where(~subset_mask)[0]

            x_subset = x_full[:, :, idx_subset, :]

            # 前向传播
            h = encoder(x_subset, idx_subset)
            x_recon = decoder(h)

            # 计算指标
            metrics = compute_metrics(x_recon, x_full, missing_indices, scaler)
            all_metrics.append(metrics)

            # 平滑度分析
            smoothness = analyze_smoothness(x_recon, x_full, missing_indices)
            all_smoothness.append(smoothness)

            # 保存第一批次数据用于可视化和详细分析
            if iter_idx == 0:
                first_batch_data = {
                    'x_full': x_full,
                    'x_recon': x_recon,
                    'idx_subset': idx_subset,
                    'missing_indices': missing_indices,
                }

                print(f"\n[第一批次统计]")
                print(f"  输入 x_full - Shape: {x_full.shape}")
                print(f"    Mean: {x_full.mean():.4f}, Std: {x_full.std():.4f}")
                print(f"  重构 x_recon - Shape: {x_recon.shape}")
                print(f"    Mean: {x_recon.mean():.4f}, Std: {x_recon.std():.4f}")
                print(f"  缺失节点重构 MSE: {metrics['mse_norm']:.6f}")
                print(f"  缺失节点重构 MAE (norm): {metrics['mae_norm']:.6f}")
                print(f"  缺失节点重构 MAE (real): {metrics['mae_real']:.4f}")

                # 子集敏感性分析
                sensitivity = analyze_subset_sensitivity(encoder, decoder, x_full, args)
                print(f"\n[子集敏感性分析]")
                print(f"  子集间方差: {sensitivity['subset_variance']:.6f}")
                print(f"  子集间最大差异: {sensitivity['max_subset_diff']:.6f}")
                print(f"  对子集敏感: {'✓ 是' if sensitivity['is_sensitive'] else '✗ 否 (问题!)'}")

                # 节点多样性分析
                diversity = analyze_node_diversity(x_recon, missing_indices)
                print(f"\n[节点多样性分析]")
                print(f"  节点间平均相关性: {diversity['avg_node_correlation']:.4f}")
                print(f"  节点均值方差: {diversity['node_mean_variance']:.6f}")
                print(f"  节点多样: {'✓ 是' if diversity['is_diverse'] else '✗ 否 (问题!)'}")

    # 汇总指标
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    avg_smoothness = {
        key: np.mean([s[key] for s in all_smoothness])
        for key in all_smoothness[0].keys()
    }

    return {
        'metrics': avg_metrics,
        'smoothness': avg_smoothness,
        'first_batch': first_batch_data,
    }


def main():
    parser = argparse.ArgumentParser(description='GAN 预训练模型 - 空间补全测试')

    # 数据参数
    parser.add_argument('--data', type=str, required=True, help='数据路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--in_dim', type=int, default=None)
    parser.add_argument('--seq_in_len', type=int, default=12)
    parser.add_argument('--seq_out_len', type=int, default=12)

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # 子集配置
    parser.add_argument('--subset_ratio', type=float, default=0.15)

    # load_dataset 需要的参数
    parser.add_argument('--predefined_S', type=str_to_bool, default=False)
    parser.add_argument('--predefined_S_frac', type=int, default=15)

    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_batches', type=int, default=10, help='评估批次数')
    parser.add_argument('--save_viz', type=str_to_bool, default=True)
    parser.add_argument('--viz_dir', type=str, default='viz_reconstruction')

    args = parser.parse_args()

    # 设备设置
    if args.device == 'cuda' and not torch.cuda.is_available():
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    device = torch.device(args.device)

    print("=" * 70)
    print(" " * 15 + "GAN 预训练模型 - 空间补全测试")
    print("=" * 70)
    print(f"数据路径: {args.data}")
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")

    # 加载数据
    print(f"\n加载数据...")
    dataloader_dict = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    test_loader = dataloader_dict['test_loader']
    scaler_obj = dataloader_dict['scaler']

    # 转换 scaler 格式
    if hasattr(scaler_obj, 'mean'):
        scaler = {'mean': scaler_obj.mean, 'std': scaler_obj.std}
    elif isinstance(scaler_obj, dict) and 'mean' in scaler_obj:
        scaler = scaler_obj
    else:
        x_train = dataloader_dict['train_loader'].xs
        scaler = {'mean': x_train[..., 0].mean(), 'std': x_train[..., 0].std()}

    args.num_nodes = test_loader.num_nodes
    if args.in_dim is None:
        args.in_dim = test_loader.xs.shape[-1]

    print(f"  节点数: {args.num_nodes}")
    print(f"  输入维度: {args.in_dim}")
    print(f"  测试样本数: {test_loader.size}")

    # 加载模型
    print(f"\n加载模型...")

    # 导入模型
    from model.encoder_v4 import NodeAwareTemporalEncoder
    from model.decoder_v2 import STDecoder

    # 创建模型
    encoder = NodeAwareTemporalEncoder(
        num_nodes=args.num_nodes,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        device=args.device
    ).to(device)

    decoder = STDecoder(
        in_dim=args.hidden_dim,
        out_dim=args.in_dim,
    ).to(device)

    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    print(f"  ✓ 模型加载成功!")
    if 'epoch' in checkpoint:
        print(f"    训练轮次: {checkpoint['epoch']}")
    if 'val_rec_loss' in checkpoint:
        print(f"    验证重构损失: {checkpoint['val_rec_loss']:.6f}")
    if 'val_temporal_loss' in checkpoint:
        print(f"    验证时序损失: {checkpoint['val_temporal_loss']:.6f}")

    print(f"  Encoder 参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Decoder 参数量: {sum(p.numel() for p in decoder.parameters()):,}")

    # 评估
    print("\n" + "=" * 70)
    print("开始评估...")
    print("=" * 70)

    results = evaluate(encoder, decoder, test_loader, args, scaler, args.num_batches)

    # 打印结果
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)

    m = results['metrics']
    print(f"\n[重构精度]")
    print(f"  MSE (归一化空间): {m['mse_norm']:.6f}")
    print(f"  MAE (归一化空间): {m['mae_norm']:.6f}")
    print(f"  MAE (原始尺度):   {m['mae_real']:.4f}")

    s = results['smoothness']
    print(f"\n[平滑度分析]")
    print(f"  时序变化 (重构): {s['recon_temporal_var']:.4f}")
    print(f"  时序变化 (真值): {s['real_temporal_var']:.4f}")
    print(f"  时序变化比率: {s['temporal_ratio']:.4f} {'✓' if 0.5 < s['temporal_ratio'] < 2.0 else '⚠ 异常'}")
    print(f"  节点差异 (重构): {s['recon_node_var']:.4f}")
    print(f"  节点差异 (真值): {s['real_node_var']:.4f}")
    print(f"  节点差异比率: {s['node_ratio']:.4f} {'✓' if 0.3 < s['node_ratio'] < 3.0 else '⚠ 异常'}")

    # 诊断
    print(f"\n[诊断]")
    is_smooth = s['temporal_ratio'] < 0.5
    if is_smooth:
        print("  ⚠ 警告: 重构结果可能过于平滑!")
    else:
        print("  ✓ 重构结果平滑度正常")

    # 可视化
    if args.save_viz and results['first_batch'] is not None:
        print(f"\n生成可视化...")
        fb = results['first_batch']
        plot_reconstruction(
            fb['x_recon'], fb['x_full'],
            fb['idx_subset'], fb['missing_indices'],
            args.viz_dir, scaler
        )

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()