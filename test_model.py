#!/usr/bin/env python3
"""
测试新模型的预测结果

重点检查：
1. 预测结果是否平滑（方差分析）
2. 不同子集是否产生不同输出（子集敏感性）
3. 不同节点的预测是否有差异（节点差异性）
4. 可视化预测曲线
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from util import load_dataset, masked_mae

# 新模型
from model.encoder_v4 import NodeAwareTemporalEncoder
from model.pred_decoder_v3 import CrossAttentionPredHead, SimplePredHead
from model.pred_decoder_v2 import TemporalPredHeadV2
from model.pred_decoder_v4 import AutoRegressivePredHeadV4
from model.pred_decoder_v5 import ResidualPredHead, StatefulCrossAttentionPredHead
from model.pred_decoder_v6 import create_pred_head_v6
from forecasters.MSTGCN import make_MSTGCN
from forecasters.ASTGCN import make_ASTGCN
from forecasters.MTGNN import gtnet


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def analyze_smoothness(pred, true):
    """
    分析预测结果的平滑度

    Args:
        pred: (B, N, T) 预测值
        true: (B, N, T) 真值

    Returns:
        dict: 平滑度指标
    """
    # 1. 时间步之间的变化（一阶差分）
    pred_diff = torch.diff(pred, dim=-1)  # (B, N, T-1)
    true_diff = torch.diff(true, dim=-1)

    pred_diff_std = pred_diff.std().item()
    true_diff_std = true_diff.std().item()

    # 2. 节点之间的变化
    pred_node_std = pred.std(dim=1).mean().item()  # 每个样本的节点方差，然后平均
    true_node_std = true.std(dim=1).mean().item()

    # 3. 整体方差
    pred_overall_std = pred.std().item()
    true_overall_std = true.std().item()

    # 4. 平滑度比率（越接近 1 越好）
    temporal_ratio = pred_diff_std / (true_diff_std + 1e-8)
    node_ratio = pred_node_std / (true_node_std + 1e-8)
    overall_ratio = pred_overall_std / (true_overall_std + 1e-8)

    return {
        'pred_temporal_var': pred_diff_std,
        'true_temporal_var': true_diff_std,
        'temporal_ratio': temporal_ratio,
        'pred_node_var': pred_node_std,
        'true_node_var': true_node_std,
        'node_ratio': node_ratio,
        'pred_overall_var': pred_overall_std,
        'true_overall_var': true_overall_std,
        'overall_ratio': overall_ratio,
    }


def analyze_subset_sensitivity(encoder, pred_head, x_full, y_full, args, scaler, num_tests=5):
    """
    分析不同子集是否产生不同输出

    如果模型正常，不同子集应该产生不同的预测
    如果所有子集产生相同输出，说明模型有问题
    """
    num_subset = max(1, int(args.num_nodes * args.subset_ratio))

    all_preds = []

    with torch.no_grad():
        for _ in range(num_tests):
            # 不同的随机子集
            idx_subset = np.random.choice(args.num_nodes, size=num_subset, replace=False)
            idx_subset = torch.tensor(idx_subset, device=args.device)

            x_subset = x_full[:, :, idx_subset, :]

            # 提取最后一个时间步（用于 V5/V6 预测头）
            # 关键修复：必须按 idx_subset 切片，确保 x_last 与当前子集对应
            x_last = x_full[:, 0:1, idx_subset, -1]  # (B, 1, N_subset)
            if scaler is not None:
                x_last_norm = (x_last - scaler['mean']) / scaler['std']
            else:
                x_last_norm = x_last

            h_all = encoder(x_subset, idx_subset)
            # V4/AR 预测头推理时不使用 Teacher Forcing
            if args.pred_head_type in ['v4', 'ar']:
                pred_all = pred_head(h_all, y_true=None, teacher_forcing_ratio=0.0)
            elif args.pred_head_type in ['v5', 'residual', 'v5_stateful', 'stateful']:
                pred_all = pred_head(h_all, x_last=x_last_norm)
            elif args.pred_head_type in ['v6_direct', 'direct', 'v6_tcn', 'tcn', 'v6_hybrid', 'v6', 'hybrid']:
                pred_all = pred_head(h_all, x_last=x_last_norm, node_idx=idx_subset)
            else:
                pred_all = pred_head(h_all)

            # 取所有节点的预测
            pred = pred_all[:, 0, :, :]  # (B, N_all, T_out)
            all_preds.append(pred)

    # 计算不同子集预测之间的差异
    all_preds = torch.stack(all_preds, dim=0)  # (num_tests, B, N, T)

    # 跨子集的方差
    subset_var = all_preds.var(dim=0).mean().item()

    # 跨子集的最大差异
    max_diff = (all_preds.max(dim=0)[0] - all_preds.min(dim=0)[0]).mean().item()

    return {
        'subset_variance': subset_var,
        'max_subset_diff': max_diff,
        'is_sensitive': subset_var > 0.01,  # 阈值判断
    }


def analyze_node_diversity(pred):
    """
    分析不同节点预测的多样性

    Args:
        pred: (B, N, T) 预测值
    """
    B, N, T = pred.shape

    # 1. 节点之间的相关性
    # 将每个节点的时序展平
    pred_flat = pred.reshape(B, N, -1)  # (B, N, T)

    # 计算节点间的平均相关性
    correlations = []
    for b in range(min(B, 5)):  # 只取前5个样本
        node_data = pred_flat[b].cpu().numpy()  # (N, T)

        # 检查节点数是否足够计算相关性
        if node_data.shape[0] < 2:
            correlations.append(0.0)
            continue

        corr_matrix = np.corrcoef(node_data)

        # 确保 corr_matrix 是 2D 数组
        if corr_matrix.ndim < 2:
            correlations.append(0.0)
            continue

        # 取上三角的平均值（不包括对角线）
        n = corr_matrix.shape[0]
        upper_tri = np.triu(corr_matrix, k=1)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_values = upper_tri[mask]
        avg_corr = float(valid_values.mean()) if len(valid_values) > 0 else 0.0
        correlations.append(avg_corr)

    avg_node_correlation = np.mean(correlations)

    # 2. 节点预测的方差
    node_means = pred.mean(dim=-1)  # (B, N)
    node_mean_var = node_means.var(dim=-1).mean().item()

    return {
        'avg_node_correlation': avg_node_correlation,
        'node_mean_variance': node_mean_var,
        'is_diverse': avg_node_correlation < 0.9,  # 相关性不应该太高
    }


def plot_predictions(pred, true, save_dir, prefix='pred'):
    """绘制预测结果"""
    os.makedirs(save_dir, exist_ok=True)

    B, N, T = pred.shape

    # 1. 多个节点对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Prediction vs Ground Truth (Different Nodes)', fontsize=14)

    for i, ax in enumerate(axes.flatten()):
        if i >= N:
            break
        ax.plot(true[0, i, :].cpu().numpy(), 'b-o', label='Ground Truth', markersize=4)
        ax.plot(pred[0, i, :].cpu().numpy(), 'r-x', label='Prediction', markersize=4)
        ax.set_title(f'Node {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_multi_nodes.png'), dpi=150)
    plt.close()

    # 2. 多个样本同一节点
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Prediction vs Ground Truth (Different Samples, Node 0)', fontsize=14)

    for i, ax in enumerate(axes):
        if i >= B:
            break
        ax.plot(true[i, 0, :].cpu().numpy(), 'b-o', label='Ground Truth', markersize=4)
        ax.plot(pred[i, 0, :].cpu().numpy(), 'r-x', label='Prediction', markersize=4)
        ax.set_title(f'Sample {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_multi_samples.png'), dpi=150)
    plt.close()

    # 3. 预测分布直方图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(pred.cpu().numpy().flatten(), bins=50, alpha=0.7, label='Prediction')
    axes[0].hist(true.cpu().numpy().flatten(), bins=50, alpha=0.7, label='Ground Truth')
    axes[0].set_title('Value Distribution')
    axes[0].legend()
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # 一阶差分分布
    pred_diff = torch.diff(pred, dim=-1).cpu().numpy().flatten()
    true_diff = torch.diff(true, dim=-1).cpu().numpy().flatten()
    axes[1].hist(pred_diff, bins=50, alpha=0.7, label='Prediction Diff')
    axes[1].hist(true_diff, bins=50, alpha=0.7, label='Ground Truth Diff')
    axes[1].set_title('Temporal Change Distribution')
    axes[1].legend()
    axes[1].set_xlabel('Δ Value')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_distribution.png'), dpi=150)
    plt.close()

    print(f"  Plots saved to {save_dir}/")

def evaluate(encoder, pred_head, dataloader, args, scaler=None, num_batches=10):
    """评估模型"""
    encoder.eval()
    if pred_head is not None:
        pred_head.eval()

    all_preds = []
    all_trues = []
    mae_list = []
    smoothness_results = []

    num_subset = max(1, int(args.num_nodes * args.subset_ratio))

    print(f"\n评估配置:")
    print(f"  子集大小: {num_subset}/{args.num_nodes} (比例: {args.subset_ratio})")
    print(f"  评估批次: {num_batches}")

    with torch.no_grad():
        for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
            if iter_idx >= num_batches:
                break

            x_full = torch.Tensor(x).to(args.device)
            x_full = x_full.transpose(1, 3)  # (B, F, N, T)

            y_real = torch.Tensor(y).to(args.device)
            if y_real.dim() == 4:
                y_real = y_real[..., 0]

            # 随机子集
            idx_subset = np.random.choice(args.num_nodes, size=num_subset, replace=False)
            idx_subset = torch.tensor(idx_subset, device=args.device)

            x_subset = x_full[:, :, idx_subset, :]

            # 提取最后一个时间步（用于 V5/V6 预测头）
            # 关键修复：必须按 idx_subset 切片，确保 x_last 与当前子集对应
            x_last = x_full[:, 0:1, idx_subset, -1]  # (B, 1, N_subset)
            if scaler is not None:
                x_last_norm = (x_last - scaler['mean']) / scaler['std']
            else:
                x_last_norm = x_last

            # 前向传播
            if pred_head is None:
                # MSTGCN/ASTGCN/MTGNN: 必须使用全图
                # x_full: (B, F, N, T)

                if args.model_type == 'mtgnn':
                    # MTGNN forward signature: input, idx=None, args=None
                    # It expects args to have .adj_identity_train_test

                    # Create a dummy args object for MTGNN forward
                    class DummyArgs:
                        def __init__(self):
                            self.adj_identity_train_test = False
                            self.device = args.device

                    dummy_args = DummyArgs()
                    pred_all = encoder(x_full, None, dummy_args)
                else:
                    pred_all = encoder(x_full, None, None) # (B, T_out, N, F_out)

                # Check output shape
                # ASTGCN/MSTGCN output: (B, T_out, N, F_in)
                # MTGNN output: (B, T_out, N, 1) usually

                # Standardize to (B, F, N, T) or (B, 1, N, T)
                if pred_all.shape[-1] == 1 or pred_all.shape[-1] == args.in_dim:
                     pred_all = pred_all.permute(0, 3, 2, 1) # (B, F, N, T)

                # 提取子集
                pred_subset = pred_all[:, 0, idx_subset, :] # (B, N_subset, T_out)
            else:
                h_all = encoder(x_subset, idx_subset)
                # V4/AR 预测头推理时不使用 Teacher Forcing
                if args.pred_head_type in ['v4', 'ar']:
                    pred_all = pred_head(h_all, y_true=None, teacher_forcing_ratio=0.0)
                elif args.pred_head_type in ['v5', 'residual', 'v5_stateful', 'stateful']:
                    pred_all = pred_head(h_all, x_last=x_last_norm)
                elif args.pred_head_type in ['v6']:
                    pred_all = pred_head(h_all, x_last=x_last_norm, node_idx=idx_subset)
                else:
                    pred_all = pred_head(h_all)

                # 提取子集预测
                # 注意：如果 pred_all 已经是子集大小 (B, 1, N_subset, T)，那么这里其实不需要 idx_subset 索引
                # 假设 pred_all 是 (B, 1, N_subset, T)
                if pred_all.shape[2] == num_subset:
                    pred_subset = pred_all[:, 0, :, :]
                else:
                    pred_subset = pred_all[:, 0, idx_subset, :]  # (B, N_subset, T_out)

            # 逆变换
            if scaler is not None:
                pred_real = pred_subset * scaler['std'] + scaler['mean']
            else:
                pred_real = pred_subset

            # 真值
            y_subset_real = y_real[:, :, idx_subset].transpose(1, 2)  # (B, N_subset, T)

            # MAE
            mae_val, _ = masked_mae(pred_real, y_subset_real, null_val=0.0)
            mae_list.append(mae_val.item())

            # 平滑度分析
            smoothness = analyze_smoothness(pred_real, y_subset_real)
            smoothness_results.append(smoothness)

            all_preds.append(pred_real)
            all_trues.append(y_subset_real)

            # 第一批次的详细统计
            if iter_idx == 0:
                print(f"\n[第一批次统计]")
                print(f"  预测 (原始尺度) - Mean: {pred_real.mean():.4f}, Std: {pred_real.std():.4f}")
                print(f"  真值 (原始尺度) - Mean: {y_subset_real.mean():.4f}, Std: {y_subset_real.std():.4f}")

                # 子集敏感性分析 (仅当有 pred_head 时)
                if pred_head is not None:
                    sensitivity = analyze_subset_sensitivity(
                        encoder, pred_head, x_full, y_real, args, scaler
                    )
                    print(f"\n[子集敏感性分析]")
                    print(f"  子集间方差: {sensitivity['subset_variance']:.6f}")
                    print(f"  子集间最大差异: {sensitivity['max_subset_diff']:.6f}")
                    print(f"  对子集敏感: {'✓ 是' if sensitivity['is_sensitive'] else '✗ 否 (问题!)'}")

                # 节点多样性分析
                diversity = analyze_node_diversity(pred_real)
                print(f"\n[节点多样性分析]")
                print(f"  节点间平均相关性: {diversity['avg_node_correlation']:.4f}")
                print(f"  节点均值方差: {diversity['node_mean_variance']:.6f}")
                print(f"  节点多样: {'✓ 是' if diversity['is_diverse'] else '✗ 否 (问题!)'}")

    # 汇总平滑度
    avg_smoothness = {
        key: np.mean([s[key] for s in smoothness_results])
        for key in smoothness_results[0].keys()
    }

    # 合并所有批次
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    return {
        'mae': np.mean(mae_list),
        'smoothness': avg_smoothness,
        'preds': all_preds,
        'trues': all_trues,
    }


def create_pred_head(args, device):
    """创建预测头"""
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
    elif args.pred_head_type in ['v4', 'ar']:
        return AutoRegressivePredHeadV4(
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(device)
    elif args.pred_head_type in ['v5', 'residual']:
        return ResidualPredHead(
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            use_residual=True
        ).to(device)
    elif args.pred_head_type in ['v5_stateful', 'stateful']:
        return StatefulCrossAttentionPredHead(
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(device)
    elif args.pred_head_type in [ 'v6']:
        return create_pred_head_v6(
            head_type='hybrid',
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            use_cross_attn=True
        ).to(device)
    else:
        raise ValueError(f"Unknown pred_head_type: {args.pred_head_type}")

def main():
    parser = argparse.ArgumentParser(description='测试新模型')

    parser.add_argument('--data', type=str, required=True, help='数据路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--model_type', type=str, default='encoder_decoder', choices=['encoder_decoder', 'mstgcn', 'astgcn', 'mtgnn'], help='模型类型')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--in_dim', type=int, default=None)
    parser.add_argument('--seq_in_len', type=int, default=12)
    parser.add_argument('--seq_out_len', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--pred_head_type', type=str, default='cross_attn',
                        choices=['cross_attn', 'simple', 'v2', 'v4', 'ar',
                                 'v5', 'residual', 'v5_stateful', 'stateful',
                                 'v6'],
                        help='预测头类型: v6=直接映射(推荐v6_hybrid)')
    parser.add_argument('--subset_ratio', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_viz', type=str_to_bool, default=True)
    parser.add_argument('--viz_dir', type=str, default='viz_new_model')
    parser.add_argument('--predefined_S', type=str_to_bool, default=False)
    parser.add_argument('--predefined_S_frac', type=int, default=15)

    args = parser.parse_args()

    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    # 加载数据
    print(f"加载数据: {args.data}...")
    dataloader_dict = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    test_loader = dataloader_dict['test_loader']
    scaler_obj = dataloader_dict['scaler']

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

    # 加载模型
    print(f"\n加载模型: {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    encoder = None
    pred_head = None

    if args.model_type == 'mstgcn':
        print("  Model Type: MSTGCN")
        args_saved = checkpoint.get('args', {})
        adj_mx = checkpoint.get('adj_mx', None)

        # 如果 checkpoint 没有 adj_mx，尝试构建一个默认的
        if adj_mx is None:
            print("  Warning: checkpoint 中没有 adj_mx，使用单位矩阵代替")
            adj_mx = np.eye(args.num_nodes)

        # 确保 adj_mx 维度正确
        if adj_mx.shape[0] != args.num_nodes:
             print(f"  Warning: adj_mx shape {adj_mx.shape} != num_nodes {args.num_nodes}. Resizing/Identity.")
             if adj_mx.shape[0] > args.num_nodes:
                 adj_mx = adj_mx[:args.num_nodes, :args.num_nodes]
             else:
                 adj_mx = np.eye(args.num_nodes)

        model = make_MSTGCN(
            device,
            args_saved.get('nb_block', 2),
            args.in_dim,
            args_saved.get('K', 3),
            args_saved.get('nb_chev_filter', 64),
            args_saved.get('nb_time_filter', 64),
            args_saved.get('time_strides', 1),
            adj_mx,
            args.seq_out_len,
            args.seq_in_len
        )
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容旧格式或直接保存的 state_dict
            try:
                model.load_state_dict(checkpoint)
            except:
                print("Error loading state dict. Keys: ", checkpoint.keys() if isinstance(checkpoint, dict) else "Not dict")
                raise

        encoder = model # MSTGCN serves as the whole model
        pred_head = None # Signal to evaluate that there is no separate pred_head

    elif args.model_type == 'astgcn':
        print("  Model Type: ASTGCN")
        args_saved = checkpoint.get('args', {})
        adj_mx = checkpoint.get('adj_mx', None)

        if adj_mx is None:
             adj_mx = np.eye(args.num_nodes)

        if adj_mx.shape[0] != args.num_nodes:
             if adj_mx.shape[0] > args.num_nodes:
                 adj_mx = adj_mx[:args.num_nodes, :args.num_nodes]
             else:
                 adj_mx = np.eye(args.num_nodes)

        model = make_ASTGCN(
            device,
            args_saved.get('nb_block', 2),
            args.in_dim,
            args_saved.get('K', 3),
            args_saved.get('nb_chev_filter', 64),
            args_saved.get('nb_time_filter', 64),
            args_saved.get('time_strides', 1),
            adj_mx,
            args.seq_out_len,
            args.seq_in_len,
            args.num_nodes
        )
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            try:
                model.load_state_dict(checkpoint)
            except:
                raise

        encoder = model
        pred_head = None

    elif args.model_type == 'mtgnn':
        print("  Model Type: MTGNN")
        args_saved = checkpoint.get('args', {})

        # MTGNN doesn't store adj in checkpoint usually as it learns it or constructs it

        model = gtnet(
            gcn_true=args_saved.get('gcn_true', True),
            buildA_true=args_saved.get('buildA_true', True),
            gcn_depth=args_saved.get('gcn_depth', 2),
            num_nodes=args.num_nodes,
            device=device,
            predefined_A=None,
            static_feat=None,
            dropout=args_saved.get('dropout', 0.3),
            subgraph_size=args_saved.get('subgraph_size', 20),
            node_dim=args_saved.get('node_dim', 40),
            dilation_exponential=args_saved.get('dilation_exponential', 1),
            conv_channels=args_saved.get('conv_channels', 32),
            residual_channels=args_saved.get('residual_channels', 32),
            skip_channels=args_saved.get('skip_channels', 64),
            end_channels=args_saved.get('end_channels', 128),
            seq_length=args.seq_in_len,
            in_dim=args.in_dim,
            out_dim=args.seq_out_len,
            layers=args_saved.get('layers', 3),
            propalpha=args_saved.get('propalpha', 0.05),
            tanhalpha=args_saved.get('tanhalpha', 3),
            layer_norm_affline=args_saved.get('layer_norm_affline', True)
        )

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            try:
                model.load_state_dict(checkpoint)
            except:
                raise

        encoder = model
        pred_head = None

    else:
        # 创建模型
        encoder = NodeAwareTemporalEncoder(
            num_nodes=args.num_nodes,
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=0.1,
            device=args.device
        ).to(device)

        pred_head = create_pred_head(args, device)

        # 加载模型权重
        try:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        except RuntimeError as e:
            print(f"Warning: Strict loading failed. Retrying with strict=False...")
            encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)

        pred_head.load_state_dict(checkpoint['pred_head_state_dict'])

    print("模型加载成功!")
    if isinstance(checkpoint, dict):
        if 'val_mae' in checkpoint:
            print(f"  验证 MAE: {checkpoint['val_mae']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  训练轮次: {checkpoint['epoch']}")

    # 评估
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)

    results = evaluate(encoder, pred_head, test_loader, args, scaler, num_batches=10)

    # ... (rest is same)

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    print(f"\n[预测精度]")
    print(f"  MAE: {results['mae']:.4f}")

    print(f"\n[平滑度分析]")
    s = results['smoothness']
    print(f"  时序变化 (预测): {s['pred_temporal_var']:.4f}")
    print(f"  时序变化 (真值): {s['true_temporal_var']:.4f}")
    print(f"  时序变化比率: {s['temporal_ratio']:.4f} {'✓' if 0.5 < s['temporal_ratio'] < 2.0 else '⚠ 异常'}")
    print(f"  节点差异 (预测): {s['pred_node_var']:.4f}")
    print(f"  节点差异 (真值): {s['true_node_var']:.4f}")
    print(f"  节点差异比率: {s['node_ratio']:.4f} {'✓' if 0.3 < s['node_ratio'] < 3.0 else '⚠ 异常'}")
    print(f"  整体方差比率: {s['overall_ratio']:.4f}")

    # 判断是否过度平滑
    is_smooth = s['temporal_ratio'] < 0.5
    print(f"\n[诊断]")
    if is_smooth:
        print("  ⚠ 警告: 预测结果可能过于平滑!")
        print("    - 时序变化比率 < 0.5 表示预测变化太小")
    else:
        print("  ✓ 预测结果平滑度正常")

    # 可视化
    if args.save_viz:
        print(f"\n生成可视化...")
        plot_predictions(results['preds'], results['trues'], args.viz_dir)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()