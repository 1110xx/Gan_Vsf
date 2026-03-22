#!/usr/bin/env python3
import os
import json
import math
import time
import glob
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from util import load_dataset, masked_mae, masked_rmse
from model.encoder_v4 import NodeAwareTemporalEncoder
from model.pred_decoder_v6 import create_pred_head_v6
from model.pred_decoder_v8 import create_pred_head_v8


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    if value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_pred_head(eval_args, device, shared_node_embed=None):
    if eval_args.pred_head_type == 'v6':
        return create_pred_head_v6(
            head_type='hybrid',
            hidden_dim=eval_args.hidden_dim,
            num_nodes=eval_args.num_nodes,
            seq_in_len=eval_args.seq_in_len,
            seq_out_len=eval_args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            use_cross_attn=getattr(eval_args, 'use_cross_attn', True),
            use_residual_pred=getattr(eval_args, 'use_residual_pred', True),
            shared_node_embed=shared_node_embed,
        ).to(device)

    if eval_args.pred_head_type == 'v8':
        return create_pred_head_v8(
            hidden_dim=eval_args.hidden_dim,
            num_nodes=eval_args.num_nodes,
            seq_in_len=eval_args.seq_in_len,
            seq_out_len=eval_args.seq_out_len,
            n_layers=getattr(eval_args, 'pred_n_layers', 4),
            kernel_size=getattr(eval_args, 'tcn_kernel_size', 3),
            dropout=0.1,
            use_residual_pred=getattr(eval_args, 'use_residual_pred', False),
        ).to(device)

    raise ValueError(f"Unknown pred_head_type: {eval_args.pred_head_type}. Only 'v6' and 'v8' are supported.")


def parse_args():
    parser = argparse.ArgumentParser(description='多次运行下游测试（真实空间 MAE/RMSE）')

    parser.add_argument('--data', type=str, required=True, help='数据路径')
    parser.add_argument('--model_path', type=str, default=None, help='单个模型路径（best_model.pt）')
    parser.add_argument('--model_glob', type=str, default=None, help='模型 glob，如 ./xx/run_*/best_model.pt')
    parser.add_argument('--mode', type=str, default='finetune', choices=['frozen', 'finetune', 'scratch'])

    parser.add_argument('--runs', type=int, default=10, help='外层 run 数（通常=模型个数）')
    parser.add_argument('--random_node_idx_split_runs', type=int, default=100, help='每个 run 的随机子集次数')
    parser.add_argument('--seed', type=int, default=2024, help='基础随机种子')
    parser.add_argument('--seed_stride', type=int, default=1000, help='split 种子偏移系数')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--in_dim', type=int, default=None)
    parser.add_argument('--seq_in_len', type=int, default=12)
    parser.add_argument('--seq_out_len', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--subset_ratio', type=float, default=0.15, help='当不使用百分比范围时使用')
    parser.add_argument('--lower_limit_random_node_selections', type=int, default=15)
    parser.add_argument('--upper_limit_random_node_selections', type=int, default=15)

    parser.add_argument('--predefined_S', type=str_to_bool, default=False)
    parser.add_argument('--predefined_S_frac', type=int, default=15)

    parser.add_argument('--output_dir', type=str, default='./multi_results')
    parser.add_argument('--save_detail', type=str_to_bool, default=True)
    parser.add_argument('--print_every_split', type=int, default=20)

    return parser.parse_args()


def resolve_model_paths(args):
    paths = []
    if args.model_glob:
        paths.extend(sorted(glob.glob(args.model_glob)))
    if args.model_path:
        paths.append(args.model_path)

    paths = [p for p in paths if p]
    if not paths:
        raise ValueError('请至少提供 --model_path 或 --model_glob')

    # 去重并保持顺序
    uniq = []
    seen = set()
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    paths = uniq

    if args.runs > 0 and len(paths) > args.runs:
        paths = paths[:args.runs]

    return paths


def merge_eval_args(cli_args, ckpt_args, num_nodes, in_dim):
    def pick(key, default=None):
        if key in ckpt_args:
            return ckpt_args[key]
        return getattr(cli_args, key, default)

    merged = {
        'num_nodes': num_nodes,
        'in_dim': in_dim,
        'seq_in_len': pick('seq_in_len', cli_args.seq_in_len),
        'seq_out_len': pick('seq_out_len', cli_args.seq_out_len),
        'hidden_dim': pick('hidden_dim', 64),
        'n_layers': pick('n_layers', 4),
        'n_heads': pick('n_heads', 4),
        'dropout': pick('dropout', 0.1),
        'pred_head_type': pick('pred_head_type', 'v8'),
        'pred_n_layers': pick('pred_n_layers', 4),
        'tcn_kernel_size': pick('tcn_kernel_size', 3),
        'use_residual_pred': pick('use_residual_pred', False),
        'use_cross_attn': pick('use_cross_attn', True),
        'use_clean_obs': pick('use_clean_obs', False),
        'pretrain_ckpt': pick('pretrain_ckpt', None),
        'device': cli_args.device,
    }
    return SimpleNamespace(**merged)


def build_models(eval_args):
    device = torch.device(eval_args.device)
    encoder = NodeAwareTemporalEncoder(
        num_nodes=eval_args.num_nodes,
        in_dim=eval_args.in_dim,
        hidden_dim=eval_args.hidden_dim,
        n_layers=eval_args.n_layers,
        n_heads=eval_args.n_heads,
        dropout=eval_args.dropout,
        device=eval_args.device,
    ).to(device)

    shared_node_embed = None
    if eval_args.pred_head_type == 'v6':
        shared_node_embed = encoder.get_shared_node_embed()

    pred_head = create_pred_head(eval_args, device, shared_node_embed=shared_node_embed)
    return encoder, pred_head


def load_scaler_dict(scaler_obj, train_loader):
    if hasattr(scaler_obj, 'mean'):
        return {'mean': scaler_obj.mean, 'std': scaler_obj.std}
    if isinstance(scaler_obj, dict) and 'mean' in scaler_obj and 'std' in scaler_obj:
        return scaler_obj
    x_train = train_loader.xs
    return {'mean': x_train[..., 0].mean(), 'std': x_train[..., 0].std()}


def select_subset(num_nodes, rng, args):
    lb = int(args.lower_limit_random_node_selections)
    ub = int(args.upper_limit_random_node_selections)
    if lb > 0 and ub >= lb:
        percent = int(rng.integers(lb, ub + 1))
        k = max(1, math.ceil(num_nodes * (percent / 100.0)))
    else:
        k = max(1, int(num_nodes * args.subset_ratio))
    idx = rng.choice(num_nodes, size=k, replace=False)
    return np.sort(idx)


def eval_one_split(encoder, pred_head, test_loader, eval_args, idx_subset_np, scaler):
    encoder.eval()
    pred_head.eval()

    idx_subset = torch.tensor(idx_subset_np, device=eval_args.device)
    horizon_mae = [[] for _ in range(eval_args.seq_out_len)]
    horizon_rmse = [[] for _ in range(eval_args.seq_out_len)]

    with torch.no_grad():
        for x, y in test_loader.get_iterator():
            x_full = torch.tensor(x, dtype=torch.float32, device=eval_args.device).transpose(1, 3)
            y_real = torch.tensor(y, dtype=torch.float32, device=eval_args.device)
            if y_real.dim() == 4:
                y_real = y_real[..., 0]

            x_subset = x_full[:, :, idx_subset, :]

            if getattr(eval_args, 'use_clean_obs', False) and getattr(eval_args, 'pretrain_ckpt', None):
                h_all, h_obs_clean = encoder(x_subset, idx_subset, return_obs_clean=True)
                h_all = encoder.replace_obs_with_clean(h_all, h_obs_clean, idx_subset)
            else:
                h_all = encoder(x_subset, idx_subset)

            x_last_full = x_full[:, 0:1, :, -1]
            if eval_args.pred_head_type == 'v8':
                pred_all = pred_head(h_all, x_last=x_last_full, return_latent=False)
            else:
                pred_all = pred_head(h_all, x_last=x_last_full)

            pred_subset = pred_all[:, 0, idx_subset, :]
            pred_real = pred_subset * scaler['std'] + scaler['mean']
            y_subset_real = y_real[:, :, idx_subset].transpose(1, 2)

            for h in range(eval_args.seq_out_len):
                mae_h, _ = masked_mae(pred_real[:, :, h], y_subset_real[:, :, h], null_val=0.0)
                rmse_h, _ = masked_rmse(pred_real[:, :, h], y_subset_real[:, :, h], null_val=0.0)
                horizon_mae[h].append(float(mae_h.item()))
                horizon_rmse[h].append(float(rmse_h.item()))

    mae_split = [float(np.mean(v)) if len(v) > 0 else float('nan') for v in horizon_mae]
    rmse_split = [float(np.mean(v)) if len(v) > 0 else float('nan') for v in horizon_rmse]
    return mae_split, rmse_split


def aggregate_metrics(mae_all, rmse_all):
    mae_arr = np.array(mae_all, dtype=np.float64)
    rmse_arr = np.array(rmse_all, dtype=np.float64)

    amae = np.mean(mae_arr, axis=0)
    armse = np.mean(rmse_arr, axis=0)
    smae = np.std(mae_arr, axis=0)
    srmse = np.std(rmse_arr, axis=0)

    return {
        'amae': amae,
        'armse': armse,
        'smae': smae,
        'srmse': srmse,
        'all_runs_avermae': float(np.mean(amae)),
        'all_runs_avermse': float(np.mean(armse)),
        'all_runs_aver_stdmae': float(np.mean(smae)),
        'all_runs_aver_stdrmse': float(np.mean(srmse)),
    }


def to_serializable_stats(stats):
    out = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA not available, using CPU')
        args.device = 'cpu'

    set_seed(args.seed)

    start_time = time.time()
    model_paths = resolve_model_paths(args)
    if len(model_paths) < args.runs:
        print(f"[Warn] 模型数量({len(model_paths)}) < runs({args.runs})，将按实际模型数量运行。")

    print(f"\n[Info] Mode: {args.mode}")
    print(f"[Info] Models: {len(model_paths)}")
    print(f"[Info] random_node_idx_split_runs: {args.random_node_idx_split_runs}")

    print(f"\nLoading data from {args.data}...")
    dataloader_dict = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    train_loader = dataloader_dict['train_loader']
    test_loader = dataloader_dict['test_loader']
    scaler = load_scaler_dict(dataloader_dict['scaler'], train_loader)

    num_nodes = test_loader.num_nodes
    if args.in_dim is None:
        args.in_dim = test_loader.xs.shape[-1]

    all_mae = []
    all_rmse = []
    detail_records = []

    for run_idx, model_path in enumerate(model_paths):
        run_start = time.time()
        if not os.path.isfile(model_path):
            print(f"[Skip] Missing checkpoint: {model_path}")
            continue

        ckpt = torch.load(model_path, map_location=args.device)
        ckpt_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}

        eval_args = merge_eval_args(args, ckpt_args, num_nodes=num_nodes, in_dim=args.in_dim)
        encoder, pred_head = build_models(eval_args)

        try:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        except RuntimeError:
            encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)

        try:
            if eval_args.pred_head_type == 'v6':
                pred_head.load_state_dict(ckpt['pred_head_state_dict'], strict=False)
            else:
                pred_head.load_state_dict(ckpt['pred_head_state_dict'])
        except RuntimeError:
            pred_head.load_state_dict(ckpt['pred_head_state_dict'], strict=False)

        run_seed = args.seed + run_idx
        run_records = []

        print(f"\n[Run {run_idx + 1}/{len(model_paths)}] {model_path}")
        for split_idx in range(args.random_node_idx_split_runs):
            split_seed = run_seed * args.seed_stride + split_idx
            rng = np.random.default_rng(split_seed)
            idx_subset = select_subset(num_nodes, rng, args)

            mae_split, rmse_split = eval_one_split(
                encoder=encoder,
                pred_head=pred_head,
                test_loader=test_loader,
                eval_args=eval_args,
                idx_subset_np=idx_subset,
                scaler=scaler,
            )

            all_mae.append(mae_split)
            all_rmse.append(rmse_split)

            run_records.append({
                'split_idx': split_idx,
                'split_seed': int(split_seed),
                'subset_size': int(len(idx_subset)),
                'mae': mae_split,
                'rmse': rmse_split,
            })

            if split_idx % max(1, args.print_every_split) == 0:
                print(f"  split {split_idx:3d}/{args.random_node_idx_split_runs:3d} | subset={len(idx_subset):3d}")

        detail_records.append({
            'run_idx': run_idx,
            'run_seed': int(run_seed),
            'model_path': model_path,
            'elapsed_sec': float(time.time() - run_start),
            'splits': run_records,
        })

    if len(all_mae) == 0:
        raise RuntimeError('没有可用评估结果，请检查模型路径与数据配置。')

    stats = aggregate_metrics(all_mae, all_rmse)
    elapsed = time.time() - start_time

    print('\n\nResults for multiple runs\n')
    for i in range(len(stats['amae'])):
        print(
            'runs {:d} ; MAE = {:.4f} +- {:.4f} ; RMSE = {:.4f} +- {:.4f}'.format(
                i + 1,
                float(stats['amae'][i]),
                float(stats['smae'][i]),
                float(stats['armse'][i]),
                float(stats['srmse'][i]),
            )
        )

    print(
        '\n Final: MAE = {:.4f} +- {:.4f} ; RMSE = {:.4f} +- {:.4f}'.format(
            stats['all_runs_avermae'],
            stats['all_runs_aver_stdmae'],
            stats['all_runs_avermse'],
            stats['all_runs_aver_stdrmse'],
        )
    )
    print(f"Mode elapsed seconds: {elapsed:.2f}")

    out_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        'mode': args.mode,
        'data': args.data,
        'num_models': len(model_paths),
        'random_node_idx_split_runs': int(args.random_node_idx_split_runs),
        'total_eval_groups': int(len(all_mae)),
        'stats': to_serializable_stats(stats),
        'elapsed_sec': float(elapsed),
    }

    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.save_detail:
        detail_path = os.path.join(out_dir, 'detail.json')
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(detail_records, f, ensure_ascii=False)

    print(f"Saved summary: {summary_path}")
    if args.save_detail:
        print(f"Saved detail: {os.path.join(out_dir, 'detail.json')}")


if __name__ == '__main__':
    main()
