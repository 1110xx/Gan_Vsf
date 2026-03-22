import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from util import load_dataset, masked_mae
from model.encoder import NodeAwareTemporalEncoder
from model.pred_decoder_v6 import create_pred_head_v6
from model.pred_decoder_v8 import create_pred_head_v8, compute_v8_loss


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def compute_pred_loss(pred_all, y_subset, idx_subset, loss_fn='mse'):
    """计算预测损失（仅子集）"""
    pred_subset = pred_all[:, 0, idx_subset, :]
    if loss_fn == 'mse':
        return F.mse_loss(pred_subset, y_subset)
    elif loss_fn == 'mae':
        return F.l1_loss(pred_subset, y_subset)
    else:
        return F.smooth_l1_loss(pred_subset, y_subset)


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


def create_pred_head(args, device, shared_node_embed=None):
    """创建预测头（仅支持 V6/V8）"""
    if args.pred_head_type == 'v6':
        return create_pred_head_v6(
            head_type='hybrid',
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            use_cross_attn=getattr(args, 'use_cross_attn', True),
            use_residual_pred=getattr(args, 'use_residual_pred', True),
            shared_node_embed=shared_node_embed
        ).to(device)
    elif args.pred_head_type == 'v8':
        return create_pred_head_v8(
            hidden_dim=args.hidden_dim,
            num_nodes=args.num_nodes,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_layers=getattr(args, 'pred_n_layers', 4),
            kernel_size=getattr(args, 'tcn_kernel_size', 3),
            dropout=0.1,
            use_residual_pred=getattr(args, 'use_residual_pred', False),
        ).to(device)
    else:
        raise ValueError(f"Unknown pred_head_type: {args.pred_head_type}. Only 'v6' and 'v8' are supported.")


def train_epoch(encoder, pred_head, dataloader, optimizer, args, epoch: int, scaler=None):
    """训练一个 epoch"""
    encoder.train()
    pred_head.train()
    dataloader.shuffle()

    pred_losses = []
    pred_subset_losses = []
    total_losses = []
    start_time = time.time()
    subset_slices, perm_refreshed = get_epoch_subset_slices(args, epoch, stream="train")
    num_split = len(subset_slices)

    for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
        # x: (B, T, N, F) -> (B, F, N, T)
        x_full = torch.Tensor(x).to(args.device)
        x_full = x_full.transpose(1, 3)

        # y: (B, T, N, F) 或 (B, T, N)
        y_full = torch.Tensor(y).to(args.device)
        if y_full.dim() == 4:
            y_full = y_full[..., 0]

        # 归一化 y
        if scaler is not None:
            y_full = (y_full - scaler['mean']) / scaler['std']
        y_full = y_full.transpose(1, 2)  # (B, N, T)

        # epoch 内按切片循环。保证覆盖全集
        idx_subset = torch.tensor(subset_slices[iter_idx % num_split], device=args.device)

        x_subset = x_full[:, :, idx_subset, :]
        y_subset = y_full[:, idx_subset, :]

        optimizer.zero_grad()

        # 前向传播
        if getattr(args, 'use_clean_obs', False) and args.pretrain_ckpt is not None:
            h_all, h_obs_clean = encoder(x_subset, idx_subset, return_obs_clean=True)
            h_all = encoder.replace_obs_with_clean(h_all, h_obs_clean, idx_subset)
        else:
            h_all = encoder(x_subset, idx_subset)

        x_last_full = x_full[:, 0:1, :, -1]

        # V6/V8 预测
        if args.pred_head_type == 'v8':
            pred_all, h_future = pred_head(h_all, x_last=x_last_full, return_latent=True)
            loss_pred, loss_dict = compute_v8_loss(
                pred_all, y_subset, idx_subset,
                h_future=h_future,
                lambda_smooth=getattr(args, 'lambda_smooth', 0.1),
                loss_type=args.loss_fn
            )
            iter_pred_subset_loss = float(loss_dict['pred_loss'])
            iter_total_loss = float(loss_dict['total_loss'])
        else:  # v6
            pred_all = pred_head(h_all, x_last=x_last_full)
            loss_pred = compute_pred_loss(pred_all, y_subset, idx_subset, args.loss_fn)
            loss_dict = None
            iter_pred_subset_loss = loss_pred.item()
            iter_total_loss = loss_pred.item()

        # 反向传播
        loss_pred.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(pred_head.parameters()),
            max_norm=args.max_grad_norm
        )
        optimizer.step()

        pred_subset_losses.append(iter_pred_subset_loss)
        total_losses.append(iter_total_loss)

        if iter_idx % args.print_every == 0:
            if loss_dict and args.pred_head_type == 'v8':
                print(f"  Iter [{iter_idx:3d}/{dataloader.num_batch:3d}] "
                      f"Total: {loss_dict['total_loss']:.4f} "
                      f"Pred: {loss_dict['pred_loss']:.4f} "
                      f"Smooth: {loss_dict['smooth_loss']:.4f}")
            else:
                print(f"  Iter [{iter_idx:3d}/{dataloader.num_batch:3d}] "
                      f"Loss: {loss_pred.item():.6f}")

    return {
        'loss_pred_subset': np.mean(pred_subset_losses),
        'loss_total': np.mean(total_losses),
        'num_split': num_split,
        'subset_cover_nodes': int(sum(len(s) for s in subset_slices)),
        'perm_refreshed': int(perm_refreshed),
        'epoch_time': time.time() - start_time,
    }


def validate(encoder, pred_head, dataloader, args, epoch: int, scaler=None):
    """验证"""
    encoder.eval()
    pred_head.eval()

    val_mae_list = []
    val_mae_norm_list = []
    horizon_mae_list = {h: [] for h in range(args.seq_out_len)}
    subset_slices, _ = get_epoch_subset_slices(args, epoch, stream="val")
    num_split = len(subset_slices)  

    with torch.no_grad():
        for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
            x_full = torch.Tensor(x).to(args.device)
            x_full = x_full.transpose(1, 3)

            y_real = torch.Tensor(y).to(args.device)
            if y_real.dim() == 4:
                y_real = y_real[..., 0]

            idx_subset = torch.tensor(subset_slices[iter_idx % num_split],device=args.device)
            x_subset = x_full[:, :, idx_subset, :]

            if getattr(args, 'use_clean_obs', False) and args.pretrain_ckpt is not None:
                h_all, h_obs_clean = encoder(x_subset, idx_subset, return_obs_clean=True)
                h_all = encoder.replace_obs_with_clean(h_all, h_obs_clean, idx_subset)
            else:
                h_all = encoder(x_subset, idx_subset)

            x_last_full = x_full[:, 0:1, :, -1]

            # V6/V8 预测
            if args.pred_head_type == 'v8':
                pred_all = pred_head(h_all, x_last=x_last_full, return_latent=False)
            else:
                pred_all = pred_head(h_all, x_last=x_last_full)

            pred_subset = pred_all[:, 0, idx_subset, :]

            # 逆变换
            if scaler is not None:
                pred_real = pred_subset * scaler['std'] + scaler['mean']
            else:
                pred_real = pred_subset

            y_subset_real = y_real[:, :, idx_subset].transpose(1, 2)

            # MAE
            mae_val, _ = masked_mae(pred_real, y_subset_real, null_val=0.0)
            val_mae_list.append(mae_val.item())

            # 标准化空间下的 MAE（pred_norm vs y_norm）
            if scaler is not None:
                y_subset_norm = (y_subset_real - scaler['mean']) / scaler['std']
                pred_subset_norm = pred_subset  # pred_subset 本来就是标准化空间输出
            else:
                y_subset_norm = y_subset_real
                pred_subset_norm = pred_real

            mae_norm_val = F.l1_loss(pred_subset_norm, y_subset_norm).item()
            val_mae_norm_list.append(mae_norm_val)

            for h in range(args.seq_out_len):
                mae_h, _ = masked_mae(pred_real[:, :, h], y_subset_real[:, :, h], null_val=0.0)
                horizon_mae_list[h].append(mae_h.item())

    return {
        'val_mae': np.mean(val_mae_list),
        'val_mae_norm': np.mean(val_mae_norm_list),
        'horizon_mae': {h: np.mean(horizon_mae_list[h]) for h in range(args.seq_out_len)},
    }


def train_loop(encoder, pred_head, train_loader, val_loader, args, scaler=None):
    """训练循环"""
    os.makedirs(args.save_dir, exist_ok=True)

    # 调度与早停设置（命令行参数）
    lr_patience = args.lr_scheduler_patience
    lr_factor = args.lr_scheduler_factor
    early_stop_patience = args.early_stop_patience

    # Plan1: 关闭 smoothness loss
    if args.plan == '1':
        args.lambda_smooth = 0.0
        print("[Plan1] Smoothness loss 已关闭 (lambda_smooth=0)")

    # Plan1: 前 10 epoch freeze encoder，需要分开优化器
    if args.plan == '1':
        # 初始冻结 encoder
        for param in encoder.parameters():
            param.requires_grad = False
        print("[Plan1] Encoder 冻结前 10 epoch")
        params = list(pred_head.parameters())  # 只优化 pred_head
    else:
        params = list(encoder.parameters()) + list(pred_head.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_factor,
        patience=lr_patience,
    )

    history = {
        'train_loss': [],
        'val_mae': [],
        'val_mae_norm': [],
        'best_val_mae_norm': float('inf')
    }
    best_epoch = 0
    no_improve_epochs = 0

    print("\n" + "=" * 70)
    print(" " * 20 + "下游预测任务训练")
    print("=" * 70)
    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    print(f"Pred head: {args.pred_head_type}")
    print(f"Num nodes: {args.num_nodes}, Subset ratio: {args.subset_ratio}")
    print(f"Subset perm refresh epochs: {args.perm_refresh_epochs}")
    print(f"Pretrain: {args.pretrain_ckpt or 'None'}")
    print(f"Plan: {args.plan or 'None'}")
    print(f"Lambda smooth: {args.lambda_smooth}")
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Pred head params: {sum(p.numel() for p in pred_head.parameters()):,}")
    print(f"LR scheduler: ReduceLROnPlateau(patience={lr_patience}, factor={lr_factor})")
    print(f"Early stopping patience: {early_stop_patience}")
    print("=" * 70 + "\n")

    for epoch in range(1, args.num_epochs + 1):
        # Plan1: 第 11 epoch unfreeze encoder
        if args.plan == '1' and epoch == 11:
            print("\n" + "=" * 70)
            print("[Plan1] Epoch 11: Unfreeze encoder，开始全模型训练")
            print("=" * 70)
            for param in encoder.parameters():
                param.requires_grad = True
            # 重建优化器，加入 encoder 参数
            params = list(encoder.parameters()) + list(pred_head.parameters())
            optimizer = torch.optim.AdamW(params, lr=args.lr * 0.5, weight_decay=args.weight_decay)
            # 重建 scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=lr_factor,
                patience=lr_patience,
            )

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 70)

        train_metrics = train_epoch(encoder, pred_head, train_loader, optimizer, args, epoch=epoch, scaler=scaler)
        val_metrics = validate(encoder, pred_head, val_loader, args, epoch=epoch, scaler=scaler)

        history['train_loss'].append(train_metrics['loss_pred_subset'])
        history['val_mae'].append(val_metrics['val_mae'])
        history['val_mae_norm'].append(val_metrics['val_mae_norm'])
        scheduler.step(val_metrics['val_mae_norm'])

        # 打印结果
        horizons = [0, 2, 5, 11]
        horizon_str = ", ".join([f"H{h+1}:{val_metrics['horizon_mae'][h]:.3f}"
                                  for h in horizons if h < args.seq_out_len])
        print(f"\n[Summary] Loss: {train_metrics['loss_pred_subset']:.6f}, "
              f"TotalLoss: {train_metrics['loss_total']:.6f}, "
              f"Val MAE (norm): {val_metrics['val_mae_norm']:.4f}, "
              f"Val MAE: {val_metrics['val_mae']:.4f}, [{horizon_str}], "
              f"Time: {train_metrics['epoch_time']:.1f}s")
        print(f"  Subset coverage: num_split={int(train_metrics.get('num_split', 0))}, "
                f"cover_nodes={int(train_metrics.get('subset_cover_nodes', 0))}, "
                f"perm_refreshed={bool(train_metrics.get('perm_refreshed', False))}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6e}")

        # 以 val_mae_norm 作为验证指标与最优模型标准
        if val_metrics['val_mae_norm'] < history['best_val_mae_norm']:
            history['best_val_mae_norm'] = val_metrics['val_mae_norm']
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'pred_head_state_dict': pred_head.state_dict(),
                'val_mae_norm': history['best_val_mae_norm'],
                'val_mae': val_metrics['val_mae'],
                'args': vars(args),
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  → Best model saved! Val MAE (norm): {history['best_val_mae_norm']:.4f}")
        else:
            no_improve_epochs += 1
            print(f"  No improvement: {no_improve_epochs}/{early_stop_patience}")

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'pred_head_state_dict': pred_head.state_dict(),
                'history': history,
                'args': vars(args),
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt'))

        if no_improve_epochs >= early_stop_patience:
            print("\n" + "=" * 70)
            print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
            print("=" * 70)
            break

    print("\n" + "=" * 70)
    print(f"训练完成! Best Val MAE (norm): {history['best_val_mae_norm']:.4f} (epoch {best_epoch})")
    print("=" * 70)
    return history
def main():
    parser = argparse.ArgumentParser(description='下游预测任务训练（V6/V8）')

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
    parser.add_argument('--pred_head_type', type=str, default='v8', choices=['v6', 'v8'],
                        help='预测头类型: v6=Hybrid, v8=纯因果TCN')
    parser.add_argument('--pred_n_layers', type=int, default=4, help='V8 TCN 层数')
    parser.add_argument('--tcn_kernel_size', type=int, default=3, help='V8 TCN 卷积核大小')
    parser.add_argument('--use_residual_pred', type=str_to_bool, default=False,
                        help='是否使用残差预测')
    parser.add_argument('--use_cross_attn', type=str_to_bool, default=True,
                        help='V6 是否使用交叉注意力')
    parser.add_argument('--lambda_smooth', type=float, default=0.1,
                        help='V8 Latent Smoothness 正则权重')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=2.0)
    parser.add_argument('--loss_fn', type=str, default='mae', choices=['mae', 'mse', 'huber'])
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                        help='ReduceLROnPlateau: 验证指标多少轮不下降后降低学习率')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                        help='ReduceLROnPlateau: 学习率衰减因子')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='早停: 验证指标多少轮不下降后停止训练')

    # 子集配置
    parser.add_argument('--subset_ratio', type=float, default=0.3)
    parser.add_argument('--perm_refresh_epochs', type=int, default=100,
                        help='子集 perm 刷新周期（单位：epoch）')

    # 预训练
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--freeze_encoder', type=str_to_bool, default=False, help='是否冻结 Encoder')
    parser.add_argument('--use_clean_obs', type=str_to_bool, default=False,
                        help='使用干净的观测节点 embedding')

    # load_dataset 需要的参数
    parser.add_argument('--predefined_S', type=str_to_bool, default=False)
    parser.add_argument('--predefined_S_frac', type=int, default=15)

    # 训练计划
    parser.add_argument('--plan', type=str, default=None,
                        help='训练计划: 1=关闭smoothness+freeze encoder前10epoch')

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

    if hasattr(scaler_obj, 'mean'):
        scaler = {'mean': scaler_obj.mean, 'std': scaler_obj.std}
    elif isinstance(scaler_obj, dict) and 'mean' in scaler_obj:
        scaler = scaler_obj
    else:
        x_train = train_loader.xs
        scaler = {'mean': x_train[..., 0].mean(), 'std': x_train[..., 0].std()}

    print(f"✓ Scaler: mean={scaler['mean']:.4f}, std={scaler['std']:.4f}")

    args.num_nodes = train_loader.num_nodes
    if args.in_dim is None:
        args.in_dim = train_loader.xs[0].shape[-1]

    print(f"✓ Data: {args.num_nodes} nodes, {args.in_dim} dim, "
          f"{train_loader.size} train, {val_loader.size} val")

    # 创建 Encoder
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
    if args.pretrain_ckpt and os.path.isfile(args.pretrain_ckpt):
        ckpt = torch.load(args.pretrain_ckpt, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        print(f"✓ Loaded pretrained encoder from {args.pretrain_ckpt}")
        if args.freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
            print("  → Encoder frozen")
    elif args.pretrain_ckpt:
        print(f"Warning: pretrain_ckpt not found: {args.pretrain_ckpt}")

    # 创建预测头
    shared_node_embed = None
    if args.pred_head_type == 'v6':
        shared_node_embed = encoder.get_shared_node_embed()
        print(f"✓ V6: Using shared node embedding")
    else:
        print(f"✓ V8: Pure causal TCN decoder")

    pred_head = create_pred_head(args, device, shared_node_embed=shared_node_embed)

    # 训练
    history = train_loop(encoder, pred_head, train_loader, val_loader, args, scaler)
    return history


if __name__ == "__main__":
    main()