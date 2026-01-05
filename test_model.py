import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from util import load_dataset, masked_mae
from model.encoder_v3 import SlotBasedEncoder
from model.pred_decoder import TemporalPredHead
from model.pred_decoder_v2 import TemporalPredHeadV2

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def plot_results(pred, true, save_dir, batch_idx):
    """
    Plot prediction vs true value for a few samples.
    pred: (B, N, T)
    true: (B, N, T)
    """
    os.makedirs(save_dir, exist_ok=True)
    # Plot first 3 samples, first node
    for i in range(min(3, pred.shape[0])):
        plt.figure(figsize=(10, 5))
        plt.plot(true[i, 0, :].cpu().numpy(), label='Ground Truth', marker='o')
        plt.plot(pred[i, 0, :].cpu().numpy(), label='Prediction', marker='x')
        plt.legend()
        plt.title(f'Batch {batch_idx}, Sample {i}, Node 0')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.savefig(os.path.join(save_dir, f'pred_b{batch_idx}_s{i}.png'))
        plt.close()

def evaluate(encoder, pred_head, dataloader, args, scaler=None, num_repeats=5, save_viz=False):
    """
    Evaluate on test set.
    """
    encoder.eval()
    pred_head.eval()

    total_mae = []
    horizon_mae_list = {h: [] for h in range(args.seq_out_len)}
    
    num_subset = max(1, int(args.num_nodes * args.subset_ratio))
    print(f"Evaluating with subset size: {num_subset}/{args.num_nodes} (Ratio: {args.subset_ratio})")
    print(f"Repeating evaluation {num_repeats} times per batch to ensure stability.")

    with torch.no_grad():
        for iter_idx, (x, y) in enumerate(dataloader.get_iterator()):
            x_full = torch.Tensor(x).to(args.device)
            x_full = x_full.transpose(1, 3)  # (B, F, N, T)

            y_real = torch.Tensor(y).to(args.device)
            if y_real.dim() == 4:
                y_real = y_real[..., 0]
            
            # Repeat for stability
            batch_maes = []
            
            for r in range(num_repeats):
                # Random subset
                idx_subset = np.random.choice(args.num_nodes, size=num_subset, replace=False)
                idx_subset = torch.tensor(idx_subset, device=args.device)

                x_subset = x_full[:, :, idx_subset, :]

                # Forward
                h_all = encoder(x_subset, idx_subset)
                pred_all = pred_head(h_all)

                # Extract subset prediction
                pred_subset = pred_all[:, 0, idx_subset, :]

                # Inverse transform
                if scaler is not None:
                    pred_real = pred_subset * scaler['std'] + scaler['mean']
                else:
                    pred_real = pred_subset

                y_subset_real = y_real[:, :, idx_subset]
                y_subset_real = y_subset_real.transpose(1, 2)

                # 统计信息
                if iter_idx == 0 and r == 0:
                    print(f"\n[Debug Stats]")
                    print(f"Pred (norm) - Mean: {pred_subset.mean():.4f}, Std: {pred_subset.std():.4f}, Min: {pred_subset.min():.4f}, Max: {pred_subset.max():.4f}")
                    # y_subset_real 是原始尺度，我们需要归一化后的 y 来比较
                    if scaler is not None:
                        y_subset_norm = (y_subset_real - scaler['mean']) / scaler['std']
                        print(f"True (norm) - Mean: {y_subset_norm.mean():.4f}, Std: {y_subset_norm.std():.4f}, Min: {y_subset_norm.min():.4f}, Max: {y_subset_norm.max():.4f}")
                        print(f"Scaler - Mean: {scaler['mean']:.4f}, Std: {scaler['std']:.4f}")

                mae_val, _ = masked_mae(pred_real, y_subset_real, null_val=0.0)
                batch_maes.append(mae_val.item())

                # Horizon MAE (only collect for the first repeat)
                if r == 0:
                    for h in range(args.seq_out_len):
                        pred_h = pred_real[:, :, h]
                        y_h = y_subset_real[:, :, h]
                        mae_h, _ = masked_mae(pred_h, y_h, null_val=0.0)
                        horizon_mae_list[h].append(mae_h.item())
                    
                    # Visualize first batch
                    if save_viz and iter_idx == 0:
                        plot_results(pred_real, y_subset_real, 'viz_results', iter_idx)

            total_mae.append(np.mean(batch_maes))

    mean_mae = np.mean(total_mae)
    horizon_mae = {h: np.mean(horizon_mae_list[h]) for h in range(args.seq_out_len)}

    return {
        'test_mae': mean_mae,
        'horizon_mae': horizon_mae
    }

def main():
    parser = argparse.ArgumentParser(description='Test Model')
    
    # Same arguments as training
    parser.add_argument('--data', type=str, required=True, help='Data path')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--in_dim', type=int, default=None)
    parser.add_argument('--seq_in_len', type=int, default=12)
    parser.add_argument('--seq_out_len', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_slots', type=int, default=16)
    parser.add_argument('--pred_kernel_size', type=int, default=3)
    parser.add_argument('--pred_head_version', type=int, default=1)
    parser.add_argument('--pred_n_layers', type=int, default=3)
    parser.add_argument('--use_node_attn', type=str_to_bool, default=True)
    parser.add_argument('--subset_ratio', type=float, default=0.15)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--predefined_S', type=str_to_bool, default=False)
    parser.add_argument('--predefined_S_frac', type=int, default=15)
    parser.add_argument('--save_viz', type=str_to_bool, default=True, help='Save visualization')

    args = parser.parse_args()

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    # Load Data
    print(f"Loading data from {args.data}...")
    dataloader_dict = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    test_loader = dataloader_dict['test_loader']
    scaler_obj = dataloader_dict['scaler']

    if hasattr(scaler_obj, 'mean'):
        scaler = {'mean': scaler_obj.mean, 'std': scaler_obj.std}
    elif isinstance(scaler_obj, dict) and 'mean' in scaler_obj:
        scaler = scaler_obj
    else:
        x_train = dataloader_dict['train_loader'].xs
        scaler = {
            'mean': x_train[..., 0].mean(),
            'std': x_train[..., 0].std()
        }
    
    args.num_nodes = test_loader.num_nodes
    if args.in_dim is None:
        args.in_dim = test_loader.xs.shape[-1]

    # Create Models
    encoder = SlotBasedEncoder(
        num_nodes=args.num_nodes,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_slots=args.num_slots,
        device=args.device
    ).to(device)

    if args.pred_head_version == 1:
        pred_head = TemporalPredHead(
            hidden_dim=args.hidden_dim,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            kernel_size=args.pred_kernel_size,
            dropout=0.1
        ).to(device)
    else:
        pred_head = TemporalPredHeadV2(
            hidden_dim=args.hidden_dim,
            seq_in_len=args.seq_in_len,
            seq_out_len=args.seq_out_len,
            n_layers=args.pred_n_layers,
            n_heads=4,
            kernel_size=args.pred_kernel_size,
            dropout=0.1,
            use_node_attn=args.use_node_attn
        ).to(device)

    # Load Checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle potential state_dict mismatch for SlotBasedEncoder
    try:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    except RuntimeError as e:
        print(f"Warning: Strict loading failed ({e}). Retrying with strict=False...")
        encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        
    pred_head.load_state_dict(checkpoint['pred_head_state_dict'])
    
    print("Model loaded successfully.")
    if 'val_mae' in checkpoint:
        print(f"Checkpoint Validation MAE: {checkpoint['val_mae']:.4f}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint Epoch: {checkpoint['epoch']}")

    # Evaluate
    print("\nStarting evaluation on TEST set...")
    metrics = evaluate(encoder, pred_head, test_loader, args, scaler, num_repeats=5, save_viz=args.save_viz)

    print("\n" + "="*40)
    print("Test Results")
    print("="*40)
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    
    horizon_str = ", ".join([f"H{h+1}:{metrics['horizon_mae'][h]:.3f}"
                              for h in [0, 2, 5, 11] if h < args.seq_out_len])
    print(f"Horizon MAE: [{horizon_str}]")
    print("="*40)
    if args.save_viz:
        print("Visualization saved to viz_results/")

if __name__ == "__main__":
    main()
