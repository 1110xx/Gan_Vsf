import torch
import torch.nn as nn
import torch.nn.functional as F
class TemporalPredHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        seq_in_len: int,
        seq_out_len: int,
        kernel_size: int = 3,
        dropout: float = 0.1
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_in_len = seq_in_len
        self.seq_out_len = seq_out_len

        # 时序处理
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 时间映射
        self.time_proj = nn.Linear(seq_in_len, seq_out_len)

        # 特征映射
        self.out_proj = nn.Linear(hidden_dim, 1)

        self._init_weights()

        param_count = sum(p.numel() for p in self.parameters())
        print(f"✓ Created TemporalPredHead with {param_count:,} parameters")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, D, N, T_in) - Encoder 输出
        return: (B, 1, N, T_out) - 全集预测
        """
        B, D, N, T = h.shape

        # (B, D, N, T) → (B*N, D, T)
        h = h.permute(0, 2, 1, 3).reshape(B * N, D, T)

        # 时序处理 + 残差
        h = h + self.temporal(h)

        # 时间映射: (B*N, D, T_in) → (B*N, D, T_out)
        h = self.time_proj(h)

        # (B*N, D, T_out) → (B*N, T_out, D)
        h = h.permute(0, 2, 1)

        # 特征映射: (B*N, T_out, D) → (B*N, T_out, 1)
        pred = self.out_proj(h)

        # Reshape: (B*N, T_out, 1) → (B, N, T_out) → (B, 1, N, T_out)
        pred = pred.squeeze(-1).reshape(B, N, self.seq_out_len)
        return pred.unsqueeze(1)


def compute_subset_pred_loss(
    pred_all: torch.Tensor,
    y_subset: torch.Tensor,
    idx_subset: torch.Tensor,
    loss_fn: str = 'mae'
) -> torch.Tensor:
    """
    计算子集预测损失

    Args:
        pred_all: (B, 1, N_all, T_out) - 全集预测
        y_subset: (B, 1, N_subset, T_out) 或 (B, N_subset, T_out) - 子集真值
        idx_subset: (N_subset,) - 子集索引
        loss_fn: 'mae' 或 'mse'

    Returns:
        loss: 标量损失
    """
    # 提取子集预测
    pred_subset = pred_all[:, :, idx_subset, :]

    # 确保形状匹配
    if y_subset.dim() == 3:
        y_subset = y_subset.unsqueeze(1)

    if loss_fn == 'mae':
        loss = F.l1_loss(pred_subset, y_subset)
    elif loss_fn == 'mse':
        loss = F.mse_loss(pred_subset, y_subset)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return loss