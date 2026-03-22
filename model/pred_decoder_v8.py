"""
预测头 V8 - Latent TCN Decoder（基于 Encoder Latent 的时序外推预测）

============================================
核心设计理念（基于 v8.md 方案 A）：
============================================

1. Latent Space 外推假设
   - Encoder 输出的 h_all 被视为每个节点在 latent 空间中的离散时序状态轨迹
   - 预测目标是在 latent 空间做时序外推

2. Latent TCN 数学形式
   - 使用因果膨胀卷积（TCN）学习映射: f_θ: R^{D×T_p} → R^{D×T_f}
   - 满足因果约束，无未来信息泄露
   - 并行预测所有输出步

3. 关键设计原则
   - 不再建模空间关系（节点间依赖已由 encoder 学习）
   - 参数在节点维度共享（强化"统一动力学假设"）
   - 仅在 latent 空间做时序外推（避免 decoder 直接拟合原始观测噪声）

4. Loss 设计
   - 子集监督预测损失：仅对输入子集节点计算预测误差
   - Latent Smoothness 正则：对所有节点的 latent 预测施加时间平滑约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class CausalConv1d(nn.Module):
    """
    因果卷积层 - 确保不会看到未来信息

    实现方式：使用左填充（left padding）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T) - 保持时间维度不变
        """
        # 左填充
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    TCN 残差块

    结构:
        x → CausalConv → GELU → Dropout → CausalConv → GELU → Dropout → + x
                                                                       ↑
                                                              (residual connection)
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv1 = CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation)
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, T)
        Returns:
            (B, D, T)
        """
        residual = x

        # 第一个卷积
        out = self.conv1(x)
        # LayerNorm 需要 (B, T, D) 格式
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)

        # 第二个卷积
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)

        # 残差连接
        return out + residual


class LatentTCN(nn.Module):
    """
    Latent TCN - 在 latent 空间做时序外推

    使用多层膨胀因果卷积，感受野随层数指数增长
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        # 膨胀率按层指数增长: 1, 2, 4, 8, ...
        self.layers = nn.ModuleList([
            TCNBlock(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(n_layers)
        ])

        # 计算感受野
        receptive_field = 1
        for i in range(n_layers):
            receptive_field += 2 * (kernel_size - 1) * (2 ** i)
        print(f"  TCN receptive field: {receptive_field}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, T)
        Returns:
            (B, D, T)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class TemporalPredHeadV8(nn.Module):
    """
    V8 预测头 - Latent TCN Decoder（纯因果时序外推）

    核心思想：
        1. 不压缩 h_all，保留完整的时序信息
        2. 使用 TCN 在 latent 空间做**严格因果**的时序外推
        3. 参数在节点维度共享（强化"统一动力学假设"）
        4. 一次性并行预测未来 T_out 步

    设计原则（简化版）：
        - 删除 input_pos_embed：decoder 只看 latent trajectory 本身
        - 删除 node_bias：避免泄露节点身份
        - 删除 temporal_proj / time_proj：保持严格因果性

    输入输出流程:
        h_all (B, D, N, T_past)
            │
            │ reshape + right-pad T_future zeros
            ▼
        (B·N, D, T_past + T_future)
            │
            ▼
        Latent Temporal TCN (causal, dilated)
            │
            ▼
        (B·N, D, T_past + T_future)
            │
            │ 取最后 T_future 步
            ▼
        (B·N, D, T_future)
            │
            │ reshape
            ▼
        (B, D, N, T_future)
            │
            ▼
        Output Projection (D → 1)
            │
            ▼
        ŷ (B, 1, N, T_future)
    """
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        seq_in_len: int,
        seq_out_len: int,
        n_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_residual_pred: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.seq_in_len = seq_in_len
        self.seq_out_len = seq_out_len
        self.use_residual_pred = use_residual_pred

        # ========== 1. Latent TCN（纯因果卷积）==========
        self.latent_tcn = LatentTCN(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # ========== 2. 输出投影 ==========
        # D → 1（特征维度到输出维度）
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 保存 latent 预测用于计算 smoothness loss
        self._h_future = None

        self._init_weights()

        param_count = sum(p.numel() for p in self.parameters())
        print(f"✓ Created TemporalPredHeadV8 (Pure Causal TCN Decoder) with {param_count:,} parameters")
        print(f"  - n_layers: {n_layers}")
        print(f"  - kernel_size: {kernel_size}")
        print(f"  - use_residual_pred: {use_residual_pred}")
        print(f"  - T_in: {seq_in_len}, T_out: {seq_out_len}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h_all: torch.Tensor,
        x_last: Optional[torch.Tensor] = None,
        node_idx: Optional[torch.Tensor] = None,
        return_latent: bool = False
    ) -> torch.Tensor:
        """
        Args:
            h_all: (B, D, N, T_in) - Encoder 输出
            x_last: (B, 1, N) - 最后一个输入值（用于残差预测）
            node_idx: 可选，节点索引
            return_latent: 是否返回 latent 预测（用于计算 smoothness loss）

        Returns:
            pred: (B, 1, N, T_out) - 预测结果
            h_future: (B, D, N, T_out) - latent 预测（仅当 return_latent=True）
        """
        B, D, N, T_in = h_all.shape
        T_out = self.seq_out_len

        # ========== 1. reshape for TCN ==========
        # (B, D, N, T_in) → (B, N, D, T_in) → (B*N, D, T_in)
        h = h_all.permute(0, 2, 1, 3).reshape(B * N, D, T_in)

        # ========== 2. 右侧 padding T_out 个零（用于时序外推）==========
        # TCN 通过因果卷积，让每个输出位置只能看到左边的输入
        # padding 后: (B*N, D, T_in + T_out)
        h = F.pad(h, (0, T_out), mode='constant', value=0.0)

        # ========== 3. Latent TCN（纯因果外推）==========
        # 因果卷积保持序列长度不变
        h = self.latent_tcn(h)  # (B*N, D, T_in + T_out)

        # ========== 4. 取最后 T_out 步作为预测 ==========
        h_future_flat = h[:, :, -T_out:]  # (B*N, D, T_out)

        # ========== 5. reshape back ==========
        # (B*N, D, T_out) → (B, N, D, T_out) → (B, D, N, T_out)
        h_future = h_future_flat.reshape(B, N, D, T_out).permute(0, 2, 1, 3)

        # 保存 latent 预测用于 smoothness loss
        self._h_future = h_future

        # ========== 6. 输出投影 ==========
        # (B, D, N, T_out) → (B, N, T_out, D) → (B, N, T_out, 1) → (B, N, T_out)
        pred = h_future.permute(0, 2, 3, 1)  # (B, N, T_out, D)
        pred = self.output_proj(pred).squeeze(-1)  # (B, N, T_out)

        # ========== 7. 残差预测（可选）==========
        if self.use_residual_pred and x_last is not None:
            # x_last: (B, 1, N) → (B, N)
            x_base = x_last.squeeze(1)  # (B, N)
            # pred 是相对于 x_last 的增量
            pred = x_base.unsqueeze(-1) + pred  # (B, N, T_out)

        # 输出格式: (B, 1, N, T_out)
        pred = pred.unsqueeze(1)

        if return_latent:
            return pred, h_future
        return pred

    def get_latent_prediction(self) -> Optional[torch.Tensor]:
        """获取最近一次 forward 的 latent 预测，用于计算 smoothness loss"""
        return self._h_future


def compute_v8_loss(
    pred_all: torch.Tensor,
    y_subset: torch.Tensor,
    idx_subset: torch.Tensor,
    h_future: Optional[torch.Tensor] = None,
    lambda_smooth: float = 0.1,
    loss_type: str = 'mse'
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 V8 的损失函数

    Args:
        pred_all: (B, 1, N_all, T_out) - 预测结果
        y_subset: (B, N_subset, T_out) - 子集真值
        idx_subset: (N_subset,) - 子集索引
        h_future: (B, D, N_all, T_out) - latent 预测（用于 smoothness loss）
        lambda_smooth: smoothness loss 的权重
        loss_type: 'mse' 或 'mae'

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典
    """
    # ========== 1. 子集监督预测损失 ==========
    # 提取子集预测
    pred_subset = pred_all[:, 0, idx_subset, :]  # (B, N_subset, T_out)

    if loss_type == 'mse':
        pred_loss = F.mse_loss(pred_subset, y_subset)
    elif loss_type == 'mae':
        pred_loss = F.l1_loss(pred_subset, y_subset)
    else:
        pred_loss = F.smooth_l1_loss(pred_subset, y_subset)

    # ========== 2. Latent Smoothness 正则 ==========
    smooth_loss = torch.tensor(0.0, device=pred_all.device)

    if h_future is not None and lambda_smooth > 0:
        # 对所有节点的 latent 预测施加时间平滑约束
        # ||h(t+k+1) - h(t+k)||^2
        # h_future: (B, D, N, T_out)
        h_diff = h_future[:, :, :, 1:] - h_future[:, :, :, :-1]  # (B, D, N, T_out-1)
        smooth_loss = (h_diff ** 2).mean()

    # ========== 3. 总损失 ==========
    total_loss = pred_loss + lambda_smooth * smooth_loss

    loss_dict = {
        'pred_loss': pred_loss.item(),
        'smooth_loss': smooth_loss.item(),
        'total_loss': total_loss.item()
    }

    return total_loss, loss_dict


def create_pred_head_v8(
    hidden_dim: int,
    num_nodes: int,
    seq_in_len: int,
    seq_out_len: int,
    n_layers: int = 4,
    kernel_size: int = 3,
    dropout: float = 0.1,
    use_residual_pred: bool = False,
) -> nn.Module:
    """
    创建 V8 预测头（纯因果 TCN Decoder）

    注意：V8 不使用 shared_node_embed，以避免泄露节点身份
    """
    return TemporalPredHeadV8(
        hidden_dim=hidden_dim,
        num_nodes=num_nodes,
        seq_in_len=seq_in_len,
        seq_out_len=seq_out_len,
        n_layers=n_layers,
        kernel_size=kernel_size,
        dropout=dropout,
        use_residual_pred=use_residual_pred,
    )