"""
时序因果预训练模块

核心思想（类比 LLM）：
- LLM 预训练：给定前 t 个 token，预测第 t+1 个 token
- 时序预训练：给定前 t 个时间步，预测第 t+1 个时间步

这样 Encoder 学到的是"时间因果关系"，而非"空间补全"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict


class TemporalMaskedPretrainLoss(nn.Module):
    """
    时序掩码预训练损失

    策略：随机掩盖部分时间步，用前面的时间步预测被掩盖的部分
    这让 Encoder 学习时序因果关系
    """
    def __init__(self, hidden_dim: int, mask_ratio: float = 0.25):
        super().__init__()
        self.mask_ratio = mask_ratio

        # 预测头：用历史表示预测未来
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, h_all: torch.Tensor, x_full: torch.Tensor = None):
        """
        h_all: (B, D, N, T) - Encoder 输出
        x_full: (B, F, N, T) - 原始输入（可选，用于重构损失）

        返回：时序预测损失
        """
        B, D, N, T = h_all.shape
        device = h_all.device

        # 随机选择要掩盖的时间步（只选后半部分，因为需要有历史）
        num_mask = max(1, int(T * self.mask_ratio))
        mask_start = T // 2  # 只在后半部分掩盖

        # 对每个 batch 随机选择掩盖的时间步
        mask_indices = torch.stack([
            torch.randperm(T - mask_start, device=device)[:num_mask] + mask_start
            for _ in range(B)
        ])  # (B, num_mask)

        total_loss = 0.0

        for b in range(B):
            for mask_t in mask_indices[b]:
                mask_t = mask_t.item()
                if mask_t == 0:
                    continue

                # 用前一个时间步的表示预测当前时间步
                h_prev = h_all[b, :, :, mask_t - 1]  # (D, N)
                h_curr = h_all[b, :, :, mask_t]      # (D, N)

                # 预测
                h_prev_t = h_prev.permute(1, 0)  # (N, D)
                h_pred = self.predictor(h_prev_t)  # (N, D)

                h_curr_t = h_curr.permute(1, 0)  # (N, D)

                # 损失：预测表示与真实表示的相似度
                loss = F.mse_loss(h_pred, h_curr_t.detach())
                total_loss = total_loss + loss

        avg_loss = total_loss / (B * num_mask)
        return avg_loss


class NextStepPredictionLoss(nn.Module):
    """
    下一步预测损失（类似 GPT）

    对每个时间步 t，用 h[t] 预测 x[t+1]
    这是最直接的时序因果学习
    """
    def __init__(self, hidden_dim: int, out_dim: int = 1):
        super().__init__()

        # 预测下一个时间步的值
        self.next_step_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h_all: torch.Tensor, x_full: torch.Tensor):
        """
        h_all: (B, D, N, T) - Encoder 输出
        x_full: (B, F, N, T) - 原始输入

        返回：下一步预测损失
        """
        B, D, N, T = h_all.shape

        # 用 t 时刻的表示预测 t+1 时刻的值
        h_prev = h_all[:, :, :, :-1]  # (B, D, N, T-1)
        x_next = x_full[:, 0:1, :, 1:]  # (B, 1, N, T-1) 只取第一个特征

        # (B, D, N, T-1) -> (B, N, T-1, D)
        h_prev = h_prev.permute(0, 2, 3, 1)

        # 预测
        pred = self.next_step_pred(h_prev)  # (B, N, T-1, 1)
        pred = pred.permute(0, 3, 1, 2)  # (B, 1, N, T-1)

        # 损失
        loss = F.mse_loss(pred, x_next)

        return loss


class CombinedPretrainLoss(nn.Module):
    """
    组合预训练损失

    = 空间重构（原 GAN 目标） + 时序预测（新增）

    这样 Encoder 同时学习：
    1. 节点间的空间关系（通过空间重构）
    2. 时间步间的因果关系（通过时序预测）
    """
    def __init__(self, hidden_dim: int, out_dim: int = 1,
                 lambda_spatial: float = 1.0,
                 lambda_temporal: float = 1.0):
        super().__init__()

        self.lambda_spatial = lambda_spatial
        self.lambda_temporal = lambda_temporal

        # 时序预测损失
        self.temporal_loss = NextStepPredictionLoss(hidden_dim, out_dim)

    def forward(self, h_all: torch.Tensor, x_full: torch.Tensor,
                recon_loss: torch.Tensor = None):
        """
        h_all: Encoder 输出
        x_full: 原始输入
        recon_loss: 空间重构损失（来自原 GAN）
        """
        # 时序预测损失
        temporal_loss = self.temporal_loss(h_all, x_full)

        # 组合损失
        if recon_loss is not None:
            total_loss = self.lambda_spatial * recon_loss + self.lambda_temporal * temporal_loss
        else:
            total_loss = self.lambda_temporal * temporal_loss

        return total_loss, {
            'temporal_loss': temporal_loss.item(),
            'spatial_loss': recon_loss.item() if recon_loss is not None else 0.0
        }
# ============================================================
# Latent Dynamics（时间动力学一致性约束）
# ============================================================

class LatentDynamics(nn.Module):
    """
    Latent 空间的时间动力学预测器

    核心思想：
    - 在 latent 空间建模时间演化：h(t+1) ≈ f(h(t))
    - 不跨节点（每个节点独立演化）
    - 只建模时间维度
    - capacity 必须小（否则 encoder 可以甩锅给 dynamics head）

    设计原则：
    - 使用因果膨胀卷积，只看历史
    - 参数量小，强迫 encoder 承担主要责任
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 很小的网络：只用 1-2 层因果卷积
        # 不跨节点，只在时间维度上做
        padding = (kernel_size - 1) * dilation  # 因果填充（左填充）

        self.f = nn.Sequential(
            # (B, D, N, T) 视为 (B*N, D, T)
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                      padding=padding, dilation=dilation, groups=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )

        self._init_weights()

        param_count = sum(p.numel() for p in self.parameters())
        print(f"✓ Created LatentDynamics with {param_count:,} parameters")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        预测 h(t+1) from h(0:t)

        Args:
            h: (B, D, N, T) - encoder 输出的 latent 表示

        Returns:
            h_pred: (B, D, N, T-1) - 预测的 h(1:T)
        """
        B, D, N, T = h.shape

        # (B, D, N, T) -> (B*N, D, T)
        h_flat = h.permute(0, 2, 1, 3).reshape(B * N, D, T)

        # 因果卷积预测
        h_pred = self.f(h_flat)  # (B*N, D, T + padding)

        # 只取前 T-1 步作为对 h(1:T) 的预测
        # 因为 h_pred[t] 是用 h(0:t) 预测的，对应目标是 h(t+1)
        h_pred = h_pred[:, :, :T-1]  # (B*N, D, T-1)

        # reshape back
        h_pred = h_pred.reshape(B, N, D, T - 1).permute(0, 2, 1, 3)  # (B, D, N, T-1)

        return h_pred


class LatentDynamicsV2(nn.Module):
    """
    Latent Dynamics V2 - 更轻量的版本

    使用简单的 MLP 而非卷积，直接预测 h(t+1) from h(t)
    这是最简单的一阶马尔可夫假设
    """

    def __init__(self, hidden_dim: int, bottleneck_ratio: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        bottleneck_dim = int(hidden_dim * bottleneck_ratio)

        # 简单的 MLP：h(t) -> h(t+1)
        self.f = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim)
        )

        self._init_weights()

        param_count = sum(p.numel() for p in self.parameters())
        print(f"✓ Created LatentDynamicsV2 with {param_count:,} parameters")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, D, N, T)

        Returns:
            h_pred: (B, D, N, T-1) - f(h(0:T-1))，预测目标是 h(1:T)
        """
        B, D, N, T = h.shape

        # 取 h(0:T-1)
        h_prev = h[:, :, :, :-1]  # (B, D, N, T-1)

        # (B, D, N, T-1) -> (B, N, T-1, D)
        h_prev = h_prev.permute(0, 2, 3, 1)

        # 预测
        h_pred = self.f(h_prev)  # (B, N, T-1, D)

        # (B, N, T-1, D) -> (B, D, N, T-1)
        h_pred = h_pred.permute(0, 3, 1, 2)

        return h_pred


def compute_latent_dynamics_loss(
    h_all: torch.Tensor,
    latent_dyn: nn.Module,
    detach_target: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 Latent Dynamics 一致性损失

    核心思想：
    - h(t+1) ≈ f(h(t))
    - 用 stop-grad 让 encoder 对时间一致性负责，而非 dynamics head

    Args:
        h_all: (B, D, N, T) - encoder 输出
        latent_dyn: LatentDynamics 模块
        detach_target: 是否对目标 stop-grad（推荐 True）

    Returns:
        loss: 标量损失
        metrics: 指标字典
    """
    B, D, N, T = h_all.shape

    # 预测 h(1:T)
    h_pred = latent_dyn(h_all)  # (B, D, N, T-1)

    # 目标是 h(1:T)
    h_target = h_all[:, :, :, 1:]  # (B, D, N, T-1)

    if detach_target:
        # :warning: 关键：stop-grad on prediction side
        # 让 encoder 学习产生"可被平滑推进的 latent"
        # 而不是让 dynamics head 学 identity
        loss = F.mse_loss(h_target, h_pred.detach())
    else:
        # 不推荐：可能导致 encoder + f 一起学 identity
        loss = F.mse_loss(h_target, h_pred)

    # 计算 consistency 指标（不参与梯度）
    with torch.no_grad():
        # latent 变化量
        h_diff = h_all[:, :, :, 1:] - h_all[:, :, :, :-1]
        latent_change = h_diff.abs().mean().item()

        # 预测误差
        pred_error = (h_target - h_pred).abs().mean().item()

    metrics = {
        'latent_dyn_loss': loss.item(),
        'latent_change': latent_change,
        'latent_pred_error': pred_error,
    }

    return loss, metrics


class LatentDynamicsLoss(nn.Module):
    """
    封装的 Latent Dynamics Loss 模块

    可以直接在训练脚本中使用
    """

    def __init__(
        self,
        hidden_dim: int,
        version: str = 'v2',
        detach_target: bool = True
    ):
        super().__init__()
        self.detach_target = detach_target

        if version == 'v1':
            self.latent_dyn = LatentDynamics(hidden_dim)
        elif version == 'v2':
            self.latent_dyn = LatentDynamicsV2(hidden_dim)
        else:
            raise ValueError(f"Unknown version: {version}")

    def forward(self, h_all: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            h_all: (B, D, N, T) - encoder 输出

        Returns:
            loss: 标量损失
            metrics: 指标字典
        """
        return compute_latent_dynamics_loss(
            h_all, self.latent_dyn, self.detach_target
        )