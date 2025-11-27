import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union
import numpy as np


from .prototype_spatial import PrototypeRoutingModule

class DataPreprocessor:
    """
    数据预处理工具 - 在DataLoader中调用
    """
    @staticmethod
    def compute_delta_vectorized(mask: torch.Tensor) -> torch.Tensor:
        """
        向量化Delta计算 - 无循环，高效
        mask: (B, 1, N, T) 或 (B, N, T)
        返回: (B, 1, N, T) 归一化时间间隔
        """
        if mask.dim() == 4:
            mask = mask.squeeze(1)  # (B, N, T)

        B, N, T = mask.shape
        device = mask.device

        # 创建时间索引
        time_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, N, T)

        # 找到每个时间点的最近观测时间
        observed_times = time_idx * (mask > 0.5)

        # 累积最大值得到最近观测时间
        cum_max = torch.cummax(observed_times, dim=-1).values

        # 计算时间间隔
        delta = time_idx - cum_max

        # 处理初始未观测的情况（累计观测为0的位置）
        # 即使 cum_max[t=0]=0 且 time_idx[t=0]=0，如果 mask[t=0]=0，应设为最大间隔
        cum_obs = (mask > 0.5).cumsum(dim=-1)  # 累计观测数
        first_unobserved = (cum_obs == 0)  # 在第一次观测之前的所有位置
        delta[first_unobserved] = T  # 设为最大间隔

        # 归一化到[0,1]
        delta_normalized = delta / T

        return delta_normalized.unsqueeze(1)  # (B, 1, N, T)
def pad_to_global(
    x_subset: torch.Tensor,
    idx: Union[torch.Tensor, np.ndarray, List],
    num_nodes: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, F, N_subset, T = x_subset.shape

    if isinstance(idx, np.ndarray):
        idx = torch.from_numpy(idx).to(device)
    elif isinstance(idx, list):
        idx = torch.tensor(idx, device=device)
    elif isinstance(idx, torch.Tensor):
        idx = idx.to(device)
    else:
        raise TypeError(f"idx must be tensor, array or list, got {type(idx)}")

    x_full = torch.zeros(B, F, num_nodes, T, device=device, dtype=x_subset.dtype)
    x_full[:, :, idx, :] = x_subset

    mask = torch.zeros(B, 1, num_nodes, T, device=device, dtype=x_subset.dtype)
    mask[:, :, idx, :] = 1.0

    return x_full, mask


def compute_confidence(mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    decay_rate = 5.0
    confidence = torch.sigmoid(-decay_rate * delta) * mask
    return confidence


class CausalDilatedConvBlock(nn.Module):

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            channels, channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )

        self.delta_net = nn.Sequential(
            nn.Conv1d(1, max(8, channels // 8), 1),
            nn.Tanh(),
            nn.Conv1d(max(8, channels // 8), channels, 1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv1d(channels, channels, 1)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        B, N, D, T = x.shape
        residual = x

        x_flat = x.reshape(B * N, D, T)
        delta_flat = delta.reshape(B * N, 1, T)

        h = self.conv(x_flat)
        if self.padding > 0:
            h = h[..., :-self.padding]

        gate, feature = torch.chunk(h, 2, dim=1)
        gate = torch.sigmoid(gate)
        feature = torch.tanh(feature)
        h = feature * gate

        gamma = self.delta_net(delta_flat)
        h = h * gamma

        h = self.out_proj(h)

        output = residual.reshape(B * N, D, T) + h
        output = output.reshape(B, N, D, T)

        return output


class TemporalBackbone(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dilations: List[int] = [1, 2, 4, 8, 16, 32],
        kernel_size: int = 3
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dilations = dilations

        self.input_proj = nn.Conv1d(in_dim + 1, hidden_dim, 1)

        self.temporal_blocks = nn.ModuleList([
            CausalDilatedConvBlock(hidden_dim, dilation=d, kernel_size=kernel_size)
            for d in dilations
        ])

        self.norm = nn.InstanceNorm1d(hidden_dim, affine=True, track_running_stats=False)

        self.gradient_scale = 1.0 / math.sqrt(len(dilations))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        B, F, N, T = x.shape

        x = x.permute(0, 2, 1, 3)
        mask = mask.permute(0, 2, 1, 3)
        delta = delta.permute(0, 2, 1, 3)

        x_flat = x.reshape(B * N, F, T)
        mask_flat = mask.reshape(B * N, 1, T)

        x_with_mask = torch.cat([x_flat, mask_flat], dim=1)
        h = self.input_proj(x_with_mask)

        h = h.reshape(B, N, self.hidden_dim, T)

        for block in self.temporal_blocks:
            h = block(h, delta)

        h_flat = h.reshape(B * N, self.hidden_dim, T)
        h_flat = self.norm(h_flat)
        h = h_flat.reshape(B, N, self.hidden_dim, T)

        if self.training:
            h = h * self.gradient_scale

        h_time = h.permute(0, 2, 1, 3)

        return h_time


class TimeFirstEncoder(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        num_prototypes: int = 32,
        temporal_dilations: List[int] = [1, 2, 4, 8, 16, 32],
        kernel_size: int = 3,
        use_projection: bool = False,
        summary_pool: str = 'mean',
        pretrain_temporal_epochs: int = 0,
        use_spatial_module_in_pretrain: bool = False,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.device = device

        self.pretrain_temporal_epochs = pretrain_temporal_epochs
        self.use_spatial_module_in_pretrain = use_spatial_module_in_pretrain
        self._current_epoch = 0

        self.temporal_backbone = TemporalBackbone(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dilations=temporal_dilations,
            kernel_size=kernel_size
        )

        self.prototype_routing = PrototypeRoutingModule(
            D=hidden_dim,
            K=num_prototypes,
            summary_pool=summary_pool,
            use_projection=use_projection
        )

        self.lambda_sp = nn.Parameter(torch.zeros(1))
        self.lambda_init = nn.Parameter(torch.zeros(1))

        self.unobserved_init = nn.Parameter(torch.randn(1, hidden_dim, 1, 1) * 0.01)

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def should_use_spatial_module(self) -> bool:
        if self._current_epoch < self.pretrain_temporal_epochs:
            return self.use_spatial_module_in_pretrain
        return True

    def forward(
        self,
        x_subset: torch.Tensor,
        idx: Union[torch.Tensor, np.ndarray, List]
    ) -> torch.Tensor:
        B, F, N_subset, T = x_subset.shape
        device = x_subset.device

        x_full, mask = pad_to_global(x_subset, idx, self.num_nodes, device)
        delta = DataPreprocessor.compute_delta_vectorized(mask)
        h_time = self.temporal_backbone(x_full, mask, delta)

        if self.should_use_spatial_module():
            confidence = compute_confidence(mask, delta)
            h_spatial = self.prototype_routing(h_time, mask, idx, confidence)
        else:
            h_spatial = torch.zeros_like(h_time)

        mask_bool = (mask > 0.5).float()

        w_sp = torch.sigmoid(self.lambda_sp)
        w_init = torch.sigmoid(self.lambda_init)

        w_sum = w_sp + w_init
        w_sp = w_sp / (w_sum + 1e-8)
        w_init = w_init / (w_sum + 1e-8)

        init = self.unobserved_init.expand(B, self.hidden_dim, self.num_nodes, T)

        h = mask_bool * h_time + (1 - mask_bool) * (w_sp * h_spatial + w_init * init)

        return h

    def get_prototypes(self) -> torch.Tensor:
        return self.prototype_routing.get_prototypes()
    