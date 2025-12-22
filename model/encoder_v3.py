import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Union
import numpy as np


def create_sinusoidal_encoding(num_positions: int, dim: int) -> torch.Tensor:
    """
    创建正弦位置编码（固定，非学习）

    这是关键：与 nn.Embedding 不同，正弦编码是确定性的、不可学习的
    模型无法为特定位置"记忆"特定输出
    """
    position = torch.arange(num_positions).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

    pe = torch.zeros(num_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class TemporalEncoder(nn.Module):
    """轻量级时序编码器"""

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden_dim, 1)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, dilation=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """x: (B, F, N, T) → (B, D, N, T)"""
        B, F, N, T = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, F, T)
        h = self.proj(x)
        for layer in self.layers:
            h = h + layer(h)
        h = h.reshape(B, N, -1, T)
        h = h.permute(0, 1, 3, 2)  # (B, N, T, D)
        h = self.norm(h)
        h = h.permute(0, 3, 1, 2)  # (B, D, N, T)
        return h


class SlotAggregation(nn.Module):
    """
    Slot Attention 聚合

    将 N_obs 个观测节点的信息聚合到 K 个 slots 中
    K << N_all，所以无法为每个节点存储独立表示
    """

    def __init__(self, hidden_dim: int, num_slots: int, n_iters: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.n_iters = n_iters

        # 可学习的 slot 初始化
        self.slots_init = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.02)

        # Attention 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # GRU 更新
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_inputs = nn.LayerNorm(hidden_dim)

    def forward(self, h_obs: torch.Tensor, density: float):
        """
        h_obs: (B, D, N_obs, T)
        density: 观测比例

        返回: (B, K, D, T) - K 个 slots 的时序表示
        """
        B, D, N_obs, T = h_obs.shape
        K = self.num_slots
        device = h_obs.device

        # 对每个时间步独立做 Slot Attention
        slots_all_t = []

        for t in range(T):
            h_t = h_obs[:, :, :, t]  # (B, D, N_obs)
            h_t = h_t.permute(0, 2, 1)  # (B, N_obs, D)
            h_t = self.norm_inputs(h_t)

            # 初始化 slots
            slots = self.slots_init.expand(B, -1, -1).clone()  # (B, K, D)

            # 迭代更新 slots
            for _ in range(self.n_iters):
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention: slots attend to inputs
                q = self.q_proj(slots)  # (B, K, D)
                k = self.k_proj(h_t)    # (B, N_obs, D)
                v = self.v_proj(h_t)    # (B, N_obs, D)

                # Attention weights
                scale = math.sqrt(D)
                attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, K, N_obs)
                attn = F.softmax(attn, dim=-1)  # 在输入维度归一化

                # 加权聚合
                updates = torch.matmul(attn, v)  # (B, K, D)

                # GRU 更新
                slots = self.gru(
                    updates.reshape(B * K, D),
                    slots_prev.reshape(B * K, D)
                ).reshape(B, K, D)

                # MLP
                slots = slots + self.mlp(slots)

            slots_all_t.append(slots)

        # Stack: (B, K, D, T)
        slots_out = torch.stack(slots_all_t, dim=-1)

        return slots_out


class SinusoidalCrossAttention(nn.Module):
    """
    使用正弦位置编码的 Cross-Attention

    关键：Query 使用固定的正弦位置编码，而非可学习的 embedding
    这防止了模型为特定位置"记忆"特定输出
    """

    def __init__(self, hidden_dim: int, num_nodes: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # 固定正弦位置编码（不可学习！）
        pe = create_sinusoidal_encoding(num_nodes, hidden_dim)
        self.register_buffer('position_encoding', pe)

        # Query 融合层：位置编码 + 密度信息
        self.query_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for density
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, slots: torch.Tensor, idx_obs: torch.Tensor, density: float):
        """
        slots: (B, K, D, T) - 聚合后的 slots
        idx_obs: (N_obs,) - 观测节点索引
        density: 观测比例

        返回: (B, D, N_all, T)
        """
        B, K, D, T = slots.shape
        N_all = self.num_nodes
        device = slots.device

        # 准备 Query：正弦位置编码 + 密度
        pe = self.position_encoding  # (N_all, D)
        density_vec = torch.full((N_all, 1), density, device=device, dtype=slots.dtype)
        query_input = torch.cat([pe, density_vec], dim=-1)  # (N_all, D+1)
        query_base = self.query_net(query_input)  # (N_all, D)

        h_all_list = []

        for t in range(T):
            slots_t = slots[:, :, :, t]  # (B, K, D)

            # Query
            q = query_base.unsqueeze(0).expand(B, -1, -1)  # (B, N_all, D)
            q = self.q_proj(q)

            # Key, Value from slots
            k = self.k_proj(slots_t)  # (B, K, D)
            v = self.v_proj(slots_t)  # (B, K, D)

            # Multi-head attention
            q = q.view(B, N_all, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

            scale = math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)  # (B, heads, N_all, head_dim)
            out = out.transpose(1, 2).reshape(B, N_all, D)
            out = self.out_proj(out)
            out = self.norm(query_base.unsqueeze(0) + out)

            h_all_list.append(out)

        h_all = torch.stack(h_all_list, dim=-1)  # (B, N_all, D, T)
        h_all = h_all.permute(0, 2, 1, 3)  # (B, D, N_all, T)

        return h_all


class SlotBasedEncoder(nn.Module):
    """
    Encoder V6: Slot-Based 编码器

    关键设计：
    1. 保留 h_obs 的完整信息，不压缩到单一向量
    2. 使用 K 个 slots 聚合信息（K << N，无法逐节点记忆）
    3. 使用固定正弦位置编码（非学习，无法记忆）
    4. 生成依赖 slots 内容 + 位置，而非固定身份

    与 V5 的对比：
    - V5: N_obs → 1 向量 → N_all (信息瓶颈)
    - V6: N_obs → K slots → N_all (保留多样性，K=16-32)

    与 nn.Embedding 的对比：
    - nn.Embedding: 节点 i → 固定向量 (可记忆)
    - V6 正弦编码: 节点 i → 固定正弦函数值 (不可学习)
    - V6 输出: 正弦编码 × Attention(slots) (依赖内容)
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        num_slots: int = 16,
        slot_iters: int = 3,
        n_temporal_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.2,
        device: str = 'cuda'
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots

        # 时序编码
        self.temporal = TemporalEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            n_layers=n_temporal_layers,
            dropout=dropout
        )

        # Slot 聚合
        self.slot_agg = SlotAggregation(
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            n_iters=slot_iters,
            dropout=dropout
        )

        # 正弦位置 Cross-Attention
        self.cross_attn = SinusoidalCrossAttention(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            n_heads=n_heads,
            dropout=dropout
        )

        # 观测节点注入
        self.inject_gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x_subset: torch.Tensor,
        idx_subset: Union[torch.Tensor, np.ndarray, List]
    ) -> torch.Tensor:
        """
        x_subset: (B, F, N_obs, T)
        idx_subset: 观测节点索引

        返回: (B, D, N_all, T)
        """
        B, F, N_obs, T = x_subset.shape
        device = x_subset.device

        if isinstance(idx_subset, np.ndarray):
            idx_subset = torch.from_numpy(idx_subset).to(device)
        elif isinstance(idx_subset, list):
            idx_subset = torch.tensor(idx_subset, device=device)

        density = N_obs / self.num_nodes

        # 1. 时序编码
        h_obs = self.temporal(x_subset)  # (B, D, N_obs, T)

        # 2. Slot 聚合（保留多样性，不压缩到 1）
        slots = self.slot_agg(h_obs, density)  # (B, K, D, T)

        # 3. 正弦位置 Cross-Attention 生成所有节点
        h_all = self.cross_attn(slots, idx_subset, density)  # (B, D, N_all, T)

        # 4. 注入观测节点特征
        gate = torch.sigmoid(self.inject_gate)
        h_all[:, :, idx_subset, :] = gate * h_all[:, :, idx_subset, :] + (1 - gate) * h_obs

        return h_all

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)