"""
Encoder V4: 节点感知时序编码器

核心改进（解决原模型问题）：
1. 保留节点 embedding（解决"不同子集输出相同"问题）
2. 因果时序注意力（解决"时序信息丢失"问题）
3. 分离空间和时间编码（更清晰的职责）
4. 去除过度压缩的 Slot Attention

设计理念（类比 LLM）：
- 节点 embedding ≈ Token embedding
- 因果时序注意力 ≈ Causal Self-Attention
- 空间注意力 ≈ Cross-Attention between tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Union, Optional
import numpy as np

class SharedNodeEmbedding(nn.Module):
    """
    共享节点 Embedding 模块

    用于在 Encoder、SubsetToFullExpander、PredHead 之间共享同一套节点表示。
    这确保：
    1. 语义一致性：同一节点在所有模块有相同的基础表示
    2. 梯度流完整：不论节点是观测还是缺失，embedding 都能得到有效更新
    3. 更好的泛化：训练和测试时节点表示一致
    """

    def __init__(self, num_nodes: int, hidden_dim: int, init_std: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.embed = nn.Parameter(torch.randn(num_nodes, hidden_dim) * init_std)

    def forward(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取节点 embedding

        Args:
            idx: 节点索引，如果为 None 则返回全部

        Returns:
            (N, D) 或 (len(idx), D)
        """
        if idx is None:
            return self.embed
        return self.embed[idx]

    def __getitem__(self, idx):
        """支持索引访问"""
        return self.embed[idx]

    @property
    def weight(self):
        """兼容旧接口"""
        return self.embed
    
def create_sinusoidal_encoding(num_positions: int, dim: int) -> torch.Tensor:
    """创建正弦位置编码"""
    position = torch.arange(num_positions).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

    pe = torch.zeros(num_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

class NodeAwareTemporalEncoder(nn.Module):
    """
    节点感知的时序编码器

    关键设计：
    1. 每个节点有独立的 embedding（保留节点差异）
    2. 时序使用因果注意力（学习时间因果）
    3. 空间使用交叉注意力（学习节点关系）
    4. 【V4.1】支持共享节点 embedding，确保语义一致性
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        device: str = 'cuda',
        shared_node_embed: Optional[SharedNodeEmbedding] = None
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # ========== 输入投影 ==========
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            nn.LayerNorm([hidden_dim]),  # 注意：这里需要调整
        )
        # 使用 1D 卷积后的 LayerNorm
        self.input_norm = nn.LayerNorm(hidden_dim)

        # ========== 节点 Embedding（关键！）==========
        # 【V4.1】支持共享 embedding 或自己创建
        if shared_node_embed is not None:
            self.node_embed = shared_node_embed
            self._owns_node_embed = False
        else:
            # 向后兼容：如果没有传入共享 embedding，自己创建
            self.node_embed = SharedNodeEmbedding(num_nodes, hidden_dim, init_std=0.1)
            self._owns_node_embed = True

        # ========== 时序位置编码 ==========
        # 固定正弦编码（非学习）
        self.register_buffer('time_pe', create_sinusoidal_encoding(512, hidden_dim))

        # ========== 编码层 ==========
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                EncoderLayer(hidden_dim, n_heads, dropout)
            )

        # ========== 子集到全集的扩展模块 ==========
        # 【V4.1】传入共享的 node_embed
        self.subset_to_full = SubsetToFullExpander(
            hidden_dim, num_nodes, n_heads, dropout,
            shared_node_embed=self.node_embed
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

    def get_shared_node_embed(self) -> SharedNodeEmbedding:
        """获取共享的节点 embedding，供 PredHead 使用"""
        return self.node_embed

    def forward(
        self,
        x_subset: torch.Tensor,
        idx_subset: Union[torch.Tensor, np.ndarray, List],
        return_obs_clean: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        x_subset: (B, F, N_obs, T)
        idx_subset: 观测节点索引

        返回: 
        (B, D, N_all, T) 或 ((B, D, N_all, T), (B, D, N_obs, T))
        """
        B, F, N_obs, T = x_subset.shape
        device = x_subset.device

        # 处理索引
        if isinstance(idx_subset, np.ndarray):
            idx_subset = torch.from_numpy(idx_subset).to(device)
        elif isinstance(idx_subset, list):
            idx_subset = torch.tensor(idx_subset, device=device)

        # ========== 1. 输入投影 ==========
        # (B, F, N_obs, T) → (B*N_obs, F, T) → (B*N_obs, D, T)
        x = x_subset.permute(0, 2, 1, 3).reshape(B * N_obs, F, T)
        x = self.input_proj[0](x)  # Conv1d
        x = x.permute(0, 2, 1)  # (B*N_obs, T, D)
        x = self.input_norm(x)

        # ========== 2. 添加节点 embedding ==========
        # 取出观测节点的 embedding（使用共享 embedding）
        node_emb = self.node_embed(idx_subset)  # (N_obs, D)
        node_emb = node_emb.unsqueeze(0).unsqueeze(2)  # (1, N_obs, 1, D)
        node_emb = node_emb.expand(B, -1, T, -1)  # (B, N_obs, T, D)
        node_emb = node_emb.reshape(B * N_obs, T, -1)  # (B*N_obs, T, D)

        x = x + node_emb

        # ========== 3. 添加时序位置编码 ==========
        time_pe = self.time_pe[:T, :].unsqueeze(0)  # (1, T, D)
        x = x + time_pe

        # ========== 4. 编码层（时序 + 空间交替）==========
        # (B*N_obs, T, D) → (B, N_obs, T, D)
        x = x.reshape(B, N_obs, T, -1)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # 可选返回观测子集的干净表示
        h_obs_clean = x.permute(0, 3, 1, 2)  # (B, D, N_obs, T)

        # ========== 5. 扩展到全集 ==========
        # x: (B, N_obs, T, D) → (B, N_all, T, D)
        h_all = self.subset_to_full(x, idx_subset)

        # (B, N_all, T, D) → (B, D, N_all, T)
        h_all = h_all.permute(0, 3, 1, 2)

        if return_obs_clean:
            return h_all, h_obs_clean
        return h_all
    
    def replace_obs_with_clean(
        self,
        h_all: torch.Tensor,
        h_obs_clean: torch.Tensor,
        idx_subset: Union[torch.Tensor, np.ndarray, List]
    ) -> torch.Tensor:
        """
        使用干净的观测子集表示替换全集中的对应部分

        h_all: (B, D, N_all, T)
        h_obs_clean: (B, D, N_obs, T)
        idx_subset: 观测节点索引

        返回:
        (B, D, N_all, T)
        """
        h_all_fixed = h_all.clone()
        h_all_fixed[:, :, idx_subset, :] = h_obs_clean
        return h_all_fixed  


class EncoderLayer(nn.Module):
    """
    编码器层：时序注意力 + 空间注意力 + FFN
    """
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()

        # 时序注意力（因果）
        self.temporal_attn = CausalTemporalAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # 空间注意力（节点间）
        self.spatial_attn = SpatialAttention(hidden_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """x: (B, N, T, D)"""
        # 时序注意力
        residual = x
        x = self.norm1(x)
        x = residual + self.temporal_attn(x)

        # 空间注意力
        residual = x
        x = self.norm2(x)
        x = residual + self.spatial_attn(x)

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)

        return x


class CausalTemporalAttention(nn.Module):
    """
    因果时序注意力

    关键：每个时间步只能看到之前的时间步（因果掩码）
    这让模型学习时间因果关系
    """
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, N, T, D)"""
        B, N, T, D = x.shape

        # (B, N, T, D) → (B*N, T, D)
        x = x.reshape(B * N, T, D)

        # QKV 投影
        qkv = self.qkv(x).reshape(B * N, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*N, heads, T, T)

        # 因果掩码
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 输出
        out = (attn @ v).transpose(1, 2).reshape(B * N, T, D)
        out = self.out_proj(out)

        return out.reshape(B, N, T, D)


class SpatialAttention(nn.Module):
    """
    空间注意力（节点间）

    对每个时间步，计算节点间的注意力
    """
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, N, T, D)"""
        B, N, T, D = x.shape

        # (B, N, T, D) → (B*T, N, D)
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, D)

        # QKV 投影
        qkv = self.qkv(x).reshape(B * T, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数（无掩码，所有节点互相可见）
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 输出
        out = (attn @ v).transpose(1, 2).reshape(B * T, N, D)
        out = self.out_proj(out)

        return out.reshape(B, T, N, D).permute(0, 2, 1, 3)  # (B, N, T, D)


class SubsetToFullExpander(nn.Module):
    """
    子集到全集扩展器

    关键改进：
    1. 使用交叉注意力而非 Slot（避免信息瓶颈）
    2. 缺失节点从观测节点的上下文中获取信息
    3. 保留节点 embedding 差异
    4. 【V4.1】使用共享节点 embedding，确保语义一致性
    """
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        n_heads: int,
        dropout: float,
        shared_node_embed: Optional[SharedNodeEmbedding] = None
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 【V4.1】使用共享节点 embedding
        # 不再创建独立的 embedding，而是引用共享的
        self.node_embed = shared_node_embed  # 可能为 None（向后兼容）

        # 交叉注意力：缺失节点从观测节点获取信息
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # 融合门控
        self.gate = nn.Parameter(torch.tensor(-2.0))  # 初始倾向于使用原始特征

    def forward(self, h_obs: torch.Tensor, idx_subset: torch.Tensor) -> torch.Tensor:
        """
        h_obs: (B, N_obs, T, D)
        idx_subset: (N_obs,)

        返回: (B, N_all, T, D)
        """
        B, N_obs, T, D = h_obs.shape
        device = h_obs.device

        # 创建全集输出
        h_full = torch.zeros(B, self.num_nodes, T, D, device=device, dtype=h_obs.dtype)

        # 1. 观测节点直接复制
        h_full[:, idx_subset, :, :] = h_obs

        # 2. 找出缺失节点
        all_idx = torch.arange(self.num_nodes, device=device)
        missing_mask = torch.ones(self.num_nodes, dtype=torch.bool, device=device)
        missing_mask[idx_subset] = False
        missing_idx = all_idx[missing_mask]

        if len(missing_idx) == 0:
            return h_full

        # 3. 缺失节点通过交叉注意力从观测节点获取信息
        # Query: 缺失节点的 embedding（使用共享 embedding）
        missing_emb = self.node_embed(missing_idx)  # (N_miss, D)

        # 【V4.1 优化】批量处理所有时间步，提高效率
        # 将 (B, N_obs, T, D) 转为 (B*T, N_obs, D)
        context_all = h_obs.permute(0, 2, 1, 3).reshape(B * T, N_obs, D)

        # 将 query 扩展为 (B*T, N_miss, D)
        query = missing_emb.unsqueeze(0).expand(B * T, -1, -1)  # (B*T, N_miss, D)
        query = self.norm(query)

        # 一次 cross-attention 处理所有时间步
        h_missing_all, _ = self.cross_attn(query, context_all, context_all)

        # 融合：节点 embedding + 上下文信息
        gate = torch.sigmoid(self.gate)
        h_missing_all = gate * query + (1 - gate) * h_missing_all

        # reshape 回 (B, T, N_miss, D) → (B, N_miss, T, D)
        h_missing_all = h_missing_all.reshape(B, T, len(missing_idx), D).permute(0, 2, 1, 3)

        h_full[:, missing_idx, :, :] = h_missing_all

        return h_full

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)