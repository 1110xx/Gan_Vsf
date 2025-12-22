import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PrototypeRoutingModule(nn.Module):

    def __init__(
        self,
        D: int,
        K: int = 32,
        num_nodes: int = 140,
        seq_len: int = 12,
        summary_pool: str = 'mean',
        use_projection: bool = False,
        proj_dim: Optional[int] = None,
        use_ema: bool = True,
        ema_momentum: float = 0.99,
        node_emb_dropout: float = 0.5
    ):
        super().__init__()
        self.D = D
        self.K = K
        self.num_nodes = num_nodes,
        self.seq_len = seq_len,
        self.summary_pool = summary_pool
        self.use_projection = use_projection
        self.proj_dim = proj_dim if proj_dim is not None else D
        self.use_ema = use_ema
        self.ema_momentum =ema_momentum
        self.node_emb_dropout_rate = node_emb_dropout

        self.prototypes = nn.Parameter(torch.randn(K, D))
        nn.init.orthogonal_(self.prototypes)

        self.prototype_temporal = nn.Parameter(torch.randn(K, D, seq_len) * 0.02)
        if use_ema:
            self.register_buffer('prototype_temporal_ema', torch.zeros(K, D, seq_len))
            self.register_buffer('ema_initialized', torch.tensor(False))
            



        self.node_embedding = nn.Embedding(num_nodes, D)
        nn.init.normal_(self.node_embedding.weight, mean=0, std=0.02)

        self.node_emb_dropout = nn.Dropout(p=node_emb_dropout)

        self.node_to_prototype = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, D)
        )

        if self.use_projection:
            self.proj_summary = nn.Sequential(
                nn.Linear(D, self.proj_dim),
                nn.LayerNorm(self.proj_dim),
                nn.GELU()
            )
            self.proj_prototypes = nn.Linear(D, self.proj_dim)

        if self.summary_pool == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(D, D // 4),
                nn.Tanh(),
                nn.Linear(D // 4, 1)
            )

    def compute_summary(self, h_time: torch.Tensor, mask: torch.Tensor, idx_obs: torch.Tensor) -> torch.Tensor:
        B, D, N, T = h_time.shape

        h_obs = h_time[:, :, idx_obs, :]
        mask_obs = mask[:, :, idx_obs, :]

        if self.summary_pool == 'mean':
            mask_sum = mask_obs.sum(dim=-1).clamp(min=1)
            s_obs = (h_obs * mask_obs).sum(dim=-1) / mask_sum
            s_obs = s_obs.permute(0, 2, 1)

        elif self.summary_pool == 'attention':
            h_obs_t = h_obs.permute(0, 2, 3, 1)
            attn_scores = self.attention_weights(h_obs_t)

            mask_obs_t = mask_obs.permute(0, 2, 3, 1)
            attn_scores = attn_scores.masked_fill(mask_obs_t < 0.5, -1e9)

            attn_weights = F.softmax(attn_scores, dim=2)
            s_obs = (h_obs_t * attn_weights).sum(dim=2)

        else:
            raise ValueError(f"Unknown summary_pool: {self.summary_pool}")

        return s_obs

    def compute_similarity(self, s: torch.Tensor) -> torch.Tensor:
        if self.use_projection:
            s_proj = self.proj_summary(s)
            prototypes_proj = self.proj_prototypes(self.prototypes)
            sim = torch.matmul(s_proj, prototypes_proj.T)
        else:
            s_norm = F.normalize(s, dim=-1)
            prototypes_norm = F.normalize(self.prototypes, dim=-1)
            sim = torch.matmul(s_norm, prototypes_norm.T)

        return sim
    def get_global_prototype_temporal(self, B: int, T: int) -> torch.Tensor:
        """
        获取全局原型时序表示

        核心改进：使用全局可学习参数，而不是从当前 batch 计算
        这确保了原型的稳定性，不受随机子集影响

        Returns:
            H_prototypes: (B, K, D, T) 全局稳定的原型时序表示
        """
        # 获取原型时序表示
        if self.use_ema and self.training:
            # 训练时更新 EMA
            with torch.no_grad():
                if not self.ema_initialized:
                    self.prototype_temporal_ema.copy_(self.prototype_temporal.data)
                    self.ema_initialized.fill_(True)
                else:
                    self.prototype_temporal_ema.mul_(self.ema_momentum).add_(
                        self.prototype_temporal.data, alpha=1 - self.ema_momentum
                    )

        # 使用可学习的原型时序
        proto_temporal = self.prototype_temporal  # (K, D, T_proto)
        K, D_dim,T_proto = proto_temporal.shape

        # 处理时间长度不匹配的情况
        if T != T_proto:
            # 插值到目标长度
            proto_temporal_flat =proto_temporal.reshape(K * D_dim, T_proto).unsqueeze(1)
            proto_temporal_interp = F.interpolate(
                proto_temporal_flat,  # (1, K, D, T_proto) -> treat K as batch
                size=T,
                mode='linear',
                align_corners=True
            ) # (K, D, T)
            proto_temporal = proto_temporal_interp.squeeze(1).reshape(K, D_dim, T)

        # 扩展到 batch 维度
        H_prototypes = proto_temporal.unsqueeze(0).expand(B, -1, -1, -1)  # (B, K, D, T)

        return H_prototypes

    def aggregate_prototype_timeseries(
        self,
        h_obs: torch.Tensor,
        alpha_obs: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, D, N_obs, T = h_obs.shape
        K = alpha_obs.shape[-1]

        alpha_expanded = alpha_obs.permute(0, 2, 1).unsqueeze(-1)
        h_obs_expanded = h_obs.unsqueeze(1)

        if confidence is not None:
            confidence_expanded = confidence.unsqueeze(1)
            weights = alpha_expanded.unsqueeze(2) * confidence_expanded
        else:
            weights = alpha_expanded.unsqueeze(2).expand(B, K, 1, N_obs, T)

        weighted_h = h_obs_expanded * weights
        H_prototypes_local = weighted_h.sum(dim=3)

        weights_sum = weights.sum(dim=3, keepdim=True).clamp(min=1e-8)
        H_prototypes_local = H_prototypes_local / weights_sum.squeeze(3)

        return H_prototypes_local

    def impute_unobserved(
        self,
        h_time: torch.Tensor,
        H_prototypes: torch.Tensor,
        mask: torch.Tensor,
        idx_obs: torch.Tensor,
        obs_context: torch.Tensor = None,
        obs_ratio: float = 0.15
    ) -> torch.Tensor:
        B, D, N, T = h_time.shape
        K = H_prototypes.shape[1]
        device = h_time.device

        all_idx = torch.arange(N, device=device)
        mask_obs_nodes = torch.zeros(N, dtype=torch.bool, device=device)
        mask_obs_nodes[idx_obs] = True
        idx_unobs = all_idx[~mask_obs_nodes]

        h_spatial = h_time.clone()

        if len(idx_unobs) > 0:
            node_emb_unobs = self.node_embedding(idx_unobs)
            if self.training:
                node_emb_unobs = self.node_emb_dropout(node_emb_unobs)
            if obs_context is not None:
                # obs_context: (B, D) - 观测节点的全局表示
                # 扩展到 N_unobs 维度并融合
                obs_context_exp = obs_context.unsqueeze(1).expand(-1, len(idx_unobs), -1)  # (B, N_unobs, D)
                node_emb_unobs_exp = node_emb_unobs.unsqueeze(0).expand(B, -1, -1)  # (B, N_unobs, D)
                context_weight = 0.1+0.8*obs_ratio

                # 融合节点身份和观测上下文（加权求和）
                context_weight = 0.5  # 观测上下文的权重
                combined_emb = (1 - context_weight) * node_emb_unobs_exp + context_weight * obs_context_exp

                # 通过投影层
                combined_emb_flat = combined_emb.reshape(B * len(idx_unobs), D)
                combined_proj = self.node_to_prototype(combined_emb_flat)
                combined_proj = combined_proj.reshape(B, len(idx_unobs), D)

                # 计算与原型的相似度
                # combined_proj: (B, N_unobs, D)
                prototypes_norm = F.normalize(self.prototypes, dim=-1)  # (K, D)
                combined_proj_norm = F.normalize(combined_proj, dim=-1)  # (B, N_unobs, D)
                sim_unobs = torch.matmul(combined_proj_norm, prototypes_norm.T)  # (B, N_unobs, K)
            else:
                # 回退到只用节点 embedding
                node_emb_proj = self.node_to_prototype(node_emb_unobs)  # (N_unobs, D)
                sim_unobs = self.compute_similarity(node_emb_proj)  # (N_unobs, K)
                sim_unobs = sim_unobs.unsqueeze(0).expand(B, -1, -1)  # (B, N_unobs, K)            

            alpha_unobs = F.softmax(sim_unobs, dim=-1)

            alpha_unobs_exp = alpha_unobs.unsqueeze(-1).unsqueeze(-1)
            H_prototypes_exp = H_prototypes.unsqueeze(1)

            h_unobs_imputed = (alpha_unobs_exp * H_prototypes_exp).sum(dim=2)
            h_unobs_imputed_per = h_unobs_imputed.permute(0, 2, 1, 3)
            h_spatial[:, :, idx_unobs, :] = h_unobs_imputed_per.to(h_spatial.dtype)

        return h_spatial

    def forward(
        self,
        h_time: torch.Tensor,
        mask: torch.Tensor,
        idx_obs: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, D, N, T = h_time.shape

        if isinstance(idx_obs, list):
            idx_obs = torch.tensor(idx_obs, device=h_time.device)
        elif not isinstance(idx_obs, torch.Tensor):
            idx_obs = torch.from_numpy(idx_obs).to(h_time.device)
        
        obs_ratio = len(idx_obs)/N

        H_prototypes_global = self.get_global_prototype_temporal(B, T)
        s_obs = self.compute_summary(h_time, mask, idx_obs)
        sim_obs = self.compute_similarity(s_obs)
        alpha_obs = F.softmax(sim_obs, dim=-1)

        h_obs = h_time[:, :, idx_obs, :]
        confidence_obs = confidence[:, :, idx_obs, :] if confidence is not None else None
        H_prototypes_local = self.aggregate_prototype_timeseries(
            h_obs, alpha_obs, confidence_obs
        )

        
        obs_context = s_obs.mean(dim=1)
        local_weight = 0.1 + 0.8 * obs_ratio
        global_weight = 1 - local_weight
        H_prototypes_mixd = global_weight * H_prototypes_global +local_weight * H_prototypes_local
        h_spatial = self.impute_unobserved(
            h_time, H_prototypes_mixd, mask, idx_obs, obs_context,obs_ratio
        )
        return h_spatial

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes.detach()

    def get_prototype_temporal(self) -> torch.Tensor:
        return self.prototype_temporal.detach()