import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PrototypeRoutingModule(nn.Module):

    def __init__(
        self,
        D: int,
        K: int = 32,
        summary_pool: str = 'mean',
        use_projection: bool = False,
        proj_dim: Optional[int] = None
    ):
        super().__init__()
        self.D = D
        self.K = K
        self.summary_pool = summary_pool
        self.use_projection = use_projection
        self.proj_dim = proj_dim if proj_dim is not None else D

        self.prototypes = nn.Parameter(torch.randn(K, D))
        nn.init.orthogonal_(self.prototypes)

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
        H_prototypes = weighted_h.sum(dim=3)

        weights_sum = weights.sum(dim=3, keepdim=True).clamp(min=1e-8)
        H_prototypes = H_prototypes / weights_sum.squeeze(3)

        return H_prototypes

    def impute_unobserved(
        self,
        h_time: torch.Tensor,
        H_prototypes: torch.Tensor,
        mask: torch.Tensor,
        idx_obs: torch.Tensor
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
            h_unobs = h_time[:, :, idx_unobs, :]
            mask_unobs = mask[:, :, idx_unobs, :]

            mask_sum = mask_unobs.sum(dim=-1).clamp(min=1)
            s_unobs = (h_unobs * mask_unobs).sum(dim=-1) / mask_sum
            s_unobs = s_unobs.permute(0, 2, 1)

            sim_unobs = self.compute_similarity(s_unobs)
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

        s_obs = self.compute_summary(h_time, mask, idx_obs)
        sim_obs = self.compute_similarity(s_obs)
        alpha_obs = F.softmax(sim_obs, dim=-1)

        h_obs = h_time[:, :, idx_obs, :]
        confidence_obs = confidence[:, :, idx_obs, :] if confidence is not None else None
        H_prototypes = self.aggregate_prototype_timeseries(
            h_obs, alpha_obs, confidence_obs
        )

        h_spatial = self.impute_unobserved(
            h_time, H_prototypes, mask, idx_obs
        )

        return h_spatial

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes.detach()