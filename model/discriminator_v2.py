import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class NodeTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, use_spectral_norm: bool = True ):
        super().__init__()

        norm_fn = spectral_norm if use_spectral_norm else lambda x: x

        self.temporal_conv = nn.Sequential(
            norm_fn(nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))),
            nn.LeakyReLU(0.2),
            norm_fn(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2))),
            nn.LeakyReLU(0.2)
        )
        self.node_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = norm_fn(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x:(B, F, N, T) -> (B, H)"""
        x = self.temporal_conv(x)
        x = self.node_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


class HybridNodeDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64,use_spectral_norm: bool = True):
        super().__init__()

        self.use_spectral_norm = use_spectral_norm
        norm_fn = spectral_norm if use_spectral_norm else lambda x: x
        
        self.sub_encoder = NodeTemporalEncoder(feature_dim, hidden_dim, use_spectral_norm)
        self.miss_encoder = NodeTemporalEncoder(feature_dim, hidden_dim, use_spectral_norm)
        self.cond_head = nn.Sequential(
            norm_fn(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.LeakyReLU(0.2),
            norm_fn(nn.Linear(hidden_dim, 1))
        )

        self.internal_encoder = NodeTemporalEncoder(feature_dim, hidden_dim, use_spectral_norm)
        self.internal_head = nn.Sequential(
            norm_fn(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            norm_fn(nn.Linear(hidden_dim // 2, 1))
        )

    def forward(self, subset_nodes: torch.Tensor, missing_nodes: torch.Tensor):
        """前向传播，返回条件判别分数和内部判别分数"""
        subset_feat = self.sub_encoder(subset_nodes)
        miss_feat = self.miss_encoder(missing_nodes)

        # 修正：使用 torch.cat() 而不是 torch.cat[]
        cond_score = self.cond_head(torch.cat([subset_feat, miss_feat], dim=-1))

        internal_feat = self.internal_encoder(missing_nodes)
        internal_score = self.internal_head(internal_feat)

        return cond_score, internal_score

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_hybrid_d_loss(cond_real, cond_fake, internal_real, internal_fake, alpha=0.7):
    """混合判别器的判别损失"""
    cond_loss = (
            F.binary_cross_entropy_with_logits(cond_real, torch.ones_like(cond_real) * 0.9) +
            F.binary_cross_entropy_with_logits(cond_fake, torch.zeros_like(cond_fake))
    )
    internal_loss = (
            F.binary_cross_entropy_with_logits(internal_real, torch.ones_like(internal_real) * 0.9) +
            F.binary_cross_entropy_with_logits(internal_fake, torch.zeros_like(internal_fake))
    )
    return alpha * cond_loss + (1 - alpha) * internal_loss


def compute_hybrid_g_loss(cond_fake, internal_fake, alpha=0.7):
    """混合判别器的生成器损失"""
    cond_loss = F.binary_cross_entropy_with_logits(cond_fake, torch.ones_like(cond_fake))
    internal_loss = F.binary_cross_entropy_with_logits(internal_fake, torch.ones_like(internal_fake))
    return alpha * cond_loss + (1 - alpha) * internal_loss


def create_discriminator(
        feature_dim: int,
        hidden_dim: int = 64,
        use_spectral_norm: bool = True 
) -> HybridNodeDiscriminator:
    """创建混合判别器的工厂函数"""
    return HybridNodeDiscriminator(feature_dim=feature_dim, hidden_dim=hidden_dim, use_spectral_norm=use_spectral_norm)