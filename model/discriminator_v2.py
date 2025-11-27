import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.dim = dim
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=dim,
            bias=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, N, T = x.shape
        residual = x

        h = x.reshape(B * N, D, T)
        h = self.conv(h)
        h = F.gelu(h)
        h = h.reshape(B, D, N, T)
        h = residual + h
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = h.permute(0, 3, 1, 2)

        return h


class FullSequenceDiscriminator(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.patch_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.tc_block1 = TemporalConvBlock(hidden_dim, dilation=1)
        self.tc_block2 = TemporalConvBlock(hidden_dim, dilation=2)

        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.patch_conv.weight, mode='fan_in', nonlinearity='linear')
        if self.patch_conv.bias is not None:
            nn.init.zeros_(self.patch_conv.bias)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F_in, N, T = x.shape

        x_flat = x.reshape(B * N, F_in, T)
        h = self.patch_conv(x_flat)
        h = F.gelu(h)
        h = h.reshape(B, self.hidden_dim, N, T)

        h = self.tc_block1(h)
        h = self.tc_block2(h)

        mean_N = h.mean(dim=2)
        agg_a = mean_N.mean(dim=2)
        agg_b = h.mean(dim=(2, 3))
        mean_T = h.mean(dim=3)
        agg_c = mean_T.mean(dim=2)
        mean_T_first = h.mean(dim=3)
        agg_d = mean_T_first.mean(dim=2)

        agg = torch.cat([agg_a, agg_b, agg_c, agg_d], dim=1)
        score = self.mlp(agg)

        return score

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_discriminator(in_dim: int, hidden_dim: int) -> FullSequenceDiscriminator:
    return FullSequenceDiscriminator(in_dim=in_dim, hidden_dim=hidden_dim)