import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class STDecoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=3,
            dilation=1,
            padding=1,
            groups=in_dim,
            bias=True
        )

        self.conv2 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=3,
            dilation=2,
            padding=2,
            groups=in_dim,
            bias=True
        )

        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='linear')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, h_final: torch.Tensor) -> torch.Tensor:
        B, D, N, T = h_final.shape

        h = h_final.reshape(B * N, D, T)
        h = self.conv1(h)
        h = F.gelu(h)
        h = self.conv2(h)
        h = F.gelu(h)
        h = h.view(B, D, N, T)

        h = h_final + h
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)

        y_pred = self.proj(h)
        y_pred = y_pred.permute(0, 3, 1, 2)

        return y_pred

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_stdecoder(encoder_dim: int, out_dim: int) -> STDecoder:
    return STDecoder(in_dim=encoder_dim, out_dim=out_dim)