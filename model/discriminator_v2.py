import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, use_spectral_norm: bool = False):
        super().__init__()
        self.dim = dim
        self.dilation = dilation

        # 选项1：保持深度可分离卷积
        # 选项2：改为标准卷积（推荐，因为谱归一化对标准卷积效果更好）
        conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            # groups=dim,  # 移除分组，改为标准卷积
            bias=True
        )
        
        # 应用谱归一化
        if use_spectral_norm:
            self.conv = spectral_norm(conv)
        else:
            self.conv = conv
        
        self.norm = nn.LayerNorm(dim)
        
        # 初始化
        self._init_weights(use_spectral_norm)

    def _init_weights(self, use_spectral_norm):
        if use_spectral_norm:
            # 谱归一化层的初始化
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        else:
            # 普通卷积的初始化
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
        
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, N, T = x.shape
        residual = x

        h = x.reshape(B * N, D, T)
        h = self.conv(h)
        h = F.gelu(h)
        h = h.reshape(B, D, N, T)
        h = residual + h  # 残差连接
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = h.permute(0, 3, 1, 2)

        return h


class FullSequenceDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, use_spectral_norm: bool = True, dropout: float = 0.1):
        """
        判别器：对整个时空序列打分。

        参数
        ----
        in_dim : 输入特征维度
        hidden_dim : 隐层维度
        use_spectral_norm : 是否对卷积层和全连接层启用谱归一化
        dropout : Dropout率，防止过拟合
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.use_spectral_norm = use_spectral_norm
        
        # 第一步：投影到隐层维度
        patch_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # 应用谱归一化到初始卷积层
        if use_spectral_norm:
            self.patch_conv = spectral_norm(patch_conv)
        else:
            self.patch_conv = patch_conv
        
        # 时序卷积块
        self.tc_block1 = TemporalConvBlock(hidden_dim, dilation=1, use_spectral_norm=use_spectral_norm)
        self.tc_block2 = TemporalConvBlock(hidden_dim, dilation=2, use_spectral_norm=use_spectral_norm)
        
        # 可选：添加更多时序卷积块
        self.tc_block3 = TemporalConvBlock(hidden_dim, dilation=4, use_spectral_norm=use_spectral_norm)
        
        # MLP 判别头 - 增强判别能力
        # 注意：spectral_norm 返回的是包装后的模块，不是原始层
        if use_spectral_norm:
            fc1 = spectral_norm(nn.Linear(4 * hidden_dim, hidden_dim * 2))
            fc2 = spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim))
            fc3 = spectral_norm(nn.Linear(hidden_dim, 1))
        else:
            fc1 = nn.Linear(4 * hidden_dim, hidden_dim * 2)
            fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
            fc3 = nn.Linear(hidden_dim, 1)
        
        self.mlp = nn.Sequential(
            fc1,
            nn.GELU(),
            nn.Dropout(dropout),
            fc2,
            nn.GELU(),
            nn.Dropout(dropout),
            fc3
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化初始卷积层
        if not self.use_spectral_norm:  # 谱归一化层不需要额外的初始化
            nn.init.kaiming_normal_(self.patch_conv.weight, mode='fan_in', nonlinearity='linear')
            if self.patch_conv.bias is not None:
                nn.init.zeros_(self.patch_conv.bias)
        
        # 初始化MLP（如果未应用谱归一化）
        for module in self.mlp:
            if isinstance(module, nn.Linear) and not self.use_spectral_norm:
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数
        ----
        x : 形状为 (B, F_in, N, T) 的输入张量
        
        返回
        ----
        score : 形状为 (B, 1) 的判别分数
        """
        B, F_in, N, T = x.shape
        
        # 1. 初始投影
        x_flat = x.reshape(B * N, F_in, T)
        h = self.patch_conv(x_flat)
        h = F.gelu(h)
        h = h.reshape(B, self.hidden_dim, N, T)
        
        # 2. 时序卷积块
        h = self.tc_block1(h)
        h = self.tc_block2(h)
        h = self.tc_block3(h)  # 新增的第三个块
        
        # 3. 聚合特征
        # 多种聚合方式，增加判别器的视野
        mean_N = h.mean(dim=2)  # 沿节点维度平均
        agg_a = mean_N.mean(dim=2)  # 沿时间维度平均
        
        mean_T = h.mean(dim=3)  # 沿时间维度平均
        agg_b = mean_T.mean(dim=2)  # 沿节点维度平均
        
        # 最大值聚合
        max_N = h.max(dim=2)[0]
        agg_c = max_N.mean(dim=2)
        
        # 标准差聚合（捕获波动信息）
        std_N = h.std(dim=2)
        agg_d = std_N.mean(dim=2)
        
        # 拼接所有聚合特征
        agg = torch.cat([agg_a, agg_b, agg_c, agg_d], dim=1)
        
        # 4. MLP判别头
        score = self.mlp(agg)
        
        return score
    
    def get_num_parameters(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_discriminator(
    in_dim: int, 
    hidden_dim: int, 
    use_spectral_norm: bool = True,
    dropout: float = 0.1
) -> FullSequenceDiscriminator:
    """创建判别器的工厂函数"""
    return FullSequenceDiscriminator(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        use_spectral_norm=use_spectral_norm,
        dropout=dropout
    )