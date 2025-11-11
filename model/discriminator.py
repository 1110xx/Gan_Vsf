import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入图卷积层
try:
    from .encoder import GraphConvLayer
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os

    sys.path.append(os.path.dirname(__file__))
    from encoder import GraphConvLayer


class GraphDiscriminator(nn.Module):
    """
    图结构判别器：判断重构全局数据是否真实
    输入：(B, F, N, T) - 完整集数据 + (N, N) - 邻接矩阵
    输出：(B, 1) - 真实概率 [0, 1]
    架构：
    数据 + 邻接矩阵 -> 图卷积 -> 时序卷积 -> 全连接 -> 概率
    """


    def __init__(self, params):
        super(GraphDiscriminator, self).__init__()
        self.in_dim = params['in_dim']  # 输入特征维度
        self.num_nodes = params['num_nodes']  # 完整集节点数
        self.seq_len = params['seq_in_len']  # 序列长度
        self.hidden_dim = params.get('disc_hidden_dim', 128)
        self.dropout = params.get('dropout', 0.3)
        # ========== 图卷积层：捕获空间依赖 ==========
        # 先将时空特征flatten，再通过图卷积学习空间模式
        self.graph_conv = GraphConvLayer(
            in_dim=self.in_dim * self.seq_len,  # 输入是flatten的时空特征
            out_dim=self.hidden_dim,
            dropout=self.dropout
        )
        # ========== 时序卷积层：捕获时间模式 ==========
        # 使用2D卷积在(节点, 时间)维度上捕获模式
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=64,
                kernel_size=(1, 3),  # (节点维度, 时间维度)
                padding=(0, 1)
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(1, 3),
                padding=(0, 1)
            ),
            nn.ReLU()
        )
        # ========== 全连接分类层 ==========
        # 计算flatten后的维度
        flatten_dim = 32 * self.num_nodes * self.seq_len
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, adj=None):
        """
        前向传播
        Args:
        x: (B, F, N, T) - 完整集数据
        adj: (N, N) - 邻接矩阵（可选，默认使用单位矩阵）
        Returns:
        prob: (B, 1) - 真实概率
        """


        batch_size = x.size(0)
        # 如果未提供邻接矩阵，使用单位矩阵
        if adj is None:
            adj = torch.eye(self.num_nodes, device=x.device)
        # ========== 步骤1: 准备图卷积输入 ==========
        # (B, F, N, T) -> (B, N, F, T)
        x = x.permute(0, 2, 1, 3)
        # Flatten时空特征：(B, N, F, T) -> (B, N, F*T)
        x_flat = x.reshape(batch_size, self.num_nodes, self.in_dim * self.seq_len)
        # ========== 步骤2: 图卷积捕获空间模式 ==========
        # (B, N, F*T) -> (B, N, hidden_dim)
        h = self.graph_conv(x_flat, adj)
        # ========== 步骤3: Reshape为2D卷积格式 ==========
        # (B, N, hidden_dim) -> (B, hidden_dim, N, 1)
        h = h.permute(0, 2, 1).unsqueeze(-1)
        # Expand时间维度
        # (B, hidden_dim, N, 1) -> (B, hidden_dim, N, T)
        h = h.repeat(1, 1, 1, self.seq_len)
        # ========== 步骤4: 时序卷积捕获时间模式 ==========
        # (B, hidden_dim, N, T) -> (B, 32, N, T)
        h = self.temporal_conv(h)
        # ========== 步骤5: 全连接输出概率 ==========
        # (B, 32, N, T) -> (B, 1)
        logits = self.classifier(h)
        prob = self.sigmoid(logits)
        return prob


class SpatioTemporalDiscriminator(nn.Module):
    """
    时空判别器 - 原始版本（保留向后兼容）
    输入: (batch, in_dim, num_nodes, seq_in_len) - 完整集数据
    输出: (batch, 1) - 真实概率
    注意：判别器判别的是输入时间步的完整集，而不是预测输出
    """


    def __init__(self, params):
        super(SpatioTemporalDiscriminator, self).__init__()


        self.seq_len = params['seq_in_len']
        self.num_nodes = params['num_nodes']
        self.in_features = params['in_dim']
        self.encoder_lstm_units = params.get('encoder_lstm_units', 128)
        self.conv_filters = params.get('conv_filters', 64)
        self.kernel_size = params.get('kernel_size', 3)
        self.dropout = params.get('dropout', 0.3)
        # LSTM Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=self.num_nodes * self.in_features,
            hidden_size=self.encoder_lstm_units,
            batch_first=True,
            dropout=self.dropout if params.get('use_dropout', True) else 0.0
        )
        # Conv块
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=self.num_nodes * self.in_features,
                out_channels=self.conv_filters,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            )
        )
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=self.conv_filters,
                out_channels=self.conv_filters,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            )
        )
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.conv_filters) for _ in range(2)
        ]) if params.get('use_batchnorm', True) else None
        # 最终分类层
        flatten_size = self.encoder_lstm_units * self.seq_len + self.conv_filters * self.seq_len
        self.fc = nn.Linear(flatten_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        前向传播 - 判别完整集（原始版本，不使用邻接矩阵）
        Args:
        x: (batch, in_dim, num_nodes, seq_in_len) - 完整集数据 (B, F, N, T)
        Returns:
        prob: (batch, 1) - 真实概率
        """


        batch_size = x.size(0)
        # 转换输入格式：(B, F, N, T) -> (B, T, N, F)
        x = x.transpose(1, 3)
        # Reshape: (B, T, N, F) -> (B, T, N*F)
        x_flat = x.reshape(batch_size, x.size(1), -1)
        # LSTM Encoder
        lstm_out, _ = self.encoder_lstm(x_flat)
        # Conv处理
        x_conv = x_flat.permute(0, 2, 1)  # (B, N*F, T)
        conv_out = x_conv
        for i, conv_layer in enumerate(self.conv_layers):
            conv_out = conv_layer(conv_out)
        if self.batch_norms is not None:
            conv_out = self.batch_norms[i](conv_out)
        conv_out = torch.tanh(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # (B, T, filters)
        # 拼接并展平
        combined = torch.cat([lstm_out, conv_out], dim=-1)
        combined_flat = combined.reshape(batch_size, -1)
        # 分类
        logits = self.fc(combined_flat)
        prob = self.sigmoid(logits)
        return prob
