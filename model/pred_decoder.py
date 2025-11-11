import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.utils import weight_norm


# ============================================================

# TCN 基础模块

# ============================================================

class TemporalBlock(nn.Module):
    """

    时序卷积块 - TCN的基础构建模块

    架构：

    - 扩张因果卷积（Dilated Causal Convolution）

    - WeightNorm + ReLU + Dropout

    - Residual Connection

    输入/输出：(B*N, C, T)

    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        """

        初始化时序卷积块

        Args:

        in_channels: 输入通道数

        out_channels: 输出通道数

        kernel_size: 卷积核大小

        dilation: 扩张率

        dropout: dropout率

        """

        super(TemporalBlock, self).__init__()

        # 计算padding以保持因果性（只看过去）

        self.padding = (kernel_size - 1) * dilation

        # 第一层因果卷积

        self.conv1 = weight_norm(nn.Conv1d(

            in_channels, out_channels, kernel_size,

            padding=self.padding, dilation=dilation

        ))

        # 第二层因果卷积

        self.conv2 = weight_norm(nn.Conv1d(

            out_channels, out_channels, kernel_size,

            padding=self.padding, dilation=dilation))

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

        # Residual connection (如果输入输出维度不同需要投影)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):

        """

        前向传播

        Args:

        x: (B*N, C_in, T) - 时序特征

        Returns:

        out: (B*N, C_out, T) - 卷积后的特征

        """

        # 第一层卷积

        out = self.conv1(x)

        # 截断右侧padding保持因果性

        if self.padding > 0:
            out = out[:, :, :-self.padding]

            out = self.relu(out)

            out = self.dropout(out)

            # 第二层卷积

            out = self.conv2(out)

        if self.padding > 0:

            out = out[:, :, :-self.padding]

            out = self.relu(out)

            out = self.dropout(out)

            # Residual connection

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class LightGraphConv(nn.Module):


    """

    轻量图卷积层

    功能：在节点间传播信息，增强图结构依赖

    架构：

    - 简单的邻接矩阵乘法

    - 线性变换

    输入：特征 (B, N, C, T) + 邻接矩阵 (N, N)

    输出：(B, N, C, T)

    """
    def __init__(self, in_channels):


        """

        初始化轻量图卷积层

        Args:

        in_channels: 输入通道数

        """

        super(LightGraphConv, self).__init__()

        # 线性变换（可选，用于特征转换）

        self.linear = nn.Linear(in_channels, in_channels)

        self.activation = nn.ReLU()


    def forward(self,

                x, adj):


        """

        前向传播

        Args:

        x: (B, N, C, T) - 节点特征

        adj: (N, N) - 归一化的邻接矩阵

        Returns:

        out: (B, N, C, T) - 图卷积后的特征

        """

        B, N, C, T = x.shape

        # 重塑为 (B, T, N, C) 以便进行矩阵乘法

        x = x.permute(0, 3, 1, 2)  # (B, T, N, C)

        # 图卷积：邻接矩阵乘法 (N, N) @ (B, T, N, C) -> (B, T, N, C)

        # 使用 einsum 进行高效计算

        x_agg = torch.einsum('nm,btmc->btnc', adj, x)

        # 线性变换

        x_agg = self.linear(x_agg)  # (B, T, N, C)

        x_agg = self.activation(x_agg)

        # 恢复原始形状 (B, N, C, T)

        x_agg = x_agg.permute(0, 2, 3, 1)

        return x_agg


# ============================================================

# Graph-aware TCN 解码器

# ============================================================


class GraphAwareTCN(nn.Module):
    """

    Graph-aware TCN 解码器

    架构：

    - 多层 TCN Block，每层后跟一个 GraphConv 层

    - TCN 捕获时序依赖，GraphConv 增强节点间结构依赖

    输入：全局embedding (B, N, embedding_dim)

    输出：全局预测 (B, 1, N, T_out)

    流程：

    (B, N, embedding_dim)

    → 投影到 hidden_dim

    → 时间扩展到 T_out

    → reshape → (B*N, hidden_dim, T_out)

    → 逐层: TCN Block → reshape → GraphConv → reshape

    → 输出层 → (B, 1, N, T_out)

    """
    def __init__(self, params):
        """

        初始化 Graph-aware TCN 解码器

        Args:

        params: 配置字典，需要包含:

        - seq_out_len: 预测序列长度

        - num_nodes: 完整集节点数

        - encoder_lstm_units: 编码器LSTM单元数

        - conv_filters: 卷积滤波器数

        - tcn_kernel_size: TCN卷积核大小（默认3）

        - tcn_num_layers: TCN层数（默认4）

        - tcn_dilations: 扩张率列表（默认[1,2,4,8]）

        - tcn_hidden_dim: TCN隐藏维度（默认128）

        - tcn_dropout: TCN dropout率（默认0.2）

        - tcn_residual: 是否使用残差连接（默认True）

        """

        super(GraphAwareTCN, self).__init__()

        # 基本参数

        self.seq_out_len = params['seq_out_len']

        self.num_nodes = params['num_nodes']

        # embedding 维度

        self.embedding_dim = params.get('encoder_lstm_units', 128) + params.get('conv_filters', 64)

        # TCN 参数

        self.kernel_size = params.get('tcn_kernel_size', 3)

        self.num_layers = params.get('tcn_num_layers', 4)

        self.dilations = params.get('tcn_dilations', [1, 2, 4, 8])

        self.hidden_dim = params.get('tcn_hidden_dim', 128)

        self.dropout = params.get('tcn_dropout', 0.2)

        self.use_residual = params.get('tcn_residual', True)

        # 确保 dilations 长度与 num_layers 匹配

        if len(self.dilations) != self.num_layers:

            # 如果不匹配，自动生成 dilations

            self.dilations = [2 ** i for i in range(self.num_layers)]

        # ========== 输入投影层 ==========

        # 将 embedding_dim 投影到 hidden_dim

        self.input_projection = nn.Linear(self.embedding_dim, self.hidden_dim)

        # ========== TCN + GraphConv 层 ==========

        self.tcn_blocks = nn.ModuleList()

        self.graph_convs = nn.ModuleList()

        for i in range(self.num_layers):

            # TCN Block（第一层输入可能不同）

            in_channels = self.hidden_dim

            out_channels = self.hidden_dim

            dilation = self.dilations[i]

            self.tcn_blocks.append(

                TemporalBlock(

                    in_channels=in_channels,

                    out_channels=out_channels,

                    kernel_size=self.kernel_size,

                    dilation=dilation,

                    dropout=self.dropout

                )

            )

            # GraphConv Layer

            self.graph_convs.append(

                LightGraphConv(in_channels=out_channels)

            )

        # ========== 输出层 ==========

        # 预测单个特征值

        self.output_layer = nn.Linear(self.hidden_dim, 1)


    def forward(self, global_embedding, adj):
        """

        前向传播

        Args:

        global_embedding: (B, N, embedding_dim) - 全局embedding

        adj: (N, N) - 归一化的邻接矩阵

        Returns:

        prediction: (B, 1, N, T_out) - 全局预测

        """

        batch_size = global_embedding.size(0)

        num_nodes = global_embedding.size(1)

        # ========== 步骤1: 输入投影 ==========

        # (B, N, embedding_dim) -> (B, N, hidden_dim)

        x = self.input_projection(global_embedding)

        # ========== 步骤2: 时间维度扩展 ==========

        # (B, N, hidden_dim) -> (B, N, hidden_dim, T_out)

        x = x.unsqueeze(-1).repeat(1, 1, 1, self.seq_out_len)

        # x: (B, N, hidden_dim, T_out)

        # ========== 步骤3: Reshape for TCN ==========

        # (B, N, hidden_dim, T_out) -> (B*N, hidden_dim, T_out)

        x = x.reshape(batch_size * num_nodes, self.hidden_dim, self.seq_out_len)

        # ========== 步骤4: 逐层 TCN + GraphConv ==========

        for tcn_block, graph_conv in zip(self.tcn_blocks, self.graph_convs):

            # TCN Block

            # (B*N, C, T) -> (B*N, C, T)

            x_tcn = tcn_block(x)

            # Reshape for GraphConv

            # (B*N, C, T) -> (B, N, C, T)

            x_tcn = x_tcn.reshape(batch_size, num_nodes, self.hidden_dim, self.seq_out_len)

            # GraphConv

            # (B, N, C, T) + (N, N) -> (B, N, C, T)

            x_graph = graph_conv(x_tcn, adj)

            # Reshape back for next TCN

            # (B, N, C, T) -> (B*N, C, T)

            x = x_graph.reshape(batch_size * num_nodes, self.hidden_dim, self.seq_out_len)

        # ========== 步骤5: Reshape 回节点维度 ==========

        # (B*N, C, T) -> (B, N, C, T)

        x = x.reshape(batch_size, num_nodes, self.hidden_dim, self.seq_out_len)

        # ========== 步骤6: 输出层 ==========

        # (B, N, C, T) -> (B, N, T, C)

        x = x.permute(0, 1, 3, 2)

        # (B, N, T, C) -> (B, N, T, 1)

        prediction = self.output_layer(x)

        # ========== 步骤7: 转换到标准格式 ==========

        # (B, N, T, 1) -> (B, 1, N, T)

        prediction = prediction.permute(0, 3, 1, 2)

        return prediction


# ============================================================

# 预测解码器（支持 LSTM 和 TCN）

# ============================================================


class PredictionDecoder(nn.Module):
    """

    预测解码器 - 全局预测版本（支持 LSTM 和 TCN）

    架构：

    - 输入：全局embedding (B, N_all, embedding_dim)

    - 输出：全局预测 (B, 1, N_all, T_out)

    支持两种解码器：

    1. LSTM 解码器（decoder_type='lstm'）

    2. Graph-aware TCN 解码器（decoder_type='tcn'）

    LSTM流程：

    全局embedding (B, N_all, dim)

    → 时间扩展 (B, N_all, T_out, dim)

    → LSTM解码 (B*N_all, T_out, decoder_units)

    → 输出层 (B*N_all, T_out, 1)

    → Reshape → (B, 1, N_all, T_out)

    TCN流程：

    全局embedding (B, N_all, dim)

    → GraphAwareTCN (带邻接矩阵)

    → (B, 1, N_all, T_out)

    """


    def __init__(self, params):
        """

        初始化预测解码器

        Args:

        params: 配置字典，需要包含:

        - seq_out_len: 预测序列长度

        - num_nodes: 完整集节点数

        - encoder_lstm_units: 编码器LSTM单元数（用于计算embedding_dim）

        - conv_filters: 卷积滤波器数（用于计算embedding_dim）

        - decoder_type: 解码器类型 'lstm' 或 'tcn'（默认'lstm'）

        LSTM 参数：

        - decoder_units: 解码器LSTM单元数（默认128）

        - dropout: dropout率（默认0.3）

        - use_dropout: 是否使用dropout（默认True）

        TCN 参数：

        - tcn_kernel_size: TCN卷积核大小（默认3）

        - tcn_num_layers: TCN层数（默认4）

        - tcn_dilations: 扩张率列表（默认[1,2,4,8]）

        - tcn_hidden_dim: TCN隐藏维度（默认128）

        - tcn_dropout: TCN dropout率（默认0.2）

        - tcn_residual: 是否使用残差连接（默认True）

        """

        super(PredictionDecoder, self).__init__()

        # 基本参数

        self.seq_out_len = params['seq_out_len']

        self.num_nodes = params['num_nodes']

        self.decoder_type = params.get('decoder_type', 'lstm')

        # embedding 维度（与encoder保持一致）

        self.embedding_dim = params.get('encoder_lstm_units', 128) + params.get('conv_filters', 64)

        # ========== 根据 decoder_type 初始化不同的解码器 ==========

        if self.decoder_type == 'tcn':

            # 使用 Graph-aware TCN 解码器

            self.decoder = GraphAwareTCN(params)

            print("✓ Using Graph-aware TCN Decoder")

        else:

            # 使用 LSTM 解码器（默认）

            self.decoder_units = params.get('decoder_units', 128)

            self.dropout = params.get('dropout', 0.3)

            # LSTM解码器

            self.lstm = nn.LSTM(

                input_size=self.embedding_dim,

                hidden_size=self.decoder_units,

                batch_first=True,

                dropout=self.dropout if params.get('use_dropout', True) else 0.0

            )

            # 输出层

            self.output_layer = nn.Linear(self.decoder_units, 1)

            print("✓ Using LSTM Decoder")


    def forward(self, global_embedding, adj=None):
        """

        前向传播

        Args:

        global_embedding: (B, N_all, embedding_dim) - 全局embedding

        adj: (N, N) - 归一化的邻接矩阵（TCN模式需要，LSTM模式可选）

        Returns:

        prediction: (B, 1, N_all, T_out) - 全局预测

        """

        if self.decoder_type == 'tcn':

        # ========== TCN 解码 ==========

            if adj is None:

                raise ValueError("TCN decoder requires adjacency matrix (adj)")

            prediction = self.decoder(global_embedding, adj)

            return prediction

        else:

            # ========== LSTM 解码 ==========

            batch_size = global_embedding.size(0)

            # 步骤1: 时间维度扩展

            # (B, N_all, dim) -> (B, N_all, T_out, dim)

            pred_input = global_embedding.unsqueeze(2).repeat(1, 1, self.seq_out_len, 1)

            # 步骤2: Reshape for LSTM

            # (B, N_all, T_out, dim) -> (B*N_all, T_out, dim)

            pred_input = pred_input.reshape(batch_size * self.num_nodes, self.seq_out_len, -1)

            # 步骤3: LSTM 解码

            # (B*N_all, T_out, dim) -> (B*N_all, T_out, decoder_units)

            lstm_out, _ = self.lstm(pred_input)

            # 步骤4: 输出层

            # (B*N_all, T_out, decoder_units) -> (B*N_all, T_out, 1)

            prediction = self.output_layer(lstm_out)

            # 步骤5: Reshape 回原始格式

            # (B*N_all, T_out, 1) -> (B, N_all, T_out, 1)

            prediction = prediction.reshape(batch_size, self.num_nodes, self.seq_out_len, 1)

            # 步骤6: 转换到标准格式

            # (B, N_all, T_out, 1) -> (B, 1, N_all, T_out)

            prediction = prediction.permute(0, 3, 1, 2)

            return prediction
