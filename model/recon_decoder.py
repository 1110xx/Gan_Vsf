import torch
import torch.nn as nn

# 导入图卷积层
try:
    from .encoder import GraphConvLayer
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os

    sys.path.append(os.path.dirname(__file__))
    from encoder import GraphConvLayer


class ReconstructionDecoder(nn.Module):
    """
    重构解码器：从全局embedding解码到时间序列（图增强版）
    输入：全局embedding (B, N_all, embedding_dim) + 邻接矩阵 (N, N)
    输出：重构的完整集数据 (B, F, N_all, T)
    流程：
    全局embedding -> 图卷积(增强空间一致性) -> 时序恢复(Conv1d) -> 重构数据
    """


    def __init__(self, params):
        super(ReconstructionDecoder, self).__init__()
        self.num_nodes = params['num_nodes']  # 完整集节点数
        self.in_dim = params['in_dim']  # 输入特征维度
        self.seq_in_len = params['seq_in_len']
        self.embedding_dim = params.get('encoder_lstm_units', 128) + params.get('conv_filters', 64)
        self.hidden_dim = params.get('recon_hidden_dim', 128)
        self.dropout = params.get('recon_dropout', 0.3)
        # 图传播层数
        self.num_graph_layers = params.get('num_recon_graph_layers', 2)
        # ========== 图卷积层：恢复空间一致性 ==========
        self.gcn_layers = nn.ModuleList()
        in_dim = self.embedding_dim
        for i in range(self.num_graph_layers):
            if i < self.num_graph_layers - 1:
                out_dim = self.hidden_dim
        else:
            out_dim = self.embedding_dim  # 最后一层恢复到embedding_dim
        self.gcn_layers.append(
            GraphConvLayer(in_dim, out_dim, dropout=self.dropout)
        )
        in_dim = out_dim
        # ========== 时序恢复模块：1D Conv ==========
        # 将节点embedding映射到时间序列
        self.temporal_decoder = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim, self.in_dim, kernel_size=3, padding=1)
        )


    def forward(self, global_embedding, adj):
        """
        前向传播：从全局embedding解码到时间序列（图增强版）
        Args:
        global_embedding: (B, N_all, embedding_dim) - 全局embedding
        adj: (N_all, N_all) - 邻接矩阵
        Returns:
        reconstruction: (B, F, N_all, T) - 重构的完整集数据
        """
        batch_size = global_embedding.size(0)
        # ========== 步骤1: 图卷积增强空间一致性 ==========
        h = global_embedding
        for layer in self.gcn_layers:
            h = layer(h, adj)  # (B, N_all, embedding_dim)
        # ========== 步骤2: 时序恢复 ==========
        # 将节点特征broadcast到时间维度
        # (B, N_all, embedding_dim) -> (B*N_all, embedding_dim, 1)
        B, N, C = h.shape
        h_time = h.reshape(B * N, C, 1)
        # Repeat到目标时间长度
        # (B*N_all, embedding_dim, 1) -> (B*N_all, embedding_dim, T)
        h_time = h_time.repeat(1, 1, self.seq_in_len)
        # 1D Conv解码到特征维度
        # (B*N_all, embedding_dim, T) -> (B*N_all, F, T)
        recon_seq = self.temporal_decoder(h_time)
        # ========== 步骤3: Reshape到标准格式 ==========
        # (B*N_all, F, T) -> (B, N_all, F, T)
        recon_seq = recon_seq.reshape(batch_size, self.num_nodes, self.in_dim, self.seq_in_len)
        # (B, N_all, F, T) -> (B, F, N_all, T)
        reconstruction = recon_seq.permute(0, 2, 1, 3)
        return reconstruction
