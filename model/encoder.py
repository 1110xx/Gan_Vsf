import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def gumbel_softmax_topk(logits, k, dim=-1, tau=1.0, hard=False):
    """
    可微的Top-k选择（基于Gumbel-Softmax）

    Args:
        logits: (*, N) 输入logits
        k: 选择的top-k数量
        dim: 操作维度
        tau: 温度参数
        hard: 是否使用硬选择（straight-through estimator）

    Returns:
        mask: (*, N) 软mask，top-k位置接近1，其他接近0
    """
    # Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    perturbed_logits = (logits + gumbel_noise) / tau

    # 获取top-k的阈值
    topk_values, _ = torch.topk(perturbed_logits, k, dim=dim)
    threshold = topk_values[..., -1:].expand_as(logits)

    # 软mask
    soft_mask = torch.sigmoid((perturbed_logits - threshold) * 10.0)  # 陡峭的sigmoid

    if hard:
        # Straight-through estimator
        hard_mask = (perturbed_logits >= threshold).float()
        mask = hard_mask - soft_mask.detach() + soft_mask
    else:
        mask = soft_mask

    return mask


def hard_topk_mask(adj, k, dim=-1):
    """
    硬Top-k mask（用于推理阶段）
    """
    values, indices = torch.topk(adj, k, dim=dim)
    mask = torch.zeros_like(adj)
    mask.scatter_(dim, indices, 1.0)
    return mask


# ============================================================
# 原始 nconv 模块（保持兼容性）
# ============================================================

class nconv(nn.Module):
    """
    原始的图卷积操作（保留用于兼容性）
    """
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        """
        Args:
            x: (B, C, N, T)
            A: (N, N) or (N, V)
        Returns:
            out: (B, C, V, T)
        """
        x = torch.einsum('bcnt,nv->bcvt', (x, A))
        return x.contiguous()


# ============================================================
# 改进的图学习模块
# ============================================================

class ImprovedGraphConstructor(nn.Module):
    """
    改进的自适应图学习模块

    主要改进：
    1. 移除 relu(tanh(...)) → 使用 softplus 保留负梯度
    2. Top-k 改为 Gumbel-Softmax（训练时可微）
    3. 邻接矩阵对称化：A = (A + A^T) / 2
    4. 可学习的温度参数
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None,
                 use_gumbel=True, temperature=1.0):
        super(ImprovedGraphConstructor, self).__init__()
        self.nnodes = nnodes
        self.k = k
        self.dim = dim
        self.device = device
        self.alpha = alpha
        self.static_feat = static_feat
        self.use_gumbel = use_gumbel

        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(temperature))

        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

    def forward(self, idx=None, training=True):
        """
        Args:
            idx: 节点索引 (可选)
            training: 是否训练模式
        Returns:
            adj: (N, N) 邻接矩阵
        """
        if idx is None:
            idx = torch.arange(self.nnodes).to(self.device)

        # 获取节点embedding
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        # 线性变换 + tanh激活
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        # 计算相似度矩阵
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))

        # 对称化
        adj = (a + a.T) / 2

        # 使用 softplus 替代 relu(tanh(...))，保留负梯度
        adj = F.softplus(adj) - 0.5  # 中心化

        # Top-k 稀疏化
        if self.use_gumbel and training:
            # Gumbel-Softmax 软稀疏化（可微）
            mask = gumbel_softmax_topk(adj, self.k, dim=1, tau=self.temperature, hard=False)
        else:
            # 硬稀疏化（推理阶段）
            mask = hard_topk_mask(adj, self.k, dim=1)

        # 应用mask，但保留原始值（不是全设为1）
        adj = adj * mask

        # 确保非负
        adj = F.relu(adj)

        return adj


class graph_constructor(nn.Module):
    """
    原始graph_constructor的包装器（向后兼容）
    内部使用ImprovedGraphConstructor
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.improved = ImprovedGraphConstructor(
            nnodes, k, dim, device, alpha, static_feat
        )
        self.nnodes = nnodes
        self.k = k
        self.dim = dim
        self.device = device
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx=None):
        training = self.training
        return self.improved(idx, training)

# ============================================================
# 改进的图卷积层
# ============================================================

class ImprovedGraphConvLayer(nn.Module):
    """
    改进的图卷积层

    主要改进：
    1. 标准对称归一化：D^(-1/2) A D^(-1/2)
    2. Highway Network 替代低alpha残差
    3. 移除mask覆盖机制
    4. 简化维度变换
    """
    def __init__(self, in_dim, out_dim, dropout=0.3, residual=True, use_highway=True):
        super(ImprovedGraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual
        self.use_highway = use_highway

        self.mlp = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        # Highway gate
        if residual and use_highway and in_dim == out_dim:
            self.gate = nn.Sequential(
                nn.Linear(in_dim + out_dim, out_dim),
                nn.Sigmoid()
            )
        else:
            self.gate = None

    def normalize_adj(self, adj):
        """
        对称归一化：D^(-1/2) A D^(-1/2)
        """
        # 添加自环
        adj = adj + torch.eye(adj.size(0), device=adj.device)

        # 计算度矩阵的逆平方根
        d = adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

        # 对称归一化
        norm_adj = adj * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)

        return norm_adj

    def forward(self, x, adj):
        """
        Args:
            x: (B, N, C) 节点特征
            adj: (N, N) 邻接矩阵
        Returns:
            out: (B, N, C') 输出特征
        """
        # 归一化邻接矩阵
        norm_adj = self.normalize_adj(adj)

        # 图卷积：(B, N, C) @ (N, N) → (B, N, C)
        h = torch.einsum('bnc,nm->bmc', x, norm_adj)

        # MLP变换
        h = self.mlp(h)
        h = self.dropout(h)

        # Highway residual
        if self.gate is not None:
            gate_input = torch.cat([x, h], dim=-1)
            gate = self.gate(gate_input)
            out = gate * h + (1 - gate) * x
        elif self.residual and self.in_dim == self.out_dim:
            out = 0.5 * x + 0.5 * h  # 简单残差
        else:
            out = h

        return out


class GraphConvLayer(nn.Module):
    """
    原始GraphConvLayer的包装器（向后兼容）
    内部使用ImprovedGraphConvLayer
    """
    def __init__(self, in_dim, out_dim, dropout=0.3, alpha=0.05):
        super(GraphConvLayer, self).__init__()
        self.improved = ImprovedGraphConvLayer(in_dim, out_dim, dropout, residual=True)
        self.alpha = alpha  # 保留以兼容

    def forward(self, x, adj, mask=None):
        # 忽略mask参数，使用改进版本
        return self.improved(x, adj)

# ============================================================
# SubsetEncoder（保持不变）
# ============================================================

class SubsetEncoder(nn.Module):
    """
    子集编码器（保持原有实现）
    """
    def __init__(self, params):
        super(SubsetEncoder, self).__init__()
        self.seq_in_len = params['seq_in_len']
        self.in_dim = params['in_dim']

        # Encoder params
        self.encoder_lstm_units = params.get('encoder_lstm_units', 128)
        self.conv_filters = params.get('conv_filters', 64)
        self.kernel_size = params.get('kernel_size', 3)
        self.dropout = params.get('dropout', 0.3)
        self.use_dropout = params.get('use_dropout', True)
        self.use_batchnorm = params.get('use_batchnorm', True)

        # LSTM encoder
        self.encoder_lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=self.encoder_lstm_units,
            batch_first=True,
            dropout=self.dropout if self.use_dropout else 0.0
        )

        # Conv encoder
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=self.in_dim,
                out_channels=self.conv_filters,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            )
        )
        for _ in range(3):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=self.conv_filters,
                    out_channels=self.conv_filters,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2
                )
            )

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.conv_filters) for _ in range(4)
        ]) if self.use_batchnorm else None

        # Embedding 维度
        self.embedding_dim = self.encoder_lstm_units + self.conv_filters

    def forward(self, x):
        """
        Args:
            x: (B, F, N, T)
        Returns:
            embedding: (B, N, embedding_dim)
        """
        batch_size = x.size(0)
        num_nodes_subset = x.size(2)

        # (B, F, N, T) → (B, N, F, T)
        x = x.permute(0, 2, 1, 3)

        # LSTM encoder
        x_lstm = x.permute(0, 1, 3, 2)  # (B, N, T, F)
        x_lstm = x_lstm.reshape(batch_size * num_nodes_subset, self.seq_in_len, self.in_dim)
        lstm_out, _ = self.encoder_lstm(x_lstm)
        lstm_features = lstm_out[:, -1, :]  # (B*N, hidden)

        # Conv encoder
        x_conv = x.reshape(batch_size * num_nodes_subset, self.in_dim, self.seq_in_len)
        conv_out = x_conv
        for i, conv_layer in enumerate(self.conv_layers):
            conv_out = conv_layer(conv_out)
            if self.batch_norms is not None:
                conv_out = self.batch_norms[i](conv_out)
            conv_out = torch.relu(conv_out)
        conv_features = conv_out[:, :, -1]  # (B*N, filters)

        # 组合embedding
        embedding = torch.cat([lstm_features, conv_features], dim=-1)
        embedding = embedding.reshape(batch_size, num_nodes_subset, -1)

        return embedding

# ============================================================
# 改进的 Embedding Expander
# ============================================================

class ImprovedEmbeddingExpander(nn.Module):
    """
    改进的 Embedding 扩展模块

    主要改进：
    1. 未观测节点使用可学习初始化（而非全0）
    2. Learnable Diffusion Gate 替代 hard mask
    3. 移除最后的强制还原
    4. Confidence-based 信息融合
    """
    def __init__(self, params):
        super(ImprovedEmbeddingExpander, self).__init__()
        self.num_nodes = params['num_nodes']
        self.embedding_dim = params.get('encoder_lstm_units', 128) + params.get('conv_filters', 64)
        self.device = params.get('device', 'cuda')

        # 图传播参数
        self.num_graph_layers = params.get('num_recon_graph_layers', 3)
        self.graph_dropout = params.get('recon_dropout', 0.3)
        self.use_adaptive_graph = params.get('use_adaptive_graph', True)

        # 预定义邻接矩阵
        self.predefined_adj = params.get('predefined_A', None)

        # 优化：预先归一化
        self.predefined_adj_normalized = None
        if self.predefined_adj is not None:
            adj_with_self = self.predefined_adj + torch.eye(self.num_nodes).to(self.device)
            d = adj_with_self.sum(1)
            d_inv_sqrt = torch.pow(d, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            self.predefined_adj_normalized = adj_with_self * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)

        # 自适应图学习
        if self.use_adaptive_graph:
            self.graph_constructor = graph_constructor(
                nnodes=self.num_nodes,
                k=params.get('graph_k', 10),
                dim=params.get('node_dim', 40),
                device=self.device,
                alpha=3
            )

        # 未观测节点的可学习初始化
        self.unobserved_init = nn.Parameter(
            torch.randn(1, 1, self.embedding_dim) * 0.01
        )

        # 多层图卷积
        self.graph_layers = nn.ModuleList()
        hidden_dim = params.get('recon_hidden_dim', self.embedding_dim)

        for i in range(self.num_graph_layers):
            in_dim = self.embedding_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < self.num_graph_layers - 1 else self.embedding_dim
            self.graph_layers.append(
                ImprovedGraphConvLayer(
                    in_dim,
                    out_dim,
                    dropout=self.graph_dropout,
                    residual=True,
                    use_highway=True
                )
            )

        # Learnable Diffusion Gates
        self.diffusion_gates = nn.ModuleList()
        for i in range(self.num_graph_layers):
            in_dim = self.embedding_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < self.num_graph_layers - 1 else self.embedding_dim
            self.diffusion_gates.append(
                nn.Sequential(
                    nn.Linear(in_dim + out_dim, out_dim),
                    nn.Sigmoid()
                )
            )

        # Confidence传播参数
        self.confidence_alpha = nn.Parameter(torch.tensor(0.3))

    def preprocess_embedding(self, subset_embedding, idx_subset, device):
        """
        初始化全局embedding（改进版）

        Args:
            subset_embedding: (B, N_subset, D)
            idx_subset: (B, N_subset) or (N_subset,)
        Returns:
            full_embedding: (B, N_all, D)
            confidence: (B, N_all, 1) - 置信度
        """
        batch_size = subset_embedding.size(0)
        embedding_dim = subset_embedding.size(2)

        # 初始化：未观测节点使用可学习embedding
        full_embedding = self.unobserved_init.expand(batch_size, self.num_nodes, -1).clone()

        # 确保 idx_subset 是 Tensor
        if isinstance(idx_subset, np.ndarray):
            idx_subset = torch.from_numpy(idx_subset).to(device)
        elif isinstance(idx_subset, list):
            idx_subset = torch.tensor(idx_subset, device=device)

        # 填充子集位置
        if idx_subset.dim() == 1:
            # 单个batch：(N_subset,)
            full_embedding[:, idx_subset, :] = subset_embedding
        else:
            # 多batch：(B, N_subset)
            for b in range(batch_size):
                full_embedding[b, idx_subset[b], :] = subset_embedding[b]

        # 创建confidence mask
        confidence = torch.zeros(batch_size, self.num_nodes, 1, device=device)
        if idx_subset.dim() == 1:
            confidence[:, idx_subset, :] = 1.0
        else:
            for b in range(batch_size):
                confidence[b, idx_subset[b], :] = 1.0

        return full_embedding, confidence

    def get_adjacency_matrix(self, idx=None):
        """
        获取邻接矩阵（保持向后兼容）
        """
        if self.predefined_adj is not None:
            return self.predefined_adj
        elif self.use_adaptive_graph:
            return self.graph_constructor(idx)
        else:
            return torch.eye(self.num_nodes).to(self.device)

    def forward(self, subset_embedding, idx_subset, args=None):
        """
        Args:
            subset_embedding: (B, N_subset, D)
            idx_subset: (N_subset,) 或 (B, N_subset)
            args: 额外参数
        Returns:
            global_embedding: (B, N_all, D)
        """
        device = subset_embedding.device
        batch_size = subset_embedding.size(0)

        # 初始化
        h, confidence = self.preprocess_embedding(subset_embedding, idx_subset, device)

        # 获取邻接矩阵
        adj = self.get_adjacency_matrix()

        # 归一化邻接矩阵（用于confidence传播）
        if self.predefined_adj_normalized is not None:
            adj_norm = self.predefined_adj_normalized
        else:
            adj_with_self = adj + torch.eye(adj.size(0), device=device)
            d = adj_with_self.sum(1)
            d_inv_sqrt = torch.pow(d, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            adj_norm = adj_with_self * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)

        # 多层GCN传播
        for i, (gcn_layer, gate_layer) in enumerate(zip(self.graph_layers, self.diffusion_gates)):
            h_prev = h

            # GCN传播（不mask输入，让梯度流通）
            h_gcn = gcn_layer(h, adj)

            # Learnable gate：决定接受多少新信息
            gate_input = torch.cat([h_prev, h_gcn], dim=-1)
            gate = gate_layer(gate_input)

            # Confidence-based融合
            # 观测节点：保留原始信息为主，GCN为辅
            # 未观测节点：GCN信息为主
            if h_prev.size(-1) == h_gcn.size(-1):
                # 维度相同：gate控制h_prev和h_gcn的融合比例
                h_intermediate = gate * h_prev + (1 - gate) * h_gcn
            else:
                # 维度不同：直接使用h_gcn（维度已通过GCN层转换）
                h_intermediate = gate * h_gcn

                # 根据confidence决定观测节点和未观测节点的策略
                # 观测节点：使用gate控制的融合结果
                # 未观测节点：直接使用GCN输出
            h = confidence * h_intermediate + (1 - confidence) * h_gcn

            # 传播confidence（邻居传播）
            if i < len(self.graph_layers) - 1:
                # (B, N, 1) @ (N, N) → (B, N, 1)
                neighbor_confidence = torch.bmm(
                    confidence.transpose(1, 2),  # (B, 1, N)
                    adj_norm.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, N)
                ).transpose(1, 2)  # (B, N, 1)

                # 更新confidence
                alpha = torch.sigmoid(self.confidence_alpha)
                confidence = torch.clamp(confidence + alpha * neighbor_confidence, 0.0, 1.0)

        return h

class EmbeddingExpander(nn.Module):
    """
    原始EmbeddingExpander的包装器（向后兼容）
    """

    def __init__(self, params):
        super(EmbeddingExpander, self).__init__()
        self.improved = ImprovedEmbeddingExpander(params)
        self.num_nodes = params['num_nodes']
        self.embedding_dim = params.get('encoder_lstm_units', 128) + params.get('conv_filters', 64)
        self.device = params.get('device', 'cuda')
        self.use_adaptive_graph = params.get('use_adaptive_graph', True)
        self.predefined_adj = params.get('predefined_A', None)

        # 兼容性：暴露内部模块
        if self.use_adaptive_graph:
            self.graph_constructor = self.improved.graph_constructor

    def get_adjacency_matrix(self, idx=None):
        return self.improved.get_adjacency_matrix(idx)

    def forward(self, subset_embedding, idx_subset, args=None):
        return self.improved(subset_embedding, idx_subset, args)

    # ============================================================
    # Global Embedding Encoder（保持不变）
    # ============================================================

class GlobalEmbeddingEncoder(nn.Module):
    """
    全局Embedding编码器
    """

    def __init__(self, params):
        super(GlobalEmbeddingEncoder, self).__init__()
        self.subset_encoder = SubsetEncoder(params)
        # 检查 是否使用 imputer
        self.use_imputer = params.get('use_embedding_imputer', False)

        if self.use_imputer:
            # 使用 GRUI Imputer
            try:
                from .embedding_imputer import EmbeddingImpacter
                self.impacter = EmbeddingImpacter(params)
                print("使用 GRUImpacter 进行缺失值填充")
            except ImportError:
                raise ImportError("GRUImpacter 模块未找到，请检查是否已安装")
                self.use_imputer = False
                self.embedding_expander = EmbeddingExpander(params)
        else:
            self.embedding_expander = EmbeddingExpander(params)
            print("使用 EmbeddingExpander 进行缺失值填充")

        
        self.embedding_dim = self.subset_encoder.embedding_dim
            

    def forward(self, x, idx_subset, args=None):
        """
        Args:
            x: (B, F, N_subset, T)
            idx_subset: (N_subset,)
        Returns:
            global_embedding: (B, N_all, embedding_dim)
        """
        subset_embedding = self.subset_encoder(x)
        if self.use_imputer:
            global_embedding = self.impacter(subset_embedding, idx_subset, args)
        else:
            global_embedding = self.embedding_expander(subset_embedding, idx_subset, args)

        return global_embedding