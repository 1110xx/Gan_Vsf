import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self, x, A):
        x = torch.einsum('bcnt,nv->bcvt', (x, A))
        return x.contiguous()

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None ):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)
        self.k = k
        self.dim = dim
        self.device = device
        self.alpha = alpha
        self.static_feat = static_feat
    def forward(self, idx=None):
        if idx is None:
            idx = torch.arange(self.nnodes).to(self.device)
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1,0)) - \
            torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha * a))

        #top-k 稀疏化
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3, alpha=0.05):
        super(GraphConvLayer, self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
    def forward(self, x, adj, mask=None):
        #归一化邻接矩阵
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        a = adj / d.view(-1,1)

        #(B,N,C)->(B,C,N,1)
        x_gcn = x.unsqueeze(-1).permute(0,2,1,3)

        #图卷积：(B,C,N,1)@(N,N)->(B,C,N,1)
        h = self.nconv(x_gcn, a)

        #转回：(B,C,N,1)->(B,N,C)
        h = h.squeeze(-1).permute(0,2,1)

        #残差连接
        h = self.alpha * x + (1 - self.alpha) * h

        #MLP变换
        out = self.mlp(h)
        out = self.dropout(out)

        if mask is not None:
            out = out * mask + x * (1 - mask)

        return out
    
class SubsetEncoder(nn.Module):
    def __init__(self, params):
        super(SubsetEncoder, self).__init__()
        self.seq_in_len = params['seq_in_len']
        self.in_dim = params['in_dim']

        #Encoder params
        self.encoder_lstm_units = params.get('encoder_lstm_units', 128)
        self.conv_filters = params.get('conv_filters', 64)
        self.kernel_size = params.get('kernel_size', 3)
        self.dropout = params.get('dropout', 0.3) 
        self.use_dropout = params.get('use_dropout', True)
        self.use_batchnorm = params.get('use_batchnorm', True)

        #LSTM encoder 逐节点
        self.encoder_lstm = nn.LSTM(
            input_size = self.in_dim,
            hidden_size = self.encoder_lstm_units,
            batch_first = True,
            dropout = self.dropout if self.use_dropout else 0.0
        )

        #Conv encoder 逐节点
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels = self.in_dim,
                out_channels = self.conv_filters,
                kernel_size = self.kernel_size,
                padding = self.kernel_size // 2
            )
        )
        for _ in range(3):    
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels = self.conv_filters,
                    out_channels = self.conv_filters,
                    kernel_size = self.kernel_size,
                    padding = self.kernel_size // 2
                )
            )

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.conv_filters) for _ in range(4)
        ]) if self.use_batchnorm else None

        #embedding 维度 = LSTM隐藏层 + Conv filters
        self.embedding_dim = self.encoder_lstm_units + self.conv_filters
       
    def forward(self, x):
        batch_size = x.size(0)
        num_nodes_subset = x.size(2)

        #(B, F, N, T)->(B, N, F, T)
        x = x.permute(0, 2, 1, 3)

        #LSTM encoder
        x_lstm = x.permute(0, 1, 3, 2)
        x_lstm = x_lstm.reshape(batch_size * num_nodes_subset, self.seq_in_len, self.in_dim)
        lstm_out, _ = self.encoder_lstm(x_lstm)
        lstm_features = lstm_out[:, -1, :]

        #Conv encoder
        x_conv = x.reshape(batch_size * num_nodes_subset, self.in_dim, self.seq_in_len)
        conv_out = x_conv
        for i, conv_layer in enumerate(self.conv_layers):
            conv_out = conv_layer(conv_out)
            if self.batch_norms is not None:
                conv_out = self.batch_norms[i](conv_out)
            conv_out = torch.relu(conv_out)
        conv_features = conv_out[:, :, -1]

        #组合embedding
        embedding = torch.cat([lstm_features, conv_features], dim=-1)
        embedding = embedding.reshape(batch_size, num_nodes_subset, -1)
        return embedding
        
class EmbeddingExpander(nn.Module):
    """
    input:
        embedding: (B, N_subset, embedding_dim) + idx_subset
    output:
        embedding: (B, N_all, embedding_dim)
    """
    def __init__(self, params):
        super(EmbeddingExpander, self).__init__()
        self.num_nodes = params['num_nodes']
        self.embedding_dim = params.get('encoder_lstm_units', 128) + params.get('conv_filters', 64)
        self.device = params.get('device', 'cuda')

        # 图传播
        self.num_graph_layers = params.get('num_recon_graph_layers', 3)
        self.graph_dropout = params.get('recon_dropout', 0.3)
        self.use_adaptive_graph = params.get('use_adaptive_graph', True)

        # 逐层扩散控制参数
        self.diffusion_type = params.get('diffusion_type', 'soft')
        self.diffusion_alpha = params.get('diffusion_alpha', 0.3)
        self.diffusion_decay = params.get('diffusion_decay', 0.8)
        self.initial_threshold = params.get('diffusion_threshold', 0.5)

        # 预定义邻接矩阵
        self.predefined_adj = params.get('predefined_A', None)

        # 优化：预定义邻接矩阵预先归一化
        self.predefined_adj_normalized = None
        if self.predefined_adj is not None:
            adj_with_self = self.predefined_adj + torch.eye(self.num_nodes).to(self.device)
            d = adj_with_self.sum(1)
            self.predefined_adj_normalized = adj_with_self / d.view(-1, 1)

        # 自适应图学习
        if self.use_adaptive_graph:
           self.graph_constructor = graph_constructor(
                nnodes=self.num_nodes,
                k=params.get('graph_k', 10),
                dim=params.get('node_dim', 40),
                device=self.device,
                alpha=3
            )
        
        # 多层图卷积
        self.graph_layers = nn.ModuleList()
        in_dim = self.embedding_dim
        hidden_dim = params.get('recon_hidden_dim', self.embedding_dim)

        for i in range(self.num_graph_layers):
            out_dim = hidden_dim if i < self.num_graph_layers - 1 else self.embedding_dim
            self.graph_layers.append(
                GraphConvLayer(
                    in_dim,
                    out_dim,
                    dropout=self.graph_dropout,
                   )
            )
            in_dim = out_dim

    def preprocess_emdedding(self, subset_embedding, idx_subset, device):
        """
        Args:
            subset_embedding: (B, N_subset, embedding_dim)
            idx_subset: (B, N_subset)
        Returns:
            full_embedding: (B, N_all, embedding_dim)
            mask: (B, N_all, 1) - 子集位置1，其他0
        """
        batch_size =subset_embedding.size(0)
        embedding_dim = subset_embedding.size(2)
        full_embedding = torch.zeros(
            batch_size, self.num_nodes, embedding_dim, 
            device=device, dtype=subset_embedding.dtype
        )
        
        # 确保 idx_subset 是 Tensor
        if isinstance(idx_subset, np.ndarray):
            idx_subset = torch.from_numpy(idx_subset).to(device)
        elif isinstance(idx_subset, list):
            idx_subset = torch.tensor(idx_subset, device=device)
        
        # 填充子集位置
        full_embedding[:, idx_subset, :] = subset_embedding

        # 创建 mask
        mask = torch.zeros(
            batch_size, self.num_nodes, 1, 
             device=device, dtype=subset_embedding.dtype
        )
        mask[:, idx_subset, :] = 1
        return full_embedding, mask
    
    def get_adjacency_matrix(self, idx=None):
        """
        获取邻接矩阵
        优先级：预定义 > 自适应图学习 > 单位矩阵
        """
        if self.predefined_adj is not None:
            return self.predefined_adj
        elif self.use_adaptive_graph:
            return self.graph_constructor(idx)
        else:
            return torch.eye(self.num_nodes).to(self.device)

    def normalize_adjacency(self, adj):
        """
        归一化邻接矩阵
        """
        adj_with_self = adj + torch.eye(adj.size(0)).to(adj.device)
        d = adj_with_self.sum(1)
        adj_normalized = adj_with_self / d.view(-1, 1)
        return adj_normalized
    
    def forward(self, subset_embedding, idx_subset, args=None):
        """
        Args:
            subset_embedding: (B, N_subset, embedding_dim)
            idx_subset: (B, N_subset)
            args: 额外参数
        Returns:
            global_embedding: (B, N_all, embedding_dim)
        """
        device = subset_embedding.device
        full_embedding, mask = self.preprocess_emdedding(
            subset_embedding, idx_subset, device)
        adj = self.get_adjacency_matrix()
        h = full_embedding
        active_mask = mask.clone()

        if self.predefined_adj_normalized is not None:
            adj_normalized = self.predefined_adj_normalized
        else:
            adj_normalized = self.normalize_adjacency(adj)

        for i, layer in enumerate(self.graph_layers):
            h_masked = h * active_mask

            active_mask_for_adj = active_mask[0].squeeze(-1) #(N,)
            adj_directed = adj * active_mask_for_adj.unsqueeze(0) #(N, N) * (1, N) -> (N, N)

            h = layer(h_masked, adj_directed, mask= None)

            if i < len(self.graph_layers) - 1:
                neighbor_scores = torch.matmul(
                    active_mask.squeeze(-1).float(),
                    adj_normalized
                )
                if self.diffusion_type == 'soft':
                    alpha_layer = self.diffusion_alpha * (self.diffusion_decay ** i)
                    active_mask = torch.clamp(
                        active_mask + alpha_layer * neighbor_scores.unsqueeze(-1),
                        0.0, 1.0
                    )
                else:
                    threshold = self.diffusion_threshold * (self.diffusion_decay ** i)
                    new_active = (neighbor_scores > threshold).float().unsqueeze(-1)
                    active_mask = torch.maximum(active_mask, new_active)
        h = h * (1 - mask) + full_embedding * mask
        return h 
        

class GlobalEmbeddingEncoder(nn.Module):
    """
    ()
    """
    def __init__(self, params):
        super(GlobalEmbeddingEncoder, self).__init__()
        self.subset_encoder = SubsetEncoder(params)
        self.embedding_expander = EmbeddingExpander(params)

        self.embedding_dim = self.subset_encoder.embedding_dim
    def forward(self, x, idx_subset, args=None):
        """
        Args:
            x: (B, F, N_subset, T)
            idx_subset: (B, N_subset)
        Returns:
            global_embedding: (B, N_all, embedding_dim)
        """
        subset_embedding = self.subset_encoder(x)

        global_embedding = self.embedding_expander(
            subset_embedding, idx_subset, args)
        
        return global_embedding

