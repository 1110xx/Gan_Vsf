import torch

import torch.nn as nn

import torch.nn.functional as F

# 导入重构后的编码器、解码器和判别器

try:

    from .encoder import GlobalEmbeddingEncoder

    from .recon_decoder import ReconstructionDecoder

    from .pred_decoder import PredictionDecoder

    from .discriminator import GraphDiscriminator, SpatioTemporalDiscriminator

    MODULES_AVAILABLE = True

except  ImportError:

    MODULES_AVAILABLE = False

    print("Warning: encoder.py, recon_decoder.py, pred_decoder.py or discriminator.py not available.")


class SpatioTemporalGenerator(nn.Module):


    """

    时空生成器 - 变量子集预测版本（重构版）

    架构：

    - 输入：变量子集 (batch, in_dim, num_nodes_subset, seq_in_len)

    - 输出1：prediction - 全局预测 (batch, 1, num_nodes, seq_out_len)

    - 输出2：reconstruction - 全局重构 (batch, in_dim, num_nodes, seq_in_len)

    新流程：

    输入子集 (B, F, N_subset, T_in)

    → GlobalEmbeddingEncoder

    → 全局embedding (B, N_all, dim)

    → ┬→ ReconstructionDecoder → 全局重构 (B, F, N_all, T_in)

    └→ 预测解码器 → 全局预测 (B, 1, N_all, T_out)

    """


    def __init__(self, params):


        super(SpatioTemporalGenerator, self).__init__()

        self.seq_in_len = params['seq_in_len']

        self.seq_out_len = params['seq_out_len']

        self.num_nodes = params['num_nodes']  # 完整集节点数

        self.in_dim = params['in_dim']

        # ========== 模块1：全局嵌入编码器 ==========

        # 子集数据 (B, F, N_subset, T) -> 全局embedding (B, N_all, dim)

        if not MODULES_AVAILABLE:

            raise ImportError("encoder.py, recon_decoder.py and pred_decoder.py must be available")

        self.encoder = GlobalEmbeddingEncoder(params)

        # ========== 模块2：重构解码器 ==========

        # 全局embedding (B, N_all, dim) -> 全局重构 (B, F, N_all, T)

        self.recon_decoder = ReconstructionDecoder(params)

        # ========== 模块3：预测解码器（全局）==========

        # 全局embedding (B, N_all, dim) -> 全局预测 (B, 1, N_all, T_out)

        self.pred_decoder = PredictionDecoder(params)


    def forward(self, x, idx=None, args=None):


        """

        前向传播（图增强版）

        Args:

        x: (B, F, N_subset, T_in) - 子集数据

        idx: 子集节点索引

        args: 额外参数

        Returns:

        prediction: (B, 1, N_all, T_out) - 全局预测

        reconstruction: (B, F, N_all, T_in) - 全局重构

        """

        batch_size = x.size(0)

        # ========== 步骤1: 编码 - 子集数据 -> 全局embedding ==========

        global_embedding = self.encoder(x, idx, args)

        # global_embedding: (B, N_all, embedding_dim)

        # ========== 步骤2: 获取邻接矩阵 ==========
        if hasattr(self.encoder, 'use_imputer') and self.encoder.use_imputer:
            adj = self.encoder.imputer.get_adjacency_matrix()
        else:
            adj = self.encoder.embedding_expander.get_adjacency_matrix()

        # adj: (N_all, N_all)

        # ========== 步骤3: 解码分支1 - 全局重构（传入邻接矩阵）==========

        reconstruction = self.recon_decoder(global_embedding, adj)

        # reconstruction: (B, F, N_all, T_in)

        # ========== 步骤4: 解码分支2 - 全局预测（传入邻接矩阵）==========

        prediction = self.pred_decoder(global_embedding, adj)

        # prediction: (B, 1, N_all, T_out)

        return prediction, reconstruction

    # 注意：SpatioTemporalDiscriminator 已移至 discriminator.py

# 这里保留别名以保持向后兼容

if MODULES_AVAILABLE:
    # 使用新的图增强判别器
    pass

else:
    # 如果导入失败，定义一个占位符
    class GraphDiscriminator:
        pass
    class SpatioTemporalDiscriminator:
        pass


def build_gan_models(params, use_graph_discriminator=True):


    """

    构建生成器和判别器（图增强版）

    Args:

    params: 配置字典

    use_graph_discriminator: 是否使用图结构判别器（默认True）

    Returns:

    generator, discriminator

    """

    generator = SpatioTemporalGenerator(params)

    # 选择判别器类型

    if use_graph_discriminator:

        discriminator = GraphDiscriminator(params)

        print("✓ Using Graph-Enhanced Discriminator")

    else:

        discriminator = SpatioTemporalDiscriminator(params)

        print("✓ Using Standard Discriminator")

    return generator, discriminator
