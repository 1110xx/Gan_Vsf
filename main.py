# ============================================================
# 基础导入 - 与GIMCC相同
# ============================================================
import torch
import numpy as np
import argparse
import time
from util import *

# ============================================================
# 训练器导入 - TOI-VSF使用TOI_Trainer，GIMCC使用Trainer
# ============================================================
from trainer import create_gan_trainer
# tyx

# ============================================================
# 工具导入 - 与GIMCC相同
# ============================================================
import ast
from copy import deepcopy
import datetime

# ============================================================
# 模型导入 - TOI-VSF特有：联合学习模块
# TOI-VSF: Jointlearning模块（集成预测器和填补器）
# GIMCC: CausalDiscovery + Imputer + OrderEmbedder
# ============================================================
from model.models import build_gan_models


# ============================================================
# 预测器导入 - 与GIMCC相同（MTGNN, ASTGCN, MSTGCN, TGCN）
# ============================================================
# from forecasters.ASTGCN import make_ASTGCN
# from forecasters.MSTGCN import make_MSTGCN
# from forecasters.TGCN import TGCN
# from forecasters.MTGNN import gtnet

# ============================================================
# 工具函数 - 与GIMCC完全相同
# ============================================================
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

# ============================================================
# 参数解析器 - 大部分参数与GIMCC相同
# ============================================================
# ============================================
# 参数解析
# ============================================
parser = argparse.ArgumentParser(description='时空GAN预测模型')

# --- 基础配置 ---
parser.add_argument('--device', type=str, default='cuda:0', help='设备')
parser.add_argument('--data', type=str, default='data/METR-LA', help='数据路径')

# --- 数据参数 ---
parser.add_argument('--seq_in_len', type=int, default=12, help='输入序列长度')
parser.add_argument('--seq_out_len', type=int, default=12, help='输出序列长度')
parser.add_argument('--in_dim', type=int, default=2, help='输入特征维度')

# --- GAN模型参数 ---
parser.add_argument('--encoder_lstm_units', type=int, default=128, help='Encoder LSTM单元数')
parser.add_argument('--conv_filters', type=int, default=64, help='卷积滤波器数量')
parser.add_argument('--kernel_size', type=int, default=3, help='卷积核大小')
parser.add_argument('--decoder_units', type=int, default=128, help='Decoder单元数')
parser.add_argument('--use_batchnorm', type=str_to_bool, default=True, help='是否使用BatchNorm')
parser.add_argument('--use_dropout', type=str_to_bool, default=True, help='是否使用Dropout')
parser.add_argument('--use_gcn', type=str_to_bool, default=False, help='是否使用图卷积')
parser.add_argument('--discriminator_features', type=int, default=2, help='判别器输入特征数')
# ------new----------------------------------------------------
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--subgraph_size', type=int, default=20, help='subgraph size')
parser.add_argument('--num_patch', type=int, default=4, help='The number of patch')
parser.add_argument('--d_model', type=int, default=256, help='hidden layers of time')

parser.add_argument('--random_node_idx_split_runs', type=int, default=100,
                    help='number of random node/variable split runs')
parser.add_argument('--lower_limit_random_node_selections', type=int, default=15,
                    help='lower limit percent value for number of nodes in any given split')
parser.add_argument('--upper_limit_random_node_selections', type=int, default=15,
                    help='upper limit percent value for number of nodes in any given split')

parser.add_argument('--model_name', type=str, default='mtgnn')

parser.add_argument('--mask_remaining', type=str_to_bool, default=False, help='the partial setting, subset S')

parser.add_argument('--predefined_S', type=str_to_bool, default=False, help='whether to use subset S selected apriori')
parser.add_argument('--predefined_S_frac', type=int, default=15,
                    help='percent of nodes in subset S selected apriori setting')
parser.add_argument('--adj_identity_train_test', type=str_to_bool, default=False,
                    help='whether to use identity matrix as adjacency during training and testing')

parser.add_argument('--do_full_set_oracle', type=str_to_bool, default=False, help='the oracle setting, where we have entire data for training and \
testing, but while computing the error metrics, we do on the subset S')
parser.add_argument('--full_set_oracle_lower_limit', type=int, default=15, help='percent of nodes in this setting')
parser.add_argument('--full_set_oracle_upper_limit', type=int, default=15, help='percent of nodes in this setting')

parser.add_argument('--borrow_from_train_data', type=str_to_bool, default=False, help="the Retrieval solution")
parser.add_argument('--num_neighbors_borrow', type=int, default=5,
                    help="number of neighbors to borrow from, during aggregation")
parser.add_argument('--dist_exp_value', type=float, default=0.5, help="the exponent value")
parser.add_argument('--neighbor_temp', type=float, default=0.1, help="the temperature paramter")
parser.add_argument('--use_ewp', type=str_to_bool, default=False,
                    help="whether to use ensemble weight predictor, ie, FDW")

parser.add_argument('--fraction_prots', type=float, default=1.0,
                    help="fraction of the training data to be used as the Retrieval Set")

# ASTGCN
parser.add_argument('--nb_block', type=int, default=2)
parser.add_argument('--K_A', type=int, default=3)
parser.add_argument('--in_channels_A', type=int, default=2)
parser.add_argument('--nb_chev_filter_A', type=int, default=64)
parser.add_argument('--nb_time_filter_A', type=int, default=64)
parser.add_argument('--time_strides_A', type=int, default=1)

parser.add_argument('--input_channels', type=int, default=207, help="input_channels = int_dim * num_nodes")
parser.add_argument('--fourier_modes', type=int, default=6, help="fourier_modes = seq_len // 2")
parser.add_argument('--sequence_len', type=int, default=12, help="sequence_len")
parser.add_argument('--final_out_channels', type=int, default=12, help="sequence_len")
# parser.add_argument('--mid_channels', type=int, default=1024, help="mid_channels")
# parser.add_argument('--kernel_size', type=int, default=5, help="kernel_size") # 已在上面定义
parser.add_argument('--features_len', type=int, default=1, help="features_len")
parser.add_argument('--runid', type=int, default=0, help="run id")
# parser.add_argument('--fore_pre_epochs',type=int,default=0,help='numbers of forecasting backbone pretrain epoch')
parser.add_argument('--epochs', type=int, default=100, help='number of jointlearning training epochs')

parser.add_argument('--w_fc', type=float, default=1, help="weight of forecast loss")
parser.add_argument('--w_ssl', type=float, default=0.8, help="weight of self-supervised loss")
parser.add_argument('--w_subgraph', type=float, default=0.1, help="weight of subgraph loss")
parser.add_argument('--w_graph', type=float, default=0.1, help="weight of graph matching loss")

parser.add_argument('--patience', type=int, default=20, help='for early stopping')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len_patchtst', type=int, default=16, help='patch length')
parser.add_argument('--stride_patchtst', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size_decomposition', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual_patchtst', type=int, default=0, help='individual head; True 1 False 0')

parser.add_argument('--embed_type', type=int, default=3,
                    help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7,
                    help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--d_model_patchtst', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads_patchtst', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers_patchtst', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers_patchtst', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff_patchtst', type=int, default=256, help='dimension of fcn')
parser.add_argument('--factor_patchtst', type=int, default=3, help='attn factor')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--dropout_patchtst', type=float, default=0.05, help='dropout')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--activation', type=str, default='gelu', help='activation')
# ------new----------------------------------------------------

# --- 训练参数 ---
parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--clip', type=float, default=5.0, help='梯度裁剪')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
# parser.add_argument('--epochs', type=int, default=100, help='训练轮数') # 已在上面定义
parser.add_argument('--print_every', type=int, default=50, help='打印间隔')
# parser.add_argument('--patience', type=int, default=20, help='早停耐心值') # 已在上面定义

# --- 课程学习参数 ---
parser.add_argument('--step_size', type=int, default=100, help='课程学习步长')
parser.add_argument('--step_size2', type=int, default=100, help='节点子集变化步长')
parser.add_argument('--num_split', type=int, default=1, help='节点分割数量')

# --- GAN特定参数 ---
parser.add_argument('--w_gan', type=float, default=0.1, help='GAN损失权重（类似train_partial_GAN的loss_weight）')
parser.add_argument('--save_discriminator', type=str_to_bool, default=True, help='是否保存判别器')

# --- 预测解码器参数 ---
parser.add_argument('--decoder_type', type=str, default='lstm', choices=['lstm', 'tcn', 'graph_tcn'],
                    help='预测解码器类型：lstm(默认) 或 tcn(Graph-aware TCN)')
parser.add_argument('--tcn_kernel_size', type=int, default=3, help='TCN卷积核大小')
parser.add_argument('--tcn_num_layers', type=int, default=4, help='TCN层数')
parser.add_argument('--tcn_hidden_dim', type=int, default=128, help='TCN隐藏维度')
parser.add_argument('--tcn_dropout', type=float, default=0.2, help='TCN dropout率')

# --- 重构解码器参数 ---
parser.add_argument('--recon_type', type=str, default='graph', choices=['attention', 'graph'],
                    help='重构解码器类型：attention(原始) 或 graph(图传播，推荐)')
parser.add_argument('--use_adaptive_graph', type=str_to_bool, default=True, help='是否使用自适应图学习')
parser.add_argument('--graph_k', type=int, default=10, help='自适应图Top-k稀疏化')
# parser.add_argument('--node_dim', type=int, default=40, help='节点embedding维度（用于图学习）') # 重复，已在第78行定义
parser.add_argument('--num_recon_graph_layers', type=int, default=3, help='图卷积层数（重构分支）')
parser.add_argument('--recon_hidden_dim', type=int, default=192, help='重构分支隐藏层维度')
parser.add_argument('--recon_dropout', type=float, default=0.3, help='重构分支Dropout率')

# --- 逐层扩散控制参数 ---
parser.add_argument('--diffusion_type', type=str, default='soft', choices=['hard', 'soft'],
                    help='扩散类型：soft(软扩散，推荐) 或 hard(硬阈值)')
parser.add_argument('--diffusion_alpha', type=float, default=0.3,
                    help='软扩散强度 (0,1)，越大扩散越快')
parser.add_argument('--diffusion_decay', type=float, default=0.8,
                    help='逐层衰减系数，控制传播半径增长速度')
parser.add_argument('--diffusion_threshold', type=float, default=0.5,
                    help='硬扩散初始阈值（仅hard模式）')

# --- 实验配置 ---
parser.add_argument('--seed', type=int, default=2024, help='随机种子')
parser.add_argument('--expid', type=int, default=1, help='实验ID')
parser.add_argument('--runs', type=int, default=1, help='运行次数')
parser.add_argument('--path_model_save', type=str, default=None, help='模型保存路径')

# ============================================================
# 参数解析和线程设置 - 与GIMCC相同
# ============================================================
args = parser.parse_args()
torch.set_num_threads(3)

# ============================================================
# 随机种子设置函数 - 与GIMCC完全相同
# ============================================================
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)


# ============================================================
# main函数定义 - 与GIMCC结构相同
# 返回：random_node_split_avg_mae, random_node_split_avg_rmse
# ============================================================
def main(runid):
    # ============================================================
    # 设备设置和数据加载 - 与GIMCC完全相同
    # ============================================================
    device = torch.device(args.device)
    dataloader = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    args.num_nodes = dataloader['train_loader'].num_nodes
    print("Number of variables/nodes = ", args.num_nodes)

    # new
    # lb=args.lower_limit_random_node_selections
    # ub=args.upper_limit_random_node_selections
    # count_percent = np.random.choice( np.arange(lb, ub+1), size=1, replace=False )[0]
    # num_subset = math.ceil(args.num_nodes * (count_percent / 100))
    # print("Number of subset variables/nodes = ", num_subset)
    # new

    if args.node_dim >= args.num_nodes:
        args.node_dim = args.num_nodes
        args.subgraph_size = args.num_nodes

    dataset_name = args.data.strip().split('/')[-1].strip()

    # new args.in_channels = args.num_nodes * args.in_dim
    # args.input_channels = num_subset
    args.final_out_channels = args.num_nodes
    args.features_len = args.num_patch * args.d_model
    args.enc_in = args.num_nodes
    args.dec_in = args.num_nodes
    args.c_out = args.num_nodes

    args.runid = runid

    if dataset_name == "METR-LA":
        args.adj_data = "../data/sensor_graph/adj_mx.pkl"
        predefined_A = load_adj(args.adj_data)
        predefined_A = torch.tensor(predefined_A) - torch.eye(dataloader['total_num_nodes'])
        predefined_A = predefined_A.to(device)
    elif dataset_name.lower() == "pems-bay":
        args.adj_data = "../data/sensor_graph/adj_mx_bay.pkl"
        predefined_A = load_adj(args.adj_data)
        predefined_A = torch.tensor(predefined_A) - torch.eye(dataloader['total_num_nodes'])
        predefined_A = predefined_A.to(device)
    else:
        predefined_A = None

    args.path_model_save = "./saved_models/" + args.model_name + "/" + dataset_name + "/" + "seed" + str(args.seed) + "/"
    import os

    if not os.path.exists(args.path_model_save):
        os.makedirs(args.path_model_save)
    if not os.path.exists("./log"):
        os.makedirs("./log")
    # 模型参数配置（变量子集预测版本）
    model_params = {
        'seq_in_len': args.seq_in_len,
        'seq_out_len': args.seq_out_len,
        'num_nodes': args.num_nodes,  # 完整集节点数
        # 'num_nodes_subset': num_subset, # 子集节点数
        # new
        'in_dim': args.in_dim,
        'encoder_lstm_units': args.encoder_lstm_units,
        'conv_filters': args.conv_filters,
        'kernel_size': args.kernel_size,
        'decoder_units': args.decoder_units,
        'dropout': args.dropout,
        'use_batchnorm': args.use_batchnorm,
        'use_dropout': args.use_dropout,
        'device': device,
        # 预测解码器参数（LSTM或TCN）
        'decoder_type': args.decoder_type,
        'tcn_kernel_size': args.tcn_kernel_size,
        'tcn_num_layers': args.tcn_num_layers,
        'tcn_dilations': [1, 2, 4, 8],  # 固定扩张率
        'tcn_hidden_dim': args.tcn_hidden_dim,
        'tcn_dropout': args.tcn_dropout,
        'tcn_residual': True,
        # 图传播重构参数
        'recon_type': args.recon_type,
        'use_adaptive_graph': args.use_adaptive_graph,
        'graph_k': args.graph_k,
        'node_dim': args.node_dim,
        'num_recon_graph_layers': args.num_recon_graph_layers,
        'recon_hidden_dim': args.recon_hidden_dim,
        'recon_dropout': args.recon_dropout,
        'predefined_A': predefined_A,  # 预定义邻接矩阵（如果有）
        # 逐层扩散控制参数
        'diffusion_type': args.diffusion_type,
        'diffusion_alpha': args.diffusion_alpha,
        'diffusion_decay': args.diffusion_decay,
        'diffusion_threshold': args.diffusion_threshold,
    }
    print(f"\n模型配置：")
    print(f" 完整集节点数: {args.num_nodes}")
    # print(f" 子集节点数: {num_subset}")
    print(f" 输入序列长度: {args.seq_in_len}")
    print(f" 预测序列长度: {args.seq_out_len}")
    print(f" 重构解码器类型: {args.recon_type}")
    if args.recon_type == 'graph':
        print(f" - 图卷积层数: {args.num_recon_graph_layers}")
        print(f" - 自适应图学习: {args.use_adaptive_graph}")
        if args.use_adaptive_graph:
            print(f" - Top-k 稀疏化: {args.graph_k}")
        if predefined_A is not None:
            print(f" - 预定义邻接矩阵: 已加载 ({predefined_A.shape})")
        print(f" - 扩散模式: {args.diffusion_type}")
        if args.diffusion_type == 'soft':
            print(f" * 扩散强度 alpha: {args.diffusion_alpha}")
            print(f" * 衰减系数: {args.diffusion_decay}")
        else:
            print(f" * 初始阈值: {args.diffusion_threshold}")
            print(f" * 衰减系数: {args.diffusion_decay}")
    generator, discriminator = build_gan_models(model_params)
    # 创建 GAN 训练器
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    engine = create_gan_trainer(
        args=args,
        generator=generator,
        discriminator=discriminator,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        clip=args.clip,
        step_size=args.step_size,
        seq_out_len=args.seq_out_len,
        scaler=scaler,
        device=device,
        cl=True  # 启用课程学习
    )
    # ============================================================
    # 训练循环初始化 - 与GIMCC结构完全相同
    # ============================================================
    print("start training...", flush=True)
    his_loss = []
    his_rmse = []
    val_time = []
    train_time = []
    minl = 1e5
    # early stopping
    early_stop_counter = 0
    args.epoch = 0

    # ============================================================
    # Epoch循环 - 与GIMCC结构相同
    # ============================================================
    for i in range(1, args.epochs + 1):
        args.epoch = i
        args.isTest = False
        # ============================================================
        # 损失记录 - GAN 版本记录多种损失
        # GAN: generator_loss, rmse, pred_loss, recon_loss, gan_loss, d_loss, recon_rmse
        # ============================================================
        train_loss = []  # generator总损失
        train_rmse = []  # 预测RMSE
        train_pred_loss = []  # 预测损失
        train_recon_loss = []  # 重构损失
        train_gan_loss = []  # 对抗损失
        train_d_loss = []  # 判别器损失
        train_recon_rmse = []  # 重构RMSE
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        # ============================================================
        # [SAME] 训练迭代 - 与GIMCC结构相同
        # ============================================================
        # 迭代的返回训练数据
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            # x：输入数据（特征
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            # 训练标签
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            # 如果迭代步长为step_size2的倍数，则随机排列节点
            if iter % args.step_size2 == 0:
            # perm:
            # 存储打乱后的节点索引顺序
                perm = np.random.permutation(range(args.num_nodes))
            # num_split 是节点分割数量参数 用于训练
            num_sub = int(args.num_nodes / args.num_split)
            # 遍历子集
            for j in range(args.num_split):
                if j != args.num_split - 1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                # id 是 子集节点索引
                id = torch.tensor(id).to(device)
                # trainx 是一个 4 维张量，通常维度为：
                # 维度 0：batch_size（批次大小，一批有多少个样本）
                # 维度 1：features/channels（特征维度或通道数）
                # 维度 2：num_nodes（节点数量）
                # 维度 3：time_steps（时间步长）
                # 切片语法解释：
                # trainx[:, :, id, :] 的四个位置分别表示：
                # : (第1维) - 保留所有批次样本
                # : (第2维) - 保留所有特征/通道
                # id (第3维) - 只选择 id 中指定的节点
                # : (第4维) - 保留所有时间步
                tx_subset = trainx[:, :, id, :]  # 子集数据
                tx_full = trainx  # 完整集数据（用于判别器）
                ty_subset = trainy[:, :, id, :]
                '''
                METR-LA 和 PEMS-BAY 是交通数据集，原始数据包含了"流量+时间"两个特征，但训练时只想用"流量"这一个特征，所以通过 [:,:1,:,:] 只取第一个特征（特征索引0），相当于把第二维从 2 缩减到 1。
                '''
                if dataset_name.lower() in {"metr-la", "pems-bay"}:
                    tx_subset = tx_subset[:, :1, :, :]
                    tx_full = tx_full[:, :1, :, :]
                # ============================================================
                # 训练逻辑 - GAN 版本（传递子集和完整集）
                # engine.train 返回7个指标：
                # (generator_loss, rmse, pred_loss, recon_loss, gan_loss, d_loss, recon_rmse)
                # ============================================================
                metrics = engine.train(args, tx_subset, tx_full, ty_subset[:, 0, :, :], id)
                train_loss.append(metrics[0])  # generator总损失
                train_rmse.append(metrics[1])  # 预测RMSE
                train_pred_loss.append(metrics[2])  # 预测损失
                train_recon_loss.append(metrics[3])  # 重构损失
                train_gan_loss.append(metrics[4])  # 对抗损失
                train_d_loss.append(metrics[5])  # 判别器损失
                train_recon_rmse.append(metrics[6])  # 重构RMSE
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, G_Loss: {:.4f}, Pred_RMSE: {:.4f}, Pred_Loss: {:.4f}, Recon_Loss: {:.4f}, GAN_Loss: {:.4f}, D_Loss: {:.4f}'
                print(log.format(iter, train_loss[-1], train_rmse[-1], train_pred_loss[-1],
                             train_recon_loss[-1], train_gan_loss[-1], train_d_loss[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # ============================================================
        # 验证阶段 - 与GIMCC结构相同
        # ============================================================
        valid_loss = []
        valid_rmse = []

        s1 = time.time()
        args.isTest = True
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            if dataset_name.lower() in {"metr-la", "pems-bay"}:
                testx = testx[:, :1, :, :]
            # id = torch.arange(args.num_nodes).to(device)
            id = get_node_random_idx_split(args, args.num_nodes, args.lower_limit_random_node_selections,
                                           args.upper_limit_random_node_selections)
            metrics = engine.eval_subset(args, testx, testy[:, 0, :, :], id)
            valid_loss.append(metrics[0])
            valid_rmse.append(metrics[1])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_rmse = np.mean(train_rmse)
        mpred_loss = np.mean(train_pred_loss)
        mrecon_loss = np.mean(train_recon_loss)
        mgan_loss = np.mean(train_gan_loss)
        md_loss = np.mean(train_d_loss)
        mrecon_rmse = np.mean(train_recon_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        his_rmse.append(mvalid_rmse)

        # ============================================================
        # 打印训练信息 - GAN 版本增加对抗损失信息
        # ============================================================
        log = 'Runid: {:d}, Epoch: {:03d}, G_Loss: {:.4f}, Pred_RMSE: {:.4f}, Pred_Loss: {:.4f}, Recon_Loss: {:.4f}, GAN_Loss: {:.4f}, D_Loss: {:.4f}, Valid_Loss: {:.4f}, Valid_RMSE: {:.4f}, Time: {:.4f}/epoch'
        print(log.format(runid, i, mtrain_loss, mtrain_rmse, mpred_loss, mrecon_loss, mgan_loss, md_loss, mvalid_loss,
                         mvalid_rmse, (t2 - t1)), flush=True)

        # ============================================================
        # 早停机制 - 与GIMCC完全相同
        # ============================================================
        if mvalid_loss < minl:
            # ============================================================
            # 模型保存 - GAN 版本保存生成器和判别器
            # ============================================================
            save_path = args.path_model_save + "exp" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth"
            engine.save_models(save_path, i)
            minl = mvalid_loss
            early_stop_counter = 0
            print(f"early_stop_counter: {early_stop_counter} / {args.patience}")
        else:
            early_stop_counter += 1
            print(f"early_stop_counter: {early_stop_counter} / {args.patience}")
            if early_stop_counter >= args.patience and i > 99:
                print("Early stopping...")
                break

    if args.epochs > 0:
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        bestid = np.argmin(his_loss)
        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
        print("The valid RMSE on best model is", str(round(his_rmse[bestid], 4)))
    # ============================================================
    # 加载最佳模型 - GAN 版本加载生成器和判别器
    # ============================================================
    load_path = args.path_model_save + "exp" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth"
    engine.load_models(load_path)
    print("\nGAN Models (Generator & Discriminator) loaded\n")

    engine.generator.eval()
    engine.discriminator.eval()

    print("\n Performing test set run. To perform the following inference on validation data, simply adjust 'y_test' to 'y_val' and 'test_loader' to 'val_loader', which\
    has been commented out for faster execution \n")

    # ============================================================
    # 测试阶段 - 随机节点分割测试（与GIMCC相同）
    # ============================================================
    random_node_split_avg_mae = []
    random_node_split_avg_rmse = []

    args.isTest = True

    for split_run in range(args.random_node_idx_split_runs):
        print("running on random node idx split ", split_run)
        if args.do_full_set_oracle:
            idx_current_nodes = np.arange(args.num_nodes, dtype=int).reshape(-1)
            assert idx_current_nodes.shape[0] == args.num_nodes
        else:
            idx_current_nodes = get_node_random_idx_split(args, args.num_nodes, args.lower_limit_random_node_selections,
                                                      args.upper_limit_random_node_selections)
        print("Number of nodes in current random split run = ", idx_current_nodes.shape)

        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]
        if not args.predefined_S:
            realy = realy[:, idx_current_nodes, :]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testx = zero_out_remaining_input(testx, idx_current_nodes,
                                         args.device)  # Remove the data corresponding to the variables that are not a part of subset "S"
            if dataset_name.lower() in {"metr-la", "pems-bay"}:
                testx = testx[:, :1, :, :]
            with torch.no_grad():
            # ============================================================
            # [DIFF] 推理逻辑 - GAN 版本
            # 提取子集数据并传递给生成器
            # ============================================================
            # testx 是完整集数据，需要根据 idx_current_nodes 提取子集
                testx_subset = testx[:, :, idx_current_nodes, :]  # (B, F, N_subset, T)
                # 生成器返回 (prediction, reconstruction)
                preds, _ = engine.generator(testx_subset, idx_current_nodes, args)
                # preds: (B, 1, N_subset, T)
                # 转换格式：(B, 1, N_subset, T) -> (B, N_subset, T)
                # preds = preds.squeeze(1) # (B, N_subset, T)
                preds = preds[:, 0, :, :]
            outputs.append(preds)

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]
        # ============================================================
        # [FIX] 从全局预测中提取子集节点的预测结果
        # 生成器返回的是所有节点的预测，需要提取对应idx_current_nodes的节点
        # ============================================================
        if not args.predefined_S:
            yhat = yhat[:, idx_current_nodes, :]

        # ============================================================
        # 评估指标计算 - 与GIMCC完全相同
        # ============================================================
        mae = []
        rmse = []

        for i in range(
            args.seq_out_len):  # this computes the metrics for multiple horizons lengths, individually, starting from 0 to args.seq_out_len
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]

            metrics = metric(pred, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[1]))
            mae.append(metrics[0])
            rmse.append(metrics[1])

        random_node_split_avg_mae.append(mae)
        random_node_split_avg_rmse.append(rmse)

    return random_node_split_avg_mae, random_node_split_avg_rmse

    # ============================================================
    # 程序入口 - 与GIMCC完全相同
    # 多次运行、统计计算、结果输出格式100%一致
    # ============================================================
if __name__ == "__main__":

    starttime = datetime.datetime.now()
    mae = []
    rmse = []

    # ============================================================
    # 多次运行循环 - 与GIMCC完全相同
    # ============================================================
    for i in range(args.runs):
        m1, m2 = main(i)
        mae.extend(m1)
        rmse.extend(m2)

    # ============================================================
    # 统计计算 - 与GIMCC完全相同
    # ============================================================
    mae = np.array(mae)
    rmse = np.array(rmse)

    amae = np.mean(mae, 0)
    armse = np.mean(rmse, 0)
    all_runs_avermae = np.mean(amae)
    all_runs_avermse = np.mean(armse)

    smae = np.std(mae, 0)
    srmse = np.std(rmse, 0)

    all_runs_aver_stdmae = np.mean(smae)
    all_runs_aver_stdrmse = np.mean(srmse)

    # ============================================================
    # 结果输出 - 与GIMCC完全相同的格式
    # ============================================================
    print('\n\nResults for multiple runs\n\n')
    for i in range(args.runs):
        print("runs {:d} ; MAE = {:.4f} +- {:.4f} ; RMSE = {:.4f} +- {:.4f}".format(
            i + 1, amae[i], smae[i], armse[i], srmse[i]))
    print("\n Final: MAE = {:.4f} +- {:.4f} ; RMSE = {:.4f} +- {:.4f}".format(all_runs_avermae, all_runs_aver_stdmae,
                                                                              all_runs_avermse, all_runs_aver_stdrmse))
    # ============================================================
    # 运行时间统计 - 与GIMCC完全相同
    # ============================================================
    # long running
    endtime = datetime.datetime.now()
    print("\n")
    print((endtime - starttime).seconds)
    print("\n")
