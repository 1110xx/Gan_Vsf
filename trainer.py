"""
时空GAN训练器 - 变量子集预测版本（重构版）

核心功能：
1. 全局预测：输入部分节点，预测所有节点的未来值
2. 损失计算：动态提取子集进行预测损失计算
3. 重构完整集：通过 embedding 重构完整集，用于对抗训练
4. 混合损失训练（预测损失 + 对抗损失）
5. 标签平滑技术
6. 课程学习（Curriculum Learning）

架构（重构后）：
- 生成器 G：输入子集 -> 全局预测 + 全局重构
- 判别器 D：判别 原始完整集 vs 重构完整集
- 损失计算：从全局预测中提取子集计算损失

损失设计：
- 判别器损失：BCE(真实完整集, 真) + BCE(重构完整集, 假)
- 生成器损失：w_fc * 预测损失（子集） + w_gan * 对抗损失

参考：TOI-VSF 的变量子集预测架构 + GAN 对抗训练

Author: Generated based on GAN-RNN train_partial_GAN + TOI-VSF
Date: 2025-10-19
Updated: 2025-01-28 (Refactored for global prediction with subset loss)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import util
from util import masked_mae, masked_rmse


class GAN_Trainer():
    """
    时空GAN训练器 - 变量子集预测版本（重构版）
    架构设计（重构后）：
    - 生成器 G：输入变量子集 -> 输出(全局预测, 全局重构)
    - 判别器 D：判别 原始完整集 vs 重构完整集
    - 损失计算：从全局预测中提取子集进行损失计算
    损失函数：
    - Generator损失 = w_fc * 预测MAE损失（子集） + w_gan * 对抗损失
    - Discriminator损失 = 标签平滑的二元交叉熵
    关键改进：
    - 全局预测：模型输出全局结果，训练时动态提取子集
    - 灵活性：支持动态变化的子集大小
    参考架构：
    - TOI-VSF: 变量子集预测 + 完整集重构
    - GAN: 对抗训练替代重构损失
    """
    def __init__(self, args, generator, discriminator, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.args = args
        self.scaler = scaler
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        # 检查是否使用图结构判别器
        self.use_graph_disc = hasattr(args, 'use_graph_discriminator') and args.use_graph_discriminator
        if self.use_graph_disc:
            print("✓ Trainer: Using Graph-Enhanced Discriminator")
        # ========== 阶段性训练参数 ==========
        self.stage1_epochs = getattr(args, 'stage1_epochs', 50)
        self.use_staged_training = getattr(args, 'use_staged_training', True)
        self.current_stage = 1
        if self.use_staged_training:
            print("=" * 60)
            print("✓ Staged Training Enabled")
            print(f" Stage 1 (Epochs 1-{self.stage1_epochs}): L_recon → Decoder + Encoder")
            print(f" Stage 2 (Epochs {self.stage1_epochs+1}+): L_recon → Decoder only")
            print("=" * 60)
        # 优化器
        self.generator_optimizer = torch.optim.Adam(
        self.generator.parameters(),
        lr=lrate,
        weight_decay=wdecay
        )
        self.discriminator_optimizer = torch.optim.Adam(
        self.discriminator.parameters(),
        lr=lrate,
        weight_decay=wdecay
        )
        # 损失函数
        self.bce_loss = nn.BCELoss() # 二元交叉熵
        self.loss = util.masked_mae
        # 损失权重（类似train_partial_GAN中的'loss_weight'）
        self.loss_weight = args.w_gan # GAN损失的权重
        self.clip = clip
        # 课程学习参数
        self.iter = 1
        self.step = step_size
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl
    def train(self, args, input_subset, input_full, real_val, idx=None):
        """
        训练方法 - 变量子集预测版本（重构版）
        Args:
        args: 参数配置
        input_subset: 输入子集序列 (B, F, N_subset, T) - 用于生成器输入
        input_full: 输入完整集序列 (B, F, N_all, T) - 用于判别器的真实样本
        real_val: 真实未来值 (B, T, N_subset) - 子集的真实预测目标
        idx: 节点索引 - 子集在完整集中的索引
        Returns:
        loss, rmse, loss_pred, loss_recon, gan_loss, d_loss, recon_rmse
        流程（重构后）：
        1. 生成器输入子集，输出：prediction（全局预测）, reconstruction（全局重构）
        2. 从全局预测中提取子集进行损失计算
        3. 判别器判别：原始完整集 vs 重构完整集
        4. 生成器损失 = w_fc * 预测损失（子集） + w_gan * 对抗损失
        """
        self.generator.train()
        self.discriminator.train()
        batch_size = input_subset.size(0)
        # ========== 训练判别器 ==========
        self.discriminator_optimizer.zero_grad()
        # 生成假样本（不需要梯度）
        with torch.no_grad():
            # 生成器返回：(prediction, reconstruction)
            # prediction: (B, 1, N_all, T) - 全局预测（重构后）
            # reconstruction: (B, F, N_all, T) - 全局重构
            pred_output, recon_output = self.generator(input_subset, idx=idx, args=args)
        # 判别器输入：都是 (B, F, N, T) 格式
        # recon_output: 重构完整集 (B, F, N_all, T)
        # input_full: 原始完整集 (B, F, N_all, T)
        # ========== 获取邻接矩阵（如果使用图判别器）==========
        if self.use_graph_disc:
            adj = self.generator.encoder.embedding_expander.get_adjacency_matrix()
            discriminator_guess_fakes = self.discriminator(recon_output, adj)
            discriminator_guess_reals = self.discriminator(input_full, adj)
        else:
            discriminator_guess_fakes = self.discriminator(recon_output)
            discriminator_guess_reals = self.discriminator(input_full)
        # 标签平滑
        smooth_real_labels = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0).to(self.device)
        smooth_fake_labels = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.2).to(self.device)
        # 判别器损失：判别 原始完整集 vs 重构完整集
        loss_reals = self.bce_loss(discriminator_guess_reals, smooth_real_labels)
        loss_fakes = self.bce_loss(discriminator_guess_fakes, smooth_fake_labels)
        discriminator_loss = loss_reals + loss_fakes
        discriminator_loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip)
        self.discriminator_optimizer.step()
        # ========== 训练生成器 ==========
        self.generator_optimizer.zero_grad()
        # ========== 阶段判断 ==========
        if self.use_staged_training:
            if args.epoch <= self.stage1_epochs:
                if self.current_stage != 1:
                    self.current_stage = 1
                    print("\n" + "=" * 60)
                    print(f"Stage 1: Stable Pre-training (Epoch {args.epoch})")
                    print(" L_recon → Decoder + Encoder ✅")
                    print("=" * 60 + "\n")
            else:
                if self.current_stage != 2:
                    self.current_stage = 2
                    print("\n" + "=" * 60)
                    print(f"Stage 2: Structure Learning (Epoch {args.epoch})")
                    print(" L_recon → Decoder only ❌ (Encoder frozen for recon)")
                    print("=" * 60 + "\n")
        # ========== 阶段1：正常训练（所有梯度流向） ==========
        if not self.use_staged_training or self.current_stage == 1:
            # 正常生成（需要梯度）
            pred_output, recon_output = self.generator(input_subset, idx=idx, args=args)
        # ========== 阶段2：重构分支截断梯度 ==========
        else: # Stage 2
            # 获取全局embedding
            global_embedding = self.generator.encoder(input_subset, idx, args)
            # 预测分支：保持梯度流向
            pred_input = global_embedding.unsqueeze(2).repeat(1, 1, self.generator.seq_out_len, 1)
            pred_input = pred_input.reshape(batch_size * self.generator.num_nodes, self.generator.seq_out_len, -1)
            pred_lstm_out, _ = self.generator.pred_decoder_lstm(pred_input)
            prediction_raw = self.generator.pred_output(pred_lstm_out)
            pred_output = prediction_raw.reshape(batch_size, self.generator.num_nodes, self.generator.seq_out_len, 1)
            pred_output = pred_output.permute(0, 3, 1, 2)
            # 重构分支：截断梯度 ⭐
            global_embedding_detached = global_embedding.detach()
            adj = self.generator.encoder.embedding_expander.get_adjacency_matrix()
            recon_output = self.generator.recon_decoder(global_embedding_detached, adj)
        # pred_output: (B, 1, N_all, T) - 全局预测（重构后）
        # recon_output: (B, F, N_all, T) - 全局重构
        # ========== 关键步骤：从全局预测中提取子集 ==========
        # pred_output: (B, 1, N_all, T) -> 提取子集 -> (B, 1, N_subset, T)
        pred_output_subset = pred_output[:, :, idx, :] # 提取子集节点
        # 预测输出转换：(B, 1, N_subset, T) -> (B, T, N_subset, 1)
        pred_output_subset = pred_output_subset.transpose(1, 3) # (B, T, N_subset, 1)
        predict = self.scaler.inverse_transform(pred_output_subset) # (B, T, N_subset, 1)

        # 重构输出：保持 (B, F, N_all, T) 格式
        recon_output_for_loss = recon_output # (B, F, N_all, T)
        reconstruction = self.scaler.inverse_transform(recon_output_for_loss) # (B, F, N_all, T)

        # 判别器判断重构完整集（使用原始格式 (B, F, N, T)）
        if self.use_graph_disc:
            adj = self.generator.encoder.embedding_expander.get_adjacency_matrix()
            discriminator_guess_fakes = self.discriminator(recon_output, adj)
        else:
            discriminator_guess_fakes = self.discriminator(recon_output)

        # 准备真实值
        # real_val: (B, N, T) -> (B, 1, N, T) -> (B, T, N, 1)
        pred_real = torch.unsqueeze(real_val, dim=1) # (B, 1, N, T) - 预测的真实值
        pred_real = pred_real.transpose(1, 3) # (B, T, N, 1) - 转置以匹配predict的维度
        # 原始完整集：保持 (B, F, N_all, T) 格式
        orig_real = input_full # (B, F, N_all, T)

        # 课程学习
        if self.iter % self.step == 0 and self.task_level <= self.seq_out_len:
            self.task_level += 1

        # 1. 预测损失（类似 TOI-VSF 的 loss_pred）
        # predict: (B, T, N_subset, 1)
        # pred_real: (B, T, N_subset, 1) ✅ 维度已对齐
        if self.cl:
            loss_pred, _ = self.loss(predict[:, :, :, :self.task_level], pred_real[:, :, :, :self.task_level], 0.0)
        else:
            loss_pred, _ = self.loss(predict, pred_real, 0.0)

        # 2. 重构损失（在全集上计算）⭐
        # 使用完整集数据作为重构目标
        # reconstruction: (B, F, N_all, T)
        # orig_real: (B, F, N_all, T) - 来自 input_full
        orig_real = input_full # (B, F, N_all, T)
        orig_real_scaled = self.scaler.inverse_transform(orig_real) # 反归一化
        if self.cl:
            loss_recon, _ = self.loss(reconstruction[:, :, :, :self.task_level],
            orig_real_scaled[:, :, :, :self.task_level], 0.0)
        else:
            loss_recon, _ = self.loss(reconstruction, orig_real_scaled, 0.0)

        # 3. GAN对抗损失（替代 TOI-VSF 的重构损失）
        ones_labels = torch.ones_like(discriminator_guess_fakes)
        gan_loss = self.bce_loss(discriminator_guess_fakes, ones_labels)

        # 4. 组合损失（参考 TOI-VSF: w_fc * loss_pred + w_imp * loss_imp）
        # 这里我们用 w_gan 替代 w_imp，用对抗损失替代重构损失
        generator_loss = args.w_fc * loss_pred + args.w_gan * gan_loss

        generator_loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip)
        self.generator_optimizer.step()

        # 计算RMSE
        rmse = util.masked_rmse(predict, pred_real, 0.0)[0].item()
        recon_rmse = util.masked_rmse(reconstruction, orig_real, 0.0)[0].item()

        self.iter += 1

        return generator_loss.item(), rmse, loss_pred.item(), loss_recon.item(), gan_loss.item(), discriminator_loss.item(), recon_rmse
    def eval(self, args, input, real_val, idx=None):
        """
        评估方法 - 变量子集预测版本（重构版）
        Args:
        args: 参数配置
        input: 输入序列 (B, F, N_subset, T) - 变量子集
        real_val: 真实未来值 (B, N_subset, T) - 子集的真实预测目标
        idx: 节点索引 - 子集在完整集中的索引
        Returns:
        loss, rmse
        注意：只评估预测损失，不评估重构损失
        """
        self.generator.eval()
        # 生成预测（返回 prediction 和 reconstruction）
        pred_output, _ = self.generator(input, idx=idx, args=args)
        # pred_output: (B, 1, N_all, T) - 全局预测（重构后）
        # ========== 关键步骤：从全局预测中提取子集 ==========
        pred_output_subset = pred_output[:, :, idx, :] # (B, 1, N_subset, T)
        # 转换：(B, 1, N_subset, T) -> (B, T, N_subset, 1)
        pred_output_subset = pred_output_subset.transpose(1, 3) # (B, T, N_subset, 1)
        # 准备真实值：(B, N_subset, T) -> (B, 1, N_subset, T) -> (B, T, N_subset, 1)
        real = torch.unsqueeze(real_val, dim=1) # (B, 1, N_subset, T)
        real = real.transpose(1, 3) # (B, T, N_subset, 1) - 转置以匹配predict的维度
        # 反归一化
        predict = self.scaler.inverse_transform(pred_output_subset) # (B, T, N_subset, 1)
        # 计算预测损失
        loss = self.loss(predict, real, 0.0)
        rmse = util.masked_rmse(predict, real, 0.0)[0].item()
        return loss[0].item(), rmse

    def eval_subset(self, args, input, real_val, idx=None):
        """
        子集评估方法 - 变量子集预测版本（重构版）
        Args:
        args: 参数配置
        input: 输入序列 (B, F, N_all, T) - 完整集（会根据idx提取子集）
        real_val: (B, N_all, T) - 完整集的真实值
        idx: 节点索引 - 指定评估的子集节点
        Returns:
        loss, rmse
        """
        self.generator.eval()
        # 提取子集
        input_subset = input[:, :, idx, :] # (B, F, N_subset, T)
        # 生成预测（返回 prediction 和 reconstruction）
        pred_output, _ = self.generator(input_subset, idx=idx, args=args)
        # pred_output: (B, 1, N_all, T) - 全局预测（重构后）
        # ========== 关键步骤：从全局预测中提取子集 ==========
        pred_output_subset = pred_output[:, :, idx, :] # (B, 1, N_subset, T)
        # 转换：(B, 1, N_subset, T) -> (B, T, N_subset, 1)
        pred_output_subset = pred_output_subset.transpose(1, 3) # (B, T, N_subset, 1)
        # 准备真实值：(B, N_all, T) -> (B, 1, N_all, T) -> (B, 1, N_subset, T) -> (B, T, N_subset, 1)
        real = torch.unsqueeze(real_val, dim=1) # (B, 1, N_all, T)
        real = real[:, :, idx, :] # 只取子集节点 (B, 1, N_subset, T)
        real = real.transpose(1, 3) # (B, T, N_subset, 1) - 转置以匹配predict的维度
        # 反归一化
        predict = self.scaler.inverse_transform(pred_output_subset) # (B, T, N_subset, 1)
        # 计算预测损失
        loss = self.loss(predict, real, 0.0)
        rmse = util.masked_rmse(predict, real, 0.0)[0].item()
        adjust_learning_rate(self.generator_optimizer, args)
        return loss[0].item(), rmse

    def eval_joint(self, args, input_subset, input_full, real_val, idx=None):
        """
        联合评估方法 - 同时评估预测和重构性能（重构版）
        Args:
        args: 参数配置
        input_subset: 输入子集序列 (B, F, N_subset, T)
        input_full: 输入完整集序列 (B, F, N_all, T)
        real_val: 真实未来值 (B, N_subset, T) - 子集的真实预测目标
        idx: 节点索引 - 子集在完整集中的索引
        Returns:
        loss, rmse_pred, loss_recon, rmse_recon
        类似 TOI-VSF 的 eval_joint，同时评估预测和重构
        """
        self.generator.eval()
        # 生成预测和重构
        pred_output, recon_output = self.generator(input_subset, idx=idx, args=args)
        # pred_output: (B, 1, N_all, T) - 全局预测（重构后）
        # recon_output: (B, F, N_all, T) - 全局重构
        # ========== 关键步骤：从全局预测中提取子集 ==========
        pred_output_subset = pred_output[:, :, idx, :] # (B, 1, N_subset, T)
        # 转换格式
        pred_output_subset = pred_output_subset.transpose(1, 3) # (B, T, N_subset, 1)
        # 准备真实值
        pred_real = torch.unsqueeze(real_val, dim=1) # (B, 1, N_subset, T)
        pred_real = pred_real.transpose(1, 3) # (B, T, N_subset, 1) - 转置以匹配predict的维度
        orig_real = input_full # (B, F, N_all, T)
        # 反归一化
        predict = self.scaler.inverse_transform(pred_output_subset) # (B, T, N_subset, 1)
        reconstruction = self.scaler.inverse_transform(recon_output) # (B, F, N_all, T)
        orig_real_scaled = self.scaler.inverse_transform(orig_real) # (B, F, N_all, T)
        # 计算预测损失（子集）
        loss_pred = self.loss(predict, pred_real, 0.0)
        rmse_pred = util.masked_rmse(predict, pred_real, 0.0)[0].item()
        # 计算重构损失（全集）⭐
        loss_recon = self.loss(reconstruction, orig_real_scaled, 0.0)
        rmse_recon = util.masked_rmse(reconstruction, orig_real_scaled, 0.0)[0].item()
        # 组合损失（类似训练时的组合）
        loss = args.w_fc * loss_pred[0].item() + args.w_gan * loss_recon[0].item()
        return loss, rmse_pred, loss_recon[0].item(), rmse_recon
    def save_models(self, save_path, epoch):
        """保存模型"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
        }, save_path)
        print(f'模型已保存至 {save_path}')
    def load_models(self, load_path):
        """加载模型"""
        checkpoint = torch.load(load_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        print(f'模型已加载自 {load_path}')
        return checkpoint['epoch']


def adjust_learning_rate(optimizer, args):
    """学习率调整函数 - 与TOI-VSF保持一致"""
    # 如果没有指定学习率调整策略，直接返回
    if not hasattr(args, 'lradj') or args.lradj is None:
        return
    epoch = args.epoch + 1
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
        20: 5e-4, 40: 1e-4, 60: 5e-5,
        80: 1e-5, 100: 5e-6, 120: 1e-6
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('更新学习率至 {}'.format(lr))


def create_gan_trainer(args, generator, discriminator, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
    """
    创建GAN训练器的工厂函数
    """
    return GAN_Trainer(args, generator, discriminator, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl)

class Optim(object):
    """优化器封装类 - 与TOI-VSF和GIMCC保持一致"""
    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()
        return grad_norm

    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()