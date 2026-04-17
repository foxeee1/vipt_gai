"""
统一Prompt生成器 - 支持三种训练/推理模式

【模式说明】
1. 'fixed'（标准训练固定Prompt）：
   - 直接返回固定的可学习Base Prompt
   - 用于标准训练，不依赖任何生成器
   - 与原ViPT的Base Prompt机制完全一致

2. 'static_gen'（摊销式FOMAML训练后生成静态Prompt）：
   - 使用轻量生成器基于平均模态状态生成Prompt
   - 用于FOMAML训练后的静态推理模式
   - 所有帧使用同一组Prompt（摊销思想）

3. 'dynamic'（摊销式FOMAML训练后动态生成每帧Prompt）：
   - 使用轻量生成器基于当前帧模态状态生成Prompt
   - 用于FOMAML训练后的动态推理模式
   - 每帧生成不同的Prompt（动态适应）

【设计原则】
- 最小改动：基于ViPT现有代码结构，不破坏主干
- 统一接口：通过mode参数一键切换三种模式
- 功能完整：包含Base Prompt、Mask Prompt、Consistency Prompt、Temporal Prompt、门控融合
- 易于扩展：预留接口，方便后续添加新的Prompt模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class PromptVisualizer:
    """
    Prompt模块可视化工具（精简版）- 只记录核心指标

    【核心指标】
    1. prompt_mean: Prompt输出均值 → 判断Prompt是否有有效输出
    2. grad_norm_total: 模块总梯度范数 → 判断梯度是否正常传播
    3. avg_rgb_weight: 模态权重均值 → 判断模块是否学到模态偏好
    4. token_consistency直方图: 权重分布 → 判断学习范围是否合理
    """
    _instance = None
    _writer = None
    _step = 0
    _enabled = False
    _log_interval = 50

    @classmethod
    def get(cls) -> 'PromptVisualizer':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def setup(cls, log_dir: str = None, enabled: bool = True, log_interval: int = 50):
        cls._enabled = enabled
        cls._log_interval = log_interval
        cls._step = 0
        if enabled and log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                cls._writer = SummaryWriter(log_dir)
            except ImportError:
                cls._writer = None

    @classmethod
    def increment_step(cls):
        cls._step += 1

    @property
    def should_log(self) -> bool:
        return self._enabled and self._writer is not None and self._step % self._log_interval == 0

    def log_scalar(self, tag: str, value: float):
        if self._writer is not None:
            self._writer.add_scalar(tag, value, self._step)

    def log_histogram(self, tag: str, tensor: torch.Tensor):
        if self._writer is not None:
            tensor_cpu = tensor.detach().cpu()
            if tensor_cpu.numel() > 0 and not torch.isnan(tensor_cpu).all() and not torch.isinf(tensor_cpu).all():
                self._writer.add_histogram(tag, tensor_cpu, self._step)

    def log_grad_norm(self, prefix: str, module: nn.Module):
        """只记录总梯度范数"""
        if not self.should_log or self._writer is None:
            return
        total_norm = 0.0
        param_count = 0
        for param in module.parameters():
            if param.requires_grad and param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
                param_count += 1
        if param_count > 0:
            self._writer.add_scalar(f'{prefix}/grad_norm', total_norm ** 0.5, self._step)

    def log_prompt_stats(self, prefix: str, prompt_tensor: torch.Tensor, weight_tensor: torch.Tensor = None):
        """【核心】统一记录Prompt关键指标：均值+权重+RGB/TIR权重"""
        if not self.should_log or self._writer is None:
            return
        with torch.no_grad():
            self._writer.add_scalar(f'{prefix}/prompt_mean', prompt_tensor.mean().item(), self._step)
            if weight_tensor is not None:
                w = weight_tensor.reshape(weight_tensor.shape[0], -1).mean(dim=1).mean().item()
                self._writer.add_scalar(f'{prefix}/avg_weight', w, self._step)
                if prefix in ['Mask', 'Consistency']:
                    self._writer.add_scalar(f'{prefix}/avg_rgb_weight', w, self._step)
                    self._writer.add_scalar(f'{prefix}/avg_tir_weight', 1 - w, self._step)
                self.log_histogram(f'{prefix}/weight_dist', weight_tensor)


class AttentionPooling(nn.Module):
    """
    自适应注意力池化模块
    使用轻量级注意力权重对序列token进行加权聚合

    公式: Prompt = Σ α_i * Prompt_i
    其中 α_i 由1层MLP生成
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] 输入序列特征
        Returns:
            [B, 1, C] 加权聚合后的全局特征
        """
        attn_weights = self.attention(x)  # [B, L, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # 在序列维度归一化
        pooled = (attn_weights * x).sum(dim=1, keepdim=True)  # [B, 1, C]
        return pooled


class MaskPromptGenerator(nn.Module):
    """
    Mask Prompt生成器 - 模态可靠性建模（v4: 目标感知+梯度-协方差联合版）

    【核心创新】Token级空间可靠性建模
    1. 空间梯度特征：计算token级空间邻域梯度，建模局部退化
    2. 局部协方差特征：3x3滑动窗口计算RGB-TIR局部相关性
    3. 联合预测：MLP融合两种特征，输出有区分度的可靠性权重

    【解决权重卡0.5问题】
    1. 目标感知监督：template_feat可选参数，计算与模板相似度的偏置
    2. 可学习温度系数：控制sigmoid锐度，替代硬clamp
    3. 模态差异增强：差异大的区域权重更有倾向性

    【双模式兼容】
    - standard模式：仅使用核心模态可靠性建模
    - fomaml模式：新增元任务感知的可学习模态偏移量

    输入：
        - rgb_feat: [B, N, C] RGB模态特征
        - tir_feat: [B, N, C] TIR模态特征
        - template_feat: [B, N_t, C] 模板特征（可选，用于目标感知偏置）
        - return_intermediate: 是否返回中间值
    输出：
        - mask_prompt: [B, num_prompt_tokens, C]
        - intermediates: dict（可选）
            - token_reliability: [B, N, 1] 可靠性权重（RGB权重）
            - tir_weight: [B, N, 1] TIR权重 = 1 - RGB权重
            - gradient_feat: [B, N, C_g] 空间梯度特征
            - covariance_feat: [B, N, C_c] 局部协方差特征
            - fused_feat: [B, N, C] 可靠性加权融合特征
    """
    def __init__(self, embed_dim: int, num_prompt_tokens: int = 8, mode: str = 'standard'):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.mode = mode
        assert mode in ['standard', 'fomaml'], f"[MaskPromptGenerator] mode必须是'standard'或'fomaml'，实际: {mode}"

        # ===== 核心创新1：空间梯度编码器 =====
        self.gradient_encoder = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2)
        )

        # ===== 核心创新2：局部协方差编码器 =====
        self.covariance_encoder = nn.Sequential(
            nn.LayerNorm(embed_dim * 2 + embed_dim),
            nn.Linear(embed_dim * 2 + embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2)
        )

        # ===== 核心创新3：联合可靠性预测头 =====
        joint_input_dim = embed_dim // 2 + embed_dim // 2 + embed_dim * 2
        self.joint_reliability_head = nn.Sequential(
            nn.LayerNorm(joint_input_dim),
            nn.Linear(joint_input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1)
        )

        # ===== 解决卡0.5：目标感知偏置网络 =====
        self.target_bias_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1)
        )

        # ===== 可学习位置编码 =====
        self.pos_encoding = nn.Parameter(torch.randn(1, num_prompt_tokens, embed_dim) * 0.01)

        # ===== 融合特征归一化（解决梯度消失） =====
        self.fused_norm = nn.LayerNorm(embed_dim)

        # ===== Prompt生成投影层 =====
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        if self.mode == 'fomaml':
            self.task_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.task_emb, std=0.01)

        self._init_weights()

    def _init_weights(self):
        """初始化所有线性层和偏置"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _compute_spatial_gradient(self, feat: torch.Tensor) -> torch.Tensor:
        """
        【核心创新】计算Token级空间邻域梯度

        Args:
            feat: [B, N, C] 输入特征

        Returns:
            spatial_grad: [B, N, C] 空间梯度特征
        """
        B, N, C = feat.shape
        feat_pad = F.pad(feat, (0, 0, 1, 0), mode='replicate')
        spatial_grad = feat_pad[:, 1:, :] - feat_pad[:, :-1, :]
        return spatial_grad

    def _compute_local_covariance(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                                   window_size: int = 3) -> torch.Tensor:
        """
        【核心创新】计算Token级局部协方差（3x3滑动窗口）

        Args:
            rgb_feat: [B, N, C] RGB模态特征
            tir_feat: [B, N, C] TIR模态特征
            window_size: 滑动窗口大小

        Returns:
            local_cov: [B, N, C] 局部协方差特征
        """
        B, N, C = rgb_feat.shape
        half_w = window_size // 2

        local_mean_rgb = F.avg_pool1d(rgb_feat.transpose(1, 2), window_size, stride=1, padding=half_w).transpose(1, 2)
        local_mean_tir = F.avg_pool1d(tir_feat.transpose(1, 2), window_size, stride=1, padding=half_w).transpose(1, 2)

        dev_rgb = rgb_feat - local_mean_rgb
        dev_tir = tir_feat - local_mean_tir

        cov_raw = dev_rgb * dev_tir
        local_cov = F.avg_pool1d(cov_raw.transpose(1, 2), window_size, stride=1, padding=half_w).transpose(1, 2)

        # 安全校验：软clamp防止数值爆炸，保留幅值信息（L2归一化会丢失幅值区分度）
        local_cov = local_cov.clamp(-10.0, 10.0)

        return local_cov

    def forward(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                template_feat: torch.Tensor = None,
                return_intermediate: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            rgb_feat: [B, N, C] RGB模态特征
            tir_feat: [B, N, C] TIR模态特征
            template_feat: [B, N_t, C] 模板特征（可选，用于目标感知偏置）
            return_intermediate: 是否返回中间值（供可视化使用）

        Returns:
            mask_prompt: [B, num_prompt_tokens, C]
            intermediates: dict（仅当return_intermediate=True时返回）
        """
        B, N, C = rgb_feat.shape

        # ===== Step1: 计算原始拼接特征 =====
        combined = torch.cat([rgb_feat, tir_feat], dim=-1)

        # ===== Step2: 核心创新1 - 空间梯度特征 =====
        rgb_gradient = self._compute_spatial_gradient(rgb_feat)
        tir_gradient = self._compute_spatial_gradient(tir_feat)
        gradient_input = torch.cat([rgb_feat, tir_feat, rgb_gradient, tir_gradient], dim=-1)
        gradient_feat = self.gradient_encoder(gradient_input)

        # ===== Step3: 核心创新2 - 局部协方差特征 =====
        local_cov = self._compute_local_covariance(rgb_feat, tir_feat)
        cov_input = torch.cat([combined, local_cov], dim=-1)
        covariance_feat = self.covariance_encoder(cov_input)

        # ===== Step4: 核心创新3 - 联合可靠性预测 =====
        joint_input = torch.cat([gradient_feat, covariance_feat, combined], dim=-1)
        reliability_logits = self.joint_reliability_head(joint_input).squeeze(-1)

        # ===== Step5: tanh残差权重（终极修复：压缩动态范围） =====
        # 【终极修复1】更严格的logits clamp，防止极端值
        reliability_logits = torch.clamp(reliability_logits, min=-3.0, max=3.0)
        # 【终极修复2】压缩权重动态范围：从[0.1, 0.9] → [0.25, 0.75]
        # 原来的：0.5 + 0.4*tanh(logits) → 范围[0.1, 0.9]（模型会顶到上限）
        # 现在的：0.5 + 0.25*tanh(logits) → 范围[0.25, 0.75]（给模型更多区分度空间）
        token_reliability = 0.5 + 0.25 * torch.tanh(reliability_logits)

        # ===== Step6: 目标感知偏置（终极修复：打破RGB偏向） =====
        target_bias = None
        if template_feat is not None:
            # 【终极修复3】用双模态融合特征计算相似度，打破RGB偏向
            # 先用基础权重计算初步融合特征
            fused_feat_prelim = token_reliability.unsqueeze(-1) * rgb_feat + (1 - token_reliability.unsqueeze(-1)) * tir_feat
            template_mean = template_feat.mean(dim=1, keepdim=True)
            # 用融合特征计算相似度
            fused_norm = F.normalize(fused_feat_prelim, dim=-1, eps=1e-8)
            template_norm = F.normalize(template_mean.repeat(1, N, 1), dim=-1, eps=1e-8)
            target_sim = F.cosine_similarity(fused_norm, template_norm, dim=-1)
            modal_diff = torch.abs(rgb_feat - tir_feat).mean(dim=-1)
            modal_diff = F.normalize(modal_diff, dim=1)
            # 【终极修复4】偏置输入改成融合特征
            target_bias = torch.sigmoid(self.target_bias_net(fused_feat_prelim).squeeze(-1))
            # 【终极修复5】更严格的偏置约束：[0.4, 0.6]
            target_bias = torch.clamp(target_bias, min=0.4, max=0.6)
            # 【终极修复6】减小logits_bias系数并做clamp
            logits_bias = (target_sim * modal_diff) * 0.5
            logits_bias = torch.clamp(logits_bias, min=-2.0, max=2.0)
            token_reliability = 0.5 + 0.25 * torch.tanh(reliability_logits + logits_bias)

        # 【终极修复7】最终安全钳制：[0.25, 0.75]
        token_reliability = torch.clamp(token_reliability, min=0.25, max=0.75)

        # ===== Step7: 可靠性加权融合 + LayerNorm =====
        fused_feat = token_reliability.unsqueeze(-1) * rgb_feat + (1 - token_reliability.unsqueeze(-1)) * tir_feat
        fused_feat = self.fused_norm(fused_feat)  # 归一化融合特征，稳定梯度流

        # ===== Step8: 全局池化生成Prompt =====
        global_feat = fused_feat.mean(dim=1, keepdim=True)

        if self.mode == 'fomaml':
            global_feat = global_feat + self.task_emb

        # ===== Step9: 生成多样化Prompt =====
        mask_prompt = global_feat.repeat(1, self.num_prompt_tokens, 1)
        mask_prompt = mask_prompt + self.pos_encoding
        mask_prompt = self.proj(mask_prompt)

        assert not torch.isnan(mask_prompt).any(), f"[MaskPromptGenerator] 输出出现NaN！"
        assert not torch.isinf(mask_prompt).any(), f"[MaskPromptGenerator] 输出出现Inf！"

        # ===== 可视化记录（保留RGB和TIR权重） =====
        vis = PromptVisualizer.get()
        if self.training:
            vis.log_prompt_stats('Mask', mask_prompt, token_reliability.unsqueeze(-1))
            vis.log_grad_norm('Mask', self)

        if return_intermediate:
            return mask_prompt, {
                'token_reliability': token_reliability.unsqueeze(-1),
                'tir_weight': (1 - token_reliability).unsqueeze(-1),
                'gradient_feat': gradient_feat,
                'covariance_feat': covariance_feat,
                'fused_feat': fused_feat,
                'target_bias': target_bias,
            }
        # 【关键修改】即使return_intermediate=False，也返回token_reliability用于计算正则化损失
        return mask_prompt, {'token_reliability': token_reliability.unsqueeze(-1)}


class ConsistencyPromptGenerator(nn.Module):
    """
    Consistency Prompt生成器 - 跨模态一致性建模（v7: 显式双峰引导版）

    【v7核心改进 — 解决"无双峰"问题】
    原v4-v6问题：
      1. MLP(concat(rgb,tir,grad,cov)) → tanh → 单标量权重
         → 所有token收到相似的梯度信号 → 自然坍塌到单峰
      2. 正则化（方差/熵）无法改变网络结构本身的局限性
      3. "一致性"是全局概念 → MLP倾向于输出相似判断

    v7解决方案：
      1. 【位置引导双峰】可学习pos_bias [1,N,2]，初始化为交替模式
         → 一半位置初始偏向RGB，另一半偏向TIR → 天然双峰种子
      2. 【Gumbel-Softmax】替代tanh，输出2维选择概率[选RGB, 选TIR]
         → 天然产生离散化/双峰分布
      3. 【温度退火】temperature: 5.0→0.5
         → 前期软探索 → 后期硬选择（明确的双峰）
      4. 【移除正则化依赖】不再需要方差/熵正则化

    输入：
        - rgb_feat: [B, N, C] RGB模态特征
        - tir_feat: [B, N, C] TIR模态特征
        - template_feat: [B, N_t, C] 模板特征（可选）
        - return_intermediate: 是否返回中间值
    输出：
        - consistency_prompt: [B, num_prompt_tokens, C]
        - intermediates: dict
            - token_consistency: [B, N, 1] RGB权重（=选择RGB的概率）
    """
    def __init__(self, embed_dim: int, num_prompt_tokens: int = 8, mode: str = 'standard',
                 num_tokens: int = 256):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.mode = mode
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        assert mode in ['standard', 'fomaml'], f"[ConsistencyPromptGenerator] mode必须是'standard'或'fomaml'，实际: {mode}"

        # ===== 空间梯度编码器 =====
        self.gradient_encoder = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2)
        )

        # ===== 局部协方差编码器 =====
        self.covariance_encoder = nn.Sequential(
            nn.LayerNorm(embed_dim * 2 + embed_dim),
            nn.Linear(embed_dim * 2 + embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2)
        )

        # ===== v7核心1: 联合预测头（输出2维logits用于Gumbel-Softmax）=====
        joint_input_dim = embed_dim // 2 + embed_dim // 2 + embed_dim * 2
        self.joint_consistency_head = nn.Sequential(
            nn.LayerNorm(joint_input_dim),
            nn.Linear(joint_input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 2)  # 【v7】输出2维：[logit_rgb, logit_tir]
        )

        # ===== v11核心: 位置引导偏置（基础值±0.5，动态系数调整有效强度）=====
        # 【v10问题】pos_bias=±0.5太弱 → 双峰种子被MLP平均梯度淹没
        # 【v11修复】动态系数：前1000步×6.0=±3.0强制双峰 → 后衰减到×1.0=±0.5让MLP主导
        pos_bias_init = torch.zeros(1, num_tokens, 2)
        for i in range(num_tokens):
            if i % 2 == 0:
                pos_bias_init[0, i, 0] = 0.5
                pos_bias_init[0, i, 1] = -0.5
            else:
                pos_bias_init[0, i, 0] = -0.5
                pos_bias_init[0, i, 1] = 0.5
        self.register_buffer('pos_bias', pos_bias_init)

        # ===== v11核心: 温度0.8，分布更清晰 =====
        self.register_buffer('temperature', torch.tensor(0.8))

        # ===== v11核心: 步数计数器（用于动态pos_bias系数）=====
        self._global_step = 0

        # ===== Prompt相关 =====
        self.pos_encoding = nn.Parameter(torch.randn(1, num_prompt_tokens, embed_dim) * 0.01)
        self.fused_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        if self.mode == 'fomaml':
            self.task_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.task_emb, std=0.01)

        self._init_weights()

    def _init_weights(self):
        """初始化所有线性层和偏置"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _compute_spatial_gradient(self, feat: torch.Tensor) -> torch.Tensor:
        """
        【v5修复】计算Token级空间邻域梯度 - 增加幅度限制

        v5改进：深层特征差分可能产生极端值，加幅度钳制防止梯度爆炸
        """
        B, N, C = feat.shape

        # 【v5修复】输入安全检测
        if torch.isnan(feat).any():
            return torch.zeros(B, N, C, device=feat.device, dtype=feat.dtype)

        # 计算相邻token的差分作为空间梯度（类似Sobel的x方向）
        feat_pad = F.pad(feat, (0, 0, 1, 0), mode='replicate')  # [B, N+1, C]
        spatial_grad = feat_pad[:, 1:, :] - feat_pad[:, :-1, :]  # [B, N, C]

        # 【v5修复】幅度限制：防止深层特征差分产生极端值
        spatial_grad = torch.clamp(spatial_grad, min=-5.0, max=5.0)

        return spatial_grad

    def _compute_local_covariance(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                                   window_size: int = 3) -> torch.Tensor:
        """
        【v5修复】计算Token级局部协方差（3x3滑动窗口）- 增强数值稳定性

        v5改进：
          1. 输入NaN/Inf检测：防止上游传播的异常值
          2. 分子clamp：归一化前钳制极端值
          3. 安全除法：分母+分子双重保护
          4. 输出NaN替换：用零向量替代异常输出（而非崩溃）
        """
        B, N, C = rgb_feat.shape
        half_w = window_size // 2

        # 【v5修复1】输入安全检测：如果输入有NaN，返回零向量避免崩溃
        if torch.isnan(rgb_feat).any() or torch.isnan(tir_feat).any():
            return torch.zeros(B, N, C, device=rgb_feat.device, dtype=rgb_feat.dtype)
        if torch.isinf(rgb_feat).any() or torch.isinf(tir_feat).any():
            rgb_feat = torch.clamp(rgb_feat, min=-10.0, max=10.0)
            tir_feat = torch.clamp(tir_feat, min=-10.0, max=10.0)

        # 计算每个位置的局部均值
        local_mean_rgb = F.avg_pool1d(rgb_feat.transpose(1, 2), window_size, stride=1, padding=half_w).transpose(1, 2)
        local_mean_tir = F.avg_pool1d(tir_feat.transpose(1, 2), window_size, stride=1, padding=half_w).transpose(1, 2)

        # 计算局部偏差
        dev_rgb = rgb_feat - local_mean_rgb  # [B, N, C]
        dev_tir = tir_feat - local_mean_tir   # [B, N, C]

        # 局部协方差 = E[(X-E[X])(Y-E[Y])]，用滑动窗口近似
        cov_raw = dev_rgb * dev_tir  # [B, N, C] 逐通道协方差
        local_cov = F.avg_pool1d(cov_raw.transpose(1, 2), window_size, stride=1, padding=half_w).transpose(1, 2)

        # 【v5修复2】分子先clamp再归一化，防止极端值放大
        local_cov = torch.clamp(local_cov, min=-5.0, max=5.0)

        # 【v5修复3】安全归一化：分母+eps + 分子双重保护
        cov_norm = local_cov.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        local_cov = local_cov / cov_norm

        # 【v5修复4】输出最终安全检查
        if torch.isnan(local_cov).any() or torch.isinf(local_cov).any():
            local_cov = torch.zeros(B, N, C, device=rgb_feat.device, dtype=rgb_feat.dtype)

        return local_cov

    def forward(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                template_feat: torch.Tensor = None,
                return_intermediate: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        v7前向传播 — 显式双峰引导 + Gumbel-Softmax

        核心流程:
          1. 计算梯度/协方差特征（保留v4-v6的物理特征）
          2. MLP → 2维logits [B,N,2] = [logit_rgb, logit_tir]
          3. +位置偏置 pos_bias [1,N,2]（交替初始化的双峰种子）
          4. Gumbel-Softmax(temperature) → 概率 [B,N,2]
          5. token_consistency = prob[:, :, 0] (RGB权重)
          6. 加权融合 → 全局池化 → Prompt

        Args:
            rgb_feat: [B, N, C] RGB模态特征
            tir_feat: [B, N, C] TIR模态特征
            template_feat: [B, N_t, C] （v7不使用）
            return_intermediate: 是否返回中间值

        Returns:
            consistency_prompt: [B, num_prompt_tokens, C]
            intermediates: dict
        """
        B, N, C = rgb_feat.shape

        # 【安全】输入NaN检测
        if torch.isnan(rgb_feat).any() or torch.isnan(tir_feat).any():
            fallback_prompt = self.pos_encoding.repeat(B, 1, 1)
            return fallback_prompt, {'token_consistency': 0.5 * torch.ones(B, N, 1, device=rgb_feat.device)}

        # ===== Step1-3: 特征计算（与v4-v6相同）=====
        combined = torch.cat([rgb_feat, tir_feat], dim=-1)

        rgb_gradient = self._compute_spatial_gradient(rgb_feat)
        tir_gradient = self._compute_spatial_gradient(tir_feat)
        gradient_input = torch.cat([rgb_feat, tir_feat, rgb_gradient, tir_gradient], dim=-1)
        gradient_feat = self.gradient_encoder(gradient_input)
        if torch.isnan(gradient_feat).any() or torch.isinf(gradient_feat).any():
            gradient_feat = torch.zeros_like(gradient_feat)

        local_cov = self._compute_local_covariance(rgb_feat, tir_feat)
        cov_input = torch.cat([combined, local_cov], dim=-1)
        covariance_feat = self.covariance_encoder(cov_input)
        if torch.isnan(covariance_feat).any() or torch.isinf(covariance_feat).any():
            covariance_feat = torch.zeros_like(covariance_feat)

        # ===== Step4: 联合预测头 → 2维logits =====
        joint_input = torch.cat([gradient_feat, covariance_feat, combined], dim=-1)
        consistency_logits = self.joint_consistency_head(joint_input)  # [B, N, 2]

        if torch.isnan(consistency_logits).any() or torch.isinf(consistency_logits).any():
            consistency_logits = torch.zeros(B, N, 2, device=rgb_feat.device)
        # 【v11核心】MLP输出范围[-3,+3]，防止太极端
        consistency_logits = torch.clamp(consistency_logits, min=-3.0, max=3.0)

        # ===== Step5: 【v11核心】动态pos_bias系数 =====
        # 前1000步：系数=6.0 → 有效pos_bias=±3.0，强制双峰
        # 1000-5000步：线性衰减到1.0 → 有效pos_bias从±3.0衰减到±0.5
        # 5000步后：系数=1.0 → MLP完全主导
        if self.training:
            self._global_step += 1
        step = self._global_step

        if step < 1000:
            pos_bias_coeff = 6.0
        elif step < 5000:
            pos_bias_coeff = 6.0 - (step - 1000) * (5.0 / 4000)
        else:
            pos_bias_coeff = 1.0

        # 位置引导偏置适配
        if self.pos_bias.size(1) != N:
            if self.pos_bias.size(1) > N:
                pos_b = self.pos_bias[:, :N, :]
            else:
                pad = torch.zeros(1, N - self.pos_bias.size(1), 2, device=self.pos_bias.device)
                pos_b = torch.cat([self.pos_bias, pad], dim=1)
        else:
            pos_b = self.pos_bias

        # 动态加权的pos_bias
        biased_logits = consistency_logits + pos_bias_coeff * pos_b  # [B,N,2]

        # ===== Step6: Gumbel-Softmax（温度=0.8）=====
        temp = self.temperature.item()

        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(biased_logits) + 1e-20) + 1e-20)
            gumbel_logits = (biased_logits + gumbel_noise) / temp
            probs = F.softmax(gumbel_logits, dim=-1)
        else:
            probs = F.softmax(biased_logits / temp, dim=-1)

        if torch.isnan(probs).any():
            probs = torch.ones(B, N, 2, device=rgb_feat.device) * 0.5

        token_consistency = probs[:, :, 0:1]

        # ===== Step7: 一致性加权融合 =====
        fused_feat = token_consistency * rgb_feat + (1 - token_consistency) * tir_feat
        fused_feat = self.fused_norm(fused_feat)

        if torch.isnan(fused_feat).any():
            fused_feat = (rgb_feat + tir_feat) / 2.0
            fused_feat = self.fused_norm(fused_feat)

        # ===== Step8-9: Prompt生成（不变）=====
        global_feat = fused_feat.mean(dim=1, keepdim=True)

        if self.mode == 'fomaml':
            global_feat = global_feat + self.task_emb

        consistency_prompt = global_feat.repeat(1, self.num_prompt_tokens, 1)
        consistency_prompt = consistency_prompt + self.pos_encoding
        consistency_prompt = self.proj(consistency_prompt)

        if torch.isnan(consistency_prompt).any() or torch.isinf(consistency_prompt).any():
            consistency_prompt = self.pos_encoding.repeat(B, 1, 1)
            token_consistency = 0.5 * torch.ones(B, N, 1, device=rgb_feat.device)

        # ===== 【v11核心】计算base_p：使用动态pos_bias系数 =====
        # base_p = softmax(pos_bias_coeff * pos_bias / temp)[:, :, 0]
        base_p = F.softmax(pos_bias_coeff * pos_b / temp, dim=-1)[:, :, 0:1]  # [B, N, 1]

        # 可视化记录
        vis = PromptVisualizer.get()
        if self.training:
            vis.log_prompt_stats('Consistency', consistency_prompt, token_consistency)
            vis.log_grad_norm('Consistency', self)
            try:
                from lib.utils.tensorboard_utils import TensorboardSummary
                tb_logger = TensorboardSummary.get_instance()
                if tb_logger.writer is not None:
                    tb_logger.writer.add_scalar('Consistency/gumbel_temp', temp, vis.step_count)
                    tb_logger.writer.add_scalar('Consistency/pos_bias_coeff', pos_bias_coeff, vis.step_count)
            except Exception:
                pass

        if return_intermediate:
            return consistency_prompt, {
                'token_consistency': token_consistency,
                'gradient_feat': gradient_feat,
                'covariance_feat': covariance_feat,
                'fused_feat': fused_feat,
                'target_bias': None,
                'base_p': base_p,
                'pos_bias_coeff': pos_bias_coeff,  # 【v11新增】用于监控
            }
        return consistency_prompt, {
            'token_consistency': token_consistency,
            'base_p': base_p,
            'pos_bias_coeff': pos_bias_coeff,
        }


class TemporalPromptGenerator(nn.Module):
    """
    Temporal Prompt生成器 - v7架构（彻底修复正反馈Bug + 机制解耦）

    【v7核心改进】
    1. 【修复正反馈Bug】用固定放大系数+LayerNorm替代自适应放大
       - 固定放大3倍，稳定可靠
       - LayerNorm自动适配不同视频的信号强度
       - 彻底杜绝帧差无限放大的正反馈循环
    
    2. 【机制解耦】三个机制各司其职
       - 运动门控：仅控制时序Prompt的整体权重，范围[0.2, 1.0]
       - 偏移放大：固定2倍，不再随运动变化
       - 运动偏置：固定强度0.15，仅方向由token_motion决定
    
    3. 【简化权重生成】两阶段逻辑
       - 第一阶段：MLP生成基础权重 sigmoid(logits)
       - 第二阶段：运动调制微调，运动大的token更有区分度
    
    4. 【元学习深度融合】task_emb调制核心参数
       - 调制帧差编码MLP的偏置
       - 调制运动门控的阈值

    输入：
      - rgb_feat/tir_feat: [B, N, C] 双模态特征
      - token_consistency: [B, N] 一致性权重（用于对齐特征）
    输出：
      - temporal_prompt: [B, num_prompt_tokens, C]
      - temporal_weight: [B, N] 时序权重
      - intermediates: dict（包含global_motion用于正则化）
    """
    def __init__(self, embed_dim: int, num_prompt_tokens: int = 8, hidden_dim: int = 256, mode: str = 'standard'):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.mode = mode
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        assert mode in ['standard', 'fomaml'], f"[TemporalPromptGenerator] mode必须是'standard'或'fomaml'，实际: {mode}"

        # ===== v7核心1: 帧差专用LayerNorm（替代自适应放大）=====
        # 彻底解决正反馈问题：固定放大+LayerNorm自动归一化
        self.frame_diff_norm = nn.LayerNorm(embed_dim)

        # ===== v7核心2: 帧差编码MLP（元学习静态参数）=====
        self.frame_diff_encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        # ===== v7核心3: 时序Prompt投影层（元学习静态参数）=====
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # ===== 可学习位置编码（元学习静态参数）=====
        self.pos_encoding = nn.Parameter(torch.randn(1, num_prompt_tokens, embed_dim) * 0.01)

        # ===== v7核心4: 元学习深度融合 =====
        if self.mode == 'fomaml':
            # task_emb调制帧差编码MLP
            self.task_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # 调制运动门控阈值
            self.gate_threshold_emb = nn.Parameter(torch.zeros(1, 1))
            nn.init.normal_(self.task_emb, std=0.01)
            nn.init.zeros_(self.gate_threshold_emb)

        # ===== v7核心5: 动态缓存（非元学习，实时维护）=====
        self._prev_aligned_feat_cache = None
        self._temporal_logits_buffer = None
        self.buffer_size = 3

        # ===== v7核心6: 固定超参数（不再自适应）=====
        self.fixed_motion_scale = 3.0  # 固定放大系数
        self.fixed_amplification = 2.0  # 固定偏移放大倍数
        self.fixed_bias_strength = 0.15  # 固定运动偏置强度

        self._init_weights()

    def _init_weights(self):
        """初始化所有线性层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_cache(self):
        """重置时序缓存（每个视频首帧调用）"""
        self._prev_aligned_feat_cache = None
        self._temporal_logits_buffer = None

    def forward(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                prev_state: Optional[torch.Tensor] = None,
                prev_features: Optional[torch.Tensor] = None,
                token_consistency: Optional[torch.Tensor] = None,
                fused_feat: Optional[torch.Tensor] = None,
                return_intermediate: bool = False) -> tuple:
        """
        Args:
            rgb_feat: [B, N, C] RGB模态特征
            tir_feat: [B, N, C] TIR模态特征
            prev_state: 兼容旧接口，不再使用
            prev_features: 兼容旧接口，不再使用
            token_consistency: [B, N] 当前帧跨模态一致性权重（用于对齐特征）
            fused_feat: [B, N, C] 融合特征（用于生成Prompt）
            return_intermediate: 是否返回中间值（供可视化使用）
        Returns:
            temporal_prompt: [B, num_prompt_tokens, C]
            temporal_weight: [B, N] 时序权重
            intermediates: dict（包含global_motion用于正则化）
        """
        B, N, C = rgb_feat.shape

        # ===== v7 Step1: 计算对齐特征 =====
        if token_consistency is None:
            token_consistency = torch.ones(B, N, device=rgb_feat.device, dtype=rgb_feat.dtype) * 0.5
        
        # 安全校验
        if torch.isnan(token_consistency).any():
            token_consistency = torch.ones(B, N, device=rgb_feat.device, dtype=rgb_feat.dtype) * 0.5
        token_consistency = torch.clamp(token_consistency, min=0.01, max=0.99)
        
        # 对齐特征: w*rgb + (1-w)*tir
        aligned_feat = token_consistency.unsqueeze(-1) * rgb_feat + (1 - token_consistency).unsqueeze(-1) * tir_feat

        # ===== v7 Step2: 计算帧差 =====
        if self._prev_aligned_feat_cache is not None and self._prev_aligned_feat_cache.shape == (B, N, C):
            frame_diff = aligned_feat - self._prev_aligned_feat_cache
        else:
            frame_diff = torch.zeros(B, N, C, device=rgb_feat.device, dtype=rgb_feat.dtype)

        # ===== v7 Step3: 固定放大 + LayerNorm（彻底解决正反馈Bug）=====
        # 【关键改进】不再使用自适应放大，而是固定放大+LayerNorm
        frame_diff_scaled = self.fixed_motion_scale * frame_diff
        frame_diff_normed = self.frame_diff_norm(frame_diff_scaled)  # LayerNorm自动归一化

        # 计算global_motion（仅用于门控，不再用于放大）
        global_motion = torch.mean(torch.abs(frame_diff), dim=[1, 2], keepdim=True)  # [B, 1, 1]

        # 更新缓存
        self._prev_aligned_feat_cache = aligned_feat.detach().clone()

        # ===== v7 Step4: 帧差编码 → 时序logits =====
        # 元学习调制：task_emb影响编码过程
        if self.mode == 'fomaml':
            # 用task_emb调制帧差特征
            frame_diff_modulated = frame_diff_normed + 0.1 * self.task_emb
        else:
            frame_diff_modulated = frame_diff_normed

        temporal_logits = self.frame_diff_encoder(frame_diff_modulated).squeeze(-1)  # [B, N]
        temporal_logits = torch.clamp(temporal_logits, min=-5.0, max=5.0)

        # ===== v7 Step5: 滑动窗口平滑 =====
        if self.training:
            current_shape = (B, N)
            if self._temporal_logits_buffer is None or len(self._temporal_logits_buffer) == 0:
                self._temporal_logits_buffer = [temporal_logits.detach().clone()]
            else:
                first_buffer_shape = tuple(self._temporal_logits_buffer[0].shape)
                if first_buffer_shape == current_shape:
                    self._temporal_logits_buffer.append(temporal_logits.detach().clone())
                    if len(self._temporal_logits_buffer) > self.buffer_size:
                        self._temporal_logits_buffer.pop(0)
                else:
                    self._temporal_logits_buffer = [temporal_logits.detach().clone()]

        # 加权平均
        if self._temporal_logits_buffer is not None and len(self._temporal_logits_buffer) > 0:
            all_same_shape = all(tuple(buf.shape) == (B, N) for buf in self._temporal_logits_buffer)
            if all_same_shape:
                weights = [0.5 ** (len(self._temporal_logits_buffer) - i) for i in range(len(self._temporal_logits_buffer))]
                weights = torch.tensor(weights, device=rgb_feat.device, dtype=rgb_feat.dtype)
                weights = weights / weights.sum()
                temporal_logits_smoothed = sum(w * buf for w, buf in zip(weights, self._temporal_logits_buffer))
            else:
                temporal_logits_smoothed = temporal_logits
        else:
            temporal_logits_smoothed = temporal_logits

        # ===== v7 Step6: 简化的两阶段权重生成 =====
        # 第一阶段：MLP生成基础权重
        base_weight = torch.sigmoid(temporal_logits_smoothed)  # [B, N], 范围[0, 1]

        # 第二阶段：运动调制微调
        # 计算每个token的运动强度
        token_motion = torch.mean(torch.abs(frame_diff.detach()), dim=-1)  # [B, N]
        token_motion_max = token_motion.max(dim=-1, keepdim=True)[0] + 1e-8
        token_motion_norm = token_motion / token_motion_max  # [B, N], 归一化到[0,1]

        # 运动调制：运动大的token向两侧微调，运动小的保持不变
        # 【关键简化】固定强度，不再随motion_gate变化
        motion_modulation = (token_motion_norm - 0.5) * self.fixed_bias_strength  # [-0.075, 0.075]
        temporal_weight = base_weight + motion_modulation  # [B, N]

        # 硬约束：限制在合理范围
        temporal_weight = torch.clamp(temporal_weight, min=0.15, max=0.85)

        # ===== v7 Step7: 运动门控（仅控制Prompt整体权重）=====
        # 【关键解耦】门控仅影响最终Prompt，不影响权重生成过程
        if self.mode == 'fomaml':
            # 元学习调制门控阈值
            gate_threshold = 0.1 + self.gate_threshold_emb.squeeze(-1)  # [B, 1]
        else:
            gate_threshold = 0.1

        motion_gate = 0.2 + 0.8 * torch.sigmoid(5.0 * (global_motion.squeeze(-1) - gate_threshold))  # [B, 1]

        # ===== v7 Step8: 生成时序Prompt =====
        if fused_feat is None:
            fused_feat = aligned_feat
        
        weighted_feat = temporal_weight.unsqueeze(-1) * fused_feat  # [B, N, C]

        # 全局池化 + 投影
        temporal_global = weighted_feat.mean(dim=1, keepdim=True)  # [B, 1, C]
        temporal_base = self.temporal_proj(temporal_global)  # [B, 1, C]

        # 扩展到num_prompt_tokens个token
        temporal_prompt = temporal_base.expand(-1, self.num_prompt_tokens, -1)  # [B, num_prompt_tokens, C]
        temporal_prompt = temporal_prompt + self.pos_encoding

        # ===== v7 Step9: 应用运动门控（仅控制Prompt整体权重）=====
        # 【关键解耦】门控乘在最终Prompt上，不影响权重生成
        temporal_prompt = motion_gate.unsqueeze(-1) * temporal_prompt

        # 安全校验
        if torch.isnan(temporal_prompt).any():
            temporal_prompt = torch.zeros_like(temporal_prompt)
        if torch.isinf(temporal_prompt).any():
            temporal_prompt = torch.clamp(temporal_prompt, min=-10.0, max=10.0)

        # ===== 可视化记录 =====
        vis = PromptVisualizer.get()
        if self.training:
            vis.log_prompt_stats('Temporal', temporal_prompt, temporal_weight)
            vis.log_grad_norm('Temporal', self)
            # v7：记录关键指标
            vis.log_scalar('Temporal/global_motion', global_motion.mean().item())
            vis.log_scalar('Temporal/motion_gate', motion_gate.mean().item())
            vis.log_scalar('Temporal/temporal_weight_mean', temporal_weight.mean().item())
            vis.log_scalar('Temporal/weight_deviation', (temporal_weight - 0.5).abs().mean().item())
            vis.log_scalar('Temporal/motion_modulation_range', (token_motion_norm.max() - token_motion_norm.min()).item())

        if return_intermediate:
            intermediates = {
                'temporal_weight': temporal_weight,
                'global_motion': global_motion,
                'motion_gate': motion_gate,
                'token_motion_norm': token_motion_norm,
            }
            return temporal_prompt, temporal_weight, intermediates

        return temporal_prompt, temporal_weight, None


class GatedFusion(nn.Module):
    """
    门控融合网络 - 平衡多种Prompt的贡献（优化版）

    【核心优化】
    1. 单类型Prompt时直接返回，避免不必要的门控计算
    2. 多类型时使用软门控加权融合
    """
    def __init__(self, num_prompt_types: int):
        super().__init__()
        self.num_prompt_types = num_prompt_types

        if num_prompt_types > 1:
            self.gate_mlp = nn.Sequential(
                nn.Linear(num_prompt_types, num_prompt_types),
                nn.Sigmoid()
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, prompt_list: list) -> torch.Tensor:
        if len(prompt_list) == 0:
            return None

        prompt_list = [p for p in prompt_list if p is not None]

        if len(prompt_list) == 0:
            return None

        # 【优化】单类型Prompt直接返回，避免门控引入噪声
        if len(prompt_list) == 1:
            return prompt_list[0]

        concatenated = torch.cat(prompt_list, dim=1)

        B, total_tokens, C = concatenated.shape
        num_types = len(prompt_list)
        tokens_per_type = total_tokens // num_types

        type_features = []
        for i, prompt in enumerate(prompt_list):
            type_features.append(prompt.mean(dim=1, keepdim=True))

        type_features_cat = torch.cat(type_features, dim=1).mean(dim=-1)

        gate_values = self.gate_mlp(type_features_cat)

        expanded_gates = []
        for i, gate in enumerate(gate_values.unbind(dim=1)):
            expanded_gates.append(gate.unsqueeze(-1).unsqueeze(-1).expand(-1, tokens_per_type, C))

        fused_prompt = torch.cat(expanded_gates, dim=1) * concatenated

        return fused_prompt


class CrossAttentionModulation(nn.Module):
    """
    极简交叉注意力调制模块 - SCI论文可用

    【设计原则】
    1. 保留注意力机制作为SCI创新点
    2. 去掉所有冗余后处理（无interpolate、无双门控、无FeatureNorm）
    3. 核心思路：prompt作为Query attend x的Key/Value → 全局调制信号 → 广播加法

    【公式】
    Q = LN(prompt), K = LN(x), V = x
    attn_out = Softmax(Q @ K^T / sqrt(d)) @ V  # [B, L_p, C]
    global_signal = MeanPooling(attn_out)         # [B, 1, C]
    output = x + alpha * global_signal            # [B, L, C]

    输入：prompt [B, L_p, C] + x [B, L, C]
    输出：[B, L, C] 调制后的主干特征
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Pre-LN: 在注意力前做LayerNorm，稳定训练
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)

        # QK投影（V直接用x，无需投影）
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        # 轻量输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 可学习调制强度（增大初始值和上限，确保梯度能传播）
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, prompt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        L_p = prompt.shape[1]

        assert B == prompt.shape[0], f"[CrossAttentionModulation] batch维度不匹配！x: {B}, prompt: {prompt.shape[0]}"
        assert C % self.num_heads == 0, f"[CrossAttentionModulation] embed_dim必须能被num_heads整除！C={C}, num_heads={self.num_heads}"

        # ===== Pre-LN =====
        q = self.norm_q(prompt)
        k = self.norm_k(x)
        v = x

        # ===== QK投影 =====
        Q = self.q_proj(q)
        K = self.k_proj(k)

        # Reshape for multi-head attention
        Q = Q.view(B, L_p, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ===== 数值稳定性保护 =====
        Q = torch.clamp(Q, min=-10.0, max=10.0)
        K = torch.clamp(K, min=-10.0, max=10.0)

        # ===== 交叉注意力计算 =====
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        # 安全校验：softmax前减去最大值，防止指数溢出导致NaN
        attn = attn - attn.max(dim=-1, keepdim=True).values.detach()
        attn = attn.softmax(dim=-1)

        # 聚合到prompt级别 [B, L_p, C]
        attn_out = (attn @ V).permute(0, 2, 1, 3).reshape(B, L_p, C)

        # 轻量投影
        attn_out = self.out_proj(attn_out)

        # ===== 关键：Mean Pooling得到全局信号 [B, 1, C] =====
        global_signal = attn_out.mean(dim=1, keepdim=True)

        # 可学习调制强度（增大上限到0.3，确保Prompt有足够影响）
        alpha = torch.sigmoid(self.alpha) * 0.3

        # 加法调制（广播到所有L个token）
        modulated_x = x + alpha * global_signal

        assert not torch.isnan(modulated_x).any(), f"[CrossAttentionModulation] 输出出现NaN！"
        assert not torch.isinf(modulated_x).any(), f"[CrossAttentionModulation] 输出出现Inf！"

        return modulated_x


class BasePromptGenerator(nn.Module):
    """
    Base Prompt生成器 - 固定可学习Prompt

    功能：生成固定的可学习Prompt，用于标准训练模式

    输入：无
    输出：Base Prompt [B, num_prompt_tokens, C]
    """
    def __init__(self, embed_dim: int, num_prompt_tokens: int = 8):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens

        self.base_prompt = nn.Parameter(torch.randn(1, num_prompt_tokens, embed_dim))

        nn.init.xavier_uniform_(self.base_prompt.data)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            batch_size: 批次大小
            device: 设备
        Returns:
            [B, num_prompt_tokens, C] Base Prompt
        """
        return self.base_prompt.repeat(batch_size, 1, 1).to(device)


class PromptGenerator(nn.Module):
    """
    统一Prompt生成器 - 支持三种训练/推理模式

    【核心接口】
    - reset_temporal_cache()：重置时序缓存，每个视频首帧调用
    - forward(rgb_feat, tir_feat, mode)：统一前向接口

    【三种模式】
    1. 'fixed'（标准训练固定Prompt）
       - 直接返回固定的可学习Base Prompt
       - 不使用任何生成器

    2. 'static_gen'（摊销式FOMAML训练后生成静态Prompt）
       - 计算平均模态状态
       - 用平均模态状态生成各功能Prompt
       - 门控融合各Prompt
       - 所有帧用同一组Prompt

    3. 'dynamic'（摊销式FOMAML训练后动态生成每帧Prompt）
       - 计算当前帧的模态状态
       - 生成各功能Prompt
       - 用GRU更新时序缓存
       - 门控融合各Prompt
       - 每帧生成不同的Prompt
    """

    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.get('EMBED_DIM', 768)
        self.num_prompt_tokens = config.get('NUM_PROMPT_TOKENS', 8)
        self.hidden_dim = config.get('HIDDEN_DIM', 256)
        self.mode = config.get('MODE', 'fixed')
        self.meta_mode = config.get('META_MODE', 'standard')  # standard/fomaml - Prompt生成器模式

        self.enable_base = config.get('ENABLE_BASE', True)
        self.enable_mask = config.get('ENABLE_MASK', True)
        self.enable_consistency = config.get('ENABLE_CONSISTENCY', True)
        self.enable_temporal = config.get('ENABLE_TEMPORAL', True)
        self.consistency_version = config.get('CONSISTENCY_VERSION', 'fusion')

        if self.enable_base:
            self.base_generator = BasePromptGenerator(self.embed_dim, self.num_prompt_tokens)

        if self.enable_mask:
            self.mask_generator = MaskPromptGenerator(self.embed_dim, self.num_prompt_tokens, mode=self.meta_mode)

        if self.enable_consistency:
            self.consistency_generator = ConsistencyPromptGenerator(self.embed_dim, self.num_prompt_tokens, mode=self.meta_mode)

        if self.enable_temporal:
            self.temporal_generator = TemporalPromptGenerator(
                self.embed_dim, self.num_prompt_tokens, self.hidden_dim, mode=self.meta_mode
            )

        num_types = 0
        if self.enable_mask:
            num_types += 1
        if self.enable_consistency:
            num_types += 1
        if self.enable_temporal:
            num_types += 1

        if num_types > 0:
            self.gated_fusion = GatedFusion(num_types)

        self.enable_cross_attn_modulation = config.get('ENABLE_CROSS_ATTN', True)
        if self.enable_cross_attn_modulation:
            self.cross_attn_modulation = CrossAttentionModulation(self.embed_dim, num_heads=4)

        self.temporal_state = None

        # 【v14】注入层数信息（用于自适应注入强度）
        self.inject_layers = config.get('META_PROMPT_INJECT_LAYERS', [])
        self._current_inject_layer = 0

    @property
    def total_prompt_len(self):
        """返回总Prompt长度，与原接口兼容"""
        return self.num_prompt_tokens

    def reset_temporal_cache(self):
        """
        重置时序缓存

        说明：每个视频的首帧调用，清除历史状态

        输入：无
        输出：无
        """
        self.temporal_state = None
        # v5: 重置时序生成器的缓存
        if self.enable_temporal and self.temporal_generator is not None:
            self.temporal_generator.reset_cache()

    def inject(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor, x_search: torch.Tensor,
               prev_features: torch.Tensor = None, consistency_version: str = 'gradient') -> Tuple[torch.Tensor, Dict]:
        """
        【核心方法】逐层注入Meta Prompt到特征

        说明：在对应层计算各功能Prompt后，立即通过交叉注意力调制注入到特征
        这是真正的"逐层注入"实现，而非最后统一融合

        输入：
            - rgb_feat: [B, N, C] 当前层RGB模态特征
            - tir_feat: [B, N, C] 当前层TIR模态特征
            - x_search: [B, N, C] 搜索区域特征（将被调制）
            - prev_features: [B, N, C] 上一帧特征（用于时序建模）
            - consistency_version: str 一致性提示版本('gradient'/'covariance')
        输出：
            - x_search: [B, N, C] 调制后的搜索区域特征
            - intermediates: Dict 包含token_consistency/token_reliability用于计算正则化损失
        """
        # 安全校验：确保输入维度正确
        assert rgb_feat.dim() == 3, f"[inject] rgb_feat维度错误！预期3D，实际: {rgb_feat.shape}"
        assert tir_feat.dim() == 3, f"[inject] tir_feat维度错误！预期3D，实际: {tir_feat.shape}"
        assert x_search.dim() == 3, f"[inject] x_search维度错误！预期3D，实际: {x_search.shape}"
        
        B, N, C = x_search.shape
        prompt_list = []
        prompt_info = []
        intermediates = {}

        if not getattr(self, '_inject_debug_printed', False):
            print(f"[DEBUG inject] enable_consistency={self.enable_consistency}, enable_mask={self.enable_mask}, enable_temporal={self.enable_temporal}")
            self._inject_debug_printed = True

        # ===== v5核心：先计算Consistency，获取token_consistency =====
        token_consistency = None
        consistency_prompt = None
        mask_prompt = None

        if self.enable_consistency and self.consistency_generator is not None:
            consistency_prompt, consistency_intermediates = self.consistency_generator(rgb_feat, tir_feat, template_feat=None)
            if consistency_prompt is not None:
                # 从intermediates获取token_consistency（用于时序模块）
                if consistency_intermediates is not None:
                    token_consistency = consistency_intermediates.get('token_consistency', None)
                    intermediates['consistency_intermediates'] = consistency_intermediates

        if self.enable_mask and self.mask_generator is not None:
            mask_prompt, mask_intermediates = self.mask_generator(rgb_feat, tir_feat, template_feat=None)
            if mask_prompt is not None:
                if mask_intermediates is not None:
                    intermediates['mask_intermediates'] = mask_intermediates

        # ===== v5核心：时序模块复用token_consistency =====
        temporal_prompt = None
        temporal_weight = None
        if self.enable_temporal and self.temporal_generator is not None:
            # 计算融合特征（用于时序Prompt生成）
            fused_feat = (rgb_feat + tir_feat) / 2.0

            # 调用v5时序模块，传入token_consistency
            temporal_prompt, temporal_weight, temporal_intermediates = self.temporal_generator(
                rgb_feat, tir_feat,
                prev_state=None,
                prev_features=prev_features,
                token_consistency=token_consistency,
                fused_feat=fused_feat
            )

            if temporal_prompt is not None:
                if temporal_intermediates is not None:
                    intermediates['temporal_intermediates'] = temporal_intermediates

        # ===== v5核心：三提示协同（时序权重动态调整）=====
        # 当目标快速运动/遮挡时（temporal_weight大），降低Consistency权重，增加Mask权重
        if temporal_weight is not None and self.enable_temporal:
            # 计算全局时序强度
            temporal_strength = temporal_weight.mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]

            # 动态调整Consistency和Mask的权重
            if consistency_prompt is not None:
                consistency_prompt = consistency_prompt * (1.0 - 0.2 * temporal_strength)
            if mask_prompt is not None:
                mask_prompt = mask_prompt * (1.0 + 0.2 * temporal_strength)

        # 构建prompt_list
        if mask_prompt is not None:
            prompt_list.append(mask_prompt)
            prompt_info.append('mask')
        if consistency_prompt is not None:
            prompt_list.append(consistency_prompt)
            prompt_info.append('consistency')
        if temporal_prompt is not None:
            prompt_list.append(temporal_prompt)
            prompt_info.append('temporal')

        if len(prompt_list) == 0:
            return x_search, intermediates

        if len(prompt_list) == 1:
            fused_prompt = prompt_list[0]
        else:
            fused_prompt = self.gated_fusion(prompt_list)

        # ===== 【v14核心】注入强度自适应（解决多层累积问题）=====
        num_inject_layers = len(self.inject_layers) if hasattr(self, 'inject_layers') and self.inject_layers else 1
        current_layer = getattr(self, '_current_inject_layer', 0)

        base_alpha = 0.1 / max(1, num_inject_layers / 2)

        layer_factor = 1.0
        if current_layer >= 9:
            layer_factor = 0.5
        elif current_layer >= 7:
            layer_factor = 0.7

        alpha = base_alpha * layer_factor
        alpha = max(alpha, 0.02)

        if self.enable_cross_attn_modulation and self.cross_attn_modulation is not None:
            x_search = self.cross_attn_modulation(fused_prompt, x_search)
        else:
            prompt_effect = alpha * fused_prompt.mean(dim=1, keepdim=True)

            x_norm = torch.norm(x_search, dim=-1, keepdim=True)
            p_norm = torch.norm(prompt_effect, dim=-1, keepdim=True)

            scale = torch.where(
                p_norm > 0.1 * (x_norm + 1e-8),
                0.1 * x_norm / (p_norm + 1e-8),
                torch.ones_like(p_norm)
            )
            scale = torch.clamp(scale, min=0.01, max=1.0)

            x_search = x_search + prompt_effect * scale

        return x_search, intermediates

    def forward_fixed(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        模式1：固定Prompt模式（标准训练）

        说明：直接返回固定的可学习Base Prompt

        输入：
            - batch_size: 批次大小
            - device: 设备
        输出：
            - base_prompt: [B, num_prompt_tokens, C]
        """
        if self.enable_base:
            return self.base_generator(batch_size, device)
        return None

    def forward_static_gen(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor) -> torch.Tensor:
        """
        模式2：静态生成Prompt模式（FOMAML训练后推理）

        说明：用平均模态状态生成各功能Prompt，所有帧用同一组

        输入：
            - rgb_feat: [B, N, C] RGB模态特征
            - tir_feat: [B, N, C] TIR模态特征
        输出：
            - fused_prompt: [B, num_prompt_tokens, C] 融合后的Prompt
        """
        prompt_list = []

        if self.enable_mask:
            mask_prompt, _ = self.mask_generator(rgb_feat, tir_feat, template_feat=None)
            prompt_list.append(mask_prompt)

        if self.enable_consistency:
            consistency_prompt, _ = self.consistency_generator(rgb_feat, tir_feat, template_feat=None)
            prompt_list.append(consistency_prompt)

        if self.enable_temporal:
            temporal_prompt, _ = self.temporal_generator(
                rgb_feat, tir_feat,
                prev_state=None,
                prev_features=None
            )
            prompt_list.append(temporal_prompt)

        if len(prompt_list) == 0:
            return None

        if self.gated_fusion is not None:
            fused_prompt = self.gated_fusion(prompt_list)
        else:
            fused_prompt = torch.cat(prompt_list, dim=1)

        return fused_prompt

    def forward_dynamic(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                        prev_features: torch.Tensor = None) -> tuple:
        """
        模式3：动态生成Prompt模式（FOMAML训练后推理）

        说明：用当前帧模态状态生成Prompt，动态更新时序状态

        输入：
            - rgb_feat: [B, N, C] RGB模态特征
            - tir_feat: [B, N, C] TIR模态特征
            - prev_features: [B, N, C] 上一帧特征（v4新增）
        输出：
            - fused_prompt: [B, num_prompt_tokens, C] 融合后的Prompt
            - temporal_state: [1, B, hidden_dim] 更新后的GRU状态
        """
        prompt_list = []

        if self.enable_mask:
            mask_prompt, _ = self.mask_generator(rgb_feat, tir_feat, template_feat=None)
            prompt_list.append(mask_prompt)

        if self.enable_consistency:
            consistency_prompt, _ = self.consistency_generator(rgb_feat, tir_feat, template_feat=None)
            prompt_list.append(consistency_prompt)

        if self.enable_temporal:
            temporal_prompt, self.temporal_state = self.temporal_generator(
                rgb_feat, tir_feat,
                prev_state=self.temporal_state,
                prev_features=prev_features
            )
            prompt_list.append(temporal_prompt)

        if len(prompt_list) == 0:
            return None, self.temporal_state

        if self.gated_fusion is not None:
            fused_prompt = self.gated_fusion(prompt_list)
        else:
            fused_prompt = torch.cat(prompt_list, dim=1)

        return fused_prompt, self.temporal_state

    def modulate(self, prompt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        交叉注意力调制方法 - 用Prompt调制主干特征

        说明：
        交叉注意力调制 = 主流顶会方法，不硬拼接token，只调制特征

        实现原理：
        1. Query = prompt tokens (可学习的调制信号)
        2. Key = 主干特征 (提供内容信息)
        3. Value = 主干特征 (提供内容信息)
        4. 输出 = 调制后的主干特征（与原特征残差连接）

        与硬拼接的区别：
        - 硬拼接：修改token序列长度，可能抢占注意力权重
        - 交叉注意力：保持token序列不变，只调制特征

        Args:
            prompt: [B, L_p, C] Prompt token序列
            x: [B, L, C] 主干特征序列
        Returns:
            modulated_x: [B, L, C] 调制后的主干特征
        """
        if not self.enable_cross_attn_modulation:
            return x

        return self.cross_attn_modulation(prompt, x)

    def get_mask_prompt(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                       batch_size: int = None) -> torch.Tensor:
        """获取Mask提示（代理方法，兼容旧接口）"""
        if hasattr(self, 'mask_generator') and self.enable_mask:
            result = self.mask_generator(rgb_feat, tir_feat, template_feat=None)
            return result[0] if isinstance(result, tuple) else result
        return None

    def get_consistency_prompt(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                               batch_size: int = None) -> torch.Tensor:
        """获取Consistency提示（代理方法，兼容旧接口）"""
        if hasattr(self, 'consistency_generator') and self.enable_consistency:
            result = self.consistency_generator(rgb_feat, tir_feat, template_feat=None)
            return result[0] if isinstance(result, tuple) else result
        return None

    def get_temporal_prompt(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                           prev_features: torch.Tensor = None, batch_size: int = None) -> torch.Tensor:
        """获取Temporal提示（代理方法，兼容旧接口）"""
        if hasattr(self, 'temporal_generator') and self.enable_temporal:
            result = self.temporal_generator(rgb_feat, tir_feat, prev_features)
            return result[0] if isinstance(result, tuple) else result
        return None

    def forward(self, rgb_feat: torch.Tensor = None, tir_feat: torch.Tensor = None,
                mode: str = None, batch_size: int = None, device: torch.device = None) -> torch.Tensor:
        """
        统一前向接口 - 通过mode参数切换三种模式

        Args:
            rgb_feat: [B, N, C] RGB模态特征（static_gen/dynamic模式必需）
            tir_feat: [B, N, C] TIR模态特征（static_gen/dynamic模式必需）
            mode: 'fixed' | 'static_gen' | 'dynamic'
            batch_size: 批次大小（fixed模式必需）
            device: 设备（fixed模式必需）

        Returns:
            - fixed模式: base_prompt [B, num_prompt_tokens, C]
            - static_gen模式: fused_prompt [B, num_prompt_tokens, C]
            - dynamic模式: (fused_prompt [B, num_prompt_tokens, C], temporal_state)

        使用示例：
            # 固定Prompt模式（标准训练）
            prompt = generator(batch_size=16, device='cuda', mode='fixed')

            # 静态生成模式（FOMAML推理）
            prompt = generator(rgb_feat=rgb, tir_feat=tir, mode='static_gen')

            # 动态生成模式（FOMAML推理）
            prompt, state = generator(rgb_feat=rgb, tir_feat=tir, mode='dynamic')
        """
        if mode is None:
            mode = self.mode

        if mode == 'fixed':
            return self.forward_fixed(batch_size, device)
        elif mode == 'static_gen':
            return self.forward_static_gen(rgb_feat, tir_feat)
        elif mode == 'dynamic':
            return self.forward_dynamic(rgb_feat, tir_feat)
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'fixed', 'static_gen', or 'dynamic'")

    def get_config(self) -> dict:
        """
        获取当前配置信息

        Returns:
            dict: 配置信息字典
        """
        return {
            'embed_dim': self.embed_dim,
            'num_prompt_tokens': self.num_prompt_tokens,
            'hidden_dim': self.hidden_dim,
            'mode': self.mode,
            'enable_base': self.enable_base,
            'enable_mask': self.enable_mask,
            'enable_consistency': self.enable_consistency,
            'enable_temporal': self.enable_temporal,
        }


def create_prompt_generator_from_config(config: dict) -> PromptGenerator:
    """
    从配置字典创建PromptGenerator的工厂函数

    Args:
        config: 配置字典

    Returns:
        PromptGenerator实例

    配置示例：
        config = {
            'EMBED_DIM': 768,
            'NUM_PROMPT_TOKENS': 8,
            'HIDDEN_DIM': 256,
            'MODE': 'fixed',
            'ENABLE_BASE': True,
            'ENABLE_MASK': True,
            'ENABLE_CONSISTENCY': True,
            'ENABLE_TEMPORAL': True,
        }
    """
    return PromptGenerator(config)


MetaPromptGenerator = PromptGenerator


if __name__ == '__main__':
    print("=" * 60)
    print("PromptGenerator 使用示例")
    print("=" * 60)

    B, N, C = 4, 256, 768
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_fixed = {
        'EMBED_DIM': C,
        'NUM_PROMPT_TOKENS': 8,
        'HIDDEN_DIM': 256,
        'MODE': 'fixed',
        'ENABLE_BASE': True,
        'ENABLE_MASK': False,
        'ENABLE_CONSISTENCY': False,
        'ENABLE_TEMPORAL': False,
    }

    generator_fixed = PromptGenerator(config_fixed).to(device)
    prompt_fixed = generator_fixed(batch_size=B, device=device, mode='fixed')
    print(f"\n[fixed模式] Base Prompt shape: {prompt_fixed.shape}")

    config_meta = {
        'EMBED_DIM': C,
        'NUM_PROMPT_TOKENS': 8,
        'HIDDEN_DIM': 256,
        'MODE': 'dynamic',
        'ENABLE_BASE': False,
        'ENABLE_MASK': True,
        'ENABLE_CONSISTENCY': True,
        'ENABLE_TEMPORAL': True,
    }

    generator_meta = PromptGenerator(config_meta).to(device)

    generator_meta.reset_temporal_cache()

    rgb_feat = torch.randn(B, N, C).to(device)
    tir_feat = torch.randn(B, N, C).to(device)

    prompt_static = generator_meta(rgb_feat=rgb_feat, tir_feat=tir_feat, mode='static_gen')
    print(f"[static_gen模式] Prompt shape: {prompt_static.shape}")

    prompt_dynamic, state = generator_meta(rgb_feat=rgb_feat, tir_feat=tir_feat, mode='dynamic')
    print(f"[dynamic模式] Prompt shape: {prompt_dynamic.shape}")
    print(f"[dynamic模式] Temporal state shape: {state.shape if state is not None else None}")

    print("\n" + "=" * 60)
    print("所有模式测试通过！")
    print("=" * 60)
