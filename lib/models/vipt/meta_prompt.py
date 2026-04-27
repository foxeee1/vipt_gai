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

import math
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
                # 【v22修复】gain从0.1提升到1.0，与Temporal同步
                nn.init.xavier_uniform_(m.weight, gain=1.0)
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
        B, N, C = rgb_feat.shape

        if torch.isnan(rgb_feat).any() or torch.isnan(tir_feat).any():
            fallback = self.pos_encoding.repeat(B, 1, 1)
            return fallback, {'token_reliability': 0.5 * torch.ones(B, N, 1, device=rgb_feat.device)}

        combined = torch.cat([rgb_feat, tir_feat], dim=-1)

        rgb_gradient = self._compute_spatial_gradient(rgb_feat)
        tir_gradient = self._compute_spatial_gradient(tir_feat)
        gradient_input = torch.cat([rgb_feat, tir_feat, rgb_gradient, tir_gradient], dim=-1)
        gradient_feat = self.gradient_encoder(gradient_input)

        if torch.isnan(gradient_feat).any():
            gradient_feat = torch.zeros_like(gradient_feat)

        local_cov = self._compute_local_covariance(rgb_feat, tir_feat)
        cov_input = torch.cat([combined, local_cov], dim=-1)
        covariance_feat = self.covariance_encoder(cov_input)

        if torch.isnan(covariance_feat).any():
            covariance_feat = torch.zeros_like(covariance_feat)

        joint_input = torch.cat([gradient_feat, covariance_feat, combined], dim=-1)
        reliability_logits = self.joint_reliability_head(joint_input).squeeze(-1)

        reliability_logits = torch.clamp(reliability_logits, min=-3.0, max=3.0)
        token_reliability = 0.5 + 0.25 * torch.tanh(reliability_logits)

        target_bias = None
        if template_feat is not None:
            fused_feat_prelim = token_reliability.unsqueeze(-1) * rgb_feat + (1 - token_reliability.unsqueeze(-1)) * tir_feat
            template_mean = template_feat.mean(dim=1, keepdim=True)
            fused_norm = F.normalize(fused_feat_prelim, dim=-1, eps=1e-8)
            template_norm = F.normalize(template_mean.repeat(1, N, 1), dim=-1, eps=1e-8)
            target_sim = F.cosine_similarity(fused_norm, template_norm, dim=-1)
            modal_diff = torch.abs(rgb_feat - tir_feat).mean(dim=-1)
            modal_diff = F.normalize(modal_diff, dim=1)
            target_bias = torch.sigmoid(self.target_bias_net(fused_feat_prelim).squeeze(-1))
            target_bias = torch.clamp(target_bias, min=0.4, max=0.6)
            logits_bias = (target_sim * modal_diff) * 0.5
            logits_bias = torch.clamp(logits_bias, min=-2.0, max=2.0)
            token_reliability = 0.5 + 0.25 * torch.tanh(reliability_logits + logits_bias)

        token_reliability = torch.clamp(token_reliability, min=0.25, max=0.75)

        fused_feat = token_reliability.unsqueeze(-1) * rgb_feat + (1 - token_reliability.unsqueeze(-1)) * tir_feat
        fused_feat = self.fused_norm(fused_feat)

        if torch.isnan(fused_feat).any():
            fused_feat = (rgb_feat + tir_feat) / 2.0
            fused_feat = self.fused_norm(fused_feat)

        global_feat = fused_feat.mean(dim=1, keepdim=True)

        if self.mode == 'fomaml':
            global_feat = global_feat + self.task_emb

        mask_prompt = global_feat.repeat(1, self.num_prompt_tokens, 1)
        mask_prompt = mask_prompt + self.pos_encoding
        mask_prompt = self.proj(mask_prompt)

        if torch.isnan(mask_prompt).any() or torch.isinf(mask_prompt).any():
            mask_prompt = self.pos_encoding.repeat(B, 1, 1)
            token_reliability = 0.5 * torch.ones(B, N, 1, device=rgb_feat.device)

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
                # 【v22修复】gain从0.1提升到1.0，与Temporal同步
                nn.init.xavier_uniform_(m.weight, gain=1.0)
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
    Temporal Prompt生成器 - v9架构（软约束 + 温度控制 + 边界陷阱修复）

    【v9核心创新】解决v8的"边界陷阱"问题：权重在0.2和0.8之间震荡
    
    v8问题诊断：
      - IoU上升但avg_weight在0.2和0.8之间极端震荡
      - weight_deviation饱和在高值（区分度过大）
      - 根因：硬clamp[0.2,0.8] + 强正则化 → 模型只能在边界取值
    
    v9解决方案：
    
    1. 【温度控制sigmoid】用可学习温度参数控制权重锐度
       - 初期T=1.0：sigmoid平滑，权重分布均匀
       - 后期自适应调整：避免过早收敛到极值
       
    2. 【软clamp替代硬clamp】
       - 放宽范围到[0.05, 0.95]
       - 用sigmoid软化边界，允许梯度流过
       
    3. 【动态正则化退火】
       - 训练初期强约束（快速探索）
       - 训练后期弱约束（稳定收敛）
       
    4. 【层间差异软引导】
       - 从"惩罚相同"变为"鼓励不同"
       - 用奖励机制而非惩罚机制

    输入/输出与v8兼容
    """
    def __init__(self, embed_dim: int, num_prompt_tokens: int = 8, hidden_dim: int = 256, mode: str = 'standard',
                 max_layers: int = 4):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.mode = mode
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_layers = max_layers
        assert mode in ['standard', 'fomaml'], f"[TemporalPromptGenerator] mode必须是'standard'或'fomaml'，实际: {mode}"

        # ===== v9核心1: 帧差专用LayerNorm =====
        self.frame_diff_norm = nn.LayerNorm(embed_dim)

        # ===== v9核心2: 帧差编码MLP =====
        self.frame_diff_encoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        # ===== v24新增：自学习模态对齐MLP（解耦Consistency依赖） =====
        # 问题：Temporal依赖Consistency的token_consistency（稳定在0.5）
        #       导致aligned_feat=0.5*rgb+0.5*tir完全固定，帧差无意义
        # 解决：Temporal自己计算模态对齐权重，不依赖上游
        self.align_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # ===== v9核心3: 时序Prompt投影层 =====
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        # 【v22修复】输出LayerNorm，稳定特征分布
        self.temporal_output_norm = nn.LayerNorm(embed_dim)

        # ===== v9核心4: 基础位置编码 =====
        self.pos_encoding = nn.Parameter(torch.randn(1, num_prompt_tokens, embed_dim) * 0.01)

        # ===== v9核心5: 层自适应参数 =====
        self.layer_pos_bias = nn.Parameter(torch.zeros(max_layers, 1, embed_dim) * 0.01)
        self.layer_logits_bias = nn.Parameter(torch.zeros(max_layers, 1))
        self.layer_gate_threshold = nn.Parameter(torch.zeros(max_layers, 1))

        # ===== v9核心6: 温度参数（关键新特性）=====
        # 控制sigmoid的锐度，避免权重过早收敛到极值
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)  # 初始温度2.0，较平滑

        # ===== v9核心7: 元学习深度融合 =====
        if self.mode == 'fomaml':
            self.task_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.task_emb, std=0.01)

        # ===== v9核心8: 动态缓存 =====
        self._prev_aligned_feat_cache = None
        self._prev_layer_weights = {}
        self._prev_frame_weights = {}  # 【v25新增】帧间平滑缓存
        self._current_layer_id = 0
        self._training_step = 0  # 用于正则化退火

        # ===== v9核心9: 固定超参数 =====
        self.fixed_motion_scale = 3.0

        self._init_weights()

    def _init_weights(self):
        """初始化所有线性层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 【v22修复】gain从0.1提升到1.0
                # 原gain=0.1导致权重仅0.01量级，temporal输出≈0，梯度99%丢失
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_cache(self):
        self._prev_aligned_feat_cache = None
        self._prev_layer_weights.clear()
        self._prev_frame_weights.clear()  # 【v25新增】清除帧间缓存

    def forward(self, rgb_feat: torch.Tensor, tir_feat: torch.Tensor,
                prev_state: Optional[torch.Tensor] = None,
                prev_features: Optional[torch.Tensor] = None,
                token_consistency: Optional[torch.Tensor] = None,
                fused_feat: Optional[torch.Tensor] = None,
                layer_id: int = 0,
                return_intermediate: bool = False) -> tuple:
        """
        Args:
            rgb_feat: [B, N, C] RGB模态特征
            tir_feat: [B, N, C] TIR模态特征
            prev_state: 兼容旧接口，不再使用
            prev_features: 兼容旧接口，不再使用
            token_consistency: [B, N] 当前帧跨模态一致性权重（用于对齐特征）
            fused_feat: [B, N, C] 融合特征（用于生成Prompt）
            layer_id: int 当前注入层编号（v8新参数，解决双层同质化）
            return_intermediate: 是否返回中间值（供可视化使用）
        Returns:
            temporal_prompt: [B, num_prompt_tokens, C]
            temporal_weight: [B, N] 时序权重
            intermediates: dict（包含global_motion用于正则化）
        """
        B, N, C = rgb_feat.shape

        # ===== v8 Step0: 层自适应参数选择（解决双层同质化）=====
        layer_idx = min(layer_id, self.max_layers - 1)  # 安全限制
        self._current_layer_id = layer_id

        # ===== v9 Step1: 计算对齐特征（v24解耦上游依赖） =====
        # 【v24关键修复】不再依赖Consistency的token_consistency（稳定在0.5导致帧差无意义）
        # 改用自学习align_mlp计算模态对齐权重，保证时序建模的有效性
        
        combined_feat = torch.cat([rgb_feat, tir_feat], dim=-1)
        self_align_weight = self.align_mlp(combined_feat).squeeze(-1)
        self_align_weight = torch.clamp(self_align_weight, min=0.3, max=0.7)
        
        if torch.isnan(self_align_weight).any():
            self_align_weight = torch.ones(B, N, device=rgb_feat.device, dtype=rgb_feat.dtype) * 0.5
        
        aligned_feat = self_align_weight.unsqueeze(-1) * rgb_feat + (1 - self_align_weight).unsqueeze(-1) * tir_feat

        # ===== v9 Step2: 计算帧差 =====
        if self._prev_aligned_feat_cache is not None and self._prev_aligned_feat_cache.shape == (B, N, C):
            if torch.isnan(self._prev_aligned_feat_cache).any() or torch.isinf(self._prev_aligned_feat_cache).any():
                self._prev_aligned_feat_cache = None
            
            if self._prev_aligned_feat_cache is not None:
                frame_diff = aligned_feat - self._prev_aligned_feat_cache
                frame_diff = torch.clamp(frame_diff, min=-5.0, max=5.0)
            else:
                frame_diff = torch.zeros(B, N, C, device=rgb_feat.device, dtype=rgb_feat.dtype)
        else:
            frame_diff = torch.zeros(B, N, C, device=rgb_feat.device, dtype=rgb_feat.dtype)

        # ===== v9 Step3: 固定放大 + LayerNorm =====
        frame_diff_scaled = self.fixed_motion_scale * frame_diff
        frame_diff_normed = self.frame_diff_norm(frame_diff_scaled)

        # 平滑近似abs：sqrt(x^2 + eps)，避免abs在0处梯度消失导致motion_gate无法学习
        global_motion = torch.mean(torch.sqrt(frame_diff_normed ** 2 + 1e-4), dim=[1, 2], keepdim=True)

        # 跨迭代缓存必须detach，否则backward会试图通过已释放的计算图
        self._prev_aligned_feat_cache = aligned_feat.detach().clone()

        # ===== v9 Step4: 帧差编码 → 时序logits（层自适应 + 温度控制）=====
        if self.mode == 'fomaml':
            frame_diff_modulated = frame_diff_normed + 0.1 * self.task_emb
        else:
            frame_diff_modulated = frame_diff_normed

        temporal_logits = self.frame_diff_encoder(frame_diff_modulated).squeeze(-1)  # [B, N]

        layer_bias = self.layer_logits_bias[layer_idx]  # [1]
        temporal_logits = temporal_logits + layer_bias

        # 【v10关键1】训练时注入高斯噪声，防止logits收敛到极端值
        if self.training:
            noise_std = 0.1
            noise = torch.randn_like(temporal_logits) * noise_std
            temporal_logits = temporal_logits + noise

        # 【v10关键2】使用标准sigmoid（T=1.0），不使用可学习温度
        temporal_weight = torch.sigmoid(temporal_logits)

        # 【v22修复】放宽clamp范围到[0.1, 0.9]，避免权重被锁死
        # 原v10的[0.25, 0.75]太严格，导致权重无法学习到合理值
        # Temporal的物理意义是运动自适应，天然权重应该在0.3-0.8范围
        temporal_weight = torch.clamp(temporal_weight, min=0.1, max=0.9)

        # 更新训练步数计数器（用于正则化退火）
        if self.training:
            self._training_step += 1

        # ===== v10 Step5: 运动门控（层自适应阈值）=====
        layer_gate_threshold = 0.3 + self.layer_gate_threshold[layer_idx]
        # 【v22修复】确保motion_gate最小值0.5，避免梯度被过度衰减
        # 原版0.3+0.7*sigmoid最小0.3，太小导致65%梯度丢失
        motion_gate = 0.5 + 0.5 * torch.sigmoid(2.0 * (global_motion.squeeze(-1) - layer_gate_threshold))

        # ===== v10 Step6: 生成时序Prompt（层自适应位置编码）=====
        if fused_feat is None:
            fused_feat = aligned_feat
        
        weighted_feat = temporal_weight.unsqueeze(-1) * fused_feat  # [B, N, C]
        temporal_global = weighted_feat.mean(dim=1, keepdim=True)  # [B, 1, C]
        temporal_base = self.temporal_proj(temporal_global)  # [B, 1, C]

        temporal_prompt = temporal_base.expand(-1, self.num_prompt_tokens, -1)
        
        layer_pos_bias = self.layer_pos_bias[layer_idx].unsqueeze(0)  # [1, 1, C]
        temporal_prompt = temporal_prompt + self.pos_encoding + layer_pos_bias

        # 【v22修复】用LayerNorm稳定输出分布，再乘motion_gate
        temporal_prompt = self.temporal_output_norm(temporal_prompt)
        temporal_prompt = motion_gate.unsqueeze(-1) * temporal_prompt

        if torch.isnan(temporal_prompt).any():
            temporal_prompt = torch.zeros_like(temporal_prompt)
        if torch.isinf(temporal_prompt).any():
            temporal_prompt = torch.clamp(temporal_prompt, min=-10.0, max=10.0)

        # ===== v10 Step7: 记录当前层权重（用于层间差异+帧间平滑约束）=====
        self._prev_layer_weights[layer_id] = temporal_weight.detach().clone()

        # 【v25修复】层间差异改为tensor（原.item()导致无法反向传播）
        layer_divergence_tensor = torch.tensor(0.0, device=temporal_weight.device)
        if layer_id > 0 and (layer_id - 1) in self._prev_layer_weights:
            prev_layer_weight = self._prev_layer_weights[layer_id - 1]
            if prev_layer_weight.shape == temporal_weight.shape:
                layer_divergence_tensor = torch.mean((temporal_weight - prev_layer_weight) ** 2)

        # 【v25新增】获取上一帧同层权重（用于帧间平滑约束）
        prev_temporal_w = self._prev_frame_weights.get(layer_id, None)
        # 更新当前帧权重缓存
        self._prev_frame_weights[layer_id] = temporal_weight.detach().clone()

        # ===== 可视化记录 =====
        vis = PromptVisualizer.get()
        if self.training:
            vis.log_prompt_stats('Temporal', temporal_prompt, temporal_weight)
            vis.log_grad_norm('Temporal', self)
            vis.log_scalar('Temporal/global_motion', global_motion.mean().item())
            vis.log_scalar('Temporal/motion_gate', motion_gate.mean().item())
            vis.log_scalar('Temporal/temporal_weight_mean', temporal_weight.mean().item())
            vis.log_scalar('Temporal/weight_deviation', (temporal_weight - 0.5).abs().mean().item())
            vis.log_scalar(f'Temporal/layer{layer_id}_weight_mean', temporal_weight.mean().item())
            vis.log_scalar(f'Temporal/layer{layer_id}_divergence', layer_divergence_tensor.item())
            vis.log_scalar('Temporal/noise_std', noise_std if self.training else 0.0)

        if return_intermediate:
            intermediates = {
                'temporal_weight': temporal_weight,
                'global_motion': global_motion,
                'motion_gate': motion_gate,
                'layer_id': layer_id,
                'layer_divergence': layer_divergence_tensor,
                'prev_temporal_weight': prev_temporal_w,
                '_debug_valid': True,
            }
            return temporal_prompt, temporal_weight, intermediates

        return temporal_prompt, temporal_weight, None


class GatedFusion(nn.Module):
    """
    门控融合网络 - 平衡多种Prompt的贡献（优化版）

    【核心优化】
    1. 单类型Prompt时直接返回，避免不必要的门控计算
    2. 多类型时使用软门控加权融合
    3. 【v21修复】支持可变数量的prompt输入
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

        # 【v21修复】当prompt数量与初始化时不匹配，使用简单平均融合
        if len(prompt_list) != self.num_prompt_types:
            return sum(prompt_list) / len(prompt_list)

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

        # 【v22修复】输出LayerNorm，稳定调制信号分布
        self.mod_norm = nn.LayerNorm(embed_dim)

        # 可学习调制强度
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

        if torch.isnan(prompt).any() or torch.isnan(x).any():
            return x.clone()

        q = self.norm_q(prompt)
        k = self.norm_k(x)
        v = x

        if torch.isnan(q).any() or torch.isnan(k).any():
            return x.clone()

        Q = self.q_proj(q)
        K = self.k_proj(k)

        if torch.isnan(Q).any() or torch.isnan(K).any():
            return x.clone()

        Q = Q.view(B, L_p, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Q = torch.clamp(Q, min=-10.0, max=10.0)
        K = torch.clamp(K, min=-10.0, max=10.0)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn - attn.max(dim=-1, keepdim=True).values.detach()
        attn = torch.clamp(attn, min=-10.0, max=10.0)
        attn = attn.softmax(dim=-1)

        if torch.isnan(attn).any():
            return x.clone()

        attn_out = (attn @ V).permute(0, 2, 1, 3).reshape(B, L_p, C)

        if torch.isnan(attn_out).any():
            return x.clone()

        attn_out = self.out_proj(attn_out)

        if torch.isnan(attn_out).any():
            return x.clone()

        global_signal = attn_out.mean(dim=1, keepdim=True)

        global_signal = self.mod_norm(global_signal)

        if torch.isnan(global_signal).any():
            return x.clone()

        alpha = torch.sigmoid(self.alpha) * 1.0
        alpha = torch.clamp(alpha, min=0.0, max=1.0)

        modulated_x = x + alpha * global_signal

        if torch.isnan(modulated_x).any() or torch.isinf(modulated_x).any():
            return x.clone()

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
        self.meta_mode = config.get('META_MODE', 'standard')
        
        # 【v15协同】协同策略配置
        # independent: 独立计算，简单相加
        # temporal_modulate: 时序动态调制一致性
        # bidirectional: 双向交互调制
        # gating: 门控自适应融合
        # temporal_modulate_anneal: 时序调制+动态温度退火（v17新增）
        # gating_modulate_hybrid: 门控+调制混合策略（v17新增）
        # gating_token_level: Token级门控（v17新增）
        self.coop_strategy = config.get('COOP_STRATEGY', 'temporal_modulate')
        
        # 【v17新增】动态温度退火计数器（用于temporal_modulate_anneal策略）
        self._coop_step = 0
        
        # 【v15协同】协同门控参数（用于gating策略）
        if self.coop_strategy == 'gating':
            self.coop_gate = nn.Parameter(torch.zeros(1))  # 【v18优化】无偏初始化，sigmoid(0)=0.5
        
        # 【v21新增】三提示门控融合
        if self.coop_strategy == 'gating_triple':
            # 三提示门控：Consistency, Temporal, Mask
            # 初始化为[0.4, 0.35, 0.25]的logits，softmax后接近这个分布
            # 使用zeros初始化，让模型自己学习最优权重
            self.coop_gate_triple = nn.Parameter(torch.zeros(3))  # [consistency, temporal, mask]
        
        # 【v18新增】门控v2策略：特征调制门控
        if self.coop_strategy == 'gating_v2':
            self.coop_gate = nn.Parameter(torch.zeros(1))  # 无偏初始化
            # 特征调制投影层：将门控信号投影到特征空间
            self.gate_feat_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # 【v17新增】门控MLP（用于gating_modulate_hybrid和gating_token_level策略）
        if self.coop_strategy in ['gating_modulate_hybrid', 'gating_token_level']:
            self.coop_gate_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 2)
            )

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
        self.training_stage = 0

        self.inject_layers = config.get('META_PROMPT_INJECT_LAYERS', [])
        self._current_inject_layer = 0
        
        self.temporal_inject_layers = config.get('TEMPORAL_PROMPT_INJECT_LAYERS', self.inject_layers)
        
        self.mask_inject_layers = config.get('MASK_PROMPT_INJECT_LAYERS', self.inject_layers)
        
        self.modality_inject_layers = config.get('MODALITY_PROMPT_INJECT_LAYERS', [1, 2, 3])
        
        if self.enable_consistency:
            self.alpha_consistency = nn.Parameter(torch.tensor(0.1))
        if self.enable_temporal:
            self.alpha_temporal = nn.Parameter(torch.tensor(0.1))
        if self.enable_mask:
            self.alpha_mask = nn.Parameter(torch.tensor(0.05))
        
        self.rgb_type_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.tir_type_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        
        self.modality_prompt_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )
        
        self.layer_gates = nn.ModuleDict()
        all_inject_layers = set(self.inject_layers) | set(self.temporal_inject_layers) | set(self.mask_inject_layers) | set(self.modality_inject_layers)
        for layer_id in all_inject_layers:
            num_gates = 4
            self.layer_gates[str(layer_id)] = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim // 4),
                nn.GELU(),
                nn.Linear(self.embed_dim // 4, num_gates)
            )
            with torch.no_grad():
                if layer_id in [1, 2, 3]:
                    self.layer_gates[str(layer_id)][-1].bias.data = torch.tensor([0.0, 0.0, -2.0, -2.0])
                elif layer_id in [5, 6]:
                    self.layer_gates[str(layer_id)][-1].bias.data = torch.tensor([0.0, 0.0, 0.0, -2.0])
                elif layer_id in [8, 9]:
                    self.layer_gates[str(layer_id)][-1].bias.data = torch.tensor([0.0, 0.0, -2.0, 0.0])
                else:
                    self.layer_gates[str(layer_id)][-1].bias.data = torch.tensor([0.0, 0.0, 0.0, 0.0])

    @property
    def total_prompt_len(self):
        return self.num_prompt_tokens

    def set_current_layer(self, layer_id: int):
        """每个Transformer block前必须调用，显式设置当前层号"""
        self._current_inject_layer = layer_id

    def reset_temporal_cache(self):
        self.temporal_state = None
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

        # 【v24关键修复】每层只计算需要注入的模块，非注入层完全不forward
        # 彻底消除无效梯度干扰：之前非注入层的Temporal也在forward，导致全局锁死
        current_layer_id = getattr(self, '_current_inject_layer', 0)
        current_stage = getattr(self, 'training_stage', 0)
        
        # Stage1: 只使用模态Prompt，完全跳过辅助分支（避免未训练Prompt引入噪声）
        # Stage2: 强制计算所有辅助分支（确保梯度连接）
        # Stage3: 正常按层注入
        if current_stage == 1:
            inject_consistency = False
            inject_temporal = False
            inject_mask = False
        elif current_stage == 2:
            # 【v25关键修复】Stage2强制计算所有辅助分支，确保梯度连接
            # 只要当前层在任意一个注入层集合中，就计算对应的prompt
            inject_consistency = self.enable_consistency and (
                current_layer_id in self.inject_layers or len(self.inject_layers) == 0
            )
            inject_temporal = self.enable_temporal and (
                current_layer_id in self.temporal_inject_layers or len(self.temporal_inject_layers) == 0
            )
            inject_mask = self.enable_mask and (
                current_layer_id in self.mask_inject_layers or len(self.mask_inject_layers) == 0
            )
        else:
            inject_consistency = self.enable_consistency and current_layer_id in self.inject_layers
            inject_temporal = self.enable_temporal and current_layer_id in self.temporal_inject_layers
            inject_mask = self.enable_mask and current_layer_id in self.mask_inject_layers
        
        token_consistency = None
        consistency_prompt = None
        mask_prompt = None

        # 只在注入层才计算Consistency
        if inject_consistency and self.consistency_generator is not None:
            consistency_prompt, consistency_intermediates = self.consistency_generator(rgb_feat, tir_feat, template_feat=None)
            if consistency_prompt is not None:
                if consistency_intermediates is not None:
                    token_consistency = consistency_intermediates.get('token_consistency', None)
                    intermediates['consistency_intermediates'] = consistency_intermediates

        # 只在注入层才计算Mask
        if inject_mask and self.mask_generator is not None:
            mask_prompt, mask_intermediates = self.mask_generator(rgb_feat, tir_feat, template_feat=None)
            if mask_prompt is not None:
                if mask_intermediates is not None:
                    intermediates['mask_intermediates'] = mask_intermediates

        # 只在注入层才计算Temporal
        temporal_prompt = None
        temporal_weight = None
        if inject_temporal and self.temporal_generator is not None:
                fused_feat = (rgb_feat + tir_feat) / 2.0

                # 调用v10时序模块，传入layer_id实现层自适应
                # 【关键修复】必须传return_intermediate=True，否则正则化无法生效
                temporal_prompt, temporal_weight, temporal_intermediates = self.temporal_generator(
                    rgb_feat, tir_feat,
                    prev_state=None,
                    prev_features=prev_features,
                    token_consistency=token_consistency,
                    fused_feat=fused_feat,
                    layer_id=current_layer_id,
                    return_intermediate=True  # 【v10修复】确保返回intermediates供正则化使用
                )

                if temporal_prompt is not None:
                    if temporal_intermediates is not None:
                        intermediates['temporal_intermediates'] = temporal_intermediates

        # ===== 【v15核心】多策略协同系统（支持任意单提示开关 + 元学习模式适配）=====
        # 协同策略仅在两种或以上提示同时启用时生效
        # 当某个提示被关闭时，自动跳过相关协同逻辑
        # 元学习模式(fomaml)下，各生成器已有task_emb，协同策略不干预
        
        # 检测哪些提示可用
        has_consistency = consistency_prompt is not None and self.enable_consistency
        has_temporal = temporal_prompt is not None and self.enable_temporal
        has_mask = mask_prompt is not None and self.enable_mask
        
        # 记录协同状态用于调试和分析
        intermediates['coop_status'] = {
            'has_consistency': has_consistency,
            'has_temporal': has_temporal,
            'has_mask': has_mask,
            'meta_mode': self.meta_mode,
            'strategy': self.coop_strategy
        }
        
        # ===== 双提示协同（Consistency + Temporal）=====
        if has_consistency and has_temporal:
            if self.coop_strategy == 'independent':
                pass
            
            elif self.coop_strategy == 'temporal_modulate':
                # 【v16修复】保护性时序调制
                # 原问题: 单向压制导致consistency持续被削弱，TIR权重降至0.08
                # 修复方案:
                #   1. 降低调制系数(0.2→0.08)，减少压制强度
                #   2. 添加对称保护：只在极端情况下才调整
                #   3. 限制调制范围[0.9, 1.05]，防止过度压制或增强
                
                temporal_strength = temporal_weight.mean(dim=1, keepdim=True).unsqueeze(-1)
                
                # 计算调制因子，限制在安全范围
                modulate_factor = 1.0 - 0.08 * temporal_strength
                modulate_factor = torch.clamp(modulate_factor, min=0.90, max=1.05)
                
                consistency_prompt = consistency_prompt * modulate_factor
                
                # 【v16调试】记录调制信息用于分析
                intermediates['coop_modulate_factor'] = modulate_factor.mean().item()
                intermediates['coop_temporal_strength_raw'] = temporal_strength.mean().item()
            
            elif self.coop_strategy == 'bidirectional':
                # 【v16修复】保护性双向交互调制
                # 原问题: 双向压制形成负反馈循环，consistency被持续削弱
                # 修复方案:
                #   1. 降低双向调制系数(0.2→0.06)
                #   2. 添加安全范围限制
                #   3. 防止极端值导致的恶性循环
                
                temporal_strength = temporal_weight.mean(dim=1, keepdim=True).unsqueeze(-1)
                
                # 时序调制一致性（降低系数+范围限制）
                consistency_modulate = 1.0 - 0.06 * temporal_strength
                consistency_modulate = torch.clamp(consistency_modulate, min=0.92, max=1.03)
                consistency_prompt = consistency_prompt * consistency_modulate
                
                # 一致性调制时序（只在token_consistency有效时）
                if token_consistency is not None:
                    consistency_strength = token_consistency.mean(dim=1, keepdim=True).unsqueeze(-1)
                    
                    # 一致性调制时序（降低系数+范围限制）
                    temporal_modulate = 0.97 + 0.06 * consistency_strength
                    temporal_modulate = torch.clamp(temporal_modulate, min=0.95, max=1.05)
                    temporal_prompt = temporal_prompt * temporal_modulate
                
                # 【v16调试】记录双向调制信息
                intermediates['coop_consistency_modulate'] = consistency_modulate.mean().item()
                if token_consistency is not None:
                    intermediates['coop_temporal_modulate'] = temporal_modulate.mean().item()
            
            elif self.coop_strategy == 'gating':
                if hasattr(self, 'coop_gate'):
                    gate = torch.sigmoid(self.coop_gate)
                    consistency_prompt = consistency_prompt * gate
                    temporal_prompt = temporal_prompt * (1.0 - gate)
            
            elif self.coop_strategy == 'gating_v2':
                # 【v18新增】特征调制门控
                # 核心改进：门控不只作用在Prompt上，而是直接调制特征融合
                # 第一步：门控调制融合特征（rgb_feat和tir_feat的加权）
                # 第二步：用调制后的特征重新生成Prompt
                # 第三步：门控正则化（降低系数，不惩罚极端门控）
                
                if hasattr(self, 'coop_gate'):
                    gate = torch.sigmoid(self.coop_gate)
                    
                    # 特征级调制：用门控加权rgb和tir特征
                    # gate→consistency(rgb主导), (1-gate)→temporal(tir主导)
                    modulated_rgb = rgb_feat * gate
                    modulated_tir = tir_feat * (1.0 - gate)
                    fused_feat = modulated_rgb + modulated_tir
                    
                    # 用调制后的特征投影生成调制信号
                    gate_signal = torch.tanh(self.gate_feat_proj(fused_feat))  # [B, N, C]
                    
                    # 将调制信号融入Prompt
                    consistency_prompt = consistency_prompt + 0.1 * gate_signal
                    temporal_prompt = temporal_prompt - 0.1 * gate_signal
                    
                    # 门控正则化：计算门控熵，用于监控（不加入loss，避免约束）
                    gate_val = gate.item()
                    gate_entropy = -(gate_val * math.log(gate_val + 1e-8) + (1 - gate_val) * math.log(1 - gate_val + 1e-8))
                    
                    # 记录调试信息
                    intermediates['coop_gate_value'] = gate_val
                    intermediates['coop_gate_entropy'] = gate_entropy
            
            elif self.coop_strategy == 'joint_regularize':
                # 【v16修复】保护性联合正则化调制
                # 原问题: 调制系数0.15仍然过大
                # 修复方案: 降低至0.05，几乎不影响consistency
                
                temporal_strength = temporal_weight.mean(dim=1, keepdim=True).unsqueeze(-1)
                
                # 极轻微的调制（几乎不调整）
                modulate_factor = 1.0 - 0.05 * temporal_strength
                modulate_factor = torch.clamp(modulate_factor, min=0.95, max=1.02)
                
                consistency_prompt = consistency_prompt * modulate_factor
                intermediates['coop_temporal_strength'] = temporal_strength
                intermediates['coop_strategy'] = 'joint_regularize'
                intermediates['coop_modulate_factor'] = modulate_factor.mean().item()
            
            elif self.coop_strategy == 'temporal_modulate_anneal':
                # 【v17新增】时序调制+动态温度退火
                # 核心思想：训练初期弱调制，后期强调制，避免早期负反馈
                # 温度退火策略：
                #   - 前2000步：modulate_coeff=0.02（几乎不调制）
                #   - 2000-8000步：线性升到0.12
                #   - 8000步后：固定0.12
                
                if self.training:
                    self._coop_step += 1
                step = self._coop_step
                
                # 动态温度退火
                if step < 2000:
                    modulate_coeff = 0.02
                elif step < 8000:
                    modulate_coeff = 0.02 + (step - 2000) * (0.10 / 6000)
                else:
                    modulate_coeff = 0.12
                
                temporal_strength = temporal_weight.mean(dim=1, keepdim=True).unsqueeze(-1)
                
                # 用动态系数计算调制因子
                modulate_factor = 1.0 - modulate_coeff * temporal_strength
                modulate_factor = torch.clamp(modulate_factor, min=0.90, max=1.05)
                
                consistency_prompt = consistency_prompt * modulate_factor
                
                # 记录调试信息
                intermediates['coop_anneal_step'] = step
                intermediates['coop_anneal_coeff'] = modulate_coeff
                intermediates['coop_modulate_factor'] = modulate_factor.mean().item()
                intermediates['coop_temporal_strength_raw'] = temporal_strength.mean().item()
            
            elif self.coop_strategy == 'gating_modulate_hybrid':
                # 【v17新增】门控+调制混合策略
                # 核心思想：门控自适应学习 + 时序调制物理先验
                # 第一步：门控MLP生成初始门控
                # 第二步：用时序强度微调门控
                # 第三步：用微调后的门控融合
                
                if hasattr(self, 'coop_gate_mlp'):
                    # 用search特征生成门控
                    gate_input = x_search.mean(dim=1)  # [B, C]
                    gate_logits = self.coop_gate_mlp(gate_input)  # [B, 2]
                    gate = F.softmax(gate_logits, dim=-1)  # [B, 2]
                    
                    # 时序强度微调门控
                    temporal_strength = temporal_weight.mean(dim=1)  # [B]
                    # 时序强度高时，稍微降低一致性门控，提高时序门控
                    gate_adjust = 0.05 * temporal_strength.unsqueeze(-1)  # [B, 1]
                    gate[:, 0:1] = gate[:, 0:1] * (1.0 - gate_adjust)
                    gate[:, 1:2] = gate[:, 1:2] * (1.0 + gate_adjust)
                    # 重新归一化
                    gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-8)
                    
                    # 用微调后的门控融合
                    consistency_prompt = consistency_prompt * gate[:, 0:1].unsqueeze(1)
                    temporal_prompt = temporal_prompt * gate[:, 1:2].unsqueeze(1)
                    
                    # 记录调试信息
                    intermediates['coop_gate'] = gate.mean(dim=0).tolist()
                    intermediates['coop_temporal_strength_raw'] = temporal_strength.mean().item()
            
            elif self.coop_strategy == 'gating_token_level':
                # 【v17新增】Token级门控
                # 核心思想：每个token独立门控，更精细的自适应融合
                
                if hasattr(self, 'coop_gate_mlp'):
                    B, N, C = x_search.shape
                    
                    # Token级门控：每个token生成独立门控
                    gate_logits = self.coop_gate_mlp(x_search)  # [B, N, 2]
                    gate = F.softmax(gate_logits, dim=-1)  # [B, N, 2]
                    
                    # Token级融合
                    consistency_prompt = consistency_prompt * gate[:, :, 0:1]
                    temporal_prompt = temporal_prompt * gate[:, :, 1:2]
                    
                    # 记录调试信息
                    intermediates['coop_gate_mean'] = gate.mean(dim=1).mean(dim=0).tolist()
                    intermediates['coop_gate_std'] = gate.std(dim=1).mean().item()
        
        # ===== 三提示协同（Mask + Consistency + Temporal）=====
        # 【v21新增】三提示门控融合
        if has_consistency and has_temporal and has_mask:
            if self.coop_strategy == 'gating_triple' and hasattr(self, 'coop_gate_triple'):
                # 三提示全局门控融合
                gate = F.softmax(self.coop_gate_triple, dim=-1)  # [3]
                
                # 门控加权融合
                consistency_prompt = consistency_prompt * gate[0]
                temporal_prompt = temporal_prompt * gate[1]
                mask_prompt = mask_prompt * gate[2]
                
                # 记录调试信息
                intermediates['coop_gate_triple'] = gate.detach().cpu().tolist()
        
        # 原有的三提示协同逻辑（非门控策略时）
        elif has_mask and has_temporal:
            temporal_strength = temporal_weight.mean(dim=1, keepdim=True).unsqueeze(-1)
            
            # 【v16修复】保护性Mask调制
            # 原问题: 0.2系数过大，导致Mask在运动大时过度增强
            # 修复方案: 降低系数+限制范围
            
            mask_modulate = 1.0 + 0.08 * temporal_strength
            mask_modulate = torch.clamp(mask_modulate, min=0.97, max=1.08)
            
            mask_prompt = mask_prompt * mask_modulate
            intermediates['coop_mask_modulate'] = mask_modulate.mean().item()

        # ===== 【v22核心】并行独立注入+分层残差融合 =====
        # CVPR 2025 CVPT方法：每个提示独立计算、独立注入，完全不互相干扰
        if self.coop_strategy == 'parallel_residual':
            current_layer_id = getattr(self, '_current_inject_layer', 0)
            
            # 记录每个提示的注入状态
            inject_info = []
            
            # 1. Consistency独立注入
            if consistency_prompt is not None and current_layer_id in self.inject_layers:
                # 【v22修复】直接使用alpha值，不用sigmoid（sigmoid(0.1)≈0.525太大）
                alpha_c = torch.clamp(self.alpha_consistency, min=0.01, max=0.5)
                if self.enable_cross_attn_modulation and self.cross_attn_modulation is not None:
                    consistency_modulation = self.cross_attn_modulation(consistency_prompt, x_search)
                else:
                    consistency_modulation = consistency_prompt.mean(dim=1, keepdim=True)
                x_search = x_search + alpha_c * consistency_modulation
                inject_info.append(f'consistency@L{current_layer_id}')
                intermediates['alpha_consistency'] = alpha_c.item()
            
            # 2. Temporal独立注入
            if temporal_prompt is not None and current_layer_id in self.temporal_inject_layers:
                alpha_t = torch.clamp(self.alpha_temporal, min=0.01, max=0.5)
                if self.enable_cross_attn_modulation and self.cross_attn_modulation is not None:
                    temporal_modulation = self.cross_attn_modulation(temporal_prompt, x_search)
                else:
                    temporal_modulation = temporal_prompt.mean(dim=1, keepdim=True)
                x_search = x_search + alpha_t * temporal_modulation
                inject_info.append(f'temporal@L{current_layer_id}')
                intermediates['alpha_temporal'] = alpha_t.item()
            
            # 3. Mask独立注入
            if mask_prompt is not None and current_layer_id in self.mask_inject_layers:
                alpha_m = torch.clamp(self.alpha_mask, min=0.01, max=0.5)
                if self.enable_cross_attn_modulation and self.cross_attn_modulation is not None:
                    mask_modulation = self.cross_attn_modulation(mask_prompt, x_search)
                else:
                    mask_modulation = mask_prompt.mean(dim=1, keepdim=True)
                x_search = x_search + alpha_m * mask_modulation
                inject_info.append(f'mask@L{current_layer_id}')
                intermediates['alpha_mask'] = alpha_m.item()
            
            intermediates['parallel_inject_info'] = inject_info
            return x_search, intermediates
        
        # ===== 【v22核心】CVPR 2025层门控（Layer-wise Gating）=====
        # 完全保留最优注入层，完全自适应强度
        # 每层学习独立的门控，控制每个提示在该层的注入强度
        if self.coop_strategy == 'layer_gating':
            current_layer_id = getattr(self, '_current_inject_layer', 0)
            
            if str(current_layer_id) in self.layer_gates:
                gate_input = torch.cat([rgb_feat.mean(dim=1), tir_feat.mean(dim=1)], dim=-1)
                gate_logits = self.layer_gates[str(current_layer_id)](gate_input)
                gate = torch.sigmoid(gate_logits)  # [B, 4] 独立门控
                
                intermediates['layer_gate'] = gate.mean(dim=0).detach().cpu().tolist()
                
                if current_layer_id <= 3:
                    alpha_scale = 0.1 + 0.15 * gate
                else:
                    alpha_scale = 0.1 + 0.2 * gate
                
                modulations = []
                
                # 0. 模态专属Prompt（低层1-3）
                if current_layer_id in self.modality_inject_layers:
                    rgb_mod = self.modality_prompt_proj(rgb_feat.mean(dim=1, keepdim=True))
                    tir_mod = self.modality_prompt_proj(tir_feat.mean(dim=1, keepdim=True))
                    modality_mod = rgb_mod + tir_mod
                    modulations.append(('modality', 0, modality_mod))
                
                # 1. Consistency调制（中层5-6）
                if consistency_prompt is not None and current_layer_id in self.inject_layers:
                    if self.enable_cross_attn_modulation and self.cross_attn_modulation is not None:
                        consistency_mod = self.cross_attn_modulation(consistency_prompt, x_search)
                    else:
                        consistency_mod = consistency_prompt.mean(dim=1, keepdim=True)
                    modulations.append(('consistency', 1, consistency_mod))
                
                # 2. Temporal调制（高层8-9）
                if temporal_prompt is not None and current_layer_id in self.temporal_inject_layers:
                    if self.enable_cross_attn_modulation and self.cross_attn_modulation is not None:
                        temporal_mod = self.cross_attn_modulation(temporal_prompt, x_search)
                    else:
                        temporal_mod = temporal_prompt.mean(dim=1, keepdim=True)
                    modulations.append(('temporal', 2, temporal_mod))
                
                # 3. Mask调制（最高层9）
                if mask_prompt is not None and current_layer_id in self.mask_inject_layers:
                    if self.enable_cross_attn_modulation and self.cross_attn_modulation is not None:
                        mask_mod = self.cross_attn_modulation(mask_prompt, x_search)
                    else:
                        mask_mod = mask_prompt.mean(dim=1, keepdim=True)
                    modulations.append(('mask', 3, mask_mod))
                
                total_modulation = torch.zeros_like(x_search)
                for name, idx, modulation in modulations:
                    effective_gate = alpha_scale[:, idx:idx+1].unsqueeze(-1)
                    total_modulation = total_modulation + effective_gate * modulation
                
                x_search = x_search + total_modulation
                
                intermediates['branch_feats'] = {
                    'rgb_mean': rgb_feat.mean(dim=1),
                    'tir_mean': tir_feat.mean(dim=1),
                    'consistency_mean': consistency_prompt.mean(dim=1) if consistency_prompt is not None else None,
                    'temporal_mean': temporal_prompt.mean(dim=1) if temporal_prompt is not None else None,
                }
            
            return x_search, intermediates

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
        elif self.coop_strategy == 'gating_triple' and len(prompt_list) == 3:
            # 【v21修复】三提示门控融合：只有三个prompt都存在时才使用
            # 检查hasattr确保参数已初始化
            if not hasattr(self, 'coop_gate_triple'):
                raise RuntimeError(f"[inject] gating_triple策略需要coop_gate_triple参数，但未初始化！")
            gate = F.softmax(self.coop_gate_triple, dim=-1)  # [3]
            intermediates['coop_gate_triple'] = gate.detach().cpu().tolist()
            fused_prompt = consistency_prompt * gate[0] + temporal_prompt * gate[1] + mask_prompt * gate[2]
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
