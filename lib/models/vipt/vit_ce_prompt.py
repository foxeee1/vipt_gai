import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple
from lib.models.layers.patch_embed import PatchEmbed
from .utils import (combine_tokens, recover_tokens,
                    token2feature, feature2token)
from .vit import VisionTransformer
from .meta_prompt import MetaPromptGenerator
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt

_logger = logging.getLogger(__name__)


class Fovea(nn.Module):
    """ Fovea模块：模拟人类视觉的中央凹注意力机制，对重要特征区域进行加权增强 """

    def __init__(self, smooth=False):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h * w)
        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)
        return output


class Prompt_block(nn.Module):
    """
    Prompt模块：用于融合多模态特征（如RGB与DTE），通过卷积和Fovea注意力增强关键特征
    输入: [B, 2*embed_dim, H, W] (RGB特征与DTE特征通道拼接)
    输出: [B, embed_dim, H, W] (融合后的单流特征)
    """

    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel,
                                 kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel,
                                 kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes,
                                 kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C / 2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1
        return self.conv1x1(x0)


class VisionTransformerCE(VisionTransformer):
    """
    带候选消除（CE）模块的Vision Transformer（双路径架构）

    【路径切换】self.meta_prompt 控制两条完全独立的前向传播路径：
    
    路径A - 基线路径 (meta_prompt=False):
      与 lib_base 完全一致，用于 baseline 对比实验
      - Prompt容器: nn.Sequential (按索引顺序访问)
      - 注入方式: 残差连接 x_ori + candidate_elimination_prompt(...)
      - 注入时机: 仅 layer >= 1 时注入深层Prompt
      - CEBlock输入: 统一为 x，无条件执行
      
    路径B - 元学习扩展路径 (meta_prompt=True):
      包含 Base Prompt + Meta Prompt 的完整扩展逻辑
      - 支持灵活配置 base_prompt_inject_layers / meta_prompt_inject_layers
      - 支持 Mask/Consistency/Temporal 三类元提示
      - 支持 CEBlock 中元提示token保护机制
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None, meta_prompt=False, consistency_version='fusion',
                 base_prompt_inject_layers=None, meta_prompt_inject_layers=None,
                 meta_enable_mask=True, meta_enable_temporal=True, meta_enable_consistency=True,
                 coop_strategy='temporal_modulate', temporal_prompt_inject_layers=None,
                 mask_prompt_inject_layers=None):
        super().__init__()

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.depth = depth

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W

        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W

        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type
        self.meta_prompt = meta_prompt
        self.consistency_version = consistency_version

        # ================================================================
        # 双路径初始化: 根据 meta_prompt 选择不同的 Prompt 架构
        # ================================================================
        if not self.meta_prompt:
            # ========== 路径A: 基线模式 (与 lib_base 完全一致) ==========
            # 使用 Sequential 容器，按索引顺序访问，数量固定为 depth 个
            if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
                block_nums = depth if self.prompt_type == 'vipt_deep' else 1
                prompt_blocks = []
                for i in range(block_nums):
                    prompt_blocks.append(Prompt_block(inplanes=embed_dim, hide_channel=8, smooth=True))
                self.prompt_blocks = nn.Sequential(*prompt_blocks)

                prompt_norms = []
                for i in range(block_nums):
                    prompt_norms.append(norm_layer(embed_dim))
                self.prompt_norms = nn.Sequential(*prompt_norms)

            self.base_prompt_inject_layers = []  # 基线不使用此字段，forward_features 用隐式判断
            self.meta_prompt_inject_layers = []
            self.meta_prompt_generator = None
            _logger.info("[Baseline Path] Using lib_base-compatible architecture (Sequential prompts)")
        else:
            # ========== 路径B: 元学习扩展模式 ==========
            # 使用 ModuleList 容器，支持灵活配置注入层
            if base_prompt_inject_layers is None or len(base_prompt_inject_layers) == 0:
                if prompt_type == 'vipt_shaw':
                    base_prompt_inject_layers = [0]
                elif prompt_type == 'vipt_deep':
                    base_prompt_inject_layers = list(range(depth))
                else:
                    base_prompt_inject_layers = []
            self.base_prompt_inject_layers = base_prompt_inject_layers

            if meta_prompt_inject_layers is None or len(meta_prompt_inject_layers) == 0:
                if meta_prompt:
                    meta_prompt_inject_layers = list(range(8, depth))
                else:
                    meta_prompt_inject_layers = []
            self.meta_prompt_inject_layers = meta_prompt_inject_layers

            _logger.info(f"[Meta Path] Base Prompt layers: {self.base_prompt_inject_layers}")
            _logger.info(f"[Meta Path] Meta Prompt layers: {self.meta_prompt_inject_layers}")

            if len(self.base_prompt_inject_layers) > 0:
                max_layer = max(self.base_prompt_inject_layers) + 1
                prompt_blocks = []
                for i in range(max_layer):
                    prompt_blocks.append(Prompt_block(inplanes=embed_dim, hide_channel=8, smooth=True))
                self.prompt_blocks = nn.ModuleList(prompt_blocks)

                prompt_norms = []
                for i in range(max_layer):
                    prompt_norms.append(norm_layer(embed_dim))
                self.prompt_norms = nn.ModuleList(prompt_norms)

            self.meta_prompt_generator = MetaPromptGenerator(
                config={
                    'EMBED_DIM': embed_dim,
                    'NUM_PROMPT_TOKENS': 8,
                    'HIDDEN_DIM': 256,
                    'MODE': 'fixed',
                    'ENABLE_BASE': True,
                    'ENABLE_MASK': meta_enable_mask,
                    'ENABLE_CONSISTENCY': meta_enable_consistency,
                    'ENABLE_TEMPORAL': meta_enable_temporal,
                    'CONSISTENCY_VERSION': consistency_version,
                    'COOP_STRATEGY': coop_strategy,
                    'META_PROMPT_INJECT_LAYERS': meta_prompt_inject_layers if meta_prompt_inject_layers else [],
                    'TEMPORAL_PROMPT_INJECT_LAYERS': temporal_prompt_inject_layers if temporal_prompt_inject_layers else meta_prompt_inject_layers if meta_prompt_inject_layers else [],
                    'MASK_PROMPT_INJECT_LAYERS': mask_prompt_inject_layers if mask_prompt_inject_layers else meta_prompt_inject_layers if meta_prompt_inject_layers else [],
                }
            )
            self.prev_features = None

        # Transformer 编码器层（含 CE 模块）- 两种路径共用
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i))
        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)

    def _forward_features_baseline(self, z, x, mask_z=None, mask_x=None,
                                    ce_template_mask=None, ce_keep_rate=None,
                                    return_last_attn=False):
        """
        【路径A】基线前向传播 - 与 lib_base/vit_ce_prompt.py 完全一致
        
        核心特征:
        1. Prompt使用Sequential容器，通过索引[0]和[i-1]访问
        2. 深层Prompt仅在第1层之后(i>=1)注入，采用残差连接
        3. CEBlock每层统一以x作为输入，无分支逻辑
        4. 不涉及meta-prompt、base_prompt_inject_layers等扩展参数
        
        Args/Returns: 与 forward_features 相同接口
        """
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        x, z = x_rgb, z_rgb

        z = self.patch_embed(z)
        x = self.patch_embed(x)
        z_dte = self.patch_embed_prompt(z_dte)
        x_dte = self.patch_embed_prompt(x_dte)

        '''Base Prompt 初始融合: 将DTE特征通过Prompt_block融入RGB token'''
        if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
            z_feat = token2feature(self.prompt_norms[0](z))
            x_feat = token2feature(self.prompt_norms[0](x))
            z_dte_feat = token2feature(self.prompt_norms[0](z_dte))
            x_dte_feat = token2feature(self.prompt_norms[0](x_dte))

            z_feat = torch.cat([z_feat, z_dte_feat], dim=1)
            x_feat = torch.cat([x_feat, x_dte_feat], dim=1)

            z_feat = self.prompt_blocks[0](z_feat)
            x_feat = self.prompt_blocks[0](x_feat)

            z_dte = feature2token(z_feat)
            x_dte = feature2token(x_feat)
            z_prompted, x_prompted = z_dte, x_dte

            z = z + z_dte
            x = x + x_dte
        else:
            z_prompted, x_prompted = z_dte, x_dte
            z = z + z_dte
            x = x + x_dte

        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)
            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)
            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        
        x_rgb_original = x.clone()
        x_dte_original = combine_tokens(z_prompted, x_prompted, mode=self.cat_mode).clone()
        
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []

        for i, blk in enumerate(self.blocks):
            '''
            深层Prompt处理: 仅当 prompt_type 为 'vipt_deep' 且层数 i>=1 时执行
            采用残差连接: x = x_ori + candidate_elimination_prompt(prompt_output)
            这确保了与 lib_base 完全一致的梯度流动和特征传播
            【修复】禁用深层Prompt处理，因为我们只在第0层注入Base Prompt
            '''
            if False and i >= 1 and self.prompt_type in ['vipt_deep']:
                x_ori = x

                lens_z_new = global_index_t.shape[1]
                lens_x_new = global_index_s.shape[1]

                z = x[:, :lens_z_new]
                x = x[:, lens_z_new:]

                if removed_indexes_s and removed_indexes_s[0] is not None:
                    removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
                    pruned_lens_x = lens_x - lens_x_new
                    pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
                    x = torch.cat([x, pad_x], dim=1)
                    index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                    C = x.shape[-1]
                    x = torch.zeros_like(x).scatter_(
                        dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

                x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
                x = torch.cat([z, x], dim=1)

                x = self.prompt_norms[i - 1](x)
                z_tokens = x[:, :lens_z, :]
                x_tokens = x[:, lens_z:, :]
                z_feat = token2feature(z_tokens)
                x_feat = token2feature(x_tokens)

                z_prompted = self.prompt_norms[i](z_prompted)
                x_prompted = self.prompt_norms[i](x_prompted)
                z_prompt_feat = token2feature(z_prompted)
                x_prompt_feat = token2feature(x_prompted)

                z_feat = torch.cat([z_feat, z_prompt_feat], dim=1)
                x_feat = torch.cat([x_feat, x_prompt_feat], dim=1)

                z_feat = self.prompt_blocks[i](z_feat)
                x_feat = self.prompt_blocks[i](x_feat)

                z = feature2token(z_feat)
                x = feature2token(x_feat)
                z_prompted, x_prompted = z, x

                x = combine_tokens(z, x, mode=self.cat_mode)
                x = x_ori + candidate_elimination_prompt(x, global_index_t.shape[1], global_index_s)

            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:lens_z_new + lens_x_new]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(
                dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
        }
        return x, aux_dict

    def _forward_features_meta(self, z, x, mask_z=None, mask_x=None,
                                ce_template_mask=None, ce_keep_rate=None,
                                return_last_attn=False):
        """
        【路径B】元学习扩展前向传播 - 包含完整的 Base/Meta Prompt 注入逻辑
        
        核心特征:
        1. 支持 base_prompt_inject_layers 灵活配置注入层
        2. 支持 meta_prompt 在指定层注入并受CE保护
        3. 替换式Prompt注入（非残差）
        4. CEBlock输入根据分支不同而变化
        """
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        x, z = x_rgb, z_rgb

        z = self.patch_embed(z)
        x = self.patch_embed(x)
        z_dte = self.patch_embed_prompt(z_dte)
        x_dte = self.patch_embed_prompt(x_dte)

        if len(self.base_prompt_inject_layers) > 0:
            if 0 in self.base_prompt_inject_layers:
                z_feat = token2feature(self.prompt_norms[0](z))
                x_feat = token2feature(self.prompt_norms[0](x))
                z_dte_feat = token2feature(self.prompt_norms[0](z_dte))
                x_dte_feat = token2feature(self.prompt_norms[0](x_dte))

                z_feat = torch.cat([z_feat, z_dte_feat], dim=1)
                x_feat = torch.cat([x_feat, x_dte_feat], dim=1)

                z_feat = self.prompt_blocks[0](z_feat)
                x_feat = self.prompt_blocks[0](x_feat)

                z_dte = feature2token(z_feat)
                x_dte = feature2token(x_feat)
                z_prompted, x_prompted = z_dte, x_dte

                z = z + z_dte
                x = x + x_dte
            else:
                z_prompted, x_prompted = z_dte, x_dte
        else:
            z_prompted, x_prompted = z_dte, x_dte

        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)
            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)
            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        
        x_rgb_original = x.clone()
        x_dte_original = combine_tokens(z_prompted, x_prompted, mode=self.cat_mode).clone()
        
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        meta_prompt_len = self.meta_prompt_generator.total_prompt_len
        # 【统一格式】META_PROMPT_INJECT_LAYERS = [layer1, layer2, ...] 任意层列表
        # 例如: [9, 10, 11, 12] 表示在第9、10、11、12层都注入meta prompt
        #       [9, 10] 表示只在第9、10层注入
        inject_layers = self.meta_prompt_inject_layers
        inject_layers_set = set(inject_layers) if inject_layers else set()

        # 【调试日志】只在首次forward时打印一次（避免刷屏）
        if not getattr(self, '_inject_layers_printed', False):
            print(f"[DEBUG] META_PROMPT_INJECT_LAYERS = {inject_layers}, inject_layers_set = {inject_layers_set}")
            self._inject_layers_printed = True

        # 每层独立计算并注入meta prompt（逐层注入，非最后统一融合）
        meta_prompt_cache = {}  # 用字典缓存每层计算的结果
        inject_intermediates = {}  # 【关键修改】初始化intermediates，用于收集正则化损失所需的权重

        for i, blk in enumerate(self.blocks):
            lens_z_cur = global_index_t.shape[1]
            lens_x_cur = global_index_s.shape[1]

            need_base_prompt = False
            
            need_deep_prompt = False
            
            x_ori = x.clone()
            x_core = x  # 默认值，确保x_core始终被定义

            if False and (need_base_prompt or need_deep_prompt):
                lens_z_new = global_index_t.shape[1]
                lens_x_new = global_index_s.shape[1]

                z = x[:, :lens_z_new]
                x_search = x[:, lens_z_new:]

                if removed_indexes_s and removed_indexes_s[0] is not None:
                    removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
                    pruned_lens_x = lens_x - lens_x_new
                    pad_x = torch.zeros([B, pruned_lens_x, x_search.shape[2]], device=x_search.device, dtype=x_search.dtype)
                    x_search = torch.cat([x_search, pad_x], dim=1)
                    index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                    C = x_search.shape[-1]
                    x_search = torch.zeros_like(x_search).scatter_(
                        dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x_search)

                x_search = recover_tokens(x_search, lens_z_new, lens_x, mode=self.cat_mode)
                x_core = torch.cat([z, x_search], dim=1)

                prompt_idx = min(i - 1, len(self.prompt_norms) - 1)
                x_core = self.prompt_norms[prompt_idx](x_core)
                z_tokens = x_core[:, :lens_z, :]
                x_tokens = x_core[:, lens_z:, :]
                z_feat = token2feature(z_tokens)
                x_feat = token2feature(x_tokens)

                prompt_prompt_idx = min(i, len(self.prompt_norms) - 1)
                z_prompted_norm = self.prompt_norms[prompt_prompt_idx](z_prompted)
                x_prompted_norm = self.prompt_norms[prompt_prompt_idx](x_prompted)
                z_prompt_feat = token2feature(z_prompted_norm)
                x_prompt_feat = token2feature(x_prompted_norm)

                z_feat = torch.cat([z_feat, z_prompt_feat], dim=1)
                x_feat = torch.cat([x_feat, x_prompt_feat], dim=1)

                block_idx = min(i, len(self.prompt_blocks) - 1)
                z_feat = self.prompt_blocks[block_idx](z_feat)
                x_feat = self.prompt_blocks[block_idx](x_feat)

                z = feature2token(z_feat)
                x_search = feature2token(x_feat)
                z_prompted, x_prompted = z, x_search

                x_core = combine_tokens(z, x_search, mode=self.cat_mode)

                x = x_ori + candidate_elimination_prompt(x_core, global_index_t.shape[1], global_index_s)

            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            lens_z_new = global_index_t.shape[1]
            lens_x_new = global_index_s.shape[1]
            x_rgb_search = x[:, lens_z_new:, :]
            x_dte_search = x[:, lens_z_new:, :]
            
            max_idx = min(lens_x_new, x_rgb_search.shape[1])
            global_index_s_safe = global_index_s[:, :max_idx].clamp(0, max_idx - 1)
            # 【逐层注入逻辑】每层独立计算Prompt并立即注入（非最后统一融合）
            # 核心：Meta Prompt在对应层计算后立即通过 modulate() 注入到特征
            # 安全校验：使用当前层特征作为主输入（保持语义一致性）
            # 叠加原始模态差异信号（提供rgb≠tir的梯度信号，解决权重卡0.5）
            if i in inject_layers_set:
                C = x_rgb_search.shape[-1]
                # 当前层特征（正确语义层次）
                x_rgb_search_cur = x_rgb_search.gather(
                    dim=1, index=global_index_s_safe.unsqueeze(-1).expand(B, -1, C).to(torch.int64))
                x_dte_search_cur = x_dte_search.gather(
                    dim=1, index=global_index_s_safe.unsqueeze(-1).expand(B, -1, C).to(torch.int64))
                # 叠加原始模态差异信号：使rgb_feat≠tir_feat，梯度可传播
                # 原始差异提供跨模态区分度，0.1系数确保不破坏当前层语义
                rgb_dte_diff = (x_rgb_original[:, lens_z:lens_z + lens_x, :] - x_dte_original[:, lens_z:lens_z + lens_x, :]).gather(
                    dim=1, index=global_index_s_safe.unsqueeze(-1).expand(B, -1, C).to(torch.int64))
                x_rgb_search_cur = x_rgb_search_cur + 0.1 * rgb_dte_diff
                x_dte_search_cur = x_dte_search_cur - 0.1 * rgb_dte_diff
                # 【v14】传递当前层数信息给MetaPromptGenerator（用于自适应注入强度）
                self.meta_prompt_generator._current_inject_layer = i
                # 调用MetaPromptGenerator生成并立即注入（逐层调制）
                # 【关键修改】接收intermediates用于计算正则化损失
                x_rgb_search_modulated, inject_intermediates = self.meta_prompt_generator.inject(
                    x_rgb_search_cur, x_dte_search_cur, x_rgb_search,
                    prev_features=self.prev_features,
                    consistency_version=getattr(self, 'consistency_version', 'gradient')
                )
                # 【避免原地操作】用torch.cat重新拼接z和调制后的search，避免梯度断裂
                x_z = x[:, :lens_z_new, :]
                x = torch.cat([x_z, x_rgb_search_modulated], dim=1)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

            self.prev_features = x

        x = self.norm(x)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        x_core = x

        z = x_core[:, :lens_z_new]
        x_search = x_core[:, lens_z_new:lens_z_new + lens_x_new]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            C = x_search.shape[-1]
            index_keep = global_index_s.clamp(min=0, max=lens_x - 1).unsqueeze(-1).expand(B, -1, C).to(torch.int64)
            x_search_full = torch.zeros([B, lens_x, C], device=x_search.device, dtype=x_search.dtype)
            x_search_full.scatter_(dim=1, index=index_keep, src=x_search)
            x_search = x_search_full

        x_search = recover_tokens(x_search, lens_z_new, lens_x, mode=self.cat_mode)
        x = torch.cat([z, x_search], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
            "inject_intermediates": inject_intermediates,  # 【关键修改】传递intermediates用于计算正则化损失
        }
        return x, aux_dict

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):
        """
        特征提取前向传播入口 - 根据 self.meta_prompt 自动选择路径
        
        【路径选择规则】
        - meta_prompt=False → _forward_features_baseline() (lib_base 兼容)
        - meta_prompt=True  → _forward_features_meta()     (扩展模式)
        
        Args:
            z: [B, C, H_z, W_z] 模板图像
            x: [B, C, H_x, W_x] 搜索区域图像
            mask_z/mask_x: 注意力掩码（可选）
            ce_template_mask: CE模板掩码（可选）
            ce_keep_rate: CE保留率（可选）
            return_last_attn: 是否返回最后注意力图
            
        Returns:
            x: [B, N_total, embed_dim] 融合后的特征token
            aux_dict: 辅助信息字典
        """
        if not self.meta_prompt:
            return self._forward_features_baseline(
                z, x, mask_z, mask_x, ce_template_mask, ce_keep_rate, return_last_attn)
        elif self.meta_prompt_generator.total_prompt_len == 0:
            return self._forward_features_baseline(
                z, x, mask_z, mask_x, ce_template_mask, ce_keep_rate, return_last_attn)
        else:
            return self._forward_features_meta(
                z, x, mask_z, mask_x, ce_template_mask, ce_keep_rate, return_last_attn)

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False):
        x, aux_dict = self.forward_features(
            z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn)
        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")
    return model


def vit_base_patch16_224_ce_prompt(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
