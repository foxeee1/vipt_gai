# 导入functools模块中的partial函数，用于创建部分应用的函数
from functools import partial

# 导入PyTorch及其神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从timm库导入多层感知机(Mlp)、随机深度(DropPath)和权重初始化函数
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
# 从当前目录导入实用函数
from .utils import combine_tokens, token2feature, feature2token
# 从lib.models.layers模块导入PatchEmbed类，用于图像分块嵌入
from lib.models.layers.patch_embed import PatchEmbed
# 从当前目录导入VisionTransformer基类
from .vit import VisionTransformer


# 定义注意力机制模块
class Attention(nn.Module):
    # 初始化注意力模块
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()  # 调用父类初始化
        self.num_heads = num_heads  # 注意力头的数量
        head_dim = dim // num_heads  # 每个注意力头的维度
        self.scale = head_dim ** -0.5  # 缩放因子，用于缩放注意力矩阵

        # 定义线性变换层，用于生成查询(q)、键(k)和值(v)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 定义注意力dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        # 定义输出投影层
        self.proj = nn.Linear(dim, dim)
        # 定义投影后的dropout层
        self.proj_drop = nn.Dropout(proj_drop)

    # 前向传播函数
    def forward(self, x, return_attention=False):
        B, N, C = x.shape  # 获取输入形状：批次大小(B)、序列长度(N)、特征维度(C)
        # 应用线性变换并重塑，得到查询、键、值
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离查询、键、值张量

        # 计算注意力权重：q与k的转置相乘，再缩放
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 应用softmax归一化
        attn = self.attn_drop(attn)  # 应用dropout

        # 应用注意力权重到值上，并重塑回原始形状
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # 应用输出投影
        x = self.proj_drop(x)  # 应用投影后dropout

        # 如果需要返回注意力权重，则同时返回特征和注意力
        if return_attention:
            return x, attn
        return x  # 否则只返回特征


# 定义Transformer的基本块
class Block(nn.Module):

    # 初始化Block
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()  # 调用父类初始化
        # 定义第一层归一化
        self.norm1 = norm_layer(dim)
        # 定义注意力模块
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # 定义随机深度层，如果drop_path>0则使用DropPath，否则使用恒等变换
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 定义第二层归一化
        self.norm2 = norm_layer(dim)
        # 计算MLP隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 定义MLP模块
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # 前向传播函数
    def forward(self, x, return_attention=False):
        # 如果需要返回注意力权重
        if return_attention:
            # 应用注意力模块并返回特征和注意力
            feat, attn = self.attn(self.norm1(x), True)
            # 残差连接并应用随机深度
            x = x + self.drop_path(feat)
            # 应用MLP并再次残差连接
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn  # 返回特征和注意力
        else:
            # 标准Transformer块前向传播
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x  # 返回特征


# 定义掩码增强模块
class MaskEnhancement(nn.Module):
    """掩码增强模块"""
    def __init__(self, dim, hide_channel=64):
        super().__init__()  # 调用父类初始化
        # 定义第一个MLP，用于处理输入特征
        self.mlp1 = Mlp(in_features=dim, hidden_features=hide_channel, out_features=hide_channel)
        # 定义第二个MLP，用于生成掩码
        self.mlp2 = Mlp(in_features=hide_channel, hidden_features=hide_channel, out_features=1)
        # 定义投影层，用于生成掩码提示
        self.proj_mask = nn.Linear(hide_channel * 2, dim)
        # 定义提示token的数量
        self.prompt_tokens = 8
        # 定义提示投影层
        self.proj_prompt = nn.Linear(hide_channel, dim)
        
    def forward(self, T_r, T_x):
        # T_r: [B, N_r, C]，模板特征
        # T_x: [B, N_x, C]，搜索特征
        
        # 分别处理模板和搜索特征
        # 对模板特征进行处理
        D_r = self.mlp1(T_r)
        # 对搜索特征进行处理
        D_x = self.mlp1(T_x)
        
        # 池化模板特征使其与搜索特征大小匹配
        if D_r.shape[1] != D_x.shape[1]:
            # 使用平均池化将较小的特征调整为较大的特征大小
            # 计算池化比例
            scale = D_x.shape[1] // D_r.shape[1]
            if scale > 1:
                # 上采样模板特征
                D_r = D_r.unsqueeze(2)  # [B, N_r, 1, C]
                D_r = D_r.repeat(1, 1, scale, 1)  # [B, N_r, scale, C]
                D_r = D_r.view(D_r.shape[0], -1, D_r.shape[-1])  # [B, N_r*scale, C]
                
                # 如果仍然不匹配，截断或填充
                if D_r.shape[1] > D_x.shape[1]:
                    D_r = D_r[:, :D_x.shape[1], :]
                elif D_r.shape[1] < D_x.shape[1]:
                    pad = torch.zeros(D_r.size(0), D_x.shape[1] - D_r.shape[1], D_r.size(2), device=D_r.device)
                    D_r = torch.cat([D_r, pad], dim=1)
            else:
                # 直接平均池化较大的特征
                D_x = D_x.view(D_x.shape[0], D_r.shape[1], -1, D_x.shape[-1])  # [B, N_r, scale, C]
                D_x = D_x.mean(dim=2)  # [B, N_r, C]
        
        # 计算跨模态差分特征的绝对值
        D = torch.abs(D_r - D_x)
        
        # 通过MLP生成掩码
        M = self.mlp2(D).sigmoid()  # [B, N, 1]，应用sigmoid归一化到0-1
        
        # 生成掩码提示
        Z_mask = torch.cat([D, M.repeat(1, 1, D.shape[-1])], dim=-1)  # 从最后一个维度连接差分特征和掩码
        P_mask = self.proj_mask(Z_mask)  # 投影到目标维度
        
        # 使用proj_prompt层增强掩码提示
        enhanced_mask = self.proj_prompt(D_r.mean(dim=1, keepdim=True).repeat(1, self.prompt_tokens, 1))
        P_mask = P_mask.mean(dim=1, keepdim=True).repeat(1, self.prompt_tokens, 1) + enhanced_mask
        
        return P_mask, M  # 返回掩码提示和掩码


# 定义一致性增强模块
class ConsistencyEnhancement(nn.Module):
    """
    一致性增强模块（改进版）
    结合：
    1. 梯度一致性（gradient consistency）
    2. 协方差一致性（covariance alignment）
    用于生成更稳健的跨模态一致性提示。
    """
    def __init__(self, embed_dim=384, prompt_len=8):
        super().__init__()  # 调用父类初始化
        self.prompt_len = prompt_len  # 提示token数量
        self.embed_dim = embed_dim  # 特征维度

        # 最终投影层：将多种一致性特征映射到 prompt
        self.proj = nn.Linear(embed_dim * 4, embed_dim * prompt_len)

    def compute_gradient(self, T):
        """计算 token 的空间梯度：沿 token 顺序求差值"""
        return T[:, 1:, :] - T[:, :-1, :]

    def compute_covariance(self, T):
        """计算每个 batch 的协方差矩阵，并降维到 token 维度"""
        B, N, C = T.shape
        T_centered = T - T.mean(dim=1, keepdim=True)
        # 协方差矩阵 (B, C, C)
        cov = torch.matmul(T_centered.transpose(1, 2), T_centered) / (N - 1)
        # 简化：取对角线（方差信息）并 repeat 到 token 维度
        diag = torch.diagonal(cov, dim1=1, dim2=2) # (B, C)
        return diag.unsqueeze(1).repeat(1, N, 1)

    def forward(self, T_r, T_x):
        # T_r: [B, N_r, C]，模板特征
        # T_x: [B, N_x, C]，搜索特征
        B, N_r, C = T_r.shape
        N_x = T_x.shape[1]
        
        # -----------------------------------------------
        # 1) 梯度一致性
        # -----------------------------------------------
        G_r = self.compute_gradient(T_r)
        G_x = self.compute_gradient(T_x)

        # pad 回 token 数量（前面补零）
        pad_r = torch.zeros(T_r.size(0), 1, T_r.size(2), device=T_r.device)
        pad_x = torch.zeros(T_x.size(0), 1, T_x.size(2), device=T_x.device)
        G_r = torch.cat([pad_r, G_r], dim=1)
        G_x = torch.cat([pad_x, G_x], dim=1)

        # 计算梯度方向一致性
        # 如果特征大小不匹配，分别计算后再池化到相同大小
        if G_r.shape[1] != G_x.shape[1]:
            # 计算各自的梯度特征
            G_r_processed = F.normalize(G_r, dim=-1)
            G_x_processed = F.normalize(G_x, dim=-1)
            
            # 计算全局梯度一致性
            C_grad_r = G_r_processed.mean(dim=1, keepdim=True)  # [B, 1, C]
            C_grad_x = G_x_processed.mean(dim=1, keepdim=True)  # [B, 1, C]
            C_grad = F.cosine_similarity(C_grad_r, C_grad_x, dim=-1, eps=1e-6).unsqueeze(-1)  # [B, 1, 1]
            
            # 扩展到各自的特征大小和维度
            C_grad_r = C_grad.repeat(1, N_r, T_r.size(-1))  # [B, N_r, C]
            C_grad_x = C_grad.repeat(1, N_x, T_x.size(-1))  # [B, N_x, C]
        else:
            C_grad = F.cosine_similarity(G_r, G_x, dim=-1, eps=1e-6).unsqueeze(-1)
            C_grad_r = C_grad.repeat(1, 1, T_r.size(-1))  # 扩展到特征维度
            C_grad_x = C_grad_r  # 特征大小匹配，两者相同

        # -----------------------------------------------
        # 2) 协方差一致性
        # -----------------------------------------------
        Cov_r = self.compute_covariance(T_r)
        Cov_x = self.compute_covariance(T_x)

        # 协方差差异（越小越一致）
        # 如果特征大小不匹配，使用全局均值比较
        if Cov_r.shape[1] != Cov_x.shape[1]:
            # 计算全局协方差信息
            Cov_r_global = Cov_r.mean(dim=1, keepdim=True)  # [B, 1, C]
            Cov_x_global = Cov_x.mean(dim=1, keepdim=True)  # [B, 1, C]
            # 计算全局差异并扩展到各自的特征大小
            C_cov_r = -torch.abs(Cov_r_global - Cov_r_global.repeat(1, N_r, 1))
            C_cov_x = -torch.abs(Cov_x_global - Cov_x_global.repeat(1, N_x, 1))
        else:
            C_cov = -torch.abs(Cov_r - Cov_x) # 负号让其成为“越大越一致”
            C_cov_r = C_cov
            C_cov_x = C_cov  # 特征大小匹配，两者相同

        # -----------------------------------------------
        # 3) 构造提示输入特征 Z
        # -----------------------------------------------
        # 如果特征大小不匹配，分别处理然后合并
        if T_r.shape[1] != T_x.shape[1]:
            # 分别处理模板和搜索特征
            # 模板特征处理：添加T_x的全局信息
            Z_r = torch.cat([
                T_r,
                T_x.mean(dim=1, keepdim=True).repeat(1, T_r.shape[1], 1),  # T_x的全局平均特征
                C_grad_r,  # 使用模板的梯度一致性特征
                C_cov_r    # 使用模板的协方差一致性特征
            ], dim=-1) # (B, N_r, 4C)
            
            # 搜索特征处理：添加T_r的全局信息
            Z_x = torch.cat([
                T_x,
                T_r.mean(dim=1, keepdim=True).repeat(1, T_x.shape[1], 1),  # T_r的全局平均特征
                C_grad_x,  # 使用搜索的梯度一致性特征
                C_cov_x    # 使用搜索的协方差一致性特征
            ], dim=-1) # (B, N_x, 4C)
            
            # 投影到 prompt 维度
            P_r = self.proj(Z_r).view(-1, T_r.size(1), self.prompt_len, self.embed_dim)
            P_x = self.proj(Z_x).view(-1, T_x.size(1), self.prompt_len, self.embed_dim)
            
            # 进行 token 汇聚（池化）
            P_r = P_r.mean(dim=1) # (B, prompt_len, embed_dim)
            P_x = P_x.mean(dim=1) # (B, prompt_len, embed_dim)
            
            # 合并两种特征的提示
            P = (P_r + P_x) / 2
        else:
            # 特征大小匹配，直接拼接
            Z = torch.cat([
                T_r,
                T_x,
                C_grad.repeat(1, 1, T_r.size(-1)),  # 梯度一致性特征
                C_cov  # 协方差一致性特征
            ], dim=-1) # (B, N, 4C)

            # 投影到 prompt
            P = self.proj(Z).view(-1, T_r.size(1), self.prompt_len, self.embed_dim)

            # 进行 token 汇聚（池化）
            P = P.mean(dim=1) # (B, prompt_len, embed_dim)
        
        return P


# 定义时序增强模块
class TemporalEnhancement(nn.Module):
    """注意力驱动的时空增强模块
    基于注意力机制，通过跨帧特征交互实现判别性目标特征的实时更新
    
    Args:
        dim: 特征维度
        hide_channel: 隐藏层通道数（未使用，保持兼容性）
        num_heads: 注意力头数
        prompt_tokens: 时序提示token数量
        history_len: 历史特征长度（帧数）
    """
    def __init__(self, dim, hide_channel=64, num_heads=4, prompt_tokens=8, history_len=3):
        super().__init__()
        
        # 定义注意力机制，用于建立当前帧与历史帧的关联
        self.temporal_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 特征更新投影层 - 用于融合当前与历史特征
        self.update_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 融合当前与历史注意力特征
            nn.LayerNorm(dim),        # 层归一化
            nn.GELU(),                # 激活函数
            nn.Linear(dim, dim)        # 输出投影
        )
        
        # 历史特征更新层 - 用于更新存储的历史特征
        self.history_update_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 融合历史与当前特征
            nn.LayerNorm(dim),        # 层归一化
            nn.GELU(),                # 激活函数
            nn.Linear(dim, dim)        # 输出投影
        )
        
        # 时序提示生成层
        self.prompt_tokens = prompt_tokens
        self.prompt_proj = nn.Linear(dim, dim)
        
        # 历史特征长度
        self.history_len = history_len
        
        # 注册空历史特征缓冲区
        self.register_buffer('empty_history', torch.zeros(1, history_len, dim))
        
    def forward(self, T_r, T_x, T_hist=None, M_hist=None):
        """前向传播函数
        
        Args:
            T_r: 当前帧模板特征，形状为 [B, N_r, C]
            T_x: 当前帧搜索特征，形状为 [B, N_x, C]
            T_hist: 历史帧特征，形状为 [B, H, C] (可选)，H为历史帧数
            M_hist: 历史掩码信息，形状为 [B, H, C] (可选)
            
        Returns:
            P_temp: 时序增强提示，形状为 [B, prompt_tokens, C]
            updated_hist: 更新后的历史特征，形状为 [B, H, C]，H为历史帧数
        """
        B, N, C = T_r.shape
        N_x = T_x.shape[1]
        
        # 初始化历史特征（如果未提供）
        if T_hist is None:
            # 使用空历史特征
            T_hist = self.empty_history.repeat(B, 1, 1).to(T_r.device)
        
        # 1. 计算当前帧与历史帧的注意力交互
        # 获取当前帧的全局特征（模板+搜索的平均）
        current_global = torch.cat([T_r.mean(dim=1, keepdim=True), T_x.mean(dim=1, keepdim=True)], dim=1)  # [B, 2, C]
        current_global = current_global.mean(dim=1, keepdim=True)  # [B, 1, C] - 当前帧全局特征
        
        # 应用跨帧注意力：当前帧全局特征与历史帧特征的交互
        attn_output, _ = self.temporal_attn(
            query=current_global,     # 查询：当前帧全局特征
            key=T_hist,               # 键：历史帧特征
            value=T_hist              # 值：历史帧特征
        )  # [B, 1, C] - 历史注意力特征
        
        # 2. 更新当前帧特征表示（融合当前与历史注意力特征）
        # 将注意力特征扩展到与当前特征相同的空间维度
        attn_output_expanded = attn_output.repeat(1, N + N_x, 1)  # [B, N_r + N_x, C]
        current_features = torch.cat([T_r, T_x], dim=1)  # [B, N_r + N_x, C]
        
        enhanced_features = torch.cat([current_features, attn_output_expanded], dim=-1)  # [B, N_r + N_x, 2C]
        updated_features = self.update_proj(enhanced_features)  # [B, N_r + N_x, C]
        
        # 3. 生成时序提示（从融合后的特征中提取）
        global_temp_info = updated_features.mean(dim=1, keepdim=True)  # [B, 1, C]
        P_temp = self.prompt_proj(global_temp_info)  # [B, 1, C]
        P_temp = P_temp.repeat(1, self.prompt_tokens, 1)  # [B, prompt_tokens, C]
        
        # 4. 更新历史特征（移除最旧的历史，使用history_update_proj融合历史与当前特征）
        # 移除最旧的历史特征
        T_hist_updated = T_hist[:, 1:, :]  # [B, history_len-1, C]
        # 将当前帧全局特征扩展到与历史特征匹配的形状
        current_global_expanded = current_global.repeat(1, self.history_len-1, 1)  # [B, history_len-1, C]
        # 使用history_update_proj融合历史与当前特征
        fused_history = self.history_update_proj(
            torch.cat([T_hist_updated, current_global_expanded], dim=-1)  # [B, history_len-1, 2C]
        )  # [B, history_len-1, C]
        # 将当前帧全局特征添加到更新后的历史特征中
        updated_hist = torch.cat([fused_history, current_global], dim=1)  # [B, history_len, C]
        
        # 5. 使用更新后的历史特征增强当前帧特征，使history_update_proj参与损失计算
        # 将更新后的历史特征通过池化和扩展与当前特征融合
        hist_pooled = updated_hist.mean(dim=1, keepdim=True)  # [B, 1, C]
        hist_expanded = hist_pooled.repeat(1, N + N_x, 1)  # [B, N_r + N_x, C]
        # 将历史特征与增强特征融合，确保history_update_proj的梯度能够回传
        # 增加融合权重，使历史特征的影响更明显
        final_features = updated_features + 0.5 * hist_expanded  # 加权融合，使历史特征影响当前特征
        
        # 6. 使用融合历史后的特征替换原来的更新特征，确保history_update_proj的梯度能够传播
        updated_features = final_features  # 直接使用融合后的特征
        
        # 7. 重新计算时序提示，使用包含历史信息的特征
        global_temp_info = updated_features.mean(dim=1, keepdim=True)  # [B, 1, C]
        P_temp = self.prompt_proj(global_temp_info)  # [B, 1, C]
        P_temp = P_temp.repeat(1, self.prompt_tokens, 1)  # [B, prompt_tokens, C]
        
        return P_temp, updated_hist  # 返回时序提示和更新后的历史特征


# 定义Meta-Consistency Prompt Vision Transformer类，继承自VisionTransformer
class VisionTransformerMetaP(VisionTransformer):
    """Meta-Consistency Prompt Vision Transformer"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
        super().__init__()  # 调用父类初始化
        self.num_classes = num_classes  # 分类类别数量
        self.num_features = self.embed_dim = embed_dim  # 特征维度
        self.num_tokens = 2 if distilled else 1  # 分类token数量（蒸馏模型有额外的token）
        # 设置归一化层，默认为LayerNorm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # 设置激活函数层，默认为GELU
        act_layer = act_layer or nn.GELU

        # 定义标准patch嵌入层
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        # 定义多模态patch嵌入层
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # 定义分类token参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 定义蒸馏token参数（如果需要）
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # 定义位置dropout层
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 计算搜索区域的patch数量
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W
        # 计算模板区域的patch数量
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W
        
        # 定义模板和搜索区域的位置嵌入
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        
        # 设置基础参数
        self.prompt_type = prompt_type  # 提示类型
        self.cat_mode = 'direct'  # 合并模式

        # 初始化三层增强模块
        self.mask_enhancement = MaskEnhancement(embed_dim)
        self.consistency_enhancement = ConsistencyEnhancement(embed_dim)
        self.temporal_enhancement = TemporalEnhancement(
            dim=embed_dim,               # 特征维度
            hide_channel=64,             # 隐藏层通道数
            num_heads=4,                 # 注意力头数
            prompt_tokens=8,             # 时序提示token数量
            history_len=3                # 历史特征长度（帧数）
        )
        
        # 初始化基础提示
        self.base_prompt_tokens = 6  # 基础提示token数量
        self.base_prompt = nn.Parameter(torch.zeros(1, self.base_prompt_tokens, embed_dim))  # 基础提示参数
        trunc_normal_(self.base_prompt, std=.02)  # 使用truncated normal初始化

        # 生成随机深度率列表
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # 构建Transformer块序列
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        # 定义最终的归一化层
        self.norm = norm_layer(embed_dim)

        # 初始化权重
        self.init_weights(weight_init)
        
        # 冻结前8层参数
        for i in range(8):
            for param in self.blocks[i].parameters():
                param.requires_grad = False

    # 特征提取前向传播
    def forward_features(self, z, x, mask_z=None, mask_x=None, 
                        return_last_attn=False, 
                        T_hist_z=None, T_hist_x=None, M_hist=None):
        # 获取输入形状
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # 分离RGB和辅助模态（深度/热成像等）
        x_rgb = x[:, :3, :, :]  # 前3通道为RGB
        z_rgb = z[:, :3, :, :]
        x_dte = x[:, 3:, :, :]  # 后通道为辅助模态
        z_dte = z[:, 3:, :, :]
        x, z = x_rgb, z_rgb  # 暂时使用RGB作为主要输入

        # 提取特征
        z = self.patch_embed(z)  # 处理模板图像
        x = self.patch_embed(x)  # 处理搜索图像
        
        # 只有当存在辅助模态时才处理
        if x_dte.shape[1] > 0 and z_dte.shape[1] > 0:
            z_dte = self.patch_embed_prompt(z_dte)  # 处理模板的辅助模态
            x_dte = self.patch_embed_prompt(x_dte)  # 处理搜索的辅助模态
            
            # 融合多模态特征
            z = z + z_dte  # 模板RGB与辅助模态特征相加
            x = x + x_dte  # 搜索RGB与辅助模态特征相加

        # 添加位置编码
        z += self.pos_embed_z
        x += self.pos_embed_x

        # 合并template和search特征
        x = combine_tokens(z, x, mode=self.cat_mode)
        x = self.pos_drop(x)  # 应用位置dropout

        # 通过前8层（冻结层）
        for i in range(8):
            x = self.blocks[i](x)
        
        # 提取第8层后的特征，用于生成Prompt
        lens_z = self.pos_embed_z.shape[1]  # 模板特征长度
        z_features = x[:, :lens_z, :]  # 模板特征
        x_features = x[:, lens_z:, :]  # 搜索特征
        
        # 生成三层增强Prompt
        P_mask, M = self.mask_enhancement(z_features, x_features)  # 掩码增强提示
        P_cons = self.consistency_enhancement(z_features, x_features)  # 一致性增强提示
        P_temp, updated_hist = self.temporal_enhancement(
            T_r=z_features,          # 当前帧模板特征
            T_x=x_features,          # 当前帧搜索特征
            T_hist=T_hist_z,         # 历史帧特征
            M_hist=M_hist            # 历史掩码信息
        )  # 时序增强提示和更新后的历史特征
        
        # 合并所有Prompt
        P_base = self.base_prompt.expand(B, -1, -1)  # 扩展基础提示到批次大小
        P_total = torch.cat([P_base, P_mask, P_cons, P_temp], dim=1)  # 按通道维度拼接
        
        # 通过后4层（注入Prompt）
        for i in range(8, 12):
            # 在Block 9-11注入Prompt
            if i >= 8:
                x = torch.cat([x, P_total], dim=1)  # 将提示与特征拼接
            x = self.blocks[i](x)  # 通过Transformer块
        
        x = self.norm(x)  # 最终归一化
        return x, updated_hist  # 返回特征和更新后的历史特征

    # 模型前向传播
    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                return_last_attn=False, T_hist_z=None, T_hist_x=None, M_hist=None):
        # 调用特征提取函数
        x, updated_hist = self.forward_features(z, x, mask_z=ce_template_mask, mask_x=None, 
                                               return_last_attn=return_last_attn,
                                               T_hist_z=T_hist_z, T_hist_x=T_hist_x, M_hist=M_hist)

        # 不使用分类头，返回特征和辅助字典
        aux_dict = {'updated_hist': updated_hist}  # 将updated_hist包装成字典
        return x, aux_dict


# 创建Vision Transformer的辅助函数
def _create_vision_transformer(pretrained=False, **kwargs):
    # 实例化模型
    model = VisionTransformerMetaP(**kwargs)

    # 如果需要加载预训练权重
    if pretrained:
        # 加载检查点
        checkpoint = torch.load(kwargs['pretrained'], map_location="cpu")
        # 加载模型状态字典，允许部分缺失
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Load pretrained model from {kwargs['pretrained']}")
        print(f"missing keys: {missing_keys}")
        print(f"unexpected keys: {unexpected_keys}")
    return model


# 创建ViT-Base模型的函数
def vit_base_patch16_224_meta_prompt(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, then fine-tuned on ImageNet-1k.
    """
    # 设置模型参数
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    # 创建并返回模型
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model