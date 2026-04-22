import pdb

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu


class ViPTActor(BaseActor):
    """ Actor for training ViPT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        
        # 【v7】纯监控模式 — 无正则化状态变量
        # v7通过网络结构本身（位置引导+Gumbel-Softmax）保证双峰
        self._adaptive_reg_state = None
        
        # 【v20分阶段训练】模块冻结逻辑（使用STAGE配置）
        if cfg is not None:
            self._apply_freeze_settings()
    
    def _apply_freeze_settings(self):
        """【v20】根据STAGE配置冻结指定模块"""
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        
        stage = getattr(self.cfg.TRAIN, 'STAGE', 0)
        
        if stage == 0:
            return
        
        print(f"[v20分阶段训练] STAGE={stage}")
        
        if hasattr(net, 'backbone') and hasattr(net.backbone, 'meta_prompt_generator'):
            meta_gen = net.backbone.meta_prompt_generator
            
            if stage == 1:
                # stage1：固定Consistency，只训练Temporal
                for name, param in net.named_parameters():
                    if 'consistency_generator' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                print("[v20 STAGE1] Consistency模块已冻结，Temporal模块可训练")
            
            elif stage == 2:
                # stage2：固定Temporal，只训练Consistency
                for name, param in net.named_parameters():
                    if 'temporal_generator' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                print("[v20 STAGE2] Temporal模块已冻结，Consistency模块可训练")
            
            elif stage == 3:
                # stage3：放开所有参数，联合微调
                for param in net.parameters():
                    param.requires_grad = True
                print("[v20 STAGE3] 所有参数已放开，联合微调")

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
            # ce_keep_rate = 0.7

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        tracking_loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        loss = tracking_loss.clone()

        # 【v7改进】纯监控模式 — 不添加任何正则化损失
        #
        # 【v7设计理念 — 彻底解决"无双峰"问题】
        #   原问题根因（v4-v6均未解决）：
        #     MLP(concat(rgb,tir,grad,cov)) → tanh → 单标量权重
        #     → 所有token收到相似梯度 → 自然坍塌到单峰
        #     → 正则化（方差/熵）无法改变网络结构本身的局限性
        #
        #   v7解决方案（架构级修复）：
        #     1. 位置引导偏置 pos_bias [1,N,2]：交替初始化为双峰种子
        #     2. Gumbel-Softmax 替代 tanh：天然产生离散化分布
        #     3. 温度退火 temperature: 3.0→0.5：前期软探索→后期硬选择
        #     4. 完全移除正则化：让网络结构本身保证双峰
        #

        reg_loss_total = torch.tensor(0.0, device=loss.device)
        current_weight_var = None
        current_weight_entropy = None

        if 'inject_intermediates' in pred_dict and pred_dict['inject_intermediates']:
            inject_intermediates = pred_dict['inject_intermediates']

            for key, intermediates in inject_intermediates.items():
                if intermediates is None:
                    continue

                token_weight = None
                base_p = None
                if 'token_consistency' in intermediates:
                    token_weight = intermediates['token_consistency'].squeeze(-1)
                    base_p = intermediates.get('base_p')
                    if base_p is not None:
                        base_p = base_p.squeeze(-1)
                elif 'token_reliability' in intermediates:
                    token_weight = intermediates['token_reliability'].squeeze(-1)

                if token_weight is not None:
                    weight_var = torch.var(token_weight, dim=1)
                    current_weight_var = weight_var.mean().item()

                    B, N = token_weight.shape
                    weight_clamped = torch.clamp(token_weight, min=0.01, max=0.99)
                    num_bins = 20
                    bin_indices = (weight_clamped * num_bins).long().clamp(0, num_bins - 1)
                    batch_entropies = []
                    for b in range(B):
                        hist = torch.bincount(bin_indices[b], minlength=num_bins).float()
                        hist = hist / (hist.sum() + 1e-8)
                        hist = torch.clamp(hist, min=1e-8)
                        entropy = -(hist * torch.log(hist)).sum()
                        batch_entropies.append(entropy)
                    current_weight_entropy = torch.stack(batch_entropies).mean().item()

                # ===== 【v20核心】修复RGB偏向问题 - 对称均值约束（降低强度）=====
                # 问题：avg_rgb_weight=0.9, avg_tir_weight=0.1 → 模型全信RGB，忽略TIR
                # 修复：强制平均权重在0.5附近，同时利用两个模态
                # 【v20关键】降低均值约束强度(0.6→0.3)，让模型有足够区分度
                if base_p is not None and token_weight is not None:
                    # 【v20修复】对称均值约束（30%权重）- 强制双模态均衡利用
                    global_mean = torch.mean(token_weight, dim=1)
                    mean_reg = torch.mean((global_mean - 0.5) ** 2)
                    
                    # 熵正则化（70%权重）- 确保双峰有区分度
                    p = token_weight
                    entropy = - (p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
                    target_entropy = 0.5
                    entropy_reg = torch.mean((entropy - target_entropy) ** 2)
                    
                    # 【v20修复】总正则化：均值约束占30%，熵正则化占70%
                    consistency_reg_loss = 0.3 * mean_reg + 0.7 * entropy_reg
                    loss = loss + 0.05 * consistency_reg_loss
                    current_reg_loss = consistency_reg_loss.item()
                else:
                    current_reg_loss = 0.0

        # ===== 【v20核心】时序正则化（简化检查逻辑 + 降低强度）=====
        # 问题：原条件检查过于复杂，导致正则化从未执行
        # 修复：简化检查条件，只检查temporal_weight是否存在
        # 【v20关键】降低均值约束强度，让模型有足够区分度
        current_temporal_reg = 0.0
        
        if 'inject_intermediates' in pred_dict and pred_dict['inject_intermediates']:
            inject_intermediates = pred_dict['inject_intermediates']
            
            if 'temporal_intermediates' in inject_intermediates and inject_intermediates['temporal_intermediates'] is not None:
                temporal_data = inject_intermediates['temporal_intermediates']
                
                # 【v20简化】只检查temporal_weight是否存在
                if 'temporal_weight' in temporal_data:
                    temporal_weight = temporal_data['temporal_weight']

                    # 【v20修复】对称均值约束（30%权重）- 强制时序权重在0.5附近
                    global_mean = torch.mean(temporal_weight, dim=1, keepdim=True)
                    mean_reg = torch.mean((global_mean - 0.5) ** 2)
                    
                    # 熵正则化（70%权重）- 确保时序权重有区分度
                    tw = temporal_weight
                    entropy = - (tw * torch.log(tw + 1e-8) + (1 - tw) * torch.log(1 - tw + 1e-8))
                    target_entropy = 0.5
                    entropy_reg = torch.mean((entropy - target_entropy) ** 2)

                    # 【v20修复】总正则化：均值约束占30%，熵正则化占70%
                    temporal_reg_loss = 0.3 * mean_reg + 0.7 * entropy_reg
                    loss = loss + 0.05 * temporal_reg_loss  # 【v20降低权重】0.2→0.05
                    current_temporal_reg = temporal_reg_loss.item()

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/consistency_reg": current_reg_loss if 'current_reg_loss' in dir() else 0.0,
                      "Loss/temporal_reg": current_temporal_reg if 'current_temporal_reg' in dir() else 0.0,
                      "Train/weight_var": current_weight_var if current_weight_var else 0.0,
                      "Train/weight_entropy": current_weight_entropy if current_weight_entropy else 0.0,
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss