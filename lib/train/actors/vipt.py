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

        # 【v22修复】动态loss平衡：正则化权重随tracking_loss自适应缩放
        # 原理：tracking_loss大时→正则化权重小（让模型专注跟踪）
        #       tracking_loss小时→正则化权重大（强化约束）
        # 避免正则化和主任务对抗
        tracking_loss_val = tracking_loss.item() + 1e-8

        current_weight_var = None
        current_weight_entropy = None
        current_reg_loss = 0.0
        current_temporal_reg = 0.0
        current_mask_reg = 0.0

        if 'inject_intermediates' in pred_dict and pred_dict['inject_intermediates']:
            inject_intermediates = pred_dict['inject_intermediates']

            # ===== Consistency正则化（双峰分布约束+最小化熵） =====
            if 'consistency_intermediates' in inject_intermediates:
                consistency_data = inject_intermediates['consistency_intermediates']
                if consistency_data is not None and isinstance(consistency_data, dict):
                    token_weight = None
                    if 'token_consistency' in consistency_data:
                        token_weight = consistency_data['token_consistency'].squeeze(-1)
                    
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

                        # 双峰分布约束：鼓励权重向0/1两端分化
                        # p*(1-p)在p=0.5时最大，在p=0/1时为0
                        # 最小化p*(1-p) → 鼓励权重远离0.5
                        bimodal_reg = torch.mean(token_weight * (1 - token_weight))
                        # 最小化熵：让权重更有区分度
                        p = torch.clamp(token_weight, min=1e-6, max=1 - 1e-6)
                        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                        entropy_min_reg = torch.mean(entropy)
                        consistency_reg_loss = 0.5 * bimodal_reg + 0.5 * entropy_min_reg
                        loss = loss + 0.02 * consistency_reg_loss
                        current_reg_loss = consistency_reg_loss.item()

            # ===== Temporal正则化（v25五维约束） =====
            # 【v25核心】v24解决了"约束失效"，v25解决"权重震荡+输出衰减"
            # 曲线证据：avg_weight在0.35-0.55剧烈震荡，prompt_mean从0.03衰减到0.025
            if 'temporal_intermediates' in inject_intermediates:
                temporal_data = inject_intermediates['temporal_intermediates']
                if temporal_data is not None and isinstance(temporal_data, dict):
                    if 'temporal_weight' in temporal_data:
                        tw = temporal_data['temporal_weight']
                        B, N = tw.shape

                        # 1. 权重均值上下限：强制[0.40, 0.58]（从0.35-0.65收紧）
                        weight_mean = tw.mean(dim=1)
                        lower_p = torch.mean(torch.relu(0.40 - weight_mean) ** 2)
                        upper_p = torch.mean(torch.relu(weight_mean - 0.58) ** 2)
                        mean_reg = lower_p + upper_p

                        # 2. 熵范围压缩：从[0.25,0.65]→[0.38,0.55]，减少震荡空间
                        entropy = -(tw * torch.log(tw + 1e-8) + (1 - tw) * torch.log(1 - tw + 1e-8))
                        ent_lower = torch.mean(torch.relu(0.38 - entropy) ** 2)
                        ent_upper = torch.mean(torch.relu(entropy - 0.55) ** 2)
                        entropy_reg = ent_lower + ent_upper

                        # 3. 【v25新增】帧间平滑性：惩罚相邻帧权重突变
                        smooth_reg = torch.tensor(0.0, device=loss.device)
                        if 'prev_temporal_weight' in temporal_data and temporal_data['prev_temporal_weight'] is not None:
                            prev_tw = temporal_data['prev_temporal_weight']
                            if prev_tw.shape == tw.shape:
                                smooth_reg = torch.mean((tw - prev_tw) ** 2)

                        # 4. motion_gate下限约束（保持不变）
                        motion_reg = torch.tensor(0.0, device=loss.device)
                        if 'motion_gate' in temporal_data:
                            mg = temporal_data['motion_gate'].squeeze()
                            motion_reg = torch.mean(torch.relu(0.85 - mg) ** 2)

                        # 5. 【v25新增】层间差异约束：鼓励分层学习
                        layer_reg = torch.tensor(0.0, device=loss.device)
                        if 'layer_divergence' in temporal_data:
                            ld = temporal_data['layer_divergence']
                            if isinstance(ld, torch.Tensor):
                                layer_reg = torch.mean(torch.relu(0.05 - ld) ** 2)

                        # 五维组合：平滑30% + 均值20% + 熵20% + 门控15% + 层间15%
                        temporal_reg_loss = 0.30*smooth_reg + 0.20*mean_reg + 0.20*entropy_reg + 0.15*motion_reg + 0.15*layer_reg
                        loss = loss + 0.02 * temporal_reg_loss
                        current_temporal_reg = temporal_reg_loss.item()

            # ===== Mask正则化（双峰分布约束+最小化熵） =====
            if 'mask_intermediates' in inject_intermediates:
                mask_data = inject_intermediates['mask_intermediates']
                if mask_data is not None and isinstance(mask_data, dict):
                    if 'token_reliability' in mask_data:
                        token_reliability = mask_data['token_reliability'].squeeze(-1)

                        # 双峰分布约束：鼓励权重向0/1两端分化
                        bimodal_reg = torch.mean(token_reliability * (1 - token_reliability))
                        # 最小化熵
                        tr = torch.clamp(token_reliability, min=1e-6, max=1 - 1e-6)
                        entropy = -(tr * torch.log(tr) + (1 - tr) * torch.log(1 - tr))
                        entropy_min_reg = torch.mean(entropy)
                        mask_reg_loss = 0.5 * bimodal_reg + 0.5 * entropy_min_reg
                        loss = loss + 0.01 * mask_reg_loss
                        current_mask_reg = mask_reg_loss.item()

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/consistency_reg": current_reg_loss,
                      "Loss/temporal_reg": current_temporal_reg,
                      "Loss/mask_reg": current_mask_reg,
                      "Train/weight_var": current_weight_var if current_weight_var else 0.0,
                      "Train/weight_entropy": current_weight_entropy if current_weight_entropy else 0.0,
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss