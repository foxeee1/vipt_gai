import pdb
import torch.nn.functional as F

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
        """三阶段训练：Stage1模态专属→Stage2一致性+时序→Stage3联合微调"""
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        
        stage = getattr(self.cfg.TRAIN, 'STAGE', 0)
        
        if stage == 0:
            return
        
        print(f"[三阶段训练] STAGE={stage}")
        
        if hasattr(net, 'backbone') and hasattr(net.backbone, 'meta_prompt_generator'):
            meta_gen = net.backbone.meta_prompt_generator
            
            if stage == 1:
                # Stage1：仅训练模态专属Prompt（低层1-3）+ 基础跟踪头
                # 冻结Consistency、Temporal、Mask分支
                for name, param in net.named_parameters():
                    if any(kw in name for kw in ['consistency_generator', 'temporal_generator', 
                                                   'mask_generator', 'cross_attn_modulation',
                                                   'layer_gates', 'alpha_consistency', 
                                                   'alpha_temporal', 'alpha_mask']):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                print("[STAGE1] 模态专属Prompt可训练，Consistency/Temporal/Mask已冻结")
            
            elif stage == 2:
                # Stage2：冻结模态专属分支，训练Consistency+Temporal
                for name, param in net.named_parameters():
                    if any(kw in name for kw in ['modality_prompt_proj', 'rgb_type_token', 'tir_type_token']):
                        param.requires_grad = False
                    elif any(kw in name for kw in ['consistency_generator', 'temporal_generator',
                                                    'mask_generator', 'cross_attn_modulation',
                                                    'layer_gates', 'alpha_consistency',
                                                    'alpha_temporal', 'alpha_mask']):
                        param.requires_grad = True
                    else:
                        param.requires_grad = True
                print("[STAGE2] Consistency+Temporal+Mask可训练，模态专属已冻结")
            
            elif stage == 3:
                # Stage3：解冻所有分支，联合微调
                for param in net.parameters():
                    param.requires_grad = True
                print("[STAGE3] 所有参数已解冻，联合微调")

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
        current_ortho_reg = 0.0
        current_kl_reg = 0.0

        if 'inject_intermediates' in pred_dict and pred_dict['inject_intermediates']:
            inject_intermediates = pred_dict['inject_intermediates']

            # ===== Consistency轻量正则（L2范数约束，归一化到token数量） =====
            if 'consistency_intermediates' in inject_intermediates:
                consistency_data = inject_intermediates['consistency_intermediates']
                if consistency_data is not None and isinstance(consistency_data, dict):
                    token_weight = None
                    if 'token_consistency' in consistency_data:
                        token_weight = consistency_data['token_consistency'].squeeze(-1)

                    if token_weight is not None:
                        weight_var = torch.var(token_weight, dim=1)
                        current_weight_var = weight_var.mean().item()
                        # 【v25修复】除以token数量归一化，避免token数多导致reg爆炸
                        num_tokens = max(token_weight.shape[1], 1)
                        consistency_reg_loss = torch.norm(token_weight, p=2) / (num_tokens ** 0.5 + 1e-8)
                        loss = loss + 1e-6 * consistency_reg_loss
                        current_reg_loss = consistency_reg_loss.item()

            # ===== Temporal轻量正则（帧间平滑+L2范数） =====
            if 'temporal_intermediates' in inject_intermediates:
                temporal_data = inject_intermediates['temporal_intermediates']
                if temporal_data is not None and isinstance(temporal_data, dict):
                    if 'temporal_weight' in temporal_data:
                        tw = temporal_data['temporal_weight']
                        temporal_reg_loss = torch.norm(tw, p=2) * 0.5
                        if 'prev_temporal_weight' in temporal_data and temporal_data['prev_temporal_weight'] is not None:
                            prev_tw = temporal_data['prev_temporal_weight']
                            if prev_tw.shape == tw.shape:
                                temporal_reg_loss = temporal_reg_loss + 0.5 * torch.mean((tw - prev_tw) ** 2)
                        # 【v25修复】权重从1e-4降到1e-6
                        loss = loss + 1e-6 * temporal_reg_loss
                        current_temporal_reg = temporal_reg_loss.item()

            # ===== Mask轻量正则（L2范数约束） =====
            if 'mask_intermediates' in inject_intermediates:
                mask_data = inject_intermediates['mask_intermediates']
                if mask_data is not None and isinstance(mask_data, dict):
                    if 'token_reliability' in mask_data:
                        token_reliability = mask_data['token_reliability'].squeeze(-1)
                        mask_reg_loss = torch.norm(token_reliability, p=2)
                        # 【v25修复】权重从1e-4降到1e-6
                        loss = loss + 1e-6 * mask_reg_loss
                        current_mask_reg = mask_reg_loss.item()

            # ===== 正交正则：强制分支特征互补无冗余（权重大幅降低） =====
            if 'branch_feats' in inject_intermediates:
                bf = inject_intermediates['branch_feats']
                rgb_f = bf.get('rgb_mean')
                tir_f = bf.get('tir_mean')
                cons_f = bf.get('consistency_mean')
                temp_f = bf.get('temporal_mean')

                ortho_terms = []
                if rgb_f is not None and tir_f is not None:
                    rgb_fn = rgb_f / (rgb_f.norm(dim=-1, keepdim=True) + 1e-8)
                    tir_fn = tir_f / (tir_f.norm(dim=-1, keepdim=True) + 1e-8)
                    ortho_terms.append(torch.norm(torch.bmm(rgb_fn.unsqueeze(1), tir_fn.unsqueeze(2)).squeeze(), p=2))
                if rgb_f is not None and cons_f is not None:
                    rgb_fn = rgb_f / (rgb_f.norm(dim=-1, keepdim=True) + 1e-8)
                    cons_fn = cons_f / (cons_f.norm(dim=-1, keepdim=True) + 1e-8)
                    ortho_terms.append(torch.norm(torch.bmm(rgb_fn.unsqueeze(1), cons_fn.unsqueeze(2)).squeeze(), p=2))
                if tir_f is not None and cons_f is not None:
                    tir_fn = tir_f / (tir_f.norm(dim=-1, keepdim=True) + 1e-8)
                    cons_fn = cons_f / (cons_f.norm(dim=-1, keepdim=True) + 1e-8)
                    ortho_terms.append(torch.norm(torch.bmm(tir_fn.unsqueeze(1), cons_fn.unsqueeze(2)).squeeze(), p=2))
                if temp_f is not None and cons_f is not None:
                    temp_fn = temp_f / (temp_f.norm(dim=-1, keepdim=True) + 1e-8)
                    cons_fn = cons_f / (cons_f.norm(dim=-1, keepdim=True) + 1e-8)
                    ortho_terms.append(torch.norm(torch.bmm(temp_fn.unsqueeze(1), cons_fn.unsqueeze(2)).squeeze(), p=2))

                if ortho_terms:
                    ortho_loss = sum(ortho_terms) / len(ortho_terms)
                    # 【v25修复】权重从5e-5降到5e-7，防止反客为主
                    loss = loss + 5e-7 * ortho_loss
                    current_ortho_reg = ortho_loss.item()

            # ===== 跨模态KL散度损失：双向对称，防止单向对齐导致分支消失 =====
            if 'branch_feats' in inject_intermediates:
                bf = inject_intermediates['branch_feats']
                rgb_f = bf.get('rgb_mean')
                tir_f = bf.get('tir_mean')
                if rgb_f is not None and tir_f is not None:
                    # 【v25修复】双向对称KL散度，避免单向对齐导致一个分支消失
                    rgb_log_prob = F.log_softmax(rgb_f, dim=-1)
                    tir_log_prob = F.log_softmax(tir_f, dim=-1)
                    rgb_prob = F.softmax(rgb_f, dim=-1)
                    tir_prob = F.softmax(tir_f, dim=-1)
                    kl_rt = F.kl_div(rgb_log_prob, tir_prob, reduction='batchmean')
                    kl_tr = F.kl_div(tir_log_prob, rgb_prob, reduction='batchmean')
                    kl_loss = (kl_rt + kl_tr) / 2.0
                    # 【v25修复】权重从1e-4降到1e-6
                    loss = loss + 1e-6 * kl_loss
                    current_kl_reg = kl_loss.item()

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/consistency_reg": current_reg_loss,
                      "Loss/temporal_reg": current_temporal_reg,
                      "Loss/mask_reg": current_mask_reg,
                      "Loss/ortho_reg": current_ortho_reg,
                      "Loss/kl_reg": current_kl_reg,
                      "Train/weight_var": current_weight_var if current_weight_var else 0.0,
                      "Train/weight_entropy": current_weight_entropy if current_weight_entropy else 0.0,
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss