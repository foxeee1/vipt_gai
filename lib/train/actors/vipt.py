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
        # 【v25关键修复】删除此处的冻结逻辑，统一由train_three_stage.py的apply_stage_freeze控制
        # 原问题：两套冻结逻辑互相矛盾，导致Stage2主干被意外解冻
        # 现在只在此处做日志输出，不修改requires_grad
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        
        stage = getattr(self.cfg.TRAIN, 'STAGE', 0)
        
        if stage == 0:
            return
        
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in net.parameters())
        print(f"[ViPTActor] STAGE={stage}, 可训练参数: {trainable:,}/{total:,} ({100*trainable/total:.1f}%) — 冻结由train_three_stage.py统一控制")

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

        # 【v26核心修复】分阶段专属Loss路由 — 解决分支失效的根本原因
        stage = getattr(self.cfg.TRAIN, 'STAGE', 0)

        if stage == 1:
            return self._compute_stage1_loss(giou_loss, l1_loss, location_loss, iou, pred_dict, return_status)
        elif stage == 2:
            return self._compute_stage2_loss(giou_loss, l1_loss, location_loss, iou, pred_dict, return_status)
        elif stage == 3:
            return self._compute_stage3_loss(giou_loss, l1_loss, location_loss, iou, pred_dict, return_status)
        else:
            return self._compute_legacy_loss(giou_loss, l1_loss, location_loss, iou, pred_dict, return_status)

    def _compute_stage1_loss(self, giou_loss, l1_loss, location_loss, iou, pred_dict, return_status):
        """Stage1: 仅定位Loss，完全屏蔽所有正则化，让模态分支专注学习表观特征"""
        tracking_loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        loss = tracking_loss

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/consistency_reg": 0.0,
                      "Loss/temporal_reg": 0.0,
                      "Loss/mask_reg": 0.0,
                      "Loss/ortho_reg": 0.0,
                      "Loss/kl_reg": 0.0,
                      "Train/weight_var": 0.0,
                      "Train/weight_entropy": 0.0,
                      "IoU": mean_iou.item(),
                      "Stage": 1}
            return loss, status
        return loss

    def _compute_stage2_loss(self, giou_loss, l1_loss, location_loss, iou, pred_dict, return_status):
        """Stage2: 定位Loss主导 + 辅助分支弱约束（避免主任务IoU下降）"""
        tracking_loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        loss = tracking_loss * 1.0  # 【v28修复】恢复tracking Loss权重，不再降权

        current_reg_loss = 0.0
        current_temporal_reg = 0.0
        current_mask_reg = 0.0
        current_ortho_reg = 0.0
        current_kl_reg = 0.0
        current_weight_var = None
        current_weight_entropy = None

        inject_intermediates = pred_dict.get('inject_intermediates', {})

        # ===== 一致性分支约束：token一致性约束（权重1e-4，平衡训练） =====
        if 'consistency_intermediates' in inject_intermediates:
            consistency_data = inject_intermediates['consistency_intermediates']
            if consistency_data is not None and isinstance(consistency_data, dict):
                token_weight = consistency_data.get('token_consistency')
                if token_weight is not None:
                    # 安全校验：检查NaN
                    if torch.isnan(token_weight).any():
                        pass
                    else:
                        token_weight = token_weight.squeeze(-1)
                        weight_var = torch.var(token_weight, dim=1)
                        current_weight_var = weight_var.mean().item()
                        num_tokens = max(token_weight.shape[1], 1)
                        # 归一化L2 norm：除以sqrt(num_tokens)确保值域合理
                        consistency_reg_loss = torch.norm(token_weight, p=2) / (num_tokens ** 0.5 + 1e-8)
                        # 裁剪到合理范围 [0, 10]
                        consistency_reg_loss = torch.clamp(consistency_reg_loss, 0, 10)
                        loss = loss + 1e-4 * consistency_reg_loss  # 【v29修复】权重提高到1e-4
                        current_reg_loss = consistency_reg_loss.item()

        # ===== 时序分支约束：帧间平滑 + 运动门控约束（权重1e-4，平衡训练） =====
        if 'temporal_intermediates' in inject_intermediates:
            temporal_data = inject_intermediates['temporal_intermediates']
            if temporal_data is not None and isinstance(temporal_data, dict):
                tw = temporal_data.get('temporal_weight')
                if tw is not None:
                    # 安全校验：检查NaN
                    if not torch.isnan(tw).any() and not torch.isinf(tw).any():
                        tw_norm = torch.norm(tw, p=2) / (tw.numel() ** 0.5 + 1e-8)
                        temporal_reg_loss = tw_norm * 0.5
                        prev_tw = temporal_data.get('prev_temporal_weight')
                        if prev_tw is not None and prev_tw.shape == tw.shape:
                            # 帧间差异损失
                            frame_diff = torch.mean((tw - prev_tw) ** 2)
                            temporal_reg_loss = temporal_reg_loss + 0.5 * frame_diff
                        # 裁剪到合理范围 [0, 10]
                        temporal_reg_loss = torch.clamp(temporal_reg_loss, 0, 10)
                        loss = loss + 1e-4 * temporal_reg_loss  # 【v29修复】权重提高到1e-4
                        current_temporal_reg = temporal_reg_loss.item()
                        global_motion = temporal_data.get('global_motion')
                        if global_motion is not None and not torch.isnan(global_motion).any():
                            motion_gate_loss = -torch.mean(global_motion * tw.mean(dim=1, keepdim=True))
                            motion_gate_loss = torch.clamp(motion_gate_loss, -10, 10)
                            loss = loss + 1e-5 * motion_gate_loss  # 【v29修复】权重提高到1e-5

        # ===== Mask分支约束：目标区域特征响应增强（权重1e-4，平衡训练） =====
        if 'mask_intermediates' in inject_intermediates:
            mask_data = inject_intermediates['mask_intermediates']
            if mask_data is not None and isinstance(mask_data, dict):
                token_reliability = mask_data.get('token_reliability')
                if token_reliability is not None:
                    # 安全校验：检查NaN
                    if not torch.isnan(token_reliability).any() and not torch.isinf(token_reliability).any():
                        token_reliability = token_reliability.squeeze(-1)
                        # 归一化L2 norm：除以sqrt(num_elements)确保值域合理
                        mask_reg_loss = torch.norm(token_reliability, p=2) / (token_reliability.numel() ** 0.5 + 1e-8)
                        mean_reliability = token_reliability.mean()
                        target_resp_loss = -torch.mean(torch.relu(mean_reliability - 0.3))
                        # 裁剪到合理范围
                        mask_reg_loss = torch.clamp(mask_reg_loss, 0, 10)
                        target_resp_loss = torch.clamp(target_resp_loss, -10, 10)
                        loss = loss + 1e-4 * mask_reg_loss + 1e-5 * target_resp_loss  # 【v29修复】权重提高到1e-4/1e-5
                        current_mask_reg = mask_reg_loss.item()

        # ===== 跨模态双向KL散度（权重1e-5，Stage2弱对齐） =====
        if 'branch_feats' in inject_intermediates:
            bf = inject_intermediates['branch_feats']
            rgb_f = bf.get('rgb_mean')
            tir_f = bf.get('tir_mean')
            if rgb_f is not None and tir_f is not None:
                # 安全校验：检查NaN
                if not torch.isnan(rgb_f).any() and not torch.isnan(tir_f).any() and not torch.isinf(rgb_f).any() and not torch.isinf(tir_f).any():
                    # 【v27关键修复】KL散度数值保护
                    # 1. 先对特征做归一化，防止极端值
                    rgb_f = (rgb_f - rgb_f.mean(dim=-1, keepdim=True)) / (rgb_f.std(dim=-1, keepdim=True) + 1e-8)
                    tir_f = (tir_f - tir_f.mean(dim=-1, keepdim=True)) / (tir_f.std(dim=-1, keepdim=True) + 1e-8)
                    # 2. 限制特征范围，避免softmax出现数值问题
                    rgb_f = torch.clamp(rgb_f, min=-5.0, max=5.0)
                    tir_f = torch.clamp(tir_f, min=-5.0, max=5.0)
                    
                    rgb_log_prob = F.log_softmax(rgb_f, dim=-1)
                    tir_log_prob = F.log_softmax(tir_f, dim=-1)
                    rgb_prob = F.softmax(rgb_f, dim=-1)
                    tir_prob = F.softmax(tir_f, dim=-1)
                    
                    # 3. 对prob添加epsilon保护，避免log(0)导致NaN
                    epsilon = 1e-8
                    rgb_prob = torch.clamp(rgb_prob, min=epsilon, max=1.0 - epsilon)
                    tir_prob = torch.clamp(tir_prob, min=epsilon, max=1.0 - epsilon)
                    rgb_log_prob = torch.clamp(rgb_log_prob, min=-50.0, max=0.0)
                    tir_log_prob = torch.clamp(tir_log_prob, min=-50.0, max=0.0)
                    
                    kl_rt = F.kl_div(rgb_log_prob, tir_prob, reduction='batchmean', log_target=False)
                    kl_tr = F.kl_div(tir_log_prob, rgb_prob, reduction='batchmean', log_target=False)
                    kl_loss = (kl_rt + kl_tr) / 2.0
                    
                    # 4. 检查并裁剪kl_loss
                    if torch.isnan(kl_loss) or torch.isinf(kl_loss) or kl_loss < 0:
                        kl_loss = torch.tensor(0.0, device=kl_loss.device)
                    kl_loss = torch.clamp(kl_loss, 0, 5.0)
                    
                    loss = loss + 1e-5 * kl_loss  # 【v29修复】权重提高到1e-5
                    current_kl_reg = kl_loss.item()

        # ===== Stage2梯度链接Loss（来自inject的grad_link_loss）=====
        # 安全校验：当当前层无注入时，通过grad_link_loss保持辅助生成器的梯度连接
        grad_link_loss_val = 0.0
        if 'inject_intermediates' in pred_dict and pred_dict['inject_intermediates']:
            inject_intermediates = pred_dict['inject_intermediates']
            if isinstance(inject_intermediates, dict) and 'grad_link_loss' in inject_intermediates:
                grad_link_loss = inject_intermediates['grad_link_loss']
                if grad_link_loss is not None and not torch.isnan(grad_link_loss) and not torch.isinf(grad_link_loss):
                    loss = loss + grad_link_loss
                    grad_link_loss_val = grad_link_loss.item()

        # 最终安全检查：确保loss不是NaN或Inf
        # 安全校验：NaN时回退到tracking_loss（保留梯度），而非硬编码0.5倍
        if torch.isnan(loss) or torch.isinf(loss):
            loss = tracking_loss

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
                      "Loss/grad_link": grad_link_loss_val,
                      "Train/weight_var": current_weight_var if current_weight_var else 0.0,
                      "Train/weight_entropy": current_weight_entropy if current_weight_entropy else 0.0,
                      "IoU": mean_iou.item(),
                      "Stage": 2}
            return loss, status
        return loss

    def _compute_stage3_loss(self, giou_loss, l1_loss, location_loss, iou, pred_dict, return_status):
        """Stage3: 定位Loss绝对主导 + 辅助Loss极低（≤1e-6），仅做弱约束"""
        tracking_loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        loss = tracking_loss.clone()

        current_weight_var = None
        current_weight_entropy = None
        current_reg_loss = 0.0
        current_temporal_reg = 0.0
        current_mask_reg = 0.0
        current_ortho_reg = 0.0
        current_kl_reg = 0.0

        if 'inject_intermediates' in pred_dict and pred_dict['inject_intermediates']:
            inject_intermediates = pred_dict['inject_intermediates']

            if 'consistency_intermediates' in inject_intermediates:
                consistency_data = inject_intermediates['consistency_intermediates']
                if consistency_data is not None and isinstance(consistency_data, dict):
                    token_weight = consistency_data.get('token_consistency')
                    if token_weight is not None:
                        token_weight = token_weight.squeeze(-1)
                        weight_var = torch.var(token_weight, dim=1)
                        current_weight_var = weight_var.mean().item()
                        num_tokens = max(token_weight.shape[1], 1)
                        consistency_reg_loss = torch.norm(token_weight, p=2) / (num_tokens ** 0.5 + 1e-8)
                        loss = loss + 1e-6 * consistency_reg_loss
                        current_reg_loss = consistency_reg_loss.item()

            if 'temporal_intermediates' in inject_intermediates:
                temporal_data = inject_intermediates['temporal_intermediates']
                if temporal_data is not None and isinstance(temporal_data, dict):
                    tw = temporal_data.get('temporal_weight')
                    if tw is not None:
                        temporal_reg_loss = torch.norm(tw, p=2) * 0.5
                        prev_tw = temporal_data.get('prev_temporal_weight')
                        if prev_tw is not None and prev_tw.shape == tw.shape:
                            temporal_reg_loss = temporal_reg_loss + 0.5 * torch.mean((tw - prev_tw) ** 2)
                        loss = loss + 1e-6 * temporal_reg_loss
                        current_temporal_reg = temporal_reg_loss.item()

            if 'mask_intermediates' in inject_intermediates:
                mask_data = inject_intermediates['mask_intermediates']
                if mask_data is not None and isinstance(mask_data, dict):
                    token_reliability = mask_data.get('token_reliability')
                    if token_reliability is not None:
                        token_reliability = token_reliability.squeeze(-1)
                        mask_reg_loss = torch.norm(token_reliability, p=2)
                        loss = loss + 1e-6 * mask_reg_loss
                        current_mask_reg = mask_reg_loss.item()

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
                    loss = loss + 5e-7 * ortho_loss
                    current_ortho_reg = ortho_loss.item()

            if 'branch_feats' in inject_intermediates:
                bf = inject_intermediates['branch_feats']
                rgb_f = bf.get('rgb_mean')
                tir_f = bf.get('tir_mean')
                if rgb_f is not None and tir_f is not None:
                    # 【v27关键修复】KL散度数值保护（Stage3/Legacy）
                    if not torch.isnan(rgb_f).any() and not torch.isnan(tir_f).any() and not torch.isinf(rgb_f).any() and not torch.isinf(tir_f).any():
                        rgb_f = (rgb_f - rgb_f.mean(dim=-1, keepdim=True)) / (rgb_f.std(dim=-1, keepdim=True) + 1e-8)
                        tir_f = (tir_f - tir_f.mean(dim=-1, keepdim=True)) / (tir_f.std(dim=-1, keepdim=True) + 1e-8)
                        rgb_f = torch.clamp(rgb_f, min=-5.0, max=5.0)
                        tir_f = torch.clamp(tir_f, min=-5.0, max=5.0)
                        
                        rgb_log_prob = F.log_softmax(rgb_f, dim=-1)
                        tir_log_prob = F.log_softmax(tir_f, dim=-1)
                        rgb_prob = F.softmax(rgb_f, dim=-1)
                        tir_prob = F.softmax(tir_f, dim=-1)
                        
                        epsilon = 1e-8
                        rgb_prob = torch.clamp(rgb_prob, min=epsilon, max=1.0 - epsilon)
                        tir_prob = torch.clamp(tir_prob, min=epsilon, max=1.0 - epsilon)
                        rgb_log_prob = torch.clamp(rgb_log_prob, min=-50.0, max=0.0)
                        tir_log_prob = torch.clamp(tir_log_prob, min=-50.0, max=0.0)
                        
                        kl_rt = F.kl_div(rgb_log_prob, tir_prob, reduction='batchmean', log_target=False)
                        kl_tr = F.kl_div(tir_log_prob, rgb_prob, reduction='batchmean', log_target=False)
                        kl_loss = (kl_rt + kl_tr) / 2.0
                        
                        if torch.isnan(kl_loss) or torch.isinf(kl_loss) or kl_loss < 0:
                            kl_loss = torch.tensor(0.0, device=kl_loss.device)
                        kl_loss = torch.clamp(kl_loss, 0, 5.0)
                        
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
                      "IoU": mean_iou.item(),
                      "Stage": 3}
            return loss, status
        return loss

    def _compute_legacy_loss(self, giou_loss, l1_loss, location_loss, iou, pred_dict, return_status):
        """Legacy: 兼容非三阶段训练的原始逻辑"""
        tracking_loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        loss = tracking_loss.clone()

        current_weight_var = None
        current_weight_entropy = None
        current_reg_loss = 0.0
        current_temporal_reg = 0.0
        current_mask_reg = 0.0
        current_ortho_reg = 0.0
        current_kl_reg = 0.0

        if 'inject_intermediates' in pred_dict and pred_dict['inject_intermediates']:
            inject_intermediates = pred_dict['inject_intermediates']

            if 'consistency_intermediates' in inject_intermediates:
                consistency_data = inject_intermediates['consistency_intermediates']
                if consistency_data is not None and isinstance(consistency_data, dict):
                    token_weight = consistency_data.get('token_consistency')
                    if token_weight is not None:
                        token_weight = token_weight.squeeze(-1)
                        weight_var = torch.var(token_weight, dim=1)
                        current_weight_var = weight_var.mean().item()
                        num_tokens = max(token_weight.shape[1], 1)
                        consistency_reg_loss = torch.norm(token_weight, p=2) / (num_tokens ** 0.5 + 1e-8)
                        loss = loss + 1e-6 * consistency_reg_loss
                        current_reg_loss = consistency_reg_loss.item()

            if 'temporal_intermediates' in inject_intermediates:
                temporal_data = inject_intermediates['temporal_intermediates']
                if temporal_data is not None and isinstance(temporal_data, dict):
                    tw = temporal_data.get('temporal_weight')
                    if tw is not None:
                        temporal_reg_loss = torch.norm(tw, p=2) * 0.5
                        prev_tw = temporal_data.get('prev_temporal_weight')
                        if prev_tw is not None and prev_tw.shape == tw.shape:
                            temporal_reg_loss = temporal_reg_loss + 0.5 * torch.mean((tw - prev_tw) ** 2)
                        loss = loss + 1e-6 * temporal_reg_loss
                        current_temporal_reg = temporal_reg_loss.item()

            if 'mask_intermediates' in inject_intermediates:
                mask_data = inject_intermediates['mask_intermediates']
                if mask_data is not None and isinstance(mask_data, dict):
                    token_reliability = mask_data.get('token_reliability')
                    if token_reliability is not None:
                        token_reliability = token_reliability.squeeze(-1)
                        mask_reg_loss = torch.norm(token_reliability, p=2)
                        loss = loss + 1e-6 * mask_reg_loss
                        current_mask_reg = mask_reg_loss.item()

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
                    loss = loss + 5e-7 * ortho_loss
                    current_ortho_reg = ortho_loss.item()

            if 'branch_feats' in inject_intermediates:
                bf = inject_intermediates['branch_feats']
                rgb_f = bf.get('rgb_mean')
                tir_f = bf.get('tir_mean')
                if rgb_f is not None and tir_f is not None:
                    # 【v27关键修复】KL散度数值保护（Stage3/Legacy）
                    if not torch.isnan(rgb_f).any() and not torch.isnan(tir_f).any() and not torch.isinf(rgb_f).any() and not torch.isinf(tir_f).any():
                        rgb_f = (rgb_f - rgb_f.mean(dim=-1, keepdim=True)) / (rgb_f.std(dim=-1, keepdim=True) + 1e-8)
                        tir_f = (tir_f - tir_f.mean(dim=-1, keepdim=True)) / (tir_f.std(dim=-1, keepdim=True) + 1e-8)
                        rgb_f = torch.clamp(rgb_f, min=-5.0, max=5.0)
                        tir_f = torch.clamp(tir_f, min=-5.0, max=5.0)
                        
                        rgb_log_prob = F.log_softmax(rgb_f, dim=-1)
                        tir_log_prob = F.log_softmax(tir_f, dim=-1)
                        rgb_prob = F.softmax(rgb_f, dim=-1)
                        tir_prob = F.softmax(tir_f, dim=-1)
                        
                        epsilon = 1e-8
                        rgb_prob = torch.clamp(rgb_prob, min=epsilon, max=1.0 - epsilon)
                        tir_prob = torch.clamp(tir_prob, min=epsilon, max=1.0 - epsilon)
                        rgb_log_prob = torch.clamp(rgb_log_prob, min=-50.0, max=0.0)
                        tir_log_prob = torch.clamp(tir_log_prob, min=-50.0, max=0.0)
                        
                        kl_rt = F.kl_div(rgb_log_prob, tir_prob, reduction='batchmean', log_target=False)
                        kl_tr = F.kl_div(tir_log_prob, rgb_prob, reduction='batchmean', log_target=False)
                        kl_loss = (kl_rt + kl_tr) / 2.0
                        
                        if torch.isnan(kl_loss) or torch.isinf(kl_loss) or kl_loss < 0:
                            kl_loss = torch.tensor(0.0, device=kl_loss.device)
                        kl_loss = torch.clamp(kl_loss, 0, 5.0)
                        
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
        return loss