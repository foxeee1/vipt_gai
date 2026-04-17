"""
ViPT元学习训练Actor - FOMAML (First-Order MAML) Support/Query分割版

【核心设计 - 借鉴标准MAML实现(learn2learn风格)】
标准FOMAML流程：
  ① DataLoader输出batch [1,B,C,H,W] (sequence模式，dim0=N_t)
  ② AutoTaskEvaluator自动判定任务类型(Base/Stress/Conflict)
  ③ 沿batch维度(dim=1)分割为Support集+Query集
  ④ 内环(K步): fast_weights = φ - β·∇L_support  (fast_weights副本，不修改原参数)
  ⑤ 外环: L_query(fast_weights) → backward → γ更新 (元学习优化原参数φ)

【关键改进 vs 旧版】
- 旧版: 原地修改param.data → save → forward → restore (易出错)
- 新版: fast_weights副本模式 (参考learn2learn)，无需save/restore

【输入】data dict含:
  - template_images: [1, B, 6, 128, 128]  (sequence模式5D)
  - search_images:   [1, B, 6, 256, 256]
  - template_anno/search_anno: list of [B, 4]
【输出】loss_outer(外环损失) + status(日志字典)
"""

import torch
import os
import math
from collections import deque
import torch.nn.functional as F
from contextlib import nullcontext
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from lib.train.data.task_evaluator import AutoTaskEvaluator
from lib.train.data.task_construction import TaskConstructor
from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate


class ViPTMetaActor(BaseActor):
    """
    ViPT FOMAML元学习训练Actor (fast_weights模式)

    【数学定义 - 借鉴标准MAML】
    内环(k=0):   fast_w = φ - β · ∇_φ L_support(φ)
    内环(k>0):   fast_w = fast_w - β · ∇_fast_w L_support(fast_w)
    外环:        loss_q = L_query(fast_w_K) → loss_q.backward() → optimizer.step()
    """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg

        self.inner_lr = cfg.TRAIN.META.INNER_LR
        self.support_ratio = getattr(cfg.TRAIN.META, 'SUPPORT_RATIO', 0.5)
        self.update_step = getattr(cfg.TRAIN.META, 'UPDATE_STEP', 1)
        self.inner_grad_clip = getattr(cfg.TRAIN.META, 'INNER_GRAD_CLIP_NORM', 2.0)
        self.warmup_epoch = getattr(cfg.TRAIN.META, 'WARMUP_EPOCH', 0)
        # 【v8.1】Support正则化权重: 外环loss = query_loss + support_reg * support_loss
        self.support_reg_weight = getattr(cfg.TRAIN.META, 'SUPPORT_REG_WEIGHT', 0.3)

        # 【方案A】Health Score监控状态 (仅用于监控, 不驱动训练参数!)
        self._adaptive_state = {
            'iou_window': deque(maxlen=80),
            'loss_window': deque(maxlen=80),
            'health_history': deque(maxlen=20),
            'prev_health': 0.5,
            'initialized': False,
        }
        self.health_smoothing = 0.8

        self.task_evaluator = AutoTaskEvaluator(modality_type_x='t')
        self.task_constructor = TaskConstructor()
        self._step_count = 0

        tb_dir = getattr(settings.env, 'tensorboard_dir', None)
        if tb_dir is not None:
            project_path = getattr(settings, 'project_path', 'train/default')
            meta_tb_dir = os.path.join(tb_dir, project_path, 'meta')
            os.makedirs(meta_tb_dir, exist_ok=True)
            try:
                self.tb_writer = SummaryWriter(meta_tb_dir, flush_secs=10)
                self.tb_writer.add_text('config', f'INNER_LR={self.inner_lr}, UPDATE_STEP={self.update_step}, SUPPORT_RATIO={self.support_ratio}, WARMUP={self.warmup_epoch}')
                self.tb_writer.flush()
                print(f"[Meta-TB] ✓ TensorBoard enabled: {meta_tb_dir}")
                print(f"[Meta-TB]   View:  tensorboard --logdir={os.path.join(tb_dir, project_path)} --port=6006")
            except Exception as e:
                self.tb_writer = None
                print(f"[Meta-TB] ✗ Init failed: {e}")
        else:
            ws_dir = getattr(settings.env, 'workspace_dir', None)
            if ws_dir is not None:
                fallback_dir = os.path.join(ws_dir, 'tensorboard_logs', 'meta')
                print(f"[Meta-TB] ⚠ No tensorboard_dir, fallback: {fallback_dir}")
                print(f"[Meta-TB]   Set tensorboard_dir in lib/train/admin/local.py to enable")
            else:
                print("[Meta-TB] ⚠ TensorBoard logging disabled (no paths configured)")
            self.tb_writer = None

    def fix_bns(self):
        net = self.net.module if hasattr(self.net, 'module') else self.net
        net.box_head.apply(lambda m: m.eval() if m.__class__.__name__.find('BatchNorm') != -1 else None)

    def __call__(self, data):
        """
        【功能】FOMAML前向传播主函数
        【输入】data: DataLoader输出的标准dict (sequence模式5D格式)
          - template_images: tensor [1, B, 6, 128, 128]
          - search_images:   tensor [1, B, 6, 256, 256]
          - template_anno:   list/tensor [B, 4]
          - search_anno:     list/tensor [B, 4]
        【输出】(loss_outer, status_dict) 外环损失和日志
        """
        template_images_raw = data['template_images'][0].view(
            -1, *data['template_images'].shape[2:])  # [B, 6, 128, 128]
        search_images_raw = data['search_images'][0].view(
            -1, *data['search_images'].shape[2:])      # [B, 6, 256, 256]
        B = template_images_raw.shape[0]

        if B < 2:
            out_dict = self._forward_pass(data)
            return self._compute_losses(out_dict, data)

        current_epoch = data.get('epoch', 0)

        rgb_imgs = template_images_raw[:, :3, :, :]
        x_imgs = template_images_raw[:, 3:, :, :]

        with torch.no_grad():
            net = self.net.module if hasattr(self.net, 'module') else self.net
            try:
                feat_rgb = net.backbone.patch_embed(rgb_imgs)
                feat_x = net.backbone.patch_embed(x_imgs)
                rgb_features = feat_rgb.flatten(2).transpose(1, 2)
                x_features = feat_x.flatten(2).transpose(1, 2)
            except Exception as e:
                rgb_features = None
                x_features = None

        # 【v7.5方案】基于你的分析: 制造真正的Support-Query分布差异
        # 核心思想:
        #   Support = 简单场景 (无退化, α=0) → 让模型在简单数据上做内环适配
        #   Query  = 困难场景 (强退化, α=0.5) → 让模型学习从简单→困难的迁移能力
        #   这样FOMAML才有意义: "学会如何快速适应困难场景"
        #
        # 对比旧版(v7.4):
        #   v7.4: Query退化α=0.15 (太轻, 支持和查询几乎一样)
        #   v7.5: Query退化α=0.5 (足够重, 制造明显分布差异)

        support_size = max(1, int(B * self.support_ratio))

        # ===== Step 1: 随机划分 =====
        perm = torch.randperm(B, device=template_images_raw.device)
        s_idx = perm[:support_size].sort().values
        q_idx = perm[support_size:].sort().values

        # ===== Step 2: 数据切片 =====
        s_tmpl = template_images_raw[s_idx]  # [S, 6, 128, 128]
        s_srch = search_images_raw[s_idx]    # [S, 6, 256, 256]
        q_tmpl = template_images_raw[q_idx]  # [Q, 6, 128, 128]
        q_srch = search_images_raw[q_idx]    # [Q, 6, 256, 256]

        # ===== Step 3: 固定退化强度 (纯FOMAML - 方案A) =====
        # 【方案A核心】训练循环内使用固定退化强度, 不受模型状态影响!
        #   Support: 无退化 (α=0, 原始数据)
        #   Query:  固定退化 (α=FIXED_ALPHA, 与epoch/health无关)
        #
        # 纯FOMAML要求: Task Distribution必须独立于当前参数θ
        # 调整策略: 通过修改YAML中的DEGRADE_ALPHA或验证集结果手动调整
        current_epoch = data.get('epoch', 0)
        query_degrade_alpha = getattr(self.cfg.TRAIN.META, 'DEGRADE_ALPHA', 0.20)

        q_tmpl_degraded = self._apply_strong_degradation(q_tmpl, alpha=query_degrade_alpha)
        q_srch_degraded = self._apply_strong_degradation(q_srch, alpha=query_degrade_alpha)

        # ===== Step 4: 构建support_data和query_data =====
        t_anno = data.get('template_anno', None)
        s_anno = data.get('search_anno', None)

        def _slice_anno(val):
            if val is None:
                return None, None
            if isinstance(val, torch.Tensor):
                if val.dim() >= 2:
                    return val[:, s_idx], val[:, q_idx]
                else:
                    return val[s_idx], val[q_idx]
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                s_val = [v[:, s_idx] if v.dim() >= 2 else v[s_idx] for v in val]
                q_val = [v[:, q_idx] if v.dim() >= 2 else v[q_idx] for v in val]
                if all(v.numel() > 0 for v in s_val) and all(v.numel() > 0 for v in q_val):
                    return s_val, q_val
            return val, val

        s_t_anno, q_t_anno = _slice_anno(t_anno)
        s_s_anno, q_s_anno = _slice_anno(s_anno)

        support_data = {
            'template_images': s_tmpl.unsqueeze(0),
            'template_anno': s_t_anno,
            'search_images': s_srch.unsqueeze(0),
            'search_anno': s_s_anno,
            'epoch': data.get('epoch', 0),
        }
        query_data = {
            'template_images': q_tmpl_degraded.unsqueeze(0),
            'template_anno': q_t_anno,
            'search_images': q_srch_degraded.unsqueeze(0),
            'search_anno': q_s_anno,
            'epoch': data.get('epoch', 0),
        }

        # ===== 诊断输出 (仅第1次) =====
        if not hasattr(self, '_simple_split_diag_done'):
            print(f"[Meta-Learning] B={B}, S={support_size}, Q={B-support_size} | 方案A v2: 纯FOMAML(α={query_degrade_alpha:.2f}, 固定LR={self.inner_lr}, sup_reg={self.support_reg_weight})")
            self._simple_split_diag_done = True

        net = self.net.module if hasattr(self.net, 'module') else self.net
        meta_param_list, meta_name_list = self._get_meta_params(net)
        if len(meta_param_list) == 0:
            out_dict = self._forward_pass(data)
            return self._compute_losses(out_dict, data)

        if not hasattr(self, '_meta_logged'):
            self._meta_logged = True

        is_ddp = isinstance(self.net, torch.nn.parallel.DistributedDataParallel)
        if is_ddp and self.inner_lr > 0:
            context = self.net.no_sync()
        else:
            context = nullcontext()

        inner_losses = []
        grad_norms = []

        with context:
            out_inner = self._forward_pass(support_data)
            loss_inner, _ = self._compute_losses(out_inner, support_data)

            if not loss_inner.requires_grad or self.inner_lr <= 0 or self.update_step < 1:
                fast_weights = list(meta_param_list)
            else:
                grads = torch.autograd.grad(loss_inner, meta_param_list, retain_graph=False,
                                            create_graph=False, allow_unused=True)

                none_count = sum(1 for g in grads if g is not None)
                total_grad_norm = sum(g.norm().item() for g in grads if g is not None)

                if none_count == 0:
                    if not hasattr(self, '_warn_grad_none'):
                        print(f"[Meta-Learning] 🚨 All {len(grads)} gradients are None! prompt params may be disconnected from loss graph")
                        print(f"[Meta-Learning]   → Check: are prompt_blocks parameters frozen? (requires_grad=False)")
                        print(f"[Meta-Learning]   → Falling back: fast_weights = original params (no adaptation)")
                        self._warn_grad_none = True
                    fast_weights = list(meta_param_list)
                    inner_losses.append(loss_inner.item())
                    grad_norms.append(0.0)
                else:
                    # 【v7.6改进】双层梯度裁剪: 先按总范数裁剪, 再按单参数范数裁剪
                    if self.inner_grad_clip > 0 and total_grad_norm > self.inner_grad_clip:
                        clip_coef = self.inner_grad_clip / (total_grad_norm + 1e-6)
                        grads = [g * clip_coef if g is not None else g for g in grads]

                    # 【v7.6修复】第二层: 按单参数范数裁剪 (防止极端大梯度, 但不再硬限制在1.0!)
                    # 旧版bug: per_param_clip = min(inner_grad_clip, 1.0) → 永远≤1.0 → 参数无法更新!
                    # 新版方案: per_param_clip = inner_grad_clip * 0.3 → 允许更大的单参数梯度
                    per_param_clip = self.inner_grad_clip * 0.3  # inner_grad_clip=10 → per_param_clip=3.0
                    grads_clipped_again = []
                    for g in grads:
                        if g is not None:
                            g_norm = g.norm().item()
                            if g_norm > per_param_clip:
                                g = g * (per_param_clip / (g_norm + 1e-6))
                        grads_clipped_again.append(g)
                    grads = grads_clipped_again

                    total_grad_norm_clipped = sum(g.norm().item() for g in grads if g is not None)

                    grad_norms.append(total_grad_norm_clipped)
                    inner_losses.append(loss_inner.item())

                    # 【方案A v2】内环学习率: 固定值 (不用cosine衰减!)
                    # 原理: cosine衰减是给外环设计的, 内环只需1步更新
                    #       内环LR必须足够大才能产生有意义的Param Shift!
                    # 对比: 外环cosine让LR从0.0004衰减到~0.00002
                    #       内环固定0.05保证每步都有足够的参数更新幅度
                    effective_lr = self.inner_lr

                    fast_weights = list(map(
                        lambda p: p[1] - effective_lr * (p[0] if p[0] is not None else 0),
                        zip(grads, meta_param_list)))

                    for k in range(1, self.update_step):
                        out_inner_k = self._forward_pass_with_params(support_data, fast_weights, meta_name_list)
                        loss_inner_k, _ = self._compute_losses(out_inner_k, support_data)

                        if loss_inner_k.requires_grad:
                            grads_k = torch.autograd.grad(loss_inner_k, fast_weights, retain_graph=False,
                                                          create_graph=False, allow_unused=True)

                            step_none = sum(1 for g in grads_k if g is not None)
                            step_grad_norm = sum(g.norm().item() for g in grads_k if g is not None)

                            # 【v7.6改进】双层梯度裁剪 (同step0)
                            if self.inner_grad_clip > 0 and step_grad_norm > self.inner_grad_clip:
                                clip_coef_k = self.inner_grad_clip / (step_grad_norm + 1e-6)
                                grads_k = [g * clip_coef_k if g is not None else g for g in grads_k]

                            # 【v7.6修复】第二层: 按单参数范数裁剪 (同step0修复)
                            per_param_clip_k = self.inner_grad_clip * 0.3  # inner_grad_clip=10 → per_param_clip=3.0
                            grads_k_clipped = []
                            for g in grads_k:
                                if g is not None:
                                    g_norm_k = g.norm().item()
                                    if g_norm_k > per_param_clip_k:
                                        g = g * (per_param_clip_k / (g_norm_k + 1e-6))
                                grads_k_clipped.append(g)
                            grads_k = grads_k_clipped

                            step_grad_norm = sum(g.norm().item() for g in grads_k if g is not None)

                            grad_norms.append(step_grad_norm)
                            inner_losses.append(loss_inner_k.item())

                            fast_weights = list(map(
                                lambda p: p[1] - effective_lr * (p[0] if p[0] is not None else 0),
                                zip(grads_k, fast_weights)))
                        else:
                            break

        # 【v8.0】在forward前保存快照 + 核心诊断 (仅前2次)
        if self._step_count < 2:
            fast_weights_snapshot = [fw.clone().detach() for fw in fast_weights]
            grads_snapshot = [g.clone().detach() if g is not None else None for g in grads]

            pre_forward_max_diff = max((fw - pw).norm().item() for fw, pw in zip(fast_weights, meta_param_list))
            snapshot_max_diff = max((fw - pw).norm().item() for fw, pw in zip(fast_weights_snapshot, meta_param_list))
            print(f"  [PRE-FORWARD] max|fw-pw|(orig)={pre_forward_max_diff:.6f}, snapshot={snapshot_max_diff:.6f}")
        else:
            fast_weights_snapshot = None

        out_outer = self._forward_pass_with_params(query_data, fast_weights, meta_name_list)
        loss_outer, status = self._compute_losses(out_outer, query_data)

        # 【方案A】Support正则化: 固定权重 (纯FOMAML)
        # 外环loss = query_loss + support_reg_weight × support_loss
        # 固定值, 不受health/epoch影响, 保证Task Distribution独立性
        if self.support_reg_weight > 0 and len(inner_losses) > 0:
            with torch.no_grad():
                out_support_clean = self._forward_pass(support_data)
            loss_support, _ = self._compute_losses(out_support_clean, support_data)
            loss_outer = loss_outer + self.support_reg_weight * loss_support
            status['Meta/support_loss'] = loss_support.item()

        if len(inner_losses) > 0:
            status['Meta/inner_loss_0'] = inner_losses[0] if len(inner_losses) > 0 else 0
            status['Meta/inner_loss_last'] = inner_losses[-1] if len(inner_losses) > 0 else 0
            status['Meta/grad_norm_0'] = grad_norms[0] if len(grad_norms) > 0 else 0
            status['Meta/grad_norm_last'] = grad_norms[-1] if len(grad_norms) > 0 else 0
            status['Meta/inner_steps'] = len(inner_losses)

            diag_fast_w = fast_weights_snapshot if fast_weights_snapshot is not None else fast_weights

            param_shifts = [
                ((fw - pw).norm().item() / max(pw.norm().item(), 1e-6))
                for fw, pw in zip(diag_fast_w, meta_param_list)
            ] if len(diag_fast_w) > 0 else []
            max_param_shift = max(param_shifts) if param_shifts else 0
            mean_param_shift = sum(param_shifts) / len(param_shifts) if param_shifts else 0
            status['Meta/max_param_shift%'] = max_param_shift * 100
            status['Meta/mean_param_shift%'] = mean_param_shift * 100

            # 【v8.0精简】只在前2次打印核心诊断
            if self._step_count < 2 and fast_weights_snapshot is not None:
                post_forward_snap_diff = max((fw - pw).norm().item() for fw, pw in zip(fast_weights_snapshot, meta_param_list))
                post_forward_orig_diff = max((fw - pw).norm().item() for fw, pw in zip(fast_weights, meta_param_list))

                print(f"\n  [PARAM SHIFT v8.0] step={self._step_count}, LR={self.inner_lr}")
                print(f"    PRE-FORWARD:  snap_max_diff={snapshot_max_diff:.6f}")
                print(f"    POST-FORWARD: snap_max_diff={post_forward_snap_diff:.6f}, orig_max_diff={post_forward_orig_diff:.6f}")
                print(f"    RESULT: max_shift={max_param_shift*100:.4f}%, mean_shift={mean_param_shift*100:.4f}%")

                # Top-5参数
                top_5 = sorted(range(len(param_shifts)), key=lambda i: param_shifts[i], reverse=True)[:5]
                print(f"    Top-5 params:")
                for idx in top_5:
                    name = meta_name_list[idx].split('.')[-1] if idx < len(meta_name_list) else f'param[{idx}]'
                    delta = (fast_weights_snapshot[idx] - meta_param_list[idx]).norm().item()
                    orig_n = meta_param_list[idx].norm().item()
                    print(f"      {idx:>3}. {name:<20} |Δ|={delta:.6f} |orig|={orig_n:.4f} shift={param_shifts[idx]*100:.4f}%")


        current_epoch = data.get('epoch', 0)
        global_step = self._step_count

        if self.tb_writer is not None and global_step % 10 == 0:
            tw = self.tb_writer

            tw.add_scalar('Train/TotalLoss', status.get('Loss/total', 0), global_step)
            tw.add_scalar('Train/PredIoU', status.get('IoU', 0), global_step)

            tw.add_scalar('Pos/CIoU_Loss', status.get('Loss/giou', 0), global_step)
            tw.add_scalar('Pos/CenterDist', status.get('CenterDist', 0), global_step)

            tw.add_scalar('Cls/ScoreLoss', status.get('Loss/location', 0), global_step)
            tw.add_scalar('Reg/BBoxL1', status.get('Loss/l1', 0), global_step)

            if len(inner_losses) > 0:
                tw.add_scalar('Meta/InnerLoss', inner_losses[0], global_step)
                tw.add_scalar('Meta/ParamShift_pct', mean_param_shift * 100, global_step)
                if len(grad_norms) > 0:
                    tw.add_scalar('Meta/GradNorm', grad_norms[0], global_step)
                # 【v8.1】Support正则化监控
                if 'Meta/support_loss' in status:
                    tw.add_scalar('Meta/SupportLoss', status['Meta/support_loss'], global_step)
                    if 'Meta/support_reg' in status:
                        tw.add_scalar('Meta/DynamicSupReg', status['Meta/support_reg'], global_step)
                # 【v8.3】自适应参数监控 (核心!)
                tw.add_scalar('Meta/FixedAlpha', query_degrade_alpha, global_step)
                tw.add_scalar('Meta/EffectiveLR', self.inner_lr, global_step)

            # 【方案A】Health Score仅作监控 (不影响训练!)
            if 'Meta/HealthScore' in status:
                tw.add_scalar('Meta/HealthScore', status['Meta/HealthScore'], global_step)
                tw.add_scalar('Meta/IoUTrend', status['Meta/IoUTrend'], global_step)
                tw.add_scalar('Meta/LossTrend', status['Meta/LossTrend'], global_step)
                tw.add_scalar('Meta/InnerGain', status['Meta/InnerGain'], global_step)

            tw.add_scalar('Meta/Phase', float(current_epoch < self.warmup_epoch), global_step)
            # 【v7.5简化】移除task_info相关TensorBoard记录 (不再使用_task_aware_split)
            tw.flush()

        self._step_count += 1

        if not hasattr(self, '_diag_done'):
            out_baseline_full = self._forward_pass(data)
            _, stat_full = self._compute_losses(out_baseline_full, data)

            out_baseline_query = self._forward_pass(query_data)
            _, stat_query = self._compute_losses(out_baseline_query, query_data)

            out_identity = self._forward_pass_with_params(query_data, list(meta_param_list), meta_name_list)
            _, stat_identity = self._compute_losses(out_identity, query_data)

            print(f"\n{'='*60}")
            print(f"  [FOMAML] B={B}, S={support_size}, Q={B-support_size}")
            print(f"  {'─'*58}")
            print(f"  FULL(B={B:>2}) Loss={stat_full['Loss/total']:.4f} IoU={stat_full['IoU']:.4f}")
            print(f"  QUERY(Q={B-support_size:>1}) Loss={stat_query['Loss/total']:.4f} IoU={stat_query['IoU']:.4f}")
            print(f"  META-OUTER     Loss={loss_outer.item():.4f} IoU={status['IoU']:.4f}")
            if len(inner_losses) > 0:
                sup_info = f" SupLoss={status.get('Meta/support_loss', 0):.4f}" if 'Meta/support_loss' in status else ""
                health_val = status.get('Meta/HealthScore', self._adaptive_state.get('prev_health', 0.5))
                print(f"  Inner: loss0={inner_losses[0]:.4f} |∇|={grad_norms[0]:.2f}{sup_info}")
                print(f"  Health={health_val:.3f} α={query_degrade_alpha:.2f}(固定) effLR={self.inner_lr:.4f}(固定)")
            print(f"{'='*60}")

            print()
            self._diag_done = True

        # 【方案A】仅计算Health Score用于监控 (不影响训练参数!)
        monitor_info = self._update_adaptive_state(status, inner_losses, grad_norms)
        if monitor_info:
            status.update(monitor_info)
            if self._step_count % 500 == 0 and self._step_count > 0:
                h = monitor_info.get('Meta/HealthScore', 0)
                it = monitor_info.get('Meta/IoUTrend', 0)
                lt = monitor_info.get('Meta/LossTrend', 0)
                ig = monitor_info.get('Meta/InnerGain', 0)
                print(f"  [MONITOR] Health={h:.3f} | IoU_trend={it:+.3f} Loss_trend={lt:+.3f} Inner_gain={ig:.3f}")

        return loss_outer, status

    def _update_adaptive_state(self, status, inner_losses, grad_norms):
        """
        【功能】方案A: 仅计算Health Score用于监控, 不影响训练参数!
        【输入】
          - status: 当前step的指标dict (含IoU, Loss/total等)
          - inner_losses: 内环loss列表
          - grad_norms: 内环梯度范数列表
        【输出】monitor_info dict (用于TensorBoard记录) 或 None
        【方案A约束】
          纯FOMAML要求: Task Distribution必须独立于当前参数θ
          因此health score仅作监控指标, 不驱动任何训练超参数!
        """
        st = self._adaptive_state
        current_iou = status.get('IoU', 0.5)
        current_loss = status.get('Loss/total', 2.0)

        st['iou_window'].append(current_iou)
        st['loss_window'].append(current_loss)

        window_size = len(st['iou_window'])
        if window_size < 20:
            st['prev_health'] = 0.5
            return None

        recent_iou = list(st['iou_window'])[-20:]
        recent_loss = list(st['loss_window'])[-20:]

        half = 10
        iou_early = sum(recent_iou[:half]) / half
        iou_later = sum(recent_iou[half:]) / half
        loss_early = sum(recent_loss[:half]) / half
        loss_later = sum(recent_loss[half:]) / half

        iou_range = max(abs(iou_later - iou_early), 0.01)
        loss_range = max(abs(loss_later - loss_early), 0.1)

        iou_trend = (iou_later - iou_early) / iou_range
        loss_trend = -(loss_later - loss_early) / loss_range

        iou_score = 0.5 * (1 + math.tanh(iou_trend * 2))
        loss_score = 0.5 * (1 + math.tanh(loss_trend * 2))

        if len(inner_losses) >= 1 and len(grad_norms) >= 1:
            gain_score = 0.5
            try:
                il0 = inner_losses[0]
                ill = inner_losses[-1] if len(inner_losses) > 1 else il0
                if il0 > 0.01:
                    improvement = (il0 - ill) / il0
                    gain_score = 0.5 + 0.5 * min(max(improvement * 5, -1), 1)
            except Exception:
                pass
        else:
            gain_score = 0.5

        raw_health = 0.35 * iou_score + 0.30 * loss_score + 0.35 * gain_score
        smoothed_health = (self.health_smoothing * st['prev_health'] +
                          (1 - self.health_smoothing) * raw_health)
        health = max(0.05, min(0.95, smoothed_health))

        st['health_history'].append(health)
        st['prev_health'] = health
        st['initialized'] = True

        return {
            'Meta/HealthScore': health,
            'Meta/IoUTrend': iou_trend,
            'Meta/LossTrend': loss_trend,
            'Meta/InnerGain': gain_score,
        }

    def _split_batch(self, data, support_size):
        """
        【功能】沿batch维度(dim=1)将5D sequence格式数据分为Support/Query
        【输入】
          - data: 原始DataLoader输出dict
          - support_size: 支持集样本数
        【输出】(support_data, query_data) — 与原始data同构的dict，可直接传入_forward_pass

        【关键设计】保持sequence模式的5D结构 [1, B_sub, C, H, W]，
        确保_forward_pass和_compute_losses与基线ViPTActor完全兼容
        """
        s_idx = slice(0, support_size)
        q_idx = slice(support_size, None)

        def _slice_tensor(val):
            """对5D tensor沿dim=1切片"""
            return val[:, s_idx], val[:, q_idx]

        def _slice_list(val):
            """对list of tensors沿dim=0切片"""
            if len(val) > 0 and isinstance(val[0], torch.Tensor):
                return [v[s_idx] for v in val], [v[q_idx] for v in val]
            return val, val

        tmpl = data['template_images']
        srch = data['search_images']
        if isinstance(tmpl, torch.Tensor):
            s_tmpl, q_tmpl = _slice_tensor(tmpl)
            s_srch, q_srch = _slice_tensor(srch)
        elif isinstance(tmpl, (list, tuple)):
            s_tmpl, q_tmpl = _slice_list(tmpl)
            s_srch, q_srch = _slice_list(srch)
        else:
            raise ValueError(f"Unexpected template_images type: {type(tmpl)}")

        t_anno = data.get('template_anno', None)
        s_anno = data.get('search_anno', None)

        def _slice_anno(val):
            if val is None:
                return None, None
            if isinstance(val, torch.Tensor):
                if val.dim() >= 2:
                    s_val, q_val = val[:, s_idx], val[:, q_idx]
                else:
                    s_val, q_val = val[s_idx], val[q_idx]
                if s_val.numel() > 0 and q_val.numel() > 0:
                    return s_val, q_val
            elif isinstance(val, list) and len(val) > 0:
                if isinstance(val[0], torch.Tensor):
                    s_val = [v[:, s_idx] if v.dim() >= 2 else v[s_idx] for v in val]
                    q_val = [v[:, q_idx] if v.dim() >= 2 else v[q_idx] for v in val]
                    if all(v.numel() > 0 for v in s_val) and all(v.numel() > 0 for v in q_val):
                        return s_val, q_val
            return val, val

        s_t_anno, q_t_anno = _slice_anno(t_anno)
        s_s_anno, q_s_anno = _slice_anno(s_anno)

        support_data = {
            'template_images': s_tmpl,
            'template_anno': s_t_anno,
            'search_images': s_srch,
            'search_anno': s_s_anno,
            'epoch': data.get('epoch', 0),
        }
        query_data = {
            'template_images': q_tmpl,
            'template_anno': q_t_anno,
            'search_images': q_srch,
            'search_anno': q_s_anno,
            'epoch': data.get('epoch', 0),
        }
        return support_data, query_data

    def _task_aware_split(self, data, support_size, task_ids, alpha_rgb, alpha_x):
        """
        【功能】基于任务类型的感知划分 — support和query来自不同分布
        【v7.1改进版 - 解决真实数据task_evaluator判定偏差问题】

        【核心思想】
          原版问题: 真实LasHeR数据94%被判为Stress → Support/Query几乎一样 → FOMAML失效!
          改进方案: 基于【综合质量分数相对排序】而非绝对阈值判定

        【新策略】
          Step 1: 计算综合质量分数 Q_combined = (Q_rgb + Q_x) / 2 + consistency * 0.1
          Step 2: 按Q_combined从高到低排序所有样本
          Step 3: Support = 排名前50% (相对高质量样本)
                  Query  = 排名后50% (相对低质量样本) + 强制退化
          → 即使所有样本都是中等质量(0.4-0.5), 也能保证support≠query!

        【输入】
          - data: 原始数据dict
          - support_size: support集大小
          - task_ids: [B, 1] task_evaluator判定的任务类型 (0=Base,1=Stress,2=Conflict)
          - alpha_rgb/x: [B, 1] 各模态退化强度
        【输出】
          - support_data/query_data: 划分后的数据dict
          - task_info: 统计信息dict
        """
        B = data['template_images'].shape[1] if data['template_images'].dim() == 5 else data['template_images'].shape[0]
        device = task_ids.device
        tid = task_ids.squeeze(-1)

        base_mask = (tid == 0)
        stress_mask = (tid == 1)
        conflict_mask = (tid == 2)

        n_base = base_mask.sum().item()
        n_stress = stress_mask.sum().item()
        n_conflict = conflict_mask.sum().item()

        # ===== v7.1新增: 自适应阈值调整 =====
        # 如果Base占比<20%, 说明阈值过严, 启用基于排序的备选方案
        use_ranking_split = (n_base / B < 0.2) if B > 0 else False

        if use_ranking_split:
            # 【方案B: 基于质量分数排序的自适应划分】
            # 核心思路: 不依赖绝对阈值, 而是按相对排名划分

            # 从evaluation_info获取质量分数 (如果可用)
            eval_info = getattr(self, '_last_eval_info', None)
            if eval_info is not None and 'quality_rgb' in eval_info:
                q_rgb = eval_info['quality_rgb'].squeeze(-1)  # [B]
                q_x = eval_info['quality_x'].squeeze(-1)      # [B]
                consistency = eval_info.get('consistency', torch.ones(B, device=device)).squeeze(-1)  # [B]

                # 综合质量分数 = 双模态平均质量 + 一致性加权
                q_combined = (q_rgb + q_x) / 2.0 + consistency * 0.1

                # 按质量从高到低排序
                sorted_indices = torch.argsort(q_combined, descending=True)

                # Support = 前50% (高质量样本)
                # Query = 后50% (低质量样本)
                half_B = B // 2
                s_idx = sorted_indices[:half_B].sort().values
                q_idx = sorted_indices[half_B:].sort().values

                # 为Query生成额外的退化强度 (基于其排名位置)
                # 排名越靠后(质量越差),退化强度越大
                q_ranks = torch.zeros(B, device=device)
                q_ranks[q_idx] = torch.arange(len(q_idx), device=device, dtype=torch.float32) / max(len(q_idx), 1)
                # 将rank转换为alpha值 [0.2, 0.8]
                q_alpha_rgb_new = 0.2 + q_ranks[q_idx] * 0.6
                q_alpha_x_new = 0.2 + q_ranks[q_idx] * 0.6
            else:
                # 如果没有quality信息, 回退到随机划分但保证Query有退化
                perm = torch.randperm(B, device=device)
                half_B = B // 2
                s_idx = perm[:half_B].sort().values
                q_idx = perm[half_B:].sort().values
                q_alpha_rgb_new = torch.rand(len(q_idx), device=device) * 0.5 + 0.2  # [0.2, 0.7]
                q_alpha_x_new = torch.rand(len(q_idx), device=device) * 0.5 + 0.2
        else:
            # 【方案A: 原有的基于task类型划分】(当Base占比正常时使用)
            base_indices = torch.where(base_mask)[0]
            stress_indices = torch.where(stress_mask)[0]
            conflict_indices = torch.where(conflict_mask)[0]

            support_target = min(support_size, B // 2)
            query_target = B - support_target

            if n_base >= support_target:
                s_idx = base_indices[:support_target].sort().values
                remaining_base = base_indices[support_target:]
                remaining_stress = stress_indices
                remaining_conflict = conflict_indices
                q_candidates = torch.cat([remaining_base, remaining_stress, remaining_conflict], dim=0)
            else:
                s_idx = base_indices
                need_more = support_target - len(s_idx)
                extra_stress = stress_indices[:need_more]
                s_idx = torch.cat([s_idx, extra_stress], dim=0).sort().values
                remaining_stress = stress_indices[need_more:]
                q_candidates = torch.cat([remaining_stress, conflict_indices], dim=0)

            if len(q_candidates) >= query_target:
                q_idx = q_candidates[:query_target].sort().values
            else:
                all_indices = torch.arange(B, device=device)
                used = torch.zeros(B, dtype=torch.bool, device=device)
                used[s_idx] = True
                if len(q_candidates) > 0:
                    used[q_candidates] = True
                unused = torch.where(~used)[0]
                need_extra = query_target - len(q_candidates)
                extra = unused[:need_extra]
                q_idx = torch.cat([q_candidates[:query_target - len(extra)], extra], dim=0).sort().values

            q_alpha_rgb_new = alpha_rgb[q_idx].squeeze(-1)
            q_alpha_x_new = alpha_x[q_idx].squeeze(-1)

        s_idx_sorted = s_idx.sort().values
        q_idx_sorted = q_idx.sort().values

        def _slice_tensor(val):
            return val[:, s_idx_sorted], val[:, q_idx_sorted]

        def _slice_list(val):
            if len(val) > 0 and isinstance(val[0], torch.Tensor):
                return [v[:, s_idx_sorted] for v in val], [v[:, q_idx_sorted] for v in val]
            return val, val

        tmpl = data['template_images']
        srch = data['search_images']
        if isinstance(tmpl, torch.Tensor):
            s_tmpl, q_tmpl = _slice_tensor(tmpl)
            s_srch, q_srch = _slice_tensor(srch)
        elif isinstance(tmpl, (list, tuple)):
            s_tmpl, q_tmpl = _slice_list(tmpl)
            s_srch, q_srch = _slice_list(srch)
        else:
            raise ValueError(f"Unexpected template_images type: {type(tmpl)}")

        t_anno = data.get('template_anno', None)
        s_anno = data.get('search_anno', None)

        def _slice_anno(val):
            if val is None:
                return None, None
            if isinstance(val, torch.Tensor):
                if val.dim() >= 2:
                    s_val, q_val = val[:, s_idx_sorted], val[:, q_idx_sorted]
                else:
                    s_val, q_val = val[s_idx_sorted], val[q_idx_sorted]
                if s_val.numel() > 0 and q_val.numel() > 0:
                    return s_val, q_val
            elif isinstance(val, list) and len(val) > 0:
                if isinstance(val[0], torch.Tensor):
                    s_val = [v[:, s_idx_sorted] if v.dim() >= 2 else v[s_idx_sorted] for v in val]
                    q_val = [v[:, q_idx_sorted] if v.dim() >= 2 else v[q_idx_sorted] for v in val]
                    if all(v.numel() > 0 for v in s_val) and all(v.numel() > 0 for v in q_val):
                        return s_val, q_val
            return val, val

        s_t_anno, q_t_anno = _slice_anno(t_anno)
        s_s_anno, q_s_anno = _slice_anno(s_anno)

        # 使用新生成的alpha值 (v7.1改进: 基于排名的自适应退化)
        if use_ranking_split:
            q_alpha_rgb = q_alpha_rgb_new
            q_alpha_x = q_alpha_x_new
        else:
            q_alpha_rgb = alpha_rgb[q_idx_sorted].squeeze(-1)
            q_alpha_x = alpha_x[q_idx_sorted].squeeze(-1)
        q_tmpl, q_srch = self._apply_query_degradation(q_tmpl, q_srch, q_alpha_rgb, q_alpha_x)

        support_data = {
            'template_images': s_tmpl,
            'template_anno': s_t_anno,
            'search_images': s_srch,
            'search_anno': s_s_anno,
            'epoch': data.get('epoch', 0),
        }
        query_data = {
            'template_images': q_tmpl,
            'template_anno': q_t_anno,
            'search_images': q_srch,
            'search_anno': q_s_anno,
            'epoch': data.get('epoch', 0),
        }

        s_tid = tid[s_idx_sorted]
        q_tid = tid[q_idx_sorted]
        task_info = {
            'split_strategy': 'ranking' if use_ranking_split else 'task_type',  # v7.1新增
            'support_base_pct': (s_tid == 0).float().mean().item(),
            'support_stress_pct': (s_tid == 1).float().mean().item(),
            'support_conflict_pct': (s_tid == 2).float().mean().item(),
            'query_base_pct': (q_tid == 0).float().mean().item(),
            'query_stress_pct': (q_tid == 1).float().mean().item(),
            'query_conflict_pct': (q_tid == 2).float().mean().item(),
            'query_degrade_mean': (q_alpha_rgb.mean().item() + q_alpha_x.mean().item()) / 2,
            'base_ratio_global': n_base / B if B > 0 else 0,  # v7.1新增: 全局Base占比
        }

        return support_data, query_data, task_info

    def _apply_light_degradation(self, images, alpha=0.15):
        """
        【功能】轻度退化 - 仅添加轻微模糊和噪声 (v7.4极简方案)
        【输入】
          - images: [B, C, H, W] 图像tensor
          - alpha: 退化强度 (0=无退化, 1=重度退化)
        【输出】degraded: 退化后的图像
        """
        import torch.nn.functional as F

        if alpha <= 0 or images is None:
            return images

        device = images.device
        B, C, H, W = images.shape

        # 轻度高斯模糊 (sigma=0.5+alpha*1.5)
        sigma = 0.5 + alpha * 1.5  # alpha=0.15 → sigma≈0.73
        ksize = int(6 * sigma) | 1  # 确保奇数
        if ksize > 1:
            kernel = self._get_gaussian_kernel(ksize=ksize, sigma=sigma, channels=C, device=device)
            kernel = kernel.to(dtype=images.dtype)
            padding = ksize // 2
            blurred = F.conv2d(images, kernel, padding=padding, groups=C)
        else:
            blurred = images

        # 轻度噪声 (std=0.01+alpha*0.05)
        noise_std = 0.01 + alpha * 0.05  # alpha=0.15 → std=0.0175
        noise = torch.randn_like(images) * noise_std
        degraded = blurred + noise * (1 - images)  # 暗部噪声更明显

        return degraded.clamp(0, 1)

    def _get_gaussian_kernel(self, ksize, sigma, channels, device):
        """
        【功能】生成2D高斯模糊核
        【输入】
          - ksize: 核大小 (奇数)
          - sigma: 高斯标准差
          - channels: 通道数 (用于分组卷积)
          - device: 设备
        【输出】kernel: [channels, 1, ksize, ksize] 高斯核
        """
        import torch.nn.functional as F

        coords = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g_2d = g.unsqueeze(1) * g.unsqueeze(0)
        g_2d = g_2d / g_2d.sum()
        kernel = g_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        return kernel

    def _apply_strong_degradation(self, images, alpha=0.5):
        """
        【功能】强退化 - 制造Support-Query分布差异 (v7.5方案)
        【输入】
          - images: [B, C, H, W] 图像tensor (C=6: 前3=RGB, 后3=X模态)
          - alpha: 退化强度 (0=无退化, 1=重度退化)
        【输出】degraded: 强退化后的图像

        【退化策略】
          1. RGB通道: 中度模糊 + 噪声
          2. X模态通道: 重度模糊 + 噪声 + 亮度变化 → 制造跨模态不一致!
        """
        import torch.nn.functional as F

        if alpha <= 0 or images is None:
            return images

        device = images.device
        B, C, H, W = images.shape

        # 分离RGB和X模态
        rgb = images[:, :3, :, :]   # [B, 3, H, W]
        x_modality = images[:, 3:, :, :]  # [B, 3, H, W]

        # ===== RGB通道: 中度退化 =====
        # 模糊 (sigma=1.0+alpha*2.0)
        sigma_rgb = 1.0 + alpha * 2.0  # alpha=0.5 → sigma=2.0
        ksize_rgb = int(6 * sigma_rgb) | 1
        if ksize_rgb > 1:
            kernel_rgb = self._get_gaussian_kernel(ksize=ksize_rgb, sigma=sigma_rgb, channels=3, device=device)
            kernel_rgb = kernel_rgb.to(dtype=images.dtype)
            padding_rgb = ksize_rgb // 2
            rgb_blurred = F.conv2d(rgb, kernel_rgb, padding=padding_rgb, groups=3)
        else:
            rgb_blurred = rgb

        # 噪声 (std=0.02+alpha*0.08)
        noise_std_rgb = 0.02 + alpha * 0.08  # alpha=0.5 → std=0.06
        noise_rgb = torch.randn_like(rgb) * noise_std_rgb
        rgb_degraded = rgb_blurred + noise_rgb

        # ===== X模态通道: 重度退化 (制造跨模态不一致!) =====
        # 更强的模糊 (sigma=1.5+alpha*3.0)
        sigma_x = 1.5 + alpha * 3.0  # alpha=0.5 → sigma=3.0
        ksize_x = int(6 * sigma_x) | 1
        if ksize_x > 1:
            kernel_x = self._get_gaussian_kernel(ksize=ksize_x, sigma=sigma_x, channels=3, device=device)
            kernel_x = kernel_x.to(dtype=images.dtype)
            padding_x = ksize_x // 2
            x_blurred = F.conv2d(x_modality, kernel_x, padding=padding_x, groups=3)
        else:
            x_blurred = x_modality

        # 更强的噪声 (std=0.03+alpha*0.12)
        noise_std_x = 0.03 + alpha * 0.12  # alpha=0.5 → std=0.09
        noise_x = torch.randn_like(x_modality) * noise_std_x
        x_degraded = x_blurred + noise_x

        # 额外的亮度变化 (模拟热成像质量波动)
        brightness_shift = (torch.rand(B, 1, 1, 1, device=device) - 0.5) * alpha * 0.4  # [-0.1α, +0.1α]
        x_degraded = x_degraded + brightness_shift

        # 合并RGB和X模态
        degraded = torch.cat([rgb_degraded, x_degraded], dim=1)

        return degraded.clamp(0, 1)

    def _apply_query_degradation(self, template, search, alpha_rgb, alpha_x):
        """
        【功能】对Query集应用基于alpha值的可控退化，制造support-query分布差异
        【输入】
          - template/search: 5D张量 [1, Q, C, H, W]
          - alpha_rgb/x: [Q] 各样本退化强度 [0,1]
        【输出】退化后的template, search (支持梯度回传)
        【v7增强版退化策略 - 确保FOMAML生效前提: support≠query分布】
          alpha < 0.1: 极轻度退化 (微小噪声, 模拟传感器噪声)
          alpha 0.1-0.4: 轻度退化 (轻微模糊+噪声, 模拟白天→黄昏)
          alpha 0.4-0.7: 中度退化 (中等模糊+噪声+亮度偏移, 模拟夜间)
          alpha > 0.7: 重度退化 (强模糊+强噪声+大亮度偏移, 模拟极端天气)
          → 所有Query样本都会被退化(即使alpha=0也有极轻度退化)!
        """
        tmpl = template.clone()
        srch = search.clone()
        C = tmpl.shape[2]

        for i in range(template.shape[1]):
            a_r = alpha_rgb[i].item() if i < len(alpha_rgb) else 0.0
            a_x = alpha_x[i].item() if i < len(alpha_x) else 0.0
            max_alpha = max(a_r, a_x)

            blur_sigma = 0.3 + max_alpha * 5.0
            noise_std = 0.01 + max_alpha * 0.15
            brightness_shift = (max_alpha - 0.3) * 0.5

            ksize = int(6 * blur_sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            ksize = min(ksize, 21)

            if a_r >= 0.01:
                tmpl[0, i, :3] = self._degrad_blur_noise_brightness(
                    tmpl[0, i, :3], ksize, blur_sigma, noise_std, brightness_shift * a_r)
                srch[0, i, :3] = self._degrad_blur_noise_brightness(
                    srch[0, i, :3], ksize, blur_sigma, noise_std, brightness_shift * a_r)

            if a_x >= 0.01 and C > 3:
                x_ch_start = 3
                tmpl[0, i, x_ch_start:] = self._degrad_blur_noise(
                    tmpl[0, i, x_ch_start:], ksize, blur_sigma, noise_std * 1.5)
                srch[0, i, x_ch_start:] = self._degrad_blur_noise(
                    srch[0, i, x_ch_start:], ksize, blur_sigma, noise_std * 1.5)

        return tmpl, srch

    def _degrad_blur_noise_brightness(self, img, ksize, sigma, noise_std, bright_shift):
        """【功能】RGB模态退化: 高斯模糊 + 噪声 + 亮度偏移
        【输入】
          - img: [C, H, W] 输入图像
          - ksize: 卷积核大小
          - sigma: 高斯模糊sigma值
          - noise_std: 噪声标准差
          - bright_shift: 亮度偏移量
        【输出】[C, H, W] 退化后的图像
        """
        img = img.unsqueeze(0)
        pad = ksize // 2
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')

        y_coords = torch.arange(ksize, device=img.device, dtype=torch.float32) - pad
        x_coords = torch.arange(ksize, device=img.device, dtype=torch.float32) - pad
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        kernel_2d = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(img.shape[1], 1, ksize, ksize)
        blurred = F.conv2d(img_padded, kernel, groups=img.shape[1])

        if noise_std > 0.001:
            noise = torch.randn_like(blurred) * noise_std
            blurred = blurred + noise

        if abs(bright_shift) > 0.001:
            blurred = blurred + bright_shift
            blurred = blurred.clamp(0, 1)

        return blurred.squeeze(0)

    def _degrad_blur_noise(self, img, ksize, sigma, noise_std):
        """【功能】X模态(红外等)退化: 高斯模糊 + 噪声
        【输入】
          - img: [C, H, W] 输入图像
          - ksize: 卷积核大小
          - sigma: 高斯模糊sigma值
          - noise_std: 噪声标准差
        【输出】[C, H, W] 退化后的图像
        """
        img = img.unsqueeze(0)
        pad = ksize // 2
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')

        y_coords = torch.arange(ksize, device=img.device, dtype=torch.float32) - pad
        x_coords = torch.arange(ksize, device=img.device, dtype=torch.float32) - pad
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        kernel_2d = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(img.shape[1], 1, ksize, ksize)
        blurred = F.conv2d(img_padded, kernel, groups=img.shape[1])

        if noise_std > 0.001:
            noise = torch.randn_like(blurred) * noise_std
            blurred = (blurred + noise).clamp(0, 1)

        return blurred.squeeze(0)

    def _get_meta_params(self, net):
        """【输出】(meta_param_list, meta_name_list) 需要元学习的参数列表"""
        meta_params = {}
        meta_names = []
        net_backbone = net.backbone if hasattr(net, 'backbone') else net

        if hasattr(net_backbone, 'meta_prompt_generator') and net_backbone.meta_prompt_generator is not None:
            for name, param in net_backbone.meta_prompt_generator.named_parameters():
                if param.requires_grad:
                    meta_params[name] = param
                    meta_names.append(name)

        if len(meta_params) == 0 and hasattr(net_backbone, 'prompt_blocks'):
            for name, param in net_backbone.prompt_blocks.named_parameters():
                key = 'prompt_blocks.' + name
                if not param.requires_grad:
                    if not hasattr(self, '_warn_frozen'):
                        print(f"[Meta-Learning] ⚠️ FROZEN param: {key} requires_grad={param.requires_grad}, shape={list(param.shape)}")
                        self._warn_frozen = True
                    continue
                meta_params[key] = param
                meta_names.append(key)
            # 【v7.3修复】排除prompt_norms参数!
            # 原因: LayerNorm/BatchNorm的bias初始化为0 → norm≈0 → Param Shift公式除零爆炸!
            # prompt_norms应该跟随prompt_blocks自适应, 不需要独立被元学习优化
            if hasattr(net_backbone, 'prompt_norms') and not hasattr(self, '_norms_excluded_logged'):
                self._norms_excluded_logged = True

        if not hasattr(self, '_param_info_logged') and len(meta_names) > 0:
            total = sum(p.numel() for p in meta_params.values())
            print(f"[Meta-Learning] {len(meta_names)} groups, {total:,} params | LR={self.inner_lr} CLIP={self.inner_grad_clip}")
            self._param_info_logged = True

        param_list = [meta_params[n] for n in meta_names]
        return param_list, meta_names

    def _forward_pass_with_params(self, data, fast_weights, weight_names):
        """
        【功能】使用fast_weights替换指定参数后做前向传播（内环更新后的参数版本）
        【输入】
          - data: 标准数据dict
          - fast_weights: 内环更新后的参数列表
          - weight_names: 参数名称列表
        【输出】out_dict: 模型预测字典
        """
        net = self.net.module if hasattr(self.net, 'module') else self.net
        net_backbone = net.backbone if hasattr(net, 'backbone') else net

        param_dict = dict(zip(weight_names, fast_weights))
        orig_values = {}

        for name, new_param in param_dict.items():
            parts = name.split('.')
            target = net_backbone
            for part in parts[:-1]:
                target = getattr(target, part)
            attr_name = parts[-1]
            old_val = getattr(target, attr_name)
            # 【v8.0关键修复】必须clone副本! 否则_replace_param的copy_()会污染引用!
            orig_values[name] = old_val.data.clone().detach()
            _replace_param(target, attr_name, new_param)

        try:
            result = self._forward_pass(data)
        finally:
            for name, orig_tensor in orig_values.items():
                parts = name.split('.')
                target = net_backbone
                for part in parts[:-1]:
                    target = getattr(target, part)
                _replace_param(target, parts[-1], orig_tensor)

        return result

    def _forward_pass(self, data):
        """
        【功能】标准前向传播，与基线ViPTActor.forward_pass()完全一致
        【输入】data: 标准数据dict (sequence模式5D)
          - template_images: [1, B_sub, 6, 128, 128]
          - search_images:   [1, B_sub, 6, 256, 256]
          - template_anno:   list/tensor [B_sub, 4]
          - search_anno:     list/tensor [B_sub, 4]
        【输出】out_dict: 模型预测字典 {pred_boxes, score_map, ...}
        """
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(
                -1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])

        box_mask_z = None
        ce_keep_rate = None

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            t_anno_raw = data.get('template_anno', None)
            if t_anno_raw is not None:
                if isinstance(t_anno_raw, (list, tuple)) and len(t_anno_raw) > 0:
                    anno_for_ce = t_anno_raw[0]
                elif isinstance(t_anno_raw, torch.Tensor):
                    if t_anno_raw.dim() == 3:
                        anno_for_ce = t_anno_raw[0]
                    else:
                        anno_for_ce = t_anno_raw
                else:
                    anno_for_ce = None
                if anno_for_ce is not None and anno_for_ce.numel() > 0:
                    box_mask_z = generate_mask_cond(
                        self.cfg, template_list[0].shape[0],
                        template_list[0].device, anno_for_ce)
                    ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
                    ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
                    ce_keep_rate = adjust_keep_rate(
                        data['epoch'], warmup_epochs=ce_start_epoch,
                        total_epochs=ce_start_epoch + ce_warm_epoch,
                        ITERS_PER_EPOCH=1,
                        base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list, search=search_img,
                            ce_template_mask=box_mask_z, ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        from lib.models.vipt.meta_prompt import PromptVisualizer
        if self.training:
            PromptVisualizer.increment_step()
        return out_dict

    def _compute_losses(self, pred_dict, gt_dict, return_status=True):
        """
        【功能】计算损失，与基线ViPTActor.compute_losses()完全一致
        【输入】
          - pred_dict: 模型预测 {pred_boxes: [B,N,4], score_map: [B,1,H,W]}
          - gt_dict: 标注数据dict
        【输出】(loss, status) 或 loss
        """
        search_anno = gt_dict['search_anno']
        if isinstance(search_anno, torch.Tensor) and search_anno.dim() == 3:
            gt_bbox = search_anno[-1]
            gt_gaussian_maps = generate_heatmap(
                search_anno, self.cfg.DATA.SEARCH.SIZE,
                self.cfg.MODEL.BACKBONE.STRIDE)
        elif isinstance(search_anno, torch.Tensor) and search_anno.dim() == 2:
            gt_bbox = search_anno
            gt_gaussian_maps = generate_heatmap(
                [search_anno], self.cfg.DATA.SEARCH.SIZE,
                self.cfg.MODEL.BACKBONE.STRIDE)
        elif isinstance(search_anno, list) and len(search_anno) > 0:
            gt_bbox = search_anno[-1]
            gt_gaussian_maps = generate_heatmap(
                search_anno, self.cfg.DATA.SEARCH.SIZE,
                self.cfg.MODEL.BACKBONE.STRIDE)
        else:
            raise ValueError(f"Unexpected search_anno type: {type(search_anno)}")
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            pred_boxes = torch.nan_to_num(pred_boxes, nan=0.0, posinf=1.0, neginf=-1.0)

        pred_b = pred_boxes.size(0)

        if gt_bbox.size(0) != pred_b:
            gt_bbox = gt_bbox[:pred_b]
            gt_gaussian_maps = gt_gaussian_maps[:pred_b]

        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat(
            (1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)

        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except Exception:
            giou_loss = torch.tensor(0.0, device=pred_boxes_vec.device)
            iou = torch.tensor(0.0, device=pred_boxes_vec.device)

        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](
                pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss)

        if return_status:
            mean_iou = iou.detach().mean()
            pred_cx = (pred_boxes_vec[:, 0] + pred_boxes_vec[:, 2]) / 2
            pred_cy = (pred_boxes_vec[:, 1] + pred_boxes_vec[:, 3]) / 2
            gt_cx = (gt_boxes_vec[:, 0] + gt_boxes_vec[:, 2]) / 2
            gt_cy = (gt_boxes_vec[:, 1] + gt_boxes_vec[:, 3]) / 2
            center_dist = torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).detach().mean().item()

            status = {
                "Loss/total": loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "IoU": mean_iou.item(),
                "CenterDist": center_dist,
            }
            return loss, status
        return loss


def _replace_param(module, attr_name, new_tensor):
    """
    【功能】安全地替换模块参数 (v7.5修复版)
    【核心改进】使用data属性原地修改, 避免创建新对象!

    【旧版bug】
      当new_tensor不是Parameter时, 会执行: setattr(m, k, Parameter(new))
      → 创建了全新的Parameter对象 → id()改变 → 导致Identity REPL不一致

    【新版方案】
      直接修改原Parameter的data属性: old_param.data = new_tensor.data
      → 保持原对象不变, 只修改数值 → 完全等价于直接前向传播!
    """
    old_val = getattr(module, attr_name)

    if isinstance(old_val, torch.nn.Parameter):
        # ✅ 关键修复: 原地修改data, 不创建新对象!
        with torch.no_grad():
            old_val.data.copy_(new_tensor.data if isinstance(new_tensor, torch.nn.Parameter) else new_tensor)
    else:
        # 普通tensor/其他类型, 直接赋值
        if isinstance(new_tensor, torch.nn.Parameter):
            setattr(module, attr_name, new_tensor)
        else:
            setattr(module, attr_name, new_tensor)
