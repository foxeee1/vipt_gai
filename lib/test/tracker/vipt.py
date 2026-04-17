import math
from lib.models.vipt import build_viptrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn as nn
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import vot
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


def _replace_param(target, name, new_tensor):
    """【功能】安全替换模块参数（支持Parameter和普通Tensor）
    【输入】target: 目标模块, name: 属性名, new_tensor: 新张量
    """
    if isinstance(new_tensor, torch.nn.Parameter):
        setattr(target, name, nn.Parameter(new_tensor.data.clone().to(target.device)))
    else:
        setattr(target, name, new_tensor.clone().to(target.device))


class ViPTTrack(BaseTracker):
    def __init__(self, params):
        super(ViPTTrack, self).__init__(params)
        network = build_viptrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = False #params.debug
        self.debug = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        # 【新增】测试时元学习适应配置
        self.adapt_steps = getattr(self.cfg.TEST, 'META_ADAPT_STEPS', 0)
        self.adapt_lr = getattr(self.cfg.TEST, 'META_ADAPT_LR', 0.00001)
        self.adapted_params = None  # 存储适应后的参数

    def _get_adaptable_params(self):
        """【功能】获取可适应的元学习参数 (与训练时一致: prompt_blocks + prompt_norms)
        【输出】(param_list, name_list) 参数列表和名称列表
        """
        net_backbone = self.network.backbone
        param_list = []
        name_list = []

        if hasattr(net_backbone, 'meta_prompt_generator') and net_backbone.meta_prompt_generator is not None:
            for name, param in net_backbone.meta_prompt_generator.named_parameters():
                if param.requires_grad:
                    param_list.append(param)
                    name_list.append(f'meta_prompt_generator.{name}')

        if len(param_list) == 0 and hasattr(net_backbone, 'prompt_blocks'):
            for name, param in net_backbone.prompt_blocks.named_parameters():
                if param.requires_grad:
                    param_list.append(param)
                    name_list.append(f'prompt_blocks.{name}')
            if hasattr(net_backbone, 'prompt_norms'):
                for name, param in net_backbone.prompt_norms.named_parameters():
                    if param.requires_grad:
                        param_list.append(param)
                        name_list.append(f'prompt_norms.{name}')

        return param_list, name_list

    def _test_time_adapt(self, template_tensor):
        """【功能】测试时内环适应：用template做K步梯度下降更新prompt参数
        【输入】template_tensor: 模板图像tensor [1,6,H,W]
        【输出】adapted_params: 适应后的参数列表 (None表示未适应)
        【原理】利用template自重建损失做self-supervised adaptation
               让模型针对当前序列的视觉特征优化prompt
        """
        if self.adapt_steps < 1:
            return None

        param_list, name_list = self._get_adaptable_params()
        if len(param_list) == 0:
            return None

        print(f"[Test-Adapt] Starting {self.adapt_steps}-step adaptation (lr={self.adapt_lr}, {len(param_list)} param groups)")

        self.network.train()  # 需要计算梯度
        fast_weights = [p.clone().detach().requires_grad_(True) for p in param_list]

        for step in range(self.adapt_steps):
            # 用当前fast_weights做前向传播
            orig_values = {}
            for name, new_p in zip(name_list, fast_weights):
                parts = name.split('.')
                target = self.network.backbone
                for part in parts[:-1]:
                    target = getattr(target, part)
                attr_name = parts[-1]
                orig_values[name] = getattr(target, attr_name)
                _replace_param(target, attr_name, new_p)

            try:
                out = self.network.forward(
                    template=template_tensor,
                    search=template_tensor,
                    ce_template_mask=self.box_mask_z)

                score_map = out['score_map']
                adapt_loss = -score_map.mean()  # 最大化响应图均值 (self-supervised)

                grads = torch.autograd.grad(adapt_loss, fast_weights,
                                            retain_graph=False, create_graph=False, allow_unused=True)

                grad_norm = sum(g.norm().item() for g in grads if g is not None)
                print(f"[Test-Adapt] Step {step+1}/{self.adapt_steps}: loss={adapt_loss.item():.4f}, |∇|={grad_norm:.4f}")

                fast_weights = [
                    p - self.adapt_lr * (g if g is not None else torch.zeros_like(p))
                    for p, g in zip(fast_weights, grads)
                ]
            finally:
                for name, orig_t in orig_values.items():
                    parts = name.split('.')
                    target = self.network.backbone
                    for part in parts[:-1]:
                        target = getattr(target, part)
                    _replace_param(target, parts[-1], orig_t)

        self.network.eval()
        print(f"[Test-Adapt] ✓ Adaptation complete!")
        return fast_weights

    def _apply_adapted_params(self, adapted_params, name_list):
        """【功能】临时应用适应后的参数做推理
        【输入】adapted_params: 适应后参数, name_list: 参数名列表
        【输出】(orig_values_dict, cleanup_fn) 原始值字典和恢复函数
        """
        orig_values = {}
        for name, new_p in zip(name_list, adapted_params):
            parts = name.split('.')
            target = self.network.backbone
            for part in parts[:-1]:
                target = getattr(target, part)
            attr_name = parts[-1]
            orig_values[name] = getattr(target, attr_name)
            _replace_param(target, attr_name, new_p.detach())

        return orig_values

    def _restore_params(self, orig_values):
        """【功能】恢复原始参数"""
        for name, orig_t in orig_values.items():
            parts = name.split('.')
            target = self.network.backbone
            for part in parts[:-1]:
                target = getattr(target, part)
            _replace_param(target, parts[-1], orig_t)

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

        # 初始化元提示的历史特征
        if hasattr(self.network.backbone, 'meta_prompt_generator'):
            self.network.backbone.prev_features = None

        # 【新增】测试时内环适应
        _, self._adapt_name_list = self._get_adaptable_params()
        self.adapted_params = self._test_time_adapt(self.z_tensor)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_tensor = search
            # 【新增】如果存在适应后的参数，临时应用
            orig_values = None
            if self.adapted_params is not None and len(self._adapt_name_list) > 0:
                orig_values = self._apply_adapted_params(self.adapted_params, self._adapt_name_list)

            try:
                # merge the template and the search
                # run the transformer
                out_dict = self.network.forward(
                    template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)
            finally:
                if orig_values is not None:
                    self._restore_params(orig_values)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return ViPTTrack
