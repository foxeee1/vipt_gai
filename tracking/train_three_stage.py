import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

import _init_paths
import lib.train.admin.settings as ws_settings
from lib.config.vipt import config as cfg_module
from lib.train.base_functions import get_optimizer_scheduler, update_settings
from lib.train.trainers.ltr_trainer import LTRTrainer
from lib.train.admin.multigpu import is_multi_gpu


def parse_args():
    parser = argparse.ArgumentParser(description='ViPT三阶段串行训练（同进程）')
    parser.add_argument('--script', type=str, default='vipt')
    parser.add_argument('--config', type=str, default='exp22_parallel_residual')
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--stage1_epochs', type=int, default=15)
    parser.add_argument('--stage2_epochs', type=int, default=20)
    parser.add_argument('--stage3_epochs', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 安全校验：开启梯度异常检测，自动定位NaN/Inf的产生位置
    torch.autograd.set_detect_anomaly(True)


def build_model_and_load_pretrain(cfg, settings):
    from lib.models.vipt.ostrack_prompt import build_viptrack
    net = build_viptrack(cfg, training=True)
    net.cuda()
    print(f"[模型] 构建完成，预训练权重已由build_viptrack自动加载")
    return net


def apply_stage_freeze(net, stage):
    # 【v25关键修复】统一冻结逻辑，确保与inject逻辑一致
    # Stage1: 仅模态Prompt+layer_gates可训练（ViT主干冻结）
    # Stage2: 仅辅助生成器+alpha可训练（ViT主干+模态Prompt冻结）
    # Stage3: 所有Prompt+门控+生成器可训练（ViT主干冻结）
    for name, param in net.named_parameters():
        param.requires_grad = False
    
    if stage == 1:
        for name, param in net.named_parameters():
            if any(kw in name for kw in ['rgb_prompt', 'tir_prompt', 'modality_prompt_proj',
                                          'rgb_type_token', 'tir_type_token', 'layer_gates']):
                param.requires_grad = True
        print(f"[Stage1] RGB/TIR模态Prompt + layer_gates可训练，ViT主干冻结")
    
    elif stage == 2:
        for name, param in net.named_parameters():
            # 安全校验：Stage2只训练辅助生成器和门控，不训练模态Prompt相关
            # modality_prompt_proj在Stage1已训练好，Stage2冻结防止破坏
            if any(kw in name for kw in ['consistency_generator', 'temporal_generator',
                                          'mask_generator', 'cross_attn_modulation',
                                          'alpha_consistency', 'alpha_temporal', 'alpha_mask',
                                          'layer_gates', 'pos_bias', 'temperature']):
                param.requires_grad = True
        print(f"[Stage2] 辅助生成器+layer_gates+pos_bias可训练，模态Prompt+ViT主干冻结")
    
    elif stage == 3:
        for name, param in net.named_parameters():
            if any(kw in name for kw in ['prompt', 'layer_gate', 'alpha_', 'generator',
                                          'cross_attn_modulation', 'type_token',
                                          'modality_prompt_proj', 'coop_gate',
                                          'gated_fusion', 'gate_feat_proj']):
                param.requires_grad = True
        print(f"[Stage3] 所有Prompt+门控+生成器可训练，ViT主干冻结")
    
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"[冻结] 可训练: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def update_optimizer_for_stage(optimizer, net, stage, cfg):
    new_param_groups = []
    
    base_lr = cfg.TRAIN.LR
    backbone_lr = base_lr * cfg.TRAIN.BACKBONE_MULTIPLIER
    gate_lr = base_lr * getattr(cfg.TRAIN, 'GATE_LR_MULTIPLIER', 0.5)
    
    default_group = {
        'params': [],
        'lr': base_lr,
        'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        'amsgrad': False,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'maximize': False,
        'foreach': None,
        'capturable': False,
        'differentiable': False,
        'fused': None
    }
    
    if stage == 1:
        group = default_group.copy()
        group['params'] = [p for n, p in net.named_parameters() if p.requires_grad]
        group['lr'] = base_lr * 1.5
        new_param_groups.append(group)
        print(f"[Stage1优化器] LR={base_lr*1.5:.2e} (Prompt需要较高LR)")
    
    elif stage == 2:
        group = default_group.copy()
        group['params'] = [p for n, p in net.named_parameters() if p.requires_grad]
        group['lr'] = base_lr * 0.5
        new_param_groups.append(group)
        print(f"[Stage2优化器] LR={base_lr*0.5:.2e} (辅助分支降半)")
    
    elif stage == 3:
        prompt_params = [p for n, p in net.named_parameters() 
                         if 'prompt' in n and p.requires_grad]
        gate_params = [p for n, p in net.named_parameters() 
                       if 'layer_gate' in n and p.requires_grad]
        other_params = [p for n, p in net.named_parameters() 
                        if 'prompt' not in n and 'layer_gate' not in n and p.requires_grad]
        
        if prompt_params:
            group = default_group.copy()
            group['params'] = prompt_params
            group['lr'] = base_lr
            new_param_groups.append(group)
        if gate_params:
            group = default_group.copy()
            group['params'] = gate_params
            group['lr'] = gate_lr
            new_param_groups.append(group)
        if other_params:
            group = default_group.copy()
            group['params'] = other_params
            group['lr'] = base_lr * 0.5
            new_param_groups.append(group)
        print(f"[Stage3优化器] Prompt LR={base_lr:.2e}, Gate LR={gate_lr:.2e}")
    
    optimizer.param_groups = new_param_groups
    # 安全校验：阶段切换时重建优化器状态
    # 原问题：保留旧阶段AdamW动量会导致新阶段参数更新方向错误
    # 修复：每个阶段开始时清空状态，让AdamW从零开始累积动量
    optimizer.state = defaultdict(dict)


def save_full_checkpoint(net, optimizer, lr_scheduler, epoch, best_iou, stage, save_path):
    net_to_save = net.module if is_multi_gpu(net) else net
    checkpoint = {
        'epoch': epoch,
        'stage': stage,
        'net_type': net_to_save.__class__.__name__,
        'net': net_to_save.state_dict(),
        'best_iou': best_iou,
    }
    try:
        checkpoint['optimizer'] = optimizer.state_dict()
    except KeyError:
        checkpoint['optimizer'] = None
        print("[警告] 优化器状态保存失败，已跳过")
    if lr_scheduler:
        try:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        except:
            checkpoint['lr_scheduler'] = None
    else:
        checkpoint['lr_scheduler'] = None
    torch.save(checkpoint, save_path)
    print(f"[保存] {save_path} (epoch={epoch}, iou={best_iou:.4f})")


def run_single_stage(stage_num, net, optimizer, lr_scheduler, trainer, settings, cfg,
                     epochs, best_iou, stage_ckpt_path, global_epoch_offset=0):
    print(f"\n{'='*70}")
    print(f"  Stage {stage_num} 开始训练 ({epochs} epochs)")
    print(f"{'='*70}")
    
    cfg.TRAIN.STAGE = stage_num
    
    # 设置当前训练阶段到meta_prompt_generator（用于控制inject行为）
    if hasattr(net, 'module'):
        meta_prompt = net.module.backbone.meta_prompt_generator
    else:
        meta_prompt = net.backbone.meta_prompt_generator
    if hasattr(meta_prompt, 'training_stage'):
        meta_prompt.training_stage = stage_num
        print(f"[Stage{stage_num}] 已设置 training_stage={stage_num}")
    
    # 安全校验：重置Temporal缓存，避免跨阶段缓存污染
    if hasattr(meta_prompt, 'reset_temporal_cache'):
        meta_prompt.reset_temporal_cache()
        print(f"[Stage{stage_num}] 已重置Temporal缓存")
    
    apply_stage_freeze(net, stage_num)
    update_optimizer_for_stage(optimizer, net, stage_num, cfg)
    
    # 安全校验：阶段切换时LR warmup — 前3个epoch使用1/10学习率
    # 避免阶段切换时Loss剧烈震荡
    warmup_epochs = 3
    original_lrs = [group['lr'] for group in optimizer.param_groups]
    
    early_stop_patience = 10
    early_stop_counter = 0
    early_stop_warmup = 5
    
    for epoch in range(1, epochs + 1):
        # 安全校验：LR warmup — 前3个epoch线性升温
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for group, orig_lr in zip(optimizer.param_groups, original_lrs):
                group['lr'] = orig_lr * warmup_factor
            if epoch == 1:
                print(f"[Stage{stage_num}] LR warmup: epoch {epoch}, factor={warmup_factor:.2f}")
        
        # 使用全局epoch作为tensorboard的step（确保曲线连续）
        global_epoch = global_epoch_offset + epoch
        trainer.epoch = global_epoch
        trainer.train_epoch()
        
        # 安全校验：每个epoch结束后梯度裁剪（防止梯度爆炸）
        trainable_params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
        if trainable_params:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if total_grad_norm > 10.0:
                print(f"[Stage{stage_num} Epoch {epoch}] 梯度裁剪: norm={total_grad_norm:.2f}")
        
        if lr_scheduler is not None:
            if cfg.TRAIN.SCHEDULER.TYPE == 'cosine':
                lr_scheduler.step(epoch - 1)
            else:
                lr_scheduler.step()
        
        current_iou = 0.0
        for loader_name, loader_stats in trainer.stats.items():
            if isinstance(loader_name, str) and 'val' in loader_name.lower() and loader_stats is not None:
                if 'IoU' in loader_stats and hasattr(loader_stats['IoU'], 'avg'):
                    current_iou = loader_stats['IoU'].avg
                    break
        
        print(f"[Stage{stage_num} Epoch {epoch}/{epochs}] IoU={current_iou:.4f} Best={best_iou:.4f}")
        
        if current_iou > best_iou:
            best_iou = current_iou
            early_stop_counter = 0
            save_full_checkpoint(net, optimizer, lr_scheduler, 
                                epoch + (stage_num-1) * epochs, best_iou, stage_num, 
                                stage_ckpt_path)
        else:
            if early_stop_patience > 0 and epoch > early_stop_warmup:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"[早停] Stage{stage_num} 连续{early_stop_patience}轮未提升")
                    break
    
    print(f"[Stage {stage_num}] 完成! Best IoU={best_iou:.4f}")
    return best_iou


def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 70)
    print("     ViPT 三阶段串行训练（同进程，权重/优化器状态完整继承）")
    print("=" * 70)
    print(f"  配置: {args.config}")
    print(f"  保存目录: {args.save_dir}")
    print(f"  Stage1: {args.stage1_epochs} epochs")
    print(f"  Stage2: {args.stage2_epochs} epochs")
    print(f"  Stage3: {args.stage3_epochs} epochs")
    print("=" * 70)
    
    cfg = cfg_module.cfg
    cfg_module.update_config_from_file(f'experiments/vipt/{args.config}.yaml')
    cfg = cfg_module.cfg
    
    cfg.TRAIN.EPOCH = args.stage1_epochs
    cfg.TRAIN.STAGE = 1
    cfg.TRAIN.EARLY_STOP_PATIENCE = 0
    
    settings = ws_settings.Settings()
    settings.script_name = 'vipt'
    settings.config_name = 'stage1'
    settings.project_path = 'train/vipt/stage1'
    settings.save_dir = os.path.abspath(args.save_dir)
    settings.local_rank = -1
    settings.device = torch.device('cuda:0')
    settings.use_lmdb = False
    settings.use_wandb = False
    settings.save_epoch_interval = 1
    settings.save_last_n_epoch = 1
    settings.early_stop_patience = 0
    settings.cfg = cfg
    
    log_dir = os.path.join(settings.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    settings.log_file = os.path.join(log_dir, 'vipt-three-stage.log')
    
    print("\n[Step1] 构建模型并加载预训练权重（仅此一次）...")
    net = build_model_and_load_pretrain(cfg, settings)
    
    print("\n[Step2] 构建数据加载器...")
    update_settings(settings, cfg)
    from lib.train.base_functions import build_dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)
    
    print("\n[Step3] 构建优化器和调度器...")
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    
    print("\n[Step4] 构建训练器...")
    from lib.train.actors.vipt import ViPTActor
    from lib.utils.focal_loss import FocalLoss
    from lib.utils.box_ops import giou_loss
    from torch.nn.functional import l1_loss
    from torch.nn import BCEWithLogitsLoss
    
    objective = {
        'giou': giou_loss,
        'l1': l1_loss,
        'focal': FocalLoss(),
        'cls': BCEWithLogitsLoss()
    }
    loss_weight = {
        'giou': cfg.TRAIN.GIOU_WEIGHT,
        'l1': cfg.TRAIN.L1_WEIGHT,
        'focal': 1.0,
        'cls': 1.0
    }
    
    actor = ViPTActor(
        net=net,
        objective=objective,
        loss_weight=loss_weight,
        settings=settings,
        cfg=cfg
    )
    
    trainer = LTRTrainer(
        actor,
        [loader_train, loader_val] if loader_val else [loader_train],
        optimizer,
        settings,
        lr_scheduler
    )
    
    ckpt_dir = os.path.join(settings.save_dir, 'checkpoints', 'train', 'vipt')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    stage1_ckpt = os.path.join(ckpt_dir, 'stage1_best.pth.tar')
    stage2_ckpt = os.path.join(ckpt_dir, 'stage2_best.pth.tar')
    stage3_ckpt = os.path.join(ckpt_dir, 'stage3_best.pth.tar')
    
    best_iou = 0.0
    total_start = time.time()
    
    try:
        print("\n" + "=" * 70)
        print("                    Stage 1: 模态Prompt训练")
        print("=" * 70)
        best_iou = run_single_stage(
            stage_num=1, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
            trainer=trainer, settings=settings, cfg=cfg,
            epochs=args.stage1_epochs, best_iou=best_iou, stage_ckpt_path=stage1_ckpt,
            global_epoch_offset=0
        )
        
        # Stage1完成后，全局epoch偏移量为stage1_epochs
        stage1_done_epochs = args.stage1_epochs
        
        print("\n" + "=" * 70)
        print("                    Stage 2: 辅助分支训练")
        print("=" * 70)
        
        # 【v25关键修复】不再重新初始化cross_attn_modulation
        # 原问题：Stage2重新初始化会破坏Stage1学到的特征表示
        # 修复：保留Stage1的权重，让辅助分支在已有特征空间上学习
        
        settings.config_name = 'stage2'
        settings.project_path = 'train/vipt/stage2'
        best_iou = run_single_stage(
            stage_num=2, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
            trainer=trainer, settings=settings, cfg=cfg,
            epochs=args.stage2_epochs, best_iou=best_iou, stage_ckpt_path=stage2_ckpt,
            global_epoch_offset=stage1_done_epochs
        )
        
        # Stage2完成后，全局epoch偏移量累加
        stage2_done_epochs = stage1_done_epochs + args.stage2_epochs
        
        print("\n" + "=" * 70)
        print("                    Stage 3: 联合微调")
        print("=" * 70)
        settings.config_name = 'stage3'
        settings.project_path = 'train/vipt/stage3'
        best_iou = run_single_stage(
            stage_num=3, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
            trainer=trainer, settings=settings, cfg=cfg,
            epochs=args.stage3_epochs, best_iou=best_iou, stage_ckpt_path=stage3_ckpt,
            global_epoch_offset=stage2_done_epochs
        )
        
        total_elapsed = time.time() - total_start
        print("\n" + "=" * 70)
        print("                    三阶段训练全部完成!")
        print("=" * 70)
        print(f"  总耗时: {total_elapsed/3600:.2f} 小时")
        print(f"  最终Best IoU: {best_iou:.4f}")
        print(f"  权重文件:")
        print(f"    Stage1: {stage1_ckpt}")
        print(f"    Stage2: {stage2_ckpt}")
        print(f"    Stage3: {stage3_ckpt}")
        print(f"  TensorBoard:")
        print(f"    tensorboard --logdir={settings.save_dir}/checkpoints/train/vipt --port 6006")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n[中断] 用户取消训练")
        save_full_checkpoint(net, optimizer, lr_scheduler, trainer.epoch, best_iou, 
                            cfg.TRAIN.STAGE, os.path.join(ckpt_dir, 'interrupted.pth.tar'))
    except Exception as e:
        import traceback
        print(f"\n[ERROR] 训练异常: {e}")
        traceback.print_exc()
        save_full_checkpoint(net, optimizer, lr_scheduler, trainer.epoch, best_iou,
                            cfg.TRAIN.STAGE, os.path.join(ckpt_dir, 'crashed.pth.tar'))


if __name__ == "__main__":
    main()
