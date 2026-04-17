#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
【功能】FOMAML v7 元任务集成验证脚本
【用途】在正式训练前验证 support/query 分布差异是否生效

【使用方式】
  python verify_meta_task.py --config experiments/vipt/exp2_baseline_metalearn.yaml

【预期输出】
  1. Task-Distribution 统计 (前3个batch)
  2. Support vs Query 的任务类型分布对比
  3. Query退化强度统计
  4. ✅/❌ 判定: FOMAML前提条件是否满足
"""

import sys
import os
import torch
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib.train.data.task_evaluator import AutoTaskEvaluator


def load_config(config_path):
    """加载YAML配置"""
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def create_mock_data(batch_size=32, img_size=128):
    """
    【v2增强版】创建模拟RGB-T数据 - 具有真实质量差异

    【输出】
      - rgb_imgs: [B, 3, H, W] RGB图像 (3种质量等级)
      - x_imgs: [B, 3, H, W] 红外图像 (3种质量等级)
    【数据分布】(模拟真实场景)
      - Base任务 (30%): 双模态都清晰 → Q_rgb>0.6 AND Q_x>0.6
      - Stress任务 (35%): 单模态退化 → 一个清晰一个模糊
      - Conflict任务 (35%): 双模态都退化 + 不一致
    """
    import torch.nn.functional as F

    B = batch_size
    n_base = int(B * 0.30)       # ~10 samples
    n_stress = int(B * 0.35)     # ~11 samples
    n_conflict = B - n_base - n_stress  # ~11 samples

    def _generate_clear_image(size, mean_val=0.5, contrast=0.25):
        """生成清晰图像 (高质量)"""
        img = torch.randn(3, size, size) * contrast + mean_val
        gradient_y = torch.linspace(0, 1, size).view(1, 1, size).expand(3, size, size)
        gradient_x = torch.linspace(0, 1, size).view(1, size, 1).expand(3, size, size)
        img = img + 0.15 * (gradient_y + gradient_x) / 2
        return img.clamp(0, 1)

    def _generate_blurry_image(size, blur_sigma=3.0, noise_std=0.08, brightness=0.4):
        """【功能】生成模糊图像 (低质量)
        【输入】
          - size: 图像尺寸 (H=W=size)
          - blur_sigma: 高斯模糊sigma值
          - noise_std: 噪声标准差
          - brightness: 基础亮度
        【输出】[3, H, W] 模糊+噪声的退化图像
        """
        img = torch.rand(3, size, size) * 0.15 + brightness
        ksize = int(6 * blur_sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        img = img.unsqueeze(0)
        pad = ksize // 2
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')

        y_coords = torch.arange(ksize, dtype=torch.float32) - pad
        x_coords = torch.arange(ksize, dtype=torch.float32) - pad
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        kernel_2d = torch.exp(-(xx ** 2 + yy ** 2) / (2 * blur_sigma ** 2))
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, ksize, ksize)

        blurred = F.conv2d(img_padded, kernel, groups=3)

        if noise_std > 0:
            noise = torch.randn_like(blurred) * noise_std
            blurred = (blurred + noise).clamp(0, 1)

        return blurred.squeeze(0)

    base_rgbs = torch.stack([_generate_clear_image(img_size, mean_val=0.5, contrast=0.28) for _ in range(n_base)])
    base_xs = torch.stack([_generate_clear_image(img_size, mean_val=0.55, contrast=0.26) for _ in range(n_base)])

    n_stress_a = n_stress // 2
    n_stress_b = n_stress - n_stress_a
    stress_a_rgbs = torch.stack([_generate_blurry_image(img_size, blur_sigma=4.0, noise_std=0.12, brightness=0.25) for _ in range(n_stress_a)])
    stress_a_xs = torch.stack([_generate_clear_image(img_size, mean_val=0.55, contrast=0.25) for _ in range(n_stress_a)])

    stress_b_rgbs = torch.stack([_generate_clear_image(img_size, mean_val=0.5, contrast=0.27) for _ in range(n_stress_b)])
    stress_b_xs = torch.stack([_generate_blurry_image(img_size, blur_sigma=5.0, noise_std=0.15, brightness=0.3) for _ in range(n_stress_b)])

    conflict_rgbs = torch.stack([_generate_blurry_image(img_size, blur_sigma=5.0, noise_std=0.18, brightness=0.2) for _ in range(n_conflict)])
    conflict_xs = torch.stack([_generate_blurry_image(img_size, blur_sigma=4.5, noise_std=0.16, brightness=0.35) for _ in range(n_conflict)])

    rgb_imgs = torch.cat([base_rgbs, stress_a_rgbs, stress_b_rgbs, conflict_rgbs], dim=0)
    x_imgs = torch.cat([base_xs, stress_a_xs, stress_b_xs, conflict_xs], dim=0)

    perm = torch.randperm(B)
    rgb_imgs = rgb_imgs[perm]
    x_imgs = x_imgs[perm]

    return rgb_imgs, x_imgs


def test_task_evaluator(cfg, num_batches=3):
    """
    测试task_evaluator的任务判定和退化生成
    【输入】
      - cfg: 配置字典
      - num_batches: 测试批次数
    【输出】打印任务分布统计
    """
    print("\n" + "=" * 70)
    print("  [FOMAML v7 验证] 元任务集成测试")
    print("=" * 70)

    meta_cfg = cfg.get('TRAIN', {}).get('META', {})

    high_thresh = meta_cfg.get('HIGH_QUALITY_THRESH', 0.6)
    low_thresh = meta_cfg.get('LOW_QUALITY_THRESH', 0.35)
    conflict_thresh = meta_cfg.get('CONFLICT_CONSISTENCY_THRESH', 0.25)

    print(f"\n[Config] 任务判定阈值:")
    print(f"  HIGH_QUALITY_THRESH = {high_thresh}")
    print(f"  LOW_QUALITY_THRESH = {low_thresh}")
    print(f"  CONFLICT_CONSISTENCY_THRESH = {conflict_thresh}")

    evaluator = AutoTaskEvaluator(
        modality_type_x='t',
        high_quality_thresh=high_thresh,
        low_quality_thresh=low_thresh,
        conflict_consistency_thresh=conflict_thresh
    )

    total_support_base = 0
    total_support_stress = 0
    total_support_conflict = 0
    total_query_base = 0
    total_query_stress = 0
    total_query_conflict = 0
    total_query_degrade = 0
    batch_count = 0

    support_ratio = meta_cfg.get('SUPPORT_RATIO', 0.5)

    for batch_idx in range(num_batches):
        B = 32
        rgb_imgs, x_imgs = create_mock_data(B)

        task_ids, alpha_rgb, alpha_x, eval_info = evaluator.evaluate(rgb_imgs, x_imgs)

        tid = task_ids.squeeze(-1)
        support_size = max(1, int(B * support_ratio))

        base_mask = (tid == 0)
        stress_mask = (tid == 1)
        conflict_mask = (tid == 2)

        n_base = base_mask.sum().item()
        n_stress = stress_mask.sum().item()
        n_conflict = conflict_mask.sum().item()

        base_indices = torch.where(base_mask)[0]
        stress_indices = torch.where(stress_mask)[0]
        conflict_indices = torch.where(conflict_mask)[0]

        if n_base >= support_size:
            s_idx = base_indices[:support_size]
            remaining = torch.cat([base_indices[support_size:], stress_indices, conflict_indices])
        else:
            s_idx = torch.cat([base_indices, stress_indices[:support_size - len(base_indices)]])
            remaining = torch.cat([stress_indices[len(s_idx) - len(base_indices):], conflict_indices])

        q_size = B - support_size
        if len(remaining) >= q_size:
            q_idx = remaining[:q_size]
        else:
            all_idx = torch.arange(B)
            used = torch.zeros(B, dtype=torch.bool)
            used[s_idx] = True
            used[remaining] = True
            unused = torch.where(~used)[0]
            need_extra = q_size - len(remaining)
            q_idx = torch.cat([remaining, unused[:need_extra]])

        s_tid = tid[s_idx]
        q_tid = tid[q_idx]

        s_base = (s_tid == 0).sum().item()
        s_stress = (s_tid == 1).sum().item()
        s_conflict = (s_tid == 2).sum().item()

        q_base = (q_tid == 0).sum().item()
        q_stress = (q_tid == 1).sum().item()
        q_conflict = (q_tid == 2).sum().item()

        q_alpha_rgb = alpha_rgb[q_idx].squeeze(-1)
        q_alpha_x = alpha_x[q_idx].squeeze(-1)
        q_degrade_mean = ((q_alpha_rgb.mean() + q_alpha_x.mean()) / 2).item()

        total_support_base += s_base
        total_support_stress += s_stress
        total_support_conflict += s_conflict
        total_query_base += q_base
        total_query_stress += q_stress
        total_query_conflict += q_conflict
        total_query_degrade += q_degrade_mean
        batch_count += 1

        print(f"\n[Batch {batch_idx + 1}] B={B}, S={support_size}, Q={q_size}")
        print(f"  全局: Base={n_base}, Stress={n_stress}, Conflict={n_conflict}")
        print(f"  Support: Base={s_base}({s_base/support_size*100:.0f}%), "
              f"Stress={s_stress}({s_stress/support_size*100:.0f}%), "
              f"Conflict={s_conflict}({s_conflict/support_size*100:.0f}%)")
        print(f"  Query:   Base={q_base}({q_base/q_size*100:.0f}%), "
              f"Stress={q_stress}({q_stress/q_size*100:.0f}%), "
              f"Conflict={q_conflict}({q_conflict/q_size*100:.0f}%)")
        print(f"  Query degrade_mean={q_degrade_mean:.3f}")

        if batch_idx == 0:
            q_rgb = eval_info['quality_rgb'].squeeze().tolist()
            q_x = eval_info['quality_x'].squeeze().tolist()
            consistency = eval_info['consistency'].squeeze().tolist()
            task_type_names = ['Base', 'Stress', 'Conflict']
            tid_list = tid.tolist()

            print(f"\n  [细粒度诊断] 前8个样本:")
            print(f"  {'Idx':>4} | {'Task':<9} | {'Q_rgb':>6} | {'Q_x':>6} | {'Consist':>7} | {'α_rgb':>6} | {'α_x':>6}")
            print(f"  {'-'*68}")
            for i in range(min(8, B)):
                tname = task_type_names[tid_list[i]] if tid_list[i] < 3 else '???'
                print(f"  {i:>4} | {tname:<9} | {q_rgb[i]:>6.3f} | {q_x[i]:>6.3f} | {consistency[i]:>7.3f} | "
                      f"{alpha_rgb[i].item():>6.3f} | {alpha_x[i].item():>6.3f}")

            print(f"\n  [质量分数统计]")
            print(f"    Q_rgb: min={min(q_rgb):.3f}, max={max(q_rgb):.3f}, mean={sum(q_rgb)/len(q_rgb):.3f}")
            print(f"    Q_x:   min={min(q_x):.3f}, max={max(q_x):.3f}, mean={sum(q_x)/len(q_x):.3f}")
            print(f"    阈值: HIGH={high_thresh}, LOW={low_thresh}")

    print("\n" + "-" * 70)
    print("  [汇总统计] (平均 over {} batches)".format(batch_count))
    print("-" * 70)

    avg_s_base = total_support_base / batch_count / (32 * support_ratio)
    avg_s_stress = total_support_stress / batch_count / (32 * support_ratio)
    avg_s_conflict = total_support_conflict / batch_count / (32 * support_ratio)

    avg_q_base = total_query_base / batch_count / (32 * (1 - support_ratio))
    avg_q_stress = total_query_stress / batch_count / (32 * (1 - support_ratio))
    avg_q_conflict = total_query_conflict / batch_count / (32 * (1 - support_ratio))
    avg_q_degrade = total_query_degrade / batch_count

    print(f"\n  Support集平均分布:")
    print(f"    Base:     {avg_s_base*100:.1f}%")
    print(f"    Stress:   {avg_s_stress*100:.1f}%")
    print(f"    Conflict: {avg_s_conflict*100:.1f}%")

    print(f"\n  Query集平均分布:")
    print(f"    Base:     {avg_q_base*100:.1f}%")
    print(f"    Stress:   {avg_q_stress*100:.1f}%")
    print(f"    Conflict: {avg_q_conflict*100:.1f}%")

    print(f"\n  Query退化强度: mean={avg_q_degrade:.3f}")

    print("\n" + "=" * 70)
    print("  [FOMAML前提条件验证]")
    print("=" * 70)

    condition1 = avg_s_base > 0.5
    condition2 = (avg_q_stress + avg_q_conflict) > 0.3
    condition3 = avg_q_degrade > 0.05

    print(f"\n  条件1: Support以Base为主 (>50%)?")
    print(f"    → {'✅ PASS' if condition1 else '❌ FAIL'} (实际: {avg_s_base*100:.1f}%)")

    print(f"\n  条件2: Query以Stress/Conflict为主 (>30%)?")
    print(f"    → {'✅ PASS' if condition2 else '❌ FAIL'} (实际: {(avg_q_stress+avg_q_conflict)*100:.1f}%)")

    print(f"\n  条件3: Query有明确退化 (degrade_mean > 0.05)?")
    print(f"    → {'✅ PASS' if condition3 else '❌ FAIL'} (实际: {avg_q_degrade:.3f})")

    all_pass = condition1 and condition2 and condition3
    print(f"\n  {'🎉 全部通过! FOMAML前提条件满足! 可以开始训练!' if all_pass else '⚠️ 未全部通过, 需要调整参数'}")
    print("=" * 70 + "\n")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description='FOMAML v7 元任务集成验证')
    parser.add_argument('--config', type=str,
                        default='experiments/vipt/exp2_baseline_metalearn.yaml',
                        help='配置文件路径')
    parser.add_argument('--num_batches', type=int, default=3,
                        help='测试批次数')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    success = test_task_evaluator(cfg, args.num_batches)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
