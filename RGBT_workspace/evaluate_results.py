"""
评估测试结果    快速测试小数据集
计算PR (Precision Rate) 和 SR (Success Rate)
"""
import os
import numpy as np
import argparse
from os.path import join, isdir


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def calculate_center_error(box1, box2):
    """计算中心点误差"""
    center1 = np.array([box1[0] + box1[2]/2, box1[1] + box1[3]/2])
    center2 = np.array([box2[0] + box2[2]/2, box2[1] + box2[3]/2])
    return np.linalg.norm(center1 - center2)


def evaluate_results(dataset_name, yaml_name, seq_home):
    """评估测试结果"""
    result_dir = f'./RGBT_workspace/results/{dataset_name}/{yaml_name}'
    
    if not os.path.exists(result_dir):
        print(f"Result directory not found: {result_dir}")
        return
    
    seq_list = [f for f in os.listdir(result_dir) if f.endswith('.txt')]
    
    if len(seq_list) == 0:
        print(f"No result files found in {result_dir}")
        return
    
    all_ious = []
    all_center_errors = []
    
    for seq_file in seq_list:
        seq_name = seq_file.replace('.txt', '')
        result_path = join(result_dir, seq_file)
        
        # 读取预测结果
        pred_boxes = np.loadtxt(result_path, delimiter=',')
        
        # 读取真值
        if dataset_name == 'LasHeR':
            gt_path = join(seq_home, seq_name, 'visible.txt')
            if os.path.exists(gt_path):
                gt_boxes = np.loadtxt(gt_path, delimiter=',')
            else:
                print(f"GT not found for {seq_name}")
                continue
        else:
            continue
        
        # 计算每帧的指标
        seq_ious = []
        seq_center_errors = []
        for i in range(min(len(pred_boxes), len(gt_boxes))):
            iou = calculate_iou(pred_boxes[i], gt_boxes[i])
            center_error = calculate_center_error(pred_boxes[i], gt_boxes[i])
            seq_ious.append(iou)
            seq_center_errors.append(center_error)
        
        all_ious.extend(seq_ious)
        all_center_errors.extend(seq_center_errors)
        
        print(f"{seq_name}: IoU={np.mean(seq_ious):.4f}, CenterError={np.mean(seq_center_errors):.2f}")
    
    # 计算整体指标
    if len(all_ious) > 0:
        # Success Rate (IoU > 0.5)
        sr = np.mean(np.array(all_ious) > 0.5) * 100
        # Precision Rate (Center Error < 20 pixels)
        pr = np.mean(np.array(all_center_errors) < 20) * 100
        # Normalized Precision Rate
        # AUC
        thresholds = np.arange(0, 1.05, 0.05)
        success_curve = [np.mean(np.array(all_ious) > t) for t in thresholds]
        auc = np.trapz(success_curve, thresholds) * 100
        
        print("\n" + "="*50)
        print(f"Overall Results on {dataset_name}")
        print("="*50)
        print(f"Success Rate (SR@0.5): {sr:.2f}%")
        print(f"Precision Rate (PR@20): {pr:.2f}%")
        print(f"AUC: {auc:.2f}%")
        print(f"Average IoU: {np.mean(all_ious):.4f}")
        print(f"Average Center Error: {np.mean(all_center_errors):.2f} pixels")
        print(f"Total frames: {len(all_ious)}")
        print(f"Total sequences: {len(seq_list)}")
        print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate tracking results.')
    parser.add_argument('--dataset_name', type=str, default='LasHeR', help='Name of dataset.')
    parser.add_argument('--yaml_name', type=str, default='lasher_meta_g4', help='Name of config file.')
    parser.add_argument('--seq_home', type=str, default='/home/apulis-dev/code/VIPT_gai/data/lasher/testingset', help='Dataset path.')
    args = parser.parse_args()
    
    evaluate_results(args.dataset_name, args.yaml_name, args.seq_home)
