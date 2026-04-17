"""
快速采样测试脚本
只测试部分视频序列，节省测试时间
"""
import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
import random
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.vipt import ViPTTrack
import lib.test.parameter.vipt as rgbt_prompt_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time
import traceback

# 缓存 Tracker 实例，避免每个序列都重新加载权重
_tracker_cache = {}

def load_text(path):
    if not os.path.exists(path):
        return None
    try:
        # 尝试逗号分隔符
        return np.loadtxt(path, delimiter=',')
    except:
        try:
            # 尝试空格分隔符
            return np.loadtxt(path)
        except:
            return None

def genConfig(seq_path, set_type):
    if set_type == 'RGBT234':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])
        RGB_gt = load_text(seq_path + '/visible.txt')
        T_gt = load_text(seq_path + '/infrared.txt')
    elif set_type == 'GTOT':
        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])
        RGB_gt = load_text(seq_path + '/groundTruth_v.txt')
        T_gt = load_text(seq_path + '/groundTruth_i.txt')
        if RGB_gt is not None:
            x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
            y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
            x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
            y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
            RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)
        if T_gt is not None:
            x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
            y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
            x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
            y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
            T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)
    elif set_type == 'LasHeR':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])
        RGB_gt = load_text(seq_path + '/visible.txt')
        T_gt = load_text(seq_path + '/infrared.txt')
    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if p.endswith(".jpg")])
        RGB_gt = load_text(seq_path + '/rgb.txt')
        T_gt = load_text(seq_path + '/ir.txt')
    return RGB_img_list, T_img_list, RGB_gt, T_gt

def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='prompt'):
    try:
        if 'VTUAV' in dataset_name:
            seq_txt = seq_name.split('/')[1]
        else:
            seq_txt = seq_name
        # 原始保存路径（兼容 evaluate_results.py 轻量评估）
        save_name = '{}'.format(yaml_name)
        save_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
        save_folder = f'./RGBT_workspace/results/{dataset_name}/' + save_name
        # 评估框架路径（兼容 analysis_results.py 标准评估，制表符分隔+整数格式）
        eval_save_path = f'./output/test/tracking_results/vipt/lasher/{seq_txt}.txt'
        eval_save_folder = f'./output/test/tracking_results/vipt/lasher'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(eval_save_folder):
            os.makedirs(eval_save_folder)
        if os.path.exists(save_path) and os.path.exists(eval_save_path):
            print(f'-1 {seq_name}')
            return
        
        try:
            worker_name = multiprocessing.current_process().name
            # 更加稳妥地从名字中提取数字，例如 SpawnPoolWorker-1 或 Terminal#1-147
            import re
            worker_numbers = re.findall(r'\d+', worker_name)
            if worker_numbers:
                worker_id = int(worker_numbers[-1]) - 1 # 取最后一个数字
            else:
                worker_id = 0
            gpu_id = worker_id % num_gpu
            torch.cuda.set_device(gpu_id)
        except Exception as e:
            print(f"Warning: Could not set GPU device: {e}")
            pass

        global _tracker_cache
        cache_key = f"{yaml_name}_{epoch}_{script_name}"
        
        if cache_key not in _tracker_cache:
            params = rgbt_prompt_params.parameters(yaml_name, epoch)
            if not os.path.exists(params.checkpoint):
                raise FileNotFoundError(f"Checkpoint not found at: {params.checkpoint}")
            
            print(f"[{multiprocessing.current_process().name}] Loading weights from {params.checkpoint} on GPU {gpu_id}...")
            mmtrack = ViPTTrack(params)
            _tracker_cache[cache_key] = (mmtrack, params)
        
        mmtrack, params = _tracker_cache[cache_key]
        tracker = ViPT_RGBT(tracker=mmtrack)

        seq_path = seq_home + '/' + seq_name
        print(f'Processing sequence: {seq_name} (Worker: {multiprocessing.current_process().name})')
        RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
        
        if RGB_gt is None or T_gt is None:
            print(f"Error: Could not load ground truth for {seq_name}")
            return

        if len(RGB_img_list) == 0:
            print(f"Error: No images found in {seq_path}")
            return

        if len(RGB_img_list) == len(RGB_gt):
            result = np.zeros_like(RGB_gt)
        else:
            result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
        
        result[0] = np.copy(RGB_gt[0])
        toc = 0
        for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
            tic = cv2.getTickCount()
            if frame_idx == 0:
                image = get_x_frame(rgb_path, T_path, dtype='rgbrgb')
                tracker.initialize(image, RGB_gt[0].tolist())
            elif frame_idx > 0:
                image = get_x_frame(rgb_path, T_path, dtype='rgbrgb')
                region, confidence = tracker.track(image)
                result[frame_idx] = np.array(region)
            toc += cv2.getTickCount() - tic
        
        toc /= cv2.getTickFrequency()
        if not debug:
            np.savetxt(save_path, result, delimiter=',')
            np.savetxt(eval_save_path, result.astype(int), delimiter='\t', fmt='%d')
        print('{} , fps:{}'.format(seq_name, frame_idx / toc))
        
    except Exception as e:
        print(f"Error in sequence {seq_name}: {str(e)}")
        traceback.print_exc()


class ViPT_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick test tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='vipt', help='Name of tracking method.')
    parser.add_argument('--yaml_name', type=str, default='lasher_meta_g4', help='Name of config file.')
    parser.add_argument('--dataset_name', type=str, default='LasHeR', help='Name of dataset.')
    parser.add_argument('--threads', default=4, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=None, type=int, help='Epochs of checkpoint (None表示使用models目录下的权重)')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='Visualize tracking results')
    parser.add_argument('--video', default='', type=str, help='Specific video name')
    parser.add_argument('--seq_home', default=None, type=str, help='Path to dataset')
    # 快速采样参数
    parser.add_argument('--sample_ratio', default=0.1, type=float, help='采样10%的数据')
    parser.add_argument('--sample_seed', default=42, type=int, help='固定种子保证结果可复现')
    parser.add_argument('--max_videos', default=25, type=int, help='限制最大25个视频')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    
    # 数据集路径配置
    if args.seq_home is not None:
        seq_home = args.seq_home
    elif dataset_name == 'GTOT':
        seq_home = '/home/lz/Videos/GTOT'
    elif dataset_name == 'RGBT234':
        seq_home = '/media/jiawen/Datasets/Tracking/DATASET_TEST/RGBT234'
    elif dataset_name == 'LasHeR':
        seq_home = '/home/apulis-dev/code/VIPT_gai/data/lasher/testingset'
    elif dataset_name == 'VTUAVST':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/short-term'
    elif dataset_name == 'VTUAVLT':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/long-term'
    else:
        raise ValueError("Error dataset!")
    
    # 获取视频序列列表
    if dataset_name in ['VTUAVST', 'VTUAVLT']:
        with open(join(seq_home, 'VTUAV-ST.txt' if dataset_name == 'VTUAVST' else 'VTUAV-LT.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    else:
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    
    # 快速采样
    total_videos = len(seq_list)
    if args.video != '':
        seq_list = [args.video]
        print(f"Testing single video: {args.video}")
    elif args.sample_ratio < 1.0 or args.max_videos is not None:
        random.seed(args.sample_seed)
        sample_size = int(total_videos * args.sample_ratio)
        if args.max_videos is not None:
            sample_size = min(sample_size, args.max_videos)
        seq_list = random.sample(seq_list, sample_size)
        print(f"Quick test mode: sampling {sample_size}/{total_videos} videos ({args.sample_ratio*100:.1f}%)")
    
    print(f"Total videos to test: {len(seq_list)}")
    print(f"Videos: {seq_list[:10]}{'...' if len(seq_list) > 10 else ''}")
    
    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    
    print(f"Total time: {time.time()-start:.2f} seconds!")
    print(f"Average time per video: {(time.time()-start)/len(seq_list):.2f} seconds")
