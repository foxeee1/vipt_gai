#!/bin/bash

# 云端小批次训练脚本
# 使用lasher数据集进行元提示模型训练

    --config lasher_meta_small \
python tracking/train.py \
    --save_dir ./output \
    --use_wandb 0

# 多GPU训练（如果需要）
# python tracking/train.py \
#     --script vipt \
#     --config lasher_meta_small \
#     --mode multiple \
#     --save_dir ./output \
#     --use_wandb 0