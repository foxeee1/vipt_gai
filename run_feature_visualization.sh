#!/bin/bash
# 一键特征可视化脚本
# 用法: bash run_feature_visualization.sh <checkpoint_path>

set -e

CHECKPOINT=${1:-"checkpoints/vipt_v25.pth"}
FEATURE_DIR="features"
OUTPUT_DIR="visualizations"

echo "=============================================="
echo "ViPT 特征可视化一键脚本"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "特征目录: $FEATURE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=============================================="

mkdir -p $FEATURE_DIR
mkdir -p $OUTPUT_DIR

echo ""
echo "[Step 1/4] 保存单提示(Temporal only)特征..."
python lib/utils/save_features.py \
    --mode temporal_only \
    --checkpoint $CHECKPOINT \
    --output_dir $FEATURE_DIR \
    --max_frames 30

echo ""
echo "[Step 2/4] 保存双提示(Consistency+Temporal)特征..."
python lib/utils/save_features.py \
    --mode double \
    --checkpoint $CHECKPOINT \
    --output_dir $FEATURE_DIR \
    --max_frames 30

echo ""
echo "[Step 3/4] 保存三提示(全量)特征..."
python lib/utils/save_features.py \
    --mode triple \
    --checkpoint $CHECKPOINT \
    --output_dir $FEATURE_DIR \
    --max_frames 30

echo ""
echo "[Step 4/4] 生成可视化分析..."
python lib/utils/visualize_features.py \
    --feature_dir $FEATURE_DIR \
    --output_dir $OUTPUT_DIR

echo ""
echo "=============================================="
echo "✓ 全部完成!"
echo "=============================================="
echo "可视化结果保存在: $OUTPUT_DIR"
echo ""
echo "生成的文件:"
ls -la $OUTPUT_DIR
echo "=============================================="
