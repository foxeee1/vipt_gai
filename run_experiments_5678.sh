#!/bin/bash
# ============================================================
# Consistency Prompt 注入层数实验 - 串行训练脚本
# ============================================================
# 【实验组】四层[5,6,7,8]、五层[5,6,7,8,9]
# 【训练顺序】5678 → 56789
# 【训练方式】标准训练(无FOMAML)
# ============================================================

set -e

echo "=========================================="
echo "Consistency Prompt 注入层数实验 (5678系列)"
echo "=========================================="

# 实验配置列表
EXPERIMENTS=(
    "exp6_layers5_6_7_8"
    "exp6_layers5_6_7_8_9"
)

# 实验描述
declare -A EXP_DESC
EXP_DESC["exp6_layers5_6_7_8"]="四层注入第5,6,7,8层"
EXP_DESC["exp6_layers5_6_7_8_9"]="五层注入第5,6,7,8,9层"

# 显存估算
declare -A EXP_MEM
EXP_MEM["exp6_layers5_6_7_8"]="~20GB (BS=64)"
EXP_MEM["exp6_layers5_6_7_8_9"]="~22GB (BS=48)"

# 训练函数
train_experiment() {
    local config=$1
    local desc=${EXP_DESC[$config]}
    local mem=${EXP_MEM[$config]}

    echo ""
    echo "=========================================="
    echo "开始训练: $config"
    echo "描述: $desc"
    echo "预估显存: $mem"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    python tracking/train.py --script vipt --config $config --mode single

    echo ""
    echo "=========================================="
    echo "完成训练: $config"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
}

# 主训练循环
total=${#EXPERIMENTS[@]}
current=0

for exp in "${EXPERIMENTS[@]}"; do
    current=$((current + 1))
    echo ""
    echo "进度: [$current/$total]"

    train_experiment $exp

    # 实验间隔
    if [ $current -lt $total ]; then
        echo ""
        echo "等待10秒后开始下一个实验..."
        sleep 10
    fi
done

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "实验结果目录:"
echo "  - checkpoints: /home/apulis-dev/code/VIPT_gai/output/checkpoints/"
echo "  - tensorboard: /home/apulis-dev/code/VIPT_gai/tensorboard/train/vipt/"
echo ""
