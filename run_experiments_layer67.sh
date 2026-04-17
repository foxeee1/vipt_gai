#!/bin/bash
# ============================================================
# Consistency Prompt 注入层数实验 - 串行训练脚本
# ============================================================
# 【实验组】单层[6]、单层[7]、三层[5,6,7]
# 【训练顺序】layer6 → layer7 → layers5_6_7
# 【训练方式】标准训练(无FOMAML)
# ============================================================

set -e

echo "=========================================="
echo "Consistency Prompt 注入层数实验"
echo "=========================================="

# 实验配置列表
EXPERIMENTS=(
    "exp6_layer6"
    "exp6_layer7"
    "exp6_layers5_6_7"
)

# 实验描述
declare -A EXP_DESC
EXP_DESC["exp6_layer6"]="单层注入第6层"
EXP_DESC["exp6_layer7"]="单层注入第7层"
EXP_DESC["exp6_layers5_6_7"]="三层注入第5,6,7层"

# 训练函数
train_experiment() {
    local config=$1
    local desc=${EXP_DESC[$config]}
    
    echo ""
    echo "=========================================="
    echo "开始训练: $config"
    echo "描述: $desc"
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
echo "  - checkpoints: /home/apulis-dev/code/VIPT_gai/checkpoints/train/vipt/"
echo "  - tensorboard: /home/apulis-dev/code/VIPT_gai/tensorboard/train/vipt/"
echo ""
