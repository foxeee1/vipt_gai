#!/bin/bash
# ============================================================
# v5 Temporal Prompt 注入层数实验 - 串行训练脚本
# ============================================================
# 【实验目的】找到时序提示单独工作时的最优注入层
# 【v5创新】跨模态-时序联合一致性建模（复用Consistency权重）
# 【训练顺序】单层5→6→7→8→9 → 双层67→78→89 → 三层678(可选)
# ============================================================

set -e

echo "=========================================="
echo "v5 Temporal Prompt 注入层数实验"
echo "=========================================="

# 实验配置列表（按优先级排序）
EXPERIMENTS=(
    # 单层注入（最高优先级）
    "exp7_temporal_layer5"
    "exp7_temporal_layer6"
    "exp7_temporal_layer7"
    "exp7_temporal_layer8"
    "exp7_temporal_layer9"
    # 双层注入（高优先级）
    "exp7_temporal_layers6_7"
    "exp7_temporal_layers7_8"
    "exp7_temporal_layers8_9"
    # 三层注入（可选）
    "exp7_temporal_layers6_7_8"
)

# 实验描述
declare -A EXP_DESC
EXP_DESC["exp7_temporal_layer5"]="单层注入第5层（低层）"
EXP_DESC["exp7_temporal_layer6"]="单层注入第6层（中层）"
EXP_DESC["exp7_temporal_layer7"]="单层注入第7层（预期最优单层）"
EXP_DESC["exp7_temporal_layer8"]="单层注入第8层（高层）"
EXP_DESC["exp7_temporal_layer9"]="单层注入第9层（超高层）"
EXP_DESC["exp7_temporal_layers6_7"]="双层注入第6,7层"
EXP_DESC["exp7_temporal_layers7_8"]="双层注入第7,8层（预期最优双层）"
EXP_DESC["exp7_temporal_layers8_9"]="双层注入第8,9层"
EXP_DESC["exp7_temporal_layers6_7_8"]="三层注入第6,7,8层（可选）"

# 显存估算
declare -A EXP_MEM
EXP_MEM["exp7_temporal_layer5"]="~16GB (BS=96)"
EXP_MEM["exp7_temporal_layer6"]="~16GB (BS=96)"
EXP_MEM["exp7_temporal_layer7"]="~16GB (BS=96)"
EXP_MEM["exp7_temporal_layer8"]="~16GB (BS=96)"
EXP_MEM["exp7_temporal_layer9"]="~16GB (BS=96)"
EXP_MEM["exp7_temporal_layers6_7"]="~18GB (BS=96)"
EXP_MEM["exp7_temporal_layers7_8"]="~18GB (BS=96)"
EXP_MEM["exp7_temporal_layers8_9"]="~18GB (BS=96)"
EXP_MEM["exp7_temporal_layers6_7_8"]="~20GB (BS=64)"

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
