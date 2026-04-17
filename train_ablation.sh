#!/bin/bash

# ============================================================
# 消融实验训练脚本
# ============================================================
# 实验分组：
#   Exp1: Baseline标准训练 - 全层Base Prompt，无Meta Prompt
#   Exp2: Meta-Learning基线 - 全层Base Prompt，无一致性/掩码模块
#   Exp3: 梯度一致性Prompt - Base Prompt + 梯度一致性Meta Prompt
#   Exp4: 融合一致性Prompt - Base Prompt + 融合一致性Meta Prompt
#   Exp5: 完整四层提示 - Base + Mask + Fusion + Temporal
# ============================================================

# 训练脚本目录
SCRIPT_DIR="./tracking"
CONFIG_DIR="./experiments/vipt"
EXP_PREFIX="exp"

# GPU配置
NUM_GPU=1
MODE="single"  # single 或 multiple

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  ViPT 消融实验训练脚本${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 循环训练所有实验组
for i in 1 2 3 4 5; do
    case $i in
        1) CONFIG="${EXP_PREFIX}1_baseline_standard" ;;
        2) CONFIG="${EXP_PREFIX}2_baseline_metalearn" ;;
        3) CONFIG="${EXP_PREFIX}3_meta_gradient" ;;
        4) CONFIG="${EXP_PREFIX}4_meta_fusion" ;;
        5) CONFIG="${EXP_PREFIX}5_meta_full" ;;
    esac

    echo -e "${GREEN}[Exp${i}] 开始训练: ${CONFIG}${NC}"
    echo "--------------------------------------------"

    python ${SCRIPT_DIR}/train.py \
        --script vipt \
        --config ${CONFIG} \
        --mode ${MODE} \
        --nproc_per_node ${NUM_GPU} \
        --save_dir ./output \
        --use_wandb 0

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[Exp${i}] ${CONFIG} 训练完成!${NC}"
    else
        echo -e "${YELLOW}[Exp${i}] ${CONFIG} 训练失败!${NC}"
    fi

    echo ""
    echo "============================================"
    echo ""
done

echo -e "${BLUE}所有实验训练完成!${NC}"
