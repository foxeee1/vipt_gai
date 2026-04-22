#!/bin/bash

# ============================================================
# 一致性+时序协同策略对比实验训练脚本
# ============================================================
# 实验分组：
#   策略A: independent - 独立协同（基线）
#   策略B: temporal_modulate - 时序调制协同
#   策略C: bidirectional - 双向交互协同
#   策略D: gating - 门控融合协同
#   策略E: joint_regularize - 联合正则化协同
#   扩展1: fomaml - FOMAML元学习模式
#   扩展2: three_prompts - 三提示协同
# ============================================================
# 注入层数（最优配置）：
#   - Consistency: [5, 6, 7, 8, 9]
#   - Temporal: [5, 6, 7, 8]
# ============================================================

SCRIPT_DIR="./tracking"
CONFIG_DIR="./experiments/vipt"
NUM_GPU=1
MODE="single"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  一致性+时序协同策略对比实验 (v15)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${BLUE}注入层配置:${NC}"
echo -e "  Consistency: [5, 6, 7, 8, 9]"
echo -e "  Temporal: [5, 6, 7, 8]"
echo ""

# ============================================================
# 协同策略对比实验
# ============================================================
declare -a COOP_CONFIGS=(
    "exp14_coop_independent"
    "exp14_coop_temporal_modulate"
    "exp14_coop_bidirectional"
    "exp14_coop_gating"
    "exp14_coop_joint_regularize"
    "exp14_coop_fomaml"
    "exp14_coop_three_prompts"
)

declare -a COOP_NAMES=(
    "策略A: 独立协同（基线）"
    "策略B: 时序调制协同"
    "策略C: 双向交互协同"
    "策略D: 门控融合协同"
    "策略E: 联合正则化协同"
    "扩展1: FOMAML元学习模式"
    "扩展2: 三提示协同"
)

for i in "${!COOP_CONFIGS[@]}"; do
    CONFIG="${COOP_CONFIGS[$i]}"
    NAME="${COOP_NAMES[$i]}"
    
    echo -e "${GREEN}[协同实验 $((i+1))/${#COOP_CONFIGS[@]}] ${NAME}${NC}"
    echo "配置文件: ${CONFIG}"
    echo "--------------------------------------------"
    
    python ${SCRIPT_DIR}/train.py \
        --script vipt \
        --config ${CONFIG} \
        --mode ${MODE} \
        --nproc_per_node ${NUM_GPU} \
        --save_dir ./output \
        --use_wandb 0
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[完成] ${NAME} 训练成功!${NC}"
    else
        echo -e "${YELLOW}[失败] ${NAME} 训练失败!${NC}"
    fi
    
    echo ""
    echo "============================================"
    echo ""
done

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  所有协同策略实验训练完成!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${BLUE}TensorBoard查看命令:${NC}"
echo "tensorboard --logdir_spec=\\"
echo "independent:./output/exp14_coop_independent,\\"
echo "temporal_modulate:./output/exp14_coop_temporal_modulate,\\"
echo "bidirectional:./output/exp14_coop_bidirectional,\\"
echo "gating:./output/exp14_coop_gating,\\"
echo "joint_regularize:./output/exp14_coop_joint_regularize,\\"
echo "fomaml:./output/exp14_coop_fomaml,\\"
echo "three_prompts:./output/exp14_coop_three_prompts \\"
echo "--port=6006"
