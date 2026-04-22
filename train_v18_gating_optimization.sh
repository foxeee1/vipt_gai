#!/bin/bash

# ============================================================
# v18 门控融合深度优化实验
# ============================================================
# 实验分组（2个配置）：
#   1. gating - 原始门控(无偏初始化优化版)
#   2. gating_v2 - 特征调制门控(核心优化)
# ============================================================
# 注入层数（最优配置）：
#   - Base: [0]
#   - Consistency: [5, 6, 7, 8, 9]
#   - Temporal: [5, 6, 7, 8]
# ============================================================
# 【v18优化】
#   - 优化1: 无偏初始化 torch.zeros(1), sigmoid(0)=0.5
#   - 优化2: 特征调制门控，门控直接调制rgb/tir特征融合
#   - 不加入门控正则化，让模型自由学习极端门控
# ============================================================
# 基线对比: 门控融合(旧版) val IoU = 74.15
# ============================================================

SCRIPT_DIR="./tracking"
CONFIG_DIR="./experiments/vipt"
NUM_GPU=1
MODE="single"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  v18 门控融合深度优化实验${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${BLUE}注入层配置:${NC}"
echo -e "  Consistency: [5, 6, 7, 8, 9]"
echo -e "  Temporal: [5, 6, 7, 8]"
echo ""
echo -e "${YELLOW}[v18优化] 无偏初始化 + 特征调制门控${NC}"
echo -e "${YELLOW}[基线] 门控融合(旧版) val IoU = 74.15${NC}"
echo ""

# ============================================================
# 实验配置列表
# ============================================================
declare -a COOP_CONFIGS=(
    "exp14_coop_gating"
    "exp15_gating_v2"
)

declare -a COOP_NAMES=(
    "基线: 门控融合(无偏初始化优化版)"
    "优化: 特征调制门控(gating_v2)"
)

declare -a COOP_DESCS=(
    "coop_gate=torch.zeros(1), 无偏初始化, Prompt乘门控"
    "特征调制门控: 门控直接调制rgb/tir特征融合 + 投影调制信号融入Prompt"
)

# ============================================================
# 开始训练循环
# ============================================================
TOTAL=${#COOP_CONFIGS[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!COOP_CONFIGS[@]}"; do
    CONFIG="${COOP_CONFIGS[$i]}"
    NAME="${COOP_NAMES[$i]}"
    DESC="${COOP_DESCS[$i]}"
    NUM=$((i + 1))
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}[实验 ${NUM}/${TOTAL}] ${NAME}${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo "配置文件: ${CONFIG}.yaml"
    echo "说明: ${DESC}"
    echo "--------------------------------------------"
    echo ""
    
    # 执行训练命令
    python ${SCRIPT_DIR}/train.py \
        --script vipt \
        --config ${CONFIG} \
        --mode ${MODE} \
        --nproc_per_node ${NUM_GPU} \
        --save_dir ./output \
        --use_wandb 0
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[✅ 成功] ${NAME} 训练完成!${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}[❌ 失败] ${NAME} 训练失败!${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    # 清理GPU显存等待
    echo ""
    sleep 5
done

# ============================================================
# 训练完成总结
# ============================================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  🎉 所有实验训练完成!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${GREEN}成功: ${SUCCESS_COUNT}/${TOTAL}${NC}"
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}失败: ${FAIL_COUNT}/${TOTAL}${NC}"
fi
echo ""

# ============================================================
# TensorBoard查看命令
# ============================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  📊 TensorBoard查看命令${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "tensorboard --logdir_spec=\\"
echo "gating_baseline:/home/apulis-dev/code/VIPT_gai/tensorboard/train/vipt/exp14_coop_gating,\\"
echo "gating_v2:/home/apulis-dev/code/VIPT_gai/tensorboard/train/vipt/exp15_gating_v2 \\"
echo "--port=6006 --bind_all"
echo ""

# ============================================================
# 关键监控指标提醒
# ============================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  🔍 关键监控指标${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "【基线 - 门控融合(无偏初始化)】:"
echo "  ✅ coop_gate_value → sigmoid后的门控值，应在训练中变化"
echo "  ✅ 最终IoU → 目标超过74.15"
echo ""
echo "【优化 - 特征调制门控】:"
echo "  ✅ coop_gate_value → 门控值，应自由学习"
echo "  ✅ coop_gate_entropy → 门控熵，越高表示越均衡"
echo "  ✅ 最终IoU → 目标超过74.15"
echo ""
echo "【关键对比】:"
echo "  📈 gating_v2 vs gating → 特征调制是否优于Prompt乘门控"
echo "  📈 无偏初始化 vs 有偏初始化 → 初始化方式的影响"
echo ""
