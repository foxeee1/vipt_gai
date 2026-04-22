#!/bin/bash

# ============================================================
# 时序提示多层注入实验训练脚本
# ============================================================
# 实验分组：
#   三层注入: [5,6,7], [6,7,8], [8,9,10]
#   四层注入: [5,6,7,8], [6,7,8,9], [7,8,9,10]
#   五层注入: [5,6,7,8,9], [6,7,8,9,10]
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
echo -e "${BLUE}  时序提示多层注入实验 (v10.2)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# ============================================================
# 三层注入实验
# ============================================================
echo -e "${GREEN}[Phase 1] 三层注入实验${NC}"
echo "--------------------------------------------"

declare -a THREE_LAYER_CONFIGS=(
    "exp11_temporal_layers5_6_7"
    "exp11_temporal_layers6_7_8"
    "exp11_temporal_layers8_9_10"
)

for CONFIG in "${THREE_LAYER_CONFIGS[@]}"; do
    echo -e "${GREEN}[三层] 开始训练: ${CONFIG}${NC}"
    python ${SCRIPT_DIR}/train.py \
        --script vipt \
        --config ${CONFIG} \
        --mode ${MODE} \
        --nproc_per_node ${NUM_GPU} \
        --save_dir ./output \
        --use_wandb 0
    echo "--------------------------------------------"
done

# ============================================================
# 四层注入实验
# ============================================================
echo -e "${GREEN}[Phase 2] 四层注入实验${NC}"
echo "--------------------------------------------"

declare -a FOUR_LAYER_CONFIGS=(
    "exp12_temporal_layers5_6_7_8"
    "exp12_temporal_layers6_7_8_9"
    "exp12_temporal_layers7_8_9_10"
)

for CONFIG in "${FOUR_LAYER_CONFIGS[@]}"; do
    echo -e "${GREEN}[四层] 开始训练: ${CONFIG}${NC}"
    python ${SCRIPT_DIR}/train.py \
        --script vipt \
        --config ${CONFIG} \
        --mode ${MODE} \
        --nproc_per_node ${NUM_GPU} \
        --save_dir ./output \
        --use_wandb 0
    echo "--------------------------------------------"
done

# ============================================================
# 五层注入实验
# ============================================================
echo -e "${GREEN}[Phase 3] 五层注入实验${NC}"
echo "--------------------------------------------"

declare -a FIVE_LAYER_CONFIGS=(
    "exp13_temporal_layers5_6_7_8_9"
    "exp13_temporal_layers6_7_8_9_10"
)

for CONFIG in "${FIVE_LAYER_CONFIGS[@]}"; do
    echo -e "${GREEN}[五层] 开始训练: ${CONFIG}${NC}"
    python ${SCRIPT_DIR}/train.py \
        --script vipt \
        --config ${CONFIG} \
        --mode ${MODE} \
        --nproc_per_node ${NUM_GPU} \
        --save_dir ./output \
        --use_wandb 0
    echo "--------------------------------------------"
done

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  所有实验训练完成!${NC}"
echo -e "${BLUE}============================================${NC}"
