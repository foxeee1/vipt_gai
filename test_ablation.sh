#!/bin/bash

# ============================================================
# 消融实验测试脚本
# ============================================================
# 测试所有实验组的模型性能
# 数据集：LasHeR (RGBT跟踪基准)
# ============================================================

# 测试脚本和配置
TEST_SCRIPT="./RGBT_workspace/test_rgbt_quick.py"
CONFIG_DIR="./experiments/vipt"
EXP_PREFIX="exp"

# 数据集配置
DATASET="LasHeR"
EPOCH=30

# GPU配置
NUM_GPU=1

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  ViPT 消融实验测试脚本${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 循环测试所有实验组
for i in 1 2 3 4 5; do
    case $i in
        1) CONFIG="${EXP_PREFIX}1_baseline_standard" ;;
        2) CONFIG="${EXP_PREFIX}2_baseline_metalearn" ;;
        3) CONFIG="${EXP_PREFIX}3_meta_gradient" ;;
        4) CONFIG="${EXP_PREFIX}4_meta_fusion" ;;
        5) CONFIG="${EXP_PREFIX}5_meta_full" ;;
    esac

    echo -e "${GREEN}[Exp${i}] 开始测试: ${CONFIG}${NC}"
    echo "--------------------------------------------"

    python ${TEST_SCRIPT} \
        --yaml_name ${CONFIG} \
        --dataset_name ${DATASET} \
        --num_gpu ${NUM_GPU} \
        --epoch ${EPOCH}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[Exp${i}] ${CONFIG} 测试完成!${NC}"
    else
        echo -e "${YELLOW}[Exp${i}] ${CONFIG} 测试失败!${NC}"
    fi

    echo ""
    echo "============================================"
    echo ""
done

echo -e "${BLUE}所有实验测试完成!${NC}"
