#!/bin/bash

# ============================================================
# v20 分阶段训练脚本 - 解决双提示"正则化集体失效"问题
# ============================================================
# 【训练策略】
#   阶段1: 固定Consistency，只训练Temporal (20 epoch)
#   阶段2: 固定Temporal，只训练Consistency (20 epoch)
#   阶段3: 放开所有参数，联合微调 (10 epoch, 学习率1/10)
# ============================================================
# 【核心修复】
#   1. 降低正则化强度(0.6→0.3)，让模型有足够区分度
#   2. 使用STAGE配置控制冻结，避免冻结整个模块
#   3. 简化正则化检查逻辑，确保正则化生效
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
echo -e "${BLUE}  v20 分阶段训练 - 解决双提示正则化失效${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${YELLOW}[核心修复]${NC}"
echo "  1. 降低正则化强度(0.6→0.3)，让模型有足够区分度"
echo "  2. 使用STAGE配置控制冻结，避免冻结整个模块"
echo "  3. 简化正则化检查逻辑，确保正则化生效"
echo ""
echo -e "${BLUE}[训练策略]${NC}"
echo "  阶段1: 固定Consistency，只训练Temporal (20 epoch)"
echo "  阶段2: 固定Temporal，只训练Consistency (20 epoch)"
echo "  阶段3: 放开所有参数，联合微调 (10 epoch, 学习率1/10)"
echo ""

# ============================================================
# 阶段1: 固定Consistency，只训练Temporal
# ============================================================
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}[阶段1/3] 固定Consistency，只训练Temporal${NC}"
echo -e "${GREEN}============================================${NC}"
echo "配置文件: exp19_stage1_temporal.yaml"
echo "训练epoch: 20"
echo ""

python ${SCRIPT_DIR}/train.py \
    --script vipt \
    --config exp19_stage1_temporal \
    --mode ${MODE} \
    --nproc_per_node ${NUM_GPU} \
    --save_dir ./output \
    --use_wandb 0

if [ $? -ne 0 ]; then
    echo -e "${RED}[❌ 失败] 阶段1训练失败!${NC}"
    exit 1
fi
echo -e "${GREEN}[✅ 成功] 阶段1训练完成!${NC}"
echo ""
sleep 5

# ============================================================
# 阶段2: 固定Temporal，只训练Consistency
# ============================================================
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}[阶段2/3] 固定Temporal，只训练Consistency${NC}"
echo -e "${GREEN}============================================${NC}"
echo "配置文件: exp19_stage2_consistency.yaml"
echo "训练epoch: 20"
echo ""

python ${SCRIPT_DIR}/train.py \
    --script vipt \
    --config exp19_stage2_consistency \
    --mode ${MODE} \
    --nproc_per_node ${NUM_GPU} \
    --save_dir ./output \
    --use_wandb 0

if [ $? -ne 0 ]; then
    echo -e "${RED}[❌ 失败] 阶段2训练失败!${NC}"
    exit 1
fi
echo -e "${GREEN}[✅ 成功] 阶段2训练完成!${NC}"
echo ""
sleep 5

# ============================================================
# 阶段3: 联合微调
# ============================================================
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}[阶段3/3] 联合微调（小学习率）${NC}"
echo -e "${GREEN}============================================${NC}"
echo "配置文件: exp19_stage3_finetune.yaml"
echo "训练epoch: 10"
echo "学习率: 0.00006 (原来的1/10)"
echo ""

python ${SCRIPT_DIR}/train.py \
    --script vipt \
    --config exp19_stage3_finetune \
    --mode ${MODE} \
    --nproc_per_node ${NUM_GPU} \
    --save_dir ./output \
    --use_wandb 0

if [ $? -ne 0 ]; then
    echo -e "${RED}[❌ 失败] 阶段3训练失败!${NC}"
    exit 1
fi
echo -e "${GREEN}[✅ 成功] 阶段3训练完成!${NC}"
echo ""

# ============================================================
# 训练完成总结
# ============================================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  🎉 所有阶段训练完成!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# ============================================================
# TensorBoard查看命令
# ============================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  📊 TensorBoard查看命令${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "tensorboard --logdir_spec=\\"
echo "stage1:/home/apulis-dev/code/VIPT_gai/tensorboard/train/vipt/exp19_stage1_temporal,\\"
echo "stage2:/home/apulis-dev/code/VIPT_gai/tensorboard/train/vipt/exp19_stage2_consistency,\\"
echo "stage3:/home/apulis-dev/code/VIPT_gai/tensorboard/train/vipt/exp19_stage3_finetune \\"
echo "--port=6006 --bind_all"
echo ""

# ============================================================
# 关键监控指标提醒
# ============================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  🔍 关键监控指标${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "【正则化是否生效】:"
echo "  ✅ Loss/consistency_reg → 应该>0，不再是0"
echo "  ✅ Loss/temporal_reg → 应该>0，不再是0"
echo ""
echo "【RGB偏向是否修复】:"
echo "  ✅ avg_rgb_weight → 应该从0.9降到0.5-0.6"
echo "  ✅ avg_tir_weight → 应该从0.1升到0.4-0.5"
echo ""
echo "【模型区分度】:"
echo "  ✅ 权重直方图 → 应该有双峰，不是所有token权重都一样"
echo "  ✅ 熵正则化 → 70%权重，确保有区分度"
echo ""
echo "【性能目标】:"
echo "  📈 最终IoU → 目标超过74.15（原门控融合）"
echo "  📈 分阶段训练 → 避免梯度干扰，稳定收敛"
echo ""
