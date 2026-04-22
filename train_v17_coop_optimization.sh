#!/bin/bash

# ============================================================
# v17协同策略优化实验训练脚本
# ============================================================
# 实验分组（3个新配置）：
#   1. temporal_modulate_anneal - 时序调制+动态温度退火
#   2. gating_modulate_hybrid - 门控+调制混合策略
#   3. gating_token_level - Token级门控+对比学习损失
# ============================================================
# 注入层数（最优配置）：
#   - Base: [0]
#   - Consistency: [5, 6, 7, 8, 9]
#   - Temporal: [5, 6, 7, 8]
# ============================================================
# 【v17改进】
#   - 方向1: 动态温度退火，训练初期弱调制后期强调制
#   - 方向2: 门控自适应学习+时序调制物理先验混合
#   - 优化1: Token级精细门控
#   - 优化2: 对比学习损失增强门控区分度
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
echo -e "${BLUE}  v17 协同策略优化实验${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${BLUE}注入层配置:${NC}"
echo -e "  Consistency: [5, 6, 7, 8, 9]"
echo -e "  Temporal: [5, 6, 7, 8]"
echo ""
echo -e "${YELLOW}[v17改进] 3个新策略快速验证${NC}"
echo ""

# ============================================================
# 实验配置列表
# ============================================================
declare -a COOP_CONFIGS=(
    "exp15_temporal_modulate_anneal"
    "exp15_gating_modulate_hybrid"
    "exp15_gating_token_level"
)

declare -a COOP_NAMES=(
    "策略A: 时序调制+动态温度退火 (temporal_modulate_anneal)"
    "策略B: 门控+调制混合 (gating_modulate_hybrid)"
    "策略C: Token级门控+对比学习 (gating_token_level)"
)

declare -a COOP_DESCS=(
    "温度退火: 0.02→0.12, 前2000步弱调制后期强调制"
    "门控MLP自适应+时序强度微调(系数0.05)"
    "Token级精细门控+对比学习损失(权重0.05)"
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
echo "anneal:./output/exp15_temporal_modulate_anneal,\\"
echo "hybrid:./output/exp15_gating_modulate_hybrid,\\"
echo "token_level:./output/exp15_gating_token_level \\"
echo "--port=6006"
echo ""

# ============================================================
# 关键监控指标提醒
# ============================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  🔍 关键监控指标${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "【策略A - 动态温度退火】:"
echo "  ✅ coop_anneal_step → 应逐步增长"
echo "  ✅ coop_anneal_coeff → 应从0.02升到0.12"
echo "  ✅ coop_modulate_factor → 应在0.90-1.05范围"
echo ""
echo "【策略B - 门控+调制混合】:"
echo "  ✅ coop_gate → 应在[0.3, 0.7]范围"
echo "  ✅ coop_temporal_strength_raw → 应与门控调整一致"
echo ""
echo "【策略C - Token级门控】:"
echo "  ✅ coop_gate_mean → 平均门控分布"
echo "  ✅ coop_gate_std → Token间门控差异"
echo "  ✅ Loss/contrast → 对比学习损失"
echo ""
echo "【性能对比】:"
echo "  📈 最终IoU → 越高越好"
echo "  📉 收敛速度 → 越快越好"
echo "  🎯 门控稳定性 → 波动适中最佳"
echo ""
