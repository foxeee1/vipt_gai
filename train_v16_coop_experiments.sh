#!/bin/bash

# ============================================================
# v16协同策略对比实验训练脚本
# ============================================================
# 实验分组（6个配置）：
#   1. independent - 独立协同（基线）
#   2. temporal_modulate - 时序调制协同（v16修复版）
#   3. bidirectional - 双向交互协同（v16修复版）
#   4. gating - 门控融合协同
#   5. joint_regularize - 联合正则化协同（v16修复版）
#   6. fomaml - FOMAML元学习模式
# ============================================================
# 注入层数（最优配置）：
#   - Base: [0]
#   - Consistency: [5, 6, 7, 8, 9]
#   - Temporal: [5, 6, 7, 8]
# ============================================================
# 【v16修复】所有协同策略已添加保护机制：
#   - 降低调制系数（0.2→0.05-0.08）
#   - 添加范围限制防止过度压制
#   - 记录调制因子用于调试分析
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
echo -e "${BLUE}  v16 协同策略对比实验 (保护性调制)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${BLUE}注入层配置:${NC}"
echo -e "  Consistency: [5, 6, 7, 8, 9]"
echo -e "  Temporal: [5, 6, 7, 8]"
echo ""
echo -e "${YELLOW}[v16改进] 所有策略已添加保护性调制机制${NC}"
echo ""

# ============================================================
# 实验配置列表
# ============================================================
declare -a COOP_CONFIGS=(
    "exp14_coop_independent"
    "exp14_coop_temporal_modulate"
    "exp14_coop_bidirectional"
    "exp14_coop_gating"
    "exp14_coop_joint_regularize"
    "exp14_coop_fomaml"
)

declare -a COOP_NAMES=(
    "策略A: 独立协同 (independent)"
    "策略B: 时序调制 (temporal_modulate)"
    "策略C: 双向交互 (bidirectional)"
    "策略D: 门控融合 (gating)"
    "策略E: 联合正则化 (joint_regularize)"
    "策略F: FOMAML元学习 (fomaml)"
)

declare -a COOP_DESCS=(
    "无交互，简单相加，作为基线对比"
    "时序强度动态调整一致性(系数0.08)"
    "双向互惠调制(系数0.06)"
    "可学习门控自适应融合"
    "软约束+正则化记录(系数0.05)"
    "FOMAML元学习+task_emb动态生成"
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
    sleep 3
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
echo "independent:./output/exp14_coop_independent,\\"
echo "temporal_modulate:./output/exp14_coop_temporal_modulate,\\"
echo "bidirectional:./output/exp14_coop_bidirectional,\\"
echo "gating:./output/exp14_coop_gating,\\"
echo "joint_regularize:./output/exp14_coop_joint_regularize,\\"
echo "fomaml:./output/exp14_coop_fomaml \\"
echo "--port=6006"
echo ""

# ============================================================
# 关键监控指标提醒
# ============================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  🔍 关键监控指标${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "【必须观察的曲线】:"
echo "  ✅ Consistency/avg_rgb_weight → 应在0.6-0.8波动(非饱和)"
echo "  ✅ Consistency/avg_tir_weight → 应在0.2-0.4(非极低)"
echo "  ✅ Consistency/weight_dist → 应呈现双峰分布"
echo "  ✅ Loss/temporal_reg → 应>0(正则化生效)"
echo "  ✅ Train/weight_entropy → 应在1.0-1.3"
echo ""
echo "【性能对比】:"
echo "  📈 最终IoU → 越高越好"
echo "  📉 收敛速度 → 越快越好"
echo "  🎯 权重稳定性 → 波动适中最佳"
echo ""
