#!/bin/bash
# ============================================================
# ViPT 串行训练脚本 - 一致性模块多高层注入实验
# ============================================================
# 【功能】按顺序执行多层注入实验，训练完成后自动开始下一个
# 【实验列表】
#   三层注入: layers6_7_8, layers7_8_9, layers8_9_10
#   四层注入: layers6_7_8_9, layers7_8_9_10
# ============================================================

# 实验配置列表（按顺序执行）
EXPERIMENTS=(
    # 三层注入
    "exp6_layers6_7_8"
    "exp6_layers7_8_9"
    "exp6_layers8_9_10"
    # 四层注入
    "exp6_layers6_7_8_9"
    "exp6_layers7_8_9_10"
)

# 训练模式
MODE="single"

# 日志目录
LOG_DIR="./output/logs"
TENSORBOARD_DIR="/home/apulis-dev/code/VIPT_gai/tensorboard"

# 创建日志目录
mkdir -p ${LOG_DIR}

echo "============================================"
echo "ViPT 一致性模块多高层注入实验"
echo "============================================"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "训练模式: ${MODE}"
echo "日志目录: ${LOG_DIR}"
echo "TensorBoard: ${TENSORBOARD_DIR}"
echo "============================================"
echo ""

# 串行执行每个实验
for i in "${!EXPERIMENTS[@]}"; do
    EXP_NAME="${EXPERIMENTS[$i]}"
    EXP_NUM=$((i + 1))
    TOTAL=${#EXPERIMENTS[@]}

    echo "============================================"
    echo "[${EXP_NUM}/${TOTAL}] 开始训练: ${EXP_NAME}"
    echo "============================================"

    # 记录开始时间
    START_TIME=$(date +%s)

    # 执行训练
    python tracking/train.py \
        --script vipt \
        --config ${EXP_NAME} \
        --mode ${MODE} \
        2>&1 | tee ${LOG_DIR}/${EXP_NAME}.log

    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_H=$((DURATION / 3600))
    DURATION_M=$(((DURATION % 3600) / 60))

    echo ""
    echo "============================================"
    echo "[${EXP_NUM}/${TOTAL}] 完成训练: ${EXP_NAME}"
    echo "耗时: ${DURATION_H}h ${DURATION_M}m"
    echo "日志: ${LOG_DIR}/${EXP_NAME}.log"
    echo "============================================"
    echo ""

    # 短暂休息（可选，防止资源争用）
    sleep 5
done

echo "============================================"
echo "所有实验训练完成!"
echo "============================================"
echo ""
echo "查看训练日志:"
echo "  ls -la ${LOG_DIR}"
echo ""
echo "启动 TensorBoard:"
echo "  tensorboard --logdir=${TENSORBOARD_DIR} --port=6006 --bind_all"
echo "============================================"
