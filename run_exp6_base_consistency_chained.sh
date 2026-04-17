#!/bin/bash
# ============================================================
# 串行训练脚本: Base + Consistency Prompt 实验组
# ============================================================
# 实验顺序:
#   6a1(gradient, layer9) -> 6a2(gradient, layer10)
#   6a3(gradient, layer11) -> 6a4(gradient, layer12)
#   6b1(covariance, layer9) -> 6b2(covariance, layer10)
#   6b3(covariance, layer11) -> 6b4(covariance, layer12)
#   6c1(gradient, layers9_10) -> 6c2(covariance, layers9_10)
#   6c3(gradient, layers10_11) -> 6c4(covariance, layers10_11)
#   6c5(gradient, layers11_12) -> 6c6(covariance, layers11_12)
# ============================================================

set -e

echo "============================================================"
echo "开始 Base + Consistency Prompt 实验"
echo "顺序: 6a(gradient单层) -> 6b(covariance单层) -> 6c(双层对照)"
echo "============================================================"
echo ""

# ===== Gradient 版本单层注入 =====
echo "[1/16] 开始训练 exp6a1_base_consistency_grad_layer9 ..."
python tracking/train.py --script vipt --config exp6a1_base_consistency_grad_layer9 --mode single
echo "[1/16] exp6a1 训练完成!"
echo ""

echo "[2/16] 开始训练 exp6a2_base_consistency_grad_layer10 ..."
python tracking/train.py --script vipt --config exp6a2_base_consistency_grad_layer10 --mode single
echo "[2/16] exp6a2 训练完成!"
echo ""

echo "[3/16] 开始训练 exp6a3_base_consistency_grad_layer11 ..."
python tracking/train.py --script vipt --config exp6a3_base_consistency_grad_layer11 --mode single
echo "[3/16] exp6a3 训练完成!"
echo ""

echo "[4/16] 开始训练 exp6a4_base_consistency_grad_layer12 ..."
python tracking/train.py --script vipt --config exp6a4_base_consistency_grad_layer12 --mode single
echo "[4/16] exp6a4 训练完成!"
echo ""

# ===== Covariance 版本单层注入 =====
echo "[5/16] 开始训练 exp6b1_base_consistency_cov_layer9 ..."
python tracking/train.py --script vipt --config exp6b1_base_consistency_cov_layer9 --mode single
echo "[5/16] exp6b1 训练完成!"
echo ""

echo "[6/16] 开始训练 exp6b2_base_consistency_cov_layer10 ..."
python tracking/train.py --script vipt --config exp6b2_base_consistency_cov_layer10 --mode single
echo "[6/16] exp6b2 训练完成!"
echo ""

echo "[7/16] 开始训练 exp6b3_base_consistency_cov_layer11 ..."
python tracking/train.py --script vipt --config exp6b3_base_consistency_cov_layer11 --mode single
echo "[7/16] exp6b3 训练完成!"
echo ""

echo "[8/16] 开始训练 exp6b4_base_consistency_cov_layer12 ..."
python tracking/train.py --script vipt --config exp6b4_base_consistency_cov_layer12 --mode single
echo "[8/16] exp6b4 训练完成!"
echo ""

# ===== 双层注入 [9,10] =====
echo "[9/16] 开始训练 exp6c1_base_consistency_grad_layers9_10 ..."
python tracking/train.py --script vipt --config exp6c1_base_consistency_grad_layers9_10 --mode single
echo "[9/16] exp6c1 训练完成!"
echo ""

echo "[10/16] 开始训练 exp6c2_base_consistency_cov_layers9_10 ..."
python tracking/train.py --script vipt --config exp6c2_base_consistency_cov_layers9_10 --mode single
echo "[10/16] exp6c2 训练完成!"
echo ""

# ===== 双层注入 [10,11] =====
echo "[11/16] 开始训练 exp6c3_base_consistency_grad_layers10_11 ..."
python tracking/train.py --script vipt --config exp6c3_base_consistency_grad_layers10_11 --mode single
echo "[11/16] exp6c3 训练完成!"
echo ""

echo "[12/16] 开始训练 exp6c4_base_consistency_cov_layers10_11 ..."
python tracking/train.py --script vipt --config exp6c4_base_consistency_cov_layers10_11 --mode single
echo "[12/16] exp6c4 训练完成!"
echo ""

# ===== 双层注入 [11,12] =====
echo "[13/16] 开始训练 exp6c5_base_consistency_grad_layers11_12 ..."
python tracking/train.py --script vipt --config exp6c5_base_consistency_grad_layers11_12 --mode single
echo "[13/16] exp6c5 训练完成!"
echo ""

echo "[14/16] 开始训练 exp6c6_base_consistency_cov_layers11_12 ..."
python tracking/train.py --script vipt --config exp6c6_base_consistency_cov_layers11_12 --mode single
echo "[14/16] exp6c6 训练完成!"
echo ""

echo "============================================================"
echo "所有 Base + Consistency Prompt 实验训练完成!"
echo "============================================================"