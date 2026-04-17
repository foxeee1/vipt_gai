# ViPT元提示框架训练指南

## 项目概述

本项目基于ViPT（Visual Prompt Multi-Modal Tracking）模型，实现了元提示框架，用于多模态目标跟踪。框架包含：

- **四层提示融合体系**：Base Prompt、Mask Prompt、Consistency Prompt、Temporal Prompt
- **元学习训练策略**：摊销式FOMAML，包含内外环优化
- **多模态支持**：RGB-D、RGB-T、RGB-E等模态

## 环境配置

### 1. 安装依赖

```bash
# 基础依赖
pip install torch torchvision numpy opencv-python pyyaml tqdm matplotlib scipy pandas pillow

# 可选依赖
pip install tensorboard wandb jpeg4py timm
```

### 2. 配置数据路径

编辑 `lib/train/admin/local.py` 文件，设置数据集路径：

```python
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/autodl-tmp/ViPT'
        self.tensorboard_dir = '/root/autodl-tmp/ViPT/tensorboard'
        self.lasher_dir = '/root/autodl-tmp/ViPT/data/lasher/trainingset'
        # ... 其他路径
```

### 3. 检查依赖

运行依赖检查脚本：

```bash
python check_dependencies.py
```

## 数据集准备

### LasHeR数据集

1. 下载LasHeR数据集：https://github.com/BUGPLEASEOUT/LasHeR
2. 解压到指定目录（如：`/root/autodl-tmp/ViPT/data/lasher/trainingset`）
3. 确保数据集结构如下：

```
lasher/
├── trainingset/
│   ├── sequence1/
│   │   ├── visible/
│   │   ├── thermal/
│   │   └── groundtruth.txt
│   ├── sequence2/
│   └── ...
└── testingset/
    └── ...
```

## 训练流程

### 1. 小批次训练（推荐用于测试）

使用lasher数据集进行小批次训练：

```bash
bash train_lasher_meta.sh
```

配置文件：`experiments/vipt/lasher_meta_small.yaml`

**关键参数**：
- BATCH_SIZE: 8（小批次）
- EPOCH: 20（快速迭代）
- SAMPLE_PER_EPOCH: 10000（减少样本数）
- META.ENABLE: True（启用元学习）

### 2. 完整训练

使用完整数据集进行训练：

```bash
bash train_vipt_meta.sh
```

配置文件：`experiments/vipt/deep_rgbd_meta.yaml`

### 3. 多GPU训练

```bash
python tracking/train.py \
    --script vipt \
    --config lasher_meta_small \
    --mode multiple \
    --nproc_per_node 4 \
    --save_dir ./output
```

## TensorBoard监控

### 1. 启动TensorBoard

```bash
tensorboard --logdir=/root/autodl-tmp/ViPT/tensorboard --port=6006
```

### 2. 访问TensorBoard

在浏览器中打开：`http://localhost:6006`

### 3. 云端访问

如果在云端服务器上训练，使用端口转发：

```bash
# 本地执行
ssh -L 6006:localhost:6006 user@remote_server
```

然后在本地浏览器访问：`http://localhost:6006`

## 训练监控指标

TensorBoard中会记录以下指标：

- **Loss/total**: 总损失
- **Loss/giou**: GIoU损失
- **Loss/l1**: L1损失
- **Loss/location**: 位置损失
- **IoU**: 预测框与真实框的IoU
- **LearningRate**: 学习率变化

## 测试流程

### 1. 准备测试数据

确保测试数据集已下载并配置路径。

### 2. 运行测试

```bash
python tracking/test.py --config lasher_meta_small
```

### 3. 评估结果

测试结果会保存在 `output/` 目录下。

## 元学习训练详解

### 参数更新逻辑

元学习采用**摊销式FOMAML**框架：

1. **内环更新**：
   ```
   θ' = θ - α_inner · ∇_θ L_inner(θ, D_train)
   ```
   - 在退化任务上单步梯度更新
   - 模拟模型快速适应能力

2. **外环更新**：
   ```
   θ ← θ - β_outer · ∇_θ L_outer(θ', D_meta)
   ```
   - 在元任务分布上优化
   - 学习良好初始化

### 元任务设计

1. **Base Task**：正常训练任务
2. **Stress Task**：单模态压力任务（RGB或X模态退化）
3. **Conflict Task**：跨模态冲突任务（两模态同时退化）

### 模态退化

- **RGB退化**：亮度、对比度变化
- **X模态退化**：随机噪声添加

## 常见问题

### 1. CUDA内存不足

**解决方案**：
- 减小BATCH_SIZE（如：4或2）
- 减小SEARCH_SIZE和TEMPLATE_SIZE
- 使用梯度累积

### 2. 训练速度慢

**解决方案**：
- 增加NUM_WORKER
- 使用AMP（自动混合精度）
- 减少SAMPLE_PER_EPOCH

### 3. TensorBoard无法访问

**解决方案**：
- 检查端口是否被占用
- 确认防火墙设置
- 使用正确的端口转发命令

### 4. 数据集加载失败

**解决方案**：
- 检查数据集路径是否正确
- 确认数据集格式是否符合要求
- 查看错误日志定位问题

## 项目结构

```
ViPT/
├── lib/
│   ├── models/
│   │   └── vipt/
│   │       ├── meta_prompt.py          # 元提示生成器
│   │       ├── vit_prompt.py           # Vision Transformer
│   │       └── ostrack_prompt.py       # ViPT模型
│   ├── train/
│   │   ├── actors/
│   │   │   ├── vipt_meta.py            # 元学习Actor
│   │   │   └── vipt.py                 # 标准Actor
│   │   ├── dataset/
│   │   │   └── lasher.py               # LasHeR数据集
│   │   └── admin/
│   │       ├── local.py                # 本地配置
│   │       └── tensorboard.py          # TensorBoard配置
│   └── test/
│       └── tracker/
│           └── vipt.py                 # 测试跟踪器
├── experiments/
│   └── vipt/
│       ├── lasher_meta_small.yaml      # 小批次配置
│       └── deep_rgbd_meta.yaml         # 完整配置
├── train_lasher_meta.sh                # 训练脚本
├── check_dependencies.py               # 依赖检查
└── README_META.md                      # 本文档
```

## 参考文献

- ViPT: Visual Prompt Multi-Modal Tracking
- FOMAML: Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
- LasHeR: A Large-scale High-diversity Benchmark for RGBT Tracking

## 联系方式

如有问题，请提交Issue或联系项目维护者。
