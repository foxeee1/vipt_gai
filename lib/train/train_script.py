"""
训练流程脚本 - ViPT模型训练主入口

【功能】
本模块是ViPT模型训练的核心脚本，负责：
1. 加载并解析配置文件
2. 构建数据加载器
3. 创建网络模型
4. 设置优化器和学习率调度器
5. 初始化训练器并启动训练循环

【训练流程】
1. 配置加载 → 2. 数据加载 → 3. 模型构建 → 4. 损失函数定义 → 5. 训练器初始化 → 6. 开始训练

【支持的任务类型】
- ViPT (Visual Prompt Multi-Modal Tracking): 主要的多模态跟踪模型
- ViPTMeta: 带元学习提示的ViPT变体（可选）

【作者】
基于ViPT框架
"""

# ==================== 导入必要的库 ====================
import os  # 操作系统库，用于路径操作、目录创建等
import torch.nn as nn  # 【v25】BN冻结需要

# ==================== 损失函数相关 ====================
from lib.utils.box_ops import giou_loss  # GIoU损失，用于边界框回归
from torch.nn.functional import l1_loss  # L1损失，用于边界框回归
from torch.nn import BCEWithLogitsLoss  # 二分类交叉熵损失，用于分类

# ==================== 训练管道相关 ====================
from lib.train.trainers import LTRTrainer  # 训练器基类，管理训练循环

# ==================== 分布式训练相关 ====================
# DDP (DistributedDataParallel): PyTorch的分布式数据并行封装
# 使得多GPU训练对用户透明
from torch.nn.parallel import DistributedDataParallel as DDP

# ==================== 基础函数 ====================
# 包含训练所需的各种辅助函数
# 如学习率调度、参数分组等
from .base_functions import *

# ==================== 网络模型 ====================
# build_ostrack: 构建OSNet跟踪网络
# build_viptrack: 构建ViPT多模态跟踪网络
from lib.models.vipt import build_ostrack, build_viptrack

# ==================== 训练Actor ====================
# Actor定义了训练过程中单个批次的前向传播和损失计算逻辑
# ViPTActor: 标准ViPT训练Actor
# ViPTMetaActor: 带元学习提示的ViPT训练Actor（FOMAML实现）
from lib.train.actors import ViPTActor, ViPTMetaActor

# ==================== 动态模块导入 ====================
# 用于运行时动态导入配置模块
import importlib

# ==================== 自定义损失函数 ====================
# Focal Loss: 用于处理类别不平衡的损失函数
from lib.utils.focal_loss import FocalLoss


def run(settings):
    """
    训练主函数

    【功能】
    执行完整的训练流程，包括：
    1. 配置加载和更新
    2. 日志目录创建
    3. 数据加载器构建
    4. 网络模型创建和DDP封装
    5. 损失函数和训练Actor创建
    6. 优化器和学习率调度器设置
    7. 训练器初始化和训练启动

    【参数】
    settings: Settings对象，包含训练所需的所有配置信息
        - script_name: 训练脚本名称（如'vipt'）
        - config_name: 配置文件名称（如'deep_rgbt'）
        - save_dir: 保存目录
        - local_rank: 本地GPU排名（-1表示非分布式）
        - cfg_file: 配置文件路径

    【返回值】
    无（训练完成后直接退出）
    """
    # ==================== 第一步：配置加载 ====================
    # 设置训练描述信息
    settings.description = 'Training script for ViPT'

    # 检查配置文件是否存在
    # cfg_file通常是 experiments/vipt/deep_rgbt.yaml 这样的路径
    if not os.path.exists(settings.cfg_file):
        raise ValueError("配置文件不存在: %s" % settings.cfg_file)

    # 动态导入配置模块
    # lib.config.<script_name>.config 是配置模块的命名规范
    # 例如: lib.config.vipt.config
    config_module_name = "lib.config.%s.config" % settings.script_name
    config_module = importlib.import_module(config_module_name)

    # 获取配置的cfg对象（EasyDict类型，支持点号访问）
    cfg = config_module.cfg

    # 从YAML文件更新配置
    # 将YAML文件中的参数覆盖到cfg对象中
    config_module.update_config_from_file(settings.cfg_file)

    # ==================== 打印配置信息（仅主进程） ====================
    # local_rank == -1 或 0 表示主进程
    # 分布式训练时只有主进程打印，避免重复输出
    if settings.local_rank in [-1, 0]:
        print("=" * 60)
        print("新配置内容如下:")
        print("=" * 60)
        for key in cfg.keys():
            print("[%s] 配置:" % key)
            print(cfg[key])
            print()
        print("=" * 60)

    # ==================== 第二步：更新训练设置 ====================
    # 根据配置更新settings对象
    # 包括数据路径、模型路径等
    update_settings(settings, cfg)

    # ==================== 第三步：创建日志目录 ====================
    # 日志保存路径: <save_dir>/logs/<script_name>-<config_name>.log
    log_dir = os.path.join(settings.save_dir, 'logs')

    # 只有主进程创建目录，避免多进程冲突
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # 日志文件路径
    settings.log_file = os.path.join(
        log_dir,
        "%s-%s.log" % (settings.script_name, settings.config_name)
    )

    # ==================== 第四步：构建数据加载器 ====================
    # build_dataloaders 返回 (train_loader, val_loader)
    # train_loader: 训练数据加载器
    # val_loader: 验证数据加载器（可能为None）
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # ==================== 第五步：创建网络模型 ====================
    # 根据script_name选择对应的网络构建函数
    if settings.script_name == "vipt":
        # 构建ViPT多模态跟踪网络
        net = build_viptrack(cfg)
    else:
        raise ValueError("非法的脚本名称: %s" % settings.script_name)

    # ==================== 第六步：模型移到GPU并封装DDP ====================
    # 将模型参数和缓冲区移到GPU
    net.cuda()

    # 分布式数据并行封装
    if settings.local_rank != -1:
        # 多GPU训练模式
        # DDP会：
        # 1. 将模型复制到对应GPU
        # 2. 自动管理梯度同步
        # 3. 使得训练代码与单GPU几乎相同

        # device_ids: 当前进程使用的GPU设备列表
        # find_unused_parameters: 是否查找未使用的参数（某些模型可能有）
        net = DDP(
            net,
            device_ids=[settings.local_rank],  # 当前进程使用的GPU
            find_unused_parameters=True
        )

        # 设置设备
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        # 单GPU训练模式
        settings.device = torch.device("cuda:0")

    # 【v25修复】全程冻结BN层，防止小batch统计不稳定导致训练崩溃
    freeze_bn = getattr(cfg.TRAIN, "FREEZE_BN", False)
    if freeze_bn:
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        print(f"[v25] 已冻结全部BN层 (FREEZE_BN={freeze_bn})")

    # ==================== 第七步：从配置中读取训练选项 ====================
    # 深度监督：在训练中间层添加辅助损失
    # 可以加速收敛，但会增加显存开销
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)

    # 知识蒸馏：使用教师模型指导学生模型训练
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)

    # 蒸馏损失类型：KL散度或其他
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    # ==================== 第八步：定义损失函数 ====================
    # 创建Focal Loss实例（用于处理正负样本不平衡）
    focal_loss = FocalLoss()

    # 损失函数字典
    # giou: 边界框回归GIoU损失
    # l1: 边界框回归L1损失
    # focal: 分类Focal损失
    # cls: 二分类损失（当前版本未使用）
    objective = {
        'giou': giou_loss,
        'l1': l1_loss,
        'focal': focal_loss,
        'cls': BCEWithLogitsLoss()
    }

    # 损失权重字典
    # 各损失成分在总损失中的加权系数
    loss_weight = {
        'giou': cfg.TRAIN.GIOU_WEIGHT,  # GIoU损失权重
        'l1': cfg.TRAIN.L1_WEIGHT,     # L1损失权重
        'focal': 1.,                   # Focal损失权重（固定为1）
        'cls': 1.0                     # 分类损失权重（固定为1）
    }

    # ==================== 第九步：创建训练Actor ====================
    # Actor负责：
    # 1. 前向传播
    # 2. 损失计算
    # 3. 反向传播（部分）
    if settings.script_name == "vipt":
        # 检查是否启用元学习
        # META.ENABLE 为 True 时使用 ViPTMetaActor
        if hasattr(cfg.TRAIN, "META") and cfg.TRAIN.META.ENABLE:
            # 元学习Actor：实现FOMAML内外环更新
            actor = ViPTMetaActor(
                net=net,
                objective=objective,
                loss_weight=loss_weight,
                settings=settings,
                cfg=cfg
            )
        else:
            # 标准Actor：普通训练流程
            actor = ViPTActor(
                net=net,
                objective=objective,
                loss_weight=loss_weight,
                settings=settings,
                cfg=cfg
            )
    else:
        raise ValueError("非法的脚本名称: %s" % settings.script_name)

    # ==================== 第十步：创建优化器和学习率调度器 ====================
    # get_optimizer_scheduler 来自 base_functions.py
    # 返回:
    # - optimizer: 优化器（如AdamW）
    # - lr_scheduler: 学习率调度器（如StepLR）
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # ==================== 第十一步：从配置中读取训练选项 ====================
    # 是否使用自动混合精度（AMP）
    # AMP可以减少显存占用，加速训练，但可能有精度损失
    use_amp = getattr(cfg.TRAIN, "AMP", False)

    # 保存checkpoint的epoch间隔
    settings.save_epoch_interval = getattr(cfg.TRAIN, "SAVE_EPOCH_INTERVAL", 1)

    # 保存最近N个epoch的checkpoint
    settings.save_last_n_epoch = getattr(cfg.TRAIN, "SAVE_LAST_N_EPOCH", 1)

    settings.early_stop_patience = getattr(cfg.TRAIN, "EARLY_STOP_PATIENCE", 0)

    settings.cfg = cfg
    settings.resume_checkpoint = getattr(settings, 'resume', None)

    # ==================== 第十二步：创建训练器 ====================
    # LTRTrainer 管理完整的训练循环
    # 参数：
    # - actor: 训练Actor
    # - loaders: 数据加载器列表 [train_loader] 或 [train_loader, val_loader]
    # - optimizer: 优化器
    # - settings: 训练设置
    # - lr_scheduler: 学习率调度器
    # - use_amp: 是否使用混合精度
    if loader_val is None:
        # 没有验证集
        trainer = LTRTrainer(
            actor,
            [loader_train],  # 只有训练集
            optimizer,
            settings,
            lr_scheduler,
            use_amp=use_amp
        )
    else:
        # 有验证集
        trainer = LTRTrainer(
            actor,
            [loader_train, loader_val],  # 训练集和验证集
            optimizer,
            settings,
            lr_scheduler,
            use_amp=use_amp
        )

    # ==================== 第十三步：启动训练循环 ====================
    # resume_checkpoint: 从指定路径恢复训练（用于三阶段权重传递）
    # load_latest: 是否从最新checkpoint恢复（仅当resume_checkpoint为None时生效）
    trainer.train(
        cfg.TRAIN.EPOCH,
        load_latest=(settings.resume_checkpoint is None),
        fail_safe=True,
        resume_checkpoint=settings.resume_checkpoint
    )
