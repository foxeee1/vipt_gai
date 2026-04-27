import os
import argparse
import random
import shutil
import time
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='ViPT三阶段串行训练')
    parser.add_argument('--script', type=str, default='vipt', help='training script name')
    parser.add_argument('--config', type=str, default='exp22_parallel_residual', help='base yaml configure file name')
    parser.add_argument('--save_dir', type=str, default='./output', help='root directory to save')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default='single')
    parser.add_argument('--nproc_per_node', type=int, default=1)
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)
    parser.add_argument('--project_name', type=str, default='vipt_three_stage')
    parser.add_argument('--stage1_epochs', type=int, default=15)
    parser.add_argument('--stage2_epochs', type=int, default=20)
    parser.add_argument('--stage3_epochs', type=int, default=25)
    args = parser.parse_args()
    return args


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(cfg, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


def build_train_cmd(args, config_name, stage_num):
    master_port = random.randint(10000, 50000)
    if args.mode == "single":
        cmd = (
            f"python lib/train/run_training.py --script {args.script} --config {config_name} "
            f"--save_dir {args.save_dir} --use_lmdb {args.use_lmdb} "
            f"--script_prv {args.script} --config_prv {config_name} "
            f"--use_wandb {args.use_wandb}"
        )
    else:
        cmd = (
            f"python -m torch.distributed.launch --nproc_per_node {args.nproc_per_node} "
            f"--master_port {master_port} lib/train/run_training.py "
            f"--script {args.script} --config {config_name} "
            f"--save_dir {args.save_dir} --use_lmdb {args.use_lmdb} "
            f"--script_prv {args.script} --config_prv {config_name} "
            f"--use_wandb {args.use_wandb}"
        )
    return cmd


def get_latest_ckpt(search_dir):
    if not os.path.exists(search_dir):
        return None
    ckpts = list(Path(search_dir).glob("**/*_ep*.pth.tar"))
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda p: p.stat().st_mtime))


def run_stage(stage_num, stage_name, epochs, inject_layers, prev_stage_ckpt=None,
               base_config=None, args=None):
    print(f"\n{'='*70}")
    print(f"  Stage {stage_num}: {stage_name}")
    print(f"{'='*70}")
    print(f"  STAGE={stage_num}, EPOCH={epochs}")
    print(f"  注入层: {inject_layers}")

    stage_config = base_config.copy()
    stage_config['TRAIN']['STAGE'] = stage_num
    stage_config['TRAIN']['EPOCH'] = epochs

    # 【v25修复】Stage1/2禁止早停（必须跑满分配的epoch），Stage3才允许早停
    if stage_num in [1, 2]:
        stage_config['TRAIN']['EARLY_STOP_PATIENCE'] = 0
        print(f"  [早停] 已禁用（Stage{stage_num}必须跑满{epochs}轮）")
    else:
        stage_config['TRAIN']['EARLY_STOP_PATIENCE'] = base_config.get('TRAIN', {}).get('EARLY_STOP_PATIENCE', 5)
        print(f"  [早停] 已启用（耐心值={stage_config['TRAIN']['EARLY_STOP_PATIENCE']}）")

    stage_config['MODEL']['BACKBONE']['META_PROMPT_INJECT_LAYERS'] = inject_layers.get('consistency', [5, 6])
    stage_config['MODEL']['BACKBONE']['TEMPORAL_PROMPT_INJECT_LAYERS'] = inject_layers.get('temporal', [8, 9])
    stage_config['MODEL']['BACKBONE']['MASK_PROMPT_INJECT_LAYERS'] = inject_layers.get('mask', [9])
    stage_config['MODEL']['BACKBONE']['MODALITY_PROMPT_INJECT_LAYERS'] = inject_layers.get('modality', [1, 2, 3])

    config_name = f"stage{stage_num}"
    config_path = os.path.join('experiments/vipt', f'{config_name}.yaml')
    save_yaml(stage_config, config_path)
    print(f"  [已创建配置] {config_path}")

    # run_training.py中project_path = train/{script}/{config}
    # 实际checkpoint保存路径: {save_dir}/checkpoints/train/{script}/{config}/
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints', 'train', args.script, config_name)

    if prev_stage_ckpt is not None and os.path.exists(prev_stage_ckpt):
        os.makedirs(ckpt_dir, exist_ok=True)
        dst_ckpt = os.path.join(ckpt_dir, 'resume_from.pth.tar')
        shutil.copy(prev_stage_ckpt, dst_ckpt)
        print(f"  [已复制权重] {prev_stage_ckpt} -> {dst_ckpt}")

    cmd = build_train_cmd(args, config_name, stage_num)
    print(f"\n[Stage {stage_num}] 执行:\n{cmd}\n")

    start = time.time()
    exit_code = os.system(cmd)
    elapsed = time.time() - start

    if exit_code != 0:
        print(f"\n[ERROR] Stage {stage_num} 训练失败，退出码: {exit_code}")
        return None

    print(f"\n[Stage {stage_num}] 完成! 耗时: {elapsed/60:.1f} 分钟")

    ckpt = get_latest_ckpt(ckpt_dir)
    if ckpt:
        print(f"  最新权重: {ckpt}")
    else:
        print(f"  [WARNING] 未找到检查点于 {ckpt_dir}")

    return ckpt


if __name__ == "__main__":
    args = parse_args()

    base_config_name = args.config
    if not base_config_name.endswith('.yaml'):
        base_config_name += '.yaml'
    base_yaml_path = os.path.join('experiments/vipt', base_config_name)

    if not os.path.exists(base_yaml_path):
        print(f"[ERROR] 配置文件不存在: {base_yaml_path}")
        exit(1)

    print("=" * 70)
    print("           ViPT 三阶段串行训练 (Three-Stage Training)")
    print("=" * 70)
    print(f"  基础配置:   {base_yaml_path}")
    print(f"  保存根目录: {args.save_dir}")
    print(f"  项目名:     {args.project_name}")
    print(f"  GPU模式:    {args.mode} ({args.nproc_per_node} GPUs)")
    print(f"  Stage1:     {args.stage1_epochs} epochs")
    print(f"  Stage2:     {args.stage2_epochs} epochs")
    print(f"  Stage3:     {args.stage3_epochs} epochs")
    print("=" * 70)

    base_config = load_yaml(base_yaml_path)

    inject_layers = {
        'modality': [1, 2, 3],
        'consistency': [5, 6],
        'temporal': [8, 9],
        'mask': [9]
    }

    stage_configs = []
    try:
        ckpt1 = run_stage(
            stage_num=1,
            stage_name="模态专属Prompt训练",
            epochs=args.stage1_epochs,
            inject_layers=inject_layers,
            prev_stage_ckpt=None,
            base_config=base_config,
            args=args
        )
        stage_configs.append('experiments/vipt/stage1.yaml')

        ckpt2 = run_stage(
            stage_num=2,
            stage_name="Consistency + Temporal训练",
            epochs=args.stage2_epochs,
            inject_layers=inject_layers,
            prev_stage_ckpt=ckpt1,
            base_config=base_config,
            args=args
        )
        stage_configs.append('experiments/vipt/stage2.yaml')

        ckpt3 = run_stage(
            stage_num=3,
            stage_name="全参数联合微调",
            epochs=args.stage3_epochs,
            inject_layers=inject_layers,
            prev_stage_ckpt=ckpt2,
            base_config=base_config,
            args=args
        )
        stage_configs.append('experiments/vipt/stage3.yaml')

        total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs
        print("\n" + "=" * 70)
        print("                    三阶段训练全部完成!")
        print("=" * 70)
        print(f"  总Epochs: {total_epochs}")
        print(f"  权重路径:")
        if ckpt1:
            print(f"    Stage1: {ckpt1}")
        if ckpt2:
            print(f"    Stage2: {ckpt2}")
        if ckpt3:
            print(f"    Stage3: {ckpt3}")
        print(f"  配置文件:")
        for cfg in stage_configs:
            print(f"    {cfg}")
        print(f"  TensorBoard:")
        print(f"    tensorboard --logdir_spec=stage1:{args.save_dir}/checkpoints/train/{args.script}/stage1/tensorboard,stage2:{args.save_dir}/checkpoints/train/{args.script}/stage2/tensorboard,stage3:{args.save_dir}/checkpoints/train/{args.script}/stage3/tensorboard --port 6006")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n[中断] 用户取消训练")
    except Exception as e:
        print(f"\n[ERROR] 训练异常: {e}")
