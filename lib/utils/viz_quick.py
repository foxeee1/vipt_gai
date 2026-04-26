#!/usr/bin/env python
"""
ViPT 逐层可视化 - 支持任意Transformer层 + Prompt注入前后对比
============================================================
核心功能:
  1. 12层Transformer逐层输出: 特征图 / Score Map / 热力图
  2. Prompt注入前后对比: Layer0输入(RGB only) vs Layer0输出(RGB+Prompt)
  3. 完全对齐实际跟踪pipeline: sample_target / map_box_back / PreprocessorMM

用法:
  # 查看所有12层
  python lib/utils/viz_quick.py \
      --checkpoint output/checkpoints/train/vipt/exp1_baseline_standard/ViPTrack_ep0060.pth.tar \
      --config experiments/vipt/exp1_baseline_standard.yaml \
      --layers all

  # 查看指定层 (0-11)
  python lib/utils/viz_quick.py \
      --checkpoint ... --config ... \
      --layers 0,3,6,11

  # 查看Prompt注入前后
  python lib/utils/viz_quick.py \
      --checkpoint ... --config ... \
      --layers all --show_prompt
"""
import os, sys, argparse, math, numpy as np, cv2, torch, yaml, copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from easydict import EasyDict as edict
from lib.models.vipt.ostrack_prompt import build_viptrack
from lib.models.vipt.utils import combine_tokens, recover_tokens


def parse_args():
    p = argparse.ArgumentParser(description='ViPT逐层可视化(对齐跟踪pipeline)')
    p.add_argument('--checkpoint', default=None,
                   help='单个checkpoint路径 (与--batch互斥)')
    p.add_argument('--config', default=None,
                   help='单个config路径 (与--batch互斥)')
    p.add_argument('--batch', action='store_true',
                   help='批量模式: 自动扫描output/checkpoints下所有实验')
    p.add_argument('--video_idx', type=int, default=0)
    p.add_argument('--output_dir', default='viz_quick')
    p.add_argument('--num_frames', type=int, default=3)
    p.add_argument('--layers', default='all',
                   help='要可视化的层, 如 "all" "0,5,11" "0-3,11"')
    p.add_argument('--show_prompt', action='store_true',
                   help='显示Prompt注入前后的对比')
    p.add_argument('--compare', action='store_true',
                   help='对比模式: 生成所有实验的并排对比图(共享colorbar)')
    args = p.parse_args()
    if not args.batch and (args.checkpoint is None or args.config is None):
        p.error('需要指定 --batch 或 --checkpoint + --config')
    return args


def find_all_experiments():
    """扫描output/checkpoints下所有有checkpoint的实验，返回[(config_path, ckpt_path), ...]"""
    ckpt_base = os.path.join(_project_root, 'output', 'checkpoints', 'train', 'vipt')
    cfg_base = os.path.join(_project_root, 'experiments', 'vipt')
    results = []
    if not os.path.isdir(ckpt_base):
        print(f"[批量] checkpoint目录不存在: {ckpt_base}")
        return results
    for exp_name in sorted(os.listdir(ckpt_base)):
        exp_dir = os.path.join(ckpt_base, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        ckpts = sorted([f for f in os.listdir(exp_dir) if f.endswith('.pth.tar')])
        if not ckpts:
            continue
        ckpt_path = os.path.join(exp_dir, ckpts[-1])
        cfg_path = os.path.join(cfg_base, exp_name + '.yaml')
        if not os.path.exists(cfg_path):
            print(f"[批量] ⚠ 跳过 {exp_name}: 找不到config {cfg_path}")
            continue
        results.append((cfg_path, ckpt_path, exp_name))
    return results


def parse_layers(layers_str, max_layers=12):
    if layers_str.lower() == 'all':
        return list(range(max_layers))
    result = []
    for part in layers_str.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    result = [l for l in sorted(set(result)) if 0 <= l < max_layers]
    return result


def sample_target(im, target_bb, search_area_factor, output_sz=None):
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    assert w > 0 and h > 0, f"[sample_target] bbox尺寸无效: w={w}, h={h}"
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    assert crop_sz >= 1, f"[sample_target] 裁剪尺寸过小: crop_sz={crop_sz}"
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                         cv2.BORDER_CONSTANT)
    resize_factor = 1.0
    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
    return im_crop_padded, resize_factor


class PreprocessorMM:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(1, 6, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(1, 6, 1, 1).cuda()

    def process(self, img_arr):
        assert img_arr.ndim == 3 and img_arr.shape[2] == 6, \
            f"[PreprocessorMM] 输入维度错误! 预期(H,W,6), 实际{img_arr.shape}"
        img_tensor = torch.tensor(img_arr).cuda().float().permute(2, 0, 1).unsqueeze(0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std
        assert not torch.isnan(img_tensor_norm).any(), "[PreprocessorMM] 归一化后出现NaN!"
        return img_tensor_norm


def map_box_back(pred_box, state, search_size, resize_factor):
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    x1 = cx_real - 0.5 * w
    y1 = cy_real - 0.5 * h
    assert w > 0 and h > 0, f"[map_box_back] 预测框尺寸无效: w={w}, h={h}"
    return [x1, y1, w, h]


def clip_box(box, H, W, margin=10):
    x1, y1, w, h = box
    x1 = max(0, min(x1, W - margin))
    y1 = max(0, min(y1, H - margin))
    w = min(w, W - x1 - margin)
    h = min(h, H - y1 - margin)
    return [x1, y1, max(w, margin), max(h, margin)]


def load_video(video_idx=0, max_frames=10):
    base = 'data/lasher/testingset'
    videos = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    if video_idx >= len(videos):
        print(f"⚠ video_idx={video_idx} 超出范围({len(videos)}个视频)")
        return None, None
    vname = videos[video_idx]
    rgb_dir = os.path.join(base, vname, 'visible')
    tir_dir = os.path.join(base, vname, 'infrared')
    gt_path = os.path.join(base, vname, 'visible.txt')
    gt_data = []
    if os.path.exists(gt_path):
        for line in open(gt_path).readlines():
            vals = [float(x) for x in line.strip().split(',')]
            if len(vals) >= 4:
                gt_data.append(vals[:4])
    frames, gt_list = [], []
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg'))])
    tir_files = sorted([f for f in os.listdir(tir_dir) if f.lower().endswith(('.png', '.jpg'))])
    for i, (rf, tf) in enumerate(zip(rgb_files, tir_files)):
        rgb_img = cv2.imread(os.path.join(rgb_dir, rf))
        tir_img = cv2.imread(os.path.join(tir_dir, tf))
        if rgb_img is None or tir_img is None:
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        tir_img = cv2.cvtColor(tir_img, cv2.COLOR_BGR2RGB)
        img_6ch = np.concatenate([rgb_img, tir_img], axis=2)
        frames.append(img_6ch)
        gt_list.append(gt_data[i] if i < len(gt_data) else None)
        if len(frames) >= max_frames:
            break
    print(f"[数据] 视频: {vname}, 帧数: {len(frames)}, GT示例: {gt_list[0] if gt_list else None}")
    return frames, gt_list


def token_to_heatmap(tokens, feat_sz, search_size=256, vmin=None, vmax=None,
                      global_index_s=None, gt_bbox_in_search=None):
    """将Search Token [B, N, C] 转为热力图 [H, W]
    
    核心修复: 适配CE剪枝，用global_index_s恢复Token的原始空间位置
    
    Args:
        tokens: Search Token [B, N, C]
        feat_sz: 特征图尺寸 (默认16)
        search_size: 搜索图尺寸 (默认256)
        vmin/vmax: 固定归一化范围
        global_index_s: 该层Search Token的原始空间索引 [B, N] (CE模块返回)
        gt_bbox_in_search: 搜索图上的GT框 [x1,y1,w,h]，用于目标引导归一化
    """
    feat = tokens[0].mean(dim=-1).detach().cpu().numpy()
    h = w = feat_sz
    total_tokens = h * w
    N_tokens = len(feat)

    feat_map = np.zeros(total_tokens, dtype=np.float32)

    if global_index_s is not None:
        idx = global_index_s[0].cpu().numpy().astype(np.int64)
        valid_mask = (idx >= 0) & (idx < total_tokens)
        feat_map[idx[valid_mask]] = feat[valid_mask]
    else:
        feat_map[:min(N_tokens, total_tokens)] = feat[:min(N_tokens, total_tokens)]

    feat_map = feat_map.reshape(h, w)

    if gt_bbox_in_search is not None:
        x1, y1, w_bbox, h_bbox = gt_bbox_in_search
        x1_f = int(max(0, (x1 / search_size) * feat_sz))
        y1_f = int(max(0, (y1 / search_size) * feat_sz))
        x2_f = int(min(feat_sz, ((x1 + w_bbox) / search_size) * feat_sz))
        y2_f = int(min(feat_sz, ((y1 + h_bbox) / search_size) * feat_sz))
        if x2_f > x1_f and y2_f > y1_f:
            target_max = feat_map[y1_f:y2_f, x1_f:x2_f].max()
            vmax_auto = max(target_max, np.percentile(feat_map, 99.5))
            vmin_auto = np.percentile(feat_map, 0.5)
        else:
            vmin_auto, vmax_auto = np.percentile(feat_map, [0.5, 99.5])
    else:
        vmin_auto, vmax_auto = np.percentile(feat_map, [0.5, 99.5])

    if vmax_auto - vmin_auto < 1e-8:
        vmin_auto, vmax_auto = feat_map.min(), feat_map.max()
    vmin = vmin_auto if vmin is None else vmin
    vmax = vmax_auto if vmax is None else vmax

    feat_clip = np.clip(feat_map, vmin, vmax)
    feat_norm = (feat_clip - vmin) / (vmax - vmin + 1e-8)

    keep_ratio = N_tokens / total_tokens if total_tokens > 0 else 1.0
    gamma = 0.55 if keep_ratio < 0.7 else 1.0
    feat_enhanced = np.power(feat_norm, gamma)

    feat_resized = cv2.resize(feat_enhanced.astype(np.float32), (search_size, search_size))
    return feat_resized, vmin, vmax


def attn_to_heatmap(attn, lens_z, feat_sz, search_size=256):
    """将注意力权重转为Search区域热力图 [H, W]"""
    if attn.dim() == 4:
        a = attn[0].mean(dim=0).detach().cpu().numpy()
    elif attn.dim() == 3:
        a = attn[0].detach().cpu().numpy()
    else:
        return None
    search_attn = a[lens_z:, lens_z:]
    h = w = feat_sz
    if search_attn.shape[0] >= h * w:
        diag = np.diag(search_attn[:h * w, :h * w])
    else:
        diag = np.zeros(h * w)
        for idx in range(min(search_attn.shape[0], h * w)):
            diag[idx] = search_attn[idx].mean()
    diag = diag.reshape(h, w)
    lo, hi = np.percentile(diag, [1, 99])
    if hi - lo < 1e-8:
        lo, hi = diag.min(), diag.max()
    diag_norm = (np.clip(diag, lo, hi) - lo) / (hi - lo + 1e-8)
    return cv2.resize(diag_norm.astype(np.float32), (search_size, search_size))


class LayerHook:
    """Hook提取每层Transformer Block的输入特征、输出特征、注意力和CE空间索引"""

    def __init__(self):
        self.features = {}
        self.inputs = {}
        self.attns = {}
        self.global_index_s = {}
        self._handles = []

    def register(self, model):
        backbone = model.backbone
        for i, blk in enumerate(backbone.blocks):
            h1 = blk.register_forward_hook(self._make_feat_hook(i))
            self._handles.append(h1)
            h2 = blk.attn.register_forward_hook(self._make_attn_hook(i))
            self._handles.append(h2)
            h3 = blk.register_forward_pre_hook(self._make_input_hook(i))
            self._handles.append(h3)

    def _make_feat_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 3:
                self.features[layer_idx] = output[0].detach()
                if len(output) >= 3 and output[2] is not None:
                    self.global_index_s[layer_idx] = output[2].detach()
                else:
                    self.global_index_s[layer_idx] = None
            else:
                self.features[layer_idx] = output.detach() if isinstance(output, torch.Tensor) else None
                self.global_index_s[layer_idx] = None
        return hook

    def _make_input_hook(self, layer_idx):
        def hook(module, input):
            if isinstance(input, tuple):
                self.inputs[layer_idx] = input[0].detach()
            else:
                self.inputs[layer_idx] = input.detach()
        return hook

    def _make_attn_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                self.attns[layer_idx] = output[1].detach()
        return hook

    def clear(self):
        self.features.clear()
        self.inputs.clear()
        self.attns.clear()
        self.global_index_s.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def run_forward_with_hooks(model, z_tensor, x_tensor, hook):
    """运行前向传播并收集逐层特征"""
    hook.clear()
    with torch.no_grad():
        out_dict = model.forward(template=z_tensor, search=x_tensor)
    return out_dict


def run_forward_no_prompt(model, z_tensor, x_tensor, hook):
    """运行无Prompt的前向传播 (仅RGB, 不注入TIR) — 不走head, 仅收集逐层特征"""
    hook.clear()
    backbone = model.backbone

    B, C, H, W = x_tensor.shape
    z_rgb = z_tensor[:, :3, :, :]
    x_rgb = x_tensor[:, :3, :, :]

    z = backbone.patch_embed(z_rgb)
    x = backbone.patch_embed(x_rgb)

    z += backbone.pos_embed_z
    x += backbone.pos_embed_x

    if backbone.add_sep_seg:
        x += backbone.search_segment_pos_embed
        z += backbone.template_segment_pos_embed

    x = combine_tokens(z, x, mode=backbone.cat_mode)

    if backbone.add_cls_token:
        cls_tokens = backbone.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + backbone.cls_pos_embed
        x = torch.cat([cls_tokens, x], dim=1)

    x = backbone.pos_drop(x)

    lens_z = backbone.pos_embed_z.shape[1]
    lens_x = backbone.pos_embed_x.shape[1]
    global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device).repeat(B, 1)
    global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device).repeat(B, 1)

    for i, blk in enumerate(backbone.blocks):
        x, global_index_t, global_index_s, _, attn = blk(
            x, global_index_t, global_index_s)
        hook.features[i] = x.detach()
        hook.global_index_s[i] = global_index_s.detach() if global_index_s is not None else None
        if attn is not None:
            hook.attns[i] = attn.detach()

    return {}


def extract_search_tokens(feature, lens_z, lens_x, cat_mode, feat_sz):
    """从合并特征中提取Search Tokens (适配CE剪枝: search token数量可能减少)"""
    if feature is None:
        return None
    actual_len = feature.shape[1]
    if cat_mode == 'direct':
        search_tokens = feature[:, lens_z:]
    elif cat_mode == 'template_central':
        central_pivot = lens_x // 2
        first_half = feature[:, :central_pivot, :]
        second_half = feature[:, central_pivot + lens_z:, :]
        search_tokens = torch.cat([first_half, second_half], dim=1)
    else:
        search_tokens = feature[:, lens_z:]
    return search_tokens


def draw_heatmap_overlay(srch_rgb, heatmap, gt_box_in_search=None, score_max_pos=None,
                         alpha=0.55, vmin=None, vmax=None):
    """绘制热力图叠加
    
    Args:
        alpha: 热力图混合权重 (默认0.55，提高区分度)
        vmin/vmax: 归一化范围，用于在图上标注
    """
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_color.astype(float) + (1 - alpha) * srch_rgb.astype(float)).astype(np.uint8)
    if gt_box_in_search is not None:
        x1, y1, w, h = [int(v) for v in gt_box_in_search]
        cv2.rectangle(blended, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
    if score_max_pos is not None:
        px, py = int(score_max_pos[0]), int(score_max_pos[1])
        cv2.circle(blended, (px, py), 5, (255, 0, 0), -1)
    if vmin is not None and vmax is not None:
        info_text = f"[{vmin:.3f},{vmax:.3f}]"
        cv2.putText(blended, info_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(blended, info_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return blended


def run_single_experiment(args, cfg_path, ckpt_path, exp_name):
    """运行单个实验的可视化"""
    cfg = edict(yaml.safe_load(open(cfg_path)))

    if 'TEST' not in cfg:
        cfg.TEST = edict()
    cfg.TEST.TEMPLATE_FACTOR = getattr(cfg.TEST, 'TEMPLATE_FACTOR', 2.0)
    cfg.TEST.TEMPLATE_SIZE = getattr(cfg.TEST, 'TEMPLATE_SIZE', 128)
    cfg.TEST.SEARCH_FACTOR = getattr(cfg.TEST, 'SEARCH_FACTOR', 4.0)
    cfg.TEST.SEARCH_SIZE = getattr(cfg.TEST, 'SEARCH_SIZE', 256)

    template_factor = cfg.TEST.TEMPLATE_FACTOR
    template_size = cfg.TEST.TEMPLATE_SIZE
    search_factor = cfg.TEST.SEARCH_FACTOR
    search_size = cfg.TEST.SEARCH_SIZE

    model = build_viptrack(cfg, training=False)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    load_result = model.load_state_dict(ckpt.get('model', ckpt.get('net', ckpt)), strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] params={n_params:,}, missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
    if load_result.missing_keys:
        print(f"[模型] ⚠ missing_keys: {load_result.missing_keys[:5]}...")
    model = model.cuda().eval()

    backbone = model.backbone
    depth = backbone.depth
    cat_mode = backbone.cat_mode
    feat_sz_s = model.feat_sz_s
    lens_z = backbone.pos_embed_z.shape[1]
    lens_x = backbone.pos_embed_x.shape[1]

    layers = parse_layers(args.layers, depth)
    meta_prompt = getattr(cfg.MODEL.BACKBONE, 'META_PROMPT', False)
    print(f"[配置] depth={depth}, cat_mode={cat_mode}, feat_sz_s={feat_sz_s}, meta_prompt={meta_prompt}")
    print(f"[配置] lens_z={lens_z}, lens_x={lens_x}")
    print(f"[配置] 可视化层: {layers}")

    hook = LayerHook()
    hook.register(model)

    if args.show_prompt:
        hook_no_prompt = LayerHook()
        hook_no_prompt.register(model)

    preprocessor = PreprocessorMM()
    frames, gt_list = load_video(args.video_idx, args.num_frames)
    if not frames:
        return

    out_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    init_bbox = gt_list[0]
    assert init_bbox is not None, "[初始化] 第一帧必须有GT bbox!"

    z_patch_arr, z_resize_factor = sample_target(frames[0], init_bbox,
                                                  template_factor, output_sz=template_size)
    z_tensor = preprocessor.process(z_patch_arr)
    state = list(init_bbox)

    n_show = min(args.num_frames, len(frames))

    for frame_i in range(n_show):
        img_6ch = frames[frame_i]
        H_img, W_img = img_6ch.shape[:2]
        rgb_orig = img_6ch[:, :, :3]

        if frame_i == 0:
            x_patch_arr, x_resize_factor = sample_target(img_6ch, init_bbox,
                                                          search_factor, output_sz=search_size)
        else:
            x_patch_arr, x_resize_factor = sample_target(img_6ch, state,
                                                          search_factor, output_sz=search_size)
        x_tensor = preprocessor.process(x_patch_arr)

        # ====== 有Prompt前向 ======
        out_dict = run_forward_with_hooks(model, z_tensor, x_tensor, hook)
        features_prompt = dict(hook.features)
        attns_prompt = dict(hook.attns)

        # 诊断: 输出特征hash和score
        import hashlib
        if frame_i == 0:
            for li in [0, 6, 11]:
                f = features_prompt.get(li)
                if f is not None:
                    h = hashlib.md5(f.cpu().numpy().tobytes()).hexdigest()[:10]
                    print(f"[诊断] Layer{li} feat_hash={h}, mean={f.mean():.4f}")
            sm = out_dict.get('score_map')
            if sm is not None:
                s = sm[0] if sm.dim()==3 else sm[0,0]
                print(f"[诊断] score_map: max={s.max():.4f}, mean={s.mean():.4f}")

        # ====== 无Prompt前向 (仅RGB) ======
        features_no_prompt = None
        attns_no_prompt = None
        if args.show_prompt:
            out_dict_no = run_forward_no_prompt(model, z_tensor, x_tensor, hook_no_prompt)
            features_no_prompt = dict(hook_no_prompt.features)
            attns_no_prompt = dict(hook_no_prompt.attns)

        # ====== 预测框 ======
        score_map = out_dict.get('score_map')
        pred_boxes_raw = out_dict.get('pred_boxes')

        sm_final = None
        if score_map is not None:
            s = score_map
            if s.dim() == 4:
                s = s[0, 0]
            elif s.dim() == 3:
                s = s[0]
            sm_final = s.cpu().numpy()

        if pred_boxes_raw is not None and pred_boxes_raw.shape[0] > 0:
            pb = pred_boxes_raw[0, 0].cpu().numpy()
            pred_box_scaled = (pb * search_size / x_resize_factor).tolist()
            if frame_i == 0:
                state = list(init_bbox)
            else:
                mapped = map_box_back(pred_box_scaled, state, search_size, x_resize_factor)
                state = clip_box(mapped, H_img, W_img, margin=10)

        # ====== 逐层可视化 ======
        srch_rgb = x_patch_arr[:, :, :3]
        inputs_prompt = dict(hook.inputs)
        global_indices_prompt = dict(hook.global_index_s)

        gt_bbox_in_search = None
        cur_bbox = init_bbox if frame_i == 0 else state
        if cur_bbox is not None:
            cx_gt = cur_bbox[0] + cur_bbox[2] / 2
            cy_gt = cur_bbox[1] + cur_bbox[3] / 2
            crop_sz = math.ceil(math.sqrt(cur_bbox[2] * cur_bbox[3]) * search_factor)
            cx_crop = cur_bbox[0] + cur_bbox[2] / 2
            cy_crop = cur_bbox[1] + cur_bbox[3] / 2
            x1_crop = round(cx_crop - crop_sz * 0.5)
            y1_crop = round(cy_crop - crop_sz * 0.5)
            gt_x1_s = (cx_gt - cur_bbox[2] / 2 - x1_crop) * (search_size / crop_sz)
            gt_y1_s = (cy_gt - cur_bbox[3] / 2 - y1_crop) * (search_size / crop_sz)
            gt_w_s = cur_bbox[2] * (search_size / crop_sz)
            gt_h_s = cur_bbox[3] * (search_size / crop_sz)
            gt_bbox_in_search = [gt_x1_s, gt_y1_s, gt_w_s, gt_h_s]

        n_cols = 4 if args.show_prompt else 3
        n_rows = len(layers)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row_i, layer_idx in enumerate(layers):
            feat_in = inputs_prompt.get(layer_idx)
            feat_out = features_prompt.get(layer_idx)
            attn = attns_prompt.get(layer_idx)
            gidx_out = global_indices_prompt.get(layer_idx)
            gidx_in = global_indices_prompt.get(layer_idx - 1) if layer_idx > 0 else None

            st_in = extract_search_tokens(feat_in, lens_z, lens_x, cat_mode, feat_sz_s) if feat_in is not None else None
            st_out = extract_search_tokens(feat_out, lens_z, lens_x, cat_mode, feat_sz_s) if feat_out is not None else None

            # 列0: Block输入
            ax = axes[row_i, 0]
            if st_in is not None:
                hm_in, vmin_in, vmax_in = token_to_heatmap(
                    st_in, feat_sz_s, search_size, global_index_s=gidx_in, gt_bbox_in_search=gt_bbox_in_search)
                ax.imshow(draw_heatmap_overlay(srch_rgb, hm_in))
                ax.set_title(f'L{layer_idx} Input [{vmin_in:.2f},{vmax_in:.2f}]', fontsize=10)
            else:
                ax.imshow(srch_rgb.astype(np.uint8))
                ax.set_title(f'L{layer_idx} Input N/A', fontsize=10)
            ax.axis('off')

            # 列1: Block输出
            ax = axes[row_i, 1]
            if st_out is not None:
                hm_out, vmin_o, vmax_o = token_to_heatmap(
                    st_out, feat_sz_s, search_size, global_index_s=gidx_out, gt_bbox_in_search=gt_bbox_in_search)
                ax.imshow(draw_heatmap_overlay(srch_rgb, hm_out))
                ax.set_title(f'L{layer_idx} Output [{vmin_o:.2f},{vmax_o:.2f}]', fontsize=10)
            else:
                ax.imshow(srch_rgb.astype(np.uint8))
                ax.set_title(f'L{layer_idx} Output N/A', fontsize=10)
            ax.axis('off')

            # 列2: 注意力热力图
            ax = axes[row_i, 2]
            if attn is not None:
                hm_attn = attn_to_heatmap(attn, lens_z, feat_sz_s, search_size)
                if hm_attn is not None:
                    ax.imshow(draw_heatmap_overlay(srch_rgb, hm_attn))
                    ax.set_title(f'L{layer_idx} Attention', fontsize=10)
                else:
                    ax.imshow(srch_rgb.astype(np.uint8))
                    ax.set_title(f'L{layer_idx} Attn N/A', fontsize=10)
            else:
                ax.imshow(srch_rgb.astype(np.uint8))
                ax.set_title(f'L{layer_idx} No Attn', fontsize=10)
            ax.axis('off')

            # 列3: Prompt前后对比
            if args.show_prompt and n_cols >= 4:
                ax = axes[row_i, 3]
                feat_no = features_no_prompt.get(layer_idx) if features_no_prompt else None
                if feat_no is not None and st_out is not None:
                    st_no = extract_search_tokens(feat_no, lens_z, lens_x, cat_mode, feat_sz_s)
                    hm_p, _, _ = token_to_heatmap(
                        st_out, feat_sz_s, search_size, global_index_s=gidx_out, gt_bbox_in_search=gt_bbox_in_search)
                    hm_n, _, _ = token_to_heatmap(
                        st_no, feat_sz_s, search_size, gt_bbox_in_search=gt_bbox_in_search)
                    diff = np.abs(hm_p - hm_n)
                    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                    diff_color = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
                    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
                    blended_diff = (0.5 * diff_color.astype(float) + 0.5 * srch_rgb.astype(float)).astype(np.uint8)
                    ax.imshow(blended_diff)
                    ax.set_title(f'L{layer_idx} Prompt Δ', fontsize=10)
                else:
                    ax.imshow(srch_rgb.astype(np.uint8))
                    ax.set_title(f'L{layer_idx} No Diff', fontsize=10)
                ax.axis('off')

        col_labels = ['Block Input', 'Block Output', 'Attention']
        if args.show_prompt:
            col_labels.append('Prompt Δ')

        for ci, clabel in enumerate(col_labels):
            fig.text(0.5 / n_cols + ci / n_cols, 0.99, clabel,
                     ha='center', va='top', fontsize=12, fontweight='bold',
                     transform=fig.transFigure)

        sm_max_str = f'{sm_final.max():.3f}' if sm_final is not None else 'N/A'
        plt.suptitle(
            f'Frame {frame_i} | {exp_name}\n'
            f'Layers: {layers} | Score_max={sm_max_str}',
            fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fpath = os.path.join(out_dir, f'frame_{frame_i:02d}_layers.png')
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Frame {frame_i}: {fpath}")

        # ====== 单层独立保存 ======
        layer_dir = os.path.join(out_dir, f'frame_{frame_i:02d}')
        os.makedirs(layer_dir, exist_ok=True)

        for layer_idx in layers:
            feat = features_prompt.get(layer_idx)
            feat_in = inputs_prompt.get(layer_idx)
            attn = attns_prompt.get(layer_idx)
            gidx_out = global_indices_prompt.get(layer_idx)
            gidx_in = global_indices_prompt.get(layer_idx - 1) if layer_idx > 0 else None

            n_cols_single = 3 + (1 if args.show_prompt else 0)
            fig_l, axes_l = plt.subplots(1, n_cols_single,
                                          figsize=(6 * n_cols_single, 5.5))

            ax = axes_l[0]
            if feat_in is not None:
                st_in = extract_search_tokens(feat_in, lens_z, lens_x, cat_mode, feat_sz_s)
                hm_in, vmin_in, vmax_in = token_to_heatmap(
                    st_in, feat_sz_s, search_size, global_index_s=gidx_in, gt_bbox_in_search=gt_bbox_in_search)
                blended_in = draw_heatmap_overlay(srch_rgb, hm_in)
                ax.imshow(blended_in)
                ax.set_title(f'Block Input [{vmin_in:.2f},{vmax_in:.2f}]', fontsize=11)
            else:
                ax.imshow(srch_rgb.astype(np.uint8))
                ax.set_title('Block Input (N/A)', fontsize=11)
            ax.axis('off')

            ax = axes_l[1]
            if feat is not None:
                search_tokens = extract_search_tokens(feat, lens_z, lens_x, cat_mode, feat_sz_s)
                hm, hvmin, hvmax = token_to_heatmap(
                    search_tokens, feat_sz_s, search_size, global_index_s=gidx_out, gt_bbox_in_search=gt_bbox_in_search)
                blended = draw_heatmap_overlay(srch_rgb, hm)
                ax.imshow(blended)
                ax.set_title(f'Block Output [{hvmin:.2f},{hvmax:.2f}]', fontsize=11)
            ax.axis('off')

            # Attention Map
            ax = axes_l[2]
            if attn is not None:
                hm_attn = attn_to_heatmap(attn, lens_z, feat_sz_s, search_size)
                if hm_attn is not None:
                    blended = draw_heatmap_overlay(srch_rgb, hm_attn)
                    ax.imshow(blended)
            ax.set_title('Attention Map', fontsize=11)
            ax.axis('off')

            # Prompt Diff
            if args.show_prompt:
                ax = axes_l[3]
                feat_no = features_no_prompt.get(layer_idx) if features_no_prompt else None
                if feat_no is not None and feat is not None:
                    st_prompt = extract_search_tokens(feat, lens_z, lens_x, cat_mode, feat_sz_s)
                    st_no = extract_search_tokens(feat_no, lens_z, lens_x, cat_mode, feat_sz_s)
                    hm_p, _, _ = token_to_heatmap(
                        st_prompt, feat_sz_s, search_size, global_index_s=gidx_out, gt_bbox_in_search=gt_bbox_in_search)
                    hm_n, _, _ = token_to_heatmap(
                        st_no, feat_sz_s, search_size, gt_bbox_in_search=gt_bbox_in_search)
                    diff = np.abs(hm_p - hm_n)
                    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                    diff_color = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
                    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
                    blended_diff = (0.5 * diff_color.astype(float) + 0.5 * srch_rgb.astype(float)).astype(np.uint8)
                    ax.imshow(blended_diff)
                ax.set_title(f'|Prompt - RGB| Diff', fontsize=12)
                ax.axis('off')

            fig_l.suptitle(f'Frame {frame_i} | Layer {layer_idx}', fontsize=13, fontweight='bold')
            plt.tight_layout()
            lpath = os.path.join(layer_dir, f'layer_{layer_idx:02d}.png')
            fig_l.savefig(lpath, dpi=150, bbox_inches='tight')
            plt.close(fig_l)

    # ====== 总览图: 所有帧 + 最终Score Map ======
    fig_overview, axes_o = plt.subplots(n_show, 5, figsize=(25, 5 * n_show))
    if n_show == 1:
        axes_o = axes_o.reshape(1, -1)

    state = list(init_bbox)
    for frame_i in range(n_show):
        img_6ch = frames[frame_i]
        H_img, W_img = img_6ch.shape[:2]
        rgb_orig = img_6ch[:, :, :3]

        if frame_i == 0:
            x_patch_arr, x_resize_factor = sample_target(img_6ch, init_bbox,
                                                          search_factor, output_sz=search_size)
        else:
            x_patch_arr, x_resize_factor = sample_target(img_6ch, state,
                                                          search_factor, output_sz=search_size)
        x_tensor = preprocessor.process(x_patch_arr)

        hook.clear()
        with torch.no_grad():
            out_dict = model.forward(template=z_tensor, search=x_tensor)

        score_map = out_dict.get('score_map')
        pred_boxes_raw = out_dict.get('pred_boxes')

        sm = None
        if score_map is not None:
            s = score_map
            if s.dim() == 4:
                s = s[0, 0]
            elif s.dim() == 3:
                s = s[0]
            sm = s.cpu().numpy()

        pred_box_orig = None
        if pred_boxes_raw is not None and pred_boxes_raw.shape[0] > 0:
            pb = pred_boxes_raw[0, 0].cpu().numpy()
            pred_box_scaled = (pb * search_size / x_resize_factor).tolist()
            if frame_i == 0:
                state = list(init_bbox)
            else:
                mapped = map_box_back(pred_box_scaled, state, search_size, x_resize_factor)
                state = clip_box(mapped, H_img, W_img, margin=10)
            pred_box_orig = state

        srch_rgb = x_patch_arr[:, :, :3]
        col_titles = ['Original+GT+Pred', 'Template', 'Search', 'Score Map', 'Heatmap Overlay']

        for j, title in enumerate(col_titles):
            ax = axes_o[frame_i, j]

            if j == 0:
                ax.imshow(rgb_orig)
                if gt_list[frame_i] is not None:
                    gx, gy, gw, gh = gt_list[frame_i]
                    ax.add_patch(Rectangle((gx, gy), gw, gh, linewidth=2,
                                           edgecolor='red', facecolor='none', linestyle='--'))
                    ax.text(gx, gy - 5, 'GT', color='red', fontsize=9, fontweight='bold')
                if pred_box_orig is not None and frame_i > 0:
                    px, py, pw, ph = pred_box_orig
                    ax.add_patch(Rectangle((px, py), pw, ph, linewidth=2,
                                           edgecolor='lime', facecolor='none', linestyle='-'))
                    ax.text(px, py + ph + 5, 'PRED', color='lime', fontsize=9, fontweight='bold')

            elif j == 1:
                ax.imshow(z_patch_arr[:, :, :3].astype(np.uint8))

            elif j == 2:
                ax.imshow(srch_rgb.astype(np.uint8))

            elif j == 3 and sm is not None:
                ax.imshow(sm, cmap='hot', vmin=0, vmax=1)
                idx = np.argmax(sm.flatten())
                iy, ix = divmod(idx, sm.shape[-1])
                ax.plot(ix, iy, 'c+', markersize=12, markeredgewidth=2)
                ax.set_title(f'score_max={sm.max():.3f}', fontsize=10)

            elif j == 4:
                if sm is not None:
                    sm_resized = cv2.resize(sm.astype(np.float32), (search_size, search_size))
                    sm_norm = (sm_resized - sm_resized.min()) / (sm_resized.max() - sm_resized.min() + 1e-8)
                    heatmap = cv2.applyColorMap((sm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    blended = (0.4 * heatmap.astype(float) + 0.6 * srch_rgb.astype(float)).astype(np.uint8)
                    ax.imshow(blended)
                    idx = np.argmax(sm.flatten())
                    iy, ix = divmod(idx, sm.shape[-1])
                    scale = search_size / sm.shape[0]
                    ax.plot((ix + 0.5) * scale, (iy + 0.5) * scale, 'wo',
                            markersize=10, markeredgewidth=2, markerfacecolor='none')
                else:
                    ax.imshow(srch_rgb.astype(np.uint8))

            ax.set_title(f'F{frame_i} {title}' if frame_i == 0 else title, fontsize=10)
            ax.axis('off')

    plt.suptitle(f'{exp_name} | Video {args.video_idx} | Overview',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    overview_path = os.path.join(out_dir, f'overview_v{args.video_idx}.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    hook.remove()
    if args.show_prompt:
        hook_no_prompt.remove()

    print(f"\n✅ 总览图: {overview_path}")
    print(f"✅ 逐层图: {out_dir}/frame_XX_layers.png (每帧1张合并)")
    print(f"✅ 单层图: {out_dir}/frame_XX/layer_YY.png (每帧每层独立)")
    print(f"\n可视化层: {layers}")
    print(f"列说明:")
    print(f"  Feature Heatmap  - Search Token均值激活热力图")
    print(f"  Attention Map     - Search区域自注意力热力图")
    if args.show_prompt:
        print(f"  Prompt Diff       - |有Prompt - 无Prompt| 差异热力图")


def run_comparison(args, experiments):
    """跨实验对比模式: 所有实验用同一输入，共享colorbar并排显示所有层"""
    import hashlib

    print(f"\n{'='*60}")
    print(f"[对比模式] {len(experiments)} 个实验 并排对比")
    print(f"{'='*60}")

    frames, gt_list = load_video(args.video_idx, max_frames=1)
    if not frames or len(frames) == 0:
        print("[对比] 无法读取视频帧")
        return

    frame_6ch = frames[0]
    init_bbox = gt_list[0] if gt_list and gt_list[0] else [
        frame_6ch.shape[1] // 4, frame_6ch.shape[0] // 4,
        frame_6ch.shape[1] // 2, frame_6ch.shape[0] // 2]

    n_exp = len(experiments)
    all_results = {}

    for i, (cfg_path, ckpt_path, exp_name) in enumerate(experiments):
        print(f"[对比] ({i+1}/{n_exp}) 加载: {exp_name}")
        try:
            cfg = edict(yaml.safe_load(open(cfg_path)))
            if 'TEST' not in cfg:
                cfg.TEST = edict()
            for k in ['TEMPLATE_FACTOR', 'TEMPLATE_SIZE', 'SEARCH_FACTOR', 'SEARCH_SIZE']:
                setattr(cfg.TEST, k, getattr(cfg.TEST, k,
                    [2.0, 128, 4.0, 256][['TEMPLATE_FACTOR', 'TEMPLATE_SIZE', 'SEARCH_FACTOR', 'SEARCH_SIZE'].index(k)]))

            model = build_viptrack(cfg, training=False)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            load_result = model.load_state_dict(ckpt.get('model', ckpt.get('net', ckpt)), strict=False)
            meta_prompt = getattr(cfg.MODEL.BACKBONE, 'META_PROMPT', False)
            model = model.cuda().eval()

            backbone = model.backbone
            depth = backbone.depth
            feat_sz_s = model.feat_sz_s
            lens_z = backbone.pos_embed_z.shape[1]
            lens_x = backbone.pos_embed_x.shape[1]
            cat_mode = backbone.cat_mode
            search_size = cfg.TEST.SEARCH_SIZE
            search_factor = cfg.TEST.SEARCH_FACTOR

            preprocessor = PreprocessorMM()
            x_patch_arr, _ = sample_target(frame_6ch, init_bbox, search_factor, output_sz=search_size)
            z_patch_arr, _ = sample_target(frame_6ch, init_bbox,
                                           getattr(cfg.TEST, 'TEMPLATE_FACTOR', 2.0),
                                           output_sz=getattr(cfg.TEST, 'TEMPLATE_SIZE', 128))
            x_tensor = preprocessor.process(x_patch_arr)
            z_tensor = preprocessor.process(z_patch_arr)

            hook = LayerHook()
            hook.register(model)
            with torch.no_grad():
                out_dict = model(z_tensor, x_tensor)

            features_with_prompt = dict(hook.features)
            inputs_with_prompt = dict(hook.inputs)
            global_indices_with_prompt = dict(hook.global_index_s)

            hook_no = LayerHook()
            hook_no.register(model)
            run_forward_no_prompt(model, z_tensor, x_tensor, hook_no)

            layer_data = {}
            for li in range(depth):
                feat_out = features_with_prompt.get(li)
                feat_in = inputs_with_prompt.get(li)
                feat_no = hook_no.features.get(li)
                gidx_out = global_indices_with_prompt.get(li)
                gidx_in = global_indices_with_prompt.get(li - 1) if li > 0 else None
                entry = {}
                if feat_out is not None:
                    st = extract_search_tokens(feat_out, lens_z, lens_x, cat_mode, feat_sz_s)
                    entry['output_raw'] = st[0].mean(dim=-1).detach().cpu().numpy()
                    entry['output_gidx'] = gidx_out
                if feat_in is not None:
                    st_in = extract_search_tokens(feat_in, lens_z, lens_x, cat_mode, feat_sz_s)
                    entry['input_raw'] = st_in[0].mean(dim=-1).detach().cpu().numpy()
                    entry['input_gidx'] = gidx_in
                if feat_no is not None:
                    st_no = extract_search_tokens(feat_no, lens_z, lens_x, cat_mode, feat_sz_s)
                    entry['noprompt_raw'] = st_no[0].mean(dim=-1).detach().cpu().numpy()
                    entry['noprompt_gidx'] = gidx_out
                if entry:
                    layer_data[li] = entry

            sm = out_dict.get('score_map')
            score_max = sm[0].max().item() if sm is not None else 0

            all_results[exp_name] = {
                'layer_data': layer_data,
                'depth': depth,
                'score_max': score_max,
                'meta_prompt': meta_prompt,
                'missing_keys': len(load_result.missing_keys),
                'feat_sz_s': feat_sz_s,
                'search_size': search_size
            }
            hook.remove()
            hook_no.remove()
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[对比] ❌ {exp_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) < 2:
        print("[对比] 成功加载的实验不足2个，无法对比")
        return

    rgb_orig = frame_6ch[:, :, :3]
    srch_rgb = cv2.resize(rgb_orig, (search_size, search_size))
    cmp_dir = os.path.join(_project_root, args.output_dir, '_compare')
    os.makedirs(cmp_dir, exist_ok=True)

    depth = list(all_results.values())[0]['depth']

    def _raw_to_hm(raw, fsz, ssz, g_lo, g_hi, gidx=None):
        h = w = fsz
        total_tokens = h * w
        feat_map = np.zeros(total_tokens, dtype=np.float32)
        if gidx is not None:
            idx = gidx.cpu().numpy().astype(np.int64) if isinstance(gidx, torch.Tensor) else np.array(gidx, dtype=np.int64)
            if idx.ndim > 1:
                idx = idx[0]
            if len(idx) == len(raw):
                valid_mask = (idx >= 0) & (idx < total_tokens)
                feat_map[idx[valid_mask]] = raw[valid_mask]
            else:
                feat_map[:min(len(raw), total_tokens)] = raw[:min(len(raw), total_tokens)]
        else:
            feat_map[:min(len(raw), total_tokens)] = raw[:min(len(raw), total_tokens)]
        feat_map = feat_map.reshape(h, w)
        clip_raw = np.clip(feat_map, g_lo, g_hi)
        normed = (clip_raw - g_lo) / (g_hi - g_lo + 1e-8)
        return cv2.resize(normed.astype(np.float32), (ssz, ssz))

    # 图1: 所有层输出对比 (每行1层, 每列1实验)
    for view_name, key in [('Output', 'output_raw'), ('Input', 'input_raw'), ('NoPrompt', 'noprompt_raw')]:
        global_vals = []
        for exp_name, res in all_results.items():
            for li in range(depth):
                ld = res['layer_data'].get(li, {})
                if key in ld:
                    global_vals.append(ld[key])
        if not global_vals:
            continue
        all_concat = np.concatenate(global_vals)
        g_lo, g_hi = np.percentile(all_concat, [1, 99])
        if g_hi - g_lo < 1e-8:
            g_lo, g_hi = all_concat.min(), all_concat.max()

        n_rows = depth
        n_cols = n_exp + 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row_i in range(depth):
            row_hms = []
            for col, (exp_name, res) in enumerate(all_results.items()):
                ax = axes[row_i, col]
                ld = res['layer_data'].get(row_i, {})
                raw = ld.get(key)
                gidx_key = key.replace('_raw', '_gidx')
                gidx = ld.get(gidx_key)
                if raw is not None:
                    hm = _raw_to_hm(raw, res['feat_sz_s'], res['search_size'], g_lo, g_hi, gidx=gidx)
                    row_hms.append(hm)
                    ax.imshow(draw_heatmap_overlay(srch_rgb, hm, alpha=0.55, vmin=g_lo, vmax=g_hi))
                    if row_i == 0:
                        tag = "META" if res['meta_prompt'] else "BASE"
                        miss = f" m{res['missing_keys']}" if res['missing_keys'] > 0 else ""
                        ax.set_title(f'{exp_name[:16]}\n{tag}{miss}', fontsize=8)
                else:
                    ax.imshow(srch_rgb.astype(np.uint8))
                    row_hms.append(None)
                if col == 0:
                    ax.set_ylabel(f'L{row_i}', fontsize=9, fontweight='bold')
                ax.axis('off')

            # 差值列: 各实验与第一个实验的像素差异
            ax_diff = axes[row_i, n_exp]
            ref_hm = row_hms[0]
            if ref_hm is not None and len([h for h in row_hms[1:] if h is not None]) > 0:
                diff_max = np.zeros_like(ref_hm)
                for hm in row_hms[1:]:
                    if hm is not None:
                        diff_max = np.maximum(diff_max, np.abs(hm - ref_hm))
                d_lo_local, d_hi_local = np.percentile(diff_max, [5, 95])
                if d_hi_local - d_lo_local < 1e-8:
                    d_lo_local, d_hi_local = 0, diff_max.max()
                diff_norm = np.clip((diff_max - d_lo_local) / (d_hi_local - d_lo_local + 1e-8), 0, 1)
                diff_color = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
                diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
                blended_diff = (0.6 * diff_color.astype(float) + 0.4 * srch_rgb.astype(float)).astype(np.uint8)
                ax_diff.imshow(blended_diff)
                max_val = diff_max.max()
                ax_diff.text(0.02, 0.96, f'max={max_val:.3f}', transform=ax_diff.transAxes,
                            fontsize=7, color='white',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
            else:
                ax_diff.imshow(srch_rgb.astype(np.uint8))
            if row_i == 0:
                ax_diff.set_title('Inter-Exp\nDiff', fontsize=8)
            ax_diff.axis('off')

            # colorbar列
            ax_cb = axes[row_i, n_exp + 1]
            sm_cb = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=g_lo, vmax=g_hi))
            sm_cb.set_array([])
            cbar = fig.colorbar(sm_cb, cax=ax_cb)

        plt.suptitle(f'Cross-Exp {view_name} | Shared [{g_lo:.3f},{g_hi:.3f}]',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fpath = os.path.join(cmp_dir, f'all_layers_{key}.png')
        plt.savefig(fpath, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"✅ 对比图: {fpath}")

    # 图2: Prompt差异对比 (每行1层, 每列1实验)
    diff_global = []
    for exp_name, res in all_results.items():
        for li in range(depth):
            ld = res['layer_data'].get(li, {})
            if 'output_raw' in ld and 'noprompt_raw' in ld:
                diff_global.append(np.abs(ld['output_raw'] - ld['noprompt_raw']))
    if diff_global:
        diff_concat = np.concatenate(diff_global)
        d_lo, d_hi = np.percentile(diff_concat, [1, 99])
        if d_hi - d_lo < 1e-8:
            d_lo, d_hi = diff_concat.min(), diff_concat.max()

        fig, axes = plt.subplots(depth, n_exp + 1, figsize=(3.5 * (n_exp + 1), 3 * depth))
        if depth == 1:
            axes = axes.reshape(1, -1)

        for row_i in range(depth):
            for col, (exp_name, res) in enumerate(all_results.items()):
                ax = axes[row_i, col]
                ld = res['layer_data'].get(row_i, {})
                if 'output_raw' in ld and 'noprompt_raw' in ld:
                    diff = np.abs(ld['output_raw'] - ld['noprompt_raw'])
                    gidx = ld.get('output_gidx')
                    hm = _raw_to_hm(diff, res['feat_sz_s'], res['search_size'], d_lo, d_hi, gidx=gidx)
                    diff_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
                    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
                    blended = (0.5 * diff_color.astype(float) + 0.5 * srch_rgb.astype(float)).astype(np.uint8)
                    ax.imshow(blended)
                    if row_i == 0:
                        ax.set_title(exp_name[:18], fontsize=8)
                else:
                    ax.imshow(srch_rgb.astype(np.uint8))
                if col == 0:
                    ax.set_ylabel(f'L{row_i}', fontsize=9, fontweight='bold')
                ax.axis('off')

            ax_cb = axes[row_i, n_exp]
            sm_cb = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=d_lo, vmax=d_hi))
            sm_cb.set_array([])
            cbar = fig.colorbar(sm_cb, cax=ax_cb)

        plt.suptitle(f'Prompt Injection Effect | |Output - NoPrompt| | Shared [{d_lo:.4f},{d_hi:.4f}]',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fpath = os.path.join(cmp_dir, 'all_layers_prompt_diff.png')
        plt.savefig(fpath, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"✅ 对比图: {fpath}")

    # 图3: 逐层统计折线图
    fig_stat, (ax_mean, ax_diff) = plt.subplots(1, 2, figsize=(14, 5))
    for exp_name, res in all_results.items():
        means, stds, diffs = [], [], []
        for li in range(depth):
            ld = res['layer_data'].get(li, {})
            if 'output_raw' in ld:
                means.append(ld['output_raw'].mean())
                stds.append(ld['output_raw'].std())
            else:
                means.append(0); stds.append(0)
            if 'output_raw' in ld and 'noprompt_raw' in ld:
                diffs.append(np.abs(ld['output_raw'] - ld['noprompt_raw']).mean())
            else:
                diffs.append(0)
        tag = "META" if res['meta_prompt'] else "BASE"
        ax_mean.plot(range(depth), means, 'o-', label=f'{exp_name[:15]}({tag})', markersize=4)
        ax_diff.plot(range(depth), diffs, 'o-', label=f'{exp_name[:15]}({tag})', markersize=4)
    ax_mean.set_xlabel('Layer'); ax_mean.set_ylabel('Mean Activation'); ax_mean.legend(fontsize=7)
    ax_mean.set_title('Per-Layer Output Mean'); ax_mean.grid(True, alpha=0.3)
    ax_diff.set_xlabel('Layer'); ax_diff.set_ylabel('|Prompt - NoPrompt| Mean'); ax_diff.legend(fontsize=7)
    ax_diff.set_title('Per-Layer Prompt Injection Effect'); ax_diff.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(cmp_dir, 'per_layer_stats.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 统计图: {fpath}")

    print(f"\n{'='*60}")
    print(f"[对比] 特征Hash汇总 (Layer6):")
    for exp_name, res in all_results.items():
        ld = res['layer_data'].get(6, {})
        if 'output_raw' in ld:
            h = hashlib.md5(ld['output_raw'].tobytes()).hexdigest()[:10]
            print(f"  {exp_name}: hash={h}, mean={ld['output_raw'].mean():.4f}, score={res['score_max']:.4f}")
    print(f"{'='*60}")


def main():
    args = parse_args()

    if args.batch:
        experiments = find_all_experiments()
        if not experiments:
            print("[批量] 未找到任何实验!")
            return
        print(f"[批量] 共找到 {len(experiments)} 个实验:")
        for i, (cfg_path, ckpt_path, exp_name) in enumerate(experiments):
            print(f"  [{i+1}] {exp_name}")
            print(f"       config: {cfg_path}")
            print(f"       ckpt:   {ckpt_path}")
        print()
        for i, (cfg_path, ckpt_path, exp_name) in enumerate(experiments):
            print(f"\n{'='*60}")
            print(f"[批量] ({i+1}/{len(experiments)}) 运行: {exp_name}")
            print(f"{'='*60}")
            try:
                run_single_experiment(args, cfg_path, ckpt_path, exp_name)
            except Exception as e:
                print(f"[批量] ❌ {exp_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        print(f"\n{'='*60}")
        print(f"[批量] 全部完成! 共 {len(experiments)} 个实验")
        print(f"{'='*60}")

        if args.compare:
            run_comparison(args, experiments)
    else:
        exp_name = os.path.basename(os.path.dirname(args.checkpoint))
        run_single_experiment(args, args.config, args.checkpoint, exp_name)


if __name__ == '__main__':
    main()
