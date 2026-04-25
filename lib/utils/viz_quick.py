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
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--video_idx', type=int, default=0)
    p.add_argument('--output_dir', default='viz_quick')
    p.add_argument('--num_frames', type=int, default=3)
    p.add_argument('--layers', default='all',
                   help='要可视化的层, 如 "all" "0,5,11" "0-3,11"')
    p.add_argument('--show_prompt', action='store_true',
                   help='显示Prompt注入前后的对比')
    return p.parse_args()


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


def token_to_heatmap(tokens, feat_sz, search_size=256):
    """将Search Token [B, N, C] 转为热力图 [H, W]"""
    feat = tokens[0].mean(dim=-1).detach().cpu().numpy()
    h = w = feat_sz
    if len(feat) < h * w:
        feat_padded = np.zeros(h * w)
        feat_padded[:len(feat)] = feat
        feat = feat_padded
    feat = feat[:h * w].reshape(h, w)
    feat_norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    feat_resized = cv2.resize(feat_norm.astype(np.float32), (search_size, search_size))
    return feat_resized


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
    diag_norm = (diag - diag.min()) / (diag.max() - diag.min() + 1e-8)
    return cv2.resize(diag_norm.astype(np.float32), (search_size, search_size))


class LayerHook:
    """Hook提取每层Transformer Block的中间特征和注意力"""

    def __init__(self):
        self.features = {}
        self.attns = {}
        self._handles = []

    def register(self, model):
        backbone = model.backbone
        for i, blk in enumerate(backbone.blocks):
            h1 = blk.register_forward_hook(self._make_feat_hook(i))
            self._handles.append(h1)
            h2 = blk.attn.register_forward_hook(self._make_attn_hook(i))
            self._handles.append(h2)

    def _make_feat_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.features[layer_idx] = output[0].detach()
            else:
                self.features[layer_idx] = output.detach()
        return hook

    def _make_attn_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                self.attns[layer_idx] = output[1].detach()
        return hook

    def clear(self):
        self.features.clear()
        self.attns.clear()

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
        if attn is not None:
            hook.attns[i] = attn.detach()

    return {}


def extract_search_tokens(feature, lens_z, lens_x, cat_mode, feat_sz):
    """从合并特征中提取Search Tokens"""
    if cat_mode == 'direct':
        search_tokens = feature[:, lens_z:lens_z + lens_x]
    elif cat_mode == 'template_central':
        central_pivot = lens_x // 2
        first_half = feature[:, :central_pivot, :]
        second_half = feature[:, central_pivot + lens_z:, :]
        search_tokens = torch.cat([first_half, second_half], dim=1)
    else:
        search_tokens = feature[:, lens_z:lens_z + lens_x]
    return search_tokens


def draw_heatmap_overlay(srch_rgb, heatmap, gt_box_in_search=None, score_max_pos=None):
    """绘制热力图叠加"""
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (0.4 * heatmap_color.astype(float) + 0.6 * srch_rgb.astype(float)).astype(np.uint8)
    return blended


def main():
    args = parse_args()
    cfg = edict(yaml.safe_load(open(args.config)))

    template_factor = cfg.TEST.TEMPLATE_FACTOR
    template_size = cfg.TEST.TEMPLATE_SIZE
    search_factor = cfg.TEST.SEARCH_FACTOR
    search_size = cfg.TEST.SEARCH_SIZE

    model = build_viptrack(cfg, training=False)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt.get('net', ckpt)), strict=False)
    model = model.cuda().eval()

    backbone = model.backbone
    depth = backbone.depth
    cat_mode = backbone.cat_mode
    feat_sz_s = model.feat_sz_s
    lens_z = backbone.pos_embed_z.shape[1]
    lens_x = backbone.pos_embed_x.shape[1]

    layers = parse_layers(args.layers, depth)
    print(f"[配置] depth={depth}, cat_mode={cat_mode}, feat_sz_s={feat_sz_s}")
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

    out_dir = os.path.join(args.output_dir, os.path.basename(args.config).replace('.yaml', ''))
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

        n_cols = 3 if args.show_prompt else 2
        n_rows = len(layers)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row_i, layer_idx in enumerate(layers):
            feat = features_prompt.get(layer_idx)
            attn = attns_prompt.get(layer_idx)

            # 提取Search Tokens
            search_tokens = None
            if feat is not None:
                search_tokens = extract_search_tokens(feat, lens_z, lens_x, cat_mode, feat_sz_s)

            # 列0: 特征热力图 (Token均值)
            ax = axes[row_i, 0]
            if search_tokens is not None:
                hm = token_to_heatmap(search_tokens, feat_sz_s, search_size)
                blended = draw_heatmap_overlay(srch_rgb, hm)
                ax.imshow(blended)
                ax.set_title(f'Layer {layer_idx} | Feature Heatmap', fontsize=11)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
                ax.set_title(f'Layer {layer_idx} | No Feature', fontsize=11)

            # 列1: 注意力热力图
            ax = axes[row_i, 1]
            if attn is not None:
                hm_attn = attn_to_heatmap(attn, lens_z, feat_sz_s, search_size)
                if hm_attn is not None:
                    blended = draw_heatmap_overlay(srch_rgb, hm_attn)
                    ax.imshow(blended)
                    ax.set_title(f'Layer {layer_idx} | Attention Map', fontsize=11)
                else:
                    ax.imshow(srch_rgb.astype(np.uint8))
                    ax.set_title(f'Layer {layer_idx} | Attn N/A', fontsize=11)
            else:
                ax.imshow(srch_rgb.astype(np.uint8))
                ax.set_title(f'Layer {layer_idx} | No Attn', fontsize=11)

            # 列2: Prompt前后对比 (可选)
            if args.show_prompt and n_cols >= 3:
                ax = axes[row_i, 2]
                feat_no = features_no_prompt.get(layer_idx) if features_no_prompt else None
                if feat_no is not None and search_tokens is not None:
                    search_tokens_no = extract_search_tokens(feat_no, lens_z, lens_x, cat_mode, feat_sz_s)
                    hm_prompt = token_to_heatmap(search_tokens, feat_sz_s, search_size)
                    hm_no_prompt = token_to_heatmap(search_tokens_no, feat_sz_s, search_size)
                    diff = np.abs(hm_prompt - hm_no_prompt)
                    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                    diff_color = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
                    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
                    blended_diff = (0.5 * diff_color.astype(float) + 0.5 * srch_rgb.astype(float)).astype(np.uint8)
                    ax.imshow(blended_diff)
                    ax.set_title(f'Layer {layer_idx} | Prompt Diff', fontsize=11)
                else:
                    ax.imshow(srch_rgb.astype(np.uint8))
                    ax.set_title(f'Layer {layer_idx} | No Diff', fontsize=11)

            for ax in axes[row_i]:
                ax.axis('off')

        col_labels = ['Feature Heatmap', 'Attention Map']
        if args.show_prompt:
            col_labels.append('|Prompt-RGB| Diff')
        sm_max_str = f'{sm_final.max():.3f}' if sm_final is not None else 'N/A'
        plt.suptitle(
            f'Frame {frame_i} | {os.path.basename(args.config)}\n'
            f'Layers: {layers} | Score_max={sm_max_str}',
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        fpath = os.path.join(out_dir, f'frame_{frame_i:02d}_layers.png')
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Frame {frame_i}: {fpath}")

        # ====== 单层独立保存 ======
        layer_dir = os.path.join(out_dir, f'frame_{frame_i:02d}')
        os.makedirs(layer_dir, exist_ok=True)

        for layer_idx in layers:
            feat = features_prompt.get(layer_idx)
            attn = attns_prompt.get(layer_idx)

            fig_l, axes_l = plt.subplots(1, 2 + (1 if args.show_prompt else 0),
                                          figsize=(7 * (2 + (1 if args.show_prompt else 0)), 6))

            # Feature Heatmap
            ax = axes_l[0]
            if feat is not None:
                search_tokens = extract_search_tokens(feat, lens_z, lens_x, cat_mode, feat_sz_s)
                hm = token_to_heatmap(search_tokens, feat_sz_s, search_size)
                blended = draw_heatmap_overlay(srch_rgb, hm)
                ax.imshow(blended)
            ax.set_title(f'Feature Heatmap', fontsize=12)
            ax.axis('off')

            # Attention Map
            ax = axes_l[1]
            if attn is not None:
                hm_attn = attn_to_heatmap(attn, lens_z, feat_sz_s, search_size)
                if hm_attn is not None:
                    blended = draw_heatmap_overlay(srch_rgb, hm_attn)
                    ax.imshow(blended)
            ax.set_title(f'Attention Map', fontsize=12)
            ax.axis('off')

            # Prompt Diff
            if args.show_prompt:
                ax = axes_l[2]
                feat_no = features_no_prompt.get(layer_idx) if features_no_prompt else None
                if feat_no is not None and feat is not None:
                    st_prompt = extract_search_tokens(feat, lens_z, lens_x, cat_mode, feat_sz_s)
                    st_no = extract_search_tokens(feat_no, lens_z, lens_x, cat_mode, feat_sz_s)
                    hm_p = token_to_heatmap(st_prompt, feat_sz_s, search_size)
                    hm_n = token_to_heatmap(st_no, feat_sz_s, search_size)
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

    plt.suptitle(f'{os.path.basename(args.config)} | Video {args.video_idx} | Overview',
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


if __name__ == '__main__':
    main()
