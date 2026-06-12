import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval_checkpoint_confusion_matrix import load_checkpoint
from train_all_baselines_config import (
    SlowFastDualEndAttentionWrapper,
    build_model,
    load_config,
    make_dual_end_mask,
    pack_slowfast_pathway,
)


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def find_frame_paths(frame_dir: Path):
    frames = []
    for ext in IMAGE_EXTS:
        frames.extend(frame_dir.glob(f"*{ext}"))
    return sorted(frames)


def sample_indices(n, num_frames, sampling, frame_stride, train=False):
    if n <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    if sampling == "uniform":
        if n == 1:
            return np.zeros(num_frames, dtype=np.int64)
        return np.linspace(0, n - 1, num_frames).round().astype(np.int64)

    span = (num_frames - 1) * frame_stride + 1
    if n >= span:
        start = 0 if train else (n - span) // 2
        return np.asarray([start + i * frame_stride for i in range(num_frames)], dtype=np.int64)

    indices = np.arange(0, n, frame_stride, dtype=np.int64)
    if len(indices) == 0:
        indices = np.asarray([0], dtype=np.int64)
    if len(indices) < num_frames:
        pad = np.full(num_frames - len(indices), indices[-1], dtype=np.int64)
        indices = np.concatenate([indices, pad], axis=0)
    return indices[:num_frames]


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(1, min(width, int(round(x2))))
    y2 = max(1, min(height, int(round(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def parse_tracks(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    tracks = []
    for track in root.findall("track"):
        boxes = {}
        for box in track.findall("box"):
            frame_idx = int(box.get("frame", "-1"))
            outside = int(box.get("outside", "0"))
            if outside == 1:
                continue
            boxes[frame_idx] = (
                float(box.get("xtl")),
                float(box.get("ytl")),
                float(box.get("xbr")),
                float(box.get("ybr")),
            )
        if boxes:
            tracks.append(
                {
                    "track_id": str(track.get("id", "")),
                    "label": str(track.get("label", "")),
                    "boxes": boxes,
                }
            )
    return tracks


def denormalize(video):
    mean = torch.tensor([0.485, 0.456, 0.406], device=video.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=video.device).view(1, 3, 1, 1)
    return (video * std + mean).clamp(0, 1)


def forward_for_cam(model, pathways, attention_masks=None, target_stage=4):
    if not isinstance(model, SlowFastDualEndAttentionWrapper):
        raise TypeError("This script expects a SlowFastDualEndAttentionWrapper model.")
    x = pathways
    captured = None
    for idx, block in enumerate(model.model.blocks):
        x = block(x)
        if attention_masks is not None and idx in model.stages:
            x = model.apply_dual_end_attention(x, attention_masks)
        if idx == target_stage:
            captured = [feat for feat in x]
            for feat in captured:
                feat.retain_grad()
    return x, captured


def pathway_cam(feature):
    grad = feature.grad
    if grad is None:
        return None
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * feature).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam_min = cam.amin(dim=(2, 3, 4), keepdim=True)
    cam_max = cam.amax(dim=(2, 3, 4), keepdim=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
    return cam


def make_overlay(rgb, heatmap, alpha=0.45):
    rgb_u8 = (rgb * 255).astype(np.uint8)
    heat_u8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_u8, 1.0 - alpha, color, alpha, 0)


def save_grid(images, path, cols=4, pad=8):
    if not images:
        return
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / cols))
    canvas = np.full((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), 255, dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        canvas[y:y + h, x:x + w] = img
    Image.fromarray(canvas).save(path)


def imread_rgb(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(buf.tobytes())


def build_track_clip(frame_paths, sampled_indices, track_boxes, image_size):
    imgs = []
    masks = []
    full_frames = []
    crop_boxes = []
    for src_idx in sampled_indices:
        frame_path = frame_paths[int(src_idx)]
        img = Image.open(frame_path).convert("RGB")
        full_rgb = np.asarray(img)
        h, w = full_rgb.shape[:2]
        box = track_boxes.get(int(src_idx))
        if box is None:
            prev = [k for k in track_boxes.keys() if k <= int(src_idx)]
            nxt = [k for k in track_boxes.keys() if k >= int(src_idx)]
            if prev:
                box = track_boxes[max(prev)]
            elif nxt:
                box = track_boxes[min(nxt)]
            else:
                raise RuntimeError(f"No boxes available for sampled frame {src_idx}")
        x1, y1, x2, y2 = clamp_box(*box, w, h)
        crop = img.crop((x1, y1, x2, y2))
        mask = make_dual_end_mask(crop, end_ratio=1.0 / 3.0)
        crop = crop.resize((image_size, image_size), Image.BILINEAR)
        mask = mask.resize((image_size, image_size), Image.NEAREST)

        img_t = torch.from_numpy(np.asarray(crop).transpose(2, 0, 1)).float() / 255.0
        img_t = torch.stack(
            [
                (img_t[0] - 0.485) / 0.229,
                (img_t[1] - 0.456) / 0.224,
                (img_t[2] - 0.406) / 0.225,
            ],
            dim=0,
        )
        mask_t = torch.from_numpy(np.asarray(mask)[None]).float() / 255.0

        imgs.append(img_t)
        masks.append(mask_t)
        full_frames.append(full_rgb)
        crop_boxes.append((x1, y1, x2, y2))
    return torch.stack(imgs, dim=0), torch.stack(masks, dim=0), full_frames, crop_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--xml_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", choices=["true", "pred"], default="true")
    parser.add_argument("--frames_per_grid", type=int, default=8)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    ckpt = load_checkpoint(checkpoint_path)
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = SimpleNamespace(**ckpt["config"])
        default_cfg = load_config(None)
        for k, v in vars(default_cfg).items():
            if not hasattr(cfg, k) or getattr(cfg, k) is None:
                setattr(cfg, k, v)
    cfg.pretrained = False
    class_names = list(ckpt.get("class_names") or cfg.class_names)
    cfg.class_names = class_names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(class_names)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    frame_dir = Path(args.frame_dir)
    xml_path = Path(args.xml_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = find_frame_paths(frame_dir)
    sampled_indices = sample_indices(len(frame_paths), int(cfg.num_frames), cfg.sampling, int(cfg.frame_stride), train=False)
    tracks = parse_tracks(xml_path)

    print(f"Device: {device}", flush=True)
    print(f"Frames: {len(frame_paths)}", flush=True)
    print(f"Sampling: {cfg.sampling}, num_frames={cfg.num_frames}, stride={cfg.frame_stride}", flush=True)
    print(f"Tracks: {len(tracks)}", flush=True)

    for track in tracks:
        video, masks, full_frames, crop_boxes = build_track_clip(
            frame_paths,
            sampled_indices,
            track["boxes"],
            int(cfg.image_size),
        )
        videos = video.unsqueeze(0).to(device)
        masks = masks.unsqueeze(0).to(device)
        clips = videos.permute(0, 2, 1, 3, 4)
        pathways = pack_slowfast_pathway(clips, alpha=getattr(cfg, "slowfast_alpha", 4))

        model.zero_grad(set_to_none=True)
        logits, captured = forward_for_cam(model, pathways, attention_masks=masks, target_stage=4)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        true_idx = class_names.index(track["label"]) if track["label"] in class_names else pred_idx
        target_idx = true_idx if args.target == "true" else pred_idx
        logits[0, target_idx].backward()

        cams = []
        for feature in captured:
            cam = pathway_cam(feature)
            if cam is not None:
                cams.append(cam)
        if not cams:
            continue
        cam_ups = []
        for cam in cams:
            cam_ups.append(
                F.interpolate(
                    cam,
                    size=(videos.shape[1], videos.shape[-2], videos.shape[-1]),
                    mode="trilinear",
                    align_corners=False,
                )
            )
        cam_full = torch.stack(cam_ups, dim=0).mean(dim=0)[0, 0].detach().cpu().numpy()
        rgb_video = denormalize(videos[0]).permute(0, 2, 3, 1).detach().cpu().numpy()

        track_dir = output_dir / f"track_{track['track_id']}_{track['label']}"
        track_dir.mkdir(parents=True, exist_ok=True)
        overlays = []
        crop_overlays = []
        for i, src_idx in enumerate(sampled_indices[: args.frames_per_grid]):
            x1, y1, x2, y2 = crop_boxes[i]
            full_frame = full_frames[i].copy()
            cam_crop = cv2.resize(cam_full[i], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            crop_rgb = full_frame[y1:y2, x1:x2]
            crop_overlay = make_overlay(rgb_video[i], cam_full[i], alpha=0.45)
            pasted = make_overlay(crop_rgb / 255.0, cam_crop, alpha=0.45)
            full_frame[y1:y2, x1:x2] = pasted
            cv2.rectangle(full_frame, (x1, y1), (x2 - 1, y2 - 1), color=(36, 92, 180), thickness=2)
            save_rgb(track_dir / f"frame_{int(src_idx):06d}_full_overlay.png", full_frame)
            Image.fromarray(crop_overlay).save(track_dir / f"frame_{int(src_idx):06d}_crop_overlay.png")
            overlays.append(full_frame)
            crop_overlays.append(crop_overlay)

        save_grid(overlays, track_dir / "full_frame_cam_grid.png", cols=4)
        save_grid(crop_overlays, track_dir / "crop_cam_grid.png", cols=4)

        report = [
            f"track_id={track['track_id']}",
            f"label={track['label']}",
            f"pred={class_names[pred_idx]}",
            f"target_for_cam={class_names[target_idx]}",
            f"sampled_indices={sampled_indices.tolist()}",
        ]
        for i, name in enumerate(class_names):
            report.append(f"p_{name}={float(probs[0, i].detach().cpu()):.6f}")
        (track_dir / "report.txt").write_text("\n".join(report), encoding="utf-8")

    print(f"Saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
