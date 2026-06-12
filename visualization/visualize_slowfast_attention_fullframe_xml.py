import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from eval_checkpoint_confusion_matrix import load_checkpoint
from train_all_baselines_config import (
    SlowFastDualEndAttentionWrapper,
    SlowFastTemporalDifferenceAttentionClassifier,
    load_config,
    build_model,
    pack_slowfast_pathway,
    make_dual_end_mask,
)
from visualize_slowfast_desa_attention import pathway_cam, save_grid


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def list_frames(folder: Path):
    frames = []
    for ext in IMAGE_EXTS:
        frames.extend(folder.glob(f"*{ext}"))
    return sorted(frames)


def sample_indices(n, num_frames, sampling="uniform", frame_stride=1):
    if n <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    if sampling == "uniform":
        if n == 1:
            return np.zeros(num_frames, dtype=np.int64)
        return np.linspace(0, n - 1, num_frames).round().astype(np.int64)

    span = (num_frames - 1) * frame_stride + 1
    if n >= span:
        start = (n - span) // 2
        return np.asarray([start + i * frame_stride for i in range(num_frames)], dtype=np.int64)

    indices = np.arange(0, n, frame_stride, dtype=np.int64)
    if len(indices) == 0:
        indices = np.asarray([0], dtype=np.int64)
    if len(indices) < num_frames:
        pad = np.full(num_frames - len(indices), indices[-1], dtype=np.int64)
        indices = np.concatenate([indices, pad], axis=0)
    return indices[:num_frames]


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


def load_tracks(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    tracks = []
    for track in root.findall("track"):
        item = {"track_id": track.get("id", ""), "label": track.get("label", ""), "boxes": {}}
        for box in track.findall("box"):
            frame_idx = int(box.get("frame", "-1"))
            if box.get("outside", "0") == "1":
                continue
            item["boxes"][frame_idx] = (
                float(box.get("xtl")),
                float(box.get("ytl")),
                float(box.get("xbr")),
                float(box.get("ybr")),
            )
        if item["boxes"]:
            tracks.append(item)
    return tracks


def make_overlay(rgb, heatmap, alpha=0.42):
    rgb_u8 = rgb.astype(np.uint8)
    heat_u8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_u8, 1.0 - alpha, color, alpha, 0)


def forward_for_cam_generic(model, pathways, attention_masks=None, target_stage=4):
    if isinstance(model, SlowFastTemporalDifferenceAttentionClassifier):
        x = pathways
        captured = None
        focus_losses = []
        model.last_focus_loss = None
        for idx, block in enumerate(model.model.blocks[:5]):
            x = block(x)
            if model.use_dual_end_attention and idx in model.attention_stages:
                x = model.apply_dual_end_attention(x, attention_masks)
            if model.use_focus_loss and attention_masks is not None and idx in model.focus_stages:
                focus_loss = model.compute_dual_end_focus_loss(x, attention_masks)
                if focus_loss is not None:
                    focus_losses.append(focus_loss)
            if idx == target_stage:
                captured = [feat for feat in x]
                for feat in captured:
                    feat.retain_grad()
        if focus_losses:
            model.last_focus_loss = torch.stack(focus_losses).mean()

        global_sequence = model.feature_norm(model.pathway_to_sequence(x))
        if model.use_dual_end_tda:
            tda_sequence = model.feature_norm(model.pathway_to_dual_end_sequence(x, attention_masks))
        else:
            tda_sequence = global_sequence

        if tda_sequence.shape[1] <= 1:
            pooled = global_sequence.mean(dim=1)
        else:
            diff = torch.zeros_like(tda_sequence)
            diff[:, 1:] = torch.abs(tda_sequence[:, 1:] - tda_sequence[:, :-1])
            diff[:, 0] = diff[:, 1]
            diff_logits = model.diff_score(diff)
            weights = torch.softmax(diff_logits, dim=1)
            attended = (tda_sequence * weights).sum(dim=1)
            pooled = global_sequence.mean(dim=1) + model.tda_alpha * attended

        logits = model.classifier(model.dropout(model.output_norm(pooled)))
        return logits, captured

    if isinstance(model, SlowFastDualEndAttentionWrapper):
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

    if hasattr(model, "blocks"):
        x = pathways
        captured = None
        for idx, block in enumerate(model.blocks):
            x = block(x)
            if idx == target_stage:
                captured = [feat for feat in x]
                for feat in captured:
                    feat.retain_grad()
        return x, captured

    raise TypeError("This script currently supports PyTorchVideo SlowFast models only.")


def build_track_clip(track, frame_paths, indices, image_size, normalize, tail_end_ratio):
    imgs = []
    masks = []
    boxes = []
    src_frames = []
    for i in indices:
        frame_idx = int(i)
        box = track["boxes"].get(frame_idx)
        if box is None:
            return None
        rgb = Image.open(frame_paths[frame_idx]).convert("RGB")
        src_np = np.asarray(rgb)
        h, w = src_np.shape[:2]
        x1, y1, x2, y2 = clamp_box(*box, w, h)
        crop = rgb.crop((x1, y1, x2, y2))
        mask = make_dual_end_mask(crop, end_ratio=tail_end_ratio)
        crop = transforms.Resize((image_size, image_size))(crop)
        mask = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        imgs.append(normalize(transforms.functional.to_tensor(crop)))
        masks.append(transforms.functional.to_tensor(mask))
        boxes.append((x1, y1, x2, y2))
        src_frames.append(src_np)
    return torch.stack(imgs, dim=0), torch.stack(masks, dim=0), boxes, src_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--xml_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", choices=["true", "pred"], default="true")
    parser.add_argument("--frames_per_grid", type=int, default=8)
    parser.add_argument("--only_frame_idx", type=int, default=None)
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

    class_names = list(ckpt.get("class_names") or cfg.class_names)
    cfg.class_names = class_names
    cfg.pretrained = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_dir = Path(args.frame_dir)
    xml_path = Path(args.xml_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}", flush=True)
    print(f"Frame dir: {frame_dir}", flush=True)
    print(f"XML: {xml_path}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Classes: {class_names}", flush=True)

    frame_paths = list_frames(frame_dir)
    indices = sample_indices(len(frame_paths), cfg.num_frames, cfg.sampling, cfg.frame_stride)
    sampled_set = {int(i) for i in indices.tolist()}
    if args.only_frame_idx is not None and args.only_frame_idx not in sampled_set:
        print(f"Warning: frame {args.only_frame_idx} is not in the sampled clip indices {indices.tolist()}.", flush=True)

    tracks = [t for t in load_tracks(xml_path) if t["label"] in class_names]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = build_model(cfg, len(class_names)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    full_frame_heatmaps = {}
    full_frame_preds = {}
    frame_base_images = {}
    for frame_idx in indices.tolist():
        rgb = imread_rgb(frame_paths[int(frame_idx)])
        frame_base_images[int(frame_idx)] = rgb
        full_frame_heatmaps[int(frame_idx)] = np.zeros(rgb.shape[:2], dtype=np.float32)
        full_frame_preds[int(frame_idx)] = []

    for track in tracks:
        built = build_track_clip(
            track,
            frame_paths,
            indices,
            cfg.image_size,
            normalize,
            getattr(cfg, "tail_end_ratio", 1.0 / 3.0),
        )
        if built is None:
            continue
        video, masks, boxes, _ = built
        videos = video.unsqueeze(0).to(device)
        masks = masks.unsqueeze(0).to(device)
        clips = videos.permute(0, 2, 1, 3, 4)
        pathways = pack_slowfast_pathway(clips, alpha=getattr(cfg, "slowfast_alpha", 4))

        model.zero_grad(set_to_none=True)
        logits, captured = forward_for_cam_generic(model, pathways, attention_masks=masks, target_stage=4)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        true_idx = class_names.index(track["label"])
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

        for t, frame_idx in enumerate(indices.tolist()):
            x1, y1, x2, y2 = boxes[t]
            crop_heat = cam_full[t]
            resized = cv2.resize(crop_heat, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            full_frame_heatmaps[int(frame_idx)][y1:y2, x1:x2] = np.maximum(
                full_frame_heatmaps[int(frame_idx)][y1:y2, x1:x2],
                resized,
            )
            full_frame_preds[int(frame_idx)].append(
                {
                    "track_id": track["track_id"],
                    "true": track["label"],
                    "pred": class_names[pred_idx],
                    "prob": float(probs[0, pred_idx].detach().cpu()),
                    "box": (x1, y1, x2, y2),
                }
            )

    selected_indices = indices.tolist()
    if args.only_frame_idx is not None:
        selected_indices = [i for i in selected_indices if int(i) == int(args.only_frame_idx)]

    overlays = []
    rows = []
    for frame_idx in selected_indices[: args.frames_per_grid] if args.only_frame_idx is None else selected_indices:
        rgb = frame_base_images[int(frame_idx)]
        heat = full_frame_heatmaps[int(frame_idx)]
        overlay = make_overlay(rgb, heat, alpha=0.42)
        for item in full_frame_preds[int(frame_idx)]:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (235, 235, 235), 1)
        out_path = output_dir / f"frame_{int(frame_idx):06d}_full_attention.png"
        save_rgb(out_path, overlay)
        overlays.append(overlay)
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "output": str(out_path),
                "tracks": len(full_frame_preds[int(frame_idx)]),
            }
        )

    if overlays:
        save_grid(overlays, output_dir / "full_attention_grid.png", cols=4)

    csv_path = output_dir / "full_attention_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "output", "tracks"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sampled indices: {indices.tolist()}", flush=True)
    print(f"Tracks used: {len(tracks)}", flush=True)
    print(f"Saved index: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
