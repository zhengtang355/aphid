import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_checkpoint_confusion_matrix import load_checkpoint
from train_all_baselines_config import (
    CroppedInstanceDataset,
    SlowFastDualEndAttentionWrapper,
    SlowFastTemporalDifferenceAttentionClassifier,
    build_model,
    load_config,
    pack_slowfast_pathway,
)


def denormalize(video):
    mean = torch.tensor([0.485, 0.456, 0.406], device=video.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=video.device).view(1, 3, 1, 1)
    return (video * std + mean).clamp(0, 1)


def enhance_heatmap(heatmap, percentile=99.0, gamma=1.0):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(np.min(heatmap))
    high = float(np.percentile(heatmap, percentile))
    if high <= 1e-8:
        high = float(np.max(heatmap))
    heatmap = heatmap / max(high, 1e-8)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    if gamma != 1.0:
        heatmap = np.power(heatmap, gamma)
    return heatmap


def make_overlay(rgb, heatmap, mask=None, alpha=0.45, percentile=99.0, gamma=1.0):
    rgb_u8 = (rgb * 255).astype(np.uint8)
    heatmap = enhance_heatmap(heatmap, percentile=percentile, gamma=gamma)
    heat_u8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_u8, 1.0 - alpha, color, alpha, 0)

    if mask is not None:
        mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        cv2.drawContours(overlay, contours, -1, (220, 30, 30), 1)
    return overlay


def save_grid(images, path, cols=4, pad=8):
    if not images:
        return
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    canvas = np.full((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), 255, dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        canvas[y : y + h, x : x + w] = img
    Image.fromarray(canvas).save(path)


def forward_for_cam(model, pathways, attention_masks=None, target_stage=4):
    if isinstance(model, SlowFastTemporalDifferenceAttentionClassifier):
        x = pathways
        captured = None
        for idx, block in enumerate(model.model.blocks[:5]):
            x = block(x)
            if model.use_dual_end_attention and idx in model.attention_stages:
                x = model.apply_dual_end_attention(x, attention_masks)
            if idx == target_stage:
                captured = [feat for feat in x]
                for feat in captured:
                    feat.retain_grad()

        global_sequence = model.feature_norm(model.pathway_to_sequence(x))
        tda_sequence = (
            model.feature_norm(model.pathway_to_dual_end_sequence(x, attention_masks))
            if getattr(model, "use_dual_end_tda", False)
            else global_sequence
        )
        if tda_sequence.shape[1] <= 1:
            pooled = global_sequence.mean(dim=1)
            logits = model.classifier(model.dropout(model.output_norm(pooled)))
            return logits, captured

        diff = torch.zeros_like(tda_sequence)
        diff[:, 1:] = torch.abs(tda_sequence[:, 1:] - tda_sequence[:, :-1])
        diff[:, 0] = diff[:, 1]
        diff_logits = model.diff_score(diff)
        weights = torch.softmax(diff_logits, dim=1)
        attended = (tda_sequence * weights).sum(dim=1)
        pooled = global_sequence.mean(dim=1) + model.tda_alpha * attended
        logits = model.classifier(model.dropout(model.output_norm(pooled)))
        return logits, captured

    if not isinstance(model, SlowFastDualEndAttentionWrapper):
        x = pathways
        captured = None
        for idx, block in enumerate(model.blocks):
            x = block(x)
            if idx == target_stage:
                captured = [feat for feat in x]
                for feat in captured:
                    feat.retain_grad()
        return x, captured

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


def sample_time_indices(num_frames, out_frames):
    if out_frames >= num_frames:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, out_frames).round().astype(int).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--target", choices=["true", "pred"], default="true")
    parser.add_argument("--max_samples_per_class", type=int, default=12)
    parser.add_argument("--frames_per_sample", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--show_dual_end_mask", action="store_true")
    parser.add_argument("--heatmap_alpha", type=float, default=0.45)
    parser.add_argument("--heatmap_percentile", type=float, default=99.0)
    parser.add_argument("--heatmap_gamma", type=float, default=1.0)
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
    val_dir = Path(args.val_dir) if args.val_dir else Path(cfg.val_dir) if getattr(cfg, "val_dir", None) else Path(cfg.data_root) / "val"
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "val_attention_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Val dir: {val_dir}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Classes: {class_names}", flush=True)

    dataset = CroppedInstanceDataset(
        val_dir,
        class_names,
        cfg.num_frames,
        cfg.image_size,
        cfg.sampling,
        cfg.frame_stride,
        train=False,
        class_map=getattr(cfg, "class_map", None),
        return_attention_masks=True,
        tail_end_ratio=getattr(cfg, "tail_end_ratio", 1.0 / 3.0),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg, len(class_names)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    counts = {name: 0 for name in class_names}
    rows = []

    for batch in tqdm(loader, desc="Visualize"):
        videos, masks, labels, names = batch
        if videos.shape[0] != 1:
            raise ValueError("Use batch_size=1 for Grad-CAM visualization.")
        true_idx = int(labels.item())
        true_name = class_names[true_idx]
        if counts[true_name] >= args.max_samples_per_class:
            continue

        videos = videos.to(device)
        masks = masks.to(device)
        clips = videos.permute(0, 2, 1, 3, 4)
        pathways = pack_slowfast_pathway(clips, alpha=getattr(cfg, "slowfast_alpha", 4))

        model.zero_grad(set_to_none=True)
        logits, captured = forward_for_cam(model, pathways, attention_masks=masks, target_stage=4)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        target_idx = true_idx if args.target == "true" else pred_idx
        score = logits[0, target_idx]
        score.backward()

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
        mask_video = masks[0, :, 0].detach().cpu().numpy()

        frame_ids = sample_time_indices(videos.shape[1], args.frames_per_sample)
        overlays = []
        sample_dir = output_dir / true_name / str(names[0])
        sample_dir.mkdir(parents=True, exist_ok=True)
        for idx in frame_ids:
            overlay = make_overlay(
                rgb_video[idx],
                cam_full[idx],
                mask=mask_video[idx] if args.show_dual_end_mask else None,
                alpha=args.heatmap_alpha,
                percentile=args.heatmap_percentile,
                gamma=args.heatmap_gamma,
            )
            overlays.append(overlay)
            Image.fromarray(overlay).save(sample_dir / f"frame_{idx:03d}_cam_overlay.png")
        save_grid(overlays, sample_dir / "cam_grid.png", cols=4)

        if args.show_dual_end_mask:
            raw_overlays = []
            for idx in frame_ids:
                raw_mask = make_overlay(
                    rgb_video[idx],
                    np.zeros_like(cam_full[idx]),
                    mask=mask_video[idx],
                    alpha=0.0,
                    percentile=args.heatmap_percentile,
                    gamma=args.heatmap_gamma,
                )
                raw_overlays.append(raw_mask)
            save_grid(raw_overlays, sample_dir / "desa_mask_grid.png", cols=4)

        rows.append(
            {
                "sample": str(names[0]),
                "true": true_name,
                "pred": class_names[pred_idx],
                "target_for_cam": class_names[target_idx],
                **{f"p_{name}": float(probs[0, i].detach().cpu()) for i, name in enumerate(class_names)},
                "cam_grid": str(sample_dir / "cam_grid.png"),
                "desa_mask_grid": str(sample_dir / "desa_mask_grid.png") if args.show_dual_end_mask else "",
            }
        )
        counts[true_name] += 1

        if all(counts[name] >= args.max_samples_per_class for name in class_names):
            break

    csv_path = output_dir / "attention_visualization_index.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved samples per class: {counts}", flush=True)
    print(f"Saved index: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
