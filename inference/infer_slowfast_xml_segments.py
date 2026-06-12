import argparse
import csv
import gc
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eval_checkpoint_confusion_matrix import load_checkpoint
from train_all_baselines_config import build_model, load_config, make_dual_end_mask, model_forward


DISPLAY_LABELS = {
    "nobirth": "non-parturition",
    "start": "parturition onset",
    "birth": "ongoing parturition",
}


def format_label(label):
    return DISPLAY_LABELS.get(label, label)


def parse_durations(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_segments(total_frames, fps, mode, durations, window_seconds, stride_seconds):
    if mode == "fixed":
        segment_ranges = []
        start = 0
        for sec in durations:
            end = min(total_frames, start + int(round(sec * fps)))
            if end > start:
                segment_ranges.append((start, end))
            start = end
            if start >= total_frames:
                break
        return segment_ranges

    window = max(1, int(round(window_seconds * fps)))
    stride = max(1, int(round(stride_seconds * fps)))
    if total_frames <= window:
        return [(0, total_frames)]

    segment_ranges = []
    start = 0
    while start + window <= total_frames:
        segment_ranges.append((start, start + window))
        start += stride
    if segment_ranges[-1][1] < total_frames:
        last_start = max(0, total_frames - window)
        if last_start > segment_ranges[-1][0]:
            segment_ranges.append((last_start, total_frames))
    return segment_ranges


def load_tracks(xml_path):
    root = ET.parse(xml_path).getroot()
    tracks = {}
    for track in root.findall("track"):
        tid = int(track.get("id", len(tracks)))
        boxes = {}
        for box in track.findall("box"):
            if box.get("outside", "0") == "1":
                continue
            frame_idx = int(box.get("frame", "-1"))
            boxes[frame_idx] = (
                float(box.get("xtl")),
                float(box.get("ytl")),
                float(box.get("xbr")),
                float(box.get("ybr")),
            )
        if boxes:
            tracks[tid] = boxes
    return tracks


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(1, min(width, int(round(x2))))
    y2 = max(1, min(height, int(round(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def sample_uniform(items, num_frames):
    if not items:
        return []
    if len(items) == 1:
        return [items[0]] * num_frames
    indices = np.linspace(0, len(items) - 1, num_frames).round().astype(int)
    return [items[int(i)] for i in indices]


def read_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")
    return frames, fps, width, height


def build_clip(frames, sampled_items, image_size, normalize, need_masks, tail_end_ratio):
    imgs = []
    masks = []
    for item in sampled_items:
        frame = frames[item["frame_idx"]]
        x1, y1, x2, y2 = item["bbox"]
        crop_bgr = frame[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            crop_bgr = frame[max(0, y1) : max(1, y2), max(0, x1) : max(1, x2)]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)
        if need_masks:
            mask = make_dual_end_mask(pil, end_ratio=tail_end_ratio)
            mask = transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            )(mask)
            masks.append(transforms.functional.to_tensor(mask))
        pil = transforms.Resize((image_size, image_size))(pil)
        imgs.append(normalize(transforms.functional.to_tensor(pil)))
    video = torch.stack(imgs, dim=0)
    if need_masks:
        return video, torch.stack(masks, dim=0)
    return video, None


def draw_label(frame, bbox, label, score, track_id):
    colors = {
        "nobirth": (90, 190, 255),
        "start": (70, 220, 120),
        "birth": (80, 80, 255),
    }
    color = colors.get(label, (255, 210, 80))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"id{track_id}: {format_label(label)} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty1 = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, ty1), (min(frame.shape[1] - 1, x1 + tw + 6), y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 3, max(th + 2, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["fixed", "sliding"], default="fixed")
    parser.add_argument("--durations", default="15,20,15")
    parser.add_argument("--window_seconds", type=float, default=15.0)
    parser.add_argument("--stride_seconds", type=float, default=5.0)
    parser.add_argument("--min_track_frames", type=int, default=4)
    args = parser.parse_args()

    video_path = Path(args.video)
    xml_path = Path(args.xml)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    model = build_model(cfg, len(class_names)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    need_masks = (
        bool(getattr(cfg, "slowfast_dual_end_attention", False))
        or bool(getattr(cfg, "slowfast_tail_attention", False))
        or bool(getattr(cfg, "temporal_difference_use_dual_end_mask", False))
        or bool(getattr(cfg, "dual_end_focus_loss", False))
    ) and cfg.model_name == "slowfast"

    frames, fps, width, height = read_video(video_path)
    tracks = load_tracks(xml_path)
    segment_ranges = build_segments(
        len(frames),
        fps,
        args.mode,
        parse_durations(args.durations),
        args.window_seconds,
        args.stride_seconds,
    )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    predictions = {}
    rows = []

    with torch.no_grad():
        for seg_id, (seg_start, seg_end) in enumerate(segment_ranges, start=1):
            for tid, boxes in sorted(tracks.items()):
                items = []
                for frame_idx in range(seg_start, seg_end):
                    box = boxes.get(frame_idx)
                    if box is None:
                        continue
                    items.append(
                        {
                            "frame_idx": frame_idx,
                            "bbox": clamp_box(box, width, height),
                        }
                    )
                if len(items) < args.min_track_frames:
                    continue
                sampled = sample_uniform(items, int(cfg.num_frames))
                video, masks = build_clip(
                    frames,
                    sampled,
                    int(cfg.image_size),
                    normalize,
                    need_masks,
                    getattr(cfg, "tail_end_ratio", 1.0 / 3.0),
                )
                videos = video.unsqueeze(0).to(device)
                tail_masks = masks.unsqueeze(0).to(device) if masks is not None else None
                logits = model_forward(
                    model,
                    videos,
                    cfg.model_name,
                    tail_masks=tail_masks,
                    slowfast_alpha=getattr(cfg, "slowfast_alpha", 4),
                )
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                label = class_names[pred_idx]
                score = float(probs[pred_idx].item())
                predictions[(seg_id, tid)] = {"label": label, "score": score}
                rows.append(
                    {
                        "segment_id": seg_id,
                        "segment_start_frame": seg_start,
                        "segment_end_frame": seg_end - 1,
                        "segment_start_sec": seg_start / fps,
                        "segment_end_sec": (seg_end - 1) / fps,
                        "track_id": tid,
                        "label": label,
                        "score": score,
                        "num_available_frames": len(items),
                        **{f"p_{name}": float(probs[i].item()) for i, name in enumerate(class_names)},
                    }
                )

    csv_path = output_dir / f"{video_path.stem}_slowfast_segments_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "segment_id",
            "segment_start_frame",
            "segment_end_frame",
            "track_id",
            "label",
            "score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    del frames
    gc.collect()

    out_path = output_dir / f"{video_path.stem}_slowfast_segments_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ok, annotated = cap.read()
        if not ok:
            break
        active_segments = []
        for seg_id, (seg_start, seg_end) in enumerate(segment_ranges, start=1):
            if seg_start <= frame_idx < seg_end:
                active_segments.append(seg_id)
        if active_segments:
            for tid, boxes in sorted(tracks.items()):
                box = boxes.get(frame_idx)
                if box is None:
                    continue
                bbox = clamp_box(box, width, height)
                pred = None
                for seg_id in reversed(active_segments):
                    pred = predictions.get((seg_id, tid))
                    if pred is not None:
                        break
                if pred is None:
                    cv2.rectangle(annotated, bbox[:2], bbox[2:], (180, 180, 180), 1)
                    continue
                draw_label(annotated, bbox, pred["label"], pred["score"], tid)
        writer.write(annotated)
        frame_idx += 1
    cap.release()
    writer.release()

    print(f"Device: {device}", flush=True)
    print(f"Video: {video_path}", flush=True)
    print(f"XML: {xml_path}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Segments: {[(s / fps, e / fps) for s, e in segment_ranges]}", flush=True)
    print(f"Predictions: {csv_path}", flush=True)
    print(f"Annotated video: {out_path}", flush=True)


if __name__ == "__main__":
    main()
