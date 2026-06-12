import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eval_checkpoint_confusion_matrix import load_checkpoint
from train_all_baselines_config import build_model, load_config, make_dual_end_mask, model_forward


IMAGE_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

DISPLAY_LABELS = {
    "nobirth": "non-parturition",
    "start": "parturition onset",
    "birth": "ongoing parturition",
}


def format_label(label):
    return DISPLAY_LABELS.get(label, label)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_videos(path_str):
    path = Path(path_str)
    if path.is_file():
        return [path]
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(1, min(w, int(round(x2))))
    y2 = max(1, min(h, int(round(y2))))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def build_frame_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def sample_uniform(items, k):
    if not items:
        return []
    if len(items) == 1:
        return [items[0] for _ in range(k)]
    idxs = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idxs]


def build_clip_tensor(items, transform, cfg, use_attention_masks):
    imgs = []
    masks = []
    for item in items:
        pil = Image.fromarray(item["crop_rgb"])
        imgs.append(transform(pil))
        if use_attention_masks:
            mask = make_dual_end_mask(pil, end_ratio=getattr(cfg, "tail_end_ratio", 1.0 / 3.0))
            mask = transforms.Resize(
                (cfg.image_size, cfg.image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            )(mask)
            masks.append(transforms.functional.to_tensor(mask))
    video = torch.stack(imgs, dim=0).unsqueeze(0)
    if use_attention_masks:
        return video, torch.stack(masks, dim=0).unsqueeze(0)
    return video, None


def draw_label(frame, bbox, label, score, track_id):
    color_map = {
        "nobirth": (80, 180, 80),
        "start": (0, 165, 255),
        "birth": (0, 0, 255),
    }
    color = color_map.get(label, (255, 255, 255))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"ID{track_id}: {format_label(label)} {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty1 = max(0, y1 - th - baseline - 6)
    cv2.rectangle(frame, (x1, ty1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def load_classifier(checkpoint_path, config_path, device):
    ckpt = load_checkpoint(checkpoint_path)
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = load_config(None)
        for k, v in ckpt["config"].items():
            setattr(cfg, k, v)
    class_names = list(ckpt.get("class_names") or cfg.class_names)
    cfg.class_names = class_names
    cfg.pretrained = False
    model = build_model(cfg, len(class_names)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg, class_names


def parse_cvat_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    tracks = []
    for track in root.findall("track"):
        track_id = int(track.get("id", "0"))
        label = track.get("label", "")
        boxes = []
        for box in track.findall("box"):
            if box.get("outside", "0") == "1":
                continue
            boxes.append(
                {
                    "frame_idx": int(box.get("frame", "0")),
                    "bbox": (
                        float(box.get("xtl", "0")),
                        float(box.get("ytl", "0")),
                        float(box.get("xbr", "0")),
                        float(box.get("ybr", "0")),
                    ),
                    "label": label,
                }
            )
        if boxes:
            tracks.append({"track_id": track_id, "label": label, "boxes": boxes})
    return tracks


def collect_tracks_from_xml(video_path, xml_path, args):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
    cap.release()

    frame_boxes = defaultdict(list)
    track_items = defaultdict(list)
    xml_tracks = parse_cvat_xml(xml_path)

    for track in xml_tracks:
        tid = track["track_id"]
        for entry in track["boxes"]:
            frame_idx = entry["frame_idx"]
            if frame_idx < 0 or frame_idx >= len(frames):
                continue
            x1, y1, x2, y2 = clamp_box(*entry["bbox"], width, height)
            crop_bgr = frames[frame_idx][y1:y2, x1:x2].copy()
            if crop_bgr.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            item = {
                "frame_idx": frame_idx,
                "bbox": (x1, y1, x2, y2),
                "crop_rgb": crop_rgb,
                "xml_label": entry["label"],
            }
            track_items[tid].append(item)
            frame_boxes[frame_idx].append((tid, (x1, y1, x2, y2)))

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "track_items": track_items,
        "frame_boxes": frame_boxes,
    }


def classify_track_windows(track_items, model, cfg, class_names, transform, device, args, total_frames):
    use_attention_masks = bool(
        getattr(cfg, "slowfast_dual_end_attention", False) or getattr(cfg, "slowfast_tail_attention", False)
    )
    window_frames = max(1, int(round(args.window_seconds * args.fps)))
    global_chunk_starts = list(range(0, total_frames, window_frames))

    per_frame_probs = defaultdict(list)
    window_rows = []

    for tid, items in sorted(track_items.items()):
        items = sorted(items, key=lambda x: x["frame_idx"])
        if len(items) < args.min_track_frames:
            continue

        for window_start in global_chunk_starts:
            window_end = window_start + window_frames - 1
            subset = [item for item in items if window_start <= item["frame_idx"] <= window_end]
            if len(subset) < args.min_track_frames:
                continue

            sampled = sample_uniform(subset, int(cfg.num_frames))
            videos, attention_masks = build_clip_tensor(sampled, transform, cfg, use_attention_masks)
            videos = videos.to(device)
            if attention_masks is not None:
                attention_masks = attention_masks.to(device)

            with torch.no_grad():
                logits = model_forward(
                    model,
                    videos,
                    cfg.model_name,
                    tail_masks=attention_masks,
                    slowfast_alpha=getattr(cfg, "slowfast_alpha", 4),
                )
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

            pred_idx = int(np.argmax(probs))
            score = float(probs[pred_idx])
            label = class_names[pred_idx]
            window_rows.append(
                {
                    "track_id": tid,
                    "window_start": window_start,
                    "window_end": window_end,
                    "label": label,
                    "score": score,
                    **{f"p_{name}": float(probs[i]) for i, name in enumerate(class_names)},
                }
            )

            for item in subset:
                per_frame_probs[(tid, item["frame_idx"])].append(probs.copy())

    frame_level_preds = {}
    for key, prob_list in per_frame_probs.items():
        mean_probs = np.mean(np.stack(prob_list, axis=0), axis=0)
        pred_idx = int(np.argmax(mean_probs))
        frame_level_preds[key] = {
            "label": class_names[pred_idx],
            "score": float(mean_probs[pred_idx]),
            "probs": mean_probs,
        }
    return frame_level_preds, window_rows


def annotate_video(video_path, output_video_path, frame_boxes, frame_level_preds, fps, width, height):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot reopen video: {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        annotated = frame.copy()
        for tid, bbox in frame_boxes.get(frame_idx, []):
            pred = frame_level_preds.get((tid, frame_idx))
            if pred is None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 180, 180), 1)
                continue
            draw_label(annotated, bbox, pred["label"], pred["score"], tid)
        writer.write(annotated)
    cap.release()
    writer.release()


def save_csvs(output_dir, frame_level_preds, window_rows, class_names):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_csv = output_dir / "frame_level_predictions.csv"
    with frame_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["track_id", "frame_idx", "label", "score"] + [f"p_{name}" for name in class_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (tid, frame_idx), pred in sorted(frame_level_preds.items()):
            row = {"track_id": tid, "frame_idx": frame_idx, "label": pred["label"], "score": pred["score"]}
            for i, name in enumerate(class_names):
                row[f"p_{name}"] = float(pred["probs"][i])
            writer.writerow(row)

    window_csv = output_dir / "window_level_predictions.csv"
    with window_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["track_id", "window_start", "window_end", "label", "score"] + [f"p_{name}" for name in class_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in window_rows:
            writer.writerow(row)


def process_video(video_path, xml_path, model, cfg, class_names, transform, device, root_output_dir, args):
    print(f"\n=== Processing {video_path.name} with XML boxes ===", flush=True)
    video_output_dir = root_output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    collected = collect_tracks_from_xml(video_path, xml_path, args)
    args.fps = collected["fps"]
    frame_level_preds, window_rows = classify_track_windows(
        collected["track_items"], model, cfg, class_names, transform, device, args, collected["total_frames"]
    )

    output_video = video_output_dir / f"{video_path.stem}_annotated.mp4"
    annotate_video(
        video_path,
        output_video,
        collected["frame_boxes"],
        frame_level_preds,
        collected["fps"],
        collected["width"],
        collected["height"],
    )
    save_csvs(video_output_dir, frame_level_preds, window_rows, class_names)
    print(f"Saved annotated video: {output_video}", flush=True)
    print(f"Saved csvs to: {video_output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Offline XML-box + SlowFast multi-target behavior inference.")
    parser.add_argument("--input_path", type=str, required=True, help="One video path or one folder of videos.")
    parser.add_argument("--xml_path", type=str, required=True, help="CVAT annotations.xml path.")
    parser.add_argument("--cls_checkpoint", type=str, required=True, help="Behavior classifier checkpoint.")
    parser.add_argument("--cls_config", type=str, required=True, help="Behavior classifier config json.")
    parser.add_argument("--output_root", type=str, default="", help="Output root directory.")
    parser.add_argument("--window_seconds", type=float, default=15.0, help="Fixed chunk size in seconds.")
    parser.add_argument("--min_track_frames", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    videos = list_videos(args.input_path)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {args.input_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, class_names = load_classifier(args.cls_checkpoint, args.cls_config, device)
    transform = build_frame_transform(int(cfg.image_size))

    root_output_dir = Path(args.output_root) if args.output_root else Path(args.input_path if Path(args.input_path).is_dir() else Path(args.input_path).parent) / "offline_xml_results"
    root_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}", flush=True)
    print(f"Videos: {len(videos)}", flush=True)
    print(f"Classes: {class_names}", flush=True)
    print(f"Window: {args.window_seconds}s", flush=True)
    print(f"Output root: {root_output_dir}", flush=True)

    xml_path = Path(args.xml_path)
    for video_path in videos:
        process_video(video_path, xml_path, model, cfg, class_names, transform, device, root_output_dir, args)
    print("\nAll videos finished.", flush=True)


if __name__ == "__main__":
    main()
