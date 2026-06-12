import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
DEFAULT_CLASSES = ["nobirth", "start", "birth"]


def imread_rgb(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, quality=95)


def find_frame_path(frame_dir: Path, frame_idx: int):
    stem = f"frame_{frame_idx:06d}"
    for ext in IMAGE_EXTS:
        p = frame_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def clamp_box(box, width, height, expand_ratio):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    x1 -= bw * expand_ratio
    y1 -= bh * expand_ratio
    x2 += bw * expand_ratio
    y2 += bh * expand_ratio

    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(1, min(width, int(round(x2))))
    y2 = max(1, min(height, int(round(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def read_annotation(csv_path: Path, class_names):
    instances = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"frame_idx", "instance_id", "label", "x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        for row in reader:
            label = row["label"].strip()
            if label not in class_names:
                continue
            instance_id = str(row["instance_id"]).strip()
            frame_idx = int(float(row["frame_idx"]))
            item = {
                "frame_idx": frame_idx,
                "label": label,
                "box": (
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ),
            }
            instances[instance_id].append(item)

    for items in instances.values():
        items.sort(key=lambda x: x["frame_idx"])
    return instances


def crop_one_split(
    split_name,
    frame_root,
    anno_dir,
    output_root,
    class_names,
    target_size,
    keep_original_size,
    expand_ratio,
    start_counts,
):
    frame_root = Path(frame_root)
    anno_dir = Path(anno_dir)
    output_root = Path(output_root) / split_name

    csv_files = sorted(anno_dir.glob("*.csv"))
    summary = {
        "clips": 0,
        "instances": 0,
        "saved_frames": 0,
        "missing_frame_dirs": 0,
        "missing_frames": 0,
        "empty_crops": 0,
    }

    for csv_path in tqdm(csv_files, desc=f"Crop {split_name}"):
        clip_name = csv_path.stem
        frame_dir = frame_root / clip_name
        if not frame_dir.exists():
            summary["missing_frame_dirs"] += 1
            continue

        instances = read_annotation(csv_path, class_names)
        if not instances:
            continue
        summary["clips"] += 1

        for instance_id, items in instances.items():
            label = items[0]["label"]
            label_set = {x["label"] for x in items}
            if len(label_set) > 1:
                label = max(label_set, key=lambda x: sum(1 for item in items if item["label"] == x))

            start_counts[label] += 1
            out_dir = output_root / label / f"{label}_{start_counts[label]:04d}"
            saved = 0

            for item in items:
                frame_path = find_frame_path(frame_dir, item["frame_idx"])
                if frame_path is None:
                    summary["missing_frames"] += 1
                    continue

                img = imread_rgb(frame_path)
                if img is None:
                    summary["missing_frames"] += 1
                    continue

                h, w = img.shape[:2]
                x1, y1, x2, y2 = clamp_box(item["box"], w, h, expand_ratio)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    summary["empty_crops"] += 1
                    continue

                if not keep_original_size:
                    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                save_rgb(out_dir / f"frame_{saved:06d}.jpg", crop)
                saved += 1

            if saved > 0:
                summary["instances"] += 1
                summary["saved_frames"] += saved

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Crop each annotated aphid instance into 224x224 frame folders."
    )
    parser.add_argument("--data_root", type=str, default=r"C:\数据\slowonly_data")
    parser.add_argument("--train_frame_dir", type=str, default=None)
    parser.add_argument("--train_anno_dir", type=str, default=None)
    parser.add_argument("--val_frame_dir", type=str, default=None)
    parser.add_argument("--val_anno_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--keep_original_size", action="store_true")
    parser.add_argument("--expand_ratio", type=float, default=0.15)
    parser.add_argument("--class_names", type=str, default=",".join(DEFAULT_CLASSES))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_frame_dir = Path(args.train_frame_dir) if args.train_frame_dir else data_root / "train" / "frames"
    train_anno_dir = Path(args.train_anno_dir) if args.train_anno_dir else data_root / "train" / "annotations"
    val_frame_dir = Path(args.val_frame_dir) if args.val_frame_dir else data_root / "val" / "frames"
    val_anno_dir = Path(args.val_anno_dir) if args.val_anno_dir else data_root / "val" / "annotations"
    default_name = "cropped_instances_original" if args.keep_original_size else "cropped_instances_224"
    output_root = Path(args.output_root) if args.output_root else data_root / default_name
    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]

    print(f"Classes: {class_names}")
    if args.keep_original_size:
        print("Target size: keep original detection box size")
    else:
        print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Expand ratio: {args.expand_ratio}")
    print(f"Output root: {output_root}")

    for split_name, frame_dir, anno_dir in [
        ("train", train_frame_dir, train_anno_dir),
        ("val", val_frame_dir, val_anno_dir),
    ]:
        counts = defaultdict(int)
        summary = crop_one_split(
            split_name=split_name,
            frame_root=frame_dir,
            anno_dir=anno_dir,
            output_root=output_root,
            class_names=class_names,
            target_size=args.target_size,
            keep_original_size=args.keep_original_size,
            expand_ratio=args.expand_ratio,
            start_counts=counts,
        )
        print(f"\n{split_name} summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("  class folders:")
        for c in class_names:
            print(f"    {c}: {counts[c]}")


if __name__ == "__main__":
    main()
