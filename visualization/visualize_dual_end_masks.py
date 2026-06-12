import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def list_frames(folder: Path):
    frames = []
    for ext in IMAGE_EXTS:
        frames.extend(folder.glob(f"*{ext}"))
    return sorted(frames)


def uniform_indices(n: int, num_frames: int):
    if n <= 0:
        return []
    if n == 1:
        return [0] * num_frames
    return np.linspace(0, n - 1, num_frames).round().astype(np.int64).tolist()


def make_dual_end_mask(rgb: Image.Image, end_ratio: float = 1.0 / 3.0):
    arr = np.asarray(rgb.convert("RGB"))
    h, w = arr.shape[:2]
    horizontal = w >= h
    mask = np.zeros((h, w), dtype=np.uint8)

    if horizontal:
        end_len = max(1, int(round(w * end_ratio)))
        mask[:, :end_len] = 255
        mask[:, max(0, w - end_len):] = 255
    else:
        end_len = max(1, int(round(h * end_ratio)))
        mask[:end_len, :] = 255
        mask[max(0, h - end_len):, :] = 255

    return Image.fromarray(mask, mode="L"), horizontal, end_len


def overlay_mask(rgb: Image.Image, mask: Image.Image, color=(255, 0, 255), alpha=0.40):
    rgb_arr = np.asarray(rgb.convert("RGB")).astype(np.float32)
    mask_arr = np.asarray(mask).astype(np.float32) / 255.0
    color_arr = np.asarray(color, dtype=np.float32)

    out = rgb_arr.copy()
    idx = mask_arr > 0
    out[idx] = np.clip((1.0 - alpha) * out[idx] + alpha * color_arr, 0, 255)
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def draw_end_boundaries(rgb: Image.Image, horizontal: bool, end_len: int, color=(0, 255, 255), width=2):
    out = rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    if horizontal:
        x1 = min(w - 1, max(0, end_len))
        x2 = min(w - 1, max(0, w - end_len))
        draw.line((x1, 0, x1, h - 1), fill=color, width=width)
        draw.line((x2, 0, x2, h - 1), fill=color, width=width)
    else:
        y1 = min(h - 1, max(0, end_len))
        y2 = min(h - 1, max(0, h - end_len))
        draw.line((0, y1, w - 1, y1), fill=color, width=width)
        draw.line((0, y2, w - 1, y2), fill=color, width=width)
    return out


def compose_triptych(src: Image.Image, overlay: Image.Image, mask: Image.Image):
    src = src.convert("RGB")
    overlay = overlay.convert("RGB")
    mask_rgb = ImageOps.colorize(mask.convert("L"), black=(0, 0, 0), white=(255, 0, 255)).convert("RGB")
    w, h = src.size
    canvas = Image.new("RGB", (w * 3, h), (255, 255, 255))
    canvas.paste(src, (0, 0))
    canvas.paste(overlay, (w, 0))
    canvas.paste(mask_rgb, (w * 2, 0))
    return canvas


def iter_instance_dirs(root_dir: Path, class_names=None):
    root_dir = Path(root_dir)
    class_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if class_names:
        keep = {x.strip() for x in class_names if x.strip()}
        class_dirs = [p for p in class_dirs if p.name in keep]

    folders = []
    for class_dir in class_dirs:
        for instance_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()]):
            folders.append((class_dir.name, instance_dir))
    return folders


def main():
    parser = argparse.ArgumentParser(description="Visualize dual-end region masks on validation clips.")
    parser.add_argument("--input_root", type=str, default=r"C:\数据\data_original_box\val")
    parser.add_argument("--output_root", type=str, default=r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\val_dual_end_mask_visualization")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--end_ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--class_names", type=str, default="nobirth,start,birth")
    parser.add_argument("--max_folders", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    folders = iter_instance_dirs(input_root, class_names=class_names)
    if args.max_folders is not None:
        folders = folders[: int(args.max_folders)]

    print(f"Input root: {input_root}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"Classes: {class_names}", flush=True)
    print(f"Folders: {len(folders)}", flush=True)
    print(f"Uniform sampled frames/folder: {args.num_frames}", flush=True)
    print(f"End ratio: {args.end_ratio}", flush=True)

    saved = 0
    skipped = 0
    for class_name, instance_dir in tqdm(folders, desc="Dual-end vis"):
        frames = list_frames(instance_dir)
        if not frames:
            skipped += 1
            continue

        out_dir = output_root / class_name / instance_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        sampled_indices = uniform_indices(len(frames), args.num_frames)

        for out_idx, frame_idx in enumerate(sampled_indices):
            frame_path = frames[frame_idx]
            triptych_path = out_dir / f"compare_{out_idx:02d}_{frame_path.stem}.png"
            overlay_path = out_dir / f"overlay_{out_idx:02d}_{frame_path.stem}.png"
            mask_path = out_dir / f"mask_{out_idx:02d}_{frame_path.stem}.png"
            if triptych_path.exists() and overlay_path.exists() and mask_path.exists() and not args.overwrite:
                continue

            src = Image.open(frame_path).convert("RGB")
            mask, horizontal, end_len = make_dual_end_mask(src, end_ratio=args.end_ratio)
            overlay = overlay_mask(src, mask, color=(255, 0, 255), alpha=0.40)
            overlay = draw_end_boundaries(overlay, horizontal, end_len, color=(0, 255, 255), width=2)
            triptych = compose_triptych(src, overlay, mask)

            overlay.save(overlay_path)
            mask.save(mask_path)
            triptych.save(triptych_path)
            saved += 3

    print(f"Saved files: {saved}", flush=True)
    print(f"Skipped empty folders: {skipped}", flush=True)


if __name__ == "__main__":
    main()
