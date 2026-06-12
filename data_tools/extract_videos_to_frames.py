import argparse
from pathlib import Path

import cv2
from PIL import Image


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


def list_videos(video_dir, prefix=None):
    video_dir = Path(video_dir)
    videos = sorted([p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
    if prefix:
        prefix = prefix.lower()
        videos = [p for p in videos if p.stem.lower().startswith(prefix)]
    return videos


def extract_one_video(video_path, output_dir, image_ext=".jpg", jpg_quality=95, overwrite=False):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob(f"*{image_ext}"))
    if existing and not overwrite:
        return len(existing), "skipped"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_idx = 0
    params = []
    if image_ext.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        save_path = output_dir / f"frame_{frame_idx:06d}{image_ext}"
        if not write_frame(save_path, frame, image_ext=image_ext, jpg_quality=jpg_quality):
            raise RuntimeError(f"Failed to write frame: {save_path}")
        frame_idx += 1

    cap.release()
    if frame_idx == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frame_idx, "done"


def write_frame(save_path, frame, image_ext=".jpg", jpg_quality=95):
    save_path = Path(save_path)
    ext = image_ext.lower()
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if ext in {".jpg", ".jpeg"}:
            img.save(str(save_path), quality=int(jpg_quality), optimize=True)
        else:
            img.save(str(save_path))
        return save_path.exists()
    except Exception:
        return False


def extract_folder(video_dir, output_root, image_ext=".jpg", jpg_quality=95, overwrite=False, prefix=None):
    video_dir = Path(video_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    videos = list_videos(video_dir, prefix=prefix)
    if not videos:
        print(f"No videos found in {video_dir}", flush=True)
        return

    print(f"Video dir: {video_dir}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"Videos: {len(videos)}", flush=True)

    total = 0
    for i, video_path in enumerate(videos, start=1):
        rel = video_path.relative_to(video_dir)
        out_dir = output_root / rel.with_suffix("")
        count, status = extract_one_video(
            video_path=video_path,
            output_dir=out_dir,
            image_ext=image_ext,
            jpg_quality=jpg_quality,
            overwrite=overwrite,
        )
        total += count
        print(f"[{i}/{len(videos)}] {video_path.name} -> {out_dir} | {count} frames | {status}", flush=True)

    print(f"Finished. Total frames: {total}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract videos to frame folders.")
    parser.add_argument("--video_dir", type=str, default=None, help="Folder containing videos")
    parser.add_argument("--output_root", type=str, default=None, help="Output frame root")
    parser.add_argument("--data_root", type=str, default=None, help="Root with train/video and val/video")
    parser.add_argument("--image_ext", type=str, default=".jpg", choices=[".jpg", ".png"])
    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prefix", type=str, default=None, help="Only process videos whose filename starts with this prefix")
    args = parser.parse_args()

    if args.data_root:
        data_root = Path(args.data_root)
        for split in ["train", "val"]:
            video_dir = data_root / split / "video"
            if not video_dir.exists():
                continue
            output_root = data_root / split / "frames"
            extract_folder(
                video_dir=video_dir,
                output_root=output_root,
                image_ext=args.image_ext,
                jpg_quality=args.jpg_quality,
                overwrite=args.overwrite,
                prefix=args.prefix,
            )
        return

    if args.video_dir is None or args.output_root is None:
        raise ValueError("Use --data_root, or provide both --video_dir and --output_root")

    extract_folder(
        video_dir=args.video_dir,
        output_root=args.output_root,
        image_ext=args.image_ext,
        jpg_quality=args.jpg_quality,
        overwrite=args.overwrite,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
