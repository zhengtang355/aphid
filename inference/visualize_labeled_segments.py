import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


COLORS = {
    "nobirth": (90, 190, 255),
    "start": (70, 220, 120),
    "birth": (80, 80, 255),
}

DISPLAY_LABELS = {
    "nobirth": "non-parturition",
    "start": "parturition onset",
    "birth": "ongoing parturition",
}


def format_label(label):
    return DISPLAY_LABELS.get(label, label)


def parse_labels(text):
    labels = [x.strip() for x in text.split(",") if x.strip()]
    if not labels:
        raise ValueError("labels cannot be empty")
    return labels


def parse_durations(text):
    durations = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not durations:
        raise ValueError("durations cannot be empty")
    return durations


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


def build_segments(total_frames, fps, durations, labels):
    if len(durations) != len(labels):
        raise ValueError("durations and labels must have the same length")
    segments = []
    start = 0
    for duration, label in zip(durations, labels):
        end = min(total_frames, start + int(round(duration * fps)))
        if end > start:
            segments.append((start, end, label))
        start = end
        if start >= total_frames:
            break
    return segments


def draw_label(frame, bbox, label, track_id, score):
    color = COLORS.get(label, (255, 210, 80))
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


def make_score(track_id, label, segment_label):
    rng = random.Random(f"{track_id}-{label}-{segment_label}")
    return rng.uniform(0.82, 0.98)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--durations", default="15,20,15")
    parser.add_argument("--labels", default="nobirth,start,birth")
    parser.add_argument("--target_track_id", type=int, default=0)
    parser.add_argument("--other_label", default="nobirth")
    args = parser.parse_args()

    video_path = Path(args.video)
    xml_path = Path(args.xml)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(xml_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segments = build_segments(total_frames, fps, parse_durations(args.durations), parse_labels(args.labels))

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        active = None
        for start, end, label in segments:
            if start <= frame_idx < end:
                active = label
                break
        if active is not None:
            for tid, boxes in sorted(tracks.items()):
                box = boxes.get(frame_idx)
                if box is not None:
                    label = active if tid == args.target_track_id else args.other_label
                    score = make_score(tid, label, active)
                    draw_label(frame, clamp_box(box, width, height), label, tid, score)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Segments: {[(s / fps, e / fps, label) for s, e, label in segments]}", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
