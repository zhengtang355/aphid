import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


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


def load_frame_boxes_from_cvat_xml(xml_path: Path, frame_idx: int):
    root = ET.parse(xml_path).getroot()
    rows = []
    for track in root.findall("track"):
        track_id = track.get("id", "")
        label = track.get("label", "")
        for box in track.findall("box"):
            if int(box.get("frame", "-1")) != frame_idx:
                continue
            if box.get("outside", "0") == "1":
                continue
            rows.append(
                {
                    "track_id": track_id,
                    "label": label,
                    "x1": float(box.get("xtl")),
                    "y1": float(box.get("ytl")),
                    "x2": float(box.get("xbr")),
                    "y2": float(box.get("ybr")),
                }
            )
    return rows


def build_dual_end_mask_for_box(height, width, box, end_ratio=1.0 / 3.0):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    horizontal = bw >= bh
    mask = np.zeros((height, width), dtype=np.uint8)

    if horizontal:
        end_len = max(1, int(round(bw * end_ratio)))
        mask[y1:y2, x1:min(x2, x1 + end_len)] = 1
        mask[y1:y2, max(x1, x2 - end_len):x2] = 1
        split_a = x1 + end_len
        split_b = x2 - end_len
        return mask, horizontal, end_len, split_a, split_b
    else:
        end_len = max(1, int(round(bh * end_ratio)))
        mask[y1:min(y2, y1 + end_len), x1:x2] = 1
        mask[max(y1, y2 - end_len):y2, x1:x2] = 1
        split_a = y1 + end_len
        split_b = y2 - end_len
        return mask, horizontal, end_len, split_a, split_b


def overlay_mask(rgb, mask, color=(104, 140, 255), alpha=0.26):
    out = rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    idx = mask > 0
    out[idx] = np.clip((1.0 - alpha) * out[idx] + alpha * color_arr, 0, 255)
    return out.astype(np.uint8)


def draw_annotations(rgb, boxes, end_ratio=1.0 / 3.0):
    h, w = rgb.shape[:2]
    out = rgb.copy()
    overlay_total = np.zeros((h, w), dtype=np.uint8)

    for row in boxes:
        box = clamp_box(row["x1"], row["y1"], row["x2"], row["y2"], w, h)
        mask, horizontal, _, split_a, split_b = build_dual_end_mask_for_box(h, w, box, end_ratio=end_ratio)
        overlay_total = np.maximum(overlay_total, mask)
        x1, y1, x2, y2 = box

        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color=(36, 92, 180), thickness=2)
        if horizontal:
            cv2.line(out, (split_a, y1), (split_a, y2 - 1), color=(235, 235, 235), thickness=1)
            cv2.line(out, (split_b, y1), (split_b, y2 - 1), color=(235, 235, 235), thickness=1)
        else:
            cv2.line(out, (x1, split_a), (x2 - 1, split_a), color=(235, 235, 235), thickness=1)
            cv2.line(out, (x1, split_b), (x2 - 1, split_b), color=(235, 235, 235), thickness=1)

    out = overlay_mask(out, overlay_total, color=(88, 126, 214), alpha=0.24)
    return out, (overlay_total * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Overlay dual-end regions on a full frame using CVAT XML boxes.")
    parser.add_argument("--frame_path", type=str, required=True)
    parser.add_argument("--xml_path", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mask_output_path", type=str, default=None)
    parser.add_argument("--end_ratio", type=float, default=1.0 / 3.0)
    args = parser.parse_args()

    frame_path = Path(args.frame_path)
    xml_path = Path(args.xml_path)
    output_path = Path(args.output_path)
    mask_output_path = Path(args.mask_output_path) if args.mask_output_path else output_path.with_name(output_path.stem + "_mask.png")

    rgb = imread_rgb(frame_path)
    if rgb is None:
        raise FileNotFoundError(f"Cannot read frame image: {frame_path}")

    boxes = load_frame_boxes_from_cvat_xml(xml_path, args.frame_idx)
    if not boxes:
        raise RuntimeError(f"No boxes found at frame {args.frame_idx} in {xml_path}")

    vis, mask = draw_annotations(rgb, boxes, end_ratio=args.end_ratio)
    save_rgb(output_path, vis)
    save_rgb(mask_output_path, np.repeat(mask[:, :, None], 3, axis=2))

    print(f"Frame: {frame_path}")
    print(f"XML: {xml_path}")
    print(f"Frame index: {args.frame_idx}")
    print(f"Boxes: {len(boxes)}")
    print(f"Saved overlay: {output_path}")
    print(f"Saved mask: {mask_output_path}")


if __name__ == "__main__":
    main()
