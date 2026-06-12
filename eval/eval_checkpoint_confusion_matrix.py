import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_all_baselines_config import (
    CroppedInstanceDataset,
    build_model,
    load_config,
    model_forward,
)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def save_confusion_csv(cm, class_names, path):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, cm):
            writer.writerow([name, *[int(x) for x in row]])


def save_confusion_png(cm, class_names, path, normalize=False):
    values = cm.astype(np.float64)
    if normalize:
        denom = values.sum(axis=1, keepdims=True)
        values = np.divide(values, np.maximum(denom, 1.0))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    im = ax.imshow(values, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    threshold = values.max() * 0.5 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    ckpt = load_checkpoint(checkpoint_path)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = SimpleNamespace(**ckpt["config"])

    class_names = list(ckpt.get("class_names") or cfg.class_names)
    cfg.class_names = class_names
    cfg.pretrained = False
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    num_workers = args.num_workers if args.num_workers is not None else cfg.num_workers
    val_dir = Path(args.val_dir) if args.val_dir else Path(cfg.val_dir) if getattr(cfg, "val_dir", None) else Path(cfg.data_root) / "val"
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Val dir: {val_dir}", flush=True)
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
        return_attention_masks=(
            (
                getattr(cfg, "slowfast_dual_end_attention", False)
                or getattr(cfg, "slowfast_tail_attention", False)
                or getattr(cfg, "endpoint_diff_use_dual_end_mask", False)
                or getattr(cfg, "temporal_difference_use_dual_end_mask", False)
                or getattr(cfg, "videomae_dual_end_token_attention", False)
            )
            and cfg.model_name in {"slowfast", "videomae"}
        ),
        tail_end_ratio=getattr(cfg, "tail_end_ratio", 1.0 / 3.0),
    )
    print(f"Val samples: {len(dataset)}", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg, len(class_names)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate"):
            if len(batch) == 4:
                videos, tail_masks, labels, _ = batch
                tail_masks = tail_masks.to(device, non_blocking=True)
            else:
                videos, labels, _ = batch
                tail_masks = None
            videos = videos.to(device, non_blocking=True)
            logits = model_forward(
                model,
                videos,
                cfg.model_name,
                tail_masks=tail_masks,
                slowfast_alpha=getattr(cfg, "slowfast_alpha", 4),
            )
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    csv_path = output_dir / "confusion_matrix.csv"
    png_path = output_dir / "confusion_matrix.png"
    norm_png_path = output_dir / "confusion_matrix_normalized.png"
    save_confusion_csv(cm, class_names, csv_path)
    save_confusion_png(cm, class_names, png_path, normalize=False)
    save_confusion_png(cm, class_names, norm_png_path, normalize=True)

    acc = np.trace(cm) / max(1, cm.sum())
    print(f"Accuracy: {acc:.4f}", flush=True)
    print("Confusion matrix rows=true, cols=pred:", flush=True)
    print(cm, flush=True)
    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved PNG: {png_path}", flush=True)
    print(f"Saved normalized PNG: {norm_png_path}", flush=True)


if __name__ == "__main__":
    main()
