import argparse
import csv
import random
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASS_NAMES = ["nobirth", "start", "birth", "end"]


PREFIX_ALIASES = {
    "nobirth": ["nobirth", "no_birth", "non_birth", "negative", "background", "非产子"],
    "start": ["start", "birth_start", "begin", "onset", "产子开始", "开始产子"],
    "birth": ["birth", "birth_middle", "middle", "mid", "ongoing", "产子中"],
    "end": ["end", "birth_end", "offset", "产子结束", "结束产子"],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def infer_label_from_name(name, class_names):
    lower = name.lower()
    for idx, class_name in enumerate(class_names):
        aliases = PREFIX_ALIASES.get(class_name, [class_name])
        for alias in aliases:
            alias = alias.lower()
            if lower == alias or lower.startswith(alias + "_") or lower.startswith(alias + "-"):
                return idx
            if lower.startswith(alias):
                next_pos = len(alias)
                if next_pos == len(lower) or lower[next_pos].isdigit():
                    return idx
    return None


class AphidStageClipDataset(Dataset):
    def __init__(
        self,
        root,
        class_names,
        frames_per_clip=16,
        frame_stride=1,
        image_size=224,
        train=True,
    ):
        self.root = Path(root)
        self.class_names = class_names
        self.frames_per_clip = frames_per_clip
        self.frame_stride = frame_stride
        self.train = train

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.75, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.4),
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.15,
                        hue=0.02,
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                        p=0.15,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No valid clip folders found in {self.root}")

    def _collect_samples(self):
        samples = []
        has_class_dirs = any((self.root / name).is_dir() for name in self.class_names)
        if has_class_dirs:
            for label, class_name in enumerate(self.class_names):
                class_dir = self.root / class_name
                if not class_dir.is_dir():
                    continue

                direct_frames = list_images(class_dir)
                if direct_frames:
                    samples.append((class_dir.name, direct_frames, label))

                for clip_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()]):
                    frames = list_images(clip_dir)
                    if frames:
                        samples.append((clip_dir.name, frames, label))
            return samples

        for clip_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            label = infer_label_from_name(clip_dir.name, self.class_names)
            if label is None:
                continue
            frames = list_images(clip_dir)
            if frames:
                samples.append((clip_dir.name, frames, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, n_frames):
        # Uniformly sample frames from the whole clip/video. This is better for
        # slow aphid behaviors because each input covers the full 15 s segment.
        indices = np.linspace(0, n_frames - 1, self.frames_per_clip)
        indices = np.round(indices)
        indices = np.clip(indices, 0, n_frames - 1)
        return indices.astype(np.int64)

    def __getitem__(self, idx):
        clip_name, frames, label = self.samples[idx]
        indices = self._sample_indices(len(frames))

        imgs = []
        for frame_idx in indices:
            img = Image.open(frames[int(frame_idx)]).convert("RGB")
            imgs.append(self.transform(img))

        # VideoMAE expects [T, C, H, W] per sample after DataLoader -> [B, T, C, H, W].
        clip = torch.stack(imgs, dim=0)
        return clip, torch.tensor(label, dtype=torch.long), clip_name


class VideoMAETailAttentionClassifier(torch.nn.Module):
    def __init__(self, model_name, num_classes, num_heads=4, dropout=0.2):
        super().__init__()
        from transformers import VideoMAEModel

        self.backbone = VideoMAEModel.from_pretrained(model_name)
        hidden_dim = self.backbone.config.hidden_size
        self.tail_norm = torch.nn.LayerNorm(hidden_dim)
        self.tail_attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        tokens = self.tail_norm(outputs.last_hidden_state)
        attn_out, _ = self.tail_attn(tokens, tokens, tokens, need_weights=False)
        feat = attn_out[:, 0]
        logits = self.classifier(feat)
        return SimpleNamespace(logits=logits)


def build_videomae(model_name, num_classes, class_names):
    return VideoMAETailAttentionClassifier(model_name, num_classes)


def compute_metrics(labels, preds, probs, class_names):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    num_classes = len(class_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    y_true = np.zeros((len(labels), num_classes), dtype=np.int64)
    y_true[np.arange(len(labels)), labels] = 1

    class_ap = []
    for class_idx in range(num_classes):
        if y_true[:, class_idx].sum() == 0:
            class_ap.append(float("nan"))
        else:
            class_ap.append(average_precision_score(y_true[:, class_idx], probs[:, class_idx]))

    valid_ap = [ap for ap in class_ap if not np.isnan(ap)]
    m_ap = float(np.mean(valid_ap)) if valid_ap else 0.0
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    return {
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "mAP": m_ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "ap": class_ap,
        "confusion_matrix": cm,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for clips, labels, _ in tqdm(loader, desc="Train", leave=False):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(pixel_values=clips)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for clips, labels, _ in tqdm(loader, desc="Val", leave=False):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(pixel_values=clips)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    return compute_metrics(all_labels, all_preds, all_probs, class_names)


def write_confusion_matrix(path, cm, class_names):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name] + row.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--class_names", type=str, default=",".join(DEFAULT_CLASS_NAMES))
    parser.add_argument("--save_path", type=str, default="best_videomae_4class.pth")
    parser.add_argument("--log_path", type=str, default="videomae_4class_train_log.csv")
    parser.add_argument("--cm_path", type=str, default="videomae_4class_best_confusion_matrix.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Classes: {class_names}", flush=True)
    print(f"VideoMAE model: {args.model_name}", flush=True)

    train_dataset = AphidStageClipDataset(
        Path(args.data_root) / "train",
        class_names=class_names,
        frames_per_clip=args.frames_per_clip,
        frame_stride=args.frame_stride,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = AphidStageClipDataset(
        Path(args.data_root) / "val",
        class_names=class_names,
        frames_per_clip=args.frames_per_clip,
        frame_stride=args.frame_stride,
        image_size=args.image_size,
        train=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train clips: {len(train_dataset)}", flush=True)
    print(f"Val clips: {len(val_dataset)}", flush=True)
    print(
        f"Sampling: {args.frames_per_clip} uniformly sampled frames/clip, image_size={args.image_size}",
        flush=True,
    )

    model = VideoMAETailAttentionClassifier(
        args.model_name,
        num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_acc",
        "val_macro_f1",
        "val_micro_f1",
        "val_mAP",
        "best_mAP",
    ]
    for class_name in class_names:
        header.extend(
            [
                f"{class_name}_AP",
                f"{class_name}_P",
                f"{class_name}_R",
                f"{class_name}_F1",
                f"{class_name}_support",
            ]
        )
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    best_map = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device, class_names)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={metrics['acc']:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} "
            f"val_micro_f1={metrics['micro_f1']:.4f} "
            f"val_mAP={metrics['mAP']:.4f}",
            flush=True,
        )

        for idx, class_name in enumerate(class_names):
            print(
                f"  {class_name}: "
                f"AP={metrics['ap'][idx]:.4f} "
                f"P={metrics['precision'][idx]:.4f} "
                f"R={metrics['recall'][idx]:.4f} "
                f"F1={metrics['f1'][idx]:.4f} "
                f"N={int(metrics['support'][idx])}",
                flush=True,
            )

        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_mAP": best_map,
                    "class_names": class_names,
                    "args": vars(args),
                },
                args.save_path,
            )
            write_confusion_matrix(args.cm_path, metrics["confusion_matrix"], class_names)
            print(f"Saved best model to {args.save_path}", flush=True)
            print(f"Saved best confusion matrix to {args.cm_path}", flush=True)

        row = [
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{metrics['acc']:.6f}",
            f"{metrics['macro_f1']:.6f}",
            f"{metrics['micro_f1']:.6f}",
            f"{metrics['mAP']:.6f}",
            f"{best_map:.6f}",
        ]
        for idx in range(num_classes):
            row.extend(
                [
                    f"{metrics['ap'][idx]:.6f}",
                    f"{metrics['precision'][idx]:.6f}",
                    f"{metrics['recall'][idx]:.6f}",
                    f"{metrics['f1'][idx]:.6f}",
                    int(metrics["support"][idx]),
                ]
            )
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        print(f"Logged epoch metrics to {log_path}", flush=True)


if __name__ == "__main__":
    main()
