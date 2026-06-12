import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AphidVideoFolderDataset(Dataset):
    def __init__(
        self,
        root,
        num_clips=16,
        frames_per_clip=16,
        frame_stride=1,
        image_size=112,
        train=True,
    ):
        self.root = Path(root)
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.frame_stride = frame_stride
        self.train = train

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.65, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.RandomRotation(degrees=8)],
                        p=0.5,
                    ),
                    transforms.ColorJitter(
                        brightness=0.35,
                        contrast=0.35,
                        saturation=0.2,
                        hue=0.03,
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                        p=0.2,
                    ),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        self.samples = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue

            name = folder.name.lower()
            if name.startswith("birth"):
                label = 1
            elif name.startswith("nobirth") or name.startswith("no_birth"):
                label = 0
            else:
                continue

            frames = sorted(
                [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
            )
            if len(frames) > 0:
                self.samples.append((folder.name, frames, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid video folders found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def _sample_clip_indices(self, n_frames):
        clip_span = self.frames_per_clip * self.frame_stride
        boundaries = np.linspace(0, n_frames, self.num_clips + 1).astype(np.int64)
        all_indices = []
        for clip_id in range(self.num_clips):
            seg_start = int(boundaries[clip_id])
            seg_end = int(boundaries[clip_id + 1])
            seg_end = max(seg_end, seg_start + 1)
            max_start = max(seg_start, seg_end - clip_span)
            if self.train:
                start = random.randint(seg_start, max_start) if max_start > seg_start else seg_start
            else:
                start = seg_start + max(0, (seg_end - seg_start - clip_span) // 2)
            indices = start + np.arange(self.frames_per_clip) * self.frame_stride
            indices = np.clip(indices, 0, n_frames - 1)
            all_indices.append(indices.astype(np.int64))
        return np.stack(all_indices, axis=0)

    def __getitem__(self, idx):
        video_name, frames, label = self.samples[idx]
        clip_indices = self._sample_clip_indices(len(frames))

        clips = []
        for one_clip in clip_indices:
            imgs = []
            for frame_idx in one_clip:
                img = Image.open(frames[int(frame_idx)]).convert("RGB")
                imgs.append(self.transform(img))
            clip = torch.stack(imgs, dim=0).permute(1, 0, 2, 3)
            clips.append(clip)

        video = torch.stack(clips, dim=0)
        return video, torch.tensor(label, dtype=torch.long), video_name


class C3D(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def forward_video(model, videos, clip_batch_size=4):
    # videos: [B, N, C, T, H, W]
    bsz, num_clips, c, t, h, w = videos.shape
    clips = videos.view(bsz * num_clips, c, t, h, w)

    clip_logits_list = []
    for start in range(0, clips.size(0), clip_batch_size):
        end = start + clip_batch_size
        clip_logits_list.append(model(clips[start:end]))

    clip_logits = torch.cat(clip_logits_list, dim=0)
    clip_logits = clip_logits.view(bsz, num_clips, -1)
    video_logits = clip_logits.mean(dim=1)
    return video_logits, clip_logits


def train_one_epoch(model, loader, optimizer, criterion, device, clip_batch_size):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for videos, labels, _ in tqdm(loader, desc="Train", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, _ = forward_video(model, videos, clip_batch_size=clip_batch_size)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device, clip_batch_size):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for videos, labels, _ in tqdm(loader, desc="Val", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, _ = forward_video(model, videos, clip_batch_size=clip_batch_size)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, zero_division=0)
    r = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    ap = average_precision_score(all_labels, all_probs) if len(set(all_labels)) == 2 else 0.0
    return {"ap": ap, "acc": acc, "p": p, "r": r, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_clips", type=int, default=16)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--clip_batch_size", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="best_c3d_video_level.pth")
    parser.add_argument("--log_path", type=str, default="c3d_train_log.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    train_root = data_root / "train"
    val_root = data_root / "val"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_dataset = AphidVideoFolderDataset(
        train_root,
        num_clips=args.num_clips,
        frames_per_clip=args.frames_per_clip,
        frame_stride=args.frame_stride,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = AphidVideoFolderDataset(
        val_root,
        num_clips=args.num_clips,
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

    print(f"Train videos: {len(train_dataset)}", flush=True)
    print(f"Val videos: {len(val_dataset)}", flush=True)
    print(
        f"Sampling: {args.num_clips} clips/video, "
        f"{args.frames_per_clip} frames/clip, stride={args.frame_stride}",
        flush=True,
    )

    model = C3D(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_ap",
                "val_acc",
                "val_p",
                "val_r",
                "val_f1",
                "best_f1",
                "lr",
            ]
        )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.clip_batch_size,
        )
        metrics = evaluate(model, val_loader, device, args.clip_batch_size)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_ap={metrics['ap']:.4f} "
            f"val_acc={metrics['acc']:.4f} "
            f"val_p={metrics['p']:.4f} "
            f"val_r={metrics['r']:.4f} "
            f"val_f1={metrics['f1']:.4f}",
            flush=True,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"Saved best model to {args.save_path}", flush=True)

        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{metrics['ap']:.6f}",
                    f"{metrics['acc']:.6f}",
                    f"{metrics['p']:.6f}",
                    f"{metrics['r']:.6f}",
                    f"{metrics['f1']:.6f}",
                    f"{best_f1:.6f}",
                    optimizer.param_groups[0]["lr"],
                ]
            )
        print(f"Logged epoch metrics to {log_path}", flush=True)


if __name__ == "__main__":
    main()
