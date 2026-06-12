import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
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
    def __init__(self, root, num_clips=16, frames_per_clip=16, frame_stride=1, image_size=160, train=True):
        self.root = Path(root)
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.frame_stride = frame_stride
        self.train = train
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(degrees=8)], p=0.5),
                transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.03),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ])

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
            frames = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])
            if frames:
                self.samples.append((folder.name, frames, label))
        if not self.samples:
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
            imgs = [self.transform(Image.open(frames[int(i)]).convert("RGB")) for i in one_clip]
            clips.append(torch.stack(imgs, dim=0).permute(1, 0, 2, 3))
        return torch.stack(clips, dim=0), torch.tensor(label, dtype=torch.long), video_name


def replace_pytorchvideo_head(model, num_classes):
    heads = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "ResNetBasicHead"
        and hasattr(module, "proj")
        and hasattr(module.proj, "in_features")
    ]
    if not heads:
        raise RuntimeError("Could not find PyTorchVideo ResNetBasicHead.")

    head = heads[-1]
    if isinstance(getattr(head, "pool", None), nn.AvgPool3d):
        head.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    elif hasattr(head, "pool") and head.pool is not None:
        head.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    head.proj = nn.Linear(head.proj.in_features, num_classes)
    print(f"Replaced I3D head: pool={head.pool}, classes={num_classes}", flush=True)
    return model


def build_i3d(num_classes=2, pretrained=True):
    model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=pretrained)
    return replace_pytorchvideo_head(model, num_classes)


def forward_video(model, videos, clip_batch_size=1):
    bsz, num_clips, c, t, h, w = videos.shape
    clips = videos.view(bsz * num_clips, c, t, h, w)
    logits = []
    for start in range(0, clips.size(0), clip_batch_size):
        logits.append(model(clips[start:start + clip_batch_size]))
    clip_logits = torch.cat(logits, dim=0).view(bsz, num_clips, -1)
    return clip_logits.mean(dim=1), clip_logits


def train_one_epoch(model, loader, optimizer, criterion, device, clip_batch_size):
    model.train()
    total_loss, all_labels, all_preds = 0.0, [], []
    for videos, labels, _ in tqdm(loader, desc="Train", leave=False):
        videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits, _ = forward_video(model, videos, clip_batch_size)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, device, clip_batch_size):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for videos, labels, _ in tqdm(loader, desc="Val", leave=False):
        videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits, _ = forward_video(model, videos, clip_batch_size)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return {
        "ap": average_precision_score(all_labels, all_probs) if len(set(all_labels)) == 2 else 0.0,
        "acc": accuracy_score(all_labels, all_preds),
        "p": precision_score(all_labels, all_preds, zero_division=0),
        "r": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_clips", type=int, default=16)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--clip_batch_size", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--save_path", type=str, default="best_i3d_video_level.pth")
    parser.add_argument("--log_path", type=str, default="i3d_train_log.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    train_dataset = AphidVideoFolderDataset(Path(args.data_root) / "train", args.num_clips, args.frames_per_clip, args.frame_stride, args.image_size, True)
    val_dataset = AphidVideoFolderDataset(Path(args.data_root) / "val", args.num_clips, args.frames_per_clip, args.frame_stride, args.image_size, False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    print(f"Train videos: {len(train_dataset)}", flush=True)
    print(f"Val videos: {len(val_dataset)}", flush=True)
    print(f"Sampling: {args.num_clips} clips/video, {args.frames_per_clip} frames/clip, stride={args.frame_stride}", flush=True)

    model = build_i3d(num_classes=2, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_ap", "val_acc", "val_p", "val_r", "val_f1", "best_f1", "lr"])

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, args.clip_batch_size)
        metrics = evaluate(model, val_loader, device, args.clip_batch_size)
        print(f"Epoch [{epoch:03d}/{args.epochs}] loss={train_loss:.4f} train_acc={train_acc:.4f} val_ap={metrics['ap']:.4f} val_acc={metrics['acc']:.4f} val_p={metrics['p']:.4f} val_r={metrics['r']:.4f} val_f1={metrics['f1']:.4f}", flush=True)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_f1": best_f1, "args": vars(args)}, args.save_path)
            print(f"Saved best model to {args.save_path}", flush=True)
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{metrics['ap']:.6f}", f"{metrics['acc']:.6f}", f"{metrics['p']:.6f}", f"{metrics['r']:.6f}", f"{metrics['f1']:.6f}", f"{best_f1:.6f}", optimizer.param_groups[0]["lr"]])
        print(f"Logged epoch metrics to {log_path}", flush=True)


if __name__ == "__main__":
    main()
