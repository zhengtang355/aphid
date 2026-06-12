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


class AphidSparseFrameDataset(Dataset):
    def __init__(self, root, num_frames=128, image_size=224, train=True):
        self.root = Path(root)
        self.num_frames = num_frames
        self.train = train

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
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
            if frames:
                self.samples.append((folder.name, frames, label))

        if not self.samples:
            raise RuntimeError(f"No valid video folders found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, n_frames):
        if self.train:
            # Divide the whole video into temporal bins and randomly sample
            # one frame per bin. This preserves long-range coverage while adding
            # temporal jitter during training.
            boundaries = np.linspace(0, n_frames, self.num_frames + 1).astype(np.int64)
            indices = []
            for i in range(self.num_frames):
                start = int(boundaries[i])
                end = int(boundaries[i + 1])
                end = max(end, start + 1)
                indices.append(random.randint(start, end - 1))
            indices = np.clip(indices, 0, n_frames - 1)
            return np.asarray(indices, dtype=np.int64)

        indices = np.linspace(0, n_frames - 1, self.num_frames)
        return np.round(indices).astype(np.int64)

    def __getitem__(self, idx):
        video_name, frames, label = self.samples[idx]
        indices = self._sample_indices(len(frames))
        imgs = []
        for frame_idx in indices:
            img = Image.open(frames[int(frame_idx)]).convert("RGB")
            imgs.append(self.transform(img))

        # [T, C, H, W]
        video = torch.stack(imgs, dim=0)
        return video, torch.tensor(label, dtype=torch.long), video_name


def build_mobilenet_backbone(name="mobilenet_v3_small", pretrained=True):
    if name == "mobilenet_v3_small":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
    elif name == "mobilenet_v3_large":
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
    elif name == "mobilenet_v2":
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    feature_dim = model.classifier[0].in_features
    backbone = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    return backbone, feature_dim


class SelfAttentionPooling(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [B, T, D]
        scores = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MobileNetAttnPoolClassifier(nn.Module):
    def __init__(
        self,
        backbone_name="mobilenet_v3_small",
        num_frames=128,
        num_classes=2,
        pretrained=True,
        temporal_layers=1,
        num_heads=4,
        dropout=0.2,
        frame_batch_size=32,
    ):
        super().__init__()
        self.backbone, feature_dim = build_mobilenet_backbone(backbone_name, pretrained)
        self.backbone_name = backbone_name
        self.frame_batch_size = frame_batch_size
        self.backbone_frozen = False
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, feature_dim))

        if temporal_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=temporal_layers,
            )
        else:
            self.temporal_encoder = nn.Identity()
        self.attn_pool = SelfAttentionPooling(feature_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def extract_frame_features(self, frames):
        # frames: [B, T, C, H, W]
        bsz, num_frames, c, h, w = frames.shape
        flat = frames.view(bsz * num_frames, c, h, w)
        features = []
        grad_context = torch.no_grad() if self.backbone_frozen else torch.enable_grad()
        with grad_context:
            for start in range(0, flat.size(0), self.frame_batch_size):
                end = start + self.frame_batch_size
                features.append(self.backbone(flat[start:end]))
        features = torch.cat(features, dim=0)
        return features.view(bsz, num_frames, -1)

    def forward(self, frames, return_attention=False):
        x = self.extract_frame_features(frames)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.temporal_encoder(x)
        pooled, weights = self.attn_pool(x)
        logits = self.classifier(pooled)
        if return_attention:
            return logits, weights
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True
        self.backbone.eval()

    def unfreeze_backbone_last_layers(self, num_layers=2):
        for param in self.backbone.parameters():
            param.requires_grad = False

        feature_layers = self.backbone[0]
        if not hasattr(feature_layers, "__len__"):
            return

        num_layers = max(1, min(num_layers, len(feature_layers)))
        for layer in feature_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.backbone_frozen = False

    def trainable_parameter_groups(self, backbone_lr, head_lr, weight_decay):
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        groups = []
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                    "name": "backbone",
                }
            )
        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                    "name": "head",
                }
            )
        return groups


def build_optimizer(model, backbone_lr, head_lr, weight_decay):
    param_groups = model.trainable_parameter_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )
    return torch.optim.AdamW(param_groups)


def describe_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    if getattr(model, "backbone_frozen", False):
        model.backbone.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for videos, labels, _ in tqdm(loader, desc="Train", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(videos)
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
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for videos, labels, _ in tqdm(loader, desc="Val", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(videos)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
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
    parser.add_argument("--num_frames", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--frame_batch_size", type=int, default=32)
    parser.add_argument("--temporal_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=10)
    parser.add_argument("--unfreeze_last_layers", type=int, default=2)
    parser.add_argument("--no_unfreeze", action="store_true")
    parser.add_argument("--save_path", type=str, default="best_mobilenet_attnpool_video_level.pth")
    parser.add_argument("--log_path", type=str, default="mobilenet_attnpool_train_log.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_dataset = AphidSparseFrameDataset(
        Path(args.data_root) / "train",
        num_frames=args.num_frames,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = AphidSparseFrameDataset(
        Path(args.data_root) / "val",
        num_frames=args.num_frames,
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
        f"Sampling: {args.num_frames} sparse frames/video, "
        f"backbone={args.backbone}",
        flush=True,
    )

    model = MobileNetAttnPoolClassifier(
        backbone_name=args.backbone,
        num_frames=args.num_frames,
        num_classes=2,
        pretrained=args.pretrained,
        temporal_layers=args.temporal_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        frame_batch_size=args.frame_batch_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    model.freeze_backbone()
    optimizer = build_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    trainable, total = describe_trainable_params(model)
    print(
        f"Stage 1: frozen MobileNet backbone. "
        f"Trainable params: {trainable / 1e6:.3f}M / {total / 1e6:.3f}M",
        flush=True,
    )

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
                "stage",
                "head_lr",
                "backbone_lr",
            ]
        )

    best_f1 = -1.0
    current_stage = "frozen_backbone"
    for epoch in range(1, args.epochs + 1):
        if (
            epoch == args.freeze_backbone_epochs + 1
            and not args.no_unfreeze
            and args.unfreeze_last_layers > 0
        ):
            model.unfreeze_backbone_last_layers(args.unfreeze_last_layers)
            optimizer = build_optimizer(
                model,
                backbone_lr=args.backbone_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay,
            )
            current_stage = f"unfreeze_last_{args.unfreeze_last_layers}"
            trainable, total = describe_trainable_params(model)
            print(
                f"Stage 2: unfreeze last {args.unfreeze_last_layers} MobileNet feature layers. "
                f"Trainable params: {trainable / 1e6:.3f}M / {total / 1e6:.3f}M",
                flush=True,
            )

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        metrics = evaluate(model, val_loader, device)

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
                    current_stage,
                    args.head_lr,
                    args.backbone_lr if current_stage != "frozen_backbone" else 0.0,
                ]
            )
        print(f"Logged epoch metrics to {log_path}", flush=True)


if __name__ == "__main__":
    main()
