import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import roi_align
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm


FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASS_NAMES = ["nobirth", "start", "birth"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_name(name):
    return Path(str(name)).stem.lower()


def list_frame_dirs(frame_root):
    frame_root = Path(frame_root)
    return sorted([p for p in frame_root.iterdir() if p.is_dir()])


def list_frame_images(frame_dir):
    frame_dir = Path(frame_dir)
    return sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in FRAME_EXTS])


def sample_uniform_indices(n_frames, num_frames):
    n_frames = max(int(n_frames), 1)
    if n_frames == 1:
        return np.zeros((num_frames,), dtype=np.int64)
    return np.linspace(0, n_frames - 1, num_frames).round().astype(np.int64)


def resize_box(box, orig_w, orig_h, target_size):
    x1, y1, x2, y2 = box
    sx = target_size / float(orig_w)
    sy = target_size / float(orig_h)
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def flip_box_h(box, orig_w):
    x1, y1, x2, y2 = box
    return orig_w - x2, y1, orig_w - x1, y2


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    return x1, y1, x2, y2


def interpolate_box_at_frame(keyframes, frame_idx):
    if not keyframes:
        return None
    if frame_idx in keyframes:
        return keyframes[frame_idx]
    frames = sorted(keyframes.keys())
    if frame_idx <= frames[0]:
        return keyframes[frames[0]]
    if frame_idx >= frames[-1]:
        return keyframes[frames[-1]]
    left = max(f for f in frames if f < frame_idx)
    right = min(f for f in frames if f > frame_idx)
    lbox = np.asarray(keyframes[left], dtype=np.float32)
    rbox = np.asarray(keyframes[right], dtype=np.float32)
    alpha = (frame_idx - left) / float(right - left)
    box = (1.0 - alpha) * lbox + alpha * rbox
    return tuple(float(x) for x in box.tolist())


def load_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_annotation_index(annotation_dir):
    annotation_dir = Path(annotation_dir)
    csv_files = sorted(annotation_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {annotation_dir}")
    index = defaultdict(list)
    for csv_path in csv_files:
        rows = load_csv_rows(csv_path)
        if rows and "video_name" in rows[0]:
            for row in rows:
                index[normalize_name(row["video_name"])].append(row)
        else:
            index[normalize_name(csv_path.stem)].extend(rows)
    return index


def group_rows_by_instance(rows):
    groups = defaultdict(list)
    for row in rows:
        inst = str(row.get("instance_id", "")).strip() or "0"
        groups[inst].append(row)
    return groups


def select_label(rows, class_names):
    labels = [str(r.get("label", "")).strip().lower() for r in rows if str(r.get("label", "")).strip()]
    if not labels:
        return None
    label = Counter(labels).most_common(1)[0][0]
    for idx, name in enumerate(class_names):
        if label == name.lower():
            return idx
    return None


class ClipAugment:
    def __init__(self, train=True, image_size=224, flip_p=0.5):
        self.train = train
        self.image_size = image_size
        self.flip_p = flip_p

    def __call__(self, frames):
        if not self.train:
            return [img.resize((self.image_size, self.image_size), Image.BILINEAR) for img in frames], False

        do_flip = random.random() < self.flip_p
        brightness = random.uniform(0.85, 1.15)
        contrast = random.uniform(0.85, 1.15)
        saturation = random.uniform(0.9, 1.1)
        blur = random.random() < 0.10

        out = []
        for img in frames:
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = ImageEnhance.Brightness(img).enhance(brightness)
            img = ImageEnhance.Contrast(img).enhance(contrast)
            img = ImageEnhance.Color(img).enhance(saturation)
            if blur:
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            out.append(img)
        return out, do_flip


class AphidFrameRoiDataset(Dataset):
    def __init__(self, frame_dir, annotation_dir, class_names, num_frames=30, image_size=224, train=True):
        self.frame_dir = Path(frame_dir)
        self.annotation_dir = Path(annotation_dir)
        self.class_names = class_names
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train
        self.augment = ClipAugment(train=train, image_size=image_size)
        self.anno_index = load_annotation_index(self.annotation_dir)
        self.skipped_unlabeled = []
        self.skipped_empty = []
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.frame_dir}")

    def _build_samples(self):
        samples = []
        for clip_dir in list_frame_dirs(self.frame_dir):
            rows = self.anno_index.get(normalize_name(clip_dir.stem), [])
            if not rows:
                self.skipped_unlabeled.append(clip_dir.name)
                continue
            frame_paths = list_frame_images(clip_dir)
            if not frame_paths:
                self.skipped_empty.append(clip_dir.name)
                continue
            for instance_id, instance_rows in group_rows_by_instance(rows).items():
                label_idx = select_label(instance_rows, self.class_names)
                if label_idx is None:
                    continue
                samples.append(
                    {
                        "clip_dir": clip_dir,
                        "frame_paths": frame_paths,
                        "rows": instance_rows,
                        "label": label_idx,
                        "instance_id": instance_id,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def labels(self):
        return [s["label"] for s in self.samples]

    def _build_keyframes(self, rows):
        keyframes = {}
        for row in rows:
            try:
                frame_idx = int(float(row["frame_idx"]))
                box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
                keyframes[frame_idx] = box
            except Exception:
                continue
        return keyframes

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]
        keyframes = self._build_keyframes(sample["rows"])
        indices = sample_uniform_indices(len(frame_paths), self.num_frames)

        frames = [Image.open(frame_paths[int(i)]).convert("RGB") for i in indices]
        orig_w, orig_h = frames[0].size
        boxes = [interpolate_box_at_frame(keyframes, int(i)) for i in indices]

        frames, do_flip = self.augment(frames)

        tensors = []
        proc_boxes = []
        for img, box in zip(frames, boxes):
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensors.append(tensor)

            if box is None:
                proc_boxes.append(None)
                continue
            if do_flip:
                box = flip_box_h(box, orig_w)
            box = resize_box(box, orig_w, orig_h, self.image_size)
            x1, y1, x2, y2 = clip_box(box, self.image_size, self.image_size)
            if x2 <= x1 or y2 <= y1:
                proc_boxes.append(None)
            else:
                proc_boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))

        frames_tensor = torch.stack(tensors, dim=0)  # [T,C,H,W]
        return frames_tensor, torch.tensor(sample["label"], dtype=torch.long), proc_boxes, sample["clip_dir"].name, sample["instance_id"]


class ResNet50RoIAlignLSTM(nn.Module):
    def __init__(
        self,
        num_classes=3,
        pretrained=True,
        weight_path=None,
        roi_out=7,
        lstm_hidden=256,
        lstm_layers=1,
        dropout=0.3,
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained and weight_path is None else None
        base = resnet50(weights=weights)
        if weight_path:
            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {k.replace("module.", ""): v for k, v in state.items()}
            base.load_state_dict(state, strict=False)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.roi_out = roi_out
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, frames, box_seqs):
        # frames: [B,T,C,H,W]
        bsz, t, c, h, w = frames.shape
        x = frames.view(bsz * t, c, h, w)
        feat = self.backbone(x)  # [B*T,2048,Hf,Wf]
        _, feat_c, feat_h, feat_w = feat.shape
        feat = feat.view(bsz, t, feat_c, feat_h, feat_w)
        spatial_scale = feat_w / float(w)
        device = frames.device

        seq_feats = []
        for b in range(bsz):
            per_t = []
            for ti in range(t):
                box = box_seqs[b][ti]
                fmap = feat[b, ti].unsqueeze(0)
                if box is None:
                    pooled = fmap.mean(dim=(2, 3)).squeeze(0)
                else:
                    roi = roi_align(
                        fmap,
                        [box.to(device).unsqueeze(0)],
                        output_size=(self.roi_out, self.roi_out),
                        spatial_scale=spatial_scale,
                        aligned=True,
                    )
                    pooled = roi.mean(dim=(2, 3)).squeeze(0)
                per_t.append(pooled)
            seq_feats.append(torch.stack(per_t, dim=0))

        seq_feats = torch.stack(seq_feats, dim=0)  # [B,T,2048]
        lstm_out, _ = self.lstm(seq_feats)
        video_feat = lstm_out.mean(dim=1)
        return self.classifier(video_feat)


def collate_fn(batch):
    frames, labels, boxes, clip_names, instance_ids = zip(*batch)
    return torch.stack(frames, dim=0), torch.stack(labels, dim=0), list(boxes), list(clip_names), list(instance_ids)


def compute_metrics(labels, preds, probs, class_names):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )
    y_true = np.zeros((len(labels), len(class_names)), dtype=np.int64)
    if len(labels):
        y_true[np.arange(len(labels)), labels] = 1
    class_ap = []
    for c in range(len(class_names)):
        if y_true[:, c].sum() == 0:
            class_ap.append(float("nan"))
        else:
            class_ap.append(average_precision_score(y_true[:, c], probs[:, c]))
    valid_ap = [x for x in class_ap if not np.isnan(x)]
    return {
        "acc": accuracy_score(labels, preds) if len(labels) else 0.0,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0) if len(labels) else 0.0,
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0) if len(labels) else 0.0,
        "mAP": float(np.mean(valid_ap)) if valid_ap else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "ap": class_ap,
    }


def make_class_weights(labels, num_classes):
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []
    for frames, labels, boxes, _, _ in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True):
        frames = frames.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(frames, boxes)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return total_loss / max(1, len(loader.dataset)), accuracy_score(all_labels, all_preds) if all_labels else 0.0


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for frames, labels, boxes, _, _ in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        frames = frames.to(device)
        labels = labels.to(device)
        logits = model(frames, boxes)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return compute_metrics(all_labels, all_preds, all_probs, class_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_frame_dir", required=True)
    parser.add_argument("--train_anno_dir", required=True)
    parser.add_argument("--val_frame_dir", required=True)
    parser.add_argument("--val_anno_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resnet_weight_path", type=str, default=None)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--save_path", type=str, default="best_resnet50_roialign_lstm.pth")
    parser.add_argument("--log_path", type=str, default="resnet50_roialign_lstm_train_log.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_names", type=str, default=",".join(DEFAULT_CLASS_NAMES))
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    save_path = Path(args.save_path)
    log_path = Path(args.log_path)
    if not save_path.is_absolute():
        save_path = script_dir / save_path
    if not log_path.is_absolute():
        log_path = script_dir / log_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Classes: {class_names}", flush=True)
    print(f"Sampling: {args.num_frames} uniformly sampled frames/sample", flush=True)

    train_dataset = AphidFrameRoiDataset(
        args.train_frame_dir, args.train_anno_dir, class_names, args.num_frames, args.image_size, train=True
    )
    val_dataset = AphidFrameRoiDataset(
        args.val_frame_dir, args.val_anno_dir, class_names, args.num_frames, args.image_size, train=False
    )
    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Val samples: {len(val_dataset)}", flush=True)
    print(f"Train unlabeled clips skipped: {len(train_dataset.skipped_unlabeled)}", flush=True)
    print(f"Val unlabeled clips skipped: {len(val_dataset.skipped_unlabeled)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = ResNet50RoIAlignLSTM(
        num_classes=len(class_names),
        pretrained=args.pretrained,
        weight_path=args.resnet_weight_path,
    ).to(device)

    if args.use_class_weights:
        weights = make_class_weights(train_dataset.labels(), len(class_names)).to(device)
        print(f"Class weights: {weights.detach().cpu().numpy().tolist()}", flush=True)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    header = ["epoch", "train_loss", "train_acc", "val_acc", "val_macro_f1", "val_micro_f1", "val_mAP", "best_mAP"]
    for c in class_names:
        header.extend([f"{c}_AP", f"{c}_P", f"{c}_R", f"{c}_F1", f"{c}_support"])
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    best_map = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device, class_names)
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={metrics['acc']:.4f} val_macro_f1={metrics['macro_f1']:.4f} "
            f"val_micro_f1={metrics['micro_f1']:.4f} val_mAP={metrics['mAP']:.4f}",
            flush=True,
        )
        for idx, c in enumerate(class_names):
            print(
                f"  {c}: AP={metrics['ap'][idx]:.4f} P={metrics['precision'][idx]:.4f} "
                f"R={metrics['recall'][idx]:.4f} F1={metrics['f1'][idx]:.4f} N={int(metrics['support'][idx])}",
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
                save_path,
            )
            print(f"Saved best model to {save_path}", flush=True)

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
        for idx in range(len(class_names)):
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
