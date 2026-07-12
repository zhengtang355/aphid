import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
import torch.nn.functional as F
from tqdm import tqdm


DEFAULT_CLASSES = ["nobirth", "start", "birth"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


DEFAULT_CONFIG = {
    "model_name": "resnet50_lstm",
    "data_root": r"C:\数据\data",
    "train_dir": None,
    "val_dir": None,
    "class_names": DEFAULT_CLASSES,
    "class_map": None,
    "epochs": 30,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "num_workers": 0,
    "image_size": 224,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "momentum": 0.9,
    "nesterov": False,
    "use_lr_scheduler": True,
    "lr_scheduler": "plateau",
    "lr_decay_patience": 5,
    "lr_decay_min_delta": 0.001,
    "lr_decay_factor": 0.1,
    "min_lr": 1e-8,
    "dropout": 0.3,
    "label_smoothing": 0.0,
    "loss_name": "ce",
    "focal_gamma": 2.0,
    "focal_lambda": 0.5,
    "crcl_lambda": 0.3,
    "crcl_positive_classes": ["start", "birth"],
    "rscl_exist_lambda": 0.2,
    "rscl_change_lambda": 0.2,
    "bce_soft_targets": None,
    "pretrained": True,
    "resnet_weight_path": r"C:\Users\zhengtang\.cache\torch\hub\checkpoints\resnet50-11ad3fa6.pth",
    "use_class_weights": True,
    "random_horizontal_flip": True,
    "random_horizontal_flip_p": 0.5,
    "random_vertical_flip": True,
    "random_vertical_flip_p": 0.2,
    "random_rotation": True,
    "random_rotation_degrees": 8,
    "random_rotation_p": 0.5,
    "random_affine": True,
    "random_affine_p": 0.5,
    "random_translate": 0.03,
    "random_scale_min": 0.95,
    "random_scale_max": 1.05,
    "random_shear_degrees": 3,
    "gaussian_blur": True,
    "gaussian_blur_p": 0.2,
    "gaussian_blur_kernel": 3,
    "gaussian_blur_sigma_min": 0.1,
    "gaussian_blur_sigma_max": 1.0,
    "gaussian_noise": True,
    "gaussian_noise_p": 0.2,
    "gaussian_noise_std": 0.01,
    "sampling": "uniform",
    "num_frames": 30,
    "frame_stride": 2,
    "clip_batch_size": 1,
    "lstm_hidden": 256,
    "transformer_layers": 2,
    "transformer_heads": 8,
    "videomae_model_name": "MCG-NJU/videomae-base-finetuned-kinetics",
    "videomae_dual_end_token_attention": False,
    "videomae_token_attention_alpha": 0.2,
    "timesformer_model_name": "facebook/timesformer-base-finetuned-k400",
    "videoswin_model_name": "MCG-NJU/videoswin-base-finetuned-kinetics400",
    "slowfast_no_last_downsample": False,
    "slowfast_alpha": 4,
    "num_segments": 3,
    "segment_temporal_module": "lstm",
    "segment_lstm_hidden": 256,
    "segment_lstm_layers": 1,
    "slowfast_bilstm": False,
    "slowfast_bilstm_hidden": 256,
    "slowfast_bilstm_layers": 1,
    "slowfast_temporal_transformer": False,
    "slowfast_temporal_transformer_layers": 2,
    "slowfast_temporal_transformer_heads": 8,
    "slowfast_temporal_transformer_dim": 512,
    "slowfast_temporal_difference_attention": False,
    "temporal_difference_attention_dim": 256,
    "temporal_difference_attention_alpha": 1.0,
    "temporal_difference_attention_dropout": 0.1,
    "temporal_difference_use_dual_end_mask": False,
    "slowfast_tail_attention": False,
    "slowfast_dual_end_attention": False,
    "tail_attention_alpha": 0.8,
    "tail_attention_alpha_slow": None,
    "tail_attention_alpha_fast": None,
    "tail_attention_stage": 4,
    "dual_end_attention_stages": [0, 1, 2, 3, 4, 5],
    "dual_end_attention_pathways": "both",
    "tail_end_ratio": 1.0 / 3.0,
    "slowfast_auxiliary_reproductive": False,
    "aux_main_lambda": 1.0,
    "aux_existence_lambda": 0.5,
    "aux_change_lambda": 0.5,
    "hierarchical_branch_backbone": "resnet18",
    "hierarchical_branch_pretrained": True,
    "hierarchical_branch_weight_path": None,
    "hierarchical_use_sampled_endpoints": True,
    "output_root": "runs",
    "run_name": None,
    "best_metric": "val_mAP",
    "save_path": None,
    "log_path": None,
    "seed": 42,
}


MODEL_DEFAULTS = {
    "resnet50_lstm": {"sampling": "uniform", "num_frames": 30, "image_size": 224},
    "resnet50_bilstm": {"sampling": "uniform", "num_frames": 30, "image_size": 224},
    "resnet50_transformer": {"sampling": "uniform", "num_frames": 30, "image_size": 224},
    "c3d": {"sampling": "consecutive", "num_frames": 16, "frame_stride": 1, "image_size": 112},
    "r3d18": {"sampling": "consecutive", "num_frames": 16, "frame_stride": 1, "image_size": 112},
    "x3d_s": {"sampling": "consecutive", "num_frames": 16, "frame_stride": 1, "image_size": 160},
    "i3d": {"sampling": "consecutive", "num_frames": 32, "frame_stride": 2, "image_size": 224},
    "slowfast": {"sampling": "consecutive", "num_frames": 64, "frame_stride": 1, "image_size": 224},
    "slowfast_segment_lstm": {"sampling": "segment_consecutive", "num_frames": 64, "frame_stride": 1, "image_size": 224},
    "slowfast_state_aggregation": {"sampling": "segment_consecutive", "num_frames": 32, "frame_stride": 2, "image_size": 224},
    "videomae": {"sampling": "consecutive", "num_frames": 16, "frame_stride": 2, "image_size": 224},
    "timesformer": {"sampling": "consecutive", "num_frames": 8, "frame_stride": 8, "image_size": 224},
    "video_swin": {"sampling": "consecutive", "num_frames": 32, "frame_stride": 2, "image_size": 224},
    "video_swin_b": {"sampling": "consecutive", "num_frames": 32, "frame_stride": 2, "image_size": 224},
    "mvit_v2_s": {"sampling": "consecutive", "num_frames": 16, "frame_stride": 4, "image_size": 224},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    for key in ("data_root", "train_dir", "val_dir"):
        value = cfg.get(key)
        if not value:
            continue
        text = str(value)
        fixed = text.replace(r"C:\方象", r"C:\数据").replace(r"C:\鏁版嵁", r"C:\数据")
        if fixed != text and not Path(text).exists() and Path(fixed).exists():
            print(f"Fixed config path {key}: {text} -> {fixed}", flush=True)
            cfg[key] = fixed
    model_name = cfg["model_name"]
    if model_name not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model_name: {model_name}. Choices: {sorted(MODEL_DEFAULTS)}")
    for k, v in MODEL_DEFAULTS[model_name].items():
        if k not in cfg or cfg[k] is None:
            cfg[k] = v
    return SimpleNamespace(**cfg)


def list_frames(folder):
    frames = []
    for ext in IMAGE_EXTS:
        frames.extend(Path(folder).glob(f"*{ext}"))
    return sorted(frames)


def make_dual_end_mask(rgb, end_ratio=1.0 / 3.0):
    arr = np.asarray(rgb.convert("RGB"))
    h, w = arr.shape[:2]
    horizontal = w >= h
    mask = np.zeros((h, w), dtype=np.uint8)
    if horizontal:
        end_len = max(1, int(round(w * end_ratio)))
        mask[:, :end_len] = 1
        mask[:, max(0, w - end_len):] = 1
    else:
        end_len = max(1, int(round(h * end_ratio)))
        mask[:end_len, :] = 1
        mask[max(0, h - end_len):, :] = 1
    return Image.fromarray((mask * 255).astype(np.uint8), mode="L")


def extract_dual_end_patches(rgb, end_ratio=1.0 / 3.0):
    arr = np.asarray(rgb.convert("RGB"))
    h, w = arr.shape[:2]
    horizontal = w >= h
    if horizontal:
        end_len = max(1, int(round(w * end_ratio)))
        patch_a = arr[:, :end_len, :]
        patch_b = arr[:, max(0, w - end_len):, :]
    else:
        end_len = max(1, int(round(h * end_ratio)))
        patch_a = arr[:end_len, :, :]
        patch_b = arr[max(0, h - end_len):, :, :]
    return Image.fromarray(patch_a), Image.fromarray(patch_b)


class CroppedInstanceDataset(Dataset):
    def __init__(
        self,
        root_dir,
        class_names,
        num_frames,
        image_size,
        sampling,
        frame_stride,
        train=True,
        class_map=None,
        return_attention_masks=False,
        tail_end_ratio=1.0 / 3.0,
        return_endpoint_patches=False,
        use_sampled_endpoints=True,
        return_segments=False,
        num_segments=3,
        random_rotation=False,
        random_rotation_degrees=8,
        random_rotation_p=0.5,
        random_horizontal_flip=True,
        random_horizontal_flip_p=0.5,
        random_vertical_flip=False,
        random_vertical_flip_p=0.2,
        random_affine=False,
        random_affine_p=0.5,
        random_translate=0.03,
        random_scale_min=0.95,
        random_scale_max=1.05,
        random_shear_degrees=3,
        gaussian_blur=False,
        gaussian_blur_p=0.2,
        gaussian_blur_kernel=3,
        gaussian_blur_sigma_min=0.1,
        gaussian_blur_sigma_max=1.0,
        gaussian_noise=False,
        gaussian_noise_p=0.2,
        gaussian_noise_std=0.01,
    ):
        self.root_dir = Path(root_dir)
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.num_frames = int(num_frames)
        self.sampling = sampling
        self.frame_stride = int(frame_stride)
        self.train = train
        self.class_map = class_map or {}
        self.return_attention_masks = return_attention_masks
        self.tail_end_ratio = tail_end_ratio
        self.return_endpoint_patches = return_endpoint_patches
        self.use_sampled_endpoints = use_sampled_endpoints
        self.return_segments = return_segments
        self.num_segments = int(num_segments)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.random_horizontal_flip_p = float(random_horizontal_flip_p)
        self.random_vertical_flip = bool(random_vertical_flip)
        self.random_vertical_flip_p = float(random_vertical_flip_p)
        self.random_rotation = bool(random_rotation)
        self.random_rotation_degrees = float(random_rotation_degrees)
        self.random_rotation_p = float(random_rotation_p)
        self.random_affine = bool(random_affine)
        self.random_affine_p = float(random_affine_p)
        self.random_translate = float(random_translate)
        self.random_scale_min = float(random_scale_min)
        self.random_scale_max = float(random_scale_max)
        self.random_shear_degrees = float(random_shear_degrees)
        self.gaussian_blur = bool(gaussian_blur)
        self.gaussian_blur_p = float(gaussian_blur_p)
        self.gaussian_blur_kernel = int(gaussian_blur_kernel)
        if self.gaussian_blur_kernel % 2 == 0:
            self.gaussian_blur_kernel += 1
        self.gaussian_blur_sigma_min = float(gaussian_blur_sigma_min)
        self.gaussian_blur_sigma_max = float(gaussian_blur_sigma_max)
        self.gaussian_noise = bool(gaussian_noise)
        self.gaussian_noise_p = float(gaussian_noise_p)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.samples = []

        if self.class_map:
            source_folders = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        else:
            source_folders = [self.root_dir / class_name for class_name in class_names]

        for class_dir in source_folders:
            if not class_dir.exists():
                continue
            source_name = class_dir.name
            target_name = self.class_map.get(source_name, source_name)
            if target_name not in self.class_to_idx:
                continue
            for instance_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()]):
                frames = list_frames(instance_dir)
                if frames:
                    self.samples.append((instance_dir.name, frames, self.class_to_idx[target_name]))

        self.image_size = image_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)

    def _sample_aug_params(self):
        params = {
            "hflip": self.train and self.random_horizontal_flip and random.random() < self.random_horizontal_flip_p,
            "vflip": self.train and self.random_vertical_flip and random.random() < self.random_vertical_flip_p,
            "rotate": self.train and self.random_rotation and random.random() < self.random_rotation_p,
            "affine": self.train and self.random_affine and random.random() < self.random_affine_p,
            "blur": self.train and self.gaussian_blur and random.random() < self.gaussian_blur_p,
            "noise": self.train and self.gaussian_noise and random.random() < self.gaussian_noise_p,
        }
        params["rotation_angle"] = (
            random.uniform(-self.random_rotation_degrees, self.random_rotation_degrees)
            if params["rotate"]
            else 0.0
        )
        params["affine_angle"] = (
            random.uniform(-self.random_rotation_degrees, self.random_rotation_degrees)
            if params["affine"]
            else 0.0
        )
        max_translate = int(round(self.random_translate * self.image_size))
        params["translate"] = (
            (
                random.randint(-max_translate, max_translate),
                random.randint(-max_translate, max_translate),
            )
            if params["affine"] and max_translate > 0
            else (0, 0)
        )
        params["scale"] = (
            random.uniform(self.random_scale_min, self.random_scale_max)
            if params["affine"]
            else 1.0
        )
        params["shear"] = (
            [
                random.uniform(-self.random_shear_degrees, self.random_shear_degrees),
                random.uniform(-self.random_shear_degrees, self.random_shear_degrees),
            ]
            if params["affine"]
            else [0.0, 0.0]
        )
        params["blur_sigma"] = (
            random.uniform(self.gaussian_blur_sigma_min, self.gaussian_blur_sigma_max)
            if params["blur"]
            else self.gaussian_blur_sigma_min
        )
        return params

    def _apply_pil_aug(self, img, params, is_mask=False):
        interpolation = (
            transforms.InterpolationMode.NEAREST
            if is_mask
            else transforms.InterpolationMode.BILINEAR
        )
        fill = 0
        if params["hflip"]:
            img = transforms.functional.hflip(img)
        if params["vflip"]:
            img = transforms.functional.vflip(img)
        if params["rotate"]:
            img = transforms.functional.rotate(
                img,
                params["rotation_angle"],
                interpolation=interpolation,
                fill=fill,
            )
        if params["affine"]:
            img = transforms.functional.affine(
                img,
                angle=params["affine_angle"],
                translate=params["translate"],
                scale=params["scale"],
                shear=params["shear"],
                interpolation=interpolation,
                fill=fill,
            )
        if (not is_mask) and params["blur"]:
            img = transforms.functional.gaussian_blur(
                img,
                kernel_size=[self.gaussian_blur_kernel, self.gaussian_blur_kernel],
                sigma=[params["blur_sigma"], params["blur_sigma"]],
            )
        return img

    def _apply_tensor_aug(self, img_t, params):
        if params["noise"]:
            noise = torch.randn_like(img_t) * self.gaussian_noise_std
            img_t = (img_t + noise).clamp(0.0, 1.0)
        return img_t

    def __len__(self):
        return len(self.samples)

    def labels(self):
        return [x[2] for x in self.samples]

    def _sample_indices(self, n):
        if n <= 0:
            return np.zeros(self.num_frames, dtype=np.int64)

        if self.sampling == "uniform":
            if n == 1:
                return np.zeros(self.num_frames, dtype=np.int64)
            return np.linspace(0, n - 1, self.num_frames).round().astype(np.int64)

        span = (self.num_frames - 1) * self.frame_stride + 1
        if n >= span:
            if self.train:
                start = random.randint(0, n - span)
            else:
                start = (n - span) // 2
            return np.asarray([start + i * self.frame_stride for i in range(self.num_frames)], dtype=np.int64)

        indices = np.arange(0, n, self.frame_stride, dtype=np.int64)
        if len(indices) == 0:
            indices = np.asarray([0], dtype=np.int64)
        if len(indices) < self.num_frames:
            pad = np.full(self.num_frames - len(indices), indices[-1], dtype=np.int64)
            indices = np.concatenate([indices, pad], axis=0)
        return indices[: self.num_frames]

    def _sample_segment_indices(self, n):
        all_indices = []
        if n <= 0:
            return [np.zeros(self.num_frames, dtype=np.int64) for _ in range(self.num_segments)]
        span = (self.num_frames - 1) * self.frame_stride + 1
        for seg_idx in range(self.num_segments):
            seg_start = int(round(seg_idx * n / self.num_segments))
            seg_end = int(round((seg_idx + 1) * n / self.num_segments))
            seg_start = max(0, min(seg_start, n - 1))
            seg_end = max(seg_start + 1, min(seg_end, n))
            seg_len = seg_end - seg_start
            if seg_len >= span:
                max_start = seg_end - span
                if self.train:
                    start = random.randint(seg_start, max_start)
                else:
                    start = seg_start + (seg_len - span) // 2
                indices = np.asarray([start + i * self.frame_stride for i in range(self.num_frames)], dtype=np.int64)
            else:
                indices = np.arange(seg_start, seg_end, self.frame_stride, dtype=np.int64)
                if len(indices) == 0:
                    indices = np.asarray([seg_start], dtype=np.int64)
                if len(indices) < self.num_frames:
                    pad = np.full(self.num_frames - len(indices), indices[-1], dtype=np.int64)
                    indices = np.concatenate([indices, pad], axis=0)
                indices = indices[: self.num_frames]
            all_indices.append(np.clip(indices, 0, n - 1))
        return all_indices

    def __getitem__(self, idx):
        name, frames, label = self.samples[idx]
        if self.return_segments:
            segment_indices = self._sample_segment_indices(len(frames))
            aug_params = self._sample_aug_params()
            segments = []
            for indices in segment_indices:
                imgs = []
                for i in indices:
                    img = Image.open(frames[int(i)]).convert("RGB")
                    img = transforms.Resize((self.image_size, self.image_size))(img)
                    img = self._apply_pil_aug(img, aug_params, is_mask=False)
                    if self.train:
                        img = self.color_jitter(img)
                    img_t = transforms.functional.to_tensor(img)
                    img_t = self._apply_tensor_aug(img_t, aug_params)
                    imgs.append(self.normalize(img_t))
                segments.append(torch.stack(imgs, dim=0))
            video = torch.stack(segments, dim=0)
            return video, torch.tensor(label, dtype=torch.long), name
        indices = self._sample_indices(len(frames))
        imgs = []
        masks = []
        aug_params = self._sample_aug_params()
        first_endpoint_img = None
        last_endpoint_img = None
        for i in indices:
            img = Image.open(frames[int(i)]).convert("RGB")
            if first_endpoint_img is None:
                first_endpoint_img = img.copy()
            last_endpoint_img = img.copy()
            mask = (
                make_dual_end_mask(
                    img,
                    end_ratio=self.tail_end_ratio,
                )
                if self.return_attention_masks
                else None
            )
            img = transforms.Resize((self.image_size, self.image_size))(img)
            if mask is not None:
                mask = transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            img = self._apply_pil_aug(img, aug_params, is_mask=False)
            if mask is not None:
                mask = self._apply_pil_aug(mask, aug_params, is_mask=True)
            if self.train:
                img = self.color_jitter(img)
            img_t = transforms.functional.to_tensor(img)
            img_t = self._apply_tensor_aug(img_t, aug_params)
            imgs.append(self.normalize(img_t))
            if mask is not None:
                masks.append(transforms.functional.to_tensor(mask))
        video = torch.stack(imgs, dim=0)
        endpoint_tensors = None
        if self.return_endpoint_patches:
            if not self.use_sampled_endpoints:
                first_endpoint_img = Image.open(frames[0]).convert("RGB")
                last_endpoint_img = Image.open(frames[-1]).convert("RGB")
            first_a, first_b = extract_dual_end_patches(first_endpoint_img, end_ratio=self.tail_end_ratio)
            last_a, last_b = extract_dual_end_patches(last_endpoint_img, end_ratio=self.tail_end_ratio)
            endpoint_tensors = []
            for patch in (first_a, first_b, last_a, last_b):
                patch = transforms.Resize((self.image_size, self.image_size))(patch)
                if do_flip:
                    patch = transforms.functional.hflip(patch)
                if self.train:
                    patch = self.color_jitter(patch)
                endpoint_tensors.append(self.normalize(transforms.functional.to_tensor(patch)))
            endpoint_tensors = tuple(endpoint_tensors)
        if self.return_attention_masks and self.return_endpoint_patches:
            attention_masks = torch.stack(masks, dim=0)
            return video, attention_masks, *endpoint_tensors, torch.tensor(label, dtype=torch.long), name
        if self.return_attention_masks:
            attention_masks = torch.stack(masks, dim=0)
            return video, attention_masks, torch.tensor(label, dtype=torch.long), name
        if self.return_endpoint_patches:
            return video, *endpoint_tensors, torch.tensor(label, dtype=torch.long), name
        return video, torch.tensor(label, dtype=torch.long), name


class ResNetTemporalBase(nn.Module):
    def __init__(self, num_classes, mode, cfg):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if cfg.pretrained and not cfg.resnet_weight_path else None
        self.backbone = resnet50(weights=weights)
        if cfg.resnet_weight_path:
            state = torch.load(cfg.resnet_weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.backbone.load_state_dict(state, strict=False)
            print(f"Loaded ResNet50 weights from: {cfg.resnet_weight_path}", flush=True)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.mode = mode

        if mode in {"lstm", "bilstm"}:
            bidirectional = mode == "bilstm"
            self.temporal = nn.LSTM(
                feat_dim,
                cfg.lstm_hidden,
                batch_first=True,
                bidirectional=bidirectional,
            )
            out_dim = cfg.lstm_hidden * (2 if bidirectional else 1)
        elif mode == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=cfg.transformer_heads,
                dim_feedforward=feat_dim * 2,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
            out_dim = feat_dim
        else:
            raise ValueError(mode)

        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, video):
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        feats = self.backbone(x).reshape(b, t, -1)
        if self.mode in {"lstm", "bilstm"}:
            out, _ = self.temporal(feats)
        else:
            out = self.temporal(feats)
        feat = out.mean(dim=1)
        return self.classifier(feat)


class SimpleC3D(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
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
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SlowFastDualEndAttentionWrapper(nn.Module):
    def __init__(
        self,
        model,
        alpha=0.8,
        stages=(0, 1, 2, 3, 4, 5),
        pathways="both",
        alpha_slow=None,
        alpha_fast=None,
    ):
        super().__init__()
        self.model = model
        self.alpha_slow = float(alpha if alpha_slow is None else alpha_slow)
        self.alpha_fast = float(alpha if alpha_fast is None else alpha_fast)
        self.stages = {int(stage) for stage in stages}
        self.pathways = str(pathways).lower()
        if self.pathways not in {"slow", "fast", "both"}:
            raise ValueError("dual_end_attention_pathways must be one of: slow, fast, both")

    def forward(self, pathways, attention_masks=None):
        x = pathways
        for idx, block in enumerate(self.model.blocks):
            x = block(x)
            if attention_masks is not None and idx in self.stages:
                x = self.apply_dual_end_attention(x, attention_masks)
        return x

    def apply_dual_end_attention(self, pathways, attention_masks):
        if not isinstance(pathways, (list, tuple)) or len(pathways) < 2:
            return pathways
        out = list(pathways)
        mask_3d = attention_masks.permute(0, 2, 1, 3, 4)
        if self.pathways in {"slow", "both"}:
            slow = out[0]
            slow_mask = mask_3d.to(device=slow.device, dtype=slow.dtype)
            slow_mask = F.interpolate(slow_mask, size=slow.shape[2:], mode="trilinear", align_corners=False)
            out[0] = slow * (1.0 + self.alpha_slow * slow_mask)
        if self.pathways in {"fast", "both"}:
            fast = out[1]
            fast_mask = mask_3d.to(device=fast.device, dtype=fast.dtype)
            fast_mask = F.interpolate(fast_mask, size=fast.shape[2:], mode="trilinear", align_corners=False)
            out[1] = fast * (1.0 + self.alpha_fast * fast_mask)
        return out


class SlowFastBiLSTMClassifier(nn.Module):
    def __init__(self, model, num_classes, cfg):
        super().__init__()
        self.model = model
        self.alpha_slow = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_slow", None) is None else cfg.tail_attention_alpha_slow)
        self.alpha_fast = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_fast", None) is None else cfg.tail_attention_alpha_fast)
        self.attention_stages = {int(stage) for stage in getattr(cfg, "dual_end_attention_stages", [])}
        self.attention_pathways = str(getattr(cfg, "dual_end_attention_pathways", "both")).lower()
        self.use_dual_end_attention = bool(
            getattr(cfg, "slowfast_dual_end_attention", False)
            or getattr(cfg, "slowfast_tail_attention", False)
        )
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0)))

        with torch.no_grad():
            dummy_fast = torch.zeros(1, 3, int(cfg.num_frames), int(cfg.image_size), int(cfg.image_size))
            dummy_slow = torch.index_select(
                dummy_fast,
                2,
                torch.linspace(0, dummy_fast.shape[2] - 1, max(1, dummy_fast.shape[2] // int(cfg.slowfast_alpha))).long(),
            )
            features = self.forward_backbone([dummy_slow, dummy_fast], attention_masks=None)
            feat_dim = int(features[0].shape[1] + features[1].shape[1])

        hidden = int(getattr(cfg, "slowfast_bilstm_hidden", 256))
        layers = int(getattr(cfg, "slowfast_bilstm_layers", 1))
        self.temporal = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=float(getattr(cfg, "dropout", 0.0)) if layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def apply_dual_end_attention(self, pathways, attention_masks):
        if attention_masks is None or not isinstance(pathways, (list, tuple)) or len(pathways) < 2:
            return pathways
        out = list(pathways)
        mask_3d = attention_masks.permute(0, 2, 1, 3, 4)
        if self.attention_pathways in {"slow", "both"}:
            slow = out[0]
            slow_mask = mask_3d.to(device=slow.device, dtype=slow.dtype)
            slow_mask = F.interpolate(slow_mask, size=slow.shape[2:], mode="trilinear", align_corners=False)
            out[0] = slow * (1.0 + self.alpha_slow * slow_mask)
        if self.attention_pathways in {"fast", "both"}:
            fast = out[1]
            fast_mask = mask_3d.to(device=fast.device, dtype=fast.dtype)
            fast_mask = F.interpolate(fast_mask, size=fast.shape[2:], mode="trilinear", align_corners=False)
            out[1] = fast * (1.0 + self.alpha_fast * fast_mask)
        return out

    def forward_backbone(self, pathways, attention_masks=None):
        x = pathways
        for idx, block in enumerate(self.model.blocks[:5]):
            x = block(x)
            if self.use_dual_end_attention and idx in self.attention_stages:
                x = self.apply_dual_end_attention(x, attention_masks)
        return x

    def pathway_to_sequence(self, pathways):
        seqs = []
        max_t = max(int(pathway.shape[2]) for pathway in pathways)
        for pathway in pathways:
            seq = pathway.mean(dim=(-1, -2)).transpose(1, 2)
            if seq.shape[1] != max_t:
                seq = F.interpolate(seq.transpose(1, 2), size=max_t, mode="linear", align_corners=False).transpose(1, 2)
            seqs.append(seq)
        return torch.cat(seqs, dim=-1)

    def forward(self, pathways, attention_masks=None):
        features = self.forward_backbone(pathways, attention_masks=attention_masks)
        sequence = self.pathway_to_sequence(features)
        output, _ = self.temporal(sequence)
        pooled = output.mean(dim=1)
        return self.classifier(self.dropout(pooled))


class SlowFastTemporalTransformerClassifier(SlowFastBiLSTMClassifier):
    def __init__(self, model, num_classes, cfg):
        nn.Module.__init__(self)
        self.model = model
        self.alpha_slow = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_slow", None) is None else cfg.tail_attention_alpha_slow)
        self.alpha_fast = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_fast", None) is None else cfg.tail_attention_alpha_fast)
        self.attention_stages = {int(stage) for stage in getattr(cfg, "dual_end_attention_stages", [])}
        self.attention_pathways = str(getattr(cfg, "dual_end_attention_pathways", "both")).lower()
        self.use_dual_end_attention = bool(
            getattr(cfg, "slowfast_dual_end_attention", False)
            or getattr(cfg, "slowfast_tail_attention", False)
        )
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0)))

        with torch.no_grad():
            dummy_fast = torch.zeros(1, 3, int(cfg.num_frames), int(cfg.image_size), int(cfg.image_size))
            dummy_slow = torch.index_select(
                dummy_fast,
                2,
                torch.linspace(0, dummy_fast.shape[2] - 1, max(1, dummy_fast.shape[2] // int(cfg.slowfast_alpha))).long(),
            )
            features = self.forward_backbone([dummy_slow, dummy_fast], attention_masks=None)
            feat_dim = int(features[0].shape[1] + features[1].shape[1])

        embed_dim = int(getattr(cfg, "slowfast_temporal_transformer_dim", 512))
        num_heads = int(getattr(cfg, "slowfast_temporal_transformer_heads", 8))
        num_layers = int(getattr(cfg, "slowfast_temporal_transformer_layers", 2))
        self.input_proj = nn.Linear(feat_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=float(getattr(cfg, "dropout", 0.0)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, pathways, attention_masks=None):
        features = self.forward_backbone(pathways, attention_masks=attention_masks)
        sequence = self.pathway_to_sequence(features)
        sequence = self.input_proj(sequence)
        output = self.temporal(sequence)
        pooled = output.mean(dim=1)
        return self.classifier(self.dropout(self.norm(pooled)))


class SlowFastTemporalSelfAttentionClassifier(SlowFastBiLSTMClassifier):
    def __init__(self, model, num_classes, cfg):
        nn.Module.__init__(self)
        self.model = model
        self.alpha_slow = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_slow", None) is None else cfg.tail_attention_alpha_slow)
        self.alpha_fast = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_fast", None) is None else cfg.tail_attention_alpha_fast)
        self.attention_stages = {int(stage) for stage in getattr(cfg, "dual_end_attention_stages", [])}
        self.attention_pathways = str(getattr(cfg, "dual_end_attention_pathways", "both")).lower()
        self.use_dual_end_attention = bool(
            getattr(cfg, "slowfast_dual_end_attention", False)
            or getattr(cfg, "slowfast_tail_attention", False)
        )
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0)))

        with torch.no_grad():
            dummy_fast = torch.zeros(1, 3, int(cfg.num_frames), int(cfg.image_size), int(cfg.image_size))
            dummy_slow = torch.index_select(
                dummy_fast,
                2,
                torch.linspace(0, dummy_fast.shape[2] - 1, max(1, dummy_fast.shape[2] // int(cfg.slowfast_alpha))).long(),
            )
            features = self.forward_backbone([dummy_slow, dummy_fast], attention_masks=None)
            feat_dim = int(features[0].shape[1] + features[1].shape[1])

        embed_dim = int(getattr(cfg, "temporal_attention_dim", getattr(cfg, "slowfast_temporal_transformer_dim", 512)))
        num_heads = int(getattr(cfg, "temporal_attention_heads", 4))
        num_layers = int(getattr(cfg, "temporal_attention_layers", 1))
        attn_dropout = float(getattr(cfg, "temporal_attention_dropout", getattr(cfg, "dropout", 0.0)))
        self.input_norm = nn.LayerNorm(feat_dim)
        self.input_proj = nn.Linear(feat_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.pool_score = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, pathways, attention_masks=None):
        features = self.forward_backbone(pathways, attention_masks=attention_masks)
        sequence = self.pathway_to_sequence(features)
        sequence = self.input_proj(self.input_norm(sequence))
        sequence = self.temporal(sequence)
        sequence = self.output_norm(sequence)
        weights = torch.softmax(self.pool_score(sequence), dim=1)
        pooled = (sequence * weights).sum(dim=1)
        return self.classifier(self.dropout(pooled))


class SlowFastTemporalDifferenceAttentionClassifier(SlowFastBiLSTMClassifier):
    def __init__(self, model, num_classes, cfg):
        nn.Module.__init__(self)
        self.model = model
        self.alpha_slow = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_slow", None) is None else cfg.tail_attention_alpha_slow)
        self.alpha_fast = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_fast", None) is None else cfg.tail_attention_alpha_fast)
        self.attention_stages = {int(stage) for stage in getattr(cfg, "dual_end_attention_stages", [])}
        self.attention_pathways = str(getattr(cfg, "dual_end_attention_pathways", "both")).lower()
        self.use_dual_end_attention = bool(
            getattr(cfg, "slowfast_dual_end_attention", False)
            or getattr(cfg, "slowfast_tail_attention", False)
        )
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0)))
        self.tda_alpha = float(getattr(cfg, "temporal_difference_attention_alpha", 1.0))
        self.use_dual_end_tda = bool(getattr(cfg, "temporal_difference_use_dual_end_mask", False))
        self.use_focus_loss = bool(getattr(cfg, "dual_end_focus_loss", False))
        self.focus_stages = {int(stage) for stage in getattr(cfg, "dual_end_focus_loss_stages", [4])}
        self.last_focus_loss = None

        with torch.no_grad():
            dummy_fast = torch.zeros(1, 3, int(cfg.num_frames), int(cfg.image_size), int(cfg.image_size))
            dummy_slow = torch.index_select(
                dummy_fast,
                2,
                torch.linspace(0, dummy_fast.shape[2] - 1, max(1, dummy_fast.shape[2] // int(cfg.slowfast_alpha))).long(),
            )
            features = self.forward_backbone([dummy_slow, dummy_fast], attention_masks=None)
            feat_dim = int(features[0].shape[1] + features[1].shape[1])

        hidden_dim = int(getattr(cfg, "temporal_difference_attention_dim", 256))
        self.feature_norm = nn.LayerNorm(feat_dim)
        self.diff_score = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(getattr(cfg, "temporal_difference_attention_dropout", getattr(cfg, "dropout", 0.0)))),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def compute_dual_end_focus_loss(self, pathways, attention_masks):
        if attention_masks is None or not isinstance(pathways, (list, tuple)):
            return None
        mask_3d = attention_masks.permute(0, 2, 1, 3, 4)
        losses = []
        for feat in pathways:
            mask = mask_3d.to(device=feat.device, dtype=feat.dtype)
            mask = F.interpolate(mask, size=feat.shape[2:], mode="trilinear", align_corners=False)
            energy = feat.abs().mean(dim=1, keepdim=True)
            total = energy.sum(dim=(1, 2, 3, 4)).clamp_min(1e-6)
            outside = (energy * (1.0 - mask)).sum(dim=(1, 2, 3, 4))
            losses.append((outside / total).mean())
        if not losses:
            return None
        return torch.stack(losses).mean()

    def forward_backbone(self, pathways, attention_masks=None):
        self.last_focus_loss = None
        x = pathways
        focus_losses = []
        for idx, block in enumerate(self.model.blocks[:5]):
            x = block(x)
            if self.use_dual_end_attention and idx in self.attention_stages:
                x = self.apply_dual_end_attention(x, attention_masks)
            if self.use_focus_loss and attention_masks is not None and idx in self.focus_stages:
                focus_loss = self.compute_dual_end_focus_loss(x, attention_masks)
                if focus_loss is not None:
                    focus_losses.append(focus_loss)
        if focus_losses:
            self.last_focus_loss = torch.stack(focus_losses).mean()
        return x

    def pathway_to_dual_end_sequence(self, pathways, attention_masks):
        if attention_masks is None:
            return self.pathway_to_sequence(pathways)
        seqs = []
        max_t = max(int(pathway.shape[2]) for pathway in pathways)
        mask_3d = attention_masks.permute(0, 2, 1, 3, 4)
        for pathway in pathways:
            mask = mask_3d.to(device=pathway.device, dtype=pathway.dtype)
            mask = F.interpolate(mask, size=pathway.shape[2:], mode="trilinear", align_corners=False)
            numerator = (pathway * mask).sum(dim=(-1, -2))
            denominator = mask.sum(dim=(-1, -2)).clamp_min(1e-6)
            seq = (numerator / denominator).transpose(1, 2)
            if seq.shape[1] != max_t:
                seq = F.interpolate(seq.transpose(1, 2), size=max_t, mode="linear", align_corners=False).transpose(1, 2)
            seqs.append(seq)
        return torch.cat(seqs, dim=-1)

    def forward(self, pathways, attention_masks=None):
        features = self.forward_backbone(pathways, attention_masks=attention_masks)
        global_sequence = self.feature_norm(self.pathway_to_sequence(features))
        tda_sequence = (
            self.feature_norm(self.pathway_to_dual_end_sequence(features, attention_masks))
            if self.use_dual_end_tda
            else global_sequence
        )
        if tda_sequence.shape[1] <= 1:
            pooled = global_sequence.mean(dim=1)
            return self.classifier(self.dropout(self.output_norm(pooled)))

        diff = torch.zeros_like(tda_sequence)
        diff[:, 1:] = torch.abs(tda_sequence[:, 1:] - tda_sequence[:, :-1])
        diff[:, 0] = diff[:, 1]
        diff_logits = self.diff_score(diff)
        weights = torch.softmax(diff_logits, dim=1)
        attended = (tda_sequence * weights).sum(dim=1)
        pooled = global_sequence.mean(dim=1) + self.tda_alpha * attended
        return self.classifier(self.dropout(self.output_norm(pooled)))


class SlowFastSegmentLSTMClassifier(nn.Module):
    def __init__(self, model, num_classes, cfg):
        super().__init__()
        self.model = model
        self.alpha = int(getattr(cfg, "slowfast_alpha", 4))
        self.num_segments = int(getattr(cfg, "num_segments", 3))
        self.temporal_module = str(getattr(cfg, "segment_temporal_module", "lstm")).lower()
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0)))

        with torch.no_grad():
            dummy = torch.zeros(1, int(cfg.num_frames), 3, int(cfg.image_size), int(cfg.image_size))
            feat = self.extract_clip_feature(dummy)
            feat_dim = int(feat.shape[1])

        self.feature_norm = nn.LayerNorm(feat_dim)
        hidden = int(getattr(cfg, "segment_lstm_hidden", 256))
        layers = int(getattr(cfg, "segment_lstm_layers", 1))
        rnn_cls = nn.GRU if self.temporal_module == "gru" else nn.LSTM
        if self.temporal_module not in {"lstm", "gru"}:
            raise ValueError("segment_temporal_module must be lstm or gru")
        self.temporal = rnn_cls(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=float(getattr(cfg, "dropout", 0.0)) if layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def pack_pathway(self, clip):
        clips = clip.permute(0, 2, 1, 3, 4)
        fast_pathway = clips
        slow_t = max(1, clips.shape[2] // self.alpha)
        slow_indices = torch.linspace(0, clips.shape[2] - 1, slow_t, device=clips.device).long()
        slow_pathway = torch.index_select(clips, 2, slow_indices)
        return [slow_pathway, fast_pathway]

    def extract_clip_feature(self, clip):
        x = self.pack_pathway(clip)
        for block in self.model.blocks[:5]:
            x = block(x)
        pooled = []
        for pathway in x:
            pooled.append(F.adaptive_avg_pool3d(pathway, (1, 1, 1)).flatten(1))
        return torch.cat(pooled, dim=1)

    def forward(self, videos):
        if videos.dim() != 6:
            raise ValueError("SlowFastSegmentLSTMClassifier expects videos with shape [B, S, T, C, H, W]")
        feats = []
        for seg_idx in range(videos.shape[1]):
            feats.append(self.extract_clip_feature(videos[:, seg_idx]))
        sequence = torch.stack(feats, dim=1)
        sequence = self.feature_norm(sequence)
        output, _ = self.temporal(sequence)
        return self.classifier(self.dropout(output[:, -1]))


class SlowFastStateAggregationClassifier(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.alpha = int(getattr(cfg, "slowfast_alpha", 4))
        self.num_segments = int(getattr(cfg, "num_segments", 3))
        if self.num_segments < 3:
            raise ValueError("SlowFastStateAggregationClassifier requires at least 3 segments")
        self.class_names = list(getattr(cfg, "class_names", DEFAULT_CLASSES))
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0)))

        with torch.no_grad():
            dummy = torch.zeros(1, int(cfg.num_frames), 3, int(cfg.image_size), int(cfg.image_size))
            feat = self.extract_clip_feature(dummy)
            feat_dim = int(feat.shape[1])

        self.feature_norm = nn.LayerNorm(feat_dim)
        self.state_head = nn.Sequential(
            nn.Dropout(float(getattr(cfg, "dropout", 0.0))),
            nn.Linear(feat_dim, 2),
        )

    def pack_pathway(self, clip):
        clips = clip.permute(0, 2, 1, 3, 4)
        fast_pathway = clips
        slow_t = max(1, clips.shape[2] // self.alpha)
        slow_indices = torch.linspace(0, clips.shape[2] - 1, slow_t, device=clips.device).long()
        slow_pathway = torch.index_select(clips, 2, slow_indices)
        return [slow_pathway, fast_pathway]

    def extract_clip_feature(self, clip):
        x = self.pack_pathway(clip)
        for block in self.model.blocks[:5]:
            x = block(x)
        pooled = []
        for pathway in x:
            pooled.append(F.adaptive_avg_pool3d(pathway, (1, 1, 1)).flatten(1))
        return torch.cat(pooled, dim=1)

    def forward(self, videos):
        if videos.dim() != 6:
            raise ValueError("SlowFastStateAggregationClassifier expects videos with shape [B, S, T, C, H, W]")
        feats = []
        for seg_idx in range(videos.shape[1]):
            feats.append(self.extract_clip_feature(videos[:, seg_idx]))
        sequence = self.feature_norm(torch.stack(feats, dim=1))
        state_logits = self.state_head(self.dropout(sequence))
        state_probs = torch.softmax(state_logits, dim=-1).clamp_min(1e-6)
        p_nobirth = state_probs[:, :, 0]
        p_birth = state_probs[:, :, 1]

        mid_idx = videos.shape[1] // 2
        transition_mid = (1.0 - torch.abs(p_birth[:, mid_idx] - p_nobirth[:, mid_idx])).clamp_min(1e-6)
        scores_by_name = {
            "nobirth": torch.log(p_nobirth).sum(dim=1),
            "start": torch.log(p_nobirth[:, 0]) + torch.log(transition_mid) + torch.log(p_birth[:, -1]),
            "birth": torch.log(p_birth).sum(dim=1),
        }
        return torch.stack([scores_by_name[name] for name in self.class_names], dim=1)


class EndpointFeatureDiffBranch(nn.Module):
    def __init__(self, cfg, out_dim=2):
        super().__init__()
        backbone_name = str(getattr(cfg, "hierarchical_branch_backbone", "resnet18")).lower()
        weight_path = getattr(cfg, "hierarchical_branch_weight_path", None)
        pretrained = bool(getattr(cfg, "hierarchical_branch_pretrained", True))
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained and not weight_path else None
            backbone = resnet18(weights=weights)
            feat_dim = backbone.fc.in_features
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained and not weight_path else None
            backbone = resnet50(weights=weights)
            feat_dim = backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported hierarchical_branch_backbone: {backbone_name}")
        if weight_path:
            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            backbone.load_state_dict(state, strict=False)
            print(f"Loaded hierarchical branch weights from: {weight_path}", flush=True)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Dropout(float(getattr(cfg, "dropout", 0.0))),
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(float(getattr(cfg, "dropout", 0.0))),
            nn.Linear(512, out_dim),
        )

    def encode(self, x):
        return self.backbone(x)

    def forward(self, first_a, first_b, last_a, last_b):
        f1a = self.encode(first_a)
        f1b = self.encode(first_b)
        f2a = self.encode(last_a)
        f2b = self.encode(last_b)
        diff_a = torch.abs(f2a - f1a)
        diff_b = torch.abs(f2b - f1b)
        return self.classifier(torch.cat([diff_a, diff_b], dim=1))


class EndpointPresenceBranch(nn.Module):
    def __init__(self, cfg, out_dim=2):
        super().__init__()
        backbone_name = str(getattr(cfg, "hierarchical_branch_backbone", "resnet18")).lower()
        weight_path = getattr(cfg, "hierarchical_branch_weight_path", None)
        pretrained = bool(getattr(cfg, "hierarchical_branch_pretrained", True))
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained and not weight_path else None
            backbone = resnet18(weights=weights)
            feat_dim = backbone.fc.in_features
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained and not weight_path else None
            backbone = resnet50(weights=weights)
            feat_dim = backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported hierarchical_branch_backbone: {backbone_name}")
        if weight_path:
            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            backbone.load_state_dict(state, strict=False)
            print(f"Loaded endpoint presence weights from: {weight_path}", flush=True)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Dropout(float(getattr(cfg, "dropout", 0.0))),
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(float(getattr(cfg, "dropout", 0.0))),
            nn.Linear(512, out_dim),
        )

    def encode(self, x):
        return self.backbone(x)

    def forward(self, last_a, last_b):
        fa = self.encode(last_a)
        fb = self.encode(last_b)
        return self.classifier(torch.cat([fa, fb], dim=1))


class SlowFastAuxiliaryReproductiveClassifier(SlowFastBiLSTMClassifier):
    def __init__(self, model, cfg):
        nn.Module.__init__(self)
        self.model = model
        self.alpha_slow = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_slow", None) is None else cfg.tail_attention_alpha_slow)
        self.alpha_fast = float(getattr(cfg, "tail_attention_alpha", 0.8) if getattr(cfg, "tail_attention_alpha_fast", None) is None else cfg.tail_attention_alpha_fast)
        self.attention_stages = {int(stage) for stage in getattr(cfg, "dual_end_attention_stages", [])}
        self.attention_pathways = str(getattr(cfg, "dual_end_attention_pathways", "both")).lower()
        self.use_dual_end_attention = bool(
            getattr(cfg, "slowfast_dual_end_attention", False)
            or getattr(cfg, "slowfast_tail_attention", False)
        )
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0)))
        self.is_auxiliary_reproductive = True
        self.class_names = list(getattr(cfg, "class_names", DEFAULT_CLASSES))
        self.name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        if "nobirth" not in self.name_to_idx or "start" not in self.name_to_idx or "birth" not in self.name_to_idx:
            raise ValueError("SlowFastAuxiliaryReproductiveClassifier requires class_names to include nobirth, start, birth")

        with torch.no_grad():
            dummy_fast = torch.zeros(1, 3, int(cfg.num_frames), int(cfg.image_size), int(cfg.image_size))
            dummy_slow = torch.index_select(
                dummy_fast,
                2,
                torch.linspace(0, dummy_fast.shape[2] - 1, max(1, dummy_fast.shape[2] // int(cfg.slowfast_alpha))).long(),
            )
            features = self.forward_backbone([dummy_slow, dummy_fast], attention_masks=None)
            feat_dim = int(features[0].shape[1] + features[1].shape[1])

        self.main_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(float(getattr(cfg, "dropout", 0.0))),
            nn.Linear(feat_dim, len(self.class_names)),
        )
        self.existence_branch = EndpointPresenceBranch(cfg, out_dim=2)
        self.change_branch = EndpointFeatureDiffBranch(cfg, out_dim=2)

    def video_feature(self, pathways, attention_masks=None):
        features = self.forward_backbone(pathways, attention_masks=attention_masks)
        pooled = []
        for pathway in features:
            pooled.append(F.adaptive_avg_pool3d(pathway, (1, 1, 1)).flatten(1))
        return torch.cat(pooled, dim=1)

    def forward(self, pathways, attention_masks=None, endpoint_patches=None):
        if endpoint_patches is None or len(endpoint_patches) != 4:
            raise ValueError("Auxiliary reproductive classifier requires endpoint_patches=(first_a, first_b, last_a, last_b)")
        first_a, first_b, last_a, last_b = endpoint_patches
        video_feat = self.video_feature(pathways, attention_masks=attention_masks)
        main_logits = self.main_head(self.dropout(video_feat))
        existence_logits = self.existence_branch(last_a, last_b)
        change_logits = self.change_branch(first_a, first_b, last_a, last_b)
        return {
            "logits": main_logits,
            "main_logits": main_logits,
            "existence_logits": existence_logits,
            "change_logits": change_logits,
        }


class VideoMAEDualEndTokenAttentionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, cfg):
        super().__init__()
        from transformers import AutoConfig, VideoMAEModel

        config = AutoConfig.from_pretrained(model_name)
        config.num_frames = int(cfg.num_frames)
        config.image_size = int(cfg.image_size)
        self.backbone = VideoMAEModel.from_pretrained(model_name, config=config)
        self.alpha = float(getattr(cfg, "videomae_token_attention_alpha", 0.2))
        self.num_frames = int(cfg.num_frames)
        self.image_size = int(cfg.image_size)
        self.patch_size = int(getattr(config, "patch_size", 16))
        self.tubelet_size = int(getattr(config, "tubelet_size", 2))
        self.hidden_size = int(config.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0)))
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def make_token_mask(self, attention_masks, token_count):
        masks = F.interpolate(
            attention_masks,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        patch_masks = F.avg_pool2d(masks, kernel_size=self.patch_size, stride=self.patch_size)
        h, w = patch_masks.shape[-2:]
        spatial_mask = patch_masks.flatten(2).transpose(1, 2)
        time_tokens = max(1, self.num_frames // self.tubelet_size)
        token_mask = spatial_mask.unsqueeze(1).repeat(1, time_tokens, 1, 1).flatten(1, 2)
        if token_mask.shape[1] != token_count:
            token_mask = token_mask[:, :token_count]
            if token_mask.shape[1] < token_count:
                pad = token_mask.new_zeros(token_mask.shape[0], token_count - token_mask.shape[1], 1)
                token_mask = torch.cat([token_mask, pad], dim=1)
        return token_mask

    def forward(self, pixel_values, attention_masks=None):
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        if attention_masks is not None:
            token_mask = self.make_token_mask(attention_masks.to(tokens.device, dtype=tokens.dtype), tokens.shape[1])
            tokens = tokens * (1.0 + self.alpha * token_mask)
        pooled = tokens.mean(dim=1)
        pooled = self.dropout(self.norm(pooled))
        return self.classifier(pooled)


def replace_pytorchvideo_head(model, num_classes):
    for block in reversed(model.blocks):
        if hasattr(block, "proj") and hasattr(block.proj, "in_features"):
            block.proj = nn.Linear(block.proj.in_features, num_classes)
            return model
    raise RuntimeError("Could not replace PyTorchVideo classification head.")


def load_pytorchvideo_model(model_name, pretrained=True):
    cache_dir = Path(torch.hub.get_dir()) / "facebookresearch_pytorchvideo_main"
    if cache_dir.exists():
        return torch.hub.load(str(cache_dir), model_name, source="local", pretrained=pretrained)
    return torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=pretrained)


def remove_slowfast_last_spatial_downsample(model):
    last_stage = model.blocks[4]
    for pathway in last_stage.multipathway_blocks:
        first_block = pathway.res_blocks[0]
        if hasattr(first_block, "branch1_conv") and first_block.branch1_conv is not None:
            first_block.branch1_conv.stride = (1, 1, 1)
        if hasattr(first_block, "branch2") and hasattr(first_block.branch2, "conv_b"):
            first_block.branch2.conv_b.stride = (1, 1, 1)
    if hasattr(model.blocks[5], "pool"):
        model.blocks[5].pool = nn.ModuleList(
            [
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            ]
        )
    return model


def use_adaptive_slowfast_head_pool(model):
    if hasattr(model.blocks[5], "pool"):
        model.blocks[5].pool = nn.ModuleList(
            [
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            ]
        )
    return model


def build_model(cfg, num_classes):
    name = cfg.model_name
    if name == "resnet50_lstm":
        return ResNetTemporalBase(num_classes, "lstm", cfg)
    if name == "resnet50_bilstm":
        return ResNetTemporalBase(num_classes, "bilstm", cfg)
    if name == "resnet50_transformer":
        return ResNetTemporalBase(num_classes, "transformer", cfg)
    if name == "c3d":
        return SimpleC3D(num_classes, cfg.dropout)
    if name == "r3d18":
        from torchvision.models.video import R3D_18_Weights, r3d_18

        weights = R3D_18_Weights.DEFAULT if cfg.pretrained else None
        model = r3d_18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "video_swin":
        from torchvision.models.video import Swin3D_T_Weights, swin3d_t

        weights = Swin3D_T_Weights.DEFAULT if cfg.pretrained else None
        model = swin3d_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if name == "video_swin_b":
        from torchvision.models.video import Swin3D_B_Weights, swin3d_b

        weights = Swin3D_B_Weights.DEFAULT if cfg.pretrained else None
        model = swin3d_b(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if name == "mvit_v2_s":
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        weights = MViT_V2_S_Weights.DEFAULT if cfg.pretrained else None
        model = mvit_v2_s(weights=weights)
        if hasattr(model, "pos_encoding") and hasattr(model.pos_encoding, "temporal_size"):
            temporal_stride = model.conv_proj.stride[0] if hasattr(model, "conv_proj") else 2
            model.pos_encoding.temporal_size = max(1, math.ceil(int(cfg.num_frames) / temporal_stride))
        model.head[1] = nn.Linear(model.head[1].in_features, num_classes)
        return model
    if name == "x3d_s":
        model = load_pytorchvideo_model("x3d_s", pretrained=cfg.pretrained)
        return replace_pytorchvideo_head(model, num_classes)
    if name == "i3d":
        model = load_pytorchvideo_model("i3d_r50", pretrained=cfg.pretrained)
        return replace_pytorchvideo_head(model, num_classes)
    if name == "slowfast_segment_lstm":
        model = load_pytorchvideo_model("slowfast_r50", pretrained=cfg.pretrained)
        print(
            f"SlowFast Segment-LSTM: segments={getattr(cfg, 'num_segments', 3)}, "
            f"clip_len={cfg.num_frames}, alpha={getattr(cfg, 'slowfast_alpha', 4)}, "
            f"temporal={getattr(cfg, 'segment_temporal_module', 'lstm')}, "
            f"hidden={getattr(cfg, 'segment_lstm_hidden', 256)}.",
            flush=True,
        )
        return SlowFastSegmentLSTMClassifier(model, num_classes, cfg)
    if name == "slowfast_state_aggregation":
        model = load_pytorchvideo_model("slowfast_r50", pretrained=cfg.pretrained)
        print(
            f"SlowFast State Aggregation: segments={getattr(cfg, 'num_segments', 3)}, "
            f"clip_len={cfg.num_frames}, stride={cfg.frame_stride}, "
            f"alpha={getattr(cfg, 'slowfast_alpha', 4)}.",
            flush=True,
        )
        return SlowFastStateAggregationClassifier(model, cfg)
    if name == "slowfast":
        model = load_pytorchvideo_model("slowfast_r50", pretrained=cfg.pretrained)
        model = use_adaptive_slowfast_head_pool(model)
        if getattr(cfg, "slowfast_no_last_downsample", False):
            model = remove_slowfast_last_spatial_downsample(model)
            print("SlowFast: removed spatial downsampling in the last ResStage.", flush=True)
        if getattr(cfg, "slowfast_auxiliary_reproductive", False):
            print(
                "SlowFast: enabled auxiliary reproductive supervision "
                "(main 3-class head + last-frame endpoint existence aux + endpoint-change aux).",
                flush=True,
            )
            return SlowFastAuxiliaryReproductiveClassifier(model, cfg)
        if getattr(cfg, "slowfast_bilstm", False):
            print(
                f"SlowFast: enabled BiLSTM temporal head, hidden={cfg.slowfast_bilstm_hidden}, "
                f"layers={cfg.slowfast_bilstm_layers}.",
                flush=True,
            )
            return SlowFastBiLSTMClassifier(model, num_classes, cfg)
        if getattr(cfg, "slowfast_temporal_difference_attention", False):
            print(
                f"SlowFast: enabled Temporal Difference Attention head, "
                f"dim={getattr(cfg, 'temporal_difference_attention_dim', 256)}, "
                f"alpha={getattr(cfg, 'temporal_difference_attention_alpha', 1.0)}, "
                f"dropout={getattr(cfg, 'temporal_difference_attention_dropout', getattr(cfg, 'dropout', 0.0))}.",
                flush=True,
            )
            if getattr(cfg, "dual_end_focus_loss", False):
                print(
                    f"SlowFast: enabled dual-end focus loss, "
                    f"lambda={getattr(cfg, 'dual_end_focus_loss_lambda', 0.01)}, "
                    f"stages={getattr(cfg, 'dual_end_focus_loss_stages', [4])}.",
                    flush=True,
                )
            return SlowFastTemporalDifferenceAttentionClassifier(model, num_classes, cfg)
        if getattr(cfg, "slowfast_temporal_self_attention", False):
            print(
                f"SlowFast: enabled Temporal Self-Attention head, "
                f"dim={getattr(cfg, 'temporal_attention_dim', 512)}, "
                f"heads={getattr(cfg, 'temporal_attention_heads', 4)}, "
                f"layers={getattr(cfg, 'temporal_attention_layers', 1)}, "
                f"dropout={getattr(cfg, 'temporal_attention_dropout', getattr(cfg, 'dropout', 0.0))}.",
                flush=True,
            )
            return SlowFastTemporalSelfAttentionClassifier(model, num_classes, cfg)
        if getattr(cfg, "slowfast_temporal_transformer", False):
            print(
                f"SlowFast: enabled Temporal Transformer head, "
                f"dim={cfg.slowfast_temporal_transformer_dim}, "
                f"heads={cfg.slowfast_temporal_transformer_heads}, "
                f"layers={cfg.slowfast_temporal_transformer_layers}.",
                flush=True,
            )
            return SlowFastTemporalTransformerClassifier(model, num_classes, cfg)
        model = replace_pytorchvideo_head(model, num_classes)
        use_dual_end_attention = getattr(cfg, "slowfast_dual_end_attention", False) or getattr(
            cfg, "slowfast_tail_attention", False
        )
        if use_dual_end_attention:
            print(
                f"SlowFast: enabled dual-end spatial attention at stages {cfg.dual_end_attention_stages}, "
                f"pathways={getattr(cfg, 'dual_end_attention_pathways', 'both')}, "
                f"alpha_default={cfg.tail_attention_alpha}, "
                f"alpha_slow={getattr(cfg, 'tail_attention_alpha_slow', None)}, "
                f"alpha_fast={getattr(cfg, 'tail_attention_alpha_fast', None)}.",
                flush=True,
            )
            model = SlowFastDualEndAttentionWrapper(
                model,
                alpha=cfg.tail_attention_alpha,
                stages=cfg.dual_end_attention_stages,
                pathways=getattr(cfg, "dual_end_attention_pathways", "both"),
                alpha_slow=getattr(cfg, "tail_attention_alpha_slow", None),
                alpha_fast=getattr(cfg, "tail_attention_alpha_fast", None),
            )
        return model
    if name == "videomae" and getattr(cfg, "videomae_dual_end_token_attention", False):
        print(
            f"VideoMAE: enabled dual-end token attention, alpha={cfg.videomae_token_attention_alpha}.",
            flush=True,
        )
        return VideoMAEDualEndTokenAttentionClassifier(cfg.videomae_model_name, num_classes, cfg)
    if name in {"videomae", "timesformer"}:
        from transformers import AutoConfig, AutoModelForVideoClassification

        hf_name = {
            "videomae": cfg.videomae_model_name,
            "timesformer": cfg.timesformer_model_name,
        }[name]
        id2label = {i: c for i, c in enumerate(cfg.class_names)}
        label2id = {c: i for i, c in enumerate(cfg.class_names)}
        if name == "timesformer" and not cfg.pretrained:
            from transformers import TimesformerConfig, TimesformerForVideoClassification

            config = TimesformerConfig(
                image_size=int(cfg.image_size),
                num_frames=int(cfg.num_frames),
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
            )
            return TimesformerForVideoClassification(config)
        config = AutoConfig.from_pretrained(hf_name, num_labels=num_classes, id2label=id2label, label2id=label2id)
        if name in {"videomae", "timesformer"}:
            config.num_frames = int(cfg.num_frames)
            config.image_size = int(cfg.image_size)
        return AutoModelForVideoClassification.from_pretrained(
            hf_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
    raise ValueError(name)


def pack_slowfast_pathway(clips, alpha=4):
    fast_pathway = clips
    slow_t = max(1, clips.shape[2] // alpha)
    slow_indices = torch.linspace(0, clips.shape[2] - 1, slow_t, device=clips.device).long()
    slow_pathway = torch.index_select(clips, 2, slow_indices)
    return [slow_pathway, fast_pathway]


def model_forward(model, videos, model_name, tail_masks=None, slowfast_alpha=4, endpoint_patches=None):
    if model_name in {"resnet50_lstm", "resnet50_bilstm", "resnet50_transformer"}:
        return model(videos)

    if model_name in {"slowfast_segment_lstm", "slowfast_state_aggregation"}:
        return model(videos)

    if model_name == "slowfast":
        clips = videos.permute(0, 2, 1, 3, 4)
        pathways = pack_slowfast_pathway(clips, alpha=slowfast_alpha)
        if getattr(model, "is_auxiliary_reproductive", False):
            return model(
                pathways,
                attention_masks=tail_masks,
                endpoint_patches=endpoint_patches,
            )
        supports_attention_masks = (
            isinstance(model, SlowFastDualEndAttentionWrapper)
            or hasattr(model, "attention_stages")
            or hasattr(model, "use_dual_end_attention")
        )
        if tail_masks is not None and supports_attention_masks:
            return model(pathways, attention_masks=tail_masks)
        return model(pathways)

    if model_name in {"c3d", "r3d18", "x3d_s", "i3d", "video_swin", "video_swin_b", "mvit_v2_s"}:
        clips = videos.permute(0, 2, 1, 3, 4)
        return model(clips)

    outputs = model(pixel_values=videos)
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def unpack_batch(batch):
    endpoint_patches = None
    if len(batch) == 8:
        videos, tail_masks, first_a, first_b, last_a, last_b, labels, names = batch
        endpoint_patches = (first_a, first_b, last_a, last_b)
    elif len(batch) == 7:
        videos, first_a, first_b, last_a, last_b, labels, names = batch
        tail_masks = None
        endpoint_patches = (first_a, first_b, last_a, last_b)
    elif len(batch) == 4:
        videos, tail_masks, labels, names = batch
        endpoint_patches = None
    else:
        videos, labels, names = batch
        tail_masks = None
        endpoint_patches = None
    return videos, tail_masks, endpoint_patches, labels, names


def make_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone() if weight is not None else None)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(1e-8)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CEFocalJointLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, focal_lambda=0.5, label_smoothing=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.focal = FocalLoss(weight=weight, gamma=gamma, label_smoothing=label_smoothing)
        self.focal_lambda = float(focal_lambda)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.focal_lambda * self.focal(logits, targets)


class MultiClassBCELoss(nn.Module):
    def __init__(self, num_classes, pos_weight=None, label_smoothing=0.0, soft_targets=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if soft_targets is None:
            self.register_buffer("soft_targets", None)
        else:
            soft_targets = torch.tensor(soft_targets, dtype=torch.float32)
            if tuple(soft_targets.shape) != (self.num_classes, self.num_classes):
                raise ValueError(
                    f"soft_targets must have shape [{self.num_classes}, {self.num_classes}], "
                    f"got {tuple(soft_targets.shape)}"
                )
            self.register_buffer("soft_targets", soft_targets)

    def forward(self, logits, targets):
        if isinstance(logits, dict):
            logits = logits["logits"]
        if self.soft_targets is not None:
            target_onehot = self.soft_targets.to(device=logits.device, dtype=logits.dtype).index_select(0, targets)
        else:
            target_onehot = torch.zeros(
                targets.size(0),
                self.num_classes,
                device=logits.device,
                dtype=logits.dtype,
            )
            target_onehot.scatter_(1, targets.view(-1, 1), 1.0)
        if self.label_smoothing > 0:
            target_onehot = target_onehot * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
        return self.bce(logits, target_onehot)


class CECRCLoss(nn.Module):
    def __init__(
        self,
        class_names,
        positive_classes,
        weight=None,
        crcl_lambda=0.3,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.crcl_lambda = float(crcl_lambda)
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        missing = [name for name in positive_classes if name not in name_to_idx]
        if missing:
            raise ValueError(f"CRCL positive classes not found in class_names: {missing}")
        positive_indices = [name_to_idx[name] for name in positive_classes]
        if not positive_indices:
            raise ValueError("CRCL requires at least one positive class.")
        self.register_buffer("positive_indices", torch.tensor(positive_indices, dtype=torch.long))
        self.nonbirth_idx = name_to_idx.get("nobirth", None)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        positive_indices = self.positive_indices.to(logits.device)
        p_positive = probs.index_select(1, positive_indices).sum(dim=1).clamp_min(1e-8)
        if self.nonbirth_idx is not None:
            p_negative = probs[:, self.nonbirth_idx].clamp_min(1e-8)
        else:
            p_negative = (1.0 - p_positive).clamp_min(1e-8)
        is_positive = (targets.unsqueeze(1) == positive_indices.unsqueeze(0)).any(dim=1)
        crcl_loss = torch.where(is_positive, -torch.log(p_positive), -torch.log(p_negative)).mean()
        return ce_loss + self.crcl_lambda * crcl_loss


class CERSCLoss(nn.Module):
    def __init__(
        self,
        class_names,
        weight=None,
        exist_lambda=0.2,
        change_lambda=0.2,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.exist_lambda = float(exist_lambda)
        self.change_lambda = float(change_lambda)
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        for required in ("nobirth", "start", "birth"):
            if required not in name_to_idx:
                raise ValueError(f"RSCL requires class_names to include {required}")
        self.nobirth_idx = name_to_idx["nobirth"]
        self.start_idx = name_to_idx["start"]
        self.birth_idx = name_to_idx["birth"]

    def forward(self, logits, targets):
        if isinstance(logits, dict):
            logits = logits["logits"]
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        eps = 1e-7

        p_exist = (probs[:, self.start_idx] + probs[:, self.birth_idx]).clamp(eps, 1.0 - eps)
        y_exist = (targets != self.nobirth_idx).to(dtype=probs.dtype)
        exist_loss = F.binary_cross_entropy(p_exist, y_exist)

        p_change = probs[:, self.start_idx].clamp(eps, 1.0 - eps)
        y_change = (targets == self.start_idx).to(dtype=probs.dtype)
        change_loss = F.binary_cross_entropy(p_change, y_change)

        return ce_loss + self.exist_lambda * exist_loss + self.change_lambda * change_loss


class AuxiliaryReproductiveLoss(nn.Module):
    def __init__(
        self,
        class_names,
        main_weight=None,
        existence_weight=None,
        change_weight=None,
        label_smoothing=0.0,
        main_lambda=1.0,
        existence_lambda=0.5,
        change_lambda=0.5,
    ):
        super().__init__()
        self.class_names = list(class_names)
        self.name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.nobirth_idx = self.name_to_idx["nobirth"]
        self.start_idx = self.name_to_idx["start"]
        self.birth_idx = self.name_to_idx["birth"]
        self.main_lambda = float(main_lambda)
        self.existence_lambda = float(existence_lambda)
        self.change_lambda = float(change_lambda)
        self.main_ce = nn.CrossEntropyLoss(weight=main_weight, label_smoothing=label_smoothing)
        self.existence_ce = nn.CrossEntropyLoss(weight=existence_weight, label_smoothing=label_smoothing)
        self.change_ce = nn.CrossEntropyLoss(weight=change_weight, label_smoothing=label_smoothing)

    def forward(self, outputs, targets):
        if not isinstance(outputs, dict):
            raise ValueError("AuxiliaryReproductiveLoss expects model outputs to be a dict")
        main_logits = outputs["main_logits"]
        existence_logits = outputs["existence_logits"]
        change_logits = outputs["change_logits"]
        main_loss = self.main_ce(main_logits, targets)
        existence_targets = (targets != self.nobirth_idx).long()
        change_targets = (targets == self.start_idx).long()
        existence_loss = self.existence_ce(existence_logits, existence_targets)
        change_loss = self.change_ce(change_logits, change_targets)
        return (
            self.main_lambda * main_loss
            + self.existence_lambda * existence_loss
            + self.change_lambda * change_loss
        )


def build_criterion(cfg, train_labels, num_classes, device):
    label_smoothing = float(getattr(cfg, "label_smoothing", 0.0))
    weights = None
    if cfg.use_class_weights:
        weights = make_class_weights(train_labels, num_classes).to(device)
        print(f"Class weights: {weights.detach().cpu().numpy().tolist()}", flush=True)

    loss_name = str(getattr(cfg, "loss_name", "ce")).lower()
    focal_gamma = float(getattr(cfg, "focal_gamma", 2.0))
    focal_lambda = float(getattr(cfg, "focal_lambda", 0.5))
    crcl_lambda = float(getattr(cfg, "crcl_lambda", 0.3))
    crcl_positive_classes = list(getattr(cfg, "crcl_positive_classes", ["start", "birth"]))
    rscl_exist_lambda = float(getattr(cfg, "rscl_exist_lambda", 0.2))
    rscl_change_lambda = float(getattr(cfg, "rscl_change_lambda", 0.2))
    if getattr(cfg, "slowfast_auxiliary_reproductive", False):
        class_names = list(cfg.class_names)
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        main_weights = None
        existence_weights = None
        change_weights = None
        if cfg.use_class_weights:
            train_labels_np = np.asarray(train_labels, dtype=np.int64)
            main_weights = make_class_weights(train_labels_np.tolist(), num_classes).to(device)
            existence_targets = (train_labels_np != name_to_idx["nobirth"]).astype(np.int64)
            change_targets = (train_labels_np == name_to_idx["start"]).astype(np.int64)
            existence_weights = make_class_weights(existence_targets.tolist(), 2).to(device)
            change_weights = make_class_weights(change_targets.tolist(), 2).to(device)
            print(
                f"Auxiliary weights: main={main_weights.detach().cpu().numpy().tolist()}, "
                f"existence={existence_weights.detach().cpu().numpy().tolist()}, "
                f"change={change_weights.detach().cpu().numpy().tolist()}",
                flush=True,
            )
        criterion = AuxiliaryReproductiveLoss(
            class_names=cfg.class_names,
            main_weight=main_weights,
            existence_weight=existence_weights,
            change_weight=change_weights,
            label_smoothing=label_smoothing,
            main_lambda=float(getattr(cfg, "aux_main_lambda", 1.0)),
            existence_lambda=float(getattr(cfg, "aux_existence_lambda", 0.5)),
            change_lambda=float(getattr(cfg, "aux_change_lambda", 0.5)),
        )
        print(
            f"Loss: auxiliary_reproductive, label_smoothing={label_smoothing}, "
            f"aux_main_lambda={getattr(cfg, 'aux_main_lambda', 1.0)}, "
            f"aux_existence_lambda={getattr(cfg, 'aux_existence_lambda', 0.5)}, "
            f"aux_change_lambda={getattr(cfg, 'aux_change_lambda', 0.5)}",
            flush=True,
        )
        return criterion
    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    elif loss_name in {"bce", "multiclass_bce", "bce_onehot", "bce_soft", "soft_bce"}:
        pos_weight = None
        if cfg.use_class_weights:
            train_labels_np = np.asarray(train_labels, dtype=np.int64)
            counts = np.bincount(train_labels_np, minlength=num_classes).astype(np.float32)
            positives = np.maximum(counts, 1.0)
            negatives = np.maximum(float(len(train_labels_np)) - counts, 1.0)
            pos_weight = torch.tensor(negatives / positives, dtype=torch.float32).to(device)
            print(f"BCE pos_weight: {pos_weight.detach().cpu().numpy().tolist()}", flush=True)
        soft_targets = getattr(cfg, "bce_soft_targets", None)
        if loss_name in {"bce_soft", "soft_bce"} and soft_targets is None:
            name_to_idx = {name: idx for idx, name in enumerate(cfg.class_names)}
            soft_targets = np.eye(num_classes, dtype=np.float32)
            if all(name in name_to_idx for name in ("nobirth", "start", "birth")):
                start_idx = name_to_idx["start"]
                soft_targets[start_idx, name_to_idx["nobirth"]] = 0.2
                soft_targets[start_idx, name_to_idx["birth"]] = 0.2
            soft_targets = soft_targets.tolist()
        if soft_targets is not None:
            print(f"BCE soft targets: {soft_targets}", flush=True)
        criterion = MultiClassBCELoss(
            num_classes=num_classes,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
            soft_targets=soft_targets,
        )
    elif loss_name == "focal":
        criterion = FocalLoss(weight=weights, gamma=focal_gamma, label_smoothing=label_smoothing)
    elif loss_name in {"ce_focal", "ce+focal", "joint"}:
        criterion = CEFocalJointLoss(
            weight=weights,
            gamma=focal_gamma,
            focal_lambda=focal_lambda,
            label_smoothing=label_smoothing,
        )
    elif loss_name in {"ce_crcl", "ce+crcl", "crcl"}:
        criterion = CECRCLoss(
            class_names=cfg.class_names,
            positive_classes=crcl_positive_classes,
            weight=weights,
            crcl_lambda=crcl_lambda,
            label_smoothing=label_smoothing,
        )
    elif loss_name in {"ce_rscl", "ce+rscl", "rscl"}:
        criterion = CERSCLoss(
            class_names=cfg.class_names,
            weight=weights,
            exist_lambda=rscl_exist_lambda,
            change_lambda=rscl_change_lambda,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(
            f"Unsupported loss_name: {cfg.loss_name}. "
            "Choices: ce, bce, bce_soft, focal, ce_focal, ce_crcl, ce_rscl"
        )

    print(
        f"Loss: {loss_name}, label_smoothing={label_smoothing}, "
        f"focal_gamma={focal_gamma}, focal_lambda={focal_lambda}, "
        f"crcl_lambda={crcl_lambda}, crcl_positive_classes={crcl_positive_classes}, "
        f"rscl_exist_lambda={rscl_exist_lambda}, rscl_change_lambda={rscl_change_lambda}",
        flush=True,
    )
    return criterion


def compute_metrics(labels, preds, probs, class_names):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    num_classes = len(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), zero_division=0
    )
    y_true = np.zeros((len(labels), num_classes), dtype=np.int64)
    y_true[np.arange(len(labels)), labels] = 1
    ap = []
    for i in range(num_classes):
        ap.append(float("nan") if y_true[:, i].sum() == 0 else average_precision_score(y_true[:, i], probs[:, i]))
    valid_ap = [x for x in ap if not math.isnan(x)]
    return {
        "acc": float((labels == preds).mean()) if len(labels) else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "micro_f1": float(precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)[2]),
        "mAP": float(np.mean(valid_ap)) if valid_ap else float("nan"),
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    model_name,
    slowfast_alpha=4,
    grad_accum_steps=1,
    focus_loss_lambda=0.0,
):
    model.train()
    total_loss, total_count = 0.0, 0
    all_labels, all_preds = [], []
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    for step_idx, batch in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        videos, tail_masks, endpoint_patches, labels, _ = unpack_batch(batch)
        if tail_masks is not None:
            tail_masks = tail_masks.to(device, non_blocking=True)
        if endpoint_patches is not None:
            endpoint_patches = tuple(p.to(device, non_blocking=True) for p in endpoint_patches)
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model_forward(
            model,
            videos,
            model_name,
            tail_masks=tail_masks,
            slowfast_alpha=slowfast_alpha,
            endpoint_patches=endpoint_patches,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        loss = criterion(outputs if isinstance(outputs, dict) else logits, labels)
        focus_loss = getattr(model, "last_focus_loss", None)
        if focus_loss is not None and float(focus_loss_lambda) > 0:
            loss = loss + float(focus_loss_lambda) * focus_loss
        (loss / grad_accum_steps).backward()
        if step_idx % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
    if len(loader) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    acc = float((np.asarray(all_labels) == np.asarray(all_preds)).mean()) if all_labels else 0.0
    return total_loss / max(1, total_count), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, model_name, class_names, slowfast_alpha=4):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        videos, tail_masks, endpoint_patches, labels, _ = unpack_batch(batch)
        if tail_masks is not None:
            tail_masks = tail_masks.to(device, non_blocking=True)
        if endpoint_patches is not None:
            endpoint_patches = tuple(p.to(device, non_blocking=True) for p in endpoint_patches)
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model_forward(
            model,
            videos,
            model_name,
            tail_masks=tail_masks,
            slowfast_alpha=slowfast_alpha,
            endpoint_patches=endpoint_patches,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        loss = criterion(outputs if isinstance(outputs, dict) else logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    metrics = compute_metrics(all_labels, all_preds, np.asarray(all_probs), class_names)
    metrics["loss"] = total_loss / max(1, total_count)
    return metrics


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def plot_series(history, series, title, ylabel, save_path):
    epochs = [int(row["epoch"]) for row in history]
    plt.figure(figsize=(8, 5), dpi=160)
    for key, label in series:
        values = [safe_float(row.get(key)) for row in history]
        if not any(not math.isnan(v) for v in values):
            continue
        plt.plot(epochs, values, marker="o", linewidth=1.8, markersize=3.5, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_training_curves(history, class_names, run_dir):
    if not history:
        return
    plot_series(
        history,
        [("train_loss", "Train Loss"), ("val_loss", "Val Loss")],
        "Training and Validation Loss",
        "Loss",
        run_dir / "loss_curve.png",
    )
    plot_series(
        history,
        [("train_acc", "Train Acc"), ("val_acc", "Val Acc")],
        "Training and Validation Accuracy",
        "Accuracy",
        run_dir / "accuracy_curve.png",
    )
    plot_series(
        history,
        [("val_macro_f1", "Macro F1"), ("val_micro_f1", "Micro F1"), ("val_mAP", "mAP")],
        "Validation F1 and mAP",
        "Score",
        run_dir / "val_metrics_curve.png",
    )
    ap_series = [(f"{c}_AP", f"{c} AP") for c in class_names]
    plot_series(history, ap_series, "Per-Class AP", "AP", run_dir / "per_class_ap_curve.png")


def get_current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def decay_learning_rate(optimizer, factor, min_lr):
    old_lr = get_current_lr(optimizer)
    new_lr = max(old_lr * factor, min_lr)
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * factor, min_lr)
    return old_lr, new_lr


def build_optimizer(model, cfg):
    optimizer_name = str(cfg.optimizer).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}. Choices: adamw, adam, sgd")


def build_lr_scheduler(optimizer, cfg):
    if not cfg.use_lr_scheduler:
        return None
    scheduler_name = str(getattr(cfg, "lr_scheduler", "plateau")).lower()
    if scheduler_name == "plateau":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.epochs),
            eta_min=float(cfg.min_lr),
        )
    raise ValueError(f"Unsupported lr_scheduler: {cfg.lr_scheduler}. Choices: plateau, cosine")


def build_run_paths(cfg, script_dir):
    output_root = Path(cfg.output_root)
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"{cfg.model_name}_{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(cfg.save_path) if cfg.save_path else run_dir / "best_model_by_mAP.pth"
    log_path = Path(cfg.log_path) if cfg.log_path else run_dir / "train_log.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return run_dir, save_path, log_path


def save_checkpoint(path, model, epoch, class_names, cfg, best_key, best_value, metrics_summary):
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        best_key: best_value,
        "best_metric_name": best_key,
        "best_metric_value": best_value,
        "best_metrics": metrics_summary,
        "class_names": class_names,
        "config": vars(cfg),
    }
    torch.save(payload, path)


def make_metrics_summary(epoch, lr, train_loss, train_acc, metrics):
    return {
        "epoch": epoch,
        "lr": lr,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": metrics["loss"],
        "val_acc": metrics["acc"],
        "val_macro_f1": metrics["macro_f1"],
        "val_micro_f1": metrics["micro_f1"],
        "val_mAP": metrics["mAP"],
        "ap": list(metrics["ap"]),
        "precision": list(metrics["precision"]),
        "recall": list(metrics["recall"]),
        "f1": list(metrics["f1"]),
        "support": list(metrics["support"]),
    }


def print_best_summary(title, summary, class_names, save_path):
    if summary is None:
        return
    print(f"\n{title}:", flush=True)
    print(
        f"  epoch={summary['epoch']} lr={summary['lr']:.8f} "
        f"train_loss={summary['train_loss']:.4f} train_acc={summary['train_acc']:.4f} "
        f"val_loss={summary['val_loss']:.4f} val_acc={summary['val_acc']:.4f} "
        f"val_macro_f1={summary['val_macro_f1']:.4f} "
        f"val_micro_f1={summary['val_micro_f1']:.4f} "
        f"val_mAP={summary['val_mAP']:.4f}",
        flush=True,
    )
    for idx, c in enumerate(class_names):
        print(
            f"  {c}: AP={summary['ap'][idx]:.4f} "
            f"P={summary['precision'][idx]:.4f} "
            f"R={summary['recall'][idx]:.4f} "
            f"F1={summary['f1'][idx]:.4f} "
            f"N={int(summary['support'][idx])}",
            flush=True,
        )
    print(f"Saved to {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.model_name:
        cfg.model_name = args.model_name
        for k, v in MODEL_DEFAULTS[cfg.model_name].items():
            setattr(cfg, k, v)

    set_seed(cfg.seed)
    script_dir = Path(__file__).resolve().parent
    class_names = list(cfg.class_names)
    cfg.class_names = class_names
    class_map = cfg.class_map if getattr(cfg, "class_map", None) else None
    train_dir = Path(cfg.train_dir) if cfg.train_dir else Path(cfg.data_root) / "train"
    val_dir = Path(cfg.val_dir) if cfg.val_dir else Path(cfg.data_root) / "val"
    run_dir, save_path, log_path = build_run_paths(cfg, script_dir)
    best_map_path = save_path
    best_acc_path = save_path.with_name("best_model_by_acc.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Model: {cfg.model_name}", flush=True)
    print(f"Output dir: {run_dir}", flush=True)
    print(f"Classes: {class_names}", flush=True)
    if class_map:
        print(f"Class map: {class_map}", flush=True)
    print(f"Train dir: {train_dir}", flush=True)
    print(f"Val dir: {val_dir}", flush=True)
    print(
        f"Sampling: {cfg.sampling}, num_frames={cfg.num_frames}, stride={cfg.frame_stride}, image_size={cfg.image_size}",
        flush=True,
    )
    if cfg.model_name == "slowfast":
        print(
            f"SlowFast pathways: fast_frames={cfg.num_frames}, slow_alpha={cfg.slowfast_alpha}, "
            f"slow_frames={max(1, int(cfg.num_frames) // int(cfg.slowfast_alpha))}",
            flush=True,
        )
    if cfg.model_name in {"slowfast_segment_lstm", "slowfast_state_aggregation"}:
        print(
            f"Segment sampling: segments={getattr(cfg, 'num_segments', 3)}, "
            f"clip_len={cfg.num_frames}, stride={cfg.frame_stride}, "
            f"slow_alpha={cfg.slowfast_alpha}, batch_size={cfg.batch_size}, "
            f"grad_accum_steps={getattr(cfg, 'gradient_accumulation_steps', 1)}, "
            f"temporal={getattr(cfg, 'segment_temporal_module', 'lstm')}, "
            f"hidden={getattr(cfg, 'segment_lstm_hidden', 256)}",
            flush=True,
        )
    print(
        f"LR schedule: enabled={cfg.use_lr_scheduler}, type={getattr(cfg, 'lr_scheduler', 'plateau')}, initial_lr={cfg.lr}, "
        f"val_loss min_delta={cfg.lr_decay_min_delta}, patience={cfg.lr_decay_patience}, "
        f"factor={cfg.lr_decay_factor}",
        flush=True,
    )
    print(
        f"Optimizer: {cfg.optimizer}, weight_decay={cfg.weight_decay}, "
        f"momentum={getattr(cfg, 'momentum', 0.0)}, nesterov={getattr(cfg, 'nesterov', False)}",
        flush=True,
    )
    print(
        f"Augmentation: random_horizontal_flip={getattr(cfg, 'random_horizontal_flip', True)}, "
        f"vertical_flip={getattr(cfg, 'random_vertical_flip', False)}, "
        f"color_jitter=True, "
        f"random_rotation={getattr(cfg, 'random_rotation', False)}, "
        f"rotation_degrees={getattr(cfg, 'random_rotation_degrees', 0)}, "
        f"rotation_p={getattr(cfg, 'random_rotation_p', 0)}, "
        f"random_affine={getattr(cfg, 'random_affine', False)}, "
        f"translate={getattr(cfg, 'random_translate', 0)}, "
        f"scale=({getattr(cfg, 'random_scale_min', 1.0)}, {getattr(cfg, 'random_scale_max', 1.0)}), "
        f"shear={getattr(cfg, 'random_shear_degrees', 0)}, "
        f"gaussian_blur={getattr(cfg, 'gaussian_blur', False)}, "
        f"gaussian_noise={getattr(cfg, 'gaussian_noise', False)}",
        flush=True,
    )
    if getattr(cfg, "slowfast_auxiliary_reproductive", False):
        print(
            "Auxiliary branch: enabled, "
            f"main_lambda={getattr(cfg, 'aux_main_lambda', 1.0)}, "
            f"existence_lambda={getattr(cfg, 'aux_existence_lambda', 0.5)}, "
            f"change_lambda={getattr(cfg, 'aux_change_lambda', 0.5)}, "
            f"branch_backbone={getattr(cfg, 'hierarchical_branch_backbone', 'resnet18')}, "
            f"use_sampled_endpoints={getattr(cfg, 'hierarchical_use_sampled_endpoints', True)}, "
            "existence_input=last_frame_dual_end",
            flush=True,
        )

    train_dataset = CroppedInstanceDataset(
        train_dir,
        class_names,
        cfg.num_frames,
        cfg.image_size,
        cfg.sampling,
        cfg.frame_stride,
        train=True,
        class_map=class_map,
        return_attention_masks=(
            (
                getattr(cfg, "slowfast_dual_end_attention", False)
                or getattr(cfg, "slowfast_tail_attention", False)
                or getattr(cfg, "dual_end_focus_loss", False)
                or getattr(cfg, "temporal_difference_use_dual_end_mask", False)
                or getattr(cfg, "videomae_dual_end_token_attention", False)
            )
            and cfg.model_name in {"slowfast", "videomae"}
        ),
        tail_end_ratio=getattr(cfg, "tail_end_ratio", 1.0 / 3.0),
        return_endpoint_patches=(
            getattr(cfg, "slowfast_auxiliary_reproductive", False)
            and cfg.model_name == "slowfast"
        ),
        use_sampled_endpoints=getattr(cfg, "hierarchical_use_sampled_endpoints", True),
        return_segments=(cfg.model_name in {"slowfast_segment_lstm", "slowfast_state_aggregation"}),
        num_segments=getattr(cfg, "num_segments", 3),
        random_horizontal_flip=getattr(cfg, "random_horizontal_flip", True),
        random_horizontal_flip_p=getattr(cfg, "random_horizontal_flip_p", 0.5),
        random_vertical_flip=getattr(cfg, "random_vertical_flip", False),
        random_vertical_flip_p=getattr(cfg, "random_vertical_flip_p", 0.2),
        random_rotation=getattr(cfg, "random_rotation", False),
        random_rotation_degrees=getattr(cfg, "random_rotation_degrees", 8),
        random_rotation_p=getattr(cfg, "random_rotation_p", 0.5),
        random_affine=getattr(cfg, "random_affine", False),
        random_affine_p=getattr(cfg, "random_affine_p", 0.5),
        random_translate=getattr(cfg, "random_translate", 0.03),
        random_scale_min=getattr(cfg, "random_scale_min", 0.95),
        random_scale_max=getattr(cfg, "random_scale_max", 1.05),
        random_shear_degrees=getattr(cfg, "random_shear_degrees", 3),
        gaussian_blur=getattr(cfg, "gaussian_blur", False),
        gaussian_blur_p=getattr(cfg, "gaussian_blur_p", 0.2),
        gaussian_blur_kernel=getattr(cfg, "gaussian_blur_kernel", 3),
        gaussian_blur_sigma_min=getattr(cfg, "gaussian_blur_sigma_min", 0.1),
        gaussian_blur_sigma_max=getattr(cfg, "gaussian_blur_sigma_max", 1.0),
        gaussian_noise=getattr(cfg, "gaussian_noise", False),
        gaussian_noise_p=getattr(cfg, "gaussian_noise_p", 0.2),
        gaussian_noise_std=getattr(cfg, "gaussian_noise_std", 0.01),
    )
    val_dataset = CroppedInstanceDataset(
        val_dir,
        class_names,
        cfg.num_frames,
        cfg.image_size,
        cfg.sampling,
        cfg.frame_stride,
        train=False,
        class_map=class_map,
        return_attention_masks=(
            (
                getattr(cfg, "slowfast_dual_end_attention", False)
                or getattr(cfg, "slowfast_tail_attention", False)
                or getattr(cfg, "dual_end_focus_loss", False)
                or getattr(cfg, "temporal_difference_use_dual_end_mask", False)
                or getattr(cfg, "videomae_dual_end_token_attention", False)
            )
            and cfg.model_name in {"slowfast", "videomae"}
        ),
        tail_end_ratio=getattr(cfg, "tail_end_ratio", 1.0 / 3.0),
        return_endpoint_patches=(
            getattr(cfg, "slowfast_auxiliary_reproductive", False)
            and cfg.model_name == "slowfast"
        ),
        use_sampled_endpoints=getattr(cfg, "hierarchical_use_sampled_endpoints", True),
        return_segments=(cfg.model_name in {"slowfast_segment_lstm", "slowfast_state_aggregation"}),
        num_segments=getattr(cfg, "num_segments", 3),
    )
    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Val samples: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg, len(class_names)).to(device)
    criterion = build_criterion(cfg, train_dataset.labels(), len(class_names), device)
    optimizer = build_optimizer(model, cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg)

    header = [
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "val_micro_f1",
        "val_mAP",
        "best_mAP",
    ]
    for c in class_names:
        header.extend([f"{c}_AP", f"{c}_P", f"{c}_R", f"{c}_F1", f"{c}_support"])
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    best_map = -1.0
    best_acc = -1.0
    best_map_summary = None
    best_acc_summary = None
    best_val_loss = float("inf")
    bad_loss_epochs = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        epoch_lr = get_current_lr(optimizer)
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            cfg.model_name,
            slowfast_alpha=getattr(cfg, "slowfast_alpha", 4),
            grad_accum_steps=getattr(cfg, "gradient_accumulation_steps", 1),
            focus_loss_lambda=getattr(cfg, "dual_end_focus_loss_lambda", 0.0),
        )
        metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg.model_name,
            class_names,
            slowfast_alpha=getattr(cfg, "slowfast_alpha", 4),
        )
        print(
            f"Epoch [{epoch:03d}/{cfg.epochs}] lr={epoch_lr:.8f} loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.4f} val_macro_f1={metrics['macro_f1']:.4f} "
            f"val_micro_f1={metrics['micro_f1']:.4f} val_mAP={metrics['mAP']:.4f}",
            flush=True,
        )
        for idx, c in enumerate(class_names):
            print(
                f"  {c}: AP={metrics['ap'][idx]:.4f} P={metrics['precision'][idx]:.4f} "
                f"R={metrics['recall'][idx]:.4f} F1={metrics['f1'][idx]:.4f} N={int(metrics['support'][idx])}",
                flush=True,
            )

        current_summary = make_metrics_summary(epoch, epoch_lr, train_loss, train_acc, metrics)
        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            best_map_summary = current_summary
            save_checkpoint(
                best_map_path,
                model,
                epoch,
                class_names,
                cfg,
                "best_mAP",
                best_map,
                best_map_summary,
            )
            print(f"Saved best mAP model to {best_map_path}", flush=True)

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_acc_summary = current_summary
            save_checkpoint(
                best_acc_path,
                model,
                epoch,
                class_names,
                cfg,
                "best_acc",
                best_acc,
                best_acc_summary,
            )
            print(f"Saved best accuracy model to {best_acc_path}", flush=True)

        if cfg.use_lr_scheduler and str(getattr(cfg, "lr_scheduler", "plateau")).lower() == "plateau":
            if metrics["loss"] < best_val_loss - cfg.lr_decay_min_delta:
                best_val_loss = metrics["loss"]
                bad_loss_epochs = 0
            else:
                bad_loss_epochs += 1
                print(
                    f"Val loss improvement < {cfg.lr_decay_min_delta}; "
                    f"bad epochs {bad_loss_epochs}/{cfg.lr_decay_patience}",
                    flush=True,
                )
                if bad_loss_epochs >= cfg.lr_decay_patience:
                    old_lr, new_lr = decay_learning_rate(optimizer, cfg.lr_decay_factor, cfg.min_lr)
                    bad_loss_epochs = 0
                    print(f"LR decayed: {old_lr:.8f} -> {new_lr:.8f}", flush=True)
        elif lr_scheduler is not None:
            old_lr = get_current_lr(optimizer)
            lr_scheduler.step()
            new_lr = get_current_lr(optimizer)
            print(f"LR scheduler step: {old_lr:.8f} -> {new_lr:.8f}", flush=True)

        row = [
            epoch,
            f"{epoch_lr:.10f}",
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{metrics['loss']:.6f}",
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
        history_row = dict(zip(header, row))
        history.append(history_row)
        save_training_curves(history, class_names, run_dir)
        print(f"Logged epoch metrics to {log_path}", flush=True)
        print(f"Updated curves in {run_dir}", flush=True)

    print_best_summary("Best validation result by val_mAP", best_map_summary, class_names, best_map_path)
    print_best_summary("Best validation result by val_acc", best_acc_summary, class_names, best_acc_path)


if __name__ == "__main__":
    main()
