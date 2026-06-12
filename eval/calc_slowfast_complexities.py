from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from eval_checkpoint_confusion_matrix import load_checkpoint
from train_all_baselines_config import build_model, load_config, model_forward


CHECKPOINTS = {
    "SlowFast": r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\runs\slowfast_uniform32_基线\best_model_by_mAP.pth",
    "SlowFast+DESA": r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\runs\slowfast_uniform32_只加DESAflow分支\best_model_by_mAP.pth",
    "SlowFast+TDEM": r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\runs\slowfast_uniform32只加时序差分增强\best_model_by_mAP.pth",
    "SlowFast+DESA+TDEM": r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\runs\slowfast_uniform32_desa_tda\best_model_by_mAP.pth",
}


def load_cfg(ckpt):
    cfg = SimpleNamespace(**ckpt["config"])
    default_cfg = load_config(None)
    for key, value in vars(default_cfg).items():
        if not hasattr(cfg, key) or getattr(cfg, key) is None:
            setattr(cfg, key, value)
    cfg.pretrained = False
    class_names = list(ckpt.get("class_names") or cfg.class_names)
    cfg.class_names = class_names
    return cfg, class_names


def needs_mask(cfg):
    return (
        bool(getattr(cfg, "slowfast_dual_end_attention", False))
        or bool(getattr(cfg, "slowfast_tail_attention", False))
        or bool(getattr(cfg, "temporal_difference_use_dual_end_mask", False))
        or bool(getattr(cfg, "dual_end_focus_loss", False))
    ) and cfg.model_name == "slowfast"


def main():
    from thop import profile

    for name, ckpt_path in CHECKPOINTS.items():
        ckpt = load_checkpoint(Path(ckpt_path))
        cfg, class_names = load_cfg(ckpt)
        model = build_model(cfg, len(class_names)).eval()
        params = sum(param.numel() for param in model.parameters())
        num_frames = int(getattr(cfg, "num_frames", 32))
        image_size = int(getattr(cfg, "image_size", 224))
        videos = torch.randn(1, num_frames, 3, image_size, image_size)
        tail_masks = torch.ones(1, num_frames, 1, image_size, image_size) if needs_mask(cfg) else None

        class ProfileWrapper(nn.Module):
            def __init__(self, inner_model, inner_cfg):
                super().__init__()
                self.inner_model = inner_model
                self.inner_cfg = inner_cfg

            def forward(self, x, mask=None):
                return model_forward(
                    self.inner_model,
                    x,
                    self.inner_cfg.model_name,
                    tail_masks=mask,
                    slowfast_alpha=getattr(self.inner_cfg, "slowfast_alpha", 4),
                )

        wrapper = ProfileWrapper(model, cfg).eval()

        if tail_masks is not None:
            flops, _ = profile(wrapper, inputs=(videos, tail_masks), verbose=False)
        else:
            flops, _ = profile(wrapper, inputs=(videos,), verbose=False)

        print(
            f"{name}\tParams={params / 1e6:.4f}M\tFLOPs={flops / 1e9:.4f}G\t"
            f"Input={num_frames}x{image_size}x{image_size}"
        )


if __name__ == "__main__":
    main()
