from pathlib import Path

import torch
from torch import nn

from train_all_baselines_config import build_model, load_config, model_forward


CONFIGS = {
    "I3D": r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\baseline_config_3class_i3d_uniform32_sgd.json",
    "X3D": r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\baseline_config_3class_x3d_s_uniform32_sgd.json",
    "VideoMAE": r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\baseline_config_3class_videomae_sgd.json",
    "MViT": r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\baseline_config_3class_mvit_v2_s_original_sgd.json",
    "VideoSwin": r"C:\Users\zhengtang\Documents\Codex\2026-05-14\new-chat\baseline_config_3class_videoswin_original_sgd.json",
}


class ProfileWrapper(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def forward(self, x):
        return model_forward(
            self.model,
            x,
            self.cfg.model_name,
            slowfast_alpha=getattr(self.cfg, "slowfast_alpha", 4),
        )


def main():
    from thop import profile

    for name, config_path in CONFIGS.items():
        cfg = load_config(config_path)
        cfg.pretrained = False
        model = build_model(cfg, len(cfg.class_names)).eval()
        params = sum(param.numel() for param in model.parameters())
        num_frames = int(getattr(cfg, "num_frames", 32))
        image_size = int(getattr(cfg, "image_size", 224))
        x = torch.randn(1, num_frames, 3, image_size, image_size)
        wrapper = ProfileWrapper(model, cfg).eval()
        flops, _ = profile(wrapper, inputs=(x,), verbose=False)
        print(
            f"{name}\tmodel_name={cfg.model_name}\tParams={params / 1e6:.4f}M\t"
            f"FLOPs={flops / 1e9:.4f}G\tInput={num_frames}x{image_size}x{image_size}"
        )


if __name__ == "__main__":
    main()
