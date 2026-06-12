import argparse
from pathlib import Path

import torch

from C3D_video_level import C3D
from I3D_video_level import build_i3d
from R3D18_video_level import build_r3d18
from SlowFast_video_level import build_slowfast, pack_slowfast_pathway
from X3D_video_level import build_x3d


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def try_count_flops(model, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None

    model.eval()
    with torch.no_grad():
        try:
            return FlopCountAnalysis(model, inputs).total()
        except Exception as exc:
            print(f"FLOPs failed: {exc}")
            return None


def fmt_params(n):
    return f"{n / 1e6:.3f} M"


def fmt_flops(n):
    if n is None:
        return "N/A"
    return f"{n / 1e9:.3f} G"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clips", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--c3d_size", type=int, default=112)
    parser.add_argument("--r3d_size", type=int, default=112)
    parser.add_argument("--x3d_size", type=int, default=160)
    parser.add_argument("--i3d_size", type=int, default=160)
    parser.add_argument("--slowfast_size", type=int, default=160)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--slowfast_frames", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)

    specs = [
        (
            "C3D",
            C3D(num_classes=2),
            torch.randn(1, 3, args.frames_per_clip, args.c3d_size, args.c3d_size),
            False,
        ),
        (
            "R3D-18",
            build_r3d18(num_classes=2, pretrained=args.pretrained),
            torch.randn(1, 3, args.frames_per_clip, args.r3d_size, args.r3d_size),
            False,
        ),
        (
            "X3D-S",
            build_x3d(num_classes=2, pretrained=args.pretrained),
            torch.randn(1, 3, args.frames_per_clip, args.x3d_size, args.x3d_size),
            False,
        ),
        (
            "I3D-R50",
            build_i3d(num_classes=2, pretrained=args.pretrained),
            torch.randn(1, 3, args.frames_per_clip, args.i3d_size, args.i3d_size),
            False,
        ),
        (
            "SlowFast-R50",
            build_slowfast(num_classes=2, pretrained=args.pretrained),
            torch.randn(1, 3, args.slowfast_frames, args.slowfast_size, args.slowfast_size),
            True,
        ),
    ]

    print("model,params,clip_flops,video_flops")
    for name, model, x, is_slowfast in specs:
        model = model.to(device).eval()
        x = x.to(device)
        inputs = pack_slowfast_pathway(x) if is_slowfast else x
        params = count_params(model)
        flops = try_count_flops(model, inputs)
        video_flops = flops * args.num_clips if flops is not None else None
        print(f"{name},{fmt_params(params)},{fmt_flops(flops)},{fmt_flops(video_flops)}")


if __name__ == "__main__":
    main()
