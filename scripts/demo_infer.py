#!/usr/bin/env python
"""Run PointDiT inference on an image (or synthetic), save a depth viz, benchmark latency.

Examples:
  python scripts/demo_infer.py --resolution high --steps 1
  python scripts/demo_infer.py --image path/to/rgb.jpg --encoder dinov2 --steps 1
  python scripts/demo_infer.py --resolution low --npu   # RK3588-friendly profile
"""
import argparse
import time

import numpy as np
import torch

from pointdit.config import ModelConfig, RESOLUTIONS
from pointdit.data import prepare_image
from pointdit.model import PointDiT


def load_image(path, H, W):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((W, H))
    arr = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0
    return arr.unsqueeze(0)


def save_depth_viz(points, out_path):
    z = points[0, 2].detach().cpu().numpy()
    z = (z - z.min()) / (np.ptp(z) + 1e-6)
    try:
        import cv2
        cm = cv2.applyColorMap((z * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(out_path, cm)
    except Exception:
        from PIL import Image
        Image.fromarray((z * 255).astype(np.uint8)).save(out_path)


@torch.no_grad()
def bench(model, img, steps, iters=50):
    for _ in range(10):
        model.sample(img, steps=steps)
    if img.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.sample(img, steps=steps)
    if img.is_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--resolution", default="high", choices=list(RESOLUTIONS))
    ap.add_argument("--encoder", default="conv", choices=["conv", "dinov2", "dinov3"])
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--npu", action="store_true", help="use NPU-friendly toggles")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="demo_depth.png")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    H, W = RESOLUTIONS[args.resolution]

    cfg = ModelConfig(encoder=args.encoder)
    if args.npu:
        cfg = cfg.npu_profile()
    model = PointDiT(cfg).to(device).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
        print(f"loaded checkpoint {args.ckpt}")

    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"encoder={cfg.encoder} npu={args.npu} params={nparam:.1f}M "
          f"resolution={args.resolution} {H}x{W} steps={args.steps} device={device}")

    if args.image:
        img01 = load_image(args.image, H, W).to(device)
        img = (img01 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)) \
            / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    else:
        img = prepare_image(torch.rand(1, 3, H, W, device=device), args.resolution)
        print("no --image: using synthetic input")

    pts = model.sample(img, steps=args.steps)
    save_depth_viz(pts, args.out)
    print(f"point map {tuple(pts.shape)} -> depth viz saved to {args.out}")

    ms = bench(model, img, args.steps)
    fps = 1000.0 / ms
    print(f"latency: {ms:.2f} ms/frame  ({fps:.1f} FPS on {device})")


if __name__ == "__main__":
    main()
