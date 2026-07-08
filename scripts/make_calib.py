#!/usr/bin/env python
"""Generate an RKNN INT8 calibration set for the 2-input (image, noise) fold-t graph.

Writes <out>/img_XXXX.npy, <out>/noise_XXXX.npy and a dataset.txt where each line lists
the inputs for one sample, space-separated in load_onnx(inputs=[...]) order:
    img_0000.npy noise_0000.npy

- Image samples come from TransCG (--data) when available (representative for accuracy),
  otherwise Gaussian noise (fine to exercise the quantization pipeline).
- The noise input is genuinely ~N(0,1) at inference, so random normal IS representative.
"""
import argparse
import os

import numpy as np
import torch

from pointdit.config import RESOLUTIONS
from pointdit.data import IMAGENET_MEAN, IMAGENET_STD, prepare_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", default="high", choices=list(RESOLUTIONS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--data", default=None, help="TransCG root for real calibration images")
    ap.add_argument("--out", default="calib")
    args = ap.parse_args()

    H, W = RESOLUTIONS[args.resolution]
    os.makedirs(args.out, exist_ok=True)

    ds = None
    if args.data:
        from pointdit.data import TransCGDataset
        ds = TransCGDataset(args.data, profile=args.resolution)
        print(f"using {len(ds)} TransCG frames for image calibration")

    lines = []
    torch.manual_seed(0)
    for i in range(args.n):
        if ds is not None and len(ds) > 0:
            img01 = ds[i % len(ds)]["image"][None]  # [1,3,H,W] in [0,1]
            img = ((img01 - IMAGENET_MEAN) / IMAGENET_STD).numpy().astype("float32")
        else:
            img = prepare_image(torch.rand(1, 3, H, W), args.resolution).numpy().astype("float32")
        noise = torch.randn(1, 3, H, W).numpy().astype("float32")
        img_name, noise_name = f"img_{i:04d}.npy", f"noise_{i:04d}.npy"
        np.save(os.path.join(args.out, img_name), img)
        np.save(os.path.join(args.out, noise_name), noise)
        # RKNN resolves dataset.txt entries relative to the file's own dir -> basenames only.
        lines.append(f"{img_name} {noise_name}")

    list_path = os.path.join(args.out, "dataset.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.n} calib samples + {list_path}")


if __name__ == "__main__":
    main()
