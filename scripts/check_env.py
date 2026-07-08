#!/usr/bin/env python
"""Quick environment + pipeline sanity check."""
import torch

from pointdit.config import ModelConfig, RESOLUTIONS
from pointdit.model import PointDiT


def main():
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0),
              "sm", torch.cuda.get_device_capability(0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for npu in (False, True):
        cfg = ModelConfig(encoder="conv")
        if npu:
            cfg = cfg.npu_profile()
        m = PointDiT(cfg).to(dev).eval()
        for name, (H, W) in RESOLUTIONS.items():
            img = torch.rand(1, 3, H, W, device=dev)
            x = m.sample(img, steps=1)
            assert torch.isfinite(x).all(), "non-finite output"
            print(f"  npu={npu} {name} {H}x{W}: sample {tuple(x.shape)} OK")
    print("ENV_OK")


if __name__ == "__main__":
    main()
