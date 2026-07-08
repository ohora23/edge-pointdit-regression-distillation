#!/usr/bin/env python
"""Export a perturbed, fold-t, NPU-profile ONNX for a given attention variant (rehearsal)."""
import argparse
from dataclasses import replace

import torch

from pointdit.config import ModelConfig, RESOLUTIONS
from pointdit.model import ExportWrapperFixedT, PointDiT

ap = argparse.ArgumentParser()
ap.add_argument("--attn-type", default="relu_linear", choices=["relu_linear", "softmax"])
ap.add_argument("--attn-norm", default="const", choices=["data", "const"])
ap.add_argument("--resolution", default="high")
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.manual_seed(0)
cfg = replace(ModelConfig(encoder="conv").npu_profile(),
              attn_type=a.attn_type, attn_norm=a.attn_norm)
m = PointDiT(cfg).eval()
H, W = RESOLUTIONS[a.resolution]
x = m.sample(torch.rand(1, 3, H, W), steps=1)
assert torch.isfinite(x).all(), "non-finite forward"
with torch.no_grad():
    for p in m.parameters():
        p.add_(0.02 * torch.randn_like(p))
w = ExportWrapperFixedT(m).eval()
torch.onnx.export(w, (torch.randn(1, 3, H, W), torch.randn(1, 3, H, W)), a.out,
                  input_names=["image", "noise"], output_names=["points"],
                  opset_version=17, dynamo=False)
print(f"exported {a.out} attn_type={a.attn_type} attn_norm={a.attn_norm} forward_ok")
