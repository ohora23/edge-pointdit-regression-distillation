#!/usr/bin/env python
"""Export PointDiT to fixed-shape ONNX (the RK3588 bridge) and check onnxruntime parity.

Exports the single-step ExportWrapper (image, noise, t) -> point map, at a fixed deploy
resolution, then verifies torch vs onnxruntime outputs match. RKNN conversion happens
later on an x86 host with rknn-toolkit2 (see pointdit/export/rknn_notes.md).

Examples:
  python scripts/export_onnx.py --resolution high --npu --out artifacts/pointdit_high.onnx
  python scripts/export_onnx.py --resolution low  --npu --out artifacts/pointdit_low.onnx
"""
import argparse
import os

import numpy as np
import torch

from pointdit.config import ModelConfig, RESOLUTIONS
from pointdit.model import ExportWrapper, ExportWrapperFixedT, ExportWrapperRegression, PointDiT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", default="high", choices=list(RESOLUTIONS))
    ap.add_argument("--encoder", default="conv", choices=["conv", "dinov2", "dinov2b", "dinov3"])
    ap.add_argument("--npu", action="store_true", help="export the NPU-friendly profile")
    ap.add_argument("--regression", action="store_true",
                    help="export the regression deploy graph: single image input -> point map "
                         "(predict path; cleanest RKNN graph, no noise/t inputs)")
    ap.add_argument("--refine", action="store_true", help="model has the RGB-guided refine head")
    ap.add_argument("--fold-t", action="store_true",
                    help="RKNN-friendly 2-input (image, noise) graph with t folded to 1")
    ap.add_argument("--rehearsal", action="store_true",
                    help="perturb weights so an UNtrained model (AdaLN-Zero gates == 0) "
                         "exports a non-degenerate graph that RKNN can constant-fold. "
                         "Only for conversion rehearsal; real exports use --ckpt.")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    H, W = RESOLUTIONS[args.resolution]
    cfg = ModelConfig(encoder=args.encoder)
    if args.npu:
        cfg = cfg.npu_profile()
    if args.refine:
        from dataclasses import replace
        cfg = replace(cfg, refine=True)
    model = PointDiT(cfg).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=False)
    if args.rehearsal:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.02 * torch.randn_like(p))

    img = torch.randn(1, 3, H, W)
    noise = torch.randn(1, 3, H, W)
    t = torch.ones(1)

    if args.regression:
        wrapper = ExportWrapperRegression(model).eval()
        example = (img,)
        input_names = ["image"]
        feed = {"image": img.numpy()}
    elif args.fold_t:
        wrapper = ExportWrapperFixedT(model).eval()
        example = (img, noise)
        input_names = ["image", "noise"]
        feed = {"image": img.numpy(), "noise": noise.numpy()}
    else:
        wrapper = ExportWrapper(model).eval()
        example = (img, noise, t)
        input_names = ["image", "noise", "t"]
        feed = {"image": img.numpy(), "noise": noise.numpy(), "t": t.numpy()}

    with torch.no_grad():
        ref = wrapper(*example).numpy()

    suffix = f"{'_npu' if args.npu else ''}{'_reg' if args.regression else ''}{'_foldt' if args.fold_t else ''}"
    out = args.out or f"artifacts/pointdit_{args.resolution}{suffix}.onnx"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper, example, out,
        input_names=input_names, output_names=["points"],
        opset_version=args.opset, dynamo=False,
    )
    print(f"exported ONNX -> {out}  (fixed shape {H}x{W}, npu={args.npu}, fold_t={args.fold_t})")

    # onnx structural check
    import onnx
    onnx.checker.check_model(onnx.load(out))
    print("onnx.checker: OK")

    # onnxruntime parity
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, feed)[0]
    diff = np.abs(ref - ort_out)
    print(f"parity: max|torch-onnx|={diff.max():.3e}  mean={diff.mean():.3e}")
    ok = diff.max() < 1e-3
    print("PARITY_OK" if ok else "PARITY_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
