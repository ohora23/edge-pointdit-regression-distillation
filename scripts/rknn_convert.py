#!/usr/bin/env python
"""ONNX -> RKNN conversion rehearsal for RK3588 (runs on x86 host, no board needed).

Purpose: build the model with rknn-toolkit2 and capture the per-op NPU/CPU placement
report so we can find CPU-fallback ops (attention/Einsum/LayerNorm/Softmax) BEFORE the
board arrives. Use do_quantization=False for pure op-support checking (fp16, no dataset).

Run with the rknn venv:
  .venv-rknn/bin/python scripts/rknn_convert.py --onnx artifacts/pointdit_high_npu_foldt.onnx \
      --resolution high --log .omc/logs/rknn_high.log
"""
import argparse
import os
import sys

RESOLUTIONS = {"high": (240, 320), "low": (128, 160)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--resolution", default="high", choices=list(RESOLUTIONS))
    ap.add_argument("--inputs", nargs="+", default=["image", "noise"])
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--dataset", default=None, help="calibration list.txt (if --quantize)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--opt-level", type=int, default=3, dest="opt_level")
    args = ap.parse_args()

    from rknn.api import RKNN

    H, W = RESOLUTIONS[args.resolution]
    n_in = len(args.inputs)
    input_size_list = [[1, 3, H, W] for _ in range(n_in)]
    # passthrough normalization (our ONNX already consumes normalized inputs)
    mean_values = [[0, 0, 0] for _ in range(n_in)]
    std_values = [[1, 1, 1] for _ in range(n_in)]

    log = args.log or f".omc/logs/rknn_{args.resolution}.log"
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    out = args.out or args.onnx.replace(".onnx", ".rknn")

    rknn = RKNN(verbose=True, verbose_file=log)
    rknn.config(target_platform="rk3588", mean_values=mean_values, std_values=std_values,
                optimization_level=args.opt_level)
    print(f"[load_onnx] {args.onnx} inputs={args.inputs} sizes={input_size_list}")
    ret = rknn.load_onnx(model=args.onnx, inputs=args.inputs, input_size_list=input_size_list)
    if ret != 0:
        print("LOAD_FAILED", ret); return 2

    print(f"[build] do_quantization={args.quantize}")
    ret = rknn.build(do_quantization=args.quantize,
                     dataset=args.dataset if args.quantize else None)
    if ret != 0:
        print("BUILD_FAILED", ret, "-> see log:", log)
        rknn.release(); return 3

    ret = rknn.export_rknn(out)
    print("EXPORT", "OK" if ret == 0 else f"FAILED({ret})", "->", out)
    rknn.release()
    sz = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
    print(f"BUILD_OK rknn={out} ({sz:.1f} MB) verbose_log={log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
