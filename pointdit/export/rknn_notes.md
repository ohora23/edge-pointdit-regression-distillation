# RK3588 (RKNN) conversion notes — the Phase 3 bridge

The PC pipeline exports fixed-shape ONNX with exact onnxruntime parity. RKNN conversion
runs on an **x86_64 Linux host** with `rknn-toolkit2` (not on the board), producing a
`.rknn` file that the board runs via `rknn-toolkit-lite2` / RKNPU2 runtime.

## 0. Why the model is already RKNN-oriented
- **Fixed shapes** (240×320, 128×160) — no dynamic shapes → no shape-driven CPU fallback.
- **NPU profile** (`--npu`): BatchNorm (folds into conv), ReLU (native), ReLU-linear
  attention (no `Softmax` on the attention path). Toggle in `ModelConfig.npu_profile()`.
- Single-step `ExportWrapper` (image, noise, t) → no diffusion loop in the graph.

## 1. Host setup (x86) — VERIFIED 2026-07-04, see ../../.omc/research/pointdit-rk3588/RKNN_REHEARSAL.md
Use a SEPARATE venv (rknn pins numpy<=1.26.4, conflicts with the torch env). Exact working set:
```bash
uv venv .venv-rknn --python 3.12
uv pip install --python .venv-rknn/bin/python \
    "rknn-toolkit2==2.3.2" "onnx==1.16.1" "numpy<=1.26.4" "setuptools<80" onnxsim onnxruntime
```
Gotchas found (all must be applied or conversion crashes):
- `onnx>=1.18` removed `onnx.mapping` -> pin **onnx==1.16.1**.
- `setuptools>=81` removed `pkg_resources` -> pin **setuptools<80**.
- Untrained AdaLN-Zero model (gates==0) crashes rknn `fold_constant` -> export with a trained
  ckpt, or `scripts/export_onnx.py --rehearsal` (perturbs weights). See rehearsal doc.
- Always run `onnxsim` on the exported ONNX before conversion.

## 2. Export ONNX (this repo)
```bash
python scripts/export_onnx.py --resolution high --npu --out artifacts/pointdit_high_npu.onnx
python scripts/export_onnx.py --resolution low  --npu --out artifacts/pointdit_low_npu.onnx
```

## 3. Convert ONNX → RKNN (INT8) — sketch
```python
from rknn.api import RKNN
rknn = RKNN(verbose=True)
rknn.config(target_platform="rk3588",
            quantized_dtype="asymmetric_quantized-8",
            optimization_level=3)
rknn.load_onnx(model="artifacts/pointdit_high_npu.onnx",
               inputs=["image", "noise", "t"],
               input_size_list=[[1,3,240,320],[1,3,240,320],[1]])
# INT8 PTQ: 100-500 in-domain (transparent-object) frames as calibration
rknn.build(do_quantization=True, dataset="calib/list.txt")
rknn.export_rknn("artifacts/pointdit_high_npu.rknn")
```

## 4. CHECK op placement (the #1 real risk — see report R4.4)
After `build`, inspect the per-layer target report: any op landing on **CPU** (esp.
attention MatMul, LayerNorm, Softmax, Gather) blows the 100 ms budget.
- If `Einsum` from ReLU-linear attention is unsupported → rewrite as explicit `MatMul`
  + `Transpose` (already the plan; see layers.py `Attention.relu_linear`).
- If LayerNorm remains anywhere → ensure `--npu` (BatchNorm) profile was used.
- Prefer running the frozen encoder on NPU and, if needed, splitting stubborn ops to CPU
  (backbone-on-NPU / head-on-CPU pattern) rather than accepting whole-graph fallback.

## 5. On-device benchmark (board)
```python
from rknnlite.api import RKNNLite
rknn = RKNNLite(); rknn.load_rknn("pointdit_high_npu.rknn")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)   # single core ~2 TOPS
# time rknn.inference(...) over 100 frames -> ms/frame, target <=100 ms (>=10 FPS)
```

## 6. Success gate (Phase 3)
`>= 10 FPS @ 320×240 on NPU_CORE_0`, INT8, with task-usable transparent-object depth.
Record latency/memory/accuracy into `.omc/autoresearch/pointdit-rk3588/` as a new mission.
