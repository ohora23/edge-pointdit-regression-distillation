# PointDiT — RK3588-portable reproduction

Point-map diffusion transformer for **transparent-object monocular geometry**
(depth / 3D point maps from a single RGB image), engineered from day one to port to a
**Rockchip RK3588 NPU (1 core, ~2 TOPS)** at real-time (≥10 FPS).

Reproduction of [PointDiT (arXiv:2607.02515)](https://haofeixu.github.io/pointdit/):
frozen image encoder + a plain ViT that denoises point-map patches (XYZ as channels),
rectified-flow, one-step inference. Methodology: `.omc/research/pointdit-rk3588/REPORT.md`.

## Status (Phase 2 — PC build)
- ✅ Runs end-to-end on PC/GPU at both deploy resolutions (320×240, 160×120), 1-step.
- ✅ NPU-friendly profile (BatchNorm / ReLU-linear const-norm attn / ReLU) runs + trains.
- ✅ Fixed-shape ONNX export with **exact onnxruntime parity** (the RK3588 bridge).
- ✅ **RKNN conversion rehearsed** on x86 (no board): converts to rk3588, real-compute CPU
  fallback reduced to 1 op (see `.omc/research/pointdit-rk3588/RKNN_REHEARSAL.md`).
- ✅ Pretrained encoders wired: **DINOv2** (`--encoder dinov2`, ungated, verified),
  **DINOv3** (`--encoder dinov3`, gated — needs HF login). Conv tokenizer is the default.
- ✅ Rectified-flow training loop + **faithful TransCG loader** (real transparent-object data).
- ⏳ Phase 3: RKNN INT8 conversion + on-device benchmark (needs the board).

## Setup
```bash
uv venv .venv --python 3.12
./.venv/bin/python -m uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[encoders,export]"
```
> RTX 50-series (Blackwell / sm_120) needs the cu128+ PyTorch index.

## Use
```bash
# sanity
./.venv/bin/python scripts/check_env.py

# inference + latency (PC baseline)
./.venv/bin/python scripts/demo_infer.py --resolution high --steps 1
./.venv/bin/python scripts/demo_infer.py --resolution high --steps 1 --npu   # RK3588 profile
./.venv/bin/python scripts/demo_infer.py --image path/to/rgb.jpg --encoder dinov2 --steps 1

# train (synthetic smoke; swap --data for TransCG)
./.venv/bin/python scripts/train.py --synthetic --profile high --steps 50 --amp
# train on real transparent-object data (TransCG) with the DINOv2 encoder + deploy config
./.venv/bin/python scripts/train.py --data /path/to/TransCG --dataset transcg --encoder dinov2 --npu --profile high --steps 100000 --amp
# DREDS sim-to-real pretraining (needs EXR support: pip install OpenEXR)
./.venv/bin/python scripts/train.py --data /path/to/DREDS-CatKnown/train --dataset dreds --domain sim --npu --steps 200000 --amp

# export ONNX (RK3588 bridge) + parity check
./.venv/bin/python scripts/export_onnx.py --resolution high --npu
./.venv/bin/python scripts/export_onnx.py --resolution low  --npu

# --- RK3588 conversion rehearsal (x86, no board; see rknn venv setup in rknn_notes.md) ---
./.venv/bin/python scripts/export_onnx.py --resolution high --npu --fold-t --rehearsal --out artifacts/r.onnx
.venv-rknn/bin/python -m onnxsim artifacts/r.onnx artifacts/r_sim.onnx
.venv-rknn/bin/python scripts/rknn_convert.py --onnx artifacts/r_sim.onnx --resolution high --log .omc/logs/r.log
# INT8 quantization (fp16 ~38MB -> int8 ~20MB)
./.venv/bin/python scripts/make_calib.py --resolution high --n 100 --out calib_high   # add --data <TransCG> for real calib
.venv-rknn/bin/python scripts/rknn_convert.py --onnx artifacts/r_sim.onnx --resolution high --quantize --dataset calib_high/dataset.txt --out artifacts/pointdit_high_int8.rknn
```

## Layout
```
pointdit/
  config.py     # ModelConfig, deploy RESOLUTIONS, npu_profile() toggles
  encoders.py   # ConvTokenizer (default) | DinoEncoder (dinov2/dinov3, frozen)
  layers.py     # NPU-friendly DiT blocks (norm/attn/act toggles, fixed 2D sincos pos-embed)
  denoiser.py   # PointMapDiT: XYZ patchify + image-token concat + AdaLN-Zero
  model.py      # PointDiT (encoder+denoiser+flow sampler), ExportWrapper (fixed-shape)
  flow.py       # rectified-flow pair sampling, flow loss, affine-invariant point-map loss
  data.py       # resolution prep, point-map norm, SyntheticPointMaps, TransCG stub
  export/rknn_notes.md   # ONNX -> RKNN INT8 conversion + op-placement checklist
scripts/        # check_env, demo_infer, train, export_onnx
```

## RK3588 portability (baked in)
- **Fixed shapes** (240×320, 128×160) — no dynamic-shape CPU fallback.
- **`--npu` profile**: BatchNorm (folds into conv), ReLU, ReLU-linear attention
  (no Softmax on the attention path) — the ops that otherwise fall back to CPU on RKNN.
- Single-step `ExportWrapper` — no diffusion loop in the exported graph.

See `pointdit/export/rknn_notes.md` for the Phase 3 conversion + on-device benchmark.
```
> Encoder note: the runnable default is a conv tokenizer (needs no download). The paper's
> DINOv3 is the accuracy-phase target (`--encoder dinov3`, gated); `--encoder dinov2` is an
> ungated stand-in. Swap freely — the denoiser is encoder-agnostic.
```
