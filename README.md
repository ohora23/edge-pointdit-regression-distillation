# Edge-PointDiT — regression-distilled PointDiT for real-time transparent-object depth on RK3588

**[Project page](https://ohora23.github.io/edge-pointdit-regression-distillation/)** ·
Unofficial reproduction & edge-deployment adaptation of
[PointDiT (arXiv:2607.02515)](https://haofeixu.github.io/pointdit/), retargeted from a
1-step diffusion ViT to a **deterministic regression** model and shipped end-to-end
through ONNX → RKNN for a **Rockchip RK3588 NPU (1 core, ~2 TOPS)**.

## Introduction

Monocular depth for **transparent and specular objects** is a problem classical sensors
can't solve — stereo, ToF, and structured light all fail on glass and clear plastic
because there's no consistent return signal. [PointDiT](https://haofeixu.github.io/pointdit/)
tackles this by skipping the usual depth-map/VAE detour and denoising **raw XYZ point
maps** directly with a plain ViT conditioned on a frozen image encoder — no lossy latent
compression, no separate depth-then-lift step.

This repo reproduces that idea from the public project page/paper description (no
official code or weights were released) and asks a different question on top of it:
**can the same architecture run in real time on a 2-TOPS edge NPU?** That constraint
drove almost every design decision here, including one that diverges from the paper —
see [Method](#method--implementation).

## Final results: two model variants

The project ships two checkpoints for two different jobs — a high-resolution quality
ceiling (GPU-only) and an economical variant actually meant for the RK3588. Both are
plane-detrended object-component correlation on the official 52-scene held-out TransCG
split (see [Method](#method--implementation) for what that means).

| | High-resolution (quality ceiling) | Economical (RK3588 deploy) |
|---|---|---|
| Resolution / tokens | 384×512, 768 tokens (`xhigh` profile) | 320×240, 300 tokens |
| Checkpoint | `artifacts/reg_xhigh130_more/pointdit.pt` | `artifacts/reg_130_long/pointdit.pt` → ONNX → RKNN |
| Target | GPU only — quality ceiling, not real-time-deployable (attention cost ≈6.5× the deploy profile) | RK3588 NPU, 1 core ~2 TOPS, ≥10 FPS target |
| Held-out object corr. | **0.884** | 0.848 (PyTorch fp32 = RKNN fp16, bit-exact) |
| Held-out transparent corr. | **0.876** | 0.817 (PyTorch fp32 = RKNN fp16, bit-exact) |
| Size / precision | fp32, GPU-only | **fp16 80.4 MB** (lossless) · **INT8 43.5 MB** (0.750 / 0.706, ~0.10 corr. loss) |
| Speed measured so far | **11 ms/frame (~90 FPS)** on an RTX 5080 (single `model.predict()` forward) | Not yet measured on-device — needs a physical RK3588 board; numbers above are from RKNN's x86 simulator |

Both checkpoints share the same denoiser (frozen DINOv2 encoder + ~45M total params) —
only the token grid changes, so parameter count is identical and only compute cost
differs between the two.

## Method / implementation

**Architecture** (`pointdit/`):
- **Frozen image encoder** — pluggable: DINOv2 (`--encoder dinov2`, ungated, default for
  real runs), DINOv3 (`--encoder dinov3`, gated, needs an HF login), or a from-scratch
  conv tokenizer (`encoders.py`, no download needed, always available). Encoder tokens
  are bilinearly resampled onto the point-map's token grid so image and point tokens
  share one positional embedding — a real bug we hit and fixed (see `RKNN_REHEARSAL.md`).
- **Denoiser** (`denoiser.py`) — a plain ViT. The (noisy) point map is patchified with
  XYZ as channels, exactly like RGB patchify, concatenated with the projected image
  tokens, and processed by joint self-attention (in-context conditioning) with
  AdaLN-Zero timestep modulation — the core PointDiT idea.
- **Diffusion → regression (the key deviation from the paper):** PointDiT is a 1-step
  rectified-flow diffusion model. On our real training data (TransCG), 1-step diffusion
  plateaued at object-correlation ≈0.55 with visibly noisy predictions — sampling a
  clean point map from pure noise in a single step is intrinsically hard. TransCG's
  ground truth is deterministic (a single real depth capture per scene, not a
  distribution), so a **deterministic regression head** (`model.predict()`: zero input,
  `t=0`, one forward pass, no diffusion loop) is both the right inductive bias for this
  data *and* strictly better for NPU deployment (one forward pass, no sampling loop).
  Switching to regression took object correlation from 0.55 → 0.92 outright.
- **NPU-friendly ops** (`layers.py`, `config.py: npu_profile()`) — every op that RKNN
  would otherwise fall back to CPU on is redesigned: BatchNorm instead of LayerNorm
  (folds into the preceding conv/linear), ReLU instead of GELU, and — after a real
  finding — **softmax attention, not linear attention**. We first tried an NPU-cheaper
  const-normalized linear attention, but on real data it NaN's under fp16/AMP when
  combined with BatchNorm. Softmax attention has zero Div ops, converts to RKNN with
  only a single CPU fallback op (the final `DepthToSpace` upsample), is fp16-stable, and
  is *more* accurate — so it won the A/B outright, not just as a fallback compromise.
- **Fixed deploy shapes** — 320×240 (patch 16 → 15×20 = 300 tokens) and 160×120→padded
  128 (8×10 = 80 tokens). RKNN needs static shapes; a 384×512 "xhigh" profile exists
  purely as a **quality ceiling** for offline evaluation (2.5× denser token grid) — it is
  not a deployment target (attention cost ≈6.5× → well under the 10 FPS bar on this NPU).

**Evaluation methodology note:** raw point-map correlation is dominated by the flat
table plane in-frame, which saturates fast and hides real progress/regressions. Every
number below is measured after **plane-detrending** — fitting and subtracting a plane
from the GT before computing correlation — which isolates the actual *object* geometry
signal (`pointdit/eval.py`).

## Results

Progression on TransCG (`train`/`test` = official disjoint scene split, 78/52 scenes),
object-component correlation after plane-detrend:

| Stage | Object corr. | Transparent-only corr. | Note |
|---|---|---|---|
| 10 scenes, diffusion | 0.38 | 0.04 | 1-step diffusion, small data |
| 10 scenes + gradient loss | 0.43 | 0.14 | multi-scale gradient loss added |
| 20 scenes | 0.53 | 0.16 | data scaling confirmed |
| 60 scenes | 0.64 | 0.13 | transparent stagnation (later found to be noise, not a wall) |
| 130 scenes (full data) | 0.73 | 0.20 | **diffusion → regression switch here: 0.55 → 0.92** |
| 130 scenes, +50k steps | 0.842 | 0.448 | longer training, standard metrics looked saturated but object signal kept rising |
| 130 scenes, high-res (xhigh) | 0.895 | 0.730 | 384×512 quality ceiling, not deployable as-is |
| + RGB-guided refine head | 0.904 | 0.751 | marginal — not a real fine-detail lever, dropped from the recommended path |
| **Held-out generalization** (52 unseen scenes, 200 frames) | **0.884** | **0.876** | best checkpoint, `artifacts/reg_xhigh130_more/`, confirms it isn't overfit to the 78 train scenes |

**On-device RK3588 FPS is not yet measured** — this project has GPU-only compute and no
physical RK3588 board; everything below is x86 simulation of the converted graph.

## Deployment (RK3588 export pipeline)

Deploy target is the 320×240 regression model (`reg_130_long` weights, not the xhigh
quality-ceiling one). Pipeline: `export_onnx.py --regression` → `onnxsim` →
`rknn_convert.py` (see `pointdit/export/rknn_notes.md` for the full op-placement
checklist and gotchas).

| Precision | Object corr. | Transparent corr. | Size | Note |
|---|---|---|---|---|
| PyTorch fp32 | 0.848 | 0.817 | — | reference |
| RKNN fp16 | 0.848 | 0.817 | 80.4 MB | **bit-for-bit equivalent to fp32** |
| RKNN INT8 | 0.750 | 0.706 | 43.5 MB | ~0.10 correlation loss |
| RKNN INT8 hybrid (softmax + head kept fp16) | 0.708 | — | 43.8 MB | attempted fix, **did not recover accuracy** |

Per-layer analysis (`accuracy_analysis`) showed 305/309 layers individually quantize
fine (cos ≥0.99); the loss is small per-layer error **accumulating across the network's
depth**, not a couple of bad layers — which is why keeping a few layers in fp16
(hybrid quantization) didn't help. Deployment decision: **fp16 (80MB) when accuracy
matters, INT8 (43MB) when size/speed matter**; INT8 loss is real but bounded.

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
# train the deployed regression model on real transparent-object data (TransCG)
./.venv/bin/python scripts/train.py --data /path/to/TransCG --dataset transcg --encoder dinov2 --npu --profile high --regression --grad_weight 2.0 --steps 100000 --amp
# DREDS sim-to-real pretraining (needs EXR support: pip install OpenEXR) — wired but not yet run
./.venv/bin/python scripts/train.py --data /path/to/DREDS-CatKnown/train --dataset dreds --domain sim --npu --steps 200000 --amp

# export ONNX (RK3588 bridge) + parity check
./.venv/bin/python scripts/export_onnx.py --resolution high --npu --regression
./.venv/bin/python scripts/export_onnx.py --resolution low  --npu --regression

# --- RK3588 conversion (x86, no board; see rknn venv setup in rknn_notes.md) ---
.venv-rknn/bin/python -m onnxsim artifacts/r.onnx artifacts/r_sim.onnx
.venv-rknn/bin/python scripts/rknn_convert.py --onnx artifacts/r_sim.onnx --resolution high --inputs image --log .omc/logs/r.log
# INT8 quantization (fp16 ~80MB -> int8 ~44MB)
./.venv/bin/python scripts/make_calib.py --resolution high --n 100 --data /path/to/TransCG --out calib_high
.venv-rknn/bin/python scripts/rknn_convert.py --onnx artifacts/r_sim.onnx --resolution high --inputs image --quantize --dataset calib_high/dataset.txt --out artifacts/pointdit_high_int8.rknn
```

## Layout
```
pointdit/
  config.py     # ModelConfig, deploy RESOLUTIONS, npu_profile() toggles
  encoders.py   # ConvTokenizer (default) | DinoEncoder (dinov2/dinov3, frozen)
  layers.py     # NPU-friendly DiT blocks (norm/attn/act toggles, fixed 2D sincos pos-embed)
  denoiser.py   # PointMapDiT: XYZ patchify + image-token concat + AdaLN-Zero
  model.py      # PointDiT (encoder+denoiser+flow sampler), ExportWrapper(Regression)
  flow.py       # rectified-flow pair sampling, flow loss, affine-invariant point-map loss
  data.py       # resolution prep, point-map norm, SyntheticPointMaps, TransCG/DREDS loaders
  eval.py       # held-out AbsRel/δ1/plane-detrended correlation + depth viz
  export/rknn_notes.md   # ONNX -> RKNN INT8 conversion + op-placement checklist
scripts/        # check_env, demo_infer, train, export_onnx, make_calib, rknn_convert
docs/           # project page (GitHub Pages, served from /docs)
```

## RK3588 portability (baked in)
- **Fixed shapes** (240×320, 128×160) — no dynamic-shape CPU fallback.
- **`--npu` profile**: BatchNorm (folds into conv), ReLU, softmax attention (no Div) —
  the ops that otherwise fall back to CPU on RKNN, resolved to a single remaining
  fallback op (`DepthToSpace`).
- Single-step regression export wrapper — no diffusion loop in the exported graph.

See `pointdit/export/rknn_notes.md` for the full conversion checklist.

> Encoder note: the runnable default is a conv tokenizer (needs no download). DINOv2 is
> the verified real-data encoder (`--encoder dinov2`); the paper's DINOv3 is available
> gated behind an HF login (`--encoder dinov3`). Swap freely — the denoiser is
> encoder-agnostic.

## Status
- ✅ PC/GPU training + eval pipeline, real TransCG data, held-out generalization verified.
- ✅ ONNX export with exact onnxruntime parity, RKNN conversion (fp16 lossless, INT8
  characterized), fixed-shape NPU-friendly graph.
- ⏳ On-device RK3588 benchmark (FPS, real NPU timing) — needs physical hardware.
- ⏳ DREDS synthetic-transparent pretraining — loader implemented, data not yet pulled.

## Credit
Reproduces the method described in [PointDiT](https://haofeixu.github.io/pointdit/)
(arXiv:2607.02515). This is an independent, unofficial implementation for edge-NPU
deployment research — not affiliated with the original authors.
