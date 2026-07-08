# RKNN 변환 리허설 결과 (x86 호스트, 보드 없음) — 2026-07-04

**목표:** RK3588 보드 도착 전에, x86에서 `rknn-toolkit2`로 PointDiT ONNX를 rk3588 타깃으로 변환해
**연산 CPU fallback(리포트 R4.4의 최대 리스크)을 미리 식별**.

**결론(요약):** PointDiT NPU-friendly 그래프는 **rk3588로 변환 성공** (high/low 두 shape 모두 `.rknn` 생성).
198 ops 중 **~185가 NPU, 실질 연산 fallback은 9개**(Div ×8 = linear-attention 정규화, DepthToSpace ×1 = unpatchify).
치명적 whole-graph fallback이나 REGTASK overflow는 **없음**. 단, 변환을 막는 환경/구조 함정 4개를 발견·해결.

## 환경
- 호스트: x86_64, 별도 venv `.venv-rknn` (Python 3.12)
- `rknn-toolkit2==2.3.2` (PyPI, cp312 x86_64 wheel)
- 변환은 보드 없이 x86에서 완결(PC-simulate). `do_quantization=False`(fp16)로 op 배치만 검사.

## 발견한 변환 차단 함정 & 해결 (재현에 필수)
| # | 증상 | 원인 | 해결 |
|---|------|------|------|
| 1 | `AttributeError: module 'onnx' has no attribute 'mapping'` | onnx≥1.18이 `onnx.mapping` 제거 | rknn venv에 **`onnx==1.16.1`**, `numpy<=1.26.4` 고정 |
| 2 | `ModuleNotFoundError: pkg_resources` (RKNN() 생성 시) | setuptools≥81이 `pkg_resources` 제거 | **`setuptools<80`** 설치 |
| 3 | `TypeError: argument of type 'NoneType' is not iterable` (fold_constant) | **AdaLN-Zero 초기화** → 미학습 gate=0이 퇴화 상수 서브그래프 생성 → rknn 상수폴딩이 None 참조 | **비영 가중치**로 export. 학습된 ckpt면 자동 OK. 미학습 리허설은 `export_onnx.py --rehearsal`(가중치 미세 섭동) |
| 4 | 그래프에 잔여 noise op / Identity·Shape·Slice 다수 | TorchScript export 부산물 | **`onnxsim`으로 사전 단순화** (261→61 Constant 등) |

> 함정 1·2·4는 일반적 rknn-toolkit2 이슈, **3은 DiT/AdaLN-Zero 특유**로 이번에 규명. 미학습 모델을 변환 리허설할 땐 반드시 `--rehearsal`.

## 연산 배치 리포트 (high 320×240, NPU-friendly, fp16)
- 총 198 ops: **NPU 185 / CPU 13** (I/O 경계 3 제외 시 실질 fallback 9)

| CPU 배치 op | 개수 | 정체 | 조치 |
|---|---|---|---|
| **Div** | 8 | ReLU-linear attention 정규화 `1/(q·k_sum)` (블록당 1) | **남은 최대 항목.** 데이터 의존 reciprocal → 고정/학습 스칼라 정규화로 대체하거나 소폭 CPU 비용 수용 검토 |
| DepthToSpace | 1 | unpatchify (pixel_shuffle) | 출력 경계 단일 op, 허용. (naive 5D Transpose+Reshape 2개 → 1개로 이미 개선) |
| Input/OutputOperator | 3 | I/O 경계 | 정상(진짜 fallback 아님) |

NPU에 안착한 핵심 연산: **Conv 29, exMatMul 16(attention/proj/mlp의 MatMul), ConvRelu 12, ConvAdd 16, BatchNorm(fold됨), Relu 16, Split 8, Transpose 대부분**. 즉 **attention·MLP·인코더의 무거운 GEMM은 전부 NPU**.

low 160×120: 동일 구조, NPU 186 / CPU 12 (`.rknn` 38.1 MB). high `.rknn` 47.4 MB(fp16; INT8 양자화 시 추가 축소).

## 이 리허설로 코드에 반영한 개선
1. **einsum 전면 제거**: unpatchify(`torch.einsum`) → reshape/permute→pixel_shuffle; relu-linear attention einsum → matmul/transpose. (rknn Einsum 취약성 회피)
2. **unpatchify → DepthToSpace**: 5D Transpose+Reshape(CPU 2개) → DepthToSpace(CPU 1개).
3. `export_onnx.py --rehearsal`, `--fold-t`(2-입력 static 그래프), `scripts/rknn_convert.py` 추가.

## Div ×8 fallback 제거 — 완료 (2026-07-04)
리허설로 A/B 비교 후 해결. 실질 연산 fallback **9 → 1**.

| 변종 | NPU ops | 실질 fallback | Div |
|------|--------|--------------|-----|
| C. relu_linear + data-norm (기존) | 184 | 9 | **8** |
| **A. relu_linear + const-norm (채택, npu_profile 기본)** | 120 | **1** (DepthToSpace) | **0** |
| B. softmax attention (batchnorm+relu) | 80 | **1** (DepthToSpace) | **0** |

- **원인:** linear-attention 정규화 `1/(q·k_sum)` → Reciprocal→Div, RKNN NPU 미지원(블록당 1개, CPU).
- **해결(A):** 데이터 의존 분모를 **고정 스칼라 `1/N`**(배포 해상도별 토큰수 고정)로 대체 → 상수 Mul로 접힘, NPU 상주. `ModelConfig.attn_norm="const"`, `npu_profile()` 기본값. 학습이 스케일 흡수.
- **뜻밖의 발견(B):** **softmax attention도 RK3588에 완전 변환**(op 80개, DepthToSpace 외 전부 NPU). "softmax는 NPU 불가"라는 통념 반증 — 토큰 600개 규모에선 REGTASK overflow 없이 fused. **정확도 이점**이 있어 Phase 3 온디바이스에서 A vs B 실측 비교 권장.
- **train==deploy:** `train.py --npu`로 배포 설정(BatchNorm/ReLU/const-norm) 그대로 학습 → 학습·배포 아키텍처 불일치 제거. 학습된 ckpt는 `--rehearsal` 없이 변환됨(비영 gate). 검증: NPU 120 / fallback 1.
- 잔여 **DepthToSpace ×1**(unpatchify, 출력 경계 단일 op)은 허용. 더 줄이려면 최종층을 ConvTranspose2d로 대체 검토(Phase 3).

## DepthToSpace fallback 조사 (c) — 제거 불가, 유지가 최선
unpatchify를 ConvTranspose2d로 대체 시도. 결과: **DepthToSpace든 ConvTranspose(`exConvTransposePad`)든
정확히 출력 경계 1개 op가 CPU로 남음**(둘 다 fallback 1). DepthToSpace(순수 메모리 재배열)가
ConvTranspose(conv+pad)보다 CPU에서 싸므로 **pixelshuffle 유지**. `head="convtranspose"`는 온디바이스
A/B용 config 옵션으로 보존. → unpatchify는 이 아키텍처의 실질 fallback 하한(1개, 출력 경계, 왕복 없음).

## INT8 양자화 파이프라인 (b) — 완성, 단 op 회귀 발견
`make_calib.py`(멀티입력 calibration: `dataset.txt` 각 줄 `img_XXXX.npy noise_XXXX.npy`, **basename 상대경로**)
+ `rknn_convert.py --quantize`. asymmetric_quantized-8, 148 레이어 양자화.
- **크기:** fp16 37.5 MB → **INT8 20.5 MB**(high) / 20.3 MB(low).
- **⚠️ op 회귀:** INT8 후 실질 fallback **1 → 9**(**Transpose ×8** 신규 CPU). attention의 reshape/permute가
  양자화로 fusion이 깨져 CPU로 낙하. → **Phase 3에서 hybrid quantization**(민감 레이어 int16 유지)으로 대응.
  단순 PTQ INT8은 크기는 줄지만 op 배치가 나빠짐을 실측 확인.

## NPU attention 최종 결정: softmax (실데이터 학습에서 확정)
Div 제거는 const-norm linear로 달성했으나, **실 TransCG 학습에서 BatchNorm+relu-linear가 fp16(AMP) NaN 발산**함을 발견.
토글 격리(norm→LN 또는 attn→softmax 시 안정)로 원인 확인. → **npu_profile attention을 softmax로 전환**:
softmax는 (i) Div 없음, (ii) RKNN 완전 변환(NPU 80 / fallback 1 = DepthToSpace), (iii) fp16 학습 안정, (iv) 정확도 우월.
const-norm은 `attn_norm="const"` 옵션으로 보존. 안정화 추가: 마스크 기반 point-map 정규화, fp32 손실, grad clip.
**실 TransCG 서브셋(scene1-10, 2785샘플) 학습 검증: loss 2.77 → 0.94 안정 수렴.**

## 정확도 경로 인코더/데이터 (완료)
- **DINOv2-small**(ungated) 실연결·검증(patch14 리사이즈, high 5.2ms/192FPS). **DINOv3**는 gated(HF 로그인).
- **TransCG 로더**(공식 포맷 충실) + **DREDS 로더**(EXR 미터 depth, sim/real intrinsics, OpenEXR 백엔드) 구현·검증.

## 남은 과제(Phase 3, 보드 필요)
- **hybrid quantization**: INT8 Transpose fallback 8개를 int16 승격으로 NPU 복귀 시도.
- **온디바이스 실측**: `RKNNLite(core_mask=NPU_CORE_0)`로 ms/frame → ≥10 FPS 게이트. **A(const-linear) vs B(softmax) 실측 비교.**
- **실학습**: TransCG/DREDS 다운로드 → sim-to-real 사전학습 → fine-tune.

## 재현 명령
```bash
# rknn venv (1회)
uv venv .venv-rknn --python 3.12
uv pip install --python .venv-rknn/bin/python "rknn-toolkit2==2.3.2" "onnx==1.16.1" "numpy<=1.26.4" "setuptools<80" onnxsim onnxruntime
# 변환 리허설
./.venv/bin/python scripts/export_onnx.py --resolution high --npu --fold-t --rehearsal --out artifacts/rehearsal_high.onnx
.venv-rknn/bin/python -m onnxsim artifacts/rehearsal_high.onnx artifacts/rehearsal_high_sim.onnx
.venv-rknn/bin/python scripts/rknn_convert.py --onnx artifacts/rehearsal_high_sim.onnx --resolution high --out artifacts/rehearsal_high.rknn --log .omc/logs/rehearsal.log
```
