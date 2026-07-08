# 방법론 리포트 — 투명물체 Depth PointDiT · 임의 소해상도 · RK3588 2-TOPS 실시간

> Autoresearch mission `pointdit-rk3588` / Phase 1 산출물. 대상: [PointDiT (arXiv:2607.02515)](https://arxiv.org/abs/2607.02515), [project](https://haofeixu.github.io/pointdit/).
> 목표: 투명·경면 물체에서도 잘 동작하는 dense depth/point map을 PointDiT로 구현하고, RK3588 NPU 1코어(~2 TOPS)에서 320×240 기준 **≥10 FPS** 실시간 구동. 정확도 허용치는 loose(task-usable).
> 하드웨어(RK3588 보드) 미보유 → 최적화 수치는 FLOP 모델·공개 벤치 유추 기반 **추정**.

## 요약 (Executive Summary)

- **재현(R1):** frozen **DINOv3 ViT** 인코더 + **DiT 계열 point-map diffusion denoiser**(flow-matching, one-step). 공식 코드 미공개 → Meta DINOv3 + facebookresearch/DiT + MoGe식 affine-invariant point-map 감독으로 밑바닥 재현.
- **투명물체(R2):** 문제 구조가 "photometric cue 부재 → multimodal ambiguity → generative completion"이라 diffusion prior가 본질적으로 유리. Primary 벤치 = **TransCG**, sim-to-real = **DREDS→STD**.
- **해상도(R3):** DINOv3의 **2D-RoPE** 덕에 인코더는 임의 해상도 native. 배포는 **320×240/160×120 고정 2-shape**.
- **최적화(R4):** 1-step diffusion 필수. 실효 예산 ~30–50 GMAC/frame. 최대 리스크는 **RKNN에서 attention/LayerNorm의 CPU fallback** → BatchNorm/ReLU-attention 재설계로 대응. 권장 경로: **DINOv3-S 재설계본 + 1-step 증류 denoiser + INT8 320×240 고정** → 10 FPS 달성권.

---

## R1 — PointDiT 재현 설계

### R1.1 대상 논문 개요

**PointDiT** ([arXiv 2607.02515](https://arxiv.org/abs/2607.02515), [project page](https://haofeixu.github.io/pointdit/), Haofei Xu — GMFlow/UniMatch/DepthSplat/MVSplat 저자)는 단일 RGB 이미지에서 dense 3D **point map**(픽셀당 XYZ)과 depth를 추정하는 monocular geometry 모델이다. 핵심 설계는 (1) VAE latent가 아니라 **raw point map 위에서 직접 diffusion을 수행**하고, (2) plain ViT denoiser가 point-map patch를 "RGB 대신 XYZ를 채널로 갖는 이미지"처럼 patchify하여 denoise하며, (3) **frozen DINOv3**가 입력 이미지를 encode해 조건으로 주입하고, (4) **one-step generation**이 가능하다는 점이다. project page는 formulation이 deterministic regression이 아닌 **flow matching**임을 controlled comparison으로 명시한다.

### R1.2 아키텍처 스펙 재구성 (근거 기반 추정)

- **이미지 인코더 (frozen):** Meta DINOv3 계열. DINOv3는 patch size 16의 ViT 제품군을 제공하며, ViT-B/16은 embed dim 768·12 heads·86M params, ViT-L/16, ViT-7B/16(embed 4096, RoPE, SwiGLU)까지 공개되어 있다([facebookresearch/dinov3 MODEL_CARD](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md), [HF facebook/dinov3-vitl16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)). 재현 시 **ViT-L/16**을 기본, 경량 실험은 ViT-B/16을 권장한다. 입력 448²이면 patch16 기준 28×28=784 image token.
- **Point-map patchify:** point map을 patch 16으로 자르고 각 패치의 XYZ(3ch)를 linear projection해 token 화한다. patch 16×16×3=768 차원 벡터 → denoiser hidden dim으로 linear embed. project page 인용: *"patchified into tokens, exactly like an image but with XYZ coordinates as the channels instead of RGB."*
- **Denoiser ViT:** DiT 계열 backbone이 자연스러운 참조점이다. [DiT](https://arxiv.org/abs/2212.09748) ([facebookresearch/DiT](https://github.com/facebookresearch/DiT))의 DiT-XL/2는 depth 28·hidden 1152·16 heads·675M params이며, DiT-L(depth 24·hidden 1024·16 heads)·DiT-B(depth 12·hidden 768·12 heads) config도 표준 제공된다. 재현 denoiser는 **DiT-L/DiT-B급 depth 12–24, hidden 768–1024, 12–16 heads**로 시작하는 것이 합리적이다. timestep은 AdaLN 또는 timestep-token으로 주입.
- **이미지 조건 주입:** project page 문구는 별도 cross-attention보다 **noisy point token + image token을 concat해 공동 self-attention(in-context conditioning)** 하는 방식을 시사한다. DINOv3 token은 frozen이므로 gradient가 흐르지 않고 key/value context 역할만 한다.
- **Diffusion / one-step:** flow matching(velocity 예측; cf. [Lipman et al. 2210.02747](https://arxiv.org/abs/2210.02747))을 학습 objective로 두고, one-step은 (a) 직선적 flow(straight-path)로 few/one-step ODE 적분, 또는 (b) consistency/distillation([Consistency Models 2303.01469](https://arxiv.org/abs/2303.01469)) 중 하나로 구현했을 가능성이 높다. Xu의 [GMFlow](https://arxiv.org/abs/2111.13680) 계보상 flow 기반 one-step이 유력하다.

### R1.3 학습 objective·좌표계·카메라 가정

point map은 **DUSt3R 계열의 픽셀-정렬 XYZ 표현**을 따르되([DUSt3R 2312.14132](https://arxiv.org/abs/2312.14132)), monocular focal 모호성을 피하기 위해 **scale-/affine-invariant 정규화**를 채택할 것으로 본다. [MoGe](https://arxiv.org/abs/2410.19115)는 affine-invariant point map + ROE global alignment solver로 scale·shift 모호성을 해소해 scale-invariant Rel 8.32%(vs DUSt3R 14.7%)를 달성했고, affine-invariant에서 6.43%까지 개선했다. PointDiT의 diffusion loss는 정규화 좌표계에서의 velocity/point regression으로, alignment(scale·shift)는 loss 내부에서 closed-form으로 풀어 적용하는 것이 정합적이다. 카메라는 unknown intrinsics를 가정(focal은 point map에서 유도), many-view로의 확장은 [VGGT 2503.11651](https://arxiv.org/abs/2503.11651)(CVPR 2025 Best Paper)가 선례다.

### R1.4 재현 계획 (공개 자산 매핑)

1. **인코더:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) ViT-L/16 frozen, dense patch feature 추출.
2. **Denoiser:** [facebookresearch/DiT](https://github.com/facebookresearch/DiT) DiT-B/L를 fork, patch-embed을 3ch(XYZ)로 교체, RGB token concat conditioning 추가, VAE 제거(raw point map).
3. **감독:** DUSt3R/[MoGe](https://arxiv.org/abs/2410.19115)/VGGT/[Depth Anything V2 2406.09414](https://arxiv.org/abs/2406.09414) 데이터·pseudo-GT 파이프라인 재사용, affine-invariant point-map loss.
4. **학습:** flow matching(rectified-flow) 사전학습 → consistency/reflow distillation으로 one-step.

### R1.5 재현 RISK·미확정 사항

- **미공개 하이퍼파라미터:** denoiser depth/width/heads, token 결합의 정확한 형태(concat vs cross-attn), noise schedule이 project page에 없음 — ablation으로 탐색 필요.
- **좌표 정규화 정의:** scale-only vs affine(scale+shift), per-scene vs per-batch 여부 불명확 → MoGe alignment를 기본값으로 가정.
- **One-step 메커니즘:** distillation인지 native straight-flow인지 불확정 — 품질/속도 trade-off의 핵심 위험.
- **데이터 스케일·라이선스:** 대규모 metric/relative 3D GT 확보가 재현의 최대 병목.
- **평가 정합성:** 논문 벤치마크 세팅이 확정 공개 전까지 절대 수치 비교는 유보.

---

## R2 — 투명 물체 depth

투명·경면(specular) 물체는 굴절·투과·정반사로 인해 stereo/ToF/structured-light 센서의 photometric 가정이 깨지므로, RGB-D 원본 depth에 대규모 hole·outlier가 발생한다. 본 절은 PointDiT(point map을 직접 denoise하는 diffusion-transformer, frozen DINOv3 encoder, [arXiv:2607.02515](https://arxiv.org/abs/2607.02515))를 이 문제에 적용하기 위한 벤치마크·지표·prior 근거·학습 전략을 정리한다.

### R2.1 벤치마크 & 데이터셋 서베이

| 데이터셋 | Real/Synth | 범위 | RGB-D | 규모 | 비고 |
|---|---|---|---|---|---|
| [ClearGrasp](https://arxiv.org/abs/1910.02550) | Synth+Real test | tabletop | O | 50K+ synth / 286 real | 투명 depth의 사실상 표준 real test set |
| [TransCG](https://arxiv.org/abs/2202.08471) | Real | tabletop, cluttered | O | 57,715 프레임 / 130 scenes | 실사 depth+normal+mask, depth completion 표준 |
| [ClearPose](https://arxiv.org/abs/2203.03890) | Real | tabletop, occlusion/liquid | O | 350K+ 프레임 / 63 objects | 대규모, pose+depth+segmentation |
| [DREDS / STD](https://arxiv.org/abs/2208.03792) | Synth+Real | tabletop | O | 130K synth / 27K real | domain-randomized 시뮬 + 실측, sim-to-real 핵심 |
| [Booster](https://arxiv.org/abs/2301.08245) | Real | general, high-res | stereo | 606 samples / 85 scenes | 12Mpx, material mask, non-tabletop 일반 장면 |
| [Trans10K-v2](https://arxiv.org/abs/2101.08461) | Real | general, in-the-wild | segmentation | 10K+ 이미지 | depth 아닌 mask 소스(평가 마스크 생성) |

**권장:** 1차 primary는 **TransCG**(대규모 real RGB-D + 투명 mask + depth-completion 프로토콜, fine-tuning·평가 모두 최적). Fallback으로 (a) zero-shot 일반화 검증용 **ClearGrasp real test set**, (b) occlusion·liquid 스트레스용 **ClearPose**, (c) sim-to-real 소스+실측 검증셋 **DREDS(synth)→STD(real)**. tabletop을 넘어선 일반 장면 robustness는 **Booster**로 보강.

### R2.2 지표 & 프로토콜

투명 depth 평가는 **transparent-region mask** 안에서 계산하는 것이 핵심이다. ClearGrasp 관례상 **ToM(투명 영역) / All(전체) / Other(비투명)** 3영역으로 분리 리포트한다. 주요 지표는 **RMSE**, **MAE**, **AbsRel(REL)**, 임계 정확도 **δ<1.25**(및 엄격한 δ<1.05/1.10)로, δ는 max(pred/gt, gt/pred)<threshold인 픽셀 비율이다([TransCG](https://arxiv.org/abs/2202.08471)). PointDiT는 point map 출력이므로 back-projection된 depth로 위 지표를 계산하되, point-cloud 기반 Chamfer/F-score를 3D geometry 보조 지표로 병기한다.

### R2.3 왜 generative(diffusion) prior가 유리한가

투명 표면에는 물체 자체의 photometric cue가 결여되고 배경 굴절/정반사만 관측되므로, 하나의 RGB로부터의 depth 매핑이 본질적으로 **다봉(multimodal)·ambiguous** 하다. Discriminative regressor는 L1/L2 최소화로 이 다봉 분포의 **평균으로 회귀**해 표면을 blur시키거나 배경으로 depth가 새는 실패를 보인다([DepthFM 2403.13788](https://arxiv.org/abs/2403.13788)). 반면 diffusion/flow prior는 조건부 분포 p(depth|image)에서 **표본을 생성(completion)** 하여, 직접 관측이 없는 표면에도 학습된 형상 prior로 그럴듯한 geometry를 채운다. 이는 [Marigold 2312.02145](https://arxiv.org/abs/2312.02145)와 DepthFM의 강한 generalization의 연장이며, 투명 특화 generative 접근이 우위를 입증한다: 좌우 consistency를 guidance로 쓴 **[D3RoMa 2409.14365](https://arxiv.org/abs/2409.14365)**, 투명→불투명 변환 후 표준 estimator에 입력하는 **[SeeClear 2603.19547](https://arxiv.org/abs/2603.19547)**, diffusion depth completion([DidSee 2506.21034](https://arxiv.org/abs/2506.21034)). PointDiT는 2D depth가 아닌 **point map을 직접 denoise**하므로 굴절 왜곡 영역의 3D 구조를 명시적으로 생성하는 데 부합한다.

### R2.4 학습 데이터 전략

**Framing:** (a) *depth-from-RGB*(RGB만으로 point map 생성), (b) *depth-completion*(센서 depth 유효 영역을 조건으로 hole 복원). 로보틱스에서는 유효 depth를 conditioning으로 주입하는 **completion framing**이 metric-scale 안정성에서 유리하며([Marigold-DC 2412.13389](https://arxiv.org/abs/2412.13389)), 센서 완전 실패 극단은 RGB-only fallback. **Sim-to-real:** [DREDS 2208.03792](https://arxiv.org/abs/2208.03792)의 PBR + 재질/조명 무작위화로 130K 합성쌍 사전학습 → TransCG/ClearPose 실측 fine-tuning. DINOv3 frozen이므로 diffusion head만 적응시켜 적은 real 라벨로도 도메인 이전 가능.

---

## R3 — 해상도 지원

PointDiT는 frozen DINOv3 encoder와 point-map을 처리하는 diffusion ViT로 구성되므로, 해상도를 바꾸면 두 ViT 계열 모듈의 **positional embedding(PE)** 처리가 핵심 문제가 된다. 표준 ViT는 학습 시 고정 grid에 대한 learnable absolute PE를 쓰기 때문에, 입력 해상도가 바뀌어 token grid가 달라지면 PE grid를 2D **bicubic interpolation** 해 재배열해야 한다. 이 기법은 원조 ViT([2010.11929](https://arxiv.org/abs/2010.11929))에서 도입되어 DeiT([2012.12877](https://arxiv.org/abs/2012.12877)), MAE([2111.06377](https://arxiv.org/abs/2111.06377)), DINOv2([2304.07193](https://arxiv.org/abs/2304.07193))의 `interpolate_pos_encoding()`에서 표준 관행이 되었다.

DINOv3([2508.10104](https://arxiv.org/abs/2508.10104))는 이 문제를 근본적으로 바꿔 **learnable absolute PE 대신 axial 2D-RoPE**를 채택한다. RoPE는 위치를 절대 좌표가 아니라 feature 공간의 rotation(상대 위치)으로 인코딩하므로, token 수가 늘거나 aspect ratio가 바뀌어도 interpolation 없이 자연스럽게 일반화된다. DINOv3는 학습 중 **RoPE-box jittering**으로 scale/resolution/aspect-ratio robustness를 강화한다. RoPE-ViT([2403.13298](https://arxiv.org/abs/2403.13298))도 vision에서의 RoPE 외삽 효과를 뒷받침한다. 즉 DINOv3 encoder 자체는 임의 해상도에 강하지만, PointDiT의 **point-map diffusion ViT branch가 absolute PE를 쓴다면** 여기에는 bicubic 보간 또는 RoPE 적용이 별도로 필요하다.

**Token-grid 계산 (patch=16, image token = point token 동일 grid 가정):**

| 해상도 (W×H) | patch=16 grid | token 수 | 비고 |
|---|---|---|---|
| 320×240 | 20×15 | **300** | 320·240 모두 16 배수 → 정합 |
| 160×120 | 10×**7.5** (비정수) | 80(→128 pad) / 70(→112 resize) | 120/16=7.5 → pad/resize 필요 |

핵심 제약: **입력은 patch 배수여야 한다.** patch=16에서 320·240은 배수이나 120은 아니므로(120/16=7.5) 120→128 zero-pad(→10×8=80 token) 또는 120→112 resize가 필요하다. DINOv2/v3 native patch=14는 320·240·160·120 모두 비정수라 crop/pad가 더 빈번하다. 따라서 배포 시 patch=16과 16-배수 정렬(예: 160×128)을 쓰면 grid가 깔끔하다.

**Accuracy vs token 수 tradeoff:** token 수는 대략 해상도의 제곱에 비례한다(300→80, 약 3.75× 감소). token이 적을수록 self-attention 연산량은 크게 줄지만(NPU 친화적) spatial 세밀도가 손실된다. 특히 **transparent-object의 얇은 edge·굴절 경계** 같은 fine geometry는 160×120에서 patch당 blur가 커져 point-map 정확도가 떨어진다. 실무상 320×240을 고정밀 모드, 160×120을 저지연 모드로 이원화한다.

**Multi-resolution TRAINING:** 하나의 weight set으로 두 배포 해상도를 커버하려면 학습 시 해상도를 섞는다. (a) multi-scale training(매 batch 320×240/160×120 랜덤 + PE 보간), (b) **NaViT식 native-resolution packing**([2307.06304](https://arxiv.org/abs/2307.06304)), (c) **FlexiViT**([2212.08013](https://arxiv.org/abs/2212.08013))식 variable patch size. 학습·구현 레벨에서 arbitrary-size 지원을 유지하되, **RK3588 NPU 배포는 320×240·160×120 두 fixed shape만** 컴파일한다: NPU 그래프는 정적 shape에서 최적 성능을 내고, 두 shape 사전 컴파일로 dynamic-shape 재컴파일·비지원 op 문제를 피하면서 정밀/저지연 모드 전환을 확보한다.

---

## R4 — RK3588 2-TOPS 최적화

본 절은 PointDiT([arXiv:2607.02515](https://arxiv.org/abs/2607.02515), frozen DINOv3 encoder + point-map diffusion ViT)를 Rockchip RK3588의 **NPU 코어 1개(~2 TOPS INT8)** 로 320×240 입력에서 **≥10 FPS(≤100 ms/frame)** 실시간 구동하기 위한 최적화 설계를 다룬다. 보드 미보유 상태이므로 모든 수치는 FLOP 기반 모델과 공개 벤치마크 유추에 근거한 **추정치**이며, 정확도 허용 범위는 loose(task-usable)로 가정한다.

### R4.1 컴퓨트 예산 모델 (per-frame MAC budget)

2 TOPS INT8 = 2×10¹² OP/s이다. 10 FPS 목표는 프레임당 **200 GOP ≈ 100 GMAC**의 *이론적* 상한을 의미한다. 그러나 실측 NPU 활용률은 GEMM/attention 워크로드에서 통상 30~50%에 그치므로, **실효 예산은 ~30–50 GMAC/frame**로 봐야 한다(RK3588 6-TOPS 3코어에서 ResNet18가 244 FPS/4.09 ms인 반면 transformer는 "추가 최적화 필요"로 보고 — [TinyComputers RK3588 NPU](https://tinycomputers.io/posts/rockchip-rk3588-npu-benchmarks.html), [IEEKER RK3588 6 TOPS](https://ieeker.com/rk3588-npu-performance-industrial-edge-ai/)).

DINOv3 인코더 크기([2508.10104](https://arxiv.org/html/2508.10104v1)): ViT-S/16 = 21M, ViT-B/16 = 86M. 224² 비용(DeiT-S/16 ≈ 4.6 GFLOPs, ViT-B/16 ≈ 17.6 GFLOPs)을 320×240(300 tokens, ~1.52×)로 스케일하면:

| 구성요소 (320×240, INT8) | MAC/frame | 비고 |
|---|---|---|
| DINOv3-**S**/16 encoder (frozen) | ~3.5 GMAC | 4.6 GFLOPs×1.52 |
| DINOv3-**B**/16 encoder (frozen) | ~13 GMAC | 예산 타이트, 활용률 하락 시 위험 |
| Denoiser ViT (6L, d=256, 300 tok) | ~1.5 GMAC / step | point-map diffusion head |
| Overhead (patch embed, upsample, I/O) | ~0.5–1 GMAC | — |
| **1-step total (S + 1 denoiser)** | **~5 GMAC** | 실효 예산 내 여유 |
| **10-step total (S + 10 denoiser)** | **~18.5–35 GMAC** | encoder 재계산 시 급증 |

**1-step diffusion이 필수인 이유**: multi-step 샘플링은 denoiser를 step 수만큼 반복하고, 조건부 인코딩을 매 step 재계산하면 encoder 비용까지 곱해진다. 10-step은 실효 예산(30–50 GMAC)을 잠식해 10 FPS를 불가능하게 만든다. 따라서 consistency/shortcut-style **single-step 증류**가 강제 요건이다.

### R4.2 DINOv3 경량화 / 대체

(a) **증류**: DINOv3는 ViT-7B teacher를 ViT-S/B/L 및 ConvNeXt로 multi-student distillation한다([2508.10104](https://arxiv.org/html/2508.10104v1)) → ViT-S(21M, ~3.5 GMAC)를 기본 인코더로 채택. (b) **토큰 감소**: ToMe(Token Merging)는 재학습 없이 유사 토큰을 병합해 **throughput 2×를 정확도 0.2–0.3% 손실**로 달성([2210.09461](https://arxiv.org/abs/2210.09461)) → 300 tokens를 ~150으로. (c) **CNN/hybrid 대체**: EfficientViT는 softmax attention을 ReLU **linear attention**으로 대체해 2차→1차 복잡도 달성([2205.14756](https://arxiv.org/html/2205.14756v6)) — RK3588 NPU conv 친화성에 결정적. loose 정확도 하에서 투명체 depth 저하는 수 % 수준 감내 가능.

### R4.3 양자화 (RKNN INT8)

RKNN-Toolkit2는 **INT8 PTQ**(calibration 기반), **hybrid quantization**(민감 레이어만 int16/fp), mixed precision을 지원한다([Ultralytics RKNN](https://docs.ultralytics.com/integrations/rockchip-rknn), [Edge AI with Rockchip NPU](https://tristanpenman.com/blog/posts/2025/07/20/edge-ai-using-the-rockchip-npu/)). Transformer는 LayerNorm/softmax의 넓은 동적 범위로 순수 INT8 PTQ에서 정확도 붕괴가 흔하다. 권고: **혼합정밀도**(attention logit/LayerNorm → int16, GEMM → int8) + 100~500장 in-domain calibration + RKNN `quantization accuracy analysis`로 레이어별 cosine 유사도 점검. QAT는 지원이 약해 PTQ 실패 레이어만 선택적 int16 승격으로 대응.

### R4.4 RKNN 연산자 지원 매트릭스 — **최대 실전 리스크**

이론 TOPS는 attention이 CPU로 fallback하면 무의미하다. RK3588 NPU는 **MatMul 등이 단일 코어에서만** 실행되고 conv 계열만 멀티코어 지원하며, 정렬/크기 제약 미충족 레이어는 자동 **CPU fallback**된다([Rockchip NPU 분석](https://clehaxze.tw/gemlog/2023/07-13-rockchip-npus-and-deploying-scikit-learn-models-on-them.gmi)). RKNN-Toolkit2는 LayerNorm/Softmax/GELU/MatMul을 지원하지만([airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)), 실전에선 transformer가 통째로 잘 안 얹힌다:

- **RF-DETR on RK3588**: "Full RF-DETR ONNX models are transformer-heavy and are **not a good fit for end-to-end RKNN**." → backbone만 NPU, transformer head는 CPU 분할. RFDETR-Nano 329→198 ms(1.66×)에 그침([rfdetr-on-rockchip-npu](https://github.com/AlexanderDhoore/rfdetr-on-rockchip-npu)).
- **ViTTracker ONNX→RKNN(2.3.2)**: 변환 성공하나 operator fusion 후 **추론 출력이 틀림**([airockchip issue #460](https://github.com/airockchip/rknn-toolkit2/issues/460)) — 변환 성공 ≠ 정확도 보존.

결론: DINOv3 attention/LayerNorm이 CPU fallback되면 100 ms 예산 초과가 확실하다. 이 리스크가 §R4.5 재설계를 정당화한다.

### R4.5 NPU-friendly 재설계

- **LayerNorm → BatchNorm**(추론 시 conv에 fold).
- **GELU → hardswish/ReLU**(RKNN 기본 지원, fusion 안정).
- **softmax attention → linear/ReLU attention**(EfficientViT식) 또는 conv-attention hybrid로 2차항·softmax fallback 제거.
- **고정 shape**: dynamic shape는 fallback 유발 → 정적 300-token 그래프 컴파일.
- Denoiser는 self-attention 대신 depthwise-conv mixer 대체 시 NPU 멀티코어 conv 경로 활용.

### R4.6 고정 shape 컴파일 & 지연/메모리 추정

두 프로파일 고정 컴파일: **320×240(300 tok)** 및 **160×120(patch16 → 10×8 = 80 tok)**. FLOP 추정(실효 ~800 GMAC/s @활용률 40%):

- **320×240, 1-step, DINOv3-S 재설계본** ~5 GMAC → NPU-only 시 ~30–60 ms, **10–20 FPS 달성권**. 활성 메모리 ~50–120 MB(INT8 가중치 21 MB + activation).
- **160×120** ~1.5 GMAC → ~15–25 ms, 40–60 FPS 여유. 저지연 fallback 프로파일.
- **주요 병목**: (1) attention/LayerNorm CPU fallback(치명적), (2) 단일코어 MatMul 제약, (3) INT8 activation 대역폭, (4) NPU↔CPU 텐서 복사.

**보드-부재 추정 방법론**: ① FLOP/실효-TOPS 상한 계산, ② RKNN-Toolkit2 시뮬레이터의 per-layer target/fallback 리포트로 CPU 낙하 레이어 조기 식별, ③ 공개 ViT-on-RK3588 벤치(ResNet18 244 FPS, YOLOv5s 54 FPS, RF-DETR 분할 1.6×)로 하한 보정. 최종 권고: **DINOv3-S + BatchNorm/ReLU-attention 재설계 + 1-step 증류 denoiser + 320×240 고정 INT8** 조합이 10 FPS 최유력 경로이며, attention이 NPU에 얹히지 못할 경우 160×120 fallback 또는 EfficientViT-style conv-encoder 전면 교체가 백업이다.

---

## R5 — 통합 로드맵 & Pareto

### R5.1 단계별 실행 계획 (1 → 2 → 3)

| Phase | 산출물 | 평가자 | 마일스톤 |
|---|---|---|---|
| **1 (현재)** 방법론 | 본 리포트 | 루브릭 커버리지 체크(≥0.85, R2·R4 필수) | 커버리지 통과 |
| **2** PC/GPU 재현 | 학습 코드 + 320×240/160×120 추론 + 1-step 증류 | 투명 벤치([TransCG](https://arxiv.org/abs/2202.08471)) AbsRel/δ1 + PC latency(ms) | [MoGe 2410.19115](https://arxiv.org/abs/2410.19115) 근접 정확도 & one-step 동작 |
| **3** RK3588 배포 | RKNN INT8 모델(2 shape) | 온디바이스 latency/memory/accuracy 실측 | **≥10 FPS @ 320×240, 1코어** |

Phase 2→3 사이에 RKNN 시뮬레이터로 fallback 레이어를 선제 점검해, 보드 도착 전에 R4.5 재설계를 코드에 반영한다.

### R5.2 속도-정확도 Pareto 후보

| 후보 | 인코더 | steps | 양자화 | 해상도 | 추정 지연 | 정확도 tier |
|---|---|---|---|---|---|---|
| A (정밀) | DINOv3-S | 4 | int8/int16 mix | 320×240 | ~120–200 ms | 최고 |
| **B (권장)** | **DINOv3-S 재설계** | **1** | **int8 (LN/logit int16)** | **320×240** | **~30–60 ms** | 높음 |
| C (저지연) | EfficientViT-conv | 1 | int8 | 160×120 | ~15–25 ms | 중간 |
| D (극한) | conv-encoder + ToMe | 1 | int8 | 160×120 | ~10–18 ms | 낮음(용도충족) |

권장 지점은 **B**(10 FPS 목표 + 품질 균형). B가 NPU fallback으로 미달 시 **C/D**로 하강, 여유 시 **A**로 정밀도 상승.

### R5.3 핵심 리스크 & 완화

- **10 FPS 미달** → 160×120 fallback(C/D), conv-encoder 전면 교체, [ToMe 2210.09461](https://arxiv.org/abs/2210.09461) 토큰 감소.
- **투명물체 일반화 실패** → [DREDS 2208.03792](https://arxiv.org/abs/2208.03792) sim-to-real 사전학습 + TransCG fine-tune, Booster OOD 검증.
- **RKNN 미지원 연산(attention/LayerNorm CPU fallback)** → BatchNorm/ReLU-attention 재설계, backbone(NPU)/head(CPU) 분할, 시뮬레이터 선제 점검.
- **공식 코드/데이터 부재** → MoGe/VGGT pseudo-GT 파이프라인, 재현 ablation으로 미공개 하이퍼파라미터 탐색.

---

*Generated by autoresearch mission `pointdit-rk3588` (Phase 1). Spec: `.omc/specs/deep-interview-pointdit-rk3588.md`.*
