# Deep Interview Spec: PointDiT — 투명 물체 Depth, 임의 소해상도, RK3588 2-TOPS 실시간

## Metadata
- Interview ID: pointdit-rk3588-2026-07-04
- Rounds: 6 (+ Round 0 토폴로지 게이트)
- Final Ambiguity Score: 11.3%
- Type: greenfield
- Generated: 2026-07-04
- Threshold: 0.20
- Threshold Source: default
- Initial Context Summarized: no
- Mode: `--autoresearch`
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.92 | 0.40 | 0.368 |
| Constraint Clarity | 0.88 | 0.30 | 0.264 |
| Success Criteria | 0.85 | 0.30 | 0.255 |
| **Total Clarity** | | | **0.887** |
| **Ambiguity** | | | **0.113** |

## Topology
프로그램 전체 목표는 **온디바이스(RK3588) 실시간 투명물체 depth**이며, 순차 진행(1 → 2 → 3). 이번 autoresearch 미션 #1은 **1단계(방법론 리포트)**를 실행 가능한 루프로 세팅한다. 2·3단계는 로드맵으로 스펙에 고정.

| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| #1 PointDiT 재현 | active (미션 #1 핵심) | 논문 기반 밑바닥 구현 + 공개 자산(Meta DINOv3, DiT 계열) | 리포트 루브릭 R1에서 재현 설계 커버 |
| #2 해상도 유연성 | active | 320×240, 160×120 (임의는 구현 레벨 pos-embed 보간, 배포는 고정 2세트) | 루브릭 R3 |
| #3 RK3588 2TOPS 최적화 | active (프로그램 최종 목표) | 1코어 ~2TOPS, 320×240 @ ≥10 FPS 실시간 | 루브릭 R4. 보드 미보유 → 미션 #1에선 **추정 평가**(FLOP/연산지원/양자화 시뮬), 보드 확보 시 실측 승격 |

## Goal
투명·반사 물체에서도 잘 동작하는 dense depth / 3D 포인트맵을 **PointDiT**(freeze DINOv3 + raw 포인트맵 diffusion ViT, 1-step 생성)로 구현하고, 이를 **RK3588 NPU 1코어(~2 TOPS)에서 320×240 기준 실시간(≥10 FPS)**으로 구동하기 위한 방법론을 확립한다. 최종 산출물은 1→2→3 단계로 진행하는 프로그램이며, 미션 #1은 이를 뒷받침하는 근거 기반 방법론 리포트다.

**핵심 동기:** 스테레오/ToF/구조광 센서가 실패하는 **투명 물체 depth**를, diffusion prior로 형상을 그럴듯하게 생성하는 PointDiT로 해결.

## Constraints
- 보유 자산: **학습/재현용 GPU만**. RK3588 보드·RKNN 툴체인 미보유. 공식 PointDiT 코드/가중치 미보유. 평가 데이터셋 미보유.
- 하드웨어 타깃: RK3588 NPU **1코어 ~2 TOPS** (3코어 합산 6 TOPS 아님).
- 속도 목표: **320×240에서 ≥10 FPS** (≤100 ms/frame) → **1-step diffusion 사실상 강제**.
- 정확도: **느슨(용도 충족)** — 공격적 양자화/모델 축소 허용. 투명 물체에 "쓸만한" depth면 성공.
- 해상도: 배포는 **고정 2세트(320×240, 160×120)**. 임의 사이즈는 구현/학습 레벨에서 pos-embed 보간으로 지원하되 NPU 배포 타깃 아님.
- 코드 부재 → 재현은 논문 기반 + 공개 자산 활용.
- 보드 부재 → 미션 #1 평가자는 실측이 아닌 **추정**(연산 예산 모델, RKNN 연산 지원 매트릭스, 양자화 시뮬레이션, 유사 모델 문헌 벤치).

## Non-Goals
- 런타임 동적 해상도(단일 모델 가변 입력), 임의 종횡비/레터박스 처리 — 범위 외(고정 2세트로 충분).
- 논문 수준 SOTA 정확도 달성 — 비목표(용도 충족이 기준).
- 다중 NPU 코어(6 TOPS) 활용 — 비목표(1코어 제약).
- 미션 #1에서 실제 온디바이스 배포/실측 — 2·3단계로 이연(보드 확보 후).
- PointDiT 외 대체 경량 depth 모델 채택 — 비목표(투명물체엔 diffusion prior가 본질).

## Acceptance Criteria (미션 #1 = 방법론 리포트)
리포트가 아래 루브릭을 커버(각 항목 근거/인용 포함)하면 통과. **평가자 = 루브릭 커버리지 체크**(결정적 앵커·인용 수 검사 + 선택적 LLM-judge). 임계: 전체 커버리지 ≥ 0.85 **AND** 크리티컬 섹션(R2, R4) 누락 0.

- [ ] **R1 PointDiT 재현 설계**: DINOv3 변형·denoiser ViT 하이퍼파라미터·포인트맵 XYZ 토큰화·조건부 주입·diffusion 스케줄·1-step 메커니즘·손실/좌표계, 공개 코드 부재 대응(자산 매핑·리스크)
- [ ] **R2 투명 물체 depth (크리티컬)**: 벤치마크 조사·선정(TransCG/ClearGrasp/DREDS/ClearPose/Booster), 지표(AbsRel/δ1/RMSE)·프로토콜, diffusion prior가 투명 표면에 유리한 근거, 학습 데이터 전략
- [ ] **R3 해상도 지원**: ViT pos-embed 2D 보간·DINOv3 고정격자 대응, 320×240·160×120 토큰 그리드 산출, 저해상도 정확도-토큰수 트레이드오프, multi-res 학습
- [ ] **R4 RK3588 2TOPS 최적화 (크리티컬)**: 프레임당 MAC 예산 모델(320×240@10FPS), DINOv3 경량화/대체(증류·CNN 대체) 및 정확도 영향, INT8/혼합정밀 양자화(PTQ/QAT)·캘리브레이션, **RKNN 연산 지원 매트릭스**(attention/LayerNorm/GELU/softmax CPU fallback 리스크·NPU 친화 재설계), 고정 shape 컴파일·추정 지연/메모리, 보드 부재 시 추정 방법론
- [ ] **R5 통합 로드맵 & Pareto**: 1→2→3 단계별 산출물/평가자/마일스톤, 속도-정확도 Pareto 후보(모델크기×양자화×스텝수)·권장 지점, 핵심 리스크·완화

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| autoresearch로 1→2→3 한 번에 | 단일 미션+고정 평가자 루프에 3단계는 평가자가 바뀜 | 미션 #1 = 1단계(리포트)만 지금 세팅, 2·3은 로드맵 |
| 공식 코드로 재현 | 2026-07 신규 arXiv, 공개 정황 없음 | 논문 기반 밑바닥 구현 + Meta DINOv3/DiT 계열 활용 |
| "빠르게" 최적화 | 측정 불가 모호어 | ≥10 FPS @ 320×240, 1코어 2TOPS로 수치화 |
| 실시간 + 논문 정확도 동시 (Contrarian) | 상충 가능 | 정확도 느슨(용도 충족) 채택 → 실시간 우선 |
| PointDiT가 요구사항? (Contrarian) | 수단일 수도 | **투명물체 depth가 본질** → PointDiT의 포인트맵/diffusion prior 필수, 경량 depth 대안 제외 |
| "임의 해상도" 동적 필요 (Simplifier) | NPU 동적 shape 고비용 | 고정 2세트로 충분, 동적/종횡비 비목표 |
| 리포트 완료 판정 (Simplifier) | 자동화 필요 | 루브릭 커버리지 체크(결정적 + LLM-judge) |

## Technical Context
- **PointDiT (arXiv 2607.02515):** monocular geometry estimation. 평범한 ViT가 포인트맵 패치(XYZ=채널)를 디노이징, VAE 없이 raw 포인트맵에서 직접 diffusion. **frozen DINOv3** 이미지 인코더 + 포인트/이미지 토큰 Transformer 블록. 1-step 생성 가능. 해상도/파라미터/임의사이즈는 논문 미기재(=연구 대상).
- **주요 병목:** DINOv3(무거운 frozen 인코더)가 2TOPS의 실제 연산 병목. ViT pos-embed 고정격자가 해상도 유연성의 핵심.
- **NPU 리스크:** RKNN에서 트랜스포머 연산(attention/LayerNorm/softmax/GELU)이 CPU fallback되면 이론 TOPS와 무관하게 지연. 연산 지원 매트릭스가 R4 크리티컬.
- **참고 자산:** Meta DINOv3(공개), DiT/point-diffusion 계열 코드. 투명물체 벤치: TransCG, ClearGrasp, DREDS, ClearPose, Booster.

## Interview Transcript
<details><summary>Full Q&A (6 rounds + topology)</summary>

- **Round 0 (토폴로지/미션):** 3컴포넌트 확정, "1부터 하고 2를 거쳐 3까지" → 순차 프로그램, 최종 온디바이스.
- **Round 1 (제약/자산):** 보유 = GPU만. 보드·코드·데이터셋 미보유.
- **Round 2 (성공기준/속도):** 실시간 ≥10 FPS @ 320×240.
- **Round 3 (성공기준/정확도):** 느슨(용도 충족이면 OK).
- **Round 4 (Contrarian/목표):** 투명 물체 depth가 본질 → PointDiT 채택, 옵션 1로 진행.
- **Round 5 (제약/해상도):** 고정 2세트(320×240, 160×120)면 충분.
- **Round 6 (Simplifier/평가자):** 루브릭 커버리지 체크.

</details>
