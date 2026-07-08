# Evaluator Rubric — pointdit-rk3588 methodology report

리포트(`REPORT.md`)는 아래 5개 섹션을 커버해야 한다. 각 항목은 근거/인용(arXiv, URL, 수치, 표)을 포함해야 하며,
`evaluate.py`가 섹션 앵커·최소 분량·최소 인용 수를 결정적으로 검사한다. 크리티컬 = R2, R4.

## R1 — PointDiT 재현 설계  (anchor: `## R1`)
- DINOv3 변형(크기/patch), denoiser ViT 깊이·폭·헤드, 포인트맵 XYZ 토큰화, 조건부 주입 방식
- diffusion 스케줄, 1-step 생성 메커니즘, 손실/좌표계(정규화 포인트맵)/카메라 가정
- 공식 코드 부재 대응: 공개 자산 매핑(Meta DINOv3, DiT 계열), 재현 리스크·불확실성
- 최소 인용 ≥ 3

## R2 — 투명 물체 depth  (crytical, anchor: `## R2`)
- 벤치마크 조사·선정: TransCG / ClearGrasp / DREDS / ClearPose / Booster 중 근거 기반 선정
- 지표(AbsRel, δ1, RMSE)·평가 프로토콜, 투명 영역 마스크 처리
- diffusion prior가 투명 표면에 유리한 이유(직접 단서 부재 → 생성적 형상 보완)
- 학습 데이터 전략(투명 물체 포함/파인튜닝)
- 최소 인용 ≥ 3

## R3 — 해상도 지원  (anchor: `## R3`)
- ViT pos-embed 2D 보간, DINOv3 고정 격자 대응
- 320×240 / 160×120 토큰 그리드 산출(patch size별), 저해상도 정확도-토큰수 트레이드오프
- multi-res 학습 전략, 고정 2세트 배포 근거
- 최소 인용 ≥ 2

## R4 — RK3588 2-TOPS 최적화  (critical, anchor: `## R4`)
- 프레임당 MAC 예산 모델: 320×240 @ 10 FPS → 예산 vs DINOv3+denoiser 추정 MAC, 1-step 강제 근거
- DINOv3 경량화/대체(증류→ViT-Tiny/Small, CNN 인코더 대체) 및 정확도 영향
- INT8/혼합정밀 양자화(PTQ/QAT)·캘리브레이션, 투명물체 정확도 영향
- **RKNN 연산 지원 매트릭스**: attention/LayerNorm/GELU/softmax NPU 지원·CPU fallback 리스크, NPU 친화 재설계
- 고정 shape 컴파일(320×240, 160×120), 추정 지연/메모리, 예상 병목
- 보드 부재 시 추정 방법론(FLOP 기반, 시뮬, 유사 모델 RK3588 문헌)
- 최소 인용 ≥ 4

## R5 — 통합 로드맵 & Pareto  (anchor: `## R5`)
- 1→2→3 단계별 산출물·평가자·마일스톤
- 속도-정확도 Pareto 후보(모델크기 × 양자화 × 스텝수)·권장 지점
- 핵심 리스크·완화(10 FPS 미달, 투명물체 일반화, RKNN 미지원 연산)
- 최소 인용 ≥ 2

## Pass threshold
- 전체 커버리지 ≥ 0.85 (총 5섹션, 각 섹션 앵커+분량+인용 충족 시 1점)
- 크리티컬(R2, R4) 누락 시 무조건 fail
