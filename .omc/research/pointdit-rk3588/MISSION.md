# Autoresearch Mission: pointdit-rk3588

## Mission (single, phase-1 of 3)
투명 물체 depth를 PointDiT로 구현하고 RK3588 1코어(~2 TOPS)에서 320×240 실시간(≥10 FPS)으로
구동하기 위한 **근거 기반 방법론 리포트**를 작성/개선한다. 리포트는 재현 설계(#1), 해상도 지원(#2),
2-TOPS 최적화(#3)를 모두 커버한다. 하드웨어(RK3588 보드) 미보유이므로 최적화 수치는 **추정**
(연산 예산 모델·RKNN 연산 지원 매트릭스·양자화 시뮬·유사 모델 문헌)으로 산출한다.

## Artifact under improvement
`.omc/research/pointdit-rk3588/REPORT.md`  (autoresearch가 매 iteration마다 개선)

## Evaluator (strict contract)
```
python3 .omc/research/pointdit-rk3588/evaluate.py
```
- 통과 기준: 전체 루브릭 커버리지 ≥ 0.85 **AND** 크리티컬 섹션(R2, R4) 누락 0.
- exit 0 = pass, exit 1 = fail. stdout에 JSON 스코어카드 출력.
- 결정적(deterministic) 검사: 섹션 앵커 존재 + 최소 분량 + 최소 인용 수.
- 선택: `EVAL_LLM_JUDGE=1`이면 LLM-judge 품질 점수를 보조 지표로 병기(미설정 시 결정적 검사만).

## Max runtime
기본 상한 권장: 45분/run (autoresearch 세팅 시 확정).

## Roadmap (deferred phases — 이 미션 범위 외)
- Phase 2: PC/GPU 재현 + 벤치 eval 스크립트로 평가자 승격(정확도+지연 실측).
- Phase 3: RK3588 보드 확보 후 RKNN 양자화 배포, 온디바이스 지연/메모리/정확도 실측.

## Provenance
- Spec: `.omc/specs/deep-interview-pointdit-rk3588.md`
- PointDiT: arXiv 2607.02515 — https://haofeixu.github.io/pointdit/
