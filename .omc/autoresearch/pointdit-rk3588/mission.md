# Autoresearch Mission — pointdit-rk3588 (v1, single mission)

- **Slug:** pointdit-rk3588
- **Started:** 2026-07-04T01:36:22+0900
- **Max runtime:** 45 min / run
- **Artifact under improvement:** `.omc/research/pointdit-rk3588/REPORT.md`
- **Evaluator:** `python3 .omc/research/pointdit-rk3588/evaluate.py`
- **Rubric:** `.omc/research/pointdit-rk3588/RUBRIC.md`
- **Spec:** `.omc/specs/deep-interview-pointdit-rk3588.md`
- **Pass:** coverage ≥ 0.85 AND critical (R2, R4) missing == 0

## Goal
투명물체 depth를 PointDiT로 구현 + 임의 소해상도(배포 고정 320×240 / 160×120) +
RK3588 1코어 ~2 TOPS 실시간(≥10 FPS) 방법론 리포트. Phase 1(리포트)만 이번 미션.
Phase 2(PC/GPU 재현), Phase 3(온디바이스 실측)은 로드맵.

## Rubric sections
- R1 PointDiT 재현 설계
- R2 투명 물체 depth (critical)
- R3 해상도 지원
- R4 RK3588 2-TOPS 최적화 (critical)
- R5 통합 로드맵 & Pareto
