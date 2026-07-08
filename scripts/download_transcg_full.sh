#!/usr/bin/env bash
# ============================================================================
# TransCG 전체(130 scene, ~181GB) 다운로드 스크립트
#
#   사용법:  bash scripts/download_transcg_full.sh
#
# - 재실행 안전: 이미 압축해제된 scene 은 건너뜀 (중간에 끊겨도 다시 실행)
# - GDrive quota 대비:
#     (1) data/dl/transcg-data-N.zip 이 이미 있으면 그걸 먼저 씀 (브라우저 수동 다운로드 지원)
#     (2) 없으면 gdown 으로 최대 3회 재시도
#     (3) 다 실패하면 브라우저/Baidu 링크를 출력하고 다음으로 넘어감
# - 무결성 검증(unzip -t) 후 압축해제, 성공 시 zip 삭제
#
# 완료 후 데이터 루트: data/TransCG/transcg/  (scene1..scene130 + camera_intrinsics/ + metadata.json)
# 그 경로로 학습:  --data data/TransCG/transcg
# ============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1          # 프로젝트 루트로 이동
ROOT="data/TransCG"                        # 압축해제 대상
DL="data/dl"                               # zip 임시 폴더
PY="./.venv/bin/python"                    # gdown 실행용 (uv venv)
mkdir -p "$ROOT" "$DL"

# archive 번호 : GDrive file id : 포함 scene 범위
#   (info.zip = 루트 metadata + camera_intrinsics, 필수)
INFO_ID="18LkbelKNTURF-8f8N-ykzs79FJ013knH"
declare -a IDS=(
  "1:1rDgbMrzyO8GvTwEyPDbyx6pS9TVVDJLI"   # scene 1-10
  "2:10etZWFV5r_wJRCR3Q38eq9WaRY8qWBFp"   # scene 11-20
  "3:19tXZ9lzpW2gUk1ibkij76I_-oknCbqyP"   # scene 21-30
  "4:13JFpSXd8d7vkVJvAUT1E6JOL6ZGC9B5f"   # scene 31-40
  "5:1sMt8pbiVTU4J2YCdN31ZV3BY6m4ywX9O"   # scene 41-50
  "6:1Uc2UDALZRG2NDZJt9CXMRLw1y6gpa_sT"   # scene 51-60
  "7:1NXCGDzC2MrWktR9HcUddtRpajeg2JMEf"   # scene 61-70
  "8:11MK-7_eX5oRNTdJPcOUXmvMaL3_cmFVh"   # scene 71-80
  "9:1oSMxQ0O9UD38Tu8w16BOhi8sPKMp6UQf"   # scene 81-90
  "10:1usAWlEJ0PE9gdgQb7CKo9CgkWOFr-Ms0"  # scene 91-100
  "11:1M-QQcEGFcTw2SJrr-Z_Gg1-fyA0PZrcb"  # scene 101-110
  "12:1KxpeNNOEOlv0t6PSNYWg0PoQ37OizDhR"  # scene 111-120
  "13:1FaZX6eWGYuMKu6riX-su3es3_XzoTawC"  # scene 121-130
)
# Baidu 폴백 (gdown 실패 시): https://graspnet.net/transcg 참고. 코드 예시:
#   data1 ncj0 · data2 3n0d · data3 umim · data4 g63o · data5 6tk3 · data6 kp14
#   data7 jmsu · data8 fqv9 · data9 equb · data10 msc7 · data11 ncg0 · data12 wntp · data13 in9v

# --- 사전 점검: gdown, 디스크 ---
$PY -c "import gdown" 2>/dev/null || { echo "gdown 없음 → 설치: uv pip install --python .venv/bin/python gdown"; exit 1; }
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "디스크 여유: ${FREE_GB}GB (전체 ~181GB 필요; 이미 받은 만큼 제외)"
[ "${FREE_GB:-0}" -lt 60 ] && echo "⚠️  여유 공간이 적습니다. 60GB+ 권장."

# --- info.zip (루트 metadata + intrinsics) ---
if [ ! -f "$ROOT/transcg/metadata.json" ]; then
  echo "[info] 루트 metadata 다운로드..."
  $PY -m gdown "$INFO_ID" -O "$DL/transcg-info.zip" && unzip -q -o "$DL/transcg-info.zip" -d "$ROOT" && rm -f "$DL/transcg-info.zip"
fi

got_zip () {  # $1=n → data/dl 에 온전한 zip 있으면 0
  local z="$DL/transcg-data-$1.zip"
  [ -f "$z" ] && unzip -t "$z" >/dev/null 2>&1
}

for pair in "${IDS[@]}"; do
  n="${pair%%:*}"; fid="${pair#*:}"
  s0=$(( (n-1)*10 + 1 )); s1=$(( n*10 ))
  # 이미 압축해제됨? (범위 첫 scene 폴더 존재로 판단)
  if [ -d "$ROOT/transcg/scene$s0" ]; then
    echo "[data-$n] scene $s0-$s1 이미 있음 → 건너뜀"; continue
  fi
  echo "==================== data-$n (scene $s0-$s1) ===================="
  # 1) 수동 다운로드 zip 있으면 사용, 없으면 gdown 3회 재시도
  if ! got_zip "$n"; then
    for try in 1 2 3; do
      echo "[data-$n] gdown 시도 $try/3 ..."
      rm -f "$DL/transcg-data-$n.zip"
      $PY -m gdown "$fid" -O "$DL/transcg-data-$n.zip"
      got_zip "$n" && break
      echo "[data-$n] 실패(quota 가능). 20초 후 재시도..."; sleep 20
    done
  fi
  # 2) 여전히 없으면 안내 후 다음으로
  if ! got_zip "$n"; then
    echo "❌ [data-$n] 자동 다운로드 실패 (GDrive quota 추정)."
    echo "   수동: 브라우저에서 https://drive.google.com/file/d/$fid/view 로 받아"
    echo "        $DL/transcg-data-$n.zip 로 저장 후 이 스크립트 재실행."
    echo "   또는 Baidu: https://graspnet.net/transcg (data-$n 코드는 스크립트 주석 참고)"
    continue
  fi
  # 3) 압축해제 + zip 삭제
  echo "[data-$n] 압축해제..."
  unzip -q -o "$DL/transcg-data-$n.zip" -d "$ROOT" && rm -f "$DL/transcg-data-$n.zip"
  echo "[data-$n] 완료. 현재 scene 수: $(ls -d "$ROOT"/transcg/scene* 2>/dev/null | wc -l)/130"
done

echo ""
echo "============================================================"
echo "완료. 최종 scene 수: $(ls -d "$ROOT"/transcg/scene* 2>/dev/null | wc -l)/130"
echo "130개면 전체 확보. 미만이면 실패한 data-N 을 수동 다운로드 후 재실행하세요."
echo "학습:  ./.venv/bin/python scripts/train.py --data data/TransCG/transcg \\"
echo "         --dataset transcg --split train --encoder dinov2 --npu --regression \\"
echo "         --grad_weight 2.0 --amp --steps 40000 --eval_every 1000 --viz_every 1000"
echo "============================================================"
