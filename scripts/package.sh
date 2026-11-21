#!/usr/bin/env bash
# 제출 묶음 만들기: dist/directon-submit/ 와 dist/directon-submit.zip
#   bash scripts/package.sh
# 맨 위 폴더에 start_server.sh, start_mock.sh, requirements.txt, .env 가 온다(공식 README 7장).
# .env 에는 대회 게이트웨이 키(BAZE_API_KEY)와 TTS 설정만 넣는다. 개발용 OPENAI_API_KEY 는 넣지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
KIT="$ROOT/DIRECT_ON 2026 Starter Kit"
OUT="$ROOT/dist/directon-submit"

need() { [ -e "$1" ] || { echo "없음: $1" >&2; exit 1; }; }
need "$ROOT/.env"; need "$ROOT/models/silero_vad.onnx"
need "$KIT/data/brand/tools.json"; need "$KIT/data/brand/scenarios.md"
need "$KIT/data/brand/information.json"; need "$KIT/data/brand/style.json"; need "$KIT/data/mock_db.json"

rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/models" "$OUT/data/brand"
cp -R "$ROOT/agent" "$ROOT/mock_api" "$OUT/"
cp "$ROOT/requirements.txt" "$ROOT/start_server.sh" "$ROOT/start_mock.sh" "$OUT/"
cp "$ROOT/models/silero_vad.onnx" "$OUT/models/"
cp "$KIT/data/brand/"{tools.json,scenarios.md,information.json,style.json} "$OUT/data/brand/"
cp "$KIT/data/mock_db.json" "$OUT/data/"
if [ -d "$ROOT/assets/tts_cache" ]; then mkdir -p "$OUT/assets"; cp -R "$ROOT/assets/tts_cache" "$OUT/assets/"; fi
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$OUT" -name "*.pyc" -delete

# 제출용 .env: 개발 .env 에서 허용한 이름만 옮긴다(값은 화면에 찍지 않는다)
: > "$OUT/.env"
for k in BAZE_API_KEY BAZE_BASE_URL GEMINI_API_KEY SONIOX_API_KEY DIRECTON_TTS_PROVIDER GEMINI_TTS_MODEL GEMINI_TTS_VOICE GEMINI_TTS_STYLE; do
  line="$(grep -E "^${k}=" "$ROOT/.env" | tail -1 || true)"
  [ -n "$line" ] && echo "$line" >> "$OUT/.env"
done
grep -q "^BAZE_API_KEY=" "$OUT/.env" || { echo "제출용 .env 에 BAZE_API_KEY 가 없다" >&2; exit 1; }
if grep -q "^OPENAI_API_KEY=" "$OUT/.env"; then echo "제출용 .env 에 개발용 키가 들어갔다" >&2; exit 1; fi
chmod +x "$OUT/start_server.sh" "$OUT/start_mock.sh"

for f in start_server.sh start_mock.sh requirements.txt .env agent/server.py mock_api/server.py \
         models/silero_vad.onnx data/brand/tools.json data/mock_db.json; do need "$OUT/$f"; done
(cd "$ROOT/dist" && zip -qr directon-submit.zip directon-submit)
echo "묶음: $OUT.zip ($(du -h "$OUT.zip" | cut -f1)), 파일 $(find "$OUT" -type f | wc -l | tr -d ' ')개, .env 이름: $(cut -d= -f1 "$OUT/.env" | tr '\n' ' ')"
