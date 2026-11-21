#!/usr/bin/env bash
# 상담 서버 시작 — 최종 측정 때 운영진이 그대로 실행한다. 9000번, 0.0.0.0. API 키는 .env에서 읽는다.
cd "$(dirname "$0")"
export DIRECTON_MANUAL="${DIRECTON_MANUAL:-official/manual.md}"
export DIRECTON_STYLE="${DIRECTON_STYLE:-official/style.md}"
export DIRECTON_MOCK_DATA="${DIRECTON_MOCK_DATA:-official/mock_db.json}"
exec python3 -m agent.server --port 9000 --host 0.0.0.0
