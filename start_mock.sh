#!/usr/bin/env bash
# 모의 함수 서버 시작 — 최종 측정 때 운영진이 `bash start_mock.sh <데이터 파일>` 로 실행한다(README 7장).
# 첫 인자로 받은 데이터 파일(측정 때는 mock_db_test.json)을 읽어 8100번 포트(0.0.0.0)에서 받는다.
cd "$(dirname "$0")"
exec python3 -m mock_api.server --host 0.0.0.0 --port 8100 --db "${1:-directon-2026-voice-cs/data/mock_db.json}"
