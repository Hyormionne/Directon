#!/usr/bin/env bash
# 모의 함수 서버 시작. 최종 측정 때 운영진이 `bash start_mock.sh <데이터 파일>`로 실행한다(8100번, 0.0.0.0).
cd "$(dirname "$0")"
exec python3 -m mock_api.server --port 8100 --db "${1:-data/mock_db.json}"
