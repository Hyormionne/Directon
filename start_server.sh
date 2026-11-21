#!/usr/bin/env bash
# 상담 서버 시작. 최종 측정 때 운영진이 `bash start_server.sh`로 실행한다(9000번, 0.0.0.0).
# 키와 설정은 같은 폴더의 .env에서 읽는다(agent.server가 직접 읽는다).
cd "$(dirname "$0")"
exec python3 -m agent.server --port 9000 --host 0.0.0.0
