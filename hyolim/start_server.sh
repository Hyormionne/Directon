#!/usr/bin/env bash
# 상담 서버 시작 — 최종 측정 때 운영진이 이 파일을 그대로 실행한다. 9000번 포트, 0.0.0.0.
# API 키는 맨 위 폴더의 .env (SONIOX_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY) 에서 읽는다.
cd "$(dirname "$0")"
exec python3 -m server.app --port 9000
