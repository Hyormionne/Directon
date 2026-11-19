#!/usr/bin/env bash
# 외부 접속용 Cloudflare quick tunnel (AG-T18). 측정기가 우리 서버에 붙을 wss 주소를 만든다.
#
# 사용:  bash scripts/tunnel.sh [포트=9000]
#   1) 먼저 상담원 서버:  uv run python -m agent.server --port 9000
#   2) 출력된 wss://xxxx.trycloudflare.com 주소를 주최 측에 전달. 측정이 끝날 때까지 끄지 않는다.
#
# cloudflared 설치: macOS `brew install cloudflared`, Linux는 https://github.com/cloudflare/cloudflared/releases
#   Windows는 scripts/tunnel.ps1 사용(winget install --id Cloudflare.cloudflared).
set -euo pipefail
PORT="${1:-9000}"
CLOUDFLARED="${CLOUDFLARED:-cloudflared}"
WAIT="${TUNNEL_WAIT_S:-60}"

if ! command -v "$CLOUDFLARED" >/dev/null 2>&1; then
  echo "cloudflared를 찾을 수 없습니다. 설치 후 다시 실행하세요." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/runs/tunnel"
LOG="$ROOT/runs/tunnel/tunnel_$(date +%Y%m%d_%H%M%S).log"

"$CLOUDFLARED" tunnel --no-autoupdate --url "http://localhost:$PORT" >"$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true' EXIT INT TERM

URL=""
for _ in $(seq 1 $((WAIT * 2))); do
  sleep 0.5
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -n1 || true)"
  [ -n "$URL" ] && break
  kill -0 "$PID" 2>/dev/null || break
done

if [ -z "$URL" ]; then
  echo "터널 주소를 받지 못했습니다. 로그: $LOG" >&2
  exit 1
fi
WSS="wss://${URL#https://}"
echo
echo "측정기 접속 주소:  $WSS"
echo "(HTTP 확인용: $URL)  로그: $LOG"
echo "점검: uv run python -m eval.stress --server $WSS --sessions 1 --rounds 1"
echo "$WSS" >"$ROOT/runs/tunnel/tunnel_url.txt"
wait "$PID"
