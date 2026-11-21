"""피드백 키트 서버. 실행: python -m feedback_kit --agent-ws ws://127.0.0.1:9000 [--port 8090] [--out DIR]

표준 라이브러리만 쓴다. 127.0.0.1에만 바인딩한다(브라우저 마이크는 http://127.0.0.1·localhost에서만 허용).
- GET  /               → static/index.html (그 밖의 /<파일>은 static/ 안의 파일만)
- GET  /api/config     → 상담원 WS 주소, tools_url, 상담원 이름, 기본 now, 저장 폴더
- POST /api/sessions   → 통화 묶음 JSON(store.py 형식)을 <out>/<session_id>/로 풀어 저장
상담원과의 음성 연결은 브라우저가 직접 WebSocket으로 한다(이 서버는 중계하지 않는다).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import store

STATIC_DIR = Path(__file__).resolve().parent / "static"
HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_AGENT_WS = "ws://127.0.0.1:9000"
MAX_SESSION_BODY = 300_000_000   # 오디오 base64 포함. 10분 통화 ≈ 55MB
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8"}


class KitServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, *, agent_ws: str, out_dir: Path, tools_url: str | None,
                 agent_label: str | None, now: str | None, static_dir: Path):
        super().__init__(addr, Handler)
        self.agent_ws = agent_ws
        self.out_dir = Path(out_dir)
        self.tools_url = tools_url
        self.agent_label = agent_label
        self.now = now
        self.static_dir = Path(static_dir).resolve()


class Handler(BaseHTTPRequestHandler):
    server: KitServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), CONTENT_TYPES[".json"])

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/config":
            s = self.server
            return self._json({"agent_ws": s.agent_ws, "tools_url": s.tools_url, "agent_label": s.agent_label,
                               "now": s.now, "out_dir": str(s.out_dir), "schema": store.SCHEMA})
        rel = urllib.parse.unquote(u.path).lstrip("/") or "index.html"
        root = self.server.static_dir
        try:
            target = (root / rel).resolve()
            target.relative_to(root)
        except (ValueError, OSError):
            return self._json({"error": "not found"}, 404)
        if not target.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = CONTENT_TYPES.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] \
            or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/api/sessions":
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_SESSION_BODY:
            self.close_connection = True
            return self._json({"error": "본문 크기가 올바르지 않습니다"}, 400)
        try:
            bundle = json.loads(self.rfile.read(n))
        except ValueError:
            return self._json({"error": "JSON이 아닙니다"}, 400)
        try:
            res = store.write_session(bundle, self.server.out_dir)
        except store.BundleError as e:
            return self._json({"error": str(e)}, 400)
        except OSError as e:
            return self._json({"error": f"저장 실패: {e}"}, 500)
        self._json({"ok": True, **res})


def create_server(port: int = DEFAULT_PORT, *, agent_ws: str = DEFAULT_AGENT_WS,
                  out_dir: Path | str = store.DEFAULT_ROOT, tools_url: str | None = None,
                  agent_label: str | None = None, now: str | None = None,
                  static_dir: Path | str = STATIC_DIR) -> KitServer:
    """127.0.0.1에 바인딩한 서버(serve_forever는 호출자가). port=0이면 임의 포트."""
    return KitServer((HOST, port), agent_ws=agent_ws, out_dir=Path(out_dir), tools_url=tools_url,
                     agent_label=agent_label, now=now, static_dir=Path(static_dir))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m feedback_kit", description="음성 상담원 휴먼 피드백 키트")
    ap.add_argument("--agent-ws", default=DEFAULT_AGENT_WS, help=f"상담원 WebSocket 주소(기본 {DEFAULT_AGENT_WS})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"페이지 포트(기본 {DEFAULT_PORT})")
    ap.add_argument("--out", type=Path, default=store.DEFAULT_ROOT, help="저장 폴더(기본 ./feedback_sessions)")
    ap.add_argument("--tools-url", default=None, help="session.start의 tools_url(Mock API 주소)")
    ap.add_argument("--agent-label", default=None, help="상담원 이름·버전(평가 기록에 남는다), 예: '팀원A v3'")
    ap.add_argument("--now", default=None, help="session.start의 now 기본값, 예: '2026-11-25 (수) 10:00'")
    a = ap.parse_args(argv)
    srv = create_server(a.port, agent_ws=a.agent_ws, out_dir=a.out, tools_url=a.tools_url,
                        agent_label=a.agent_label, now=a.now)
    print(f"피드백 키트: http://{HOST}:{srv.server_address[1]}  (상담원 {a.agent_ws}, 저장 {a.out.resolve()})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
