"""AG-T14 데모 서버(demo/server.py): 정적 파일, 회원 목록, /calls 프록시, 설정, 피드백 저장.

네트워크 금지: 데모 서버는 127.0.0.1 임의 포트, mock API는 테스트 안의 가짜 HTTP 서버.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent.normalize import digits_for_speech
from demo import server as demo_server

ROOT = Path(__file__).resolve().parents[1]


class _FakeMock(BaseHTTPRequestHandler):
    seen: list[str] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        _FakeMock.seen.append(self.path)
        body = json.dumps([{"session_id": "DEMO-x", "name": "track_delivery",
                            "arguments": {"order_id": "2611150231"}, "matched": True,
                            "t_response": 1.0}]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start(srv):
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


@pytest.fixture
def fake_mock():
    _FakeMock.seen = []
    srv = _start(ThreadingHTTPServer(("127.0.0.1", 0), _FakeMock))
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def demo(tmp_path, fake_mock):
    srv = demo_server.create_server(port=0, mock_url=fake_mock, agent_ws="ws://127.0.0.1:9999",
                                    feedback_dir=tmp_path / "fb", date_fn=lambda: "20261120")
    _start(srv)
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}", tmp_path / "fb"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def _post(url, body: bytes, ctype="application/json"):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_binds_loopback_only(demo):
    srv, _, _ = demo
    assert srv.server_address[0] == "127.0.0.1"


def test_static_files(demo):
    _, base, _ = demo
    st, ct, body = _get(base + "/")
    assert st == 200 and ct.startswith("text/html")
    html = body.decode("utf-8")
    assert "app.js" in html and "style.css" in html
    st, ct, body = _get(base + "/app.js")
    assert st == 200 and "javascript" in ct
    st, ct, _ = _get(base + "/style.css")
    assert st == 200 and ct.startswith("text/css")
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/../agent/config.py")
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/nope.js")
    assert e.value.code == 404


def test_static_has_no_external_cdn():
    static = ROOT / "demo" / "static"
    for name in ("index.html", "app.js", "style.css"):
        text = (static / name).read_text(encoding="utf-8")
        for bad in ("cdn.", "googleapis", "unpkg", "jsdelivr", "<script src=\"http"):
            assert bad not in text, (name, bad)




def test_config(demo):
    _, base, _ = demo
    _, _, body = _get(base + "/api/config")
    cfg = json.loads(body)
    assert cfg["agent_ws"] == "ws://127.0.0.1:9999"
    assert cfg["mock_url"].startswith("http://127.0.0.1:")
    assert cfg["default_now"] == "2026-11-20 14:00"


def test_calls_proxy(demo):
    _, base, _ = demo
    st, _, body = _get(base + "/api/calls?session_id=DEMO-x")
    assert st == 200
    rows = json.loads(body)
    assert rows[0]["name"] == "track_delivery" and rows[0]["matched"] is True
    assert _FakeMock.seen == ["/calls?session_id=DEMO-x"]


def test_calls_proxy_mock_down(tmp_path):
    srv = _start(demo_server.create_server(port=0, mock_url="http://127.0.0.1:1",
                                           feedback_dir=tmp_path / "fb"))
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(f"http://127.0.0.1:{srv.server_address[1]}/api/calls?session_id=x")
        assert e.value.code == 502
    finally:
        srv.shutdown()
        srv.server_close()


def test_feedback_appends_jsonl(demo):
    _, base, fb = demo
    rec = {"session_id": "DEMO-1", "rating": 4, "comment": "좋아요",
           "turns": [{"user": "배송 언제 와요?", "agent": "네, 고객님.", "vote": "up", "latency_ms": 820}]}
    st, res = _post(base + "/api/feedback", json.dumps(rec, ensure_ascii=False).encode("utf-8"))
    assert st == 200 and res["ok"] is True
    st, _ = _post(base + "/api/feedback", json.dumps({"session_id": "DEMO-2"}).encode())
    assert st == 200
    path = fb / "feedback-20261120.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["session_id"] == "DEMO-1" and first["comment"] == "좋아요"
    assert first["turns"][0]["latency_ms"] == 820
    assert "received_at" in first


def test_feedback_rejects_bad_body(demo):
    _, base, fb = demo
    st, _ = _post(base + "/api/feedback", b"not json")
    assert st == 400
    st, _ = _post(base + "/api/feedback", b"[1,2]")
    assert st == 400
    assert not (fb / "feedback-20261120.jsonl").exists()
