"""AG-T14b 데모 이전 대화(history) 주입: 프리셋 파일·/api/history_presets·정적 UI 요소.

네트워크 금지: 데모 서버는 127.0.0.1 임의 포트. 기대 함수 호출은 mock 서버 없이 load_shop().lookup으로 판정.
"""
import json
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pytest

from demo import server as demo_server
from mock_api.server import KEY_ARGS, load_shop

ROOT = Path(__file__).resolve().parents[1]
PRESETS_PATH = ROOT / "demo" / "presets" / "history_presets.json"
STATIC = ROOT / "demo" / "static"


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    srv = demo_server.create_server(port=0, mock_url="http://127.0.0.1:1", agent_ws="ws://127.0.0.1:9999",
                                    feedback_dir=tmp_path_factory.mktemp("fb"))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture(scope="module")
def presets(base):
    with urllib.request.urlopen(base + "/api/history_presets", timeout=5) as r:
        assert r.status == 200
        assert r.headers.get("Content-Type", "").startswith("application/json")
        body = json.loads(r.read())
    assert isinstance(body, list)
    return body


@pytest.fixture(scope="module")
def mock_data():
    return json.loads((ROOT / "legacy" / "shop_mock" / "mock_data.json").read_text(encoding="utf-8"))


def test_preset_file_exists_and_not_served_as_static(base):
    assert PRESETS_PATH.is_file()
    json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/../presets/history_presets.json", timeout=5)
    assert e.value.code == 404


def test_presets_format(presets):
    assert 6 <= len(presets) <= 8
    ids = [p["id"] for p in presets]
    assert len(set(ids)) == len(ids)
    for p in presets:
        for k in ("id", "name", "description", "caller", "caller_name", "now", "history", "try_saying", "expected"):
            assert k in p, (p.get("id"), k)
        assert re.fullmatch(r"[a-z0-9_]+", p["id"])
        assert p["name"].strip() and p["description"].strip()
        datetime.strptime(p["now"], "%Y-%m-%d %H:%M")
        assert isinstance(p["try_saying"], list) and p["try_saying"]
        assert all(isinstance(s, str) and s.strip() for s in p["try_saying"])
        exp = p["expected"]
        assert isinstance(exp.get("tools"), list) and isinstance(exp.get("note"), str) and exp["note"].strip()


def test_history_roles_and_last_is_agent(presets):
    for p in presets:
        h = p["history"]
        assert isinstance(h, list) and len(h) >= 2, p["id"]
        for turn in h:
            assert set(turn) == {"role", "text"}, (p["id"], turn)   # 서버가 받는 형식 그대로
            assert turn["role"] in ("agent", "customer"), (p["id"], turn)
            assert isinstance(turn["text"], str) and turn["text"].strip(), (p["id"], turn)
        assert h[-1]["role"] == "agent", p["id"]
        assert h[0]["role"] == "customer", p["id"]       # 대회 형식: 고객이 먼저 말한다


def test_caller_matches_mock_member(presets, mock_data):
    members = {m["phone_number"]: m["name"] for m in mock_data["members"]}
    assert any(p["caller"] is None for p in presets)
    assert any(p["caller"] is not None for p in presets)
    for p in presets:
        if p["caller"] is None:
            assert p["caller_name"] is None
        else:
            assert p["caller"] in members, p["id"]
            assert p["caller_name"] == members[p["caller"]], p["id"]


def test_expected_tools_match_mock(presets, mock_data):
    shop = load_shop()
    orders = {o["order_id"]: o for o in mock_data["orders"]}
    for p in presets:
        exp = p["expected"]
        assert exp["tools"], p["id"]
        for call in exp["tools"] + exp.get("optional_tools", []):
            assert call["name"] in shop.tool_names, (p["id"], call)
            resp, matched = shop.lookup(call["name"], dict(call["arguments"]))
            assert matched is True, (p["id"], call, resp)
            key = {k: v for k, v in call["arguments"].items() if k in KEY_ARGS}
            # 회원 프리셋이면 키 인자가 그 회원 것이어야 한다
            if p["caller"]:
                if "phone_number" in key:
                    assert key["phone_number"] == p["caller"], (p["id"], call)
                if "order_id" in key:
                    assert orders[key["order_id"]]["phone_number"] == p["caller"], (p["id"], call)
        for name in exp.get("not_tools", []):
            assert name in shop.tool_names, (p["id"], name)
            assert name not in {c["name"] for c in exp["tools"]}, (p["id"], name)


def test_presets_cover_required_situations(presets):
    names = {c["name"] for p in presets for c in p["expected"]["tools"]}
    for need in ("cancel_order", "request_exchange", "transfer_to_agent", "request_callback", "check_stock",
                 "change_shipping_address"):
        assert need in names, need
    # 영업시간 외 프리셋이 하나는 있다(평일 09~18 밖 또는 주말)
    def after_hours(now):
        t = datetime.strptime(now, "%Y-%m-%d %H:%M")
        return t.weekday() >= 5 or not (9 <= t.hour < 18)
    assert any(after_hours(p["now"]) for p in presets)
    # check_stock은 옵션까지 넣는 경우가 있다
    assert any("option" in c["arguments"] for p in presets for c in p["expected"]["tools"]
               if c["name"] == "check_stock")


def test_history_not_copied_from_testset(presets):
    lines = (ROOT / "legacy" / "testset" / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    test_texts = set()
    test_utts = set()
    for line in lines:
        if not line.strip():
            continue
        c = json.loads(line)
        test_texts.update(h["text"].strip() for h in c.get("history") or [])
        test_utts.add(c["customer_utterance"].strip())
    for p in presets:
        for turn in p["history"]:
            assert turn["text"].strip() not in test_texts, (p["id"], turn["text"])


def test_static_has_history_ui():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for el_id in ("history-panel", "history-preset", "history-turns", "history-add", "history-clear",
                  "history-json-toggle", "history-json", "history-warn", "history-desc"):
        assert f'id="{el_id}"' in html, el_id
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "/api/history_presets" in js
    assert "history: []" not in js                      # 실제로 주입한 history를 보낸다
    assert re.search(r"history:\s*call\.history", js)
    for word in ("agent", "customer"):
        assert f'"{word}"' in js
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert ".history-turn" in css and ".log .injected" in css
