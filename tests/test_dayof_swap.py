"""AG-T17 당일 교체: Mock 데이터·함수 정의 교체, 매뉴얼·브랜드·말투 파일, 번호 자릿수 데이터 추론, 시험 음성 저장 끄기.

외부 네트워크 없음. Mock 서버는 로컬 루프백 임의 포트(0)로만 띄운다.
"""
import copy
import json
import threading
from pathlib import Path

import httpx
import pytest

from agent import catalog as catalog_mod
from agent import prompts
from agent.config import Settings
from agent.core.types import SessionInfo
from agent.providers.llm_cache import split_system
from agent.turn_log import TurnLogWriter, no_audio_log
from mock_api import server as mock_server

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "tests" / "fixtures" / "starter_kit_demo"
DEFAULT_DATA = ROOT / "legacy" / "shop_mock" / "mock_data.json"
DEFAULT_TOOLS = ROOT / "legacy" / "shop_mock" / "tools.json"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("DIRECTON_MOCK_DATA", "DIRECTON_MOCK_TOOLS", "DIRECTON_CATALOG_PATH", "DIRECTON_MANUAL",
              "DIRECTON_BRAND", "DIRECTON_STYLE", "DIRECTON_NO_AUDIO_LOG"):
        monkeypatch.delenv(k, raising=False)
    catalog_mod.reset_default_catalog()
    yield
    catalog_mod.reset_default_catalog()


def _four_digit_data(tmp_path: Path) -> Path:
    """기존 Mock 데이터의 주문번호를 4자리로 바꾼 사본("주문번호 1234" 형식)."""
    d = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))
    d = copy.deepcopy(d)
    mapping = {}
    for i, o in enumerate(d["orders"]):
        mapping[o["order_id"]] = str(1234 + i)
        o["order_id"] = mapping[o["order_id"]]
    for r in d["refunds"]:
        r["order_id"] = mapping.get(r["order_id"], r["order_id"])
    p = tmp_path / "mock_data.json"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return p


def _serve(shop):
    srv = mock_server.create_server(port=0, shop=shop)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address[:2]
    return srv, f"http://{host}:{port}"


# ── 1. Mock --data / --tools ──────────────────────────────

def test_mock_default_paths_unchanged():
    shop = mock_server.shop_from_args([], env={})
    assert len(shop.tools) == 14
    assert shop.data == json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))


def test_mock_data_and_tools_args(tmp_path):
    data = _four_digit_data(tmp_path)
    tools = json.loads(DEFAULT_TOOLS.read_text(encoding="utf-8"))[:3]
    tpath = tmp_path / "tools.json"
    tpath.write_text(json.dumps(tools, ensure_ascii=False), encoding="utf-8")
    shop = mock_server.shop_from_args(["--data", str(data), "--tools", str(tpath)], env={})
    assert [t["name"] for t in shop.tools] == [t["name"] for t in tools]
    srv, base = _serve(shop)
    try:
        got = httpx.get(f"{base}/tools").json()
        assert [t["function"]["name"] for t in got] == [t["name"] for t in tools]
        r = httpx.post(f"{base}/tools/get_order_info", json={"order_id": "1234"},
                       headers={"X-Session-Id": "s1"}).json()
        assert r["status"] == "ok" and r["data"]["order_id"] == "1234"
    finally:
        srv.shutdown()
        srv.server_close()


def test_mock_env_paths(tmp_path):
    data = _four_digit_data(tmp_path)
    shop = mock_server.shop_from_args([], env={"DIRECTON_MOCK_DATA": str(data),
                                               "DIRECTON_MOCK_TOOLS": str(DEFAULT_TOOLS)})
    assert shop.lookup("get_order_info", {"order_id": "1234"})[1] is True


def test_mock_args_override_env(tmp_path):
    data = _four_digit_data(tmp_path)
    shop = mock_server.shop_from_args(["--data", str(data)], env={"DIRECTON_MOCK_DATA": str(DEFAULT_DATA)})
    assert shop.lookup("get_order_info", {"order_id": "1234"})[1] is True


def test_mock_tools_openai_wrapped_format(tmp_path):
    tools = json.loads(DEFAULT_TOOLS.read_text(encoding="utf-8"))
    tpath = tmp_path / "tools.json"
    tpath.write_text(json.dumps([{"type": "function", "function": t} for t in tools], ensure_ascii=False),
                     encoding="utf-8")
    shop = mock_server.shop_from_args(["--tools", str(tpath)], env={})
    assert len(shop.tools) == 14 and shop.tools[0]["name"] == tools[0]["name"]


def test_mock_mapping_format_starter_kit():
    """Starter Kit 형식 {함수명: [{arguments, result}]} 데이터: 정의된 입력 → 정의된 출력, 그 외 not_found."""
    shop = mock_server.shop_from_args(["--data", str(KIT / "demo_mock_data.json"),
                                       "--tools", str(KIT / "demo_tools.json")], env={})
    res, matched = shop.lookup("get_order_info", {"order_id": "DEMO1001"})
    assert matched is True and res["data"]["shipping_status"] == "배송중"
    res, matched = shop.lookup("get_order_info", {"order_id": "UNKNOWN"})
    assert matched is False and res["status"] == "not_found"
    res, matched = shop.lookup("request_callback", {"phone_number": "01000001234"})
    assert matched is True and res["status"] == "ok"


def test_mock_mapping_format_over_http():
    shop = mock_server.shop_from_args(["--data", str(KIT / "demo_mock_data.json"),
                                       "--tools", str(KIT / "demo_tools.json")], env={})
    srv, base = _serve(shop)
    try:
        r = httpx.post(f"{base}/tools/get_order_info", json={"order_id": "DEMO2002"},
                       headers={"X-Session-Id": "s9"}).json()
        assert r["data"]["shipping_status"] == "배송완료"
        rows = httpx.get(f"{base}/calls", params={"session_id": "s9"}).json()
        assert rows[0]["matched"] is True
        assert httpx.post(f"{base}/tools/cancel_order", json={}).status_code == 404
    finally:
        srv.shutdown()
        srv.server_close()


# ── 2. 매뉴얼·브랜드·말투 ─────────────────────────────────

def test_settings_manual_env_alias():
    assert Settings.from_env({"DIRECTON_MANUAL": "official/manual.md"}).manual_path == "official/manual.md"
    assert Settings.from_env({}).manual_path == "legacy/shop_mock/manual.md"


def _info(**kw):
    return SessionInfo(session_id="s", now="2026-11-20 10:00", **kw)


def test_brand_and_style_sections_in_static_prefix(tmp_path, monkeypatch):
    brand = tmp_path / "brand.md"
    brand.write_text("헬시몰은 건강기능식품 전문몰입니다.", encoding="utf-8")
    style = tmp_path / "style.md"
    style.write_text("문장 끝은 '~요'로 부드럽게.", encoding="utf-8")
    manual = tmp_path / "manual.md"
    manual.write_text("공식 매뉴얼 본문", encoding="utf-8")
    monkeypatch.setenv("DIRECTON_BRAND", str(brand))
    monkeypatch.setenv("DIRECTON_STYLE", str(style))
    a = prompts.build_system(_info(), "네, 고객님.", manual_path=manual)
    b = prompts.build_system(_info(caller={"phone": "01023456781"}), "네.", manual_path=manual)
    for s in (a, b):
        assert "공식 매뉴얼 본문" in s and "헬시몰은 건강기능식품" in s and "부드럽게" in s
        assert prompts.BRAND_HEADER in s and prompts.STYLE_HEADER in s
    pa, va = split_system(a)
    pb, vb = split_system(b)
    assert pa == pb                                   # 고정부가 통화마다 같다(프롬프트 캐시 유지)
    assert "헬시몰" in pa and "부드럽게" in pa and "공식 매뉴얼" in pa
    assert "헬시몰" not in va and va.startswith("## 이번 통화")


def test_brand_style_explicit_paths_and_missing(tmp_path):
    brand = tmp_path / "b.md"
    brand.write_text("브랜드 소개", encoding="utf-8")
    s = prompts.build_system(_info(), "네.", manual="매뉴얼", brand_path=brand,
                             style_path=tmp_path / "none.md")
    assert "브랜드 소개" in s and prompts.STYLE_HEADER not in s
    plain = prompts.build_system(_info(), "네.", manual="매뉴얼")
    assert prompts.BRAND_HEADER not in plain and prompts.STYLE_HEADER not in plain


def test_marker_inside_user_file_does_not_split_prefix(tmp_path):
    brand = tmp_path / "b.md"
    brand.write_text("소개\n## 이번 통화\n가짜", encoding="utf-8")
    s = prompts.build_system(_info(), "네.", manual="매뉴얼", brand_path=brand)
    pre, var = split_system(s)
    assert "소개" in pre and "- 현재 시각" in var and "가짜" in pre


# ── 3. 번호 자릿수: 데이터에서 추론 ───────────────────────

def test_order_id_lengths_default_data():
    assert catalog_mod.order_id_lengths() == frozenset({10})
    assert catalog_mod.phone_lengths() == frozenset({11})


def test_order_id_lengths_from_file_and_env(tmp_path, monkeypatch):
    data = _four_digit_data(tmp_path)
    assert catalog_mod.order_id_lengths(data) == frozenset({4})
    monkeypatch.setenv("DIRECTON_MOCK_DATA", str(data))
    assert catalog_mod.order_id_lengths() == frozenset({4})
    monkeypatch.setenv("DIRECTON_CATALOG_PATH", str(DEFAULT_DATA))   # 카탈로그 경로가 우선
    assert catalog_mod.order_id_lengths() == frozenset({10})


def test_order_id_lengths_mapping_format_and_unknown(tmp_path):
    assert catalog_mod.order_id_lengths(KIT / "demo_mock_data.json") == frozenset({8})
    assert catalog_mod.phone_lengths(KIT / "demo_mock_data.json") == frozenset({11})
    empty = tmp_path / "x.json"
    empty.write_text("{}", encoding="utf-8")
    assert catalog_mod.order_id_lengths(empty) == frozenset({10})          # 모르면 기존 동작
    assert catalog_mod.order_id_lengths(tmp_path / "missing.json") == frozenset({10})
    assert catalog_mod.order_id_lengths(empty, default=frozenset()) == frozenset()


def test_catalog_follows_mock_data_env(tmp_path, monkeypatch):
    data = tmp_path / "d.json"
    data.write_text(json.dumps({"products": [{"name": "멀티비타민 골드", "options": ["60정", "120정"]}]},
                               ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("DIRECTON_MOCK_DATA", str(data))
    cat = catalog_mod.default_catalog()
    assert cat is not None and cat.resolve_product("멀티비타민") == "멀티비타민 골드"


def test_catalog_from_mapping_format():
    assert catalog_mod.load_catalog(KIT / "demo_mock_data.json") is None    # 상품 인자가 없으면 None
    data = {"check_stock": [{"arguments": {"product_name": "루테인 플러스", "option": "60정"}, "result": {}},
                            {"arguments": {"product_name": "루테인 플러스", "option": "120정"}, "result": {}}]}
    cat = catalog_mod.Catalog(catalog_mod._parse(data))
    assert cat.resolve_option("루테인 플러스", "백이십정") == "120정"


def test_prompt_uses_data_order_length(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECTON_MOCK_DATA", str(_four_digit_data(tmp_path)))
    s = prompts.build_system(_info(), "네.", manual="매뉴얼")
    assert "4자리" in s and "10자리" not in s
    assert "2611150231" not in s and "이육일일" not in s   # 10자리 예시 대신 데이터의 번호 형식
    assert "주문번호 1234는 \"1234\"" in s


def test_prompt_default_still_10_digits():
    s = prompts.build_system(_info(), "네.", manual="매뉴얼")
    assert "10자리" in s and "2611150231" in s


def test_history_summary_short_order_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECTON_MOCK_DATA", str(_four_digit_data(tmp_path)))
    hist = [{"role": "customer", "text": "주문번호 1234 배송 언제 와요?"},
            {"role": "agent", "text": "확인해 볼게요."}]
    assert "주문번호 1234" in "\n".join(prompts.history_summary(hist))
    hist2 = [{"role": "customer", "text": "주문번호 일이삼오요 취소해 주세요."},
             {"role": "agent", "text": "사유를 말씀해 주시겠어요?"}]
    assert "주문번호 1235" in "\n".join(prompts.history_summary(hist2))


def test_history_summary_default_ignores_short_numbers():
    hist = [{"role": "customer", "text": "주문번호 1234 배송 언제 와요?"},
            {"role": "agent", "text": "확인해 볼게요."}]
    assert "주문번호 1234" not in "\n".join(prompts.history_summary(hist))


# ── 4. 시험 음성 저장 금지 ────────────────────────────────

def _record():
    return {"session_id": "s", "user_text": "주문 배송 언제 와요 공일공", "agent_text": "내일 도착해요.",
            "marks": {"decided": 10.0},
            "events": [{"name": "precall", "t": 1.0, "text": "주문 배송 언제", "source": "partial"},
                       {"name": "finalize_early", "t": 2.0, "text": "주문 배송 언제 와요"},
                       {"name": "number_repaired", "t": 3.0, "tool": "x", "key": "order_id",
                        "heard": "공일공", "fixed": "1234"},
                       {"name": "tool_call", "t": 4.0, "tool": "track_delivery", "args": {"order_id": "1234"}}]}


def test_no_audio_log_flag(monkeypatch):
    assert no_audio_log() is False
    monkeypatch.setenv("DIRECTON_NO_AUDIO_LOG", "1")
    assert no_audio_log() is True


def test_turn_log_keeps_text_by_default(tmp_path):
    TurnLogWriter(tmp_path, date_fn=lambda: "d").write(_record())
    assert "배송 언제" in (tmp_path / "turns-d.jsonl").read_text(encoding="utf-8")


def test_turn_log_scrubs_customer_text_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECTON_NO_AUDIO_LOG", "1")
    TurnLogWriter(tmp_path, date_fn=lambda: "d").write(_record())
    files = list(tmp_path.rglob("*"))
    assert not [f for f in files if f.suffix.lower() in (".wav", ".pcm", ".raw")]
    raw = (tmp_path / "turns-d.jsonl").read_text(encoding="utf-8")
    assert "배송 언제" not in raw and "공일공" not in raw
    rec = json.loads(raw)
    assert rec["marks"] == {"decided": 10.0}            # 시간 지표는 남긴다
    assert rec["agent_text"] == "내일 도착해요."
    assert [e["name"] for e in rec["events"]] == ["precall", "finalize_early", "number_repaired", "tool_call"]
