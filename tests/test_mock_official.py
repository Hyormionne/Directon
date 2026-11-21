"""AG-T30 공식 규격 모의 함수 서버(mock_api/server.py). 규칙: 공식 키트 SPEC.md 「모의 함수 서버」.

픽스처(tests/fixtures/official_mini)는 공식 형식의 지어낸 값이다. 네트워크는 127.0.0.1 임의 포트만 쓴다.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest

from mock_api import server as mock

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "fixtures" / "official_mini"
KIT = ROOT / "DIRECT_ON 2026 Starter Kit" / "data"


@pytest.fixture(scope="module")
def store() -> mock.Store:
    return mock.load_store(MINI / "mock_db.json", MINI / "brand" / "tools.json", env={})


@pytest.fixture(scope="module")
def base(store):
    srv = mock.create_server(port=0, host="127.0.0.1", store=store)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _orders(res: dict) -> list[str]:
    return [o["주문번호"] for o in res["data"]["orders"]]


# ── 조회: 숫자만 비교 ─────────────────────────────────
@pytest.mark.parametrize("order_id", ["20260101-0000001", "202601010000001", "20260101 0000001"])
def test_order_lookup_ignores_hyphen_and_space(store, order_id):
    res, matched = store.lookup("get_order_info", {"order_id": order_id})
    assert matched and res["status"] == "ok" and _orders(res) == ["20260101-0000001"]


@pytest.mark.parametrize("phone", ["01011112222", "010-1111-2222", "010 1111 2222"])
def test_order_lookup_by_phone_only(store, phone):
    res, matched = store.lookup("get_order_info", {"buyer_cellphone": phone})
    assert matched and _orders(res) == ["20260102-0000002"]


def test_order_lookup_prefers_order_id_over_phone(store):
    # 주문번호는 1번 기록, 전화번호는 다른 기록 → 주문번호가 맞는 기록
    res, _ = store.lookup("get_order_info", {"order_id": "20260101-0000001", "buyer_cellphone": "01011112222"})
    assert _orders(res) == ["20260101-0000001"]


def test_order_lookup_same_phone_picks_record_with_same_argument_shape(store):
    only_phone, _ = store.lookup("get_order_info", {"buyer_cellphone": "010-3333-4444"})
    assert _orders(only_phone) == ["20260103-0000003", "20260104-0000004"]
    both, _ = store.lookup("get_order_info", {"order_id": "20260103-0000003", "buyer_cellphone": "01033334444"})
    assert _orders(both) == ["20260103-0000003"]


def test_null_argument_is_ignored(store):
    res, matched = store.lookup("get_order_info", {"order_id": None, "buyer_cellphone": "01011112222"})
    assert matched and res["status"] == "ok"


def test_lookup_not_found(store):
    res, matched = store.lookup("get_order_info", {"order_id": "20269999-9999999"})
    assert not matched and res["status"] == "not_found"
    res, matched = store.lookup("get_return_info", {"customer_phone_number": "01000000000"})
    assert not matched and res["status"] == "not_found"


def test_recorded_not_found_result_is_returned_as_is(store):
    res, matched = store.lookup("get_return_info", {"customer_phone_number": "01077778888"})
    assert matched and res == {"status": "not_found", "message": "반품 상품이 없습니다."}


def test_lookup_without_any_id_is_invalid(store):
    assert store.lookup("get_order_info", {})[0]["status"] == "invalid"
    assert store.lookup("get_order_info", {"order_id": "", "buyer_cellphone": None})[0]["status"] == "invalid"
    assert store.lookup("get_return_info", {})[0]["status"] == "invalid"


# ── 접수: 필수 값 → ok / invalid, 기록이 있으면 그 결과 ──
@pytest.mark.parametrize("name,args", [
    ("cancel_order", {"order_id": "20260101-0000001", "reason": "단순 변심"}),
    ("exchange_order", {"order_number": "20260102-0000002", "reason": "파손"}),
    ("return_order", {"order_number": "20260102-0000002", "reason": "단순 변심"}),
    ("request_callback", {"customer_name": "라마바", "phone_number": "01012341234"}),
])
def test_action_ok_when_required_present_even_without_record(store, name, args):
    res, _ = store.lookup(name, args)
    assert res["status"] == "ok"


@pytest.mark.parametrize("name,args", [
    ("cancel_order", {"order_id": "20260101-0000001"}),                 # reason 없음
    ("cancel_order", {"order_id": "", "reason": "단순 변심"}),
    ("return_order", {"reason": "단순 변심"}),
    ("return_order_item", {"return_items": [], "reason": "일부 반품"}),
    ("exchange_order", {"order_number": "20260102-0000002", "reason": "  "}),
    ("request_callback", {"phone_number": "01012341234"}),              # 이름 없음
])
def test_action_invalid_when_required_missing(store, name, args):
    res, matched = store.lookup(name, args)
    assert res["status"] == "invalid" and not matched


def test_action_returns_recorded_result(store):
    res, matched = store.lookup("return_order", {"order_number": "202601030000003", "reason": "다른 사유"})
    assert matched and res["status"] == "already_requested"
    res, matched = store.lookup("return_order_item", {
        "return_items": [{"order_item_code": "20260101-0000001-01", "quantity": 1}], "reason": "일부만"})
    assert matched and res["message"] == "상품 반품 접수 성공"
    res, matched = store.lookup("request_callback", {"customer_name": "가나다", "phone_number": "01099990000"})
    assert matched and res["data"] == {"성공여부": True}


# ── 검색: 가장 가까운 기록 ────────────────────────────
@pytest.mark.parametrize("query,expected", [
    ("가상 관절 영양제", "가상 관절 영양제"),
    ("관절영양제", "가상 관절 영양제"),
    ("눈 영양제", "가상 눈 영양제"),
    ("가상눈영양제 두 개", "가상 눈 영양제"),
])
def test_search_returns_nearest_record(store, query, expected):
    res, matched = store.lookup("search_products", {"query": query})
    assert matched and res["data"]["결과"][0]["제품명"] == expected


def test_search_without_query_is_invalid(store):
    assert store.lookup("search_products", {})[0]["status"] == "invalid"


# ── HTTP ─────────────────────────────────────────────
def test_get_tools_matches_definition_file(base):
    tools = httpx.get(base + "/tools").json()
    raw = json.loads((MINI / "brand" / "tools.json").read_text(encoding="utf-8"))
    assert [t["function"] for t in tools] == raw and all(t["type"] == "function" for t in tools)
    assert len(tools) == 8


def test_unknown_function_is_404(base):
    assert httpx.post(base + "/tools/transfer_to_agent", json={"reason": "x"}).status_code == 404
    assert httpx.post(base + "/nope", json={}).status_code == 404


def test_post_returns_result_and_logs_session(base):
    r = httpx.post(base + "/tools/get_order_info", json={"buyer_cellphone": "010-1111-2222"},
                   headers={"X-Session-Id": "S-1"})
    assert r.status_code == 200 and r.json()["status"] == "ok"
    calls = httpx.get(base + "/calls", params={"session_id": "S-1"}).json()
    assert calls[-1]["name"] == "get_order_info" and calls[-1]["matched"] is True


def test_bad_json_is_400(base):
    r = httpx.post(base + "/tools/get_order_info", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


# ── 실행 인자·기본 경로 ───────────────────────────────
def test_db_flag_and_data_alias():
    p = mock.build_parser()
    assert p.parse_args(["--db", "x.json"]).db == "x.json"
    assert p.parse_args(["--data", "y.json"]).db == "y.json"
    assert p.parse_args([]).host == "0.0.0.0"


def test_env_paths_and_argument_priority(tmp_path):
    env = {mock.DATA_ENV: str(MINI / "mock_db.json"), mock.TOOLS_ENV: str(MINI / "brand" / "tools.json")}
    assert len(mock.load_store(env=env).tools) == 8
    other = tmp_path / "db.json"
    other.write_text("{}", encoding="utf-8")
    s = mock.load_store(other, env=env)                # 인자가 환경변수보다 우선
    assert s.db == {}


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="데이터 파일"):
        mock.load_store(tmp_path / "none.json", MINI / "brand" / "tools.json", env={})
    with pytest.raises(FileNotFoundError, match="함수 정의"):
        mock.load_store(MINI / "mock_db.json", tmp_path / "none.json", env={})


def test_main_exits_cleanly_when_files_missing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        mock.main(["--db", str(tmp_path / "none.json"), "--tools", str(MINI / "brand" / "tools.json")])
    assert "시작할 수 없습니다" in capsys.readouterr().err


# ── 공식 데이터 자체 검사(키트가 있을 때만) ────────────
@pytest.mark.skipif(not (KIT / "mock_db.json").is_file(), reason="공식 키트 없음")
def test_official_db_every_record_returns_its_own_status():
    s = mock.load_store(KIT / "mock_db.json", KIT / "brand" / "tools.json", env={})
    assert len(s.tools) == 8
    bad = []
    for name, rows in s.db.items():
        for i, r in enumerate(rows):
            res, _ = s.lookup(name, r["arguments"])
            if res.get("status") != r["result"].get("status"):
                bad.append((name, i, res.get("status"), r["result"].get("status")))
    assert not bad, bad
