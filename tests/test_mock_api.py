"""AG-T36 공식 규격 Mock 서버 테스트(directon-2026-voice-cs SPEC.md 「모의 함수 서버」).

데이터는 합성 소형 DB(가짜 값, AD14: 공식 mock_db.json은 저장소에 넣지 않는다).
서버를 임의 포트(0)로 띄우고 실제 HTTP로 부른다(로컬 루프백만, 외부 네트워크 없음).
"""
import json
import threading
from pathlib import Path

import httpx
import pytest

from mock_api.server import DEFAULT_TOOLS_PATH, Store, build_parser, create_server, load_store

OFFICIAL_TOOLS = {
    "get_order_info", "get_return_info", "search_products", "cancel_order",
    "return_order", "return_order_item", "exchange_order", "request_callback",
}

ORDER_A = {"status": "ok", "data": {"orders": [{"주문번호": "20990101-0000001", "배송상태": "배송중"}]},
           "message": "주문 조회 완료"}
ORDER_B = {"status": "ok", "data": {"orders": [{"주문번호": "20990202-0000002", "배송상태": "배송완료"}]},
           "message": "주문 조회 완료"}
ORDER_C = {"status": "ok", "data": {"orders": [{"주문번호": "20990303-0000003", "배송상태": "결제완료"}]},
           "message": "주문 조회 완료"}

DB = {
    "get_order_info": [
        {"arguments": {"order_id": "20990101-0000001", "buyer_cellphone": "01000000001"}, "result": ORDER_A},
        {"arguments": {"order_id": "20990202-0000002", "buyer_cellphone": None}, "result": ORDER_B},
        {"arguments": {"order_id": None, "buyer_cellphone": "010 0000 0003"}, "result": ORDER_C},
    ],
    "get_return_info": [
        {"arguments": {"customer_phone_number": "010-0000-0001"},
         "result": {"status": "ok", "data": {"반품": [{"상태": "수거완료"}]}, "message": "반품 조회 완료"}},
    ],
    "search_products": [
        {"arguments": {"query": "하루온 비타민 2개"}, "result": {"status": "ok", "data": {"id": "vita"}}},
        {"arguments": {"query": "하루온 유산균 효능"}, "result": {"status": "ok", "data": {"id": "probio"}}},
    ],
    "return_order": [
        {"arguments": {"order_number": "20990101-0000001", "reason": "단순 변심"},
         "result": {"status": "already_requested", "message": "이미 접수됨"}},
    ],
    "request_callback": [
        {"arguments": {"customer_name": "김가짜", "phone_number": "010-0000-0009"},
         "result": {"status": "ok", "data": {"접수": True}, "message": "전화 요청 등록"}},
    ],
}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("mock") / "mock_db.json"
    p.write_text(json.dumps(DB, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def store(db_path):
    return load_store(db_path)


@pytest.fixture(scope="module")
def base(store):
    srv = create_server(port=0, store=store)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    host, port = srv.server_address[:2]
    yield f"http://{host}:{port}"
    srv.shutdown()
    srv.server_close()


def post(base, name, args, sid="s-default"):
    return httpx.post(f"{base}/tools/{name}", json=args, headers={"X-Session-Id": sid})


# ---------- /tools ----------

def test_tools_are_the_8_official_functions_in_openai_format(base):
    r = httpx.get(f"{base}/tools")
    assert r.status_code == 200
    tools = r.json()
    assert all(t["type"] == "function" for t in tools)
    assert {t["function"]["name"] for t in tools} == OFFICIAL_TOOLS


def test_tools_match_official_copy_exactly(base):
    official = json.loads(Path(DEFAULT_TOOLS_PATH).read_text(encoding="utf-8"))
    assert [t["function"] for t in httpx.get(f"{base}/tools").json()] == official


# ---------- 조회 함수 ----------

def test_order_info_by_order_id(store):
    res, matched = store.lookup("get_order_info", {"order_id": "20990202-0000002"})
    assert matched and res == ORDER_B


def test_order_info_ignores_hyphen_and_space(store):
    assert store.lookup("get_order_info", {"order_id": "20990202 0000002"})[0] == ORDER_B
    assert store.lookup("get_order_info", {"buyer_cellphone": "010-0000-0003"})[0] == ORDER_C
    assert store.lookup("get_order_info", {"buyer_cellphone": "01000000003"})[0] == ORDER_C


def test_order_info_by_phone_only_matches_record_that_has_both(store):
    res, matched = store.lookup("get_order_info", {"buyer_cellphone": "010-0000-0001", "order_id": None})
    assert matched and res == ORDER_A


def test_order_info_prefers_record_matching_both_ids(store):
    # 주문번호는 B, 전화는 A의 것 → 둘 다 맞는 기록이 없으면 주문번호 우선
    res, _ = store.lookup("get_order_info", {"order_id": "20990202-0000002", "buyer_cellphone": "01000000001"})
    assert res == ORDER_B


def test_order_info_not_found(store):
    res, matched = store.lookup("get_order_info", {"order_id": "20990909-0000009"})
    assert res["status"] == "not_found" and not matched


def test_order_info_without_any_id_is_invalid(store):
    assert store.lookup("get_order_info", {})[0]["status"] == "invalid"
    assert store.lookup("get_order_info", {"order_id": "", "buyer_cellphone": None})[0]["status"] == "invalid"


def test_return_info_by_phone_and_missing(store):
    assert store.lookup("get_return_info", {"customer_phone_number": "01000000001"})[0]["status"] == "ok"
    assert store.lookup("get_return_info", {"customer_phone_number": "01099999999"})[0]["status"] == "not_found"
    assert store.lookup("get_return_info", {})[0]["status"] == "invalid"


# ---------- 접수 함수 ----------

def test_write_with_required_values_is_ok_even_without_record(store):
    res, matched = store.lookup("cancel_order", {"order_id": "20990303-0000003", "reason": "단순 변심"})
    assert res["status"] == "ok" and not matched
    res, _ = store.lookup("exchange_order", {"order_number": "20990303-0000003", "reason": "파손"})
    assert res["status"] == "ok"
    res, _ = store.lookup("return_order_item", {
        "return_items": [{"order_item_code": "20990303-0000003-01", "quantity": 1}], "reason": "일부 반품"})
    assert res["status"] == "ok"


def test_write_missing_required_is_invalid(store):
    assert store.lookup("cancel_order", {"order_id": "20990303-0000003"})[0]["status"] == "invalid"
    assert store.lookup("return_order", {"order_number": "", "reason": "x"})[0]["status"] == "invalid"
    assert store.lookup("return_order_item", {"return_items": [], "reason": "x"})[0]["status"] == "invalid"
    assert store.lookup("request_callback", {"customer_name": "김가짜"})[0]["status"] == "invalid"


def test_write_returns_recorded_result_when_ids_match(store):
    res, matched = store.lookup("return_order", {"order_number": "2099-0101-0000001", "reason": "다른 말로 한 사유"})
    assert matched and res["status"] == "already_requested"
    res, matched = store.lookup("request_callback", {"customer_name": "김가짜", "phone_number": "01000000009"})
    assert matched and res["data"] == {"접수": True}


# ---------- 상품 검색 ----------

def test_search_returns_closest_record(store):
    assert store.lookup("search_products", {"query": "하루온 유산균"})[0]["data"]["id"] == "probio"
    assert store.lookup("search_products", {"query": "비타민 두 개"})[0]["data"]["id"] == "vita"
    assert store.lookup("search_products", {"query": "하루온비타민2개"})[0]["data"]["id"] == "vita"


def test_search_without_query_is_invalid(store):
    assert store.lookup("search_products", {})[0]["status"] == "invalid"


def test_search_with_no_records_is_not_found(tmp_path):
    p = tmp_path / "db.json"
    p.write_text("{}", encoding="utf-8")
    assert load_store(p).lookup("search_products", {"query": "아무거나"})[0]["status"] == "not_found"


# ---------- HTTP ----------

def test_http_call_returns_result_and_records_session(base):
    r = post(base, "get_order_info", {"order_id": "20990202-0000002"}, sid="s-abc")
    assert r.status_code == 200 and r.json() == ORDER_B
    rec = httpx.get(f"{base}/calls", params={"session_id": "s-abc"}).json()
    assert len(rec) == 1 and rec[0]["name"] == "get_order_info" and rec[0]["matched"] is True
    assert rec[0]["arguments"] == {"order_id": "20990202-0000002"}


def test_unknown_function_is_404(base):
    assert post(base, "transfer_to_agent", {"reason": "x"}).status_code == 404
    assert post(base, "get_order_list", {"phone_number": "01000000001"}).status_code == 404


def test_bad_json_is_400(base):
    r = httpx.post(f"{base}/tools/get_order_info", content=b"{not json", headers={"X-Session-Id": "s"})
    assert r.status_code == 400


def test_missing_db_file_starts_empty(tmp_path):
    s = load_store(tmp_path / "nope.json")
    assert isinstance(s, Store) and s.lookup("get_order_info", {"order_id": "1"})[0]["status"] == "not_found"


def test_cli_accepts_db_and_tools_and_binds_all_interfaces_by_default():
    a = build_parser().parse_args(["--port", "8100", "--db", "x.json"])
    assert a.db == "x.json" and a.port == 8100 and a.host == "0.0.0.0"
