"""AG-T02 Mock API 서버(가상 쇼핑몰 '다이렉트몰') 테스트.

서버를 임의 포트(0)로 띄우고 실제 HTTP로 부른다(로컬 루프백만, 외부 네트워크 없음).
"""
import json
import re
import threading

import httpx
import pytest

from mock_api.server import create_server, load_shop

EXPECTED_TOOLS = {
    "get_member_info", "get_order_list", "get_order_info", "track_delivery",
    "cancel_order", "request_return", "request_exchange", "change_shipping_address",
    "get_refund_status", "get_product_info", "check_stock", "get_coupons",
    "request_callback", "transfer_to_agent",
}
HANGUL = re.compile(r"[가-힣]")


@pytest.fixture(scope="module")
def server():
    srv = create_server(port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture(scope="module")
def base(server):
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


@pytest.fixture(scope="module")
def data():
    return load_shop().data


def call(base, name, args, sid="s-default"):
    r = httpx.post(f"{base}/tools/{name}", json=args, headers={"X-Session-Id": sid})
    return r


def calls(base, sid):
    return httpx.get(f"{base}/calls", params={"session_id": sid}).json()


def first_order(data, status):
    return next(o for o in data["orders"] if o["status"] == status)


# ---------- /tools ----------

def test_tools_lists_14_functions_in_openai_format(base):
    r = httpx.get(f"{base}/tools")
    assert r.status_code == 200
    tools = r.json()
    assert len(tools) == 14
    assert {t["function"]["name"] for t in tools} == EXPECTED_TOOLS
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert set(fn) >= {"name", "description", "parameters"}
        assert HANGUL.search(fn["description"]), fn["name"]
        assert len(fn["description"]) >= 15, fn["name"]
        p = fn["parameters"]
        assert p["type"] == "object"
        assert isinstance(p["required"], list) and p["required"], fn["name"]
        assert set(p["required"]) <= set(p["properties"])
        for pname, schema in p["properties"].items():
            assert schema["type"] == "string", (fn["name"], pname)
            assert HANGUL.search(schema["description"]), (fn["name"], pname)


def test_tools_key_parameters(base):
    tools = {t["function"]["name"]: t["function"]["parameters"] for t in httpx.get(f"{base}/tools").json()}
    assert tools["get_order_info"]["required"] == ["order_id"]
    assert tools["get_member_info"]["required"] == ["phone_number"]
    assert tools["check_stock"]["required"] == ["product_name"]
    assert "option" in tools["check_stock"]["properties"]
    assert set(tools["change_shipping_address"]["required"]) == {"order_id", "new_address"}
    assert tools["transfer_to_agent"]["required"] == ["reason"]


# ---------- 데이터 규모 ----------

def test_mock_data_size_and_formats(data):
    assert len(data["members"]) >= 12
    assert len(data["orders"]) >= 30
    assert len(data["products"]) >= 15
    for m in data["members"]:
        assert re.fullmatch(r"010\d{8}", m["phone_number"])
    statuses = {}
    for o in data["orders"]:
        assert re.fullmatch(r"\d{10}", o["order_id"])
        statuses[o["status"]] = statuses.get(o["status"], 0) + 1
    for s in ["결제완료", "상품준비중", "배송준비", "배송중", "배송완료", "취소완료", "반품접수", "교환접수"]:
        assert statuses.get(s, 0) >= 2, s
    ids = [o["order_id"] for o in data["orders"]]
    assert len(ids) == len(set(ids))
    phones = {m["phone_number"] for m in data["members"]}
    assert all(o["phone_number"] in phones for o in data["orders"])


# ---------- matched=true 경로 + /calls 기록 ----------

def test_defined_input_returns_output_and_records_call(base, data):
    order = data["orders"][0]
    r = call(base, "get_order_info", {"order_id": order["order_id"]}, sid="s-match")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["data"]["order_id"] == order["order_id"]
    assert body["data"]["status"] == order["status"]
    rows = calls(base, "s-match")
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == "s-match"
    assert row["name"] == "get_order_info"
    assert row["arguments"] == {"order_id": order["order_id"]}
    assert row["matched"] is True
    assert isinstance(row["t_response"], float)


def test_calls_filtered_by_session(base, data):
    oid = data["orders"][1]["order_id"]
    call(base, "get_order_info", {"order_id": oid}, sid="s-a")
    call(base, "track_delivery", {"order_id": oid}, sid="s-b")
    call(base, "get_refund_status", {"order_id": oid}, sid="s-b")
    assert [c["name"] for c in calls(base, "s-a")] == ["get_order_info"]
    assert [c["name"] for c in calls(base, "s-b")] == ["track_delivery", "get_refund_status"]


def test_same_input_is_deterministic(base, data):
    oid = first_order(data, "결제완료")["order_id"]
    a = call(base, "cancel_order", {"order_id": oid}).json()
    b = call(base, "cancel_order", {"order_id": oid}).json()
    assert a == b


@pytest.mark.parametrize("name,argf", [
    ("get_member_info", lambda d: {"phone_number": d["members"][0]["phone_number"]}),
    ("get_order_list", lambda d: {"phone_number": d["members"][0]["phone_number"]}),
    ("get_coupons", lambda d: {"phone_number": d["members"][0]["phone_number"]}),
    ("request_callback", lambda d: {"phone_number": d["members"][0]["phone_number"], "preferred_time": "내일 오후 2시"}),
    ("get_order_info", lambda d: {"order_id": d["orders"][0]["order_id"]}),
    ("track_delivery", lambda d: {"order_id": first_order(d, "배송중")["order_id"]}),
    ("cancel_order", lambda d: {"order_id": first_order(d, "결제완료")["order_id"], "reason": "단순 변심"}),
    ("request_return", lambda d: {"order_id": first_order(d, "배송완료")["order_id"], "reason": "사이즈가 안 맞아요"}),
    ("request_exchange", lambda d: {"order_id": first_order(d, "배송완료")["order_id"], "reason": "색상 교환"}),
    ("change_shipping_address", lambda d: {"order_id": first_order(d, "결제완료")["order_id"], "new_address": "서울시 마포구 월드컵로 1"}),
    ("get_refund_status", lambda d: {"order_id": first_order(d, "취소완료")["order_id"]}),
    ("get_product_info", lambda d: {"product_name": d["products"][0]["name"]}),
    ("check_stock", lambda d: {"product_name": d["products"][0]["name"], "option": d["products"][0]["options"][0]["name"]}),
    ("check_stock", lambda d: {"product_name": d["products"][0]["name"]}),
    ("transfer_to_agent", lambda d: {"reason": "고객이 상담원 연결을 요청"}),
])
def test_every_function_has_matched_ok_path(base, data, name, argf):
    sid = f"s-ok-{name}-{len(argf(data))}"
    r = call(base, name, argf(data), sid=sid)
    assert r.status_code == 200
    assert r.json()["status"] == "ok", r.json()
    assert calls(base, sid)[-1]["matched"] is True


def test_free_text_args_do_not_affect_match_or_output(base, data):
    oid = first_order(data, "결제완료")["order_id"]
    a = call(base, "cancel_order", {"order_id": oid, "reason": "단순 변심"}, sid="s-free")
    b = call(base, "cancel_order", {"order_id": oid, "reason": "다른 데서 더 싸게 샀어요"}, sid="s-free")
    c = call(base, "cancel_order", {"order_id": oid}, sid="s-free")
    assert a.json() == b.json() == c.json()
    assert [x["matched"] for x in calls(base, "s-free")] == [True, True, True]


# ---------- 상태 규칙 ----------

@pytest.mark.parametrize("status,expected", [
    ("결제완료", "ok"), ("상품준비중", "ok"), ("배송준비", "ok"),
    ("배송중", "not_allowed"), ("배송완료", "not_allowed"), ("취소완료", "not_allowed"),
    ("반품접수", "not_allowed"), ("교환접수", "not_allowed"),
])
def test_cancel_order_depends_on_status(base, data, status, expected):
    oid = first_order(data, status)["order_id"]
    body = call(base, "cancel_order", {"order_id": oid}).json()
    assert body["status"] == expected, body
    assert HANGUL.search(body["message"])


def test_cancel_in_transit_guides_return(base, data):
    oid = first_order(data, "배송중")["order_id"]
    body = call(base, "cancel_order", {"order_id": oid}).json()
    assert "반품" in body["message"]


def test_return_only_after_delivery(base, data):
    assert call(base, "request_return", {"order_id": first_order(data, "배송중")["order_id"]}).json()["status"] == "not_allowed"
    assert call(base, "request_return", {"order_id": first_order(data, "결제완료")["order_id"]}).json()["status"] == "not_allowed"
    expired = next(o for o in data["orders"] if o["status"] == "배송완료" and o.get("return_window_expired"))
    body = call(base, "request_return", {"order_id": expired["order_id"]}).json()
    assert body["status"] == "not_allowed" and "7일" in body["message"]


def test_address_change_only_before_shipping_prep(base, data):
    ok = call(base, "change_shipping_address", {"order_id": first_order(data, "상품준비중")["order_id"], "new_address": "부산시"}).json()
    no = call(base, "change_shipping_address", {"order_id": first_order(data, "배송중")["order_id"], "new_address": "부산시"}).json()
    assert ok["status"] == "ok" and no["status"] == "not_allowed"


def test_refund_status_without_refund_is_matched(base, data):
    oid = first_order(data, "배송중")["order_id"]
    body = call(base, "get_refund_status", {"order_id": oid}, sid="s-norefund").json()
    assert body["status"] == "no_refund"
    assert calls(base, "s-norefund")[-1]["matched"] is True


# ---------- matched=false 경로 ----------

@pytest.mark.parametrize("name,args", [
    ("get_order_info", {"order_id": "9999999999"}),
    ("get_order_info", {"order_id": "261115023"}),
    ("track_delivery", {"order_id": "0000000000"}),
    ("cancel_order", {"order_id": "1234567890", "reason": "변심"}),
    ("get_member_info", {"phone_number": "01099999999"}),
    ("get_member_info", {"phone_number": "010-1234-5678"}),
    ("get_order_list", {"phone_number": "01000000000"}),
    ("get_coupons", {"phone_number": "01000000000"}),
    ("request_callback", {"phone_number": "01000000000"}),
    ("get_product_info", {"product_name": "없는 상품"}),
    ("check_stock", {"product_name": "없는 상품", "option": "블랙"}),
])
def test_undefined_input_not_found(base, name, args):
    sid = f"s-nf-{name}-{json.dumps(args, ensure_ascii=True)}"
    r = call(base, name, args, sid=sid)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "not_found"
    assert HANGUL.search(body["message"])
    row = calls(base, sid)[-1]
    assert row["matched"] is False and row["arguments"] == args


def test_unknown_option_not_found(base, data):
    p = data["products"][0]
    body = call(base, "check_stock", {"product_name": p["name"], "option": "없는옵션"}, sid="s-opt").json()
    assert body["status"] == "not_found"
    assert calls(base, "s-opt")[-1]["matched"] is False


@pytest.mark.parametrize("name,args", [
    ("get_order_info", {}),
    ("get_order_info", {"order_id": 2611150231}),
    ("get_order_info", {"order_id": "2611150231", "extra": "x"}),
    ("transfer_to_agent", {}),
])
def test_invalid_arguments_not_matched(base, name, args):
    sid = f"s-inv-{name}-{len(args)}-{type(args.get('order_id')).__name__}"
    body = call(base, name, args, sid=sid).json()
    assert body["status"] == "invalid_arguments"
    assert calls(base, sid)[-1]["matched"] is False


# ---------- HTTP 오류 ----------

def test_unknown_function_404(base):
    r = call(base, "launch_rocket", {}, sid="s-404")
    assert r.status_code == 404
    assert calls(base, "s-404") == []


def test_unknown_get_path_404(base):
    assert httpx.get(f"{base}/nope").status_code == 404


def test_invalid_json_400(base):
    r = httpx.post(f"{base}/tools/get_order_info", content=b"{not json", headers={"X-Session-Id": "s-400"})
    assert r.status_code == 400
    r2 = httpx.post(f"{base}/tools/get_order_info", content=b"[1,2]", headers={"X-Session-Id": "s-400"})
    assert r2.status_code == 400
    assert calls(base, "s-400") == []


def test_servers_are_isolated():
    a, b = create_server(port=0), create_server(port=0)
    try:
        assert a.server_address[1] != b.server_address[1]
        assert a.calls is not b.calls
    finally:
        a.server_close(); b.server_close()
