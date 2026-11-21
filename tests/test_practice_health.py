"""AG-T22 건강기능식품 연습판(legacy/practice_health/) 검사.

- 함수 정의·Mock 데이터·매뉴얼·시나리오 40개가 기존 테스트셋 형식(legacy/testset, eval/runner.py)과 맞는다.
- 공식 업무 7종(배송·주문 취소·반품/환불·교환·결제·상품·상담원 연결)과 '섭취 후 몸 불편 → 상담원 연결'이 있다.
- 모든 문항의 expected_tools가 연습 Mock에서 matched가 된다. Mock은 AG-T17 계약의 실제 로더
  (`mock_api.server.shop_from_args(["--data", ..., "--tools", ...])`)로 띄운다. mock_data.json은 Starter Kit 매핑 형식,
  사람이 읽는 원본은 source_data.json(쇼핑몰 형식). 둘 다 generate.py가 만든다.
- 주문번호는 공식 예시("주문번호 1234")처럼 숫자 4자리.
"""
import importlib.util
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

import mock_api.server as mock_server

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "legacy" / "practice_health"
TOOLS = DIR / "tools.json"
DATA = DIR / "source_data.json"
MAPPING = DIR / "mock_data.json"
GEN = DIR / "generate.py"
CASES = DIR / "cases.jsonl"
MANUAL = DIR / "manual.md"
README = DIR / "README.md"

TASK_TYPES = {"배송", "주문취소", "반품환불", "교환", "결제", "상품", "상담원연결"}
CATEGORIES = {"faq", "tool_lookup", "tool_action", "need_info", "out_of_scope", "complaint_handoff", "followup"}
TAGS = {"mid_pause", "filler_words", "connective_end", "digits", "self_correction", "negation",
        "banmal", "long", "short", "caller_known"}
REQUIRED = {"case_id": str, "now": str, "history": list, "customer_utterance": str, "segments": list,
            "voice": str, "env": dict, "category": str, "tags": list, "expected_tools": list,
            "expected_points": list, "must_not": list, "task_type": str}
ENV = {"channel": {"wideband", "phone", "phone_ulaw"}, "noise": {"none", "white", "pink", "babble", "hum"},
       "snr_db": {30, 20, 10, 5}, "gain_db": {0, -12, 6}, "speed": {0.85, 1.0, 1.2}, "packet_loss": {0.0, 0.02}}
PHONE_TOOLS = {"get_member_info", "get_order_list", "request_callback"}
DISCOMFORT_RE = re.compile(r"불편|두드러기|가려|복통|설사|메스꺼|어지러|울렁|붓|아파|토할|두근")
ORDER_ID_RE = re.compile(r"^\d{4}$")
PHONE_RE = re.compile(r"^010\d{8}$")


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@pytest.fixture(scope="module")
def tools():
    return json.loads(TOOLS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cases():
    return _jsonl(CASES)


@pytest.fixture(scope="module")
def shop():
    """AG-T17 계약 경로: python -m mock_api.server --data <mock_data.json> --tools <tools.json>"""
    if not hasattr(mock_server, "shop_from_args"):
        pytest.skip("mock_api.server에 --data/--tools 로더(shop_from_args)가 아직 없다(AG-T17)")
    return mock_server.shop_from_args(["--data", str(MAPPING), "--tools", str(TOOLS)], env={})


# ---------- 파일 ----------

def test_files_exist():
    for p in (TOOLS, DATA, MAPPING, CASES, MANUAL, README, GEN):
        assert p.exists(), p


def test_manual_covers_tasks_and_health_handoff():
    m = MANUAL.read_text(encoding="utf-8")
    for word in ("배송", "취소", "반품", "환불", "교환", "결제", "무통장", "섭취", "상담원"):
        assert word in m, word
    assert re.search(r"몸.{0,10}불편|이상 반응|부작용", m), "섭취 후 이상 반응 인계 규칙 없음"
    assert re.search(r"의학|의료|진단", m), "의료 조언 금지 규칙 없음"


# ---------- tools.json ----------

def test_tools_schema(tools):
    assert isinstance(tools, list) and tools
    names = [t["name"] for t in tools]
    assert len(names) == len(set(names)), names
    for t in tools:
        assert set(t) >= {"name", "description", "parameters"}, t.get("name")
        assert re.fullmatch(r"[a-z_]+", t["name"]), t["name"]
        assert t["description"].strip(), t["name"]
        p = t["parameters"]
        assert p["type"] == "object" and isinstance(p["properties"], dict), t["name"]
        assert set(p["required"]) <= set(p["properties"]), t["name"]
        for k, v in p["properties"].items():
            assert v.get("type") == "string" and v.get("description"), (t["name"], k)


def test_tools_cover_seven_tasks(tools):
    names = {t["name"] for t in tools}
    need = {"track_delivery", "cancel_order", "request_return", "get_refund_status", "request_exchange",
            "get_payment_info", "check_deposit", "get_product_info", "transfer_to_agent", "get_order_list",
            "get_order_info"}
    assert need <= names, need - names


# ---------- mock_data.json ----------

def test_source_data_shape(data):
    assert {"shop", "members", "products", "orders", "refunds", "payments"} <= set(data)


def test_mock_loads_as_mapping_with_all_tools(shop, tools, data):
    assert shop.mapping, "mock_data.json은 Starter Kit 매핑 형식이어야 한다"
    assert shop.tool_names == {t["name"] for t in tools}
    m = json.loads(MAPPING.read_text(encoding="utf-8"))
    assert set(m) == shop.tool_names
    order_fns = {t["name"] for t in tools if t["parameters"]["required"][:1] == ["order_id"]}
    for name in order_fns:   # 모든 주문 × 주문 함수가 정의돼 있다
        got = {r["arguments"]["order_id"] for r in m[name]}
        assert got == {o["order_id"] for o in data["orders"]}, name
    for o in data["orders"]:
        for name in order_fns:
            res, matched = shop.lookup(name, {"order_id": o["order_id"], **(
                {"new_address": "서울"} if name == "change_shipping_address" else {})})
            assert matched and res["status"] in {"ok", "not_allowed", "no_refund"}, (name, o["order_id"], res)
    res, matched = shop.lookup("track_delivery", {"order_id": "9999"})
    assert not matched and res["status"] == "not_found"
    res, matched = shop.lookup("transfer_to_agent", {"reason": "아무 사유"})
    assert matched


def test_payment_results(shop, data):
    pending = [p["order_id"] for p in data["payments"] if p["status"] == "입금대기"]
    res, matched = shop.lookup("check_deposit", {"order_id": pending[0]})
    assert matched and res["data"]["deposit_confirmed"] is False
    card = next(p["order_id"] for p in data["payments"] if p["method"] == "신용카드")
    res, matched = shop.lookup("check_deposit", {"order_id": card})
    assert matched and res["status"] == "not_allowed"
    res, _ = shop.lookup("get_payment_info", {"order_id": pending[0]})
    assert res["data"]["account_number"] and res["data"]["deposit_deadline"]
    res, _ = shop.lookup("get_product_info", {"product_name": data["products"][0]["name"]})
    assert {"how_to_take", "ingredients", "cautions", "price"} <= set(res["data"])


def test_files_in_sync_with_generator(tools, data):
    spec = importlib.util.spec_from_file_location("practice_health_generate", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = mod.build_data()
    assert mod.TOOLS == tools and src == data, "generate.py를 다시 실행"
    assert mod.build_mapping(src) == json.loads(MAPPING.read_text(encoding="utf-8")), "generate.py를 다시 실행"
    assert mod.build_cases() == _jsonl(CASES), "generate.py를 다시 실행"


def test_order_ids_are_four_digits(data):
    ids = [o["order_id"] for o in data["orders"]]
    assert len(ids) == len(set(ids))
    for i in ids:
        assert ORDER_ID_RE.match(i), i
    assert "1234" in ids, "공식 예시 주문번호 1234"
    for r in data["refunds"] + data["payments"]:
        assert r["order_id"] in ids, r["order_id"]


def test_members_and_order_refs(data):
    phones = {m["phone_number"] for m in data["members"]}
    for p in phones:
        assert PHONE_RE.match(p), p
    products = {p["name"]: p for p in data["products"]}
    for o in data["orders"]:
        assert o["phone_number"] in phones, o["order_id"]
        for it in o["items"]:
            assert it["product_name"] in products, (o["order_id"], it["product_name"])
            assert it["option"] in {x["name"] for x in products[it["product_name"]]["options"]}, o["order_id"]
        assert o["total_amount"] == o["product_amount"] + o["shipping_fee"], o["order_id"]
        assert o["product_amount"] == sum(i["unit_price"] * i["quantity"] for i in o["items"]), o["order_id"]


def test_health_products(data):
    ps = data["products"]
    assert len(ps) >= 15
    for p in ps:
        for k in ("name", "price", "category", "description", "options", "ingredients", "how_to_take", "cautions"):
            assert p.get(k), (p.get("name"), k)
        assert isinstance(p["price"], int) and p["price"] > 0
        assert "섭취" in p["how_to_take"] or "드세요" in p["how_to_take"] or "드십시오" in p["how_to_take"], p["name"]


def test_payments(data):
    orders = {o["order_id"]: o for o in data["orders"]}
    pays = data["payments"]
    assert {p["order_id"] for p in pays} == set(orders), "모든 주문에 결제 정보"
    methods = Counter(p["method"] for p in pays)
    assert methods["무통장입금"] >= 3 and methods["신용카드"] >= 3, methods
    pending = [p for p in pays if p["status"] == "입금대기"]
    assert len(pending) >= 2
    for p in pays:
        o = orders[p["order_id"]]
        assert p["method"] == o["payment_method"], p["order_id"]
        assert p["amount"] == o["total_amount"], p["order_id"]
        if p["method"] == "무통장입금":
            for k in ("bank", "account_number", "account_holder", "depositor_name", "deposit_deadline"):
                assert p.get(k), (p["order_id"], k)
        if p["status"] == "입금대기":
            assert o["status"] == "입금대기", p["order_id"]


# ---------- cases.jsonl ----------

def test_cases_schema(cases):
    assert len(cases) == 40, len(cases)
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        cid = c.get("case_id")
        assert re.fullmatch(r"H\d{4}", cid), cid
        for k, t in REQUIRED.items():
            assert isinstance(c.get(k), t), (cid, k)
        assert "customer_profile" in c, cid
        assert c["category"] in CATEGORIES, cid
        assert set(c["tags"]) <= TAGS, (cid, set(c["tags"]) - TAGS)
        assert c["task_type"] in TASK_TYPES, (cid, c["task_type"])
        datetime.strptime(c["now"], "%Y-%m-%d %H:%M")
        assert c["now"].startswith("2026-11-"), cid
        assert 1 <= len(c["expected_points"]) <= 3 and len(c["must_not"]) <= 2, cid
        e = c["env"]
        for k, allowed in ENV.items():
            assert e[k] in allowed, (cid, k, e[k])
        assert isinstance(e["lead_ms"], int) and 200 <= e["lead_ms"] <= 1500, cid
        assert re.fullmatch(r"gemini:[A-Z][a-z]+", c["voice"]) and c["voice"] != "gemini:Kore", cid


def test_segments_and_speech_text(cases):
    for c in cases:
        segs = c["segments"]
        assert segs and segs[-1]["pause_after_ms"] == 0, c["case_id"]
        for s in segs[:-1]:
            assert 300 <= s["pause_after_ms"] <= 1200, c["case_id"]
        norm = lambda s: re.sub(r"\s+", "", s)
        assert norm("".join(s["text"] for s in segs)) == norm(c["customer_utterance"]), c["case_id"]
        assert not re.search(r"[0-9]", c["customer_utterance"]), c["case_id"]
        if {"mid_pause", "connective_end"} & set(c["tags"]):
            assert len(segs) >= 2, c["case_id"]


def test_all_task_types_present(cases):
    cnt = Counter(c["task_type"] for c in cases)
    for t in TASK_TYPES:
        assert cnt[t] >= 3, (t, cnt[t])


def test_body_discomfort_handoff_present(cases):
    hits = [c for c in cases if c["task_type"] == "상담원연결"
            and DISCOMFORT_RE.search(c["customer_utterance"])
            and any(t["name"] == "transfer_to_agent" for t in c["expected_tools"])]
    assert len(hits) >= 2, [c["case_id"] for c in hits]
    for c in hits:
        assert any(re.search(r"의학|진단|복용|섭취를 계속|괜찮다", m) for m in c["must_not"]), c["case_id"]


def test_multi_turn_cases(cases):
    multi = [c for c in cases if c["history"]]
    assert len(multi) >= 2
    for c in multi:
        for h in c["history"]:
            assert h["role"] in ("agent", "customer") and h["text"].strip(), c["case_id"]
        assert c["history"][-1]["role"] == "agent", c["case_id"]


def test_caller_profile_matches_member(cases, data):
    members = {m["phone_number"]: m for m in data["members"]}
    for c in cases:
        p = c["customer_profile"]
        assert ("caller_known" in c["tags"]) == (p is not None), c["case_id"]
        if p:
            assert set(p) == {"name", "phone", "email"} and members[p["phone"]]["name"] == p["name"], c["case_id"]


def test_expected_order_ids_four_digits(cases):
    for c in cases:
        for t in c["expected_tools"]:
            if "order_id" in t["arguments"] and not t.get("expect_not_found"):
                assert ORDER_ID_RE.match(t["arguments"]["order_id"]), (c["case_id"], t)


def test_expected_tools_match_practice_data(cases, shop, tools):
    names = {t["name"] for t in tools}
    for c in cases:
        for call in c["expected_tools"]:
            assert set(call) <= {"name", "arguments", "expect_not_found"}, c["case_id"]
            assert call["name"] in names, (c["case_id"], call["name"])
            res, matched = shop.lookup(call["name"], call["arguments"])
            if call.get("expect_not_found"):
                assert not matched and res["status"] == "not_found", (c["case_id"], call, res)
            else:
                assert matched, (c["case_id"], call, res)


def test_caller_owns_orders(cases, data):
    orders = {o["order_id"]: o for o in data["orders"]}
    for c in cases:
        p = c["customer_profile"]
        if not p:
            continue
        for call in c["expected_tools"]:
            a = call["arguments"]
            if call.get("expect_not_found"):
                continue
            if call["name"] in PHONE_TOOLS:
                assert a["phone_number"] == p["phone"], c["case_id"]
            if "order_id" in a:
                assert orders[a["order_id"]]["phone_number"] == p["phone"], c["case_id"]


def test_category_rules(cases):
    for c in cases:
        names = {t["name"] for t in c["expected_tools"]}
        if c["category"] in ("faq", "out_of_scope"):
            assert not names, c["case_id"]
        if c["category"] == "complaint_handoff":
            assert names & {"transfer_to_agent", "request_callback"}, c["case_id"]
        if c["category"] in ("tool_lookup", "tool_action"):
            assert names, c["case_id"]
        if c["category"] == "followup":
            assert c["history"], c["case_id"]




# ---------- 음성 생성 경로 ----------

def test_build_testset_accepts_paths():
    from eval import build_testset as B
    a = B.parse_args(["--cases", str(CASES), "--out", str(DIR / "audio"), "--cache", str(DIR / "cache")])
    assert Path(a.cases) == CASES and Path(a.out) == DIR / "audio" and Path(a.cache) == DIR / "cache"
    assert len(B.load_cases(a.cases)) == 40


def test_practice_audio_is_gitignored():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "legacy/practice_health/audio/" in gi and "legacy/practice_health/cache/" in gi
