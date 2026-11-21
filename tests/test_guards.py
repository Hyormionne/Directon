"""함수 호출 가드(AG-T33) 테스트. 번호는 모두 지어낸 값이다."""
from __future__ import annotations

import copy

import pytest

from agent import guards
from agent.guards import (ACTION_FUNCS, LOOKUP_FUNCS, ORDER_KEYS, PHONE_KEYS, SEARCH_FUNCS, GuardContext,
                          Verdict, check, format_order_id, known_numbers)

ORDER = "20261105-0012345"
ORDER_D = "202611050012345"
ORDER_NEAR = "20261105-0012346"      # ORDER와 한 자리 차이
ORDER_OTHER = "20261107-0067890"
ORDER16 = "4812345678901234"
ITEM = "20261105-0012345-01"
PHONE = "01055501234"
PHONE_OTHER = "01077709876"

_STR = {"type": "string"}
_NULLABLE = {"type": ["string", "null"]}
_REFUND = {"type": ["object", "null"], "properties": {"bank_name": _STR, "account_no": _STR, "holder": _STR}}


def _tool(name: str, props: dict, required: list[str]) -> dict:
    return {"name": name, "description": name,
            "parameters": {"type": "object", "properties": props, "required": required}}


TOOLS = [
    _tool("get_order_info", {"order_id": _NULLABLE, "buyer_cellphone": _NULLABLE}, []),
    _tool("get_return_info", {"customer_phone_number": _STR}, ["customer_phone_number"]),
    _tool("search_products", {"query": _STR}, ["query"]),
    _tool("cancel_order", {"order_id": _STR, "customer_phone_number": _NULLABLE, "reason": _STR},
          ["order_id", "reason"]),
    _tool("return_order", {"order_number": _STR, "reason": _STR, "pickup_address": _NULLABLE,
                           "refund_account": _REFUND}, ["order_number", "reason"]),
    _tool("return_order_item",
          {"return_items": {"type": "array", "items": {
              "type": "object",
              "properties": {"order_item_code": _STR, "quantity": {"type": "integer"}},
              "required": ["order_item_code", "quantity"]}},
           "reason": _STR, "pickup_address": _NULLABLE, "refund_account": _REFUND},
          ["return_items", "reason"]),
    _tool("exchange_order", {"order_number": _STR, "reason": _STR, "pickup_address": _NULLABLE},
          ["order_number", "reason"]),
    _tool("request_callback", {"customer_name": _STR, "phone_number": _STR},
          ["customer_name", "phone_number"]),
]

LOOKUP_RESULT = {"status": "ok", "message": "조회했습니다", "data": {"orders": [{
    "주문번호": ORDER, "상세주문번호": ITEM, "주문일자": "2026-11-05 10:11:12", "제품명": "테스트 상품",
    "수량": 2, "배송상태": "배송완료", "송장번호": "555566667777", "출고예정일자": None}]}}
LOOKUP_HISTORY = [
    {"role": "customer", "text": "주문 확인해 주세요"},
    {"role": "tool_call", "name": "get_order_info", "arguments": {"buyer_cellphone": "010 5550 1234"}},
    {"role": "tool_result", "name": "get_order_info", "result": LOOKUP_RESULT},
    {"role": "agent", "text": "주문 한 건 확인됩니다."},
]


def ctx(**kw) -> GuardContext:
    kw.setdefault("tools", TOOLS)
    return GuardContext(**kw)


def passed(v: Verdict) -> bool:
    return v.blocked is None and v.reason == ""


def blocked(v: Verdict, reason: str) -> bool:
    return (v.reason == reason and isinstance(v.blocked, dict) and v.blocked.get("status") == "blocked"
            and bool(v.blocked.get("message")))


# ── 상수·보조 함수 ────────────────────────────────────
def test_constants():
    assert ORDER_KEYS == ("order_id", "order_number")
    assert PHONE_KEYS == ("buyer_cellphone", "customer_phone_number", "phone_number")
    assert LOOKUP_FUNCS == {"get_order_info", "get_return_info"}
    assert SEARCH_FUNCS == {"search_products"}
    assert ACTION_FUNCS == {"cancel_order", "return_order", "return_order_item", "exchange_order",
                            "request_callback"}
    assert {t["name"] for t in TOOLS} == LOOKUP_FUNCS | SEARCH_FUNCS | ACTION_FUNCS


@pytest.mark.parametrize("raw, want", [
    (ORDER_D, ORDER),
    (ORDER, ORDER),
    ("20261105 0012345", ORDER),
    (ORDER16, ORDER16),                 # 16자리는 그대로
    ("2026110500123", "2026110500123"),  # 자리 수가 모자라면 지어 맞추지 않는다
    ("", ""),
])
def test_format_order_id(raw, want):
    assert format_order_id(raw) == want


# ── 1. 정의에 없는 함수 ───────────────────────────────
@pytest.mark.parametrize("tools", [
    TOOLS,
    [{"type": "function", "function": t} for t in TOOLS],   # OpenAI 래핑
    [],                                                     # 정의를 못 받았으면 기본 8개로 본다
])
def test_undefined_function_blocked(tools):
    v = check("transfer_to_agent", {"reason": "상담원 연결"}, ctx(tools=tools))
    assert blocked(v, "undefined")
    assert "없습니다" in v.blocked["message"] and "말로" in v.blocked["message"]
    assert passed(check("search_products", {"query": "테스트 상품"}, ctx(tools=tools)))


# ── 2. 번호 인수 정리 ─────────────────────────────────
@pytest.mark.parametrize("args, want", [
    ({"order_id": None, "buyer_cellphone": "010-5550-1234"}, {"buyer_cellphone": PHONE}),
    ({"order_id": "", "buyer_cellphone": PHONE}, {"buyer_cellphone": PHONE}),
    ({"order_id": "  ", "buyer_cellphone": PHONE}, {"buyer_cellphone": PHONE}),
    ({"order_id": "모름", "buyer_cellphone": PHONE}, {"buyer_cellphone": PHONE}),
    ({"order_id": ORDER_D, "buyer_cellphone": None}, {"order_id": ORDER}),
])
def test_null_keys_dropped(args, want):
    v = check("get_order_info", args, ctx())
    assert passed(v) and v.args == want


@pytest.mark.parametrize("raw", [
    "010-5550-1234", "010 5550 1234", "010.5550.1234", "공일공 오오오공 일이삼사", "공일공오오오공일이삼사",
    "+82 10-5550-1234", 1055501234, PHONE,
])
def test_phone_normalized(raw):
    v = check("get_return_info", {"customer_phone_number": raw}, ctx())
    assert passed(v) and v.args == {"customer_phone_number": PHONE}


@pytest.mark.parametrize("raw, want", [
    (ORDER_D, ORDER),
    ("20261105 0012345", ORDER),
    ("이공이육일일공오 다시 공공일이삼사오", ORDER),
    ("이공이육일일공오공공일이삼사오번", ORDER),
    (ORDER16, ORDER16),
    (int(ORDER16), ORDER16),
])
def test_order_id_formatted(raw, want):
    v = check("get_order_info", {"order_id": raw}, ctx())
    assert passed(v) and v.args == {"order_id": want}


def test_item_code_formatted_and_extra_args_untouched():
    args = {"return_items": [{"order_item_code": "20261105001234501", "quantity": 1, "memo": " 그대로 "}],
            "reason": "단순 변심", "pickup_address": " 서울시 가상구 1-2 ", "extra": {"a": [1, 2]}}
    v = check("return_order_item", args, ctx(history=LOOKUP_HISTORY))
    assert passed(v)
    assert v.args["return_items"] == [{"order_item_code": ITEM, "quantity": 1, "memo": " 그대로 "}]
    assert v.args["pickup_address"] == " 서울시 가상구 1-2 " and v.args["extra"] == {"a": [1, 2]}


# ── 3. 아는 번호에 맞춤 ───────────────────────────────
HIST_ORDER = [{"role": "customer", "text": f"네 {ORDER} 입니다"}]
HIST_TWO = HIST_ORDER + [{"role": "customer", "text": "20261105-0012347 도 있어요"}]


@pytest.mark.parametrize("label, name, args, kw, key, want", [
    ("고객 말(STT)이 한 자리 틀림 → 이전 대화 번호로", "cancel_order",
     {"order_id": ORDER_NEAR, "reason": "변심"}, {"history": HIST_ORDER, "user_text": f"{ORDER_NEAR} 취소요"},
     "order_id", ORDER),
    ("모델이 숫자를 섞음 → 조회 결과 번호로", "return_order",
     {"order_number": "20261105-0012354", "reason": "불량"}, {"history": LOOKUP_HISTORY},
     "order_number", ORDER),
    ("조회 함수도 맞춘다", "get_order_info",
     {"order_id": ORDER_NEAR}, {"history": HIST_ORDER}, "order_id", ORDER),
    ("전화번호는 발신자 번호로", "request_callback",
     {"customer_name": "가나다", "phone_number": "010-5550-1284"}, {"caller": {"name": "가나다", "phone": PHONE}},
     "phone_number", PHONE),
    ("모델이 틀리게 옮김 → 고객 말의 번호로", "get_order_info",
     {"order_id": ORDER_NEAR}, {"user_text": "주문번호 이공이육일일공오 다시 공공일이삼사오"}, "order_id", ORDER),
    ("후보가 둘이면 그대로", "get_order_info",
     {"order_id": ORDER_NEAR}, {"history": HIST_TWO}, "order_id", ORDER_NEAR),
    ("많이 다르면 그대로", "get_order_info",
     {"order_id": ORDER_OTHER}, {"history": HIST_ORDER}, "order_id", ORDER_OTHER),
    ("정확히 아는 번호는 그대로", "get_order_info",
     {"order_id": "20261105-0012347"}, {"history": HIST_TWO}, "order_id", "20261105-0012347"),
    ("글 속 번호는 모양으로 종류를 정한다(0으로 시작하는 11자리는 전화번호)", "get_return_info",
     {"customer_phone_number": "01055501235"}, {"history": [{"role": "customer", "text": "번호 01055501234"}]},
     "customer_phone_number", PHONE),
    ("종류가 다르면 그대로(주문번호를 전화번호에 맞추지 않는다)", "get_order_info",
     {"order_id": "01055501235"},
     {"history": [{"role": "tool_call", "name": "get_order_info", "arguments": {"buyer_cellphone": PHONE}}]},
     "order_id", "01055501235"),
])
def test_snap(label, name, args, kw, key, want):
    v = check(name, args, ctx(**kw))
    assert v.args[key] == want, label
    assert passed(v), label


def test_item_code_not_snapped():
    """-01과 -02는 서로 다른 상품이라 맞추지 않고 막는다."""
    args = {"return_items": [{"order_item_code": "20261105-0012345-02", "quantity": 1}], "reason": "변심"}
    v = check("return_order_item", args, ctx(history=LOOKUP_HISTORY))
    assert blocked(v, "ungrounded")
    assert v.args["return_items"][0]["order_item_code"] == "20261105-0012345-02"


# ── 4. 필수 인수 ──────────────────────────────────────
GROUNDED = {"user_text": f"주문번호 {ORDER} 전화번호 {PHONE}", "history": LOOKUP_HISTORY}


@pytest.mark.parametrize("name, args, words", [
    ("cancel_order", {"order_id": ORDER}, ["사유"]),
    ("cancel_order", {"order_id": ORDER, "reason": "  "}, ["사유"]),
    ("cancel_order", {"reason": "변심"}, ["주문번호"]),
    ("cancel_order", {"order_id": None, "reason": None}, ["주문번호", "사유"]),
    ("return_order", {"order_number": ORDER}, ["사유"]),
    ("return_order", {"reason": "불량"}, ["주문번호"]),
    ("exchange_order", {"order_number": ORDER}, ["사유"]),
    ("exchange_order", {"order_number": "모름", "reason": "사이즈"}, ["주문번호"]),
    ("return_order_item", {"return_items": [{"order_item_code": ITEM, "quantity": 1}]}, ["사유"]),
    ("return_order_item", {"reason": "변심"}, ["반품할 상품"]),
    ("return_order_item", {"return_items": [], "reason": "변심"}, ["반품할 상품"]),
    ("return_order_item", {"return_items": [{"quantity": 1}], "reason": "변심"}, ["상세주문번호"]),
    ("return_order_item", {"return_items": [{"order_item_code": ITEM}], "reason": "변심"}, ["수량"]),
    ("request_callback", {"phone_number": PHONE}, ["이름"]),
    ("request_callback", {"customer_name": "가나다"}, ["전화번호"]),
    ("request_callback", {"customer_name": "", "phone_number": ""}, ["이름", "전화번호"]),
    ("get_return_info", {}, ["전화번호"]),
    ("search_products", {"query": ""}, ["검색어"]),
])
def test_required_missing(name, args, words):
    v = check(name, args, ctx(**GROUNDED))
    assert blocked(v, "missing"), (name, args)
    msg = v.blocked["message"]
    assert all(w in msg for w in words), msg
    assert "처리됐다고 말하지" in msg


@pytest.mark.parametrize("args", [{}, {"order_id": None, "buyer_cellphone": ""}])
def test_order_lookup_without_any_number(args):
    v = check("get_order_info", args, ctx(user_text="배송 언제 와요"))
    assert blocked(v, "missing") and "주문번호 또는 전화번호" in v.blocked["message"]
    v = check("get_order_info", args, ctx(user_text="배송 언제 와요", caller={"phone": PHONE}))
    assert passed(v) and v.args == {"buyer_cellphone": PHONE}      # 발신자 번호로 조회


def test_required_from_defaults_when_tools_empty():
    assert blocked(check("cancel_order", {"order_id": ORDER}, ctx(tools=[], user_text=ORDER)), "missing")
    assert blocked(check("get_order_info", {}, ctx(tools=[])), "missing")
    assert passed(check("cancel_order", {"order_id": ORDER, "reason": "변심"}, ctx(tools=[], user_text=ORDER)))


# ── 5. 접수 함수의 번호 근거 ──────────────────────────
@pytest.mark.parametrize("label, name, args, kw", [
    ("고객 말(아라비아 숫자)", "cancel_order", {"order_id": ORDER, "reason": "변심"},
     {"user_text": f"주문번호 {ORDER} 취소해 주세요"}),
    ("고객 말(끊어 읽은 숫자)", "cancel_order", {"order_id": ORDER, "reason": "변심"},
     {"user_text": "2026 1105 0012 345 취소해 주세요"}),
    ("고객 말(한국어로 읽음)", "cancel_order", {"order_id": ORDER, "reason": "변심"},
     {"user_text": "주문번호 이공이육일일공오 다시 공공일이삼사오 취소해 주세요"}),
    ("고객 말(16자리와 전화번호가 붙음)", "return_order", {"order_number": ORDER16, "reason": "불량"},
     {"user_text": f"{ORDER16} {PHONE}"}),
    ("고객 말(다른 숫자와 섞임)", "return_order", {"order_number": ORDER, "reason": "불량"},
     {"user_text": f"-주문일 : 2026-11-05\n-주문번호 : {ORDER}\n47,700원 반품할게요"}),
    ("이전 대화의 고객 말", "exchange_order", {"order_number": ORDER, "reason": "사이즈"}, {"history": HIST_ORDER}),
    ("이전 대화의 조회 결과", "exchange_order", {"order_number": ORDER, "reason": "사이즈"},
     {"history": LOOKUP_HISTORY}),
    ("이전 대화의 함수 인수", "return_order", {"order_number": ORDER, "reason": "불량"},
     {"history": [{"role": "tool_call", "name": "get_order_info", "arguments": {"order_id": ORDER}}]}),
    ("이번 턴 조회 결과(전화번호로 조회해 나온 주문)", "cancel_order", {"order_id": ORDER_D, "reason": "변심"},
     {"user_text": "방금 주문한 거 취소해 주세요", "caller": {"name": "가나다", "phone": PHONE},
      "results": [LOOKUP_RESULT], "called": ["get_order_info"]}),
    ("이번 턴 결과(이름·인수와 함께 싼 모양)", "cancel_order", {"order_id": ORDER, "reason": "변심"},
     {"results": [{"name": "get_order_info", "arguments": {"buyer_cellphone": PHONE}, "result": LOOKUP_RESULT}]}),
    ("콜백: 발신자 전화번호", "request_callback", {"customer_name": "가나다", "phone_number": "010-5550-1234"},
     {"user_text": "상담원 전화 주세요", "caller": {"name": "가나다", "phone": "010-5550-1234", "email": None}}),
    ("콜백: 고객 말", "request_callback", {"customer_name": "가나다", "phone_number": PHONE},
     {"user_text": "가나다 공일공 오오오공 일이삼사입니다"}),
    ("콜백: 고객 말(받침 없는 숫자 뒤 입니다)", "request_callback",
     {"customer_name": "가나다", "phone_number": "01055501239"},
     {"user_text": "공일공, 오오오공, 일이삼구입니다."}),
    ("콜백: 이전 대화의 조회 인수", "request_callback", {"customer_name": "가나다", "phone_number": PHONE},
     {"history": LOOKUP_HISTORY}),
])
def test_action_grounded(label, name, args, kw):
    v = check(name, args, ctx(**kw))
    assert passed(v), (label, v)


@pytest.mark.parametrize("label, name, args, kw, word", [
    ("아는 번호가 없음", "cancel_order", {"order_id": ORDER, "reason": "변심"},
     {"user_text": "주문 취소해 주세요"}, "주문번호"),
    ("다른 주문번호", "return_order", {"order_number": ORDER_OTHER, "reason": "불량"},
     {"history": LOOKUP_HISTORY}, "주문번호"),
    ("콜백 전화번호를 지어냄", "request_callback", {"customer_name": "가나다", "phone_number": PHONE_OTHER},
     {"user_text": "전화 주세요", "caller": {"name": "가나다", "phone": PHONE}}, "전화번호"),
    ("발신자 정보 없음", "request_callback", {"customer_name": "가나다", "phone_number": PHONE},
     {"user_text": "전화 주세요", "caller": None}, "전화번호"),
    ("모르는 상세주문번호", "return_order_item",
     {"return_items": [{"order_item_code": "20261107-0067890-01", "quantity": 1}], "reason": "변심"},
     {"history": LOOKUP_HISTORY}, "상세주문번호"),
])
def test_action_ungrounded_blocked(label, name, args, kw, word):
    v = check(name, args, ctx(**kw))
    assert blocked(v, "ungrounded"), (label, v)
    assert word in v.blocked["message"] and "확인" in v.blocked["message"]


@pytest.mark.parametrize("name, args, want", [
    ("get_order_info", {"order_id": ORDER_OTHER}, {"order_id": ORDER_OTHER}),
    ("get_order_info", {"buyer_cellphone": PHONE_OTHER}, {"buyer_cellphone": PHONE_OTHER}),
    ("get_return_info", {"customer_phone_number": "010-7770-9876"}, {"customer_phone_number": PHONE_OTHER}),
    ("search_products", {"query": "테스트 상품 2개"}, {"query": "테스트 상품 2개"}),
])
def test_lookup_and_search_not_blocked_for_grounding(name, args, want):
    v = check(name, args, ctx(user_text="배송 언제 와요"))
    assert passed(v) and v.args == want


def test_optional_ungrounded_phone_dropped_not_blocked():
    """필수가 아닌 전화번호가 근거 없으면 막지 않고 뺀다(주문번호는 확인된 번호)."""
    args = {"order_id": ORDER, "customer_phone_number": PHONE_OTHER, "reason": "변심"}
    v = check("cancel_order", args, ctx(history=LOOKUP_HISTORY))
    assert passed(v) and v.args == {"order_id": ORDER, "reason": "변심"}
    args["customer_phone_number"] = "010-5550-1234"         # 이전 조회에 쓴 번호는 남긴다
    v = check("cancel_order", args, ctx(history=LOOKUP_HISTORY))
    assert passed(v) and v.args == {"order_id": ORDER, "customer_phone_number": PHONE, "reason": "변심"}


def test_return_order_item_with_looked_up_item_code():
    args = {"return_items": [{"order_item_code": ITEM, "quantity": 1}], "reason": "하나만 반품"}
    for kw in ({"history": LOOKUP_HISTORY}, {"results": [LOOKUP_RESULT]}, {"user_text": f"{ITEM} 이거 반품"}):
        v = check("return_order_item", args, ctx(**kw))
        assert passed(v) and v.args == args, kw
    assert blocked(check("return_order_item", args, ctx(user_text="하나만 반품할게요")), "ungrounded")


def test_grounding_switch_off(monkeypatch):
    args = {"order_id": ORDER, "reason": "변심"}
    assert blocked(check("cancel_order", args, ctx()), "ungrounded")
    monkeypatch.setenv("DIRECTON_GUARD_GROUNDING", "0")
    assert passed(check("cancel_order", args, ctx()))
    assert blocked(check("cancel_order", {"order_id": ORDER}, ctx()), "missing")          # 다른 규칙은 그대로
    assert blocked(check("cancel_order", args, ctx(called=["return_order"])), "second_action")
    monkeypatch.setenv("DIRECTON_GUARD_GROUNDING", "1")
    assert blocked(check("cancel_order", args, ctx()), "ungrounded")


# ── 6. 한 턴에 접수 함수는 하나 ───────────────────────
@pytest.mark.parametrize("name, args, called, reason", [
    ("return_order", {"order_number": ORDER, "reason": "불량"}, ["cancel_order"], "second_action"),
    ("cancel_order", {"order_id": ORDER, "reason": "변심"}, ["get_order_info", "cancel_order"], "second_action"),
    ("request_callback", {"customer_name": "가나다", "phone_number": PHONE}, ["exchange_order"], "second_action"),
    ("cancel_order", {"order_id": ORDER, "reason": "변심"}, ["get_order_info", "search_products"], ""),
    ("get_order_info", {"order_id": ORDER}, ["cancel_order"], ""),
    ("get_order_info", {"order_id": ORDER}, ["get_order_info"], ""),
    ("search_products", {"query": "테스트"}, ["request_callback"], ""),
])
def test_one_action_per_turn(name, args, called, reason):
    v = check(name, args, ctx(called=called, **GROUNDED))
    assert v.reason == reason
    assert (v.blocked is None) == (reason == "")
    if reason:
        assert blocked(v, reason) and "이미" in v.blocked["message"]


def test_rule_order_missing_before_ungrounded_before_second_action():
    c = ctx(called=["cancel_order"], user_text="취소해 주세요")
    assert check("return_order", {"order_number": ORDER}, c).reason == "missing"
    assert check("return_order", {"order_number": ORDER, "reason": "불량"}, c).reason == "ungrounded"
    assert check("no_such", {}, c).reason == "undefined"


# ── 7. 잘못된 입력, 원본 보존 ─────────────────────────
@pytest.mark.parametrize("name, args", [
    ("get_order_info", None), ("get_order_info", "20261105-0012345"), ("get_order_info", [1, 2]),
    ("get_order_info", {"order_id": {"a": 1}, "buyer_cellphone": [PHONE]}),
    ("get_order_info", {"order_id": 3.5, "buyer_cellphone": True}),
    ("cancel_order", {"order_id": ["x"], "reason": 3}),
    ("return_order_item", {"return_items": "아무거나", "reason": "변심"}),
    ("return_order_item", {"return_items": [None, 3, "x"], "reason": "변심"}),
    ("return_order_item", {"return_items": {"order_item_code": ITEM, "quantity": 1}, "reason": "변심"}),
    ("return_order_item", {"return_items": '[{"order_item_code": "20261105001234501", "quantity": 1}]',
                           "reason": "변심"}),
    ("request_callback", {"customer_name": None, "phone_number": {"n": 1}}),
    (None, {}), ("", None),
])
def test_malformed_args_do_not_raise(name, args):
    before = copy.deepcopy(args)
    v = check(name, args, ctx(history=LOOKUP_HISTORY))
    assert isinstance(v, Verdict) and isinstance(v.args, dict)
    assert v.blocked is None or v.blocked["status"] == "blocked"
    assert args == before


def test_malformed_context_does_not_raise():
    bad = GuardContext(tools=[None, 3, {"x": 1}, {"function": "x"}, TOOLS[0], {"name": "cancel_order"}],
                       user_text=None, history=[None, 3, "주문번호 20261105-0012345", {"role": "customer"},
                                                {"role": "tool_result", "result": "문자열 결과"},
                                                {"role": "tool_call", "arguments": None}],
                       caller={"phone": None}, results=None, called=None)
    assert passed(check("get_order_info", {"order_id": ORDER_D}, bad))
    assert passed(check("cancel_order", {"order_id": ORDER}, bad))       # 정의에 required가 없으면 필수 검사 없음
    assert known_numbers(bad)["orders"] == [ORDER_D]
    weird = GuardContext(tools="tools", history="기록", caller="발신자", results="결과", called="cancel_order")
    assert check("get_order_info", {"order_id": ORDER}, weird).args == {"order_id": ORDER}
    assert isinstance(known_numbers(weird), dict)


def test_check_fails_open_on_internal_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(guards, "_collect", boom)
    args = {"order_id": ORDER_D, "reason": "변심"}
    v = check("cancel_order", args, ctx())
    assert passed(v) and v.args == args and v.args is not args


def test_input_not_mutated():
    args = {"order_id": "이공이육일일공오 다시 공공일이삼사오", "customer_phone_number": None, "reason": "변심",
            "nested": {"k": [1]}}
    items = {"return_items": [{"order_item_code": "20261105001234501", "quantity": 1}], "reason": "변심"}
    c = ctx(history=copy.deepcopy(LOOKUP_HISTORY), results=[copy.deepcopy(LOOKUP_RESULT)], called=["get_order_info"])
    before = copy.deepcopy((args, items, c))
    v1 = check("cancel_order", args, c)
    v2 = check("return_order_item", items, c)
    assert (args, items, c) == before
    assert v1.args == {"order_id": ORDER, "reason": "변심", "nested": {"k": [1]}}
    assert v2.args["return_items"][0]["order_item_code"] == ITEM
    assert v1.args["nested"] is not args["nested"] and v2.args["return_items"] is not items["return_items"]


# ── 아는 번호 모으기 ──────────────────────────────────
def test_known_numbers_sources_and_kinds():
    kn = known_numbers(ctx(
        user_text=f"{ORDER16} 공일공 칠칠칠공 구팔칠육이에요",
        history=LOOKUP_HISTORY + [{"role": "agent", "text": f"주문번호 {ORDER_OTHER} 로 접수되었습니다."}],
        results=[{"status": "ok", "data": {"결과": [{"order_number": "20261109-0000011", "연락처": "010-3330-4444",
                                                     "결제일자": "2026-11-09 09:08:07"}]}}],
        caller={"name": "가나다", "phone": "010-2220-3333", "email": None}))
    assert set(kn) == {"orders", "phones", "items"}
    assert {ORDER_D, "202611070067890", "202611090000011", ORDER16} <= set(kn["orders"])
    assert {PHONE, PHONE_OTHER, "01033304444", "01022203333"} <= set(kn["phones"])
    assert kn["items"] == ["20261105001234501"]
    every = kn["orders"] + kn["phones"] + kn["items"]
    assert all(isinstance(n, str) and n.isdigit() for n in every)
    assert "555566667777" not in every                     # 송장번호
    assert not any(n.startswith("20261105101112") or n.startswith("20261109090807") for n in every)   # 날짜


def test_known_numbers_empty_context():
    assert known_numbers(ctx()) == {"orders": [], "phones": [], "items": []}
    assert known_numbers(GuardContext(tools=[])) == {"orders": [], "phones": [], "items": []}


# ── 형식에 안 맞는 번호는 믿는 출처의 번호로(AG-T33 실측 뒤 추가) ──
def _ctx_fb(**kw):
    from agent.guards import GuardContext
    return GuardContext(tools=[], **kw)


def test_fallback_garbled_phone_uses_caller_phone_for_lookup():
    from agent.guards import check
    v = check("get_return_info", {"customer_phone_number": "515110736"},
              _ctx_fb(user_text="515110736 반품 회수해 갔다네요", caller={"phone": "010-1234-5678"}))
    assert v.blocked is None and v.args["customer_phone_number"] == "01012345678"


def test_fallback_prefers_phone_from_history_over_caller():
    from agent.guards import check
    hist = [{"role": "customer", "text": "연락처는 010 2222 3333 입니다"}]
    v = check("get_return_info", {"customer_phone_number": "5162273"},
              _ctx_fb(user_text="5162273", history=hist, caller={"phone": "01012345678"}))
    assert v.args["customer_phone_number"] == "01022223333"


def test_fallback_keeps_valid_new_phone():
    from agent.guards import check
    v = check("get_return_info", {"customer_phone_number": "010 9999 8888"},
              _ctx_fb(user_text="010 9999 8888 입니다", caller={"phone": "01012345678"}))
    assert v.args["customer_phone_number"] == "01099998888"


def test_fallback_garbled_phone_without_trusted_number_is_kept():
    from agent.guards import check
    v = check("get_return_info", {"customer_phone_number": "515110736"}, _ctx_fb(user_text="515110736"))
    assert v.blocked is None and v.args["customer_phone_number"] == "515110736"


def test_order_lookup_adds_caller_phone_and_replaces_garbled_order_from_history():
    from agent.guards import check
    hist = [{"role": "agent", "text": "주문번호 20260924-0035429 로 확인됩니다"}]
    v = check("get_order_info", {"order_id": "20679292003428"},
              _ctx_fb(user_text="주문 확인 20679292003428", history=hist, caller={"phone": "01012345678"}))
    assert v.args == {"order_id": "20260924-0035429", "buyer_cellphone": "01012345678"}


def test_order_lookup_valid_order_kept_and_phone_added():
    from agent.guards import check
    v = check("get_order_info", {"order_id": "20260101-0000001"},
              _ctx_fb(user_text="20260101-0000001", caller={"phone": "01012345678"}))
    assert v.args == {"order_id": "20260101-0000001", "buyer_cellphone": "01012345678"}


def test_action_garbled_order_snaps_only_when_single_trusted_order():
    from agent.guards import check
    one = [{"role": "tool_result", "name": "get_order_info",
            "result": {"status": "ok", "data": {"orders": [{"주문번호": "20260101-0000001"}]}}}]
    v = check("return_order", {"order_number": "2026010001", "reason": "단순 변심"},
              _ctx_fb(user_text="2026010001 반품해 주세요", history=one))
    assert v.blocked is None and v.args["order_number"] == "20260101-0000001"
    two = one + [{"role": "agent", "text": "다른 주문 20260202-0000002 도 있어요"}]
    v = check("return_order", {"order_number": "2026010001", "reason": "단순 변심"},
              _ctx_fb(user_text="2026010001 반품해 주세요", history=two))
    assert v.args["order_number"] == "2026010001"      # 후보가 둘이면 고르지 않는다


def test_callback_phone_is_not_replaced_by_caller_phone():
    from agent.guards import check
    v = check("request_callback", {"customer_name": "가나다", "phone_number": "50051824029"},
              _ctx_fb(user_text="50051824029 가나다", caller={"phone": "01012345678"}))
    assert v.args["phone_number"] == "50051824029"
