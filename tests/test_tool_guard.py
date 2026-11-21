"""AG-T45 함수 호출 안전장치(agent/tool_guard.py + 응답기 배선).

공식 측정(docs/14_공식측정_결과보고서.md 6절 1~3)의 세 가지 실패를 막는다.
1. 전체 반품인데 return_order_item → 주문 조회 결과로 전체임이 확인되면 return_order로 바꾼다(일부 반품 표현이 있으면 그대로).
2. 접수 함수를 부르지 않고 "접수해 두었습니다"라고 말함 → 그 문장은 내보내지 않고 교정 지시로 한 번 다시 생성한다.
3. 함수 호출 문자열·JSON이 음성으로 나감 → 내보내지 않고, 알려진 함수 호출이면 실제 호출로 바꾼다.
외부 호출 없음(가짜 LLM·TTS·함수).
"""
import asyncio
import json
from pathlib import Path

import pytest

from agent import catalog as catalog_mod
from agent.config import Settings
from agent.core.types import LLMEvent, Message, SessionInfo, ToolCall
from agent.prompts import build_system, fixed_prompt
from agent.responder import Responder
from agent.tool_guard import (FALLBACK_TEXT, fix_full_return, is_claim, is_leak, order_items,
                              parse_leaked_calls, registration_tools)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = json.loads((ROOT / "mock_api" / "data" / "tools.json").read_text(encoding="utf-8"))
TOOLS = [{"type": "function", "function": t} for t in OFFICIAL]
SCHEMAS = {t["name"]: t["parameters"] for t in OFFICIAL}
NAMES = [t["name"] for t in OFFICIAL]

TWO_ITEMS = {"status": "ok", "data": {"orders": [
    {"주문번호": "20260929-0004550", "상세주문번호": "20260929-0004550-01", "제품명": "이눌린 배변 가드", "수량": 1},
    {"주문번호": "20260929-0004550", "상세주문번호": "20260929-0047714-02", "제품명": "간 가드 맥스2", "수량": 1}]},
    "message": "주문 조회 완료"}
ONE_ITEM_QTY4 = {"status": "ok", "data": {"orders": [
    {"주문번호": "20261001-0060343", "상세주문번호": "20261001-0060343-01", "제품명": "간 가드 맥스2", "수량": 4}]},
    "message": "주문 조회 완료"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("DIRECTON_CATALOG_PATH", "DIRECTON_MOCK_DATA", "DIRECTON_CONFIRM_ORDER_ID", "DIRECTON_REPAIR_ORDER_ID",
              "DIRECTON_TOOL_TEMPLATE"):
        monkeypatch.delenv(k, raising=False)
    catalog_mod.reset_default_catalog()
    yield
    catalog_mod.reset_default_catalog()


def system_with(history=None, caller=None):
    info = SessionInfo("S", now="2026-10-07 (수) 10:23", caller=caller, history=history or [])
    return build_system(info, "네, 고객님.", manual="매뉴얼 본문")


def order_history(result, customer="전부 반품해 주세요"):
    return [{"role": "customer", "text": "반품하고 싶어요"},
            {"role": "tool_call", "name": "get_order_info", "arguments": {"order_id": None, "buyer_cellphone": "01047775671"}},
            {"role": "tool_result", "name": "get_order_info", "result": result},
            {"role": "agent", "text": "반품 사유를 알려 주시겠어요?"},
            {"role": "customer", "text": customer}]


def item_call(items, **extra):
    return ToolCall("c1", "return_order_item", {"return_items": items, "reason": "효과가 없어요", **extra})


# ── 1. 전체 반품 판별 ─────────────────────────────────────
def test_registration_tools_from_definitions():
    assert registration_tools(NAMES) == {"cancel_order", "return_order", "return_order_item",
                                         "exchange_order", "request_callback"}


def test_order_items_from_history_system_line():
    items = order_items(system_with(order_history(TWO_ITEMS)), [])
    assert items["20260929-0004550"] == {"20260929000455001": 1, "20260929004771402": 1}


def test_order_items_from_tool_message():
    msgs = [Message("user", "반품할게요"),
            Message("assistant", "", tool_calls=[ToolCall("t1", "get_order_info", {"buyer_cellphone": "01047775671"})]),
            Message("tool", json.dumps(TWO_ITEMS, ensure_ascii=False), tool_call_id="t1", name="get_order_info")]
    assert "20260929-0004550" in order_items("", msgs)


def test_full_return_item_call_becomes_return_order():
    system = system_with(order_history(TWO_ITEMS, "2개 주문건 모두 환불신청합니다"))
    tc = item_call([{"order_item_code": "20260929-0004550-01", "quantity": 1},
                    {"order_item_code": "20260929-0047714-02", "quantity": 1}],
                   refund_account={"bank_name": "우리은행", "account_no": "123", "holder": "김가짜"}, pickup_address=None)
    fixed = fix_full_return(tc, system, [Message("user", "2개 주문건 모두 환불신청합니다")], SCHEMAS)
    assert fixed is not None
    assert fixed.id == "c1" and fixed.name == "return_order"
    assert fixed.arguments == {"order_number": "20260929-0004550", "reason": "효과가 없어요",
                               "refund_account": {"bank_name": "우리은행", "account_no": "123", "holder": "김가짜"}}


def test_full_quantity_single_item_becomes_return_order():
    system = system_with(order_history(ONE_ITEM_QTY4, "다 환불입니다 1+1이였으면 8개가와야지 4개만 와서요"))
    tc = item_call([{"order_item_code": "20261001-0060343-01", "quantity": 4}])
    fixed = fix_full_return(tc, system, [Message("user", "다 환불입니다 1+1이였으면 8개가와야지 4개만 와서요")], SCHEMAS)
    assert fixed is not None and fixed.name == "return_order"
    assert fixed.arguments["order_number"] == "20261001-0060343"


def test_true_partial_item_subset_stays():
    system = system_with(order_history(TWO_ITEMS))
    tc = item_call([{"order_item_code": "20260929-0047714-02", "quantity": 1}])
    assert fix_full_return(tc, system, [Message("user", "간 가드만 반품할게요")], SCHEMAS) is None


def test_true_partial_quantity_stays():
    system = system_with(order_history(ONE_ITEM_QTY4))
    tc = item_call([{"order_item_code": "20261001-0060343-01", "quantity": 2}])
    assert fix_full_return(tc, system, [Message("user", "환불해 주세요")], SCHEMAS) is None


def test_customer_said_partial_stays_even_if_covering():
    """공식 P0147-07: 상품 1개 주문이라도 고객이 '일부반품'이라고 하면 return_order_item이 정답."""
    one = {"status": "ok", "data": {"orders": [
        {"주문번호": "20261002-0048338", "상세주문번호": "20261002-0048338-01", "제품명": "이눌린", "수량": 1}]}}
    system = system_with(order_history(one, "두개씩이틀을 먹었지만 아무런반응이없어요"))
    tc = item_call([{"order_item_code": "20261002-0048338-01", "quantity": 1}])
    assert fix_full_return(tc, system, [Message("user", "일부반품 요제가먹은게있으니까요")], SCHEMAS) is None


def test_unknown_order_info_stays():
    tc = item_call([{"order_item_code": "20260929-0004550-01", "quantity": 1}])
    assert fix_full_return(tc, system_with(), [Message("user", "전부 반품할게요")], SCHEMAS) is None


def test_other_calls_untouched():
    tc = ToolCall("c1", "return_order", {"order_number": "20260929-0004550", "reason": "효과 없음"})
    assert fix_full_return(tc, system_with(order_history(TWO_ITEMS)), [], SCHEMAS) is None


# ── 2. 접수 주장 판별 ─────────────────────────────────────
@pytest.mark.parametrize("text", [
    "반품 접수가 완료되었습니다.", "상담원 전화 요청을 남겨 두었습니다.", "접수해 두었습니다.",
    "취소 처리해 드렸어요.", "요청해 드렸어요.", "반품 신청이 완료됐어요.", "교환 접수 해드렸습니다.",
    "말씀하신 내용으로 접수했습니다."])
def test_claim_positive(text):
    assert is_claim(text)


@pytest.mark.parametrize("text", [
    "반품 접수해 드릴까요?", "접수해 드릴게요.", "이미 접수되었는지 확인해 볼게요.", "주문하신 제품은 배송이 완료되었어요.",
    "취소가 안 됐어요.", "접수되면 문자로 알려 드려요.", "결제가 완료되었어요.", "이미 취소된 주문이에요.",
    "반품 사유를 말씀해 주시겠어요?", "반품 완료 후 환불까지 2~3일 걸려요."])
def test_claim_negative(text):
    assert not is_claim(text)


# ── 3. 호출 문자열 판별·변환 ─────────────────────────────────
@pytest.mark.parametrize("text", [
    'searchproducts({"query": "간 가드"})', '{"order_id": "20260929-0004550"}',
    'functions.get_order_info({"buyer_cellphone": "01047775671"})', "search_products 호출할게요."])
def test_leak_positive(text):
    assert is_leak(text, NAMES)


@pytest.mark.parametrize("text", [
    "간 가드 맥스2는 하루 1포 드시면 돼요.", "하루온 Joint Flex 제품이에요.", "주문번호 20260929-0004550 확인했어요.",
    "회수는 3~4일 안에 방문해요 (영업일 기준)."])
def test_leak_negative(text):
    assert not is_leak(text, NAMES)


def test_parse_leaked_calls():
    calls = parse_leaked_calls('search_products({"query": "간 가드 섭취 방법"})', NAMES)
    assert [(c.name, c.arguments) for c in calls] == [("search_products", {"query": "간 가드 섭취 방법"})]
    calls = parse_leaked_calls('functions.searchproducts({"query": "유산균"})', NAMES)
    assert [(c.name, c.arguments) for c in calls] == [("search_products", {"query": "유산균"})]
    assert parse_leaked_calls('{"결과": "확인"}', NAMES) == []
    assert parse_leaked_calls('unknown_tool({"a": 1})', NAMES) == []


def test_prompt_rules():
    p = fixed_prompt()
    assert "상세주문번호" in p
    assert "일부만" in p and "return_order_item" in p
    assert "글자로 쓰지" in p


# ── 응답기 배선 ──────────────────────────────────────────
class ScriptLLM:
    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.calls: list[dict] = []

    async def stream(self, system, messages, tools):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        for ev in self.rounds[min(len(self.calls) - 1, len(self.rounds) - 1)]:
            await asyncio.sleep(0)
            yield ev


class RecTTS:
    def __init__(self):
        self.requests: list[str] = []

    async def synthesize(self, text):
        self.requests.append(text)
        yield b"\x10\x27" * 8


async def _sink(pcm):
    return None


def text(*chunks):
    return [LLMEvent("text", text=c) for c in chunks] + [LLMEvent("done")]


def call(name, args, cid="c1"):
    return [LLMEvent("tool_call", tool_call=ToolCall(cid, name, args)), LLMEvent("done")]


def make(llm, results=None):
    tts = RecTTS()
    seen: list[tuple[str, dict]] = []

    async def call_tool(name, args):
        seen.append((name, args))
        return (results or {}).get(name, {"status": "ok", "message": "접수 성공"})

    r = Responder(llm, tts, call_tool, tools=TOOLS, settings=Settings())
    return r, tts, seen


def run(r, system, msgs):
    return asyncio.run(r.respond(system, msgs, _sink))


def test_responder_rewrites_full_return():
    system = system_with(order_history(TWO_ITEMS, "2개 주문건 모두 환불신청합니다"))
    llm = ScriptLLM(call("return_order_item", {"return_items": [
        {"order_item_code": "20260929-0004550-01", "quantity": 1},
        {"order_item_code": "20260929-0047714-02", "quantity": 1}], "reason": "효과 없음"}),
        text("반품 접수가 완료되었어요."))
    r, tts, seen = make(llm)
    spoken = run(r, system, [Message("user", "2개 주문건 모두 환불신청합니다")])
    assert [n for n, _ in seen] == ["return_order"]
    assert seen[0][1]["order_number"] == "20260929-0004550"
    assert llm.calls[1]["messages"][-2].tool_calls[0].name == "return_order"   # 기록도 바뀐 호출로
    assert spoken == ["반품 접수가 완료되었어요."]


def test_responder_false_claim_regenerates_with_tool():
    llm = ScriptLLM(text("네, 반품 접수해 두었습니다."),
                    call("return_order", {"order_number": "20260929-0004550", "reason": "효과 없음"}),
                    text("반품 접수가 완료되었어요."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(), [Message("user", "20260929-0004550 주문 효과가 없어서 전체 반품해 주세요")])
    assert "네, 반품 접수해 두었습니다." not in tts.requests
    assert [n for n, _ in seen] == ["return_order"]
    assert "접수" in llm.calls[1]["system"] and llm.calls[1]["system"] != llm.calls[0]["system"]
    assert llm.calls[1]["tools"]                                    # 다시 생성할 때 함수를 부를 수 있다
    assert spoken == ["반품 접수가 완료되었어요."]


def test_responder_claim_after_successful_registration_passes():
    llm = ScriptLLM(call("request_callback", {"customer_name": "김가짜", "phone_number": "01047775671"}),
                    text("상담원 전화 요청을 남겨 두었습니다. 곧 연락드릴게요."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(), [Message("user", "김가짜고 01047775671로 전화 주세요")])
    assert spoken == ["상담원 전화 요청을 남겨 두었습니다.", "곧 연락드릴게요."]
    assert len(llm.calls) == 2


def test_responder_claim_after_failed_registration_is_blocked():
    llm = ScriptLLM(call("request_callback", {"customer_name": "", "phone_number": "01047775671"}),
                    text("요청을 남겨 두었습니다."), text("성함을 말씀해 주시겠어요?"))
    r, tts, seen = make(llm, {"request_callback": {"status": "invalid", "message": "필수 값 누락"}})
    spoken = run(r, system_with(), [Message("user", "01047775671로 전화 주세요")])
    assert "요청을 남겨 두었습니다." not in tts.requests
    assert spoken == ["성함을 말씀해 주시겠어요?"]


def test_responder_repeated_false_claim_falls_back():
    llm = ScriptLLM(text("접수해 두었습니다."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(), [Message("user", "반품해 주세요")])
    assert not any("접수해 두었습니다" in t for t in tts.requests)
    assert spoken == [FALLBACK_TEXT]
    assert seen == []


def test_responder_claim_with_prior_registration_in_history_passes():
    hist = [{"role": "customer", "text": "반품해 주세요"},
            {"role": "tool_call", "name": "return_order", "arguments": {"order_number": "20260929-0004550", "reason": "효과 없음"}},
            {"role": "tool_result", "name": "return_order", "result": {"status": "ok", "message": "주문 반품 접수 성공"}},
            {"role": "agent", "text": "반품 접수가 완료되었습니다."}]
    llm = ScriptLLM(text("네, 반품은 이미 접수되었어요."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(hist), [Message("user", "접수된 거 맞죠?")])
    assert spoken == ["네, 반품은 이미 접수되었어요."]
    assert len(llm.calls) == 1


def test_responder_leaked_call_becomes_tool_call():
    llm = ScriptLLM(text('search_products({"query": "간 가드 섭취 방법"})'), text("하루 한 포 드시면 돼요."))
    r, tts, seen = make(llm, {"search_products": {"status": "ok", "data": {"결과": []}}})
    spoken = run(r, system_with(), [Message("user", "간 가드 어떻게 먹어요?")])
    assert seen == [("search_products", {"query": "간 가드 섭취 방법"})]
    assert spoken == ["하루 한 포 드시면 돼요."]
    assert not any("search" in t or "{" in t for t in tts.requests)


def test_responder_leaked_json_regenerates():
    llm = ScriptLLM(text('{"결과": "확인", "상태": "배송중"}'), text("현재 배송 중이에요."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(), [Message("user", "배송 어디쯤이에요?")])
    assert spoken == ["현재 배송 중이에요."]
    assert not any("{" in t for t in tts.requests)


def test_responder_leak_split_across_sentences_never_voiced():
    llm = ScriptLLM(text('lookup({"note": "확인 중. ', '잠시"}) ', "네, 확인해 볼게요."), text("배송 중이에요."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(), [Message("user", "배송 어디쯤이에요?")])
    assert not any("{" in t or "}" in t or "잠시\"" in t for t in tts.requests)
    assert "네, 확인해 볼게요." in spoken


def test_responder_normal_text_untouched():
    llm = ScriptLLM(text("배송은 내일 도착해요. ", "다른 문의 있으시면 말씀해 주세요."))
    r, tts, seen = make(llm)
    spoken = run(r, system_with(), [Message("user", "언제 와요?")])
    assert spoken == ["배송은 내일 도착해요.", "다른 문의 있으시면 말씀해 주세요."]
    assert len(llm.calls) == 1 and seen == []


def test_responder_claim_backed_by_status_lookup_passes():
    """반품 현황 조회 결과가 '반품접수'면 "이미 반품 접수되어 있어요"는 사실이다(다시 생성하지 않는다)."""
    llm = ScriptLLM(call("get_return_info", {"customer_phone_number": "01047775671"}),
                    text("이미 반품 접수되어 있어요."))
    r, tts, seen = make(llm, {"get_return_info": {"status": "ok", "data": {"현재상태": "반품접수"}}})
    spoken = run(r, system_with(), [Message("user", "01047775671 반품 어떻게 됐어요?")])
    assert spoken == ["이미 반품 접수되어 있어요."]
    assert len(llm.calls) == 2
