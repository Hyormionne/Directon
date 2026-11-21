"""답 후보(agent.drafts): 결과별 선택·빈칸 채우기, 응답기가 결과 뒤 LLM 재호출 없이 후보로 답하는지."""
from __future__ import annotations

from agent import drafts as D
from agent.config import Settings
from agent.core.types import LLMEvent, Message, ToolCall
from agent.responder import Responder

ORDER = {"status": "ok", "data": {"orders": [
    {"주문번호": "20260924-0035429", "제품명": "🔒비밀특가🔒 하루온 간 가드 맥스2", "배송상태": "배송완료",
     "배송완료일자": "2026-10-01"}]}}
ANSWERS = {"answers": [
    {"when": "배송상태=배송중", "text": "지금 배송 중이에요.", "rank": 1},
    {"when": "배송상태=배송완료", "text": "{제품명}은 {배송완료일자}에 배송 완료됐어요.", "rank": 2},
    {"when": "status=not_found", "text": "그 번호로는 주문이 안 보여요.", "rank": 3}]}


def test_choose_fills_slots_and_cleans_values():
    drafts = D.parse(ANSWERS)
    d, text = D.choose(drafts, ORDER)
    assert text == "하루온 간 가드 맥스2은 10월 1일에 배송 완료됐어요."
    assert D.choose(drafts, {"status": "not_found"})[1] == "그 번호로는 주문이 안 보여요."
    assert [x.text for x in D.presynth_targets(drafts, 3)] == ["지금 배송 중이에요.", "그 번호로는 주문이 안 보여요."]


def test_no_match_or_missing_slot_returns_none():
    drafts = D.parse({"answers": [{"when": "배송상태=배송완료", "text": "{출고예정일자} 출고예요."}]})
    assert D.choose(drafts, ORDER) is None                       # 결과에 없는 빈칸
    assert D.choose(D.parse(ANSWERS), {"status": "ok", "data": {"orders": [{"배송상태": "입금전"}]}}) is None
    mixed = {"status": "ok", "data": {"orders": [{"배송상태": "배송중"}, {"배송상태": "배송완료"}]}}
    assert D.choose(D.parse(ANSWERS), mixed) is None             # 상품마다 상태가 다르면 고르지 않는다


class _LLM:
    def __init__(self):
        self.calls = 0

    async def stream(self, system, messages, tools):
        self.calls += 1
        if self.calls == 1:
            assert any(t["function"]["name"] == D.DRAFT_NAME for t in tools)
            yield LLMEvent("tool_call", tool_call=ToolCall("c1", "get_order_info", {"order_id": "20260924-0035429"}))
            yield LLMEvent("tool_call", tool_call=ToolCall("c2", D.DRAFT_NAME, ANSWERS))
        else:
            assert all(t["function"]["name"] != D.DRAFT_NAME for t in tools)   # 재호출에는 후보 함수를 안 보인다
            yield LLMEvent("text", text="LLM 재호출 답입니다. ")
        yield LLMEvent("done")


class _TTS:
    def __init__(self):
        self.texts = []

    async def synthesize(self, text):
        self.texts.append(text)
        yield b"\x10\x27" * 320


TOOLS = [{"type": "function", "function": {"name": "get_order_info", "description": "", "parameters": {
    "type": "object", "properties": {"order_id": {"type": "string"}}}}}]


async def _sink(pcm):
    pass


async def test_responder_uses_draft_without_second_llm_call():
    sent = []

    async def tool(name, args):
        sent.append(name)
        return ORDER

    llm, tts = _LLM(), _TTS()
    r = Responder(llm, tts, tool, tools=TOOLS, settings=Settings(filler_delay_ms=10_000))
    spoken = await r.respond("S", [Message("user", "20260924-0035429 언제 와요?")], _sink)
    assert spoken == ["하루온 간 가드 맥스2은 10월 1일에 배송 완료됐어요."]
    assert llm.calls == 1 and sent == ["get_order_info"]          # draft_answers는 함수로 보내지 않는다
    assert "지금 배송 중이에요." in tts.texts                       # 빈칸 없는 상위 후보는 미리 합성


async def test_responder_falls_back_to_llm_when_no_draft_matches():
    async def tool(name, args):
        return {"status": "ok", "data": {"orders": [{"배송상태": "입금전"}]}}

    llm = _LLM()
    r = Responder(llm, _TTS(), tool, tools=TOOLS, settings=Settings(filler_delay_ms=10_000))
    spoken = await r.respond("S", [Message("user", "x")], _sink)
    assert spoken == ["LLM 재호출 답입니다."] and llm.calls == 2
