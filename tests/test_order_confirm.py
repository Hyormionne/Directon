"""AG-T28 번호 확인 되묻기(기본 끔 스위치 DIRECTON_CONFIRM_ORDER_ID).

번호를 확실히 고칠 수 없어 되묻던 경우(자릿수 오류 + 교정 실패, 자릿수는 맞는데 not_found)에만,
발신 고객 본인의 주문 중 가장 가까운 1~2개를 끝자리·상품명으로 확인하라는 결과(needs_confirmation)로 바꾼다.
조용한 교정(후보 하나뿐 → 바로 호출, AG-T15a)과 스위치를 끈 동작은 그대로다.
"""
import asyncio
import json

import pytest

from agent.config import Settings
from agent.core.types import LLMEvent, Message, SessionInfo, ToolCall
from agent.prompts import build_system
from agent.responder import Responder, confirm_candidates, last_digits
from eval import llm_eval
from mock_api.legacy_shop import load_shop

CALLER = {"name": "한소율", "phone": "01076543210", "email": None}
ORDERS = [{"order_id": "2611160457", "status": "배송중", "product_name": "무선 충전기"},
          {"order_id": "2611140498", "status": "배송완료", "product_name": "러닝화"},
          {"order_id": "2610300266", "status": "배송완료", "product_name": "기모 후드티"}]
ENV = "DIRECTON_CONFIRM_ORDER_ID"


@pytest.fixture(scope="module")
def tools():
    return llm_eval.shop_tools(load_shop())


def system_for(caller=None):
    info = SessionInfo("S", now="2026-11-20 14:00", caller=caller, history=[])
    return build_system(info, "네, 고객님.", manual="매뉴얼 본문")


class _TTS:
    async def synthesize(self, text):
        yield text.encode("utf-8")


async def _sink(pcm):
    return None


class ScriptLLM:
    def __init__(self, rounds):
        self.rounds = rounds
        self.seen: list[list[Message]] = []

    async def stream(self, system, messages, tools):
        self.seen.append(list(messages))
        for ev in self.rounds[min(len(self.seen) - 1, len(self.rounds) - 1)]:
            await asyncio.sleep(0)
            yield ev


def calls(*tcs):
    return [LLMEvent("tool_call", tool_call=ToolCall(f"c{i}", n, a)) for i, (n, a) in enumerate(tcs)] + [LLMEvent("done")]


def say(t):
    return [LLMEvent("text", text=t), LLMEvent("done")]


class FakeTools:
    """get_order_list는 고객 주문, 주문 함수는 목록에 있는 번호만 ok, 아니면 not_found."""

    def __init__(self, orders=ORDERS):
        self.orders = orders
        self.seen: list[tuple[str, dict]] = []

    async def __call__(self, name, args):
        self.seen.append((name, dict(args)))
        if name == "get_order_list":
            return {"status": "ok", "data": {"name": "한소율", "orders": list(self.orders)}}
        if args.get("order_id") in {o["order_id"] for o in self.orders}:
            return {"status": "ok", "data": {"order_id": args.get("order_id")}}
        return {"status": "not_found", "message": "해당 주문번호의 주문을 찾을 수 없습니다."}


async def run(llm, tool, tools, system, utt="주문 상태 봐 주세요."):
    r = Responder(llm, _TTS(), tool, tools=tools, settings=Settings())
    await r.respond(system, [Message("user", utt)], _sink)


def tool_body(llm, round_idx=1):
    return json.loads([m for m in llm.seen[round_idx] if m.role == "tool"][0].content)


# ── 후보 고르기 ────────────────────────────────────────
def test_candidates_single_closest_same_length():
    c = confirm_candidates("2611140497", ORDERS, 10)          # 끝자리 하나 틀림
    assert [x["order_id"] for x in c] == ["2611140498"]
    assert c[0]["product_name"] == "러닝화" and c[0]["last_digits"] == "0498"


def test_candidates_two_when_tied():
    orders = [{"order_id": "2611150231", "product_name": "A"}, {"order_id": "2611150232", "product_name": "B"}]
    c = confirm_candidates("261115023", orders, 10)           # 한 자리 빠짐, 둘 다 같은 거리
    assert sorted(x["order_id"] for x in c) == ["2611150231", "2611150232"]


def test_candidates_none_when_three_or_more_tied():
    orders = [{"order_id": f"261115023{i}", "product_name": str(i)} for i in (1, 2, 3)]
    assert confirm_candidates("261115023", orders, 10) == []


def test_candidates_none_when_far_or_short_or_exact():
    assert confirm_candidates("2612999999", ORDERS, 10) == []  # 먼 번호
    assert confirm_candidates("123", ORDERS, 10) == []         # 번호라 보기 어려운 짧은 수
    assert confirm_candidates("2611140498", ORDERS, 10) == []  # 같은 번호는 확인 대상이 아님
    assert confirm_candidates("2611140497", [], 10) == []


def test_candidates_four_digit_ids():
    orders = [{"order_id": "1302", "product_name": "유산균"}, {"order_id": "2417", "product_name": "비타민"}]
    c = confirm_candidates("130", orders, 4)                  # 앞자리 '일' 누락이 아니라 끝자리 누락
    assert [x["order_id"] for x in c] == ["1302"] and c[0]["last_digits"] == "1302"


def test_last_digits():
    assert last_digits("2611140498") == "0498"
    assert last_digits("1302") == "1302"


# ── 응답기: 스위치 끔(기본) → 이전과 같음 ─────────────────
async def test_off_by_default_not_found_unchanged(tools, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("DIRECTON_NUMBER_CANDIDATES", "0")     # AG-T43 약한 확인도 끈 옛 동작
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "2611140497"})), say("다시 말씀해 주세요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(CALLER))
    assert [n for n, _ in tool.seen] == ["get_order_info"]     # 목록 조회를 더하지 않는다
    assert tool_body(llm)["status"] == "not_found"


async def test_off_ambiguous_length_error_unchanged(tools, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("DIRECTON_NUMBER_CANDIDATES", "0")     # AG-T43 약한 확인도 끈 옛 동작
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "261115023"})), say("다시 말씀해 주세요. ")])
    tool = FakeTools(orders=[{"order_id": "2611150231", "product_name": "A"},
                             {"order_id": "2611150232", "product_name": "B"}])
    await run(llm, tool, tools, system_for(CALLER))
    body = tool_body(llm)
    assert body["status"] == "invalid_arguments" and "한 자리씩" in body["message"]


# ── 응답기: 스위치 켬 ─────────────────────────────────
async def test_on_not_found_becomes_confirmation(tools, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "2611140497"})), say("끝자리 0498 러닝화 주문 맞으실까요? ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(CALLER))
    assert [n for n, _ in tool.seen] == ["get_order_info", "get_order_list"]
    body = tool_body(llm)
    assert body["status"] == "needs_confirmation" and body["original_status"] == "not_found"
    assert [c["order_id"] for c in body["candidates"]] == ["2611140498"]
    msg = body["message"]
    assert "끝자리 0498" in msg and "한글" not in msg and "러닝화" in msg and "맞으실까요" in msg
    assert "2611140498" not in msg and "한 자리씩" not in msg  # 전체 번호를 읽게 하지 않는다


async def test_on_ambiguous_length_error_asks_which_one(tools, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "261115023"})), say("어느 주문일까요? ")])
    tool = FakeTools(orders=[{"order_id": "2611150231", "product_name": "무선 이어폰"},
                             {"order_id": "2611150232", "product_name": "보조배터리"}])
    await run(llm, tool, tools, system_for(CALLER))
    assert [n for n, _ in tool.seen] == ["get_order_list"]     # 목록은 한 번, 틀린 번호는 보내지 않는다
    body = tool_body(llm)
    assert body["status"] == "needs_confirmation" and body["original_status"] == "invalid_arguments"
    assert len(body["candidates"]) == 2
    assert "무선 이어폰" in body["message"] and "보조배터리" in body["message"] and "어느" in body["message"]


async def test_on_unique_wrong_length_still_repaired_silently(tools, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "26140498"})), say("배송완료예요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(CALLER))
    assert tool.seen[-1] == ("get_order_info", {"order_id": "2611140498"})
    assert tool_body(llm)["status"] == "ok"


async def test_on_without_caller_keeps_old_behavior(tools, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "2611140497"})), say("다시 말씀해 주세요. ")])
    tool = FakeTools()
    await run(llm, tool, tools, system_for(None))
    assert [n for n, _ in tool.seen] == ["get_order_info"]     # 남의 주문 목록은 보지 않는다
    assert tool_body(llm)["status"] == "not_found"


async def test_on_too_many_candidates_keeps_reask(tools, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "261115023"})), say("다시 말씀해 주세요. ")])
    tool = FakeTools(orders=[{"order_id": f"261115023{i}", "product_name": str(i)} for i in (1, 2, 3)])
    await run(llm, tool, tools, system_for(CALLER))
    assert tool_body(llm)["status"] == "invalid_arguments"
    assert [n for n, _ in tool.seen] == ["get_order_list"]
