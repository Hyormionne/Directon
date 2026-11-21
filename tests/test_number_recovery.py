"""AG-T41·T42 STT 숫자 오인식 대응(AG-T40 진단: 연습 음성의 숫자를 STT가 틀리게 받아 적는다).

- AG-T41: 자리 수는 맞지만 발신번호와 거의 같은 전화번호 인자는 발신번호로 교정한다.
- AG-T42: 자리 수 검사에 막힐 회차엔 채움말을 틀지 않고, 되묻기 안내는 짧게(묻지 않은 안내·자리 수 규칙 설명 없이).
"""
import asyncio
import json
from pathlib import Path

import pytest

from agent import catalog as catalog_mod
from agent.config import Settings
from agent.core.types import LLMEvent, Message, SessionInfo, ToolCall
from agent.prompts import build_fixed, build_system
from agent.responder import Responder

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = json.loads((ROOT / "mock_api" / "data" / "tools.json").read_text(encoding="utf-8"))
TOOLS = [{"type": "function", "function": t} for t in OFFICIAL]
CALLER = {"name": "안가짜", "phone": "01052072967", "email": None}
FILLER = "잠시만 기다려 주세요."


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("DIRECTON_CATALOG_PATH", "DIRECTON_MOCK_DATA", "DIRECTON_CONFIRM_ORDER_ID", "DIRECTON_REPAIR_ORDER_ID"):
        monkeypatch.delenv(k, raising=False)
    catalog_mod.reset_default_catalog()
    yield
    catalog_mod.reset_default_catalog()


def system_for(caller=None):
    info = SessionInfo("S", now="2026-11-04 (수) 11:00", caller=caller, history=[])
    return build_system(info, "네, 고객님.", manual="매뉴얼 본문")


class _TTS:
    async def synthesize(self, text):
        yield text.encode("utf-8")


class _Cache:
    def get(self, text):
        return b"\x11\x11" if text == FILLER else None


async def _sink(pcm):
    return None


class ScriptLLM:
    def __init__(self, rounds, second_delay=0.0):
        self.rounds, self.second_delay = rounds, second_delay
        self.seen: list[list[Message]] = []

    async def stream(self, system, messages, tools):
        self.seen.append(list(messages))
        if len(self.seen) > 1:
            await asyncio.sleep(self.second_delay)
        for ev in self.rounds[min(len(self.seen) - 1, len(self.rounds) - 1)]:
            await asyncio.sleep(0)
            yield ev


def calls(*tcs):
    return [LLMEvent("tool_call", tool_call=ToolCall(f"c{i}", n, a)) for i, (n, a) in enumerate(tcs)] + [LLMEvent("done")]


def say(t):
    return [LLMEvent("text", text=t), LLMEvent("done")]


class FakeTools:
    def __init__(self):
        self.seen: list[tuple[str, dict]] = []

    async def __call__(self, name, args):
        self.seen.append((name, dict(args)))
        if name == "get_order_info":
            return {"status": "ok", "data": {"orders": []}, "message": "주문 조회 완료"}
        return {"status": "ok"}


def run(llm, tool, system, utt="전화 주세요", **settings):
    r = Responder(llm, _TTS(), tool, tools=TOOLS, cache=_Cache(), settings=Settings(**settings))
    return asyncio.run(r.respond(system, [Message("user", utt)], _sink))


def tool_body(llm, round_idx=1):
    return json.loads([m for m in llm.seen[round_idx] if m.role == "tool"][0].content)


# ── AG-T41: 자리 수는 맞는데 발신번호와 거의 같은 번호 ─────────
@pytest.mark.parametrize("name,args,key", [
    ("request_callback", {"customer_name": "안가짜", "phone_number": "01050207267"}, "phone_number"),   # P0119-04 유형
    ("request_callback", {"customer_name": "안가짜", "phone_number": "010-5207-2907"}, "phone_number"),
    ("get_return_info", {"customer_phone_number": "01052072667"}, "customer_phone_number"),
])
def test_valid_length_phone_close_to_caller_uses_caller(name, args, key):
    llm = ScriptLLM([calls((name, args)), say("접수됐어요.")])
    tool = FakeTools()
    run(llm, tool, system_for(CALLER))
    sent = [a for n, a in tool.seen if n == name]
    assert sent and sent[-1][key] == "01052072967"


def test_different_valid_phone_is_kept():
    llm = ScriptLLM([calls(("request_callback", {"customer_name": "안가짜", "phone_number": "01098761234"})),
                     say("접수됐어요.")])
    tool = FakeTools()
    run(llm, tool, system_for(CALLER))
    assert tool.seen == [("request_callback", {"customer_name": "안가짜", "phone_number": "01098761234"})]


def test_no_caller_keeps_valid_phone():
    llm = ScriptLLM([calls(("request_callback", {"customer_name": "안가짜", "phone_number": "01050207267"})),
                     say("접수됐어요.")])
    tool = FakeTools()
    run(llm, tool, system_for(None))
    assert tool.seen[-1][1]["phone_number"] == "01050207267"


def test_repair_switch_off_keeps_valid_phone(monkeypatch):
    monkeypatch.setenv("DIRECTON_REPAIR_ORDER_ID", "0")
    llm = ScriptLLM([calls(("request_callback", {"customer_name": "안가짜", "phone_number": "01050207267"})),
                     say("접수됐어요.")])
    tool = FakeTools()
    run(llm, tool, system_for(CALLER))
    assert tool.seen[-1][1]["phone_number"] == "01050207267"


# ── AG-T42: 막힐 회차엔 채움말 없음 ─────────────────────────
def test_no_filler_when_every_call_is_blocked_by_length():
    llm = ScriptLLM([calls(("get_return_info", {"customer_phone_number": "515110736"})),
                     say("번호를 한 자리씩 천천히 다시 말씀해 주시겠어요?")], second_delay=0.15)
    tool = FakeTools()
    spoken = run(llm, tool, system_for(None), utt="반품 회수해 갔대요", filler_delay_ms=10, fillers=(FILLER,))
    assert tool.seen == []
    assert FILLER not in spoken


def test_no_filler_when_required_number_missing():
    llm = ScriptLLM([calls(("get_return_info", {})), say("휴대폰 번호를 말씀해 주시겠어요?")], second_delay=0.15)
    spoken = run(llm, FakeTools(), system_for(None), utt="반품 됐나요", filler_delay_ms=10, fillers=(FILLER,))
    assert FILLER not in spoken


def test_filler_kept_when_call_goes_through():
    llm = ScriptLLM([calls(("get_return_info", {"customer_phone_number": "01018167368"})),
                     say("반품이 접수돼 있어요.")], second_delay=0.15)
    tool = FakeTools()
    spoken = run(llm, tool, system_for(None), utt="반품 됐나요", filler_delay_ms=10, fillers=(FILLER,))
    assert tool.seen and spoken[0] == FILLER


# ── AG-T42: 되묻기 안내문 ───────────────────────────────────
def test_length_block_message_is_short_reask_without_extras():
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "513755499451", "buyer_cellphone": None})), say("다시요.")])
    run(llm, FakeTools(), system_for(None), utt="01079044351")
    msg = tool_body(llm)["message"]
    assert "한 자리씩 천천히" in msg
    assert "덧붙이지 않는다" in msg                      # 묻지 않은 안내 금지
    assert "설명하지 않는다" in msg                      # 자리 수 규칙을 고객에게 설명하지 않음
    assert "직접 물었을 때만" in msg                     # 정책 답은 고객이 이번에 물은 경우만


def test_fixed_prompt_invalid_arguments_rule_has_no_extras():
    rule = [l for l in build_fixed().split("\n") if l.startswith("- invalid_arguments:")][0]
    assert "직접 물었을 때만" in rule and "덧붙이지 않는다" in rule
