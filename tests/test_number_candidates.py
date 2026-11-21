"""AG-T43 번호 후보 컨텍스트 + 확신도별 처리.

이미 아는 번호(발신번호, 이전 대화의 번호·조회 결과 주문번호)를 '번호 후보'로 프롬프트에 넣는다.
들은 번호가 후보와 강하게 맞으면 후보로 바꿔 호출하고, 약하게 맞으면(자리 수가 틀려 막힐 번호) 그 후보 끝자리로 되묻는다.
"""
import asyncio
import json
from pathlib import Path

import pytest

from agent import catalog as catalog_mod
from agent.config import Settings
from agent.core.types import LLMEvent, Message, SessionInfo, ToolCall
from agent.prompts import build_system, number_candidates
from agent.responder import Responder, classify_number, number_candidates_from_system

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = json.loads((ROOT / "mock_api" / "data" / "tools.json").read_text(encoding="utf-8"))
TOOLS = [{"type": "function", "function": t} for t in OFFICIAL]
CALLER = {"name": "안가짜", "phone": "01079044351", "email": None}
ORDER_HIST = [
    {"role": "customer", "text": "주문 확인하려고요"},
    {"role": "tool_call", "name": "get_order_info", "arguments": {"order_id": None, "buyer_cellphone": "01079044351"}},
    {"role": "tool_result", "name": "get_order_info",
     "result": {"status": "ok", "data": {"orders": [{"주문번호": "20261001-0005987", "상품명": "무선 충전기",
                                                    "배송상태": "배송중"}]}, "message": "주문 조회 완료"}},
    {"role": "agent", "text": "무선 충전기 주문은 배송중입니다."},
]
PHONE_HIST = [
    {"role": "customer", "text": "제 번호는 01033334444예요"},
    {"role": "agent", "text": "네, 확인했습니다. 더 도와드릴까요?"},
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("DIRECTON_CATALOG_PATH", "DIRECTON_MOCK_DATA", "DIRECTON_CONFIRM_ORDER_ID", "DIRECTON_REPAIR_ORDER_ID",
              "DIRECTON_NUMBER_CANDIDATES"):
        monkeypatch.delenv(k, raising=False)
    catalog_mod.reset_default_catalog()
    yield
    catalog_mod.reset_default_catalog()


def info_for(caller=None, history=None):
    return SessionInfo("S", now="2026-11-04 (수) 11:00", caller=caller, history=history or [])


def system_for(caller=None, history=None):
    return build_system(info_for(caller, history), "네, 고객님.", manual="매뉴얼 본문")


def _var(s):
    return s[s.index("## 이번 통화"):]


def _cand_line(system):
    lines = [l for l in _var(system).splitlines() if l.startswith("- 번호 후보:")]
    return lines[0] if lines else ""


# ── 프롬프트: 번호 후보 줄 ─────────────────────────────────
def test_caller_phone_is_candidate():
    line = _cand_line(system_for(CALLER))
    assert "전화번호 01079044351(발신번호)" in line
    assert "끝자리" in line                     # 약한 확신이면 끝자리로 확인
    assert "그 후보 번호를 그대로" in line       # 강한 확신이면 후보로 바꿔 쓴다


def test_history_order_result_is_candidate_with_product():
    line = _cand_line(system_for(CALLER, ORDER_HIST))
    assert "주문번호 20261001-0005987(무선 충전기)" in line
    assert line.count("01079044351") == 1        # 발신번호와 이전 호출 인자의 같은 번호는 한 번만


def test_history_spoken_phone_is_candidate():
    cands = number_candidates(info_for(None, PHONE_HIST))
    assert [(c["kind"], c["number"]) for c in cands] == [("phone", "01033334444")]


def test_no_candidates_no_line():
    assert _cand_line(system_for(None)) == ""


def test_switch_off_removes_line(monkeypatch):
    monkeypatch.setenv("DIRECTON_NUMBER_CANDIDATES", "0")
    assert _cand_line(system_for(CALLER, ORDER_HIST)) == ""


def test_candidates_parsed_back_from_system():
    cands = number_candidates_from_system(system_for(CALLER, ORDER_HIST))
    got = {(c["kind"], c["number"], c["raw"], c["note"]) for c in cands}
    assert ("phone", "01079044351", "01079044351", "발신번호") in got
    assert ("order", "202610010005987", "20261001-0005987", "무선 충전기") in got


# ── 확신도 분류 ───────────────────────────────────────────
PHONE = {"kind": "phone", "number": "01079044351", "raw": "01079044351", "note": "발신번호"}
O1 = {"kind": "order", "number": "202610010005987", "raw": "20261001-0005987", "note": "무선 충전기"}
O2 = {"kind": "order", "number": "202610010005988", "raw": "20261001-0005988", "note": "케이블"}


@pytest.mark.parametrize("heard,level", [
    ("01079044351", "exact"),
    ("01079044851", "strong"),          # 한 자리 오인식
    ("0107904435", "strong"),           # 한 자리 빠짐
    ("513755499451", "weak"),           # AG-T40 실제 오인식(Soniox "51375. 5499. 451.")
    ("55555555555", "none"),
    ("98765", "none"),
])
def test_classify_phone(heard, level):
    got, cands = classify_number(heard, [PHONE])
    assert got == level
    assert (cands == [PHONE]) == (level != "none")


def test_classify_order_stt_example_is_strong():
    assert classify_number("2026202000987", [O1]) == ("strong", [O1])   # AG-T40 "2026년 2020. 00987"


def test_classify_two_close_orders_is_weak_with_both():
    level, cands = classify_number("2026100100059", [O1, O2])
    assert level == "weak" and cands == [O1, O2]


# ── 응답기: 강한 확신 → 교체 ────────────────────────────────
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


def calls(name, args):
    return [LLMEvent("tool_call", tool_call=ToolCall("c0", name, args)), LLMEvent("done")]


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


def run(llm, tool, system, utt="확인해 주세요"):
    r = Responder(llm, _TTS(), tool, tools=TOOLS, settings=Settings())
    return asyncio.run(r.respond(system, [Message("user", utt)], _sink))


def tool_body(llm):
    return json.loads([m for m in llm.seen[1] if m.role == "tool"][0].content)


def test_same_length_order_close_to_history_order_is_swapped():
    llm = ScriptLLM([calls("get_order_info", {"order_id": "202610010005387", "buyer_cellphone": None}), say("배송중이에요.")])
    tool = FakeTools()
    run(llm, tool, system_for(None, ORDER_HIST))
    assert tool.seen == [("get_order_info", {"order_id": "20261001-0005987", "buyer_cellphone": None})]


def test_phone_close_to_history_phone_is_swapped():
    llm = ScriptLLM([calls("request_callback", {"customer_name": "안가짜", "phone_number": "01033384444"}), say("접수됐어요.")])
    tool = FakeTools()
    run(llm, tool, system_for(None, PHONE_HIST))
    assert tool.seen[-1][1]["phone_number"] == "01033334444"


def test_different_valid_order_is_kept():
    llm = ScriptLLM([calls("get_order_info", {"order_id": "202609150001234", "buyer_cellphone": None}), say("네.")])
    tool = FakeTools()
    run(llm, tool, system_for(None, ORDER_HIST))
    assert tool.seen[-1][1]["order_id"] == "202609150001234"


# ── 응답기: 약한 확신 → 그 후보로 되묻기 ──────────────────────
def test_weak_phone_match_asks_with_candidate_last_digits():
    llm = ScriptLLM([calls("get_return_info", {"customer_phone_number": "513755499451"}),
                     say("끝자리 4351 번호 맞으실까요?")])
    tool = FakeTools()
    run(llm, tool, system_for(CALLER))
    assert [n for n, _ in tool.seen if n == "get_return_info"] == []      # 확인 전에는 호출하지 않는다
    body = tool_body(llm)
    assert body["status"] == "needs_confirmation" and body["original_status"] == "invalid_arguments"
    assert body["candidates"][0]["last_digits"] == "4351"
    assert "끝자리 4351" in body["message"]


def test_no_close_candidate_keeps_invalid_arguments():
    llm = ScriptLLM([calls("get_return_info", {"customer_phone_number": "98765"}), say("다시 말씀해 주시겠어요?")])
    run(llm, FakeTools(), system_for(CALLER))
    assert tool_body(llm)["status"] == "invalid_arguments"


def test_switch_off_keeps_invalid_arguments(monkeypatch):
    monkeypatch.setenv("DIRECTON_NUMBER_CANDIDATES", "0")
    llm = ScriptLLM([calls("get_return_info", {"customer_phone_number": "513755499451"}), say("다시요.")])
    run(llm, FakeTools(), system_for(CALLER))
    assert tool_body(llm)["status"] == "invalid_arguments"
