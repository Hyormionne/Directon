"""AG-T49 번호를 받고도 조회 없이 번호 전체를 다시 묻는 문제(공식 측정 runs/official_t45 약 15건).

- 같은 회차에 다른 조회가 성공했으면 막힌 조회는 건너뛰고 그 결과로 답한다(P0173-02).
- 막힌 조회는 발신번호로 대신 조회한다(get_order_info·get_return_info). 결과가 있으면 다시 묻지 않는다.
- get_order_info의 한 인자만 자리 수가 틀리고 다른 번호가 맞으면 틀린 인자를 버리고 맞는 번호로 조회한다(P0086-02 유형).
- 번호 인자가 모두 빈 조회는 발신번호로 채우고, 없으면 부르지 않는다(P0104-01·P0124-01·P0152-01). 같은 회차에 이미
  번호를 물었으면 다시 생성하지 않고 턴을 끝낸다(P0104-01 같은 질문 두 번).
- 들린 번호가 어떤 형식에도 맞지 않으면 첫 마디를 "네, 확인해 드릴게요." 대신 기본 첫 마디로(되묻기와 엇갈림 방지).
- 이전 대화에서 조회 결과가 없던 번호는 다시 묻지 말고 다른 정보를 묻게 한다(P0095-06 "번호는 동일").
"""
import asyncio
import json
from pathlib import Path

import pytest

from agent import catalog as catalog_mod
from agent.config import Settings
from agent.core.types import LLMEvent, Message, SessionInfo, ToolCall
from agent.prompts import build_system, history_summary
from agent.responder import Responder
from agent.tts_cache import opener_kind, select_opener_for_text

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = json.loads((ROOT / "mock_api" / "data" / "tools.json").read_text(encoding="utf-8"))
TOOLS = [{"type": "function", "function": t} for t in OFFICIAL]
CALLER = {"name": None, "phone": "01031980722", "email": None}
FILLER = "잠시만 기다려 주세요."
ORDER = {"주문번호": "20261010-0039399", "상세주문번호": "20261010-0039399-01", "제품명": "간 가드", "배송상태": "배송중"}
RETURN = {"상품명": "위 가드 루틴", "주문번호": "20260930-0010832", "현재상태": "반품처리중[수거전]"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("DIRECTON_CATALOG_PATH", "DIRECTON_MOCK_DATA", "DIRECTON_CONFIRM_ORDER_ID", "DIRECTON_REPAIR_ORDER_ID",
              "DIRECTON_NUMBER_CANDIDATES"):
        monkeypatch.delenv(k, raising=False)
    catalog_mod.reset_default_catalog()
    yield
    catalog_mod.reset_default_catalog()


def system_for(caller=None, history=None):
    info = SessionInfo("S", now="2026-11-04 (수) 11:00", caller=caller, history=history or [])
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


def calls(*tcs, text=""):
    head = [LLMEvent("text", text=text)] if text else []
    return head + [LLMEvent("tool_call", tool_call=ToolCall(f"c{i}", n, a)) for i, (n, a) in enumerate(tcs)] \
        + [LLMEvent("done")]


def say(t):
    return [LLMEvent("text", text=t), LLMEvent("done")]


class FakeTools:
    """발신번호(01031980722) 주문·반품은 있고, 그 밖 번호는 비어 있다."""

    def __init__(self, phone="01031980722"):
        self.phone = phone
        self.seen: list[tuple[str, dict]] = []

    async def __call__(self, name, args):
        self.seen.append((name, dict(args)))
        if name == "get_order_info":
            hit = self.phone and args.get("buyer_cellphone") == self.phone
            return {"status": "ok", "data": {"orders": [ORDER] if hit else []}, "message": "주문 조회 완료"}
        if name == "get_return_info":
            if self.phone and args.get("customer_phone_number") == self.phone:
                return {"status": "ok", "data": {"반품정보": [RETURN]}, "message": "반품 정보 조회가 완료되었습니다."}
            return {"status": "not_found", "message": "반품 상품이 없습니다."}
        return {"status": "ok"}


def run(llm, tool, system, utt="확인해 주세요", **settings):
    r = Responder(llm, _TTS(), tool, tools=TOOLS, cache=_Cache(), settings=Settings(**settings))
    return asyncio.run(r.respond(system, [Message("user", utt)], _sink))


def tool_bodies(llm, round_idx=1):
    return [json.loads(m.content) for m in llm.seen[round_idx] if m.role == "tool"]


# ── P0173-02: 번호 교정용 발신번호 자동 조회는 성공했는데 LLM은 막힌 결과만 받아 다시 물었다 ─────────
def test_caller_auto_lookup_result_reaches_llm_when_number_blocked():
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "520101811", "buyer_cellphone": None})),
                     say("간 가드 주문은 배송중이에요.")])
    tool = FakeTools()
    run(llm, tool, system_for(CALLER), utt="520101811. 222G22.")
    assert tool.seen == [("get_order_info", {"buyer_cellphone": "01031980722"})]   # 자동 조회 한 번뿐
    body = tool_bodies(llm)[0]
    assert body["status"] == "ok" and body["data"]["orders"]
    assert "다시 묻지" in body["_note"]


# ── 같은 회차에 같은 조회가 통과하면 막힌 조회는 건너뛴다 ─────────
def test_blocked_lookup_skipped_when_same_round_lookup_succeeds():
    llm = ScriptLLM([calls(("get_order_info", {"buyer_cellphone": "01031980722"}),
                           ("get_order_info", {"order_id": "520101811", "buyer_cellphone": None})),
                     say("간 가드 주문은 배송중이에요.")])
    tool = FakeTools()
    run(llm, tool, system_for(None), utt="010 3198 0722, 520101811")
    assert tool.seen == [("get_order_info", {"buyer_cellphone": "01031980722"})]
    ok, skipped = tool_bodies(llm)
    assert ok["status"] == "ok" and ok["data"]["orders"]
    assert skipped["status"] == "skipped"
    assert "다시 묻지" in skipped["message"]


# ── 막힌 조회 → 발신번호로 대신 조회 ─────────────────────────────
def test_blocked_order_lookup_falls_back_to_caller_phone():
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "513755499451", "buyer_cellphone": None})),
                     say("간 가드 주문은 배송중이에요.")])
    tool = FakeTools()
    run(llm, tool, system_for(CALLER))
    assert tool.seen and all(a.get("buyer_cellphone") == "01031980722" and not a.get("order_id") for _, a in tool.seen)
    assert len(tool.seen) == 1                                   # 발신번호 조회는 턴당 한 번(AG-T28 목록 조회 재사용)
    body = tool_bodies(llm)[0]
    assert body["status"] == "ok" and body["data"]["orders"]
    assert "발신번호" in body["_note"] and "다시 묻지" in body["_note"]


def test_blocked_return_lookup_falls_back_to_caller_phone():
    llm = ScriptLLM([calls(("get_return_info", {"customer_phone_number": "520101811"})),
                     say("반품 회수 전 상태예요.")])
    tool = FakeTools()
    run(llm, tool, system_for(CALLER))
    assert tool.seen == [("get_return_info", {"customer_phone_number": "01031980722"})]
    body = tool_bodies(llm)[0]
    assert body["status"] == "ok" and body["data"]["반품정보"]
    assert "발신번호" in body["_note"]


def test_fallback_without_result_keeps_reask():
    llm = ScriptLLM([calls(("get_return_info", {"customer_phone_number": "520101811"})), say("다시 말씀해 주시겠어요?")])
    tool = FakeTools(phone="")                                  # 발신번호로도 반품 없음
    run(llm, tool, system_for(CALLER))
    body = tool_bodies(llm)[0]
    assert body["status"] == "invalid_arguments"
    assert "발신번호" in body["message"]                        # 발신번호로도 없었다는 사실을 알린다


def test_no_fallback_without_caller():
    llm = ScriptLLM([calls(("get_return_info", {"customer_phone_number": "515110736"})), say("다시요.")])
    tool = FakeTools()
    run(llm, tool, system_for(None))
    assert tool.seen == []
    assert tool_bodies(llm)[0]["status"] == "invalid_arguments"


# ── get_order_info: 틀린 인자 하나는 버리고 맞는 번호로(P0086-02 유형) ─────
def test_wrong_length_optional_arg_dropped_when_other_number_valid():
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "2956331155831032310510878976",
                                               "buyer_cellphone": "01082389762"})), say("네.")])
    tool = FakeTools()
    run(llm, tool, system_for(None))
    assert tool.seen == [("get_order_info", {"order_id": None, "buyer_cellphone": "01082389762"})]


# ── 빈 인자 조회(P0104-01·P0124-01·P0152-01) ──────────────────────
def test_empty_lookup_not_called_without_caller():
    llm = ScriptLLM([calls(("get_order_info", {"order_id": None, "buyer_cellphone": None})),
                     say("주문번호나 주문하신 분 연락처를 말씀해 주시겠어요?")], second_delay=0.15)
    tool = FakeTools()
    spoken = run(llm, tool, system_for(None), utt="주문한 건강제 언제 도착하나요?", filler_delay_ms=10, fillers=(FILLER,))
    assert tool.seen == []
    assert FILLER not in spoken
    body = tool_bodies(llm)[0]
    assert body["status"] == "invalid_arguments" and "받지 못했습니다" in body["message"]


def test_empty_lookup_filled_with_caller_phone():
    llm = ScriptLLM([calls(("get_order_info", {"order_id": None, "buyer_cellphone": None})), say("배송중이에요.")])
    tool = FakeTools()
    run(llm, tool, system_for(CALLER), utt="주문한 거 언제 와요?")
    assert tool.seen == [("get_order_info", {"order_id": None, "buyer_cellphone": "01031980722"})]


def test_question_then_empty_lookup_ends_turn_without_repeat():
    q = "주문번호나 주문하신 분 연락처를 말씀해 주시겠어요?"
    llm = ScriptLLM([calls(("get_order_info", {"buyer_cellphone": None, "order_id": None}), text=q), say(q)],
                    second_delay=0.15)
    tool = FakeTools()
    spoken = run(llm, tool, system_for(None), utt="입금 확인이 안 돼요.", filler_delay_ms=10, fillers=(FILLER,))
    assert tool.seen == []
    assert len(llm.seen) == 1                                     # 다시 생성하지 않는다
    assert spoken.count(q) == 1 and FILLER not in spoken


# ── 첫 마디: 형식에 맞는 번호가 없으면 "확인해 드릴게요"를 쓰지 않는다 ───────
@pytest.mark.parametrize("text", [
    "51375. 5499. 451.",                                         # P0052-02
    "5109580? 5497.",                                            # P0154-02
    "515110736 반품 횟수 횟 같다네요.",                           # P0032-09(키워드 '반품'이 있어도)
    "어, 채 씨. 010-3710. 35.",                                   # P0193-02
    "주문 1. 2026. 18. 33. 주문번호. 20162031. 008422. 금액. 68,500원.",   # P0110-02
])
def test_garbled_number_uses_plain_opener(text):
    s = Settings()
    assert select_opener_for_text(text, s) == s.openers[0]
    assert opener_kind(text, s) == "simple"


@pytest.mark.parametrize("text", [
    "01079044351",
    "어, 채 씨. 010-3537-2635.",
    "2956732601515352 01082389762",
    "주문번호. 20261003-0084222. 금액. 68,500원",
    "공일공 칠구공사 사삼오일",                                  # 한글로 읽은 번호는 그대로(조회 첫 마디)
    "반품하고 싶어요",                                          # 번호 없는 조회 요청은 그대로
])
def test_valid_number_or_lookup_keeps_lookup_opener(text):
    s = Settings()
    assert select_opener_for_text(text, s) == s.opener_lookup
    assert opener_kind(text, s) == "lookup"


# ── 이전 대화에서 결과가 없던 번호(P0095-06) ───────────────────────
EMPTY_HIST = [
    {"role": "customer", "text": "주문한거 언제도착하나요"},
    {"role": "tool_call", "name": "get_order_info", "arguments": {"order_id": None, "buyer_cellphone": "01038137325"}},
    {"role": "tool_result", "name": "get_order_info",
     "result": {"status": "ok", "data": {"orders": []}, "message": "주문 조회 완료"}},
    {"role": "agent", "text": "01038137325 번호로는 주문서 조회가 되지 않습니다. 주문 시 연락처나 주문번호를 확인 부탁드려도 될까요?"},
]


def test_history_failed_lookup_number_not_asked_again():
    lines = history_summary(EMPTY_HIST)
    line = next(l for l in lines if "결과가 없었" in l)
    assert "01038137325" in line
    assert "다시 묻지" in line and "다른 연락처" in line and "주문번호" in line
    assert "동일" in line or "같은 번호" in line


def test_history_not_found_return_lookup_also_flagged():
    hist = [{"role": "tool_call", "name": "get_return_info", "arguments": {"customer_phone_number": "01035726035"}},
            {"role": "tool_result", "name": "get_return_info", "result": {"status": "not_found", "message": "반품 상품이 없습니다."}},
            {"role": "agent", "text": "반품 내역이 없어요."}]
    assert any("01035726035" in l and "결과가 없었" in l for l in history_summary(hist))


def test_history_successful_lookup_not_flagged():
    hist = [{"role": "tool_call", "name": "get_order_info", "arguments": {"buyer_cellphone": "01031980722"}},
            {"role": "tool_result", "name": "get_order_info", "result": {"status": "ok", "data": {"orders": [ORDER]}}},
            {"role": "agent", "text": "배송중이에요."}]
    assert not any("결과가 없었" in l for l in history_summary(hist))
