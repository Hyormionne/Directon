"""AG-T38 공식 8개 함수 체계에서의 응답기·정규화·카탈로그·인계 판정.

함수 정의는 공식 사본 `mock_api/data/tools.json`. 번호 형식은 공식 데이터 기준(주문번호 `YYYYMMDD-NNNNNNN`
15자리, 일부 16자리, 전화번호 11자리). 상담원 연결 함수는 없다(인계는 말로) — 정답에 없는 접수 함수를
부르면 그 문항 함수 점수가 0이라 자동으로 부르면 안 된다.
"""
import asyncio
import json
from pathlib import Path

import pytest

from agent import catalog as catalog_mod
from agent import handoff_jev
from agent.config import Settings
from agent.core.types import LLMEvent, Message, SessionInfo, ToolCall
from agent.normalize import normalize_args
from agent.prompts import build_system
from agent.responder import Responder, _is_number_key, tool_result_template

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = json.loads((ROOT / "mock_api" / "data" / "tools.json").read_text(encoding="utf-8"))
TOOLS = [{"type": "function", "function": t} for t in OFFICIAL]
SCHEMAS = {t["name"]: t["parameters"] for t in OFFICIAL}
CALLER = {"name": "김가짜", "phone": "01047775671", "email": None}
ORDERS = [{"주문번호": "20260924-0035429", "상품명": "하루온 비타민", "배송상태": "배송완료"},
          {"주문번호": "20261001-0011111", "상품명": "하루온 유산균", "배송상태": "배송중"}]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("DIRECTON_CATALOG_PATH", "DIRECTON_MOCK_DATA", "DIRECTON_CONFIRM_ORDER_ID", "DIRECTON_REPAIR_ORDER_ID"):
        monkeypatch.delenv(k, raising=False)
    catalog_mod.reset_default_catalog()
    yield
    catalog_mod.reset_default_catalog()


def system_for(caller=None):
    info = SessionInfo("S", now="2026-11-04 (수) 12:39", caller=caller, history=[])
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


def _digits(v):
    return "".join(ch for ch in str(v or "") if ch.isdigit())


class FakeTools:
    """get_order_info: 전화번호만이면 고객 주문 목록, 주문번호가 목록에 있으면 그 주문, 아니면 not_found. 접수는 ok."""

    def __init__(self, orders=ORDERS):
        self.orders = orders
        self.seen: list[tuple[str, dict]] = []

    async def __call__(self, name, args):
        self.seen.append((name, dict(args)))
        if name == "get_order_info":
            oid = _digits(args.get("order_id"))
            if oid:
                hit = [o for o in self.orders if _digits(o["주문번호"]) == oid]
                return ({"status": "ok", "data": {"orders": hit}, "message": "주문 조회 완료"} if hit
                        else {"status": "not_found"})
            return {"status": "ok", "data": {"orders": list(self.orders)}, "message": "주문 조회 완료"}
        return {"status": "ok"}


async def run(llm, tool, system, utt="주문 취소해 주세요."):
    r = Responder(llm, _TTS(), tool, tools=TOOLS, settings=Settings())
    await r.respond(system, [Message("user", utt)], _sink)
    return r


def tool_body(llm, round_idx=1):
    return json.loads([m for m in llm.seen[round_idx] if m.role == "tool"][0].content)


# ── 인계: 함수 없음 ───────────────────────────────────
def test_promising_handoff_calls_no_function():
    llm = ScriptLLM([say("불편을 드려 죄송합니다. 상담원에게 연결해 드릴게요.")])
    tool = FakeTools()
    asyncio.run(run(llm, tool, system_for(CALLER), utt="상담원 연결해 주세요"))
    assert tool.seen == []


def test_settings_have_no_transfer_filler_skip():
    assert Settings().filler_skip_tools == ()


def test_handoff_hints_name_no_function():
    j = handoff_jev.HandoffJudgment(probs={}, handoff=True, reasons=["human_request"])
    for now in ("2026-11-04 (수) 11:00", "2026-11-07 (토) 11:00"):
        hint = handoff_jev.system_hint(j, now)
        assert hint and "transfer_to_agent" not in hint and "request_callback" not in hint
        assert "함수" in hint                              # 함수를 부르지 말라는 지시


# ── 번호 인자 정규화·인식 ─────────────────────────────
def test_order_number_is_number_key():
    assert _is_number_key("order_number") and _is_number_key("order_id")
    assert _is_number_key("buyer_cellphone") and _is_number_key("customer_phone_number")


def test_normalize_spoken_order_number_and_phones():
    out = normalize_args("return_order", {"order_number": "이공이육공구이사 공공삼오사이구", "reason": "변심"},
                         SCHEMAS["return_order"], "2026-11-04 12:39")
    assert _digits(out["order_number"]) == "202609240035429"
    out = normalize_args("get_return_info", {"customer_phone_number": "공일공 육공칠오 구팔공구"},
                         SCHEMAS["get_return_info"], "2026-11-04 12:39")
    assert out["customer_phone_number"] == "01060759809"
    out = normalize_args("get_order_info", {"order_id": None, "buyer_cellphone": "010-6075-9809"},
                         SCHEMAS["get_order_info"], "2026-11-04 12:39")
    assert out["buyer_cellphone"] == "01060759809" and out.get("order_id") is None


# ── 공식 번호 형식 ─────────────────────────────────────
def test_catalog_has_no_default_file_and_official_lengths():
    assert catalog_mod.default_catalog() is None                # mock 데이터를 읽지 않는다
    assert catalog_mod.order_id_lengths() == frozenset({15, 16})
    assert catalog_mod.phone_lengths() == frozenset({11})


@pytest.mark.parametrize("oid", ["20260924-0035429", "202609240035429", "2956732601515352"])
def test_valid_order_numbers_are_called(oid):
    llm = ScriptLLM([calls(("cancel_order", {"order_id": oid, "reason": "변심"})), say("접수됐어요.")])
    tool = FakeTools()
    asyncio.run(run(llm, tool, system_for()))
    assert [n for n, _ in tool.seen] == ["cancel_order"]


def test_wrong_length_order_number_is_blocked_without_caller():
    llm = ScriptLLM([calls(("cancel_order", {"order_id": "2026092400354", "reason": "변심"})), say("다시 말씀해 주세요.")])
    tool = FakeTools()
    asyncio.run(run(llm, tool, system_for()))
    assert tool.seen == []
    assert tool_body(llm)["status"] == "invalid_arguments"


def test_wrong_length_phone_is_blocked_and_valid_phone_called():
    llm = ScriptLLM([calls(("get_return_info", {"customer_phone_number": "0106075980"})), say("다시요.")])
    tool = FakeTools()
    asyncio.run(run(llm, tool, system_for(), utt="반품 현황 알려 주세요"))
    assert tool.seen == []
    llm = ScriptLLM([calls(("get_order_info", {"order_id": None, "buyer_cellphone": "01060759809"})), say("네.")])
    asyncio.run(run(llm, tool, system_for(), utt="배송 언제 와요"))
    assert [n for n, _ in tool.seen] == ["get_order_info"]


# ── 발신 고객 주문으로 교정·되묻기 ─────────────────────
def test_repair_uses_order_info_by_caller_phone():
    llm = ScriptLLM([calls(("cancel_order", {"order_id": "20260924003542", "reason": "변심"})), say("접수됐어요.")])
    tool = FakeTools()
    asyncio.run(run(llm, tool, system_for(CALLER)))
    assert tool.seen[0] == ("get_order_info", {"buyer_cellphone": "01047775671"})
    name, args = tool.seen[-1]
    assert name == "cancel_order" and _digits(args["order_id"]) == "202609240035429"


def test_confirm_uses_caller_orders_with_korean_keys(monkeypatch):
    monkeypatch.setenv("DIRECTON_CONFIRM_ORDER_ID", "1")
    llm = ScriptLLM([calls(("get_order_info", {"order_id": "20260924-0035428"})), say("확인해 볼게요.")])
    tool = FakeTools()
    asyncio.run(run(llm, tool, system_for(CALLER), utt="주문 조회해 주세요"))
    body = tool_body(llm)
    assert body["status"] == "needs_confirmation"
    assert body["candidates"][0]["product_name"] == "하루온 비타민"
    assert _digits(body["candidates"][0]["order_id"]) == "202609240035429"


# ── 결과 템플릿(한글 키) ───────────────────────────────
def test_template_reads_single_order_delivery_status():
    res = {"status": "ok", "data": {"orders": [ORDERS[1]]}, "message": "주문 조회 완료"}
    assert tool_result_template("get_order_info", res) == "확인해 보니 현재 배송중 상태입니다."


def test_template_skips_multiple_orders():
    res = {"status": "ok", "data": {"orders": list(ORDERS)}, "message": "주문 조회 완료"}
    assert tool_result_template("get_order_info", res) is None
