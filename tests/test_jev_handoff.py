"""AG-T24 Jev 사람 인계 판정: 요청 모양, 기준별 임계값, 시간 초과 → 강제 인계 없음, 스위치 끔 = 기존 동작.

외부 호출 없음: typesafe_sdk 클라이언트에 httpx2.MockTransport를 주입한다. 근거: docs/decisions/jev_handoff.md
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from agent import handoff_jev as hj


def _resp(**probs) -> dict:
    p = {k: 0.02 for k in hj.CRITERIA}
    p.update(probs)
    return {"model": hj.MODEL, "usage": {"input_tokens": 900, "output_tokens": 80},
            "answers": {k: {"type": "noul", "noul": v} for k, v in p.items()}}


def _client(handler) -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(handler),
                               retry=RetryPolicy(max_retries=0))


def test_criteria_cover_required_conditions():
    assert set(hj.CRITERIA) == {"health", "human_request", "abuse_anger", "repeated_failure", "legal_dispute",
                                "out_of_policy"}
    assert set(hj.THRESHOLDS) == set(hj.CRITERIA)
    assert all(0 < v < 1 for v in hj.THRESHOLDS.values())


async def test_request_shape():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx2.Response(200, json=_resp(human_request=0.97))

    j = hj.JevHandoff(client=_client(handler), timeout_s=2.0)
    hist = [{"role": "customer", "text": "벌써 세 번째 물어보는데"}, {"role": "agent", "text": "불편을 드려 죄송합니다."}]
    r = await j.judge("상담원 바꿔 주세요.", history=hist, failures=2)
    b = seen["body"]
    assert b["model"] == hj.MODEL
    assert set(b["questions"]) == set(hj.CRITERIA)
    assert all(q["type"] == "noul" for q in b["questions"].values())
    st = b["state"]
    assert st["customer_utterance"] == "상담원 바꿔 주세요."
    assert st["recent_conversation"][-1] == {"role": "agent", "text": "불편을 드려 죄송합니다."}
    assert st["agent_failed_to_understand_count"] == 2
    assert r is not None and r.handoff and r.reasons == ["human_request"]
    assert r.probs["human_request"] == pytest.approx(0.97)


def test_decide_per_criterion_threshold():
    th = dict(hj.THRESHOLDS)
    lo = {k: 0.0 for k in hj.CRITERIA}
    assert hj.decide(lo, th) == (False, [])
    for k in hj.CRITERIA:
        p = dict(lo, **{k: th[k]})
        assert hj.decide(p, th) == (True, [k])                        # 임계값 이상이면 그 기준만으로 인계
        p = dict(lo, **{k: th[k] - 0.01})
        assert hj.decide(p, th) == (False, [])
    both = dict(lo, health=0.99, abuse_anger=0.99)
    ok, why = hj.decide(both, th)
    assert ok and why == ["health", "abuse_anger"]                    # CRITERIA 순서
    assert hj.decide({"health": 0.99}, th) == (True, ["health"])      # 빠진 답은 0으로 본다


async def test_timeout_no_forced_handoff():
    async def handler(req):
        await asyncio.sleep(1.0)
        return httpx2.Response(200, json=_resp(health=0.99))

    j = hj.JevHandoff(client=_client(handler), timeout_s=0.05)
    assert await j.judge("먹고 나서 두드러기가 났어요.") is None


async def test_error_no_forced_handoff():
    j = hj.JevHandoff(client=_client(lambda r: httpx2.Response(503, json={"detail": "x"})), timeout_s=1.0)
    assert await j.judge("먹고 나서 두드러기가 났어요.") is None


def test_transfer_reason_and_hint():
    r = hj.HandoffJudgment(probs={"health": 0.95}, handoff=True, reasons=["health"], ms=100.0)
    assert "섭취" in hj.transfer_reason(r) or "몸" in hj.transfer_reason(r)
    hint = hj.system_hint(r)
    assert hint and "transfer_to_agent" not in hint and "함수를 부르지" in hint   # AG-T38: 인계는 말로
    assert hj.system_hint(hj.HandoffJudgment(probs={}, handoff=False, reasons=[], ms=1.0)) == ""
    assert hj.system_hint(None) == ""


def test_route_business_hours():
    yes = hj.HandoffJudgment(probs={"human_request": 0.99}, handoff=True, reasons=["human_request"], ms=1.0)
    no = hj.HandoffJudgment(probs={}, handoff=False, reasons=[], ms=1.0)
    assert hj.route(None, "2026-11-20 10:00") is None and hj.route(no, "2026-11-20 10:00") is None
    assert hj.route(yes, "2026-11-20 10:00") == "transfer"          # 금 오전
    assert hj.route(yes, "2026-11-20 12:20") == "callback"          # 점심
    assert hj.route(yes, "2026-11-20 19:40") == "callback"          # 영업 끝
    assert hj.route(yes, "2026-11-21 10:30") == "callback"          # 토
    assert hj.route(yes, "") == "transfer"                          # 모르면 인계
    off, on = hj.system_hint(yes, "2026-11-21 10:30"), hj.system_hint(yes, "2026-11-20 10:00")
    assert "전화 문의 방법" in off and "연결 시간이 아니" in off          # AG-T38: 영업시간 밖은 전화 문의 안내
    assert "request_callback" not in off and "transfer_to_agent" not in on


async def test_switch_off_no_call(monkeypatch):
    monkeypatch.delenv("DIRECTON_JEV_HANDOFF", raising=False)
    assert hj.handoff_enabled() is False
    calls = []
    j = hj.JevHandoff(client=_client(lambda r: calls.append(1) or httpx2.Response(200, json=_resp())))
    assert await hj.maybe_judge(j, "상담원 바꿔 주세요.") is None
    assert calls == []
    monkeypatch.setenv("DIRECTON_JEV_HANDOFF", "1")
    assert hj.handoff_enabled() is True
    r = await hj.maybe_judge(j, "상담원 바꿔 주세요.")
    assert r is not None and calls == [1]


async def test_empty_utterance_no_call():
    calls = []
    j = hj.JevHandoff(client=_client(lambda r: calls.append(1) or httpx2.Response(200, json=_resp())))
    assert await j.judge("") is None and calls == []


@pytest.mark.live
async def test_live_handoff():
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY 없음")
    j = hj.JevHandoff(timeout_s=5.0)
    a = await j.judge("홍삼 먹고 나서 두드러기가 났어요. 계속 먹어도 돼요?")
    b = await j.judge("배송 언제 와요?")
    assert a is not None and a.handoff and "health" in a.reasons
    assert b is not None and not b.handoff
