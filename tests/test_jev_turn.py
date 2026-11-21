"""AG-T24 Jev 턴 종료 판정: 요청 모양, 확률 파싱, 시간 초과 → 표 결정기, 스위치 끔 = 기존 동작.

외부 호출 없음: typesafe_sdk 클라이언트에 httpx2.MockTransport를 주입한다. 실제 호출은 @pytest.mark.live.
근거: docs/decisions/jev_turn.md
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from agent.core.types import TurnSignals
from agent.turn import detector as dm
from agent.turn import jev_turn as jt


def _answers(finished=0.9, cut_off=0.1, more=0.2) -> dict:
    return {"model": jt.MODEL, "usage": {"input_tokens": 700, "output_tokens": 54},
            "answers": {"finished": {"type": "noul", "noul": finished},
                        "cut_off": {"type": "noul", "noul": cut_off},
                        "more_expected": {"type": "noul", "noul": more}}}


def _client(handler) -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(handler),
                               retry=RetryPolicy(max_retries=0))


# ── 요청 모양 ──
async def test_request_shape_and_parse():
    seen = {}

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization", "")
        seen["body"] = json.loads(req.content)
        return httpx2.Response(200, json=_answers(0.88, 0.07, 0.31))

    j = jt.JevTurnJudge(client=_client(handler), timeout_s=2.0)
    hist = [{"role": "agent", "text": "주문번호 말씀해 주시겠어요?"}]
    r = await j.judge("이육일일 일삼공", history=hist, caller_known=True)
    assert seen["url"].endswith("/v1/systemone")
    assert seen["auth"] == "Bearer test-key"
    b = seen["body"]
    assert b["model"] == jt.MODEL
    assert set(b["questions"]) == {"finished", "cut_off", "more_expected"}
    assert all(q["type"] == "noul" and q["instructions"] and set(q["criteria"]) == {"true", "false"}
               for q in b["questions"].values())
    st = b["state"]
    assert st["customer_speech_so_far"] == "이육일일 일삼공"
    assert st["previous_conversation"] == [{"role": "agent", "text": "주문번호 말씀해 주시겠어요?"}]
    assert st["caller_already_identified"] is True
    assert r is not None and r.finished == pytest.approx(0.88) and r.cut_off == pytest.approx(0.07)
    assert r.more_expected == pytest.approx(0.31) and r.ms >= 0




def test_parse_raw_dict_and_missing():
    r = jt.parse_answers(_answers(0.4, 0.6, 0.7))
    assert (r.finished, r.cut_off, r.more_expected) == (0.4, 0.6, 0.7)
    assert jt.parse_answers({"answers": {}}) is None
    flat = jt.parse_answers({"answers": {"finished": 0.3, "cut_off": 0.5, "more_expected": 0.5}})  # 캐시 형식
    assert flat.finished == 0.3


async def test_timeout_returns_none():
    async def handler(req):
        await asyncio.sleep(1.0)
        return httpx2.Response(200, json=_answers())

    j = jt.JevTurnJudge(client=_client(handler), timeout_s=0.05)
    t0 = asyncio.get_running_loop().time()
    assert await j.judge("주문번호가요") is None
    assert asyncio.get_running_loop().time() - t0 < 0.5


async def test_http_error_returns_none():
    j = jt.JevTurnJudge(client=_client(lambda req: httpx2.Response(500, json={"detail": "x"})), timeout_s=1.0)
    assert await j.judge("취소해 주세요.") is None


async def test_empty_text_no_call():
    calls = []
    j = jt.JevTurnJudge(client=_client(lambda req: calls.append(1) or httpx2.Response(200, json=_answers())))
    assert await j.judge("  ") is None
    assert calls == []


# ── 대기 결합 정책 ──
def test_combine_wait():
    pol = jt.JevTurnPolicy(hi=0.85, lo=0.25, fast_ms=400)
    assert jt.combine_wait(1200, 1300, None, pol) == (1200, "")
    assert jt.combine_wait(1200, 1300, 0.9, pol)[0] == 400           # 확실히 끝 → 짧게
    assert jt.combine_wait(300, 1300, 0.9, pol)[0] == 300            # 표가 더 짧으면 표
    assert jt.combine_wait(900, 1300, 0.1, pol)[0] == 1300           # 확실히 중간 → 상한까지
    assert jt.combine_wait(900, 2000, 0.1, pol)[0] == 2000
    assert jt.combine_wait(900, 1300, 0.5, pol) == (900, "")


# ── 결정기 래퍼 ──
SIGS = [TurnSignals(silence_ms=s, speech_ms=sp, text=t, expect_short=es)
        for s in (0, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1800, 2100)
        for sp in (100, 900, 2000, 3000)
        for t in ("", "네", "취소해 주세요.", "그래서", "이육일일", "배송 언제 와요?", "아 잠깐만요")
        for es in (False, True)]


def test_wrapper_without_p_identical_to_table():
    base = dm.TableTurnDetector(dm.CALIBRATED_CONFIG)
    w = jt.JevTurnDetector(dm.TableTurnDetector(dm.CALIBRATED_CONFIG), judge=None)
    for s in SIGS:
        assert w.decide(s) == base.decide(s), s
    assert w.config is dm.CALIBRATED_CONFIG
    assert w.wait_ms(SIGS[5]) == base.wait_ms(SIGS[5])


def test_wrapper_uses_supplied_p_for_same_text_only():
    w = jt.JevTurnDetector(dm.TableTurnDetector(dm.CALIBRATED_CONFIG), judge=None,
                           policy=jt.JevTurnPolicy(hi=0.85, lo=0.25, fast_ms=400))
    t = "취소해 주세요."
    s = TurnSignals(silence_ms=420, speech_ms=900, text=t)
    assert not w.decide(s).end                       # 표: final s1 1250ms
    w.set_jev(t, 0.95)
    d = w.decide(s)
    assert d.end and "jev" in d.reason
    # 텍스트가 바뀌면(부분 결과 갱신) 옛 p는 쓰지 않는다
    assert not w.decide(TurnSignals(silence_ms=440, speech_ms=900, text="취소해 주세요. 아니")).end


def test_wrapper_low_p_extends_to_cap():
    w = jt.JevTurnDetector(dm.TableTurnDetector(dm.CALIBRATED_CONFIG), judge=None,
                           policy=jt.JevTurnPolicy(hi=0.85, lo=0.25, fast_ms=400))
    t = "배송 언제 와요?"                                 # question s0 600ms
    w.set_jev(t, 0.1)
    assert not w.decide(TurnSignals(silence_ms=700, speech_ms=500, text=t)).end
    assert w.decide(TurnSignals(silence_ms=1300, speech_ms=500, text=t)).end   # 상한은 그대로


def test_wrapper_resets_on_new_silence():
    w = jt.JevTurnDetector(dm.TableTurnDetector(dm.CALIBRATED_CONFIG), judge=None)
    w.set_jev("취소해 주세요.", 0.99)
    w.decide(TurnSignals(silence_ms=0, speech_ms=900, text="취소해 주세요."))   # 음성 재개
    assert w.latest() is None


class FakeJudge:
    def __init__(self, result=None, delay=0.0, exc=None):
        self.calls, self.result, self.delay, self.exc = [], result, delay, exc

    async def judge(self, text, history=None, caller_known=False):
        self.calls.append((text, history, caller_known))
        await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.result


async def test_wrapper_fires_request_at_200ms_and_consumes():
    fj = FakeJudge(jt.JevTurnResult(0.95, 0.05, 0.1, 5.0))
    w = jt.JevTurnDetector(dm.TableTurnDetector(dm.CALIBRATED_CONFIG), judge=fj,
                           policy=jt.JevTurnPolicy(hi=0.85, lo=0.25, fast_ms=400))
    w.set_context([{"role": "agent", "text": "더 도와드릴까요?"}], caller_known=True)
    t = "취소해 주세요."
    assert not w.decide(TurnSignals(silence_ms=180, speech_ms=900, text=t)).end
    assert fj.calls == []                                     # 200ms 전에는 안 부른다
    w.decide(TurnSignals(silence_ms=200, speech_ms=900, text=t))
    w.decide(TurnSignals(silence_ms=220, speech_ms=900, text=t))
    await asyncio.sleep(0)                                    # 작업 시작
    assert len(fj.calls) == 1 and fj.calls[0][0] == t and fj.calls[0][2] is True
    await asyncio.sleep(0.01)
    assert w.decide(TurnSignals(silence_ms=420, speech_ms=900, text=t)).end


async def test_wrapper_judge_failure_falls_back_to_table():
    base = dm.TableTurnDetector(dm.CALIBRATED_CONFIG)
    for fj in (FakeJudge(None), FakeJudge(exc=RuntimeError("boom")), FakeJudge(jt.JevTurnResult(0.99, 0, 0, 1), delay=5)):
        w = jt.JevTurnDetector(dm.TableTurnDetector(dm.CALIBRATED_CONFIG), judge=fj)
        t = "취소해 주세요."
        for ms in range(200, 1400, 20):
            s = TurnSignals(silence_ms=ms, speech_ms=900, text=t)
            assert w.decide(s) == base.decide(s)
            await asyncio.sleep(0)
        w.close()


def test_wrapper_no_loop_no_crash():
    fj = FakeJudge(jt.JevTurnResult(0.99, 0, 0, 1))
    w = jt.JevTurnDetector(dm.TableTurnDetector(dm.CALIBRATED_CONFIG), judge=fj)
    s = TurnSignals(silence_ms=300, speech_ms=900, text="취소해 주세요.")
    assert w.decide(s) == dm.TableTurnDetector(dm.CALIBRATED_CONFIG).decide(s)


# ── 스위치 ──
def test_make_detector_switch(monkeypatch):
    monkeypatch.delenv("DIRECTON_JEV_TURN", raising=False)
    d = dm.make_detector()
    assert type(d) is dm.TableTurnDetector and d.config is dm.CALIBRATED_CONFIG
    monkeypatch.setenv("DIRECTON_JEV_TURN", "0")
    assert type(dm.make_detector()) is dm.TableTurnDetector
    monkeypatch.setenv("DIRECTON_JEV_TURN", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    d = dm.make_detector()                     # 생성 시 네트워크 호출 없음(클라이언트는 첫 요청 때 만든다)
    assert isinstance(d, jt.JevTurnDetector) and d.config is dm.CALIBRATED_CONFIG


def test_make_detector_switch_without_key_falls_back(monkeypatch):
    monkeypatch.setenv("DIRECTON_JEV_TURN", "1")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jt, "_dotenv_key", lambda: None)
    assert type(dm.make_detector()) is dm.TableTurnDetector


# ── 실제 호출 ──
@pytest.mark.live
async def test_live_jev_turn():
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY 없음")
    j = jt.JevTurnJudge(timeout_s=5.0)
    done = await j.judge("주문 취소해 주세요.")
    mid = await j.judge("제 주문번호가 이육일일")
    assert done is not None and mid is not None
    assert done.finished > mid.finished
