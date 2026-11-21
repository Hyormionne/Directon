"""받아쓰기가 빈 채 끝난 턴(2026-11-21 공식 측정 P0029-01·P0120-01): 말소리가 충분했으면 끝맺음을 한 번 더 요청하고
글을 잠깐 기다린다. 그래도 비면 지금처럼 다시 묻는다. fake는 tests/test_session.py의 것을 쓴다.
"""
import asyncio
import json
import time

from agent.config import Settings
from test_session import (OPENER_PCM, REPROMPT_PCM, FakeLLM, FakeWS, make_deps, pcm_for, run, start_msg,
                          text_events, utterance)

TEXT = "하루온 혈관 청소 구입 방법이 궁금해요."
ANSWER = "온라인이나 전화로 주문하실 수 있어요."


class LateStream:
    """첫 끝맺음은 빈 글로 끝난다. 글은 그로부터 at_s초 뒤에야 생긴다(None이면 끝까지 없다).
    fin_waits=True면 두 번째 끝맺음이 글이 올 때까지(시간 한도 안에서) 기다린다 — 늦게 온 <fin>."""

    def __init__(self, text=TEXT, at_s=None, fin_waits=False):
        self.text, self.at_s, self.fin_waits = text, at_s, fin_waits
        self.t0 = None
        self.finalize_timeouts = []
        self.closed = False

    def _now(self):
        return asyncio.get_running_loop().time()

    async def send_audio(self, pcm):
        pass

    def latest_text(self):
        if self.t0 is None or self.at_s is None:
            return ""
        return self.text if self._now() - self.t0 >= self.at_s else ""

    async def finalize(self, timeout=1.0):
        self.finalize_timeouts.append(timeout)
        await asyncio.sleep(0)
        if self.t0 is None:
            self.t0 = self._now()
            return ""
        if self.fin_waits:
            end = self._now() + timeout
            while self._now() < end and not self.latest_text():
                await asyncio.sleep(0.005)
        return self.latest_text()

    async def close(self):
        self.closed = True


class LateSTT:
    def __init__(self, **kw):
        self.kw = kw
        self.streams = []

    async def open_stream(self, context=""):
        s = LateStream(**self.kw)
        self.streams.append(s)
        return s


def deps_for(tmp_path, ws, llm, stt, **settings_kw):
    kw = {"empty_stt_wait_s": 0.4, "empty_stt_min_speech_ms": 400, **settings_kw}
    return make_deps(tmp_path, ws=ws, llm=llm, stt=stt, settings_kw=kw)


def test_settings_defaults_and_env():
    s = Settings()
    assert s.empty_stt_wait_s == 1.5 and s.empty_stt_min_speech_ms == 400
    e = Settings.from_env({"DIRECTON_EMPTY_STT_WAIT_S": "0", "DIRECTON_EMPTY_STT_MIN_SPEECH_MS": "900"})
    assert e.empty_stt_wait_s == 0.0 and e.empty_stt_min_speech_ms == 900
    assert s.finalize_timeout_s == 1.0 and s.safety_timeout_s == 20.0       # 기존 값은 그대로


async def test_late_fin_after_empty_finalize_is_used_for_the_answer(tmp_path):
    ws = FakeWS(start_msg(), utterance(50))
    llm = FakeLLM(text_events(ANSWER))
    stt = LateSTT(at_s=0.1, fin_waits=True)
    deps, det = deps_for(tmp_path, ws, llm, stt)
    await run(ws, deps)

    stream = stt.streams[0]
    assert stream.finalize_timeouts == [deps.settings.finalize_timeout_s, 0.4]   # 끝맺음을 한 번 더 요청했다
    assert len(llm.calls) == 1 and llm.calls[0]["messages"][-1].content == TEXT
    bins = ws.binaries()
    assert all(i >= det.decided_at[0] for i, _ in bins)                 # 고객 말이 끝나기 전에는 소리를 내지 않는다
    assert [m for _, m in bins] == [OPENER_PCM, pcm_for(ANSWER)]          # 다시 묻는 문구는 나가지 않는다
    assert ws.texts()[-2] == {"type": "agent.transcript", "text": f"네, 고객님. {ANSWER}"}
    assert ws.texts()[-1] == {"type": "turn.end"}


async def test_text_that_shows_up_in_latest_text_is_used(tmp_path):
    """두 번째 끝맺음도 빈 채 바로 끝났지만 조금 뒤 부분 결과가 생긴 경우: 폴링으로 받아 쓴다."""
    ws = FakeWS(start_msg(), utterance(50))
    llm = FakeLLM(text_events(ANSWER))
    stt = LateSTT(at_s=0.1, fin_waits=False)
    deps, _ = deps_for(tmp_path, ws, llm, stt, empty_stt_wait_s=0.6)
    await run(ws, deps)
    assert len(llm.calls) == 1 and llm.calls[0]["messages"][-1].content == TEXT
    assert [m for _, m in ws.binaries()] == [OPENER_PCM, pcm_for(ANSWER)]


async def test_still_empty_after_wait_plays_reprompt(tmp_path):
    ws = FakeWS(start_msg(), utterance(50))
    llm = FakeLLM(text_events("x."))
    stt = LateSTT(at_s=None)
    deps, _ = deps_for(tmp_path, ws, llm, stt)
    t0 = time.perf_counter()
    await run(ws, deps)
    took = time.perf_counter() - t0
    assert llm.calls == []
    assert len(stt.streams[0].finalize_timeouts) == 2                    # 한 번만 더 기다린다
    assert 0.4 <= took < 0.4 + 0.25 + 0.5                                # 기다림은 설정값(+여유 0.25초)을 넘지 않는다
    assert [m for _, m in ws.binaries()] == [OPENER_PCM, REPROMPT_PCM]
    assert ws.texts()[-1] == {"type": "turn.end"}


async def test_no_extra_wait_for_short_sound(tmp_path):
    """말소리가 짧으면(기준 미만) 잡음일 수 있다: 기다리지 않고 바로 다시 묻는다."""
    ws = FakeWS(start_msg(), utterance(10))                             # 200ms
    stt = LateSTT(at_s=0.05, fin_waits=True)
    deps, _ = deps_for(tmp_path, ws, FakeLLM(text_events("x.")), stt)
    await run(ws, deps)
    assert len(stt.streams[0].finalize_timeouts) == 1
    assert [m for _, m in ws.binaries()] == [OPENER_PCM, REPROMPT_PCM]


async def test_wait_can_be_turned_off(tmp_path):
    ws = FakeWS(start_msg(), utterance(50))
    stt = LateSTT(at_s=0.05, fin_waits=True)
    deps, _ = deps_for(tmp_path, ws, FakeLLM(text_events("x.")), stt, empty_stt_wait_s=0.0)
    await run(ws, deps)
    assert len(stt.streams[0].finalize_timeouts) == 1
    assert [m for _, m in ws.binaries()] == [OPENER_PCM, REPROMPT_PCM]


async def test_turn_log_records_retry_without_customer_text(tmp_path):
    from agent.turn_log import TurnLogWriter

    ws = FakeWS(start_msg(), utterance(50))
    stt = LateSTT(at_s=0.1, fin_waits=True)
    deps, _ = deps_for(tmp_path, ws, FakeLLM(text_events(ANSWER)), stt)
    deps.log_writer = TurnLogWriter(tmp_path / "logs")
    await run(ws, deps)
    rec = json.loads(next((tmp_path / "logs").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    ev = [e for e in rec["events"] if e["name"] == "stt_empty_retry"]
    assert len(ev) == 1 and ev[0]["chars"] == len(TEXT) and ev[0]["waited_ms"] >= 90
    assert "text" not in ev[0]                                           # 글 내용은 이 이벤트에 넣지 않는다(글자 수만)
    assert rec["user_text"] == TEXT
