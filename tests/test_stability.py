"""AG-T18 외부 접속·안정성: TTS 대체(edge → Gemini), 세션 예외 격리, 부하 도구의 순수 부분. 네트워크 없음.

- FallbackTTS: 1차가 첫 덩어리 전에 실패·무응답·빈 출력이면 그 문장은 2차로 이어 말하고, 1차를 쿨다운 동안 건너뛴다.
- 세션 격리: 한 세션의 예외(STT·LLM·TTS·함수·턴 결정·송신)가 다른 세션과 서버를 죽이지 않는다.
  실패한 턴도 가능하면 캐시 사과 + agent.transcript + turn.end로 끝난다.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
from types import SimpleNamespace as NS

import httpx
import numpy as np
import pytest

from agent.config import Settings
from agent.core.types import LLMEvent, TurnDecision

# ══════════════════════════════════════════════════════
# FallbackTTS
# ══════════════════════════════════════════════════════
class ScriptTTS:
    """mode: ok | raise | empty | hang | raise_after_first. 호출 기록을 남긴다."""

    def __init__(self, name: str, mode: str = "ok", chunks: list[bytes] | None = None, voice: str = ""):
        self.name, self.mode = name, mode
        self.chunks = chunks if chunks is not None else [name.encode() * 2, b"\x01\x00" * 4]
        self.requests: list[str] = []
        self.closed = 0
        self.voice = voice or f"{name}-voice"

    async def synthesize(self, text):
        self.requests.append(text)
        try:
            if self.mode == "raise":
                raise RuntimeError(f"{self.name} down")
            if self.mode == "hang":
                await asyncio.Event().wait()
            if self.mode == "empty":
                return
            for i, c in enumerate(self.chunks):
                await asyncio.sleep(0)
                yield c
                if self.mode == "raise_after_first" and i == 0:
                    raise RuntimeError(f"{self.name} broke mid-sentence")
        finally:
            self.closed += 1


async def collect(agen) -> list[bytes]:
    return [c async for c in agen]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t
















def _tone(amp: float, seconds: float = 0.6) -> bytes:
    t = np.arange(int(16000 * seconds)) / 16000
    return np.round(amp * np.sin(2 * np.pi * 220 * t)).astype(np.int16).tobytes()






# ── factory ──────────────────────────────────────────






# ══════════════════════════════════════════════════════
# 세션 격리 (fake deps, 서버 deps 공유, 4세션 동시)
# ══════════════════════════════════════════════════════
SPEECH = b"\x10\x27" * 320          # 20ms, 진폭 10000
SILENCE = b"\x00\x00" * 320
OPENER_PCM = b"\x11\x22" * 1600
APOLOGY_PCM = b"\x33\x44" * 1600
REPROMPT_PCM = b"\x77\x66" * 1600
FILLER_PCM = b"\x55\x66" * 1600
TOOLS = [{"type": "function", "function": {
    "name": "get_order_info", "description": "주문 조회",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}}]

FAULT: contextvars.ContextVar[str] = contextvars.ContextVar("fault", default="")


class Boom(RuntimeError):
    pass


class StabWS:
    """측정기 흉내: session.start → 발화 프레임 → turn.end까지 무음 → 몇 프레임 뒤 종료.
    send_fail: 처음 n번의 바이너리 송신에서 예외(ConnectionClosed가 아닌 오류)."""

    def __init__(self, sid: str, speech: int = 40, send_fail: int = 0, max_frames: int = 20000):
        self.sid = sid
        self._start = json.dumps({"type": "session.start", "session_id": sid, "case_id": sid,
                                  "now": "2026-11-21 11:00", "history": [], "caller": None,
                                  "tools_url": "http://mock:8100",
                                  "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}})
        self.speech = speech
        self.started = False
        self.idx = 0
        self.send_fail = send_fail
        self.max_frames = max_frames
        self.sent: list[tuple[int, object]] = []
        self.turn_ends = 0
        self.after_end = 0

    async def recv(self):
        return await self.__anext__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if not self.started:
            self.started = True
            return self._start
        if self.turn_ends:
            if self.after_end >= 5:
                raise StopAsyncIteration
            self.after_end += 1
        if self.idx > self.max_frames:
            raise StopAsyncIteration
        if self.idx > 3000:
            await asyncio.sleep(0.001)
        self.idx += 1
        return SPEECH if self.idx <= self.speech else SILENCE

    async def send(self, msg):
        if isinstance(msg, bytes) and self.send_fail > 0:
            self.send_fail -= 1
            raise OSError("socket write failed")
        self.sent.append((self.idx, msg))
        if isinstance(msg, str) and json.loads(msg).get("type") == "turn.end":
            self.turn_ends += 1

    def binaries(self):
        return [(i, m) for i, m in self.sent if isinstance(m, bytes)]

    def texts(self):
        return [json.loads(m) for _, m in self.sent if isinstance(m, str)]


class StabVAD:
    def __init__(self):
        if FAULT.get() == "vad_init":
            raise Boom("vad init failed")

    def process(self, frame):
        return 1.0 if any(frame) else 0.0

    def reset(self):
        pass


class StabDetector:
    def decide(self, s):
        if s.silence_ms >= 300:
            if FAULT.get() == "detector":
                raise Boom("detector crashed")
            return TurnDecision(True, "silence")
        return TurnDecision(False, "wait")


class StabStream:
    def __init__(self, text):
        self.text = text

    async def send_audio(self, pcm):
        if FAULT.get() == "stt_send":
            raise Boom("stt send failed")

    def latest_text(self):
        return self.text

    async def finalize(self, timeout=1.0):
        if FAULT.get() == "stt_final":
            raise Boom("stt finalize failed")
        await asyncio.sleep(0)
        return self.text

    async def close(self):
        pass


class StabSTT:
    async def open_stream(self, context=""):
        if FAULT.get() == "stt_open":
            raise Boom("stt open failed")
        return StabStream("주문 2611150231 조회해 주세요")


class StabLLM:
    model = "stab"

    async def stream(self, system, messages, tools):
        f = FAULT.get()
        if f == "llm":
            raise Boom("llm down")
        await asyncio.sleep(0.01)
        if f == "tool" and not any(m.role == "tool" for m in messages):
            from agent.core.types import ToolCall
            yield LLMEvent("tool_call", tool_call=ToolCall("c1", "get_order_info", {"order_id": "2611150231"}))
            yield LLMEvent("done")
            return
        yield LLMEvent("text", text="확인해 드렸습니다. 내일 도착합니다.")
        yield LLMEvent("done")


class StabTTS:
    voice = "stab"

    async def synthesize(self, text):
        if FAULT.get() == "tts":
            raise Boom("tts down")
        await asyncio.sleep(0)
        yield b"\x10\x27" * 4 + text.encode("utf-8")


def stab_deps(tmp_path):
    from agent.session import SessionDeps
    from agent.tools_client import ToolsClient
    from agent.tts_cache import TTSCache

    s = Settings(filler_delay_ms=700, safety_timeout_s=5.0, llm_timeout_s=5.0, tts_timeout_s=5.0)
    cache = TTSCache(tmp_path / "cache")
    for p in s.cached_phrases():
        cache.put(p, FILLER_PCM if p in s.fillers else OPENER_PCM)
    cache.put(s.apology, APOLOGY_PCM)
    cache.put(s.reprompt, REPROMPT_PCM)

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=TOOLS)
        if FAULT.get() == "tool":
            raise httpx.ConnectError("mock api down")
        return httpx.Response(200, json={"status": "ok"})

    tools = ToolsClient(http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return SessionDeps(stt=StabSTT(), tts=StabTTS(), llm=StabLLM(), vad_factory=StabVAD,
                       smart_turn=NS(predict=lambda pcm: 0.9), detector_factory=StabDetector,
                       tools=tools, cache=cache, settings=s, log_writer=None)


async def _run_with_fault(ws, deps, fault: str):
    from agent.server import handle

    FAULT.set(fault)                        # 이 태스크(과 그 하위 태스크)에만 보이는 고장
    await asyncio.wait_for(handle(ws, deps), 15.0)


def _assert_turn_completed(ws: StabWS, *, speech_frames: int):
    texts = ws.texts()
    assert texts and texts[-1] == {"type": "turn.end"}, (ws.sid, texts)
    assert sum(1 for t in texts if t.get("type") == "turn.end") == 1, ws.sid
    assert any(t.get("type") == "agent.transcript" for t in texts), ws.sid
    bins = ws.binaries()
    assert bins, ws.sid
    assert min(i for i, _ in bins) > speech_frames, ws.sid    # 고객 발화가 끝나기 전에는 소리를 보내지 않는다


@pytest.mark.parametrize("fault", ["llm", "tts", "tool", "stt_open", "stt_final", "stt_send", "detector",
                                   "vad_init"])
async def test_four_concurrent_sessions_survive_one_fault(tmp_path, fault):
    """서버 deps를 공유하는 4세션 동시 처리. 하나에 고장을 넣어도 나머지 3세션은 정상 응답,
    고장 난 세션도 예외를 밖으로 내지 않고 turn.end로 끝난다(LLM·TTS·턴 결정 실패 시 캐시 사과)."""
    deps = stab_deps(tmp_path)
    wss = [StabWS(f"S-{i}") for i in range(4)]
    faults = ["", fault, "", ""]
    res = await asyncio.gather(*(_run_with_fault(w, deps, f) for w, f in zip(wss, faults)),
                               return_exceptions=True)
    assert [r for r in res if isinstance(r, BaseException)] == []
    for w, f in zip(wss, faults):
        _assert_turn_completed(w, speech_frames=w.speech)
        joined = " ".join(t.get("text", "") for t in w.texts() if t.get("type") == "agent.transcript")
        if not f:
            assert "내일 도착합니다" in joined, w.sid
    bad = wss[1]
    if fault in ("llm", "tts", "detector", "vad_init"):
        assert APOLOGY_PCM in [m for _, m in bad.binaries()], fault
        assert deps.settings.apology in " ".join(t.get("text", "") for t in bad.texts())


async def test_session_crash_mid_turn_still_sends_apology_and_turn_end(tmp_path):
    """첫 마디 송신이 (연결 종료가 아닌) 오류로 실패해 세션이 죽어도, 턴은 사과 + turn.end로 닫힌다."""
    deps = stab_deps(tmp_path)
    good = [StabWS(f"G-{i}") for i in range(3)]
    bad = StabWS("BAD", send_fail=1)
    res = await asyncio.gather(*(_run_with_fault(w, deps, "") for w in [bad, *good]), return_exceptions=True)
    assert [r for r in res if isinstance(r, BaseException)] == []
    for w in good:
        _assert_turn_completed(w, speech_frames=w.speech)
    _assert_turn_completed(bad, speech_frames=bad.speech)
    assert APOLOGY_PCM in [m for _, m in bad.binaries()]


async def test_handle_never_raises_when_connection_is_broken(tmp_path):
    """송신이 계속 실패(망가진 연결)해도 handle은 조용히 끝난다(서버 로그만)."""
    from agent.server import handle

    deps = stab_deps(tmp_path)
    ws = StabWS("DEAD", send_fail=10_000, max_frames=600)
    await asyncio.wait_for(handle(ws, deps), 15.0)


async def test_real_websocket_server_keeps_serving_after_session_crash(tmp_path):
    """실제 websockets 서버(127.0.0.1, 임시 포트)에서 세션 하나가 죽어도 다음 연결을 받는다."""
    import websockets

    from agent.server import handle

    deps = stab_deps(tmp_path)
    deps.vad_factory = lambda: (_ for _ in ()).throw(Boom("vad init failed"))   # 모든 세션 생성 실패
    async with websockets.serve(lambda ws: handle(ws, deps), "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        for _ in range(2):
            async with websockets.connect(f"ws://127.0.0.1:{port}") as c:
                await c.send(json.dumps({"type": "session.start", "session_id": "X"}))
                for k in range(40 + 80):
                    await c.send(SPEECH if k < 40 else SILENCE)
                got = []
                async with asyncio.timeout(5):
                    async for m in c:
                        if isinstance(m, str):
                            got.append(json.loads(m))
                            if got[-1].get("type") == "turn.end":
                                break
                assert got[-1] == {"type": "turn.end"}


# ══════════════════════════════════════════════════════
# eval/stress.py 순수 부분
# ══════════════════════════════════════════════════════










# ══════════════════════════════════════════════════════
# 시험 음성 저장 금지 모드: 콘솔 로그에도 고객 발화 원문을 남기지 않는다
# (콘솔을 파일로 돌리면 디스크에 남는다 → 실격 위험. DIRECTON_NO_AUDIO_LOG=1)
# ══════════════════════════════════════════════════════
class _RecheckStream:
    def __init__(self, interim, final):
        self.interim, self.final = interim, final

    async def send_audio(self, pcm):
        pass

    def latest_text(self):
        return self.interim

    async def finalize(self, timeout=1.0):
        return self.final

    async def close(self):
        pass


class _RecheckSTT:
    def __init__(self, interim, final):
        self.interim, self.final = interim, final

    async def open_stream(self, context=""):
        return _RecheckStream(self.interim, self.final)


async def _recheck_logs(caplog, interim, final, recheck=None):
    import logging

    from agent.providers.stt_recheck import DigitRecheckSTT

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="agent.providers.stt_recheck"):
        s = await DigitRecheckSTT(_RecheckSTT(interim, final), recheck, timeout=0.5).open_stream()
        await s.send_audio(b"\x01\x00" * 10)
        text = await s.finalize(1.0)
    return text, "\n".join(r.getMessage() for r in caplog.records)




def test_openai_bad_args_log_omits_text_when_no_audio_log(caplog, monkeypatch):
    import logging

    from agent.providers.openai_llm import _parse_args

    monkeypatch.setenv("DIRECTON_NO_AUDIO_LOG", "1")
    with caplog.at_level(logging.DEBUG, logger="agent.providers.openai_llm"):
        assert _parse_args('{"order_id": "2611200517", "name": "홍길동"') == {}
    logs = "\n".join(r.getMessage() for r in caplog.records)
    assert logs and "2611200517" not in logs and "홍길동" not in logs
