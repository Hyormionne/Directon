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


async def test_fallback_tts_uses_primary_when_healthy():
    from agent.providers.fallback_tts import FallbackTTS

    p, s = ScriptTTS("p"), ScriptTTS("s")
    tts = FallbackTTS(p, s)
    out = await collect(tts.synthesize("안녕하세요."))
    assert out == p.chunks and s.requests == []
    assert tts.stats.get("served:primary") == 1


@pytest.mark.parametrize("mode", ["raise", "empty"])
async def test_fallback_tts_switches_to_secondary_before_first_chunk(mode):
    from agent.providers.fallback_tts import FallbackTTS

    p, s = ScriptTTS("p", mode), ScriptTTS("s")
    tts = FallbackTTS(p, s)
    out = await collect(tts.synthesize("배송 안내해 드릴게요."))
    assert out == s.chunks
    assert p.requests == s.requests == ["배송 안내해 드릴게요."]
    assert p.closed == 1                                   # 1차 스트림을 닫았다


async def test_fallback_tts_switches_on_first_chunk_timeout():
    from agent.providers.fallback_tts import FallbackTTS

    p, s = ScriptTTS("p", "hang"), ScriptTTS("s")
    tts = FallbackTTS(p, s, first_chunk_timeout_s=0.05)
    out = await asyncio.wait_for(collect(tts.synthesize("잠시만요.")), 2.0)
    assert out == s.chunks and p.closed == 1
    assert any(k.startswith("fallback:timeout") for k in tts.stats)


async def test_fallback_tts_skips_primary_during_cooldown_then_retries():
    from agent.providers.fallback_tts import FallbackTTS

    clock = Clock()
    p, s = ScriptTTS("p", "raise"), ScriptTTS("s")
    tts = FallbackTTS(p, s, cooldown_s=30.0, clock=clock)
    await collect(tts.synthesize("첫 문장."))
    assert not tts.primary_healthy()
    await collect(tts.synthesize("둘째 문장."))
    assert p.requests == ["첫 문장."]                      # 쿨다운 중에는 1차를 부르지 않는다
    assert s.requests == ["첫 문장.", "둘째 문장."]
    clock.t += 31.0
    p.mode = "ok"
    out = await collect(tts.synthesize("셋째 문장."))
    assert out == p.chunks and tts.primary_healthy()


async def test_fallback_tts_failure_after_first_chunk_raises_without_duplicate():
    """첫 덩어리를 보낸 뒤의 실패는 2차로 다시 말하지 않는다(같은 말 반복 방지). 1차는 쿨다운."""
    from agent.providers.fallback_tts import FallbackTTS

    p, s = ScriptTTS("p", "raise_after_first"), ScriptTTS("s")
    tts = FallbackTTS(p, s)
    got: list[bytes] = []
    with pytest.raises(RuntimeError):
        async for c in tts.synthesize("길게 말하는 문장입니다."):
            got.append(c)
    assert got == p.chunks[:1] and s.requests == []
    assert not tts.primary_healthy()


async def test_fallback_tts_empty_text_calls_nobody():
    from agent.providers.fallback_tts import FallbackTTS

    p, s = ScriptTTS("p"), ScriptTTS("s")
    assert await collect(FallbackTTS(p, s).synthesize("  ")) == []
    assert p.requests == s.requests == []


async def test_fallback_tts_secondary_failure_propagates():
    from agent.providers.fallback_tts import FallbackTTS

    tts = FallbackTTS(ScriptTTS("p", "raise"), ScriptTTS("s", "raise"))
    with pytest.raises(RuntimeError):
        await collect(tts.synthesize("둘 다 실패."))


def _tone(amp: float, seconds: float = 0.6) -> bytes:
    t = np.arange(int(16000 * seconds)) / 16000
    return np.round(amp * np.sin(2 * np.pi * 220 * t)).astype(np.int16).tobytes()


async def test_level_normalization_holds_through_fallback():
    """2차(조용한 목소리)로 넘어가도 LevelTTS가 같은 레벨(음성 구간 RMS≈3000)로 맞춘다. 16k s16le 짝수 바이트."""
    from agent.providers.fallback_tts import FallbackTTS
    from agent.tts_cache import LevelTTS, speech_rms

    quiet = _tone(300.0)
    s = ScriptTTS("s", chunks=[quiet[i:i + 2048] for i in range(0, len(quiet), 2048)])
    tts = LevelTTS(FallbackTTS(ScriptTTS("p", "raise"), s), target_rms=3000.0)
    out = b"".join(await collect(tts.synthesize("조용한 대체 음성.")))
    assert len(out) % 2 == 0 and len(out) == len(quiet)
    assert 2500 <= speech_rms(out) <= 3500


def test_cache_voice_key_unchanged_by_fallback_wrapper():
    """FallbackTTS로 감싸도 캐시 네임스페이스는 1차(edge) 기준 그대로(기존 캐시 재사용)."""
    from agent.providers.edge_tts import EdgeTTS
    from agent.providers.fallback_tts import FallbackTTS
    from agent.server import _cache_voice_key
    from agent.tts_cache import LevelTTS

    s = Settings()
    edge = EdgeTTS()
    wrapped = LevelTTS(FallbackTTS(edge, ScriptTTS("gem")), target_rms=3000.0)
    assert _cache_voice_key(s, wrapped) == _cache_voice_key(s, LevelTTS(edge, target_rms=3000.0))


# ── factory ──────────────────────────────────────────
# AG-T46: 기본 TTS는 대회 지정 Gemini(대체 없음). edge+Gemini 대체 구성은 명시적으로 켠다.
_EDGE_FB = {"DIRECTON_TTS_PROVIDER": "edge", "DIRECTON_TTS_FALLBACK": "gemini"}

def test_factory_tts_with_fallback_edge_then_gemini():
    from agent.providers.edge_tts import EdgeTTS
    from agent.providers.factory import ProviderConfig, make_tts_with_fallback
    from agent.providers.fallback_tts import FallbackTTS
    from agent.providers.gemini_tts import GeminiTTS

    fake = NS()
    tts = make_tts_with_fallback(ProviderConfig.from_env({**_EDGE_FB, "GEMINI_API_KEY": "g"}), gemini_client=fake)
    assert isinstance(tts, FallbackTTS)
    assert isinstance(tts.primary, EdgeTTS) and isinstance(tts.secondary, GeminiTTS)
    assert tts.secondary.client is fake
    assert tts.voice == tts.primary.voice == "ko-KR-SunHiNeural"
    t2 = make_tts_with_fallback(ProviderConfig.from_env({**_EDGE_FB, "GEMINI_API_KEY": "g", "TTS_FALLBACK_TIMEOUT_S": "1.5",
                                                         "TTS_FALLBACK_COOLDOWN_S": "10"}), gemini_client=fake)
    assert t2.first_chunk_timeout_s == 1.5 and t2.cooldown_s == 10.0


def test_factory_tts_with_fallback_plain_when_no_gemini_key_or_disabled():
    from agent.providers.edge_tts import EdgeTTS
    from agent.providers.factory import ProviderConfig, make_tts_with_fallback
    from agent.providers.gemini_tts import GeminiTTS

    assert isinstance(make_tts_with_fallback(ProviderConfig.from_env({**_EDGE_FB})), EdgeTTS)
    # BAZE는 비활성(토큰 소진) — BAZE 키만 있으면 대체 TTS를 붙이지 않는다(BAZE 경로는 그대로)
    assert isinstance(make_tts_with_fallback(ProviderConfig.from_env({**_EDGE_FB, "BAZE_API_KEY": "bz"})), EdgeTTS)
    off = ProviderConfig.from_env({**_EDGE_FB, "GEMINI_API_KEY": "g", "DIRECTON_TTS_FALLBACK": "none"})
    assert isinstance(make_tts_with_fallback(off, gemini_client=NS()), EdgeTTS)
    g = ProviderConfig.from_env({**_EDGE_FB, "GEMINI_API_KEY": "g", "DIRECTON_TTS_PROVIDER": "gemini"})
    assert isinstance(make_tts_with_fallback(g, gemini_client=NS()), GeminiTTS)   # 1차가 이미 Gemini


def test_build_deps_prefers_tts_with_fallback(monkeypatch, tmp_path):
    import agent.providers.factory as factory
    import agent.turn.smart_turn as st
    import agent.turn.vad as vad
    from agent.server import build_deps
    from agent.tts_cache import LevelTTS

    sentinel = ScriptTTS("fb")
    monkeypatch.setattr(factory, "make_stt", lambda *a, **k: NS())
    monkeypatch.setattr(factory, "make_llm_with_fallback", lambda *a, **k: NS(model="m"))
    monkeypatch.setattr(factory, "make_tts_with_fallback", lambda *a, **k: sentinel)
    monkeypatch.setattr(vad, "SileroVAD", lambda: NS(process=lambda f: 0.0, reset=lambda: None))
    monkeypatch.setattr(st, "SmartTurnV3", lambda: NS(predict=lambda p: 0.5))
    monkeypatch.delattr(vad, "make_vad", raising=False)
    monkeypatch.delattr(st, "make_smart_turn", raising=False)
    monkeypatch.delattr(st, "SmartTurnONNX", raising=False)
    deps = build_deps(Settings(tts_cache_dir=str(tmp_path / "c"), log_dir=str(tmp_path / "l")))
    assert isinstance(deps.tts, LevelTTS) and deps.tts.inner is sentinel


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
def test_stress_percentile_and_summary():
    from eval import stress

    assert stress.percentile([], 50) is None
    assert stress.percentile([1.0], 95) == 1.0
    assert stress.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    runs = [
        {"round": 0, "case_id": "A", "first_sound": 1.0, "turn_end": 3.0, "spoke_before_end": False},
        {"round": 0, "case_id": "B", "first_sound": 2.0, "turn_end": 4.0, "spoke_before_end": False},
        {"round": 1, "case_id": "C", "error": "ConnectionRefusedError()"},
        {"round": 1, "case_id": "D", "first_sound": None, "turn_end": None, "spoke_before_end": False},
        {"round": 1, "case_id": "E", "first_sound": -0.1, "turn_end": 2.0, "spoke_before_end": True},
    ]
    mem = [{"t": 0, "rss_mb": 300.0}, {"t": 1, "rss_mb": 340.0}, {"t": 2, "rss_mb": 320.0}]
    s = stress.summarize(runs, mem)
    assert s["sessions"] == 5 and s["ok"] == 3
    assert s["failures"] == 2 and s["errors"] == 1 and s["no_turn_end"] == 1
    assert s["spoke_before_end"] == 1
    assert s["first_sound_p50"] == pytest.approx(1.0)
    assert s["mem_start_mb"] == 300.0 and s["mem_peak_mb"] == 340.0 and s["mem_end_mb"] == 320.0
    assert s["mem_growth_mb"] == pytest.approx(20.0)
    assert stress.summarize([], [])["first_sound_p50"] is None


def test_stress_pick_cases_rotates_without_repeating_within_round():
    from eval import stress

    cases = [{"case_id": f"T{i}"} for i in range(6)]
    r0 = stress.pick_cases(cases, 4, 0)
    r1 = stress.pick_cases(cases, 4, 1)
    assert [c["case_id"] for c in r0] == ["T0", "T1", "T2", "T3"]
    assert [c["case_id"] for c in r1] == ["T4", "T5", "T0", "T1"]
    assert len(stress.pick_cases(cases[:2], 4, 0)) == 4               # 문항이 모자라면 돌려 쓴다


def test_stress_first_voiced_and_spoke_before_end():
    from eval import stress

    loud = b"\x10\x27" * 320
    quiet = b"\x01\x00" * 320
    frames = [(-0.5, quiet), (0.2, quiet), (0.8, loud), (1.0, loud)]
    assert stress.first_voiced(frames) == 0.8
    assert stress.first_voiced([(0.1, quiet)]) is None
    assert stress.first_voiced([(-0.3, loud)]) == -0.3


def test_stress_pid_from_netstat_windows():
    from eval import stress

    win = """
  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:8100           0.0.0.0:0              LISTENING       1111
  TCP    0.0.0.0:9000           0.0.0.0:0              LISTENING       4242
  TCP    127.0.0.1:9000         127.0.0.1:50000        ESTABLISHED     4242
  TCP    [::]:19000             [::]:0                 LISTENING       7
"""
    assert stress.pid_from_netstat(win, 9000) == 4242
    assert stress.pid_from_netstat(win, 9001) is None


def test_stress_parse_args_defaults():
    from eval import stress

    a = stress.parse_args(["--server", "ws://127.0.0.1:9000"])
    assert a.sessions == 4 and a.rounds >= 1
    assert a.cases.endswith("cases.jsonl") and a.audio.endswith("audio")


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


@pytest.mark.parametrize("path", ["interim", "batch"])
async def test_stt_recheck_logs_omit_customer_text_when_no_audio_log(caplog, monkeypatch, path):
    async def recheck(pcm, context=""):
        return "주문번호 2611200517 취소해 주세요"

    interim = "2611 180042 취소해" if path == "interim" else "주문번호 261200517 취소해 주세요"
    final = "26111800042 취소해 주세요." if path == "interim" else "주문번호 261200517 취소해 주세요"
    rc = None if path == "interim" else recheck
    monkeypatch.setenv("DIRECTON_NO_AUDIO_LOG", "1")
    text, logs = await _recheck_logs(caplog, interim, final, rc)
    assert "2611" in text                                   # 동작은 그대로(교체됨)
    assert logs                                             # 교체 사실은 남긴다
    assert "취소" not in logs and "2611" not in logs and "180042" not in logs and "261200517" not in logs
    monkeypatch.delenv("DIRECTON_NO_AUDIO_LOG")
    _, logs2 = await _recheck_logs(caplog, interim, final, rc)
    assert "취소" in logs2                                  # 개발 중에는 원문 로그 유지


def test_openai_bad_args_log_omits_text_when_no_audio_log(caplog, monkeypatch):
    import logging

    from agent.providers.openai_llm import _parse_args

    monkeypatch.setenv("DIRECTON_NO_AUDIO_LOG", "1")
    with caplog.at_level(logging.DEBUG, logger="agent.providers.openai_llm"):
        assert _parse_args('{"order_id": "2611200517", "name": "홍길동"') == {}
    logs = "\n".join(r.getMessage() for r in caplog.records)
    assert logs and "2611200517" not in logs and "홍길동" not in logs
