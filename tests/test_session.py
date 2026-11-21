"""AG-T07 session/server: 프레임 수신 → 턴 감지 → 첫 마디 → 본 응답 → transcript → turn.end.

시간은 프레임 수 기준(20ms/프레임)이라 실시간 대기 없이 돈다.
FakeWS.idx = 지금까지 우리 서버에 전달한 PCM 프레임 수. 송신 메시지마다 그 시점의 idx를 기록한다.
"""
import asyncio
import json
import time

import httpx
import pytest

from agent.config import Settings
from agent.core.types import LLMEvent, TurnDecision, ToolCall
from agent.server import handle
from agent.session import SessionDeps
from agent.tools_client import ToolsClient
from agent.tts_cache import TTSCache
from agent.turn_log import TurnLogWriter

SPEECH = b"\x10\x27" * 320          # 20ms, 진폭 10000
SILENCE = b"\x00\x00" * 320
OPENER = "네, 고객님."
OPENER_PCM = b"\x11\x22" * 1600     # 100ms = 송신 청크 1개
APOLOGY_PCM = b"\x33\x44" * 1600
FILLER_PCM = b"\x55\x66" * 1600
REPROMPT_PCM = b"\x77\x66" * 1600
TOOLS = [{"type": "function", "function": {
    "name": "get_order_info", "description": "주문 조회",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}}]


def pcm_for(text: str) -> bytes:
    return b"\x10\x27" * 4 + text.encode("utf-8")


# ── fakes ────────────────────────────────────────────
class FakeWS:
    """측정기 흉내. segments[i]는 i번째 turn.end를 받은 뒤에야 시작되는 발화 프레임들(사이는 무음)."""

    def __init__(self, start: dict, *segments, after_end_frames=5, turns=1, fast_frames=4000):
        self._start = json.dumps(start, ensure_ascii=False)
        self.segments = [list(s) for s in segments]
        self.seg_i = 0
        self.seg_pos = 0
        self.started = False
        self.idx = 0
        self.sent: list[tuple[int, object]] = []
        self.turn_ends = 0
        self.turns = turns
        self.after_end_frames = after_end_frames
        self.frames_after_last_end = 0
        self.fast_frames = fast_frames

    async def recv(self):
        return await self.__anext__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if not self.started:
            self.started = True
            return self._start
        if self.turn_ends >= self.turns:
            if self.frames_after_last_end >= self.after_end_frames:
                raise StopAsyncIteration
            self.frames_after_last_end += 1
        if self.idx > self.fast_frames:
            await asyncio.sleep(0.001)
        if self.idx > 200000:
            raise StopAsyncIteration
        frame = SILENCE
        if self.seg_i < len(self.segments) and self.seg_i <= self.turn_ends:
            seg = self.segments[self.seg_i]
            if self.seg_pos < len(seg):
                frame = seg[self.seg_pos]
                self.seg_pos += 1
            if self.seg_pos >= len(seg):
                self.seg_i += 1
                self.seg_pos = 0
        self.idx += 1
        return frame

    async def send(self, msg):
        self.sent.append((self.idx, msg))
        if isinstance(msg, str) and json.loads(msg).get("type") == "turn.end":
            self.turn_ends += 1

    # helpers
    def binaries(self):
        return [(i, m) for i, m in self.sent if isinstance(m, bytes)]

    def texts(self):
        return [json.loads(m) for _, m in self.sent if isinstance(m, str)]


class FakeVAD:
    def __init__(self):
        self.resets = 0

    def process(self, frame):
        return 1.0 if any(frame) else 0.0

    def reset(self):
        self.resets += 1


class FakeSmartTurn:
    def __init__(self):
        self.calls = 0

    def predict(self, pcm):
        self.calls += 1
        return 0.9


class FakeDetector:
    def __init__(self, ws=None, end_ms=600):
        self.ws = ws
        self.end_ms = end_ms
        self.signals = []
        self.decided_at = []

    def decide(self, s):
        self.signals.append(s)
        if s.silence_ms >= self.end_ms:
            self.decided_at.append(self.ws.idx if self.ws else None)
            return TurnDecision(True, "silence")
        return TurnDecision(False, "wait")


class FakeStream:
    def __init__(self, partial, final):
        self.partial, self.final = partial, final
        self.audio = 0
        self.finalize_calls = 0
        self.closed = False

    async def send_audio(self, pcm):
        self.audio += 1

    def latest_text(self):
        return self.partial if self.audio > 5 else ""

    async def finalize(self, timeout=1.0):
        self.finalize_calls += 1
        await asyncio.sleep(0)
        return self.final

    async def close(self):
        self.closed = True


class FakeSTT:
    def __init__(self, partial="배송 언제 와요?", final=None, texts=None):
        self.texts = texts or [(partial, final if final is not None else partial)]
        self.streams = []

    async def open_stream(self, context=""):
        p, f = self.texts[min(len(self.streams), len(self.texts) - 1)]
        s = FakeStream(p, f)
        self.streams.append(s)
        return s


class FakeLLM:
    def __init__(self, *scripts, ws=None):
        self.scripts = list(scripts)
        self.calls = []
        self.cancelled = []
        self.ws = ws

    async def stream(self, system, messages, tools):
        n = len(self.calls)
        self.calls.append({"system": system, "messages": list(messages), "tools": tools,
                           "idx": self.ws.idx if self.ws else None})
        script = self.scripts[min(n, len(self.scripts) - 1)]
        try:
            if isinstance(script, Exception):
                raise script
            if script == "hang":
                await asyncio.Event().wait()
            for ev in script:
                await asyncio.sleep(0)
                yield ev
        except asyncio.CancelledError:
            self.cancelled.append(n)
            raise
        except GeneratorExit:
            self.cancelled.append(n)
            raise


def text_events(*chunks):
    return [LLMEvent("text", text=c) for c in chunks] + [LLMEvent("done")]


class FakeTTS:
    def __init__(self, fail=False):
        self.fail = fail
        self.requests = []

    async def synthesize(self, text):
        self.requests.append(text)
        if self.fail:
            raise RuntimeError("tts down")
        await asyncio.sleep(0)
        yield pcm_for(text)


def start_msg(**kw):
    m = {"type": "session.start", "session_id": "S-1", "case_id": "C1", "now": "2026-11-21 11:00",
         "history": [], "caller": None, "tools_url": "http://mock:8100",
         "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}}
    m.update(kw)
    return m


def utterance(speech=50, silence=0):
    return [SPEECH] * speech + [SILENCE] * silence


def make_deps(tmp_path, *, ws, llm, stt=None, tts=None, detector=None, settings_kw=None,
              tool_handler=None, log_writer=None):
    s = Settings(**{"filler_delay_ms": 700, "safety_timeout_s": 5.0, "llm_timeout_s": 5.0,
                    "tts_timeout_s": 5.0, **(settings_kw or {})})
    cache = TTSCache(tmp_path / "cache")
    cache.put(OPENER, OPENER_PCM)
    cache.put(s.apology, APOLOGY_PCM)
    cache.put(s.reprompt, REPROMPT_PCM)
    for f in s.fillers:
        cache.put(f, FILLER_PCM)

    def default_handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=TOOLS)
        return httpx.Response(200, json={"status": "ok"})

    tools = ToolsClient(http=httpx.AsyncClient(transport=httpx.MockTransport(tool_handler or default_handler)))
    det = detector or FakeDetector(ws)
    return SessionDeps(stt=stt or FakeSTT(), tts=tts or FakeTTS(), llm=llm,
                       vad_factory=FakeVAD, smart_turn=FakeSmartTurn(),
                       detector_factory=lambda: det, tools=tools, cache=cache, settings=s,
                       log_writer=log_writer), det


async def run(ws, deps):
    await asyncio.wait_for(handle(ws, deps), 10.0)


# ── 테스트 ───────────────────────────────────────────
async def test_no_audio_before_end_then_opener_first(tmp_path):
    ws = FakeWS(start_msg(), utterance(50))
    llm = FakeLLM(text_events("주문하신 상품은 내일 도착합니다."))
    deps, det = make_deps(tmp_path, ws=ws, llm=llm)
    await run(ws, deps)

    # 결정기: 600ms(30프레임) 전에는 계속 False
    assert all(not (s.silence_ms >= 600) for s in det.signals[:-1])
    assert det.decided_at == [80]                    # 음성 50 + 무음 30
    bins = ws.binaries()
    assert bins, "no audio sent"
    assert all(i >= 80 for i, _ in bins)             # 확정 전 송신 0
    assert bins[0] == (80, OPENER_PCM)               # 확정 즉시 첫 마디
    audio = b"".join(m for _, m in bins)
    assert audio == OPENER_PCM + pcm_for("주문하신 상품은 내일 도착합니다.")

    texts = ws.texts()
    assert texts[-1] == {"type": "turn.end"}
    assert texts[-2]["type"] == "agent.transcript"
    assert texts[-2]["text"] == "네, 고객님. 주문하신 상품은 내일 도착합니다."
    assert ws.sent[-1][1] == json.dumps({"type": "turn.end"})   # 마지막 메시지
    # turn.end 이후 프레임도 읽어서 버렸다(drain)
    assert ws.frames_after_last_end == ws.after_end_frames


async def test_precall_at_200ms_and_reused_when_final_text_same(tmp_path):
    ws = FakeWS(start_msg(), utterance(50))
    llm = FakeLLM(text_events("네 확인해 드리겠습니다."), ws=ws)
    deps, det = make_deps(tmp_path, ws=ws, llm=llm)
    await run(ws, deps)
    assert len(llm.calls) == 1
    assert 60 <= llm.calls[0]["idx"] < 80            # 침묵 200ms(10프레임)에 선호출, 확정 전
    assert llm.calls[0]["messages"][-1].content == "배송 언제 와요?"
    assert llm.calls[0]["tools"] == TOOLS
    assert deps.stt.streams[0].finalize_calls >= 1
    assert deps.smart_turn.calls >= 1


async def test_recall_when_final_text_differs(tmp_path):
    ws = FakeWS(start_msg(), utterance(50))
    llm = FakeLLM(text_events("앞부분 답."), text_events("배송은 내일 도착합니다."), ws=ws)
    stt = FakeSTT(partial="배송 언제", final="배송 언제 와요?")
    deps, _ = make_deps(tmp_path, ws=ws, llm=llm, stt=stt)
    await run(ws, deps)
    assert len(llm.calls) == 2
    assert llm.calls[1]["messages"][-1].content == "배송 언제 와요?"
    assert ws.texts()[-2]["text"] == "네, 고객님. 배송은 내일 도착합니다."


async def test_speech_resume_cancels_precall(tmp_path):
    frames = utterance(30, 15) + utterance(20)       # 300ms 쉬고 다시 말함
    ws = FakeWS(start_msg(), frames)
    llm = FakeLLM("hang", text_events("네 확인해 드리겠습니다."), ws=ws)
    deps, det = make_deps(tmp_path, ws=ws, llm=llm)
    await run(ws, deps)
    assert 0 in llm.cancelled                        # 첫 선호출 취소
    assert len(llm.calls) == 2
    assert det.decided_at == [30 + 15 + 20 + 30]
    assert all(i >= det.decided_at[0] for i, _ in ws.binaries())
    assert ws.texts()[-1] == {"type": "turn.end"}


async def test_llm_error_plays_cached_apology(tmp_path):
    ws = FakeWS(start_msg(), utterance(50))
    deps, _ = make_deps(tmp_path, ws=ws, llm=FakeLLM(RuntimeError("boom")))
    await run(ws, deps)
    bins = [m for _, m in ws.binaries()]
    assert bins[0] == OPENER_PCM and APOLOGY_PCM in bins
    texts = ws.texts()
    assert texts[-1] == {"type": "turn.end"}
    assert deps.settings.apology in texts[-2]["text"]


async def test_tts_error_plays_cached_apology(tmp_path):
    ws = FakeWS(start_msg(), utterance(50))
    deps, _ = make_deps(tmp_path, ws=ws, llm=FakeLLM(text_events("안내해 드리겠습니다.")),
                        tts=FakeTTS(fail=True))
    await run(ws, deps)
    bins = [m for _, m in ws.binaries()]
    assert bins[0] == OPENER_PCM and bins[-1] == APOLOGY_PCM
    assert ws.texts()[-1] == {"type": "turn.end"}


async def test_safety_cap_ends_turn(tmp_path):
    ws = FakeWS(start_msg(), utterance(50), fast_frames=200)
    deps, _ = make_deps(tmp_path, ws=ws, llm=FakeLLM("hang"),
                        settings_kw={"safety_timeout_s": 0.3, "llm_timeout_s": 30.0})
    t0 = time.perf_counter()
    await run(ws, deps)
    assert time.perf_counter() - t0 < 3.0
    assert APOLOGY_PCM in [m for _, m in ws.binaries()]
    assert ws.texts()[-1] == {"type": "turn.end"}


async def test_tool_call_uses_session_id_and_tools_url(tmp_path):
    seen = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=TOOLS)
        seen.append((str(req.url), req.headers.get("X-Session-Id"), json.loads(req.content)))
        return httpx.Response(200, json={"status": "ok", "delivery": "배송중"})

    ws = FakeWS(start_msg(session_id="S-77", tools_url="http://mock:8100/"), utterance(50))
    llm = FakeLLM([LLMEvent("tool_call", tool_call=ToolCall("c1", "get_order_info", {"order_id": "20261115-0000231"})),
                   LLMEvent("done")], text_events("배송 중입니다."))
    deps, _ = make_deps(tmp_path, ws=ws, llm=llm, tool_handler=handler)
    await run(ws, deps)
    assert seen == [("http://mock:8100/tools/get_order_info", "S-77", {"order_id": "20261115-0000231"})]
    assert ws.texts()[-2]["text"].endswith("배송 중입니다.")


async def test_empty_transcript_plays_reprompt_without_llm(tmp_path):
    ws = FakeWS(start_msg(), utterance(50))
    llm = FakeLLM(text_events("x."))
    deps, _ = make_deps(tmp_path, ws=ws, llm=llm, stt=FakeSTT(partial="", final=""))
    await run(ws, deps)
    assert llm.calls == []
    assert [m for _, m in ws.binaries()] == [OPENER_PCM, REPROMPT_PCM]
    assert ws.texts()[-1] == {"type": "turn.end"}


async def test_expect_short_from_history_question(tmp_path):
    ws = FakeWS(start_msg(history=[{"role": "agent", "text": "주문번호를 알려 주시겠어요?"}]), utterance(20))
    deps, det = make_deps(tmp_path, ws=ws, llm=FakeLLM(text_events("네 확인했습니다.")))
    await run(ws, deps)
    assert det.signals and all(s.expect_short for s in det.signals)


async def test_turn_log_written(tmp_path):
    writer = TurnLogWriter(tmp_path / "logs", date_fn=lambda: "20261121")
    ws = FakeWS(start_msg(), utterance(50))
    deps, _ = make_deps(tmp_path, ws=ws, llm=FakeLLM(text_events("배송 중입니다.")), log_writer=writer)
    await run(ws, deps)
    lines = (tmp_path / "logs" / "turns-20261121.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["session_id"] == "S-1"
    for k in ["first_speech", "silence_start", "finalize_request", "finalize_done", "smart_turn",
              "commit", "opener_sent", "llm_first_token", "turn_end"]:
        assert k in rec["marks"], k
    assert rec["user_text"] == "배송 언제 와요?"
    assert rec["agent_text"] == "네, 고객님. 배송 중입니다."
    assert rec["decision_reason"] == "silence"
    assert rec["marks"]["commit_frame"] == 80


async def test_second_utterance_on_same_connection(tmp_path):
    ws = FakeWS(start_msg(), utterance(30), utterance(30), turns=2)
    llm = FakeLLM(text_events("첫 답변입니다."), text_events("두 번째 답변입니다."))
    stt = FakeSTT(texts=[("첫 질문", "첫 질문"), ("두번째 질문", "두번째 질문")])
    deps, _ = make_deps(tmp_path, ws=ws, llm=llm, stt=stt,
                        detector=FakeDetector(None))
    await run(ws, deps)
    texts = ws.texts()
    assert [t["type"] for t in texts] == ["agent.transcript", "turn.end", "agent.transcript", "turn.end"]
    assert texts[2]["text"] == "네, 고객님. 두 번째 답변입니다."
    assert llm.calls[-1]["messages"][-1].content == "두번째 질문"
