"""bench.naive_server(AG-T13) 프로토콜 순서. fake STT/LLM/TTS/VAD, Mock API는 httpx.MockTransport(네트워크 없음)."""
import asyncio
import json

import httpx

from agent.core.types import LLMEvent, ToolCall
from agent.tools_client import ToolsClient
from bench.naive_server import NaiveConfig, NaiveDeps, build_system, handle
from agent.core.types import SessionInfo

SPEECH = b"\x10\x27" * 320          # 20ms, 유성(fake VAD가 음성으로 봄)
SILENCE = bytes(640)
TOOLS = [{"type": "function", "function": {"name": "get_order_info", "description": "주문 조회",
          "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}}]
START = {"type": "session.start", "session_id": "S-1", "case_id": "T1", "now": "2026-11-20 10:00",
         "history": [{"role": "agent", "text": "주문번호를 말씀해 주세요."}], "caller": None,
         "customer_profile": {"name": "김민수", "phone": "01012345678"},
         "tools_url": "http://mock", "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}}


class FakeWS:
    """프레임 목록을 차례로 주고, 다 쓰면 turn.end 받을 때까지 무음, 그 뒤 몇 프레임 후 끝."""

    def __init__(self, frames, tail=1000):
        self.frames = list(frames)
        self.idx = 0
        self.started = False
        self.sent: list[tuple[int, object]] = []
        self.ended = False
        self.after = 0
        self.tail = tail

    async def recv(self):
        return await self.__anext__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if not self.started:
            self.started = True
            return json.dumps(START, ensure_ascii=False)
        if self.ended:
            self.after += 1
            if self.after > 3:
                raise StopAsyncIteration
        if self.idx >= len(self.frames) + self.tail:
            raise StopAsyncIteration
        f = self.frames[self.idx] if self.idx < len(self.frames) else SILENCE
        self.idx += 1
        if self.idx > len(self.frames):
            await asyncio.sleep(0.001)      # 말 끝 뒤에는 실시간처럼 천천히(응답이 끼어들 틈)
        return f

    async def send(self, msg):
        self.sent.append((self.idx, msg))
        if isinstance(msg, str) and json.loads(msg).get("type") == "turn.end":
            self.ended = True

    def texts(self):
        return [json.loads(m) for _, m in self.sent if isinstance(m, str)]


class FakeVAD:
    def process(self, frame):
        return 1.0 if any(frame) else 0.0

    def reset(self):
        pass


class FakeStream:
    def __init__(self, text):
        self.text, self.n = text, 0

    async def send_audio(self, pcm):
        self.n += 1

    def latest_text(self):
        return self.text

    async def finalize(self, timeout=1.0):
        return self.text

    async def close(self):
        pass


class FakeSTT:
    def __init__(self, text="주문 조회해 주세요"):
        self.stream = FakeStream(text)

    async def open_stream(self, context=""):
        return self.stream


class FakeLLM:
    """1회차: 툴 호출(있으면), 2회차: 텍스트."""

    def __init__(self, call=True):
        self.call = call
        self.calls: list[tuple[str, list, list]] = []

    async def stream(self, system, messages, tools):
        self.calls.append((system, list(messages), tools))
        if self.call and len(self.calls) == 1:
            yield LLMEvent("tool_call", tool_call=ToolCall("c1", "get_order_info", {"order_id": "이육일일"}))
        else:
            yield LLMEvent("text", text="배송 중이에요. ")
            yield LLMEvent("text", text="내일 도착해요.")
        yield LLMEvent("done")


class FakeTTS:
    def __init__(self):
        self.texts: list[str] = []

    async def synthesize(self, text):
        self.texts.append(text)
        yield b"\x10\x27" * 1600          # 100ms


def make_deps(llm=None, tts=None, handler=None, cfg=None):
    seen: list[httpx.Request] = []

    def default(req: httpx.Request):
        seen.append(req)
        if req.method == "GET":
            return httpx.Response(200, json=TOOLS)
        return httpx.Response(200, json={"status": "ok", "data": {"state": "배송중"}})

    tools = ToolsClient(http=httpx.AsyncClient(transport=httpx.MockTransport(handler or default)))
    deps = NaiveDeps(stt=FakeSTT(), llm=llm or FakeLLM(), tts=tts or FakeTTS(), tools=tools,
                     vad_factory=FakeVAD, manual="매뉴얼 본문", cfg=cfg or NaiveConfig())
    return deps, seen


async def test_no_send_before_800ms_silence_then_audio_transcript_turn_end():
    n_speech = 50
    ws = FakeWS([SILENCE] * 10 + [SPEECH] * n_speech)
    deps, seen = make_deps()
    await asyncio.wait_for(handle(ws, deps), 10)
    end_idx = 10 + n_speech + 40                       # 800ms = 40프레임 침묵째에서 말 끝
    assert ws.sent, "응답이 없음"
    assert min(i for i, _ in ws.sent) >= end_idx      # 800ms 침묵 전에는 아무것도 안 보냄
    kinds = ["pcm" if isinstance(m, bytes) else json.loads(m)["type"] for _, m in ws.sent]
    assert kinds[-2:] == ["agent.transcript", "turn.end"]
    assert set(kinds[:-2]) == {"pcm"}
    assert ws.texts()[0]["text"] == "배송 중이에요. 내일 도착해요."
    # 전체 텍스트로 TTS 1회(문장 분할 없음)
    assert deps.tts.texts == ["배송 중이에요. 내일 도착해요."]


async def test_silence_shorter_than_800ms_does_not_end_turn():
    # 음성 - 780ms 쉼 - 음성: 쉼에서 끊지 않는다
    frames = [SPEECH] * 30 + [SILENCE] * 39 + [SPEECH] * 30
    ws = FakeWS(frames)
    deps, _ = make_deps()
    await asyncio.wait_for(handle(ws, deps), 10)
    assert min(i for i, _ in ws.sent) >= len(frames) + 40


async def test_tool_call_has_session_header_and_raw_args():
    ws = FakeWS([SPEECH] * 30)
    llm = FakeLLM()
    deps, seen = make_deps(llm=llm)
    await asyncio.wait_for(handle(ws, deps), 10)
    posts = [r for r in seen if r.method == "POST"]
    assert len(posts) == 1
    assert posts[0].url.path == "/tools/get_order_info"
    assert posts[0].headers["X-Session-Id"] == "S-1"
    assert json.loads(posts[0].content) == {"order_id": "이육일일"}   # 인자 정규화 없음
    # 2회차 LLM에 툴 결과가 들어감, 툴 목록은 GET /tools 그대로
    assert llm.calls[0][2] == TOOLS
    assert [m.role for m in llm.calls[1][1]] == ["user", "assistant", "tool"]


async def test_turn_end_sent_on_llm_error():
    class BoomLLM:
        async def stream(self, system, messages, tools):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    ws = FakeWS([SPEECH] * 30)
    deps, _ = make_deps(llm=BoomLLM())
    await asyncio.wait_for(handle(ws, deps), 10)
    assert ws.texts()[-1] == {"type": "turn.end"}


async def test_sentence_mode_splits_tts():
    ws = FakeWS([SPEECH] * 30)
    deps, _ = make_deps(llm=FakeLLM(call=False), cfg=NaiveConfig(tts_mode="sentence"))
    await asyncio.wait_for(handle(ws, deps), 10)
    assert deps.tts.texts == ["배송 중이에요.", "내일 도착해요."]
    assert ws.texts()[-2:] == [{"type": "agent.transcript", "text": "배송 중이에요. 내일 도착해요."},
                               {"type": "turn.end"}]


def test_system_prompt_is_role_manual_and_raw_context():
    info = SessionInfo.from_start(START)
    info.caller = START["customer_profile"]
    s = build_system(info, "매뉴얼 본문")
    assert "매뉴얼 본문" in s and "2026-11-20 10:00" in s and "01012345678" in s and "주문번호를 말씀해" in s
