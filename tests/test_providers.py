"""AG-T04 Provider 테스트. 모두 fake client/websocket을 주입한다(네트워크 금지)."""
from __future__ import annotations

import asyncio
import io
import json
import wave
from types import SimpleNamespace as NS

import numpy as np
import pytest

from agent.core.types import LLMEvent, Message, ToolCall


# ── 공용 도우미 ───────────────────────────────────────
def sine_pcm(sr: int, sec: float, hz: float = 440.0) -> bytes:
    t = np.arange(int(sr * sec)) / sr
    return (np.sin(2 * np.pi * hz * t) * 8000).astype(np.int16).tobytes()


async def collect(aiter):
    return [x async for x in aiter]


async def wait_for(cond, timeout=1.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        if loop.time() > end:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.005)


# ══════════════════════════════════════════════════════
# STT — Gemini Live
# ══════════════════════════════════════════════════════
def live_msg(interim=None, final=None):
    return NS(server_content=NS(
        interim_input_transcription=NS(text=interim) if interim is not None else None,
        input_transcription=NS(text=final) if final is not None else None,
        turn_complete=None))


class FakeLiveSession:
    """오디오 바이트 누적량에 따라 interim을, audio_stream_end에 final을 돌려준다."""

    def __init__(self, words, final_on_end=True):
        self.words = words
        self.final_on_end = final_on_end
        self.sent = []
        self.q: asyncio.Queue = asyncio.Queue()
        self.audio_bytes = 0
        self.closed = False

    async def send_realtime_input(self, **kw):
        self.sent.append(kw)
        if kw.get("audio") is not None:
            self.audio_bytes += len(kw["audio"].data)
            n = min(len(self.words), self.audio_bytes // 3200)
            if n:
                self.q.put_nowait(live_msg(interim=" ".join(self.words[:n])))
        if kw.get("audio_stream_end") and self.final_on_end:
            self.q.put_nowait(live_msg(final=" ".join(self.words)))

    async def receive(self):
        while True:
            m = await self.q.get()
            if m is None:
                return
            yield m


class FakeLiveConnect:
    def __init__(self, client, session):
        self.client, self.session = client, session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *a):
        self.session.closed = True
        self.session.q.put_nowait(None)


class FakeLiveClient:
    def __init__(self, words=("주문", "번호", "알려", "주세요"), final_on_end=True, fail=False):
        self.words = list(words)
        self.final_on_end = final_on_end
        self.fail = fail
        self.connects = []
        self.sessions = []
        self.aio = NS(live=NS(connect=self._connect))

    def _connect(self, *, model, config):
        if self.fail:
            raise RuntimeError("connect failed")
        self.connects.append({"model": model, "config": config})
        s = FakeLiveSession(self.words, self.final_on_end)
        self.sessions.append(s)
        return FakeLiveConnect(self, s)








# ══════════════════════════════════════════════════════
# STT — Gemini Batch
# ══════════════════════════════════════════════════════
class FakeModels:
    def __init__(self, gen=None, stream=None):
        self._gen = gen
        self._stream = stream
        self.calls = []

    async def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._gen(model, contents, config)

    async def generate_content_stream(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._stream(model, contents, config, len(self.calls))


def fake_gen_client(gen=None, stream=None):
    models = FakeModels(gen, stream)
    return NS(aio=NS(models=models)), models








# ══════════════════════════════════════════════════════
# STT — Soniox
# ══════════════════════════════════════════════════════
class FakeSonioxWS:
    def __init__(self):
        self.sent = []
        self.q: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, bytes):
            n = sum(len(d) for d in self.sent if isinstance(d, bytes))
            if n == 640 * 5:
                self.push({"tokens": [{"text": "환불", "is_final": True},
                                      {"text": " 하고", "is_final": False}]})
        elif data and json.loads(data).get("type") == "finalize":
            self.push({"tokens": [{"text": " 싶어요", "is_final": True},
                                  {"text": "<fin>", "is_final": True}]})

    def push(self, obj):
        self.q.put_nowait(json.dumps(obj, ensure_ascii=False))

    async def recv(self):
        m = await self.q.get()
        if m is None:
            raise ConnectionError("closed")
        return m

    async def close(self):
        self.closed = True
        self.q.put_nowait(None)


async def test_soniox_stt_protocol():
    from agent.providers.soniox_stt import SonioxSTT

    ws = FakeSonioxWS()
    seen = {}

    async def connect(url, headers):
        seen["url"], seen["headers"] = url, headers
        return ws

    stt = SonioxSTT(api_key="k-test", connect=connect)
    stream = await stt.open_stream(context="주문번호, 반품")
    for _ in range(6):
        await stream.send_audio(b"\x00\x00" * 320)
    await wait_for(lambda: stream.latest_text() == "환불 하고")
    text = await stream.finalize(timeout=1.0)
    assert text == "환불 싶어요"
    assert "<fin>" not in stream.latest_text()
    await stream.close()

    assert seen["url"] == "wss://stt-rt.soniox.com/transcribe-websocket"
    assert seen["headers"] == {"Authorization": "Bearer k-test"}
    start = json.loads(ws.sent[0])
    assert start["model"] == "stt-rt-v6"
    assert start["audio_format"] == "pcm_s16le"
    assert start["sample_rate"] == 16000 and start["num_channels"] == 1
    assert start["language_hints"] == ["ko"]
    assert "api_key" not in start
    assert "반품" in json.dumps(start["context"], ensure_ascii=False)
    assert ws.sent[-1] == ""      # 정상 종료는 빈 텍스트 프레임
    assert ws.closed


async def test_soniox_keepalive_when_idle():
    from agent.providers.soniox_stt import SonioxSTT

    ws = FakeSonioxWS()

    async def connect(url, headers):
        return ws

    stream = await SonioxSTT(api_key="k", connect=connect, keepalive_s=0.05).open_stream()
    await wait_for(lambda: any(isinstance(d, str) and d and json.loads(d).get("type") == "keepalive"
                               for d in ws.sent), timeout=1.0)
    await stream.close()


async def test_soniox_error_message_raises_on_finalize():
    from agent.providers.soniox_stt import SonioxSTT

    ws = FakeSonioxWS()

    async def connect(url, headers):
        return ws

    stream = await SonioxSTT(api_key="k", connect=connect).open_stream()
    ws.push({"error_code": 401, "error_type": "unauthenticated", "error_message": "bad"})
    await asyncio.sleep(0.02)
    # 에러 후에는 finalize가 기다리지 않고 지금까지의 텍스트를 돌려준다
    assert await stream.finalize(timeout=0.5) == ""
    await stream.close()


# ══════════════════════════════════════════════════════
# TTS — Gemini
# ══════════════════════════════════════════════════════
def tts_chunk(data, mime="audio/l16; rate=24000; channels=1"):
    return NS(candidates=[NS(content=NS(parts=[NS(inline_data=NS(data=data, mime_type=mime), text=None)]))])


def make_tts_stream(chunks_list, fail_times=0):
    state = {"n": 0}

    def stream(model, contents, config, ncall):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise RuntimeError("503")

        async def gen():
            for c in chunks_list:
                yield c
        return gen()
    return stream


async def test_gemini_tts_resamples_24k_to_16k_in_order():
    from agent.providers.gemini_tts import GeminiTTS

    pcm24 = sine_pcm(24000, 0.6)
    pieces = [pcm24[:9601], pcm24[9601:20001], pcm24[20001:]]  # 홀수 바이트 경계 포함
    client, models = fake_gen_client(stream=make_tts_stream([tts_chunk(p) for p in pieces]))
    tts = GeminiTTS(client=client)
    out = await collect(tts.synthesize("안녕하세요."))
    assert len(out) >= 2
    assert all(isinstance(c, bytes) and len(c) % 2 == 0 for c in out)
    total = b"".join(out)
    assert abs(len(total) // 2 - int(0.6 * 16000)) <= 16      # 길이 2/3
    # 주파수 보존(440Hz)
    x = np.frombuffer(total, np.int16).astype(np.float32)
    f = np.fft.rfftfreq(len(x), 1 / 16000)[np.argmax(np.abs(np.fft.rfft(x)))]
    assert abs(f - 440) < 5
    call = models.calls[0]
    assert call["model"] == "gemini-3.8-flash-tts"
    vc = call["config"].speech_config.voice_config.prebuilt_voice_config
    assert vc.voice_name == "Kore"
    assert call["config"].response_modalities == ["AUDIO"]


async def test_gemini_tts_16k_passthrough_and_settings():
    from agent.providers.gemini_tts import GeminiTTS

    pcm16 = sine_pcm(16000, 0.2)
    client, models = fake_gen_client(stream=make_tts_stream(
        [tts_chunk(pcm16[:3200], "audio/l16;rate=16000"), tts_chunk(pcm16[3200:], "audio/l16;rate=16000")]))
    tts = GeminiTTS(client=client, model="gemini-3.8-flash-lite-tts", voice="Aoede",
                    style="밝고 친절하게")
    out = await collect(tts.synthesize("네."))
    assert b"".join(out) == pcm16
    call = models.calls[0]
    assert call["model"] == "gemini-3.8-flash-lite-tts"
    assert call["config"].speech_config.voice_config.prebuilt_voice_config.voice_name == "Aoede"
    prompt = call["contents"] if isinstance(call["contents"], str) else str(call["contents"])
    assert "밝고 친절하게" in prompt and "네." in prompt


async def test_gemini_tts_empty_text_yields_nothing():
    from agent.providers.gemini_tts import GeminiTTS

    client, models = fake_gen_client(stream=make_tts_stream([]))
    assert await collect(GeminiTTS(client=client).synthesize("   ")) == []
    assert models.calls == []


async def test_gemini_tts_retries_once():
    from agent.providers.gemini_tts import GeminiTTS

    pcm24 = sine_pcm(24000, 0.1)
    client, models = fake_gen_client(stream=make_tts_stream([tts_chunk(pcm24)], fail_times=1))
    out = await collect(GeminiTTS(client=client).synthesize("네"))
    assert len(b"".join(out)) > 0
    assert len(models.calls) == 2

    client, models = fake_gen_client(stream=make_tts_stream([tts_chunk(pcm24)], fail_times=2))
    with pytest.raises(RuntimeError):
        await collect(GeminiTTS(client=client).synthesize("네"))
    assert len(models.calls) == 2


async def test_gemini_tts_does_not_retry_quota_error():
    from agent.providers.gemini_tts import GeminiTTS

    def stream(model, contents, config, n):
        raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

    client, models = fake_gen_client(stream=stream)
    with pytest.raises(RuntimeError):
        await collect(GeminiTTS(client=client).synthesize("네"))
    assert len(models.calls) == 1


# ══════════════════════════════════════════════════════
# TTS — edge-tts (AD6)
# ══════════════════════════════════════════════════════
class FakePassDecoder:
    """'MP3'를 그대로 PCM으로 내보내는 fake 디코더."""
    instances: list = []

    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()
        self.written = []
        self.closed = False
        FakePassDecoder.instances.append(self)

    async def write(self, data):
        self.written.append(data)
        self.q.put_nowait(data)

    async def close_input(self):
        self.q.put_nowait(None)

    async def output(self):
        while (b := await self.q.get()) is not None:
            yield b

    async def aclose(self):
        self.closed = True


def fake_communicate(chunks_by_call, calls):
    class FakeComm:
        def __init__(self, text, voice, **kw):
            calls.append({"text": text, "voice": voice, **kw})
            self.idx = len(calls) - 1

        async def stream(self):
            chunks = chunks_by_call[min(self.idx, len(chunks_by_call) - 1)]
            if isinstance(chunks, Exception):
                raise chunks
            yield {"type": "WordBoundary", "text": "x"}
            for c in chunks:
                yield {"type": "audio", "data": c}
    return FakeComm


async def _fake_decoder():
    return FakePassDecoder()










# ══════════════════════════════════════════════════════
# LLM — 공용 픽스처
# ══════════════════════════════════════════════════════
TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_info", "description": "주문 조회",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "transfer_to_agent", "description": "상담원 연결",
        "parameters": {"type": "object", "properties": {}}}},
]

HISTORY = [
    Message(role="user", content="주문 2611150231 어디예요?"),
    Message(role="assistant", content="확인해 드릴게요.",
            tool_calls=[ToolCall(id="c1", name="get_order_info", arguments={"order_id": "2611150231"})]),
    Message(role="tool", tool_call_id="c1", name="get_order_info",
            content='{"status": "배송중"}'),
    Message(role="assistant", content="배송 중입니다."),
    Message(role="user", content="고마워요"),
]


# ══════════════════════════════════════════════════════
# LLM — Gemini
# ══════════════════════════════════════════════════════








def gem_chunk(*parts):
    return NS(candidates=[NS(content=NS(parts=list(parts)))])


def gpart(text=None, fc=None, thought=None, sig=None):
    return NS(text=text, function_call=fc, thought=thought, thought_signature=sig)






# ══════════════════════════════════════════════════════
# LLM — OpenAI (Responses API)
# ══════════════════════════════════════════════════════
def test_openai_input_from_messages():
    from agent.providers.openai_llm import to_responses_input

    items = to_responses_input(HISTORY)
    assert items[0] == {"role": "user", "content": "주문 2611150231 어디예요?"}
    assert items[1] == {"role": "assistant", "content": "확인해 드릴게요."}
    assert items[2]["type"] == "function_call"
    assert items[2]["call_id"] == "c1" and items[2]["name"] == "get_order_info"
    assert json.loads(items[2]["arguments"]) == {"order_id": "2611150231"}
    assert items[3] == {"type": "function_call_output", "call_id": "c1", "output": '{"status": "배송중"}'}
    assert items[4] == {"role": "assistant", "content": "배송 중입니다."}
    assert items[5] == {"role": "user", "content": "고마워요"}


def test_openai_tools_flattened():
    from agent.providers.openai_llm import to_responses_tools

    t = to_responses_tools(TOOLS)
    assert t[0] == {"type": "function", "name": "get_order_info", "description": "주문 조회",
                    "parameters": TOOLS[0]["function"]["parameters"]}


class FakeResponses:
    def __init__(self, events):
        self.events = events
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        events = self.events

        async def g():
            for e in events:
                yield e
        return g()


async def test_openai_llm_stream():
    from agent.providers.openai_llm import OpenAILLM

    fc_item = NS(type="function_call", name="get_order_info", call_id="call_9", id="fc_1",
                 arguments='{"order_id":"2611150231"}')
    events = [
        NS(type="response.created"),
        NS(type="response.output_text.delta", delta="확인"),
        NS(type="response.output_text.delta", delta="할게요."),
        NS(type="response.output_item.added", output_index=1,
           item=NS(type="function_call", name="get_order_info", call_id="call_9", id="fc_1", arguments="")),
        NS(type="response.function_call_arguments.delta", output_index=1, delta='{"order_id":'),
        NS(type="response.function_call_arguments.delta", output_index=1, delta='"2611150231"}'),
        NS(type="response.output_item.done", output_index=1, item=fc_item),
        NS(type="response.completed", response=NS()),
    ]
    resp = FakeResponses(events)
    client = NS(responses=resp)
    llm = OpenAILLM(client=client)
    out = await collect(llm.stream("시스템", HISTORY, TOOLS))
    assert [e.type for e in out] == ["text", "text", "tool_call", "done"]
    tc = out[2].tool_call
    assert (tc.id, tc.name, tc.arguments) == ("call_9", "get_order_info", {"order_id": "2611150231"})
    kw = resp.calls[0]
    assert kw["model"] == "gpt-5.4-mini"
    assert kw["instructions"] == "시스템"
    assert kw["reasoning"] == {"effort": "none"}
    assert kw["stream"] is True
    assert kw["tools"][0]["name"] == "get_order_info"
    assert kw["input"][0] == {"role": "user", "content": "주문 2611150231 어디예요?"}


async def test_openai_llm_failed_raises():
    from agent.providers.openai_llm import OpenAILLM

    resp = FakeResponses([NS(type="error", message="boom")])
    with pytest.raises(RuntimeError):
        await collect(OpenAILLM(client=NS(responses=resp)).stream("s", [Message("user", "x")], []))


# ══════════════════════════════════════════════════════
# factory
# ══════════════════════════════════════════════════════




def test_factory_picks_soniox_and_openai_with_keys():
    from agent.providers.factory import ProviderConfig, make_llm, make_stt
    from agent.providers.openai_llm import OpenAILLM
    from agent.providers.soniox_stt import SonioxSTT

    cfg = ProviderConfig.from_env({"GEMINI_API_KEY": "g", "SONIOX_API_KEY": "s", "OPENAI_API_KEY": "o"})
    assert isinstance(make_stt(cfg), SonioxSTT)
    llm = make_llm(cfg, openai_client=NS())
    assert isinstance(llm, OpenAILLM) and llm.model == "gpt-5.4-mini"




def test_factory_soniox_requested_without_key_errors():
    from agent.providers.factory import make_stt

    with pytest.raises(ValueError):
        make_stt({"GEMINI_API_KEY": "g", "stt": "soniox"})




# ══════════════════════════════════════════════════════
# BAZE 게이트웨이 (AG-T16, AD9)
# ══════════════════════════════════════════════════════
BAZE = "https://api.baze.co/v1/gateway"




def test_direct_keys_win_over_baze():
    from agent.providers.factory import ProviderConfig, make_llm, make_stt
    from agent.providers.soniox_stt import SONIOX_URL

    cfg = ProviderConfig.from_env({"BAZE_API_KEY": "bz", "SONIOX_API_KEY": "s", "OPENAI_API_KEY": "o"})
    stt = make_stt(cfg)
    assert stt.api_key == "s" and stt.url == SONIOX_URL
    llm = make_llm(cfg)
    assert llm.api_key == "o" and llm.base_url == ""




def baze_transport(pcm: bytes, seen: list, status: int = 200):
    import httpx

    def handler(req):
        seen.append(req)
        if status != 200:
            return httpx.Response(status, json={"detail": {"code": status, "message": "nope"}})
        return httpx.Response(200, content=pcm, headers={"content-type": "audio/pcm"})
    return httpx.MockTransport(handler)


async def test_baze_tts_request_and_resample_24k_to_16k():
    from agent.providers.baze_tts import BazeTTS

    pcm24 = sine_pcm(24000, 0.6)
    seen: list = []
    tts = BazeTTS(api_key="bz", transport=baze_transport(pcm24, seen), voice="Aoede", style="밝고 친절하게")
    out = await collect(tts.synthesize("안녕하세요."))
    assert out and all(isinstance(c, bytes) and len(c) % 2 == 0 for c in out)
    total = b"".join(out)
    assert abs(len(total) // 2 - int(0.6 * 16000)) <= 16
    x = np.frombuffer(total, np.int16).astype(np.float32)
    f = np.fft.rfftfreq(len(x), 1 / 16000)[np.argmax(np.abs(np.fft.rfft(x)))]
    assert abs(f - 440) < 5
    req = seen[0]
    assert req.method == "POST" and str(req.url) == BAZE + "/audio/speech/"
    assert req.headers["authorization"] == "Bearer bz"
    body = json.loads(req.content)
    assert body == {"model": "gemini-3.8-flash-tts", "input": "안녕하세요.", "voice": "Aoede",
                    "instructions": "밝고 친절하게"}


async def test_baze_tts_empty_text_and_http_error():
    from agent.providers.baze_tts import BazeTTS

    seen: list = []
    tts = BazeTTS(api_key="bz", transport=baze_transport(b"", seen))
    assert await collect(tts.synthesize("  ")) == [] and seen == []
    bad = BazeTTS(api_key="bz", transport=baze_transport(b"", seen, status=403), retries=0)
    with pytest.raises(Exception):
        await collect(bad.synthesize("네."))


def test_factory_baze_tts_selection():
    from agent.providers.baze_tts import BazeTTS
    from agent.providers.factory import ProviderConfig, make_tts

    tts = make_tts(ProviderConfig.from_env({"BAZE_API_KEY": "bz", "DIRECTON_TTS_PROVIDER": "baze",
                                            "GEMINI_TTS_VOICE": "Leda"}))
    assert isinstance(tts, BazeTTS) and tts.model == "gemini-3.8-flash-tts" and tts.voice == "Leda"
    assert tts.url == BAZE + "/audio/speech/" and tts.api_key == "bz"
    with pytest.raises(ValueError):
        make_tts(ProviderConfig.from_env({"DIRECTON_TTS_PROVIDER": "baze"}))


# ══════════════════════════════════════════════════════
# 실제 호출 스모크 (기본 실행 제외: uv run pytest -m live tests/test_providers.py)
# ══════════════════════════════════════════════════════
@pytest.mark.live
async def test_live_gemini_tts_llm_stt_smoke():
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY 없음")
    from agent.providers.factory import ProviderConfig, make_llm, make_stt, make_tts

    cfg = ProviderConfig.from_env()
    pcm = b"".join(await collect(make_tts(cfg).synthesize("배송 언제 와요?")))
    assert len(pcm) > 16000
    stream = await make_stt(cfg).open_stream()
    for i in range(0, len(pcm), 640):
        await stream.send_audio(pcm[i:i + 640])
    text = await stream.finalize(timeout=5.0)
    await stream.close()
    assert "배송" in text
    events = await collect(make_llm(cfg).stream("짧게 답해.", [Message("user", "주문 2611150231 조회해 줘")], TOOLS))
    assert events[-1].type == "done"
    assert any(e.type == "tool_call" and e.tool_call.name == "get_order_info" for e in events)




# ── TTS 끔(개발용 신호음, AG-T31 보강) ─────────────────
async def test_tts_off_makes_local_tone_without_any_key():
    from agent.providers.factory import ProviderConfig, make_tts
    from agent.providers.tone_tts import ToneTTS

    tts = make_tts(ProviderConfig.from_env({}, tts="off"))          # 키가 하나도 없어도 된다
    assert isinstance(tts, ToneTTS)
    chunks = [c async for c in tts.synthesize("확인해 볼게요.")]
    assert len(chunks) == 1 and len(chunks[0]) == 16000 * 2 * 300 // 1000   # 300ms, 16kHz s16le
    assert [c async for c in tts.synthesize("  ")] == []


def test_package_script_never_ships_tts_off():
    from pathlib import Path
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "package.sh").read_text(encoding="utf-8")
    assert "DIRECTON_TTS_PROVIDER=off*" in sh and "TTS 끔 설정이 들어갔다" in sh
