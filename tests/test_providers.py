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


async def test_gemini_live_stt_partial_and_final():
    from agent.providers.gemini_stt import GeminiLiveSTT

    client = FakeLiveClient()
    stt = GeminiLiveSTT(client=client)
    stream = await stt.open_stream()
    frame = b"\x01\x00" * 320  # 20ms
    for _ in range(25):           # 500ms
        await stream.send_audio(frame)
    await wait_for(lambda: stream.latest_text() != "")
    assert stream.latest_text().startswith("주문")
    text = await stream.finalize(timeout=1.0)
    assert text == "주문 번호 알려 주세요"
    assert stream.latest_text() == "주문 번호 알려 주세요"
    await stream.close()

    sess = client.sessions[0]
    assert client.connects[0]["model"] == "gemini-3.5-transcribe-live"
    cfg = client.connects[0]["config"]
    assert cfg.input_audio_transcription.language_codes == ["ko-KR"]
    audio_sends = [k["audio"] for k in sess.sent if k.get("audio") is not None]
    # 20ms 프레임을 그대로 보내지 않고 100ms(3200바이트) 이상으로 묶는다
    assert len(audio_sends) <= 6
    assert all(len(a.data) >= 3200 for a in audio_sends[:-1])
    assert audio_sends[0].mime_type == "audio/pcm;rate=16000"
    assert sum(len(a.data) for a in audio_sends) == 25 * 640
    assert any(k.get("audio_stream_end") for k in sess.sent)
    assert sess.closed


async def test_gemini_live_stt_appends_segments_and_model_configurable():
    from agent.providers.gemini_stt import GeminiLiveStream

    client = FakeLiveClient()
    stt_cls = __import__("agent.providers.gemini_stt", fromlist=["GeminiLiveSTT"]).GeminiLiveSTT
    stt = stt_cls(client=client, model="my-live", language="en-US")
    stream = await stt.open_stream()
    assert isinstance(stream, GeminiLiveStream)
    await wait_for(lambda: client.sessions)
    sess = client.sessions[0]
    # 서버 VAD가 먼저 세그먼트를 확정한 경우: 세그먼트는 이어붙인다
    sess.q.put_nowait(live_msg(final="첫 문장."))
    sess.q.put_nowait(live_msg(interim="둘째"))
    await wait_for(lambda: stream.latest_text() == "첫 문장. 둘째")
    await stream.close()
    assert client.connects[0]["model"] == "my-live"
    assert client.connects[0]["config"].input_audio_transcription.language_codes == ["en-US"]


async def test_gemini_live_stt_finalize_timeout_returns_best_effort():
    from agent.providers.gemini_stt import GeminiLiveSTT

    client = FakeLiveClient(final_on_end=False)
    stream = await GeminiLiveSTT(client=client).open_stream()
    for _ in range(10):
        await stream.send_audio(b"\x01\x00" * 320)
    await wait_for(lambda: stream.latest_text() != "")
    text = await stream.finalize(timeout=0.1)
    assert text == stream.latest_text() != ""
    await stream.close()


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


async def test_gemini_batch_stt_transcribe_model():
    from agent.providers.gemini_stt import GeminiBatchSTT

    def gen(model, contents, config):
        return NS(candidates=[NS(content=NS(parts=[
            NS(text=None, audio_transcription=NS(text="배송 언제 와요?"))]))])

    client, models = fake_gen_client(gen=gen)
    stream = await GeminiBatchSTT(client=client).open_stream()
    pcm = sine_pcm(16000, 0.5)
    for i in range(0, len(pcm), 640):
        await stream.send_audio(pcm[i:i + 640])
    assert stream.latest_text() == ""
    assert await stream.finalize() == "배송 언제 와요?"
    assert stream.latest_text() == "배송 언제 와요?"
    call = models.calls[0]
    assert call["model"] == "gemini-3.5-transcribe"
    part = call["contents"][0]
    assert part.inline_data.mime_type == "audio/wav"
    with wave.open(io.BytesIO(part.inline_data.data)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.readframes(w.getnframes()) == pcm
    assert call["config"].audio_transcription_config.language_codes == ["ko-KR"]
    # 추가 오디오 없이 다시 finalize하면 호출 없이 같은 텍스트
    assert await stream.finalize() == "배송 언제 와요?"
    assert len(models.calls) == 1
    await stream.close()


async def test_gemini_batch_stt_llm_model_uses_prompt_and_text():
    from agent.providers.gemini_stt import GeminiBatchSTT

    def gen(model, contents, config):
        return NS(candidates=[NS(content=NS(parts=[NS(text="네 맞아요", audio_transcription=None)]))])

    client, models = fake_gen_client(gen=gen)
    stream = await GeminiBatchSTT(client=client, model="gemini-3.5-flash-lite").open_stream()
    await stream.send_audio(sine_pcm(16000, 0.2))
    assert await stream.finalize() == "네 맞아요"
    contents = models.calls[0]["contents"]
    assert any(isinstance(c, str) and "받아" in c for c in contents)


async def test_gemini_batch_stt_empty_audio_no_call():
    from agent.providers.gemini_stt import GeminiBatchSTT

    client, models = fake_gen_client(gen=lambda *a: None)
    stream = await GeminiBatchSTT(client=client).open_stream()
    assert await stream.finalize() == ""
    assert models.calls == []


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


async def test_edge_tts_streams_decoded_pcm_in_order():
    from agent.providers.edge_tts import EdgeTTS

    calls = []
    pcm = sine_pcm(16000, 0.3)
    chunks = [pcm[:3200], pcm[3200:6400], pcm[6400:]]
    tts = EdgeTTS(communicate_cls=fake_communicate([chunks], calls), decoder_factory=_fake_decoder)
    out = await collect(tts.synthesize("네, 확인해 드릴게요."))
    assert b"".join(out) == pcm
    assert len(out) >= 2
    assert calls[0]["text"] == "네, 확인해 드릴게요." and calls[0]["voice"] == "ko-KR-SunHiNeural"
    assert FakePassDecoder.instances[-1].closed


async def test_edge_tts_empty_text_and_voice_setting():
    from agent.providers.edge_tts import EdgeTTS

    calls = []
    tts = EdgeTTS(voice="ko-KR-InJoonNeural", communicate_cls=fake_communicate([[b"\x01\x00"]], calls),
                  decoder_factory=_fake_decoder)
    assert await collect(tts.synthesize("  ")) == []
    assert calls == []
    await collect(tts.synthesize("네"))
    assert calls[0]["voice"] == "ko-KR-InJoonNeural"


async def test_edge_tts_retries_once_then_raises():
    from agent.providers.edge_tts import EdgeTTS

    calls = []
    tts = EdgeTTS(communicate_cls=fake_communicate([RuntimeError("No audio"), [b"\x01\x00" * 10]], calls),
                  decoder_factory=_fake_decoder)
    assert b"".join(await collect(tts.synthesize("네"))) == b"\x01\x00" * 10
    assert len(calls) == 2

    calls = []
    tts = EdgeTTS(communicate_cls=fake_communicate([RuntimeError("x"), RuntimeError("y")], calls),
                  decoder_factory=_fake_decoder)
    with pytest.raises(RuntimeError):
        await collect(tts.synthesize("네"))
    assert len(calls) == 2


def test_ffmpeg_decoder_args_target_16k_mono():
    from agent.providers.edge_tts import FfmpegMp3Decoder

    a = FfmpegMp3Decoder.ARGS
    assert a[a.index("-ar") + 1] == "16000" and a[a.index("-ac") + 1] == "1"
    assert a[a.index("-f", a.index("pipe:0")) + 1] == "s16le"


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
def test_gemini_function_declarations_from_openai_tools():
    from agent.providers.gemini_llm import to_function_declarations

    decls = to_function_declarations(TOOLS)
    assert [d.name for d in decls] == ["get_order_info", "transfer_to_agent"]
    assert decls[0].description == "주문 조회"
    assert decls[0].parameters_json_schema["properties"]["order_id"]["type"] == "string"
    assert to_function_declarations([]) == []


def test_gemini_contents_from_messages():
    from agent.providers.gemini_llm import to_gemini_contents

    contents = to_gemini_contents(HISTORY)
    assert [c.role for c in contents] == ["user", "model", "user", "model", "user"]
    assert contents[0].parts[0].text == "주문 2611150231 어디예요?"
    model_parts = contents[1].parts
    assert model_parts[0].text == "확인해 드릴게요."
    fc = model_parts[1].function_call
    assert (fc.id, fc.name, fc.args) == ("c1", "get_order_info", {"order_id": "2611150231"})
    fr = contents[2].parts[0].function_response
    assert (fr.id, fr.name) == ("c1", "get_order_info")
    assert fr.response == {"status": "배송중"}


def test_gemini_contents_merges_parallel_tool_results_and_wraps_non_dict():
    from agent.providers.gemini_llm import to_gemini_contents

    msgs = [
        Message(role="user", content="둘 다"),
        Message(role="assistant", tool_calls=[ToolCall("a", "f1", {}), ToolCall("b", "f2", {"x": 1})]),
        Message(role="tool", tool_call_id="a", name="f1", content="plain text"),
        Message(role="tool", tool_call_id="b", name="f2", content="[1, 2]"),
    ]
    contents = to_gemini_contents(msgs)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert len(contents[1].parts) == 2           # 빈 텍스트 파트 없음
    frs = [p.function_response for p in contents[2].parts]
    assert frs[0].response == {"result": "plain text"}
    assert frs[1].response == {"result": [1, 2]}


def test_gemini_contents_attaches_thought_signature():
    from agent.providers.gemini_llm import to_gemini_contents

    msgs = [Message(role="user", content="q"),
            Message(role="assistant", tool_calls=[ToolCall("c1", "f", {})])]
    contents = to_gemini_contents(msgs, signatures={"c1": b"SIG"})
    assert contents[1].parts[0].thought_signature == b"SIG"
    # 모르는 id는 검증 생략용 더미 서명
    contents = to_gemini_contents(msgs)
    assert contents[1].parts[0].thought_signature


def gem_chunk(*parts):
    return NS(candidates=[NS(content=NS(parts=list(parts)))])


def gpart(text=None, fc=None, thought=None, sig=None):
    return NS(text=text, function_call=fc, thought=thought, thought_signature=sig)


async def test_gemini_llm_stream_text_toolcall_done():
    from agent.providers.gemini_llm import GeminiLLM

    chunks = [
        gem_chunk(gpart(text="생각중", thought=True)),
        gem_chunk(gpart(text="확인해 ")),
        gem_chunk(gpart(text="드릴게요.")),
        gem_chunk(gpart(fc=NS(id=None, name="get_order_info", args={"order_id": "2611150231"}), sig=b"S1")),
        NS(candidates=None),
    ]

    def stream(model, contents, config, n):
        async def g():
            for c in chunks:
                yield c
        return g()

    client, models = fake_gen_client(stream=stream)
    llm = GeminiLLM(client=client, model="gemini-x")
    events = await collect(llm.stream("너는 상담원", HISTORY, TOOLS))
    types_ = [e.type for e in events]
    assert types_ == ["text", "text", "tool_call", "done"]
    assert "".join(e.text for e in events if e.type == "text") == "확인해 드릴게요."
    tc = events[2].tool_call
    assert tc.name == "get_order_info" and tc.arguments == {"order_id": "2611150231"}
    assert isinstance(tc.arguments, dict) and tc.id
    call = models.calls[0]
    assert call["model"] == "gemini-x"
    cfg = call["config"]
    assert cfg.system_instruction == "너는 상담원"
    assert [d.name for d in cfg.tools[0].function_declarations] == ["get_order_info", "transfer_to_agent"]
    assert cfg.thinking_config is not None
    assert cfg.automatic_function_calling.disable is True
    # 서명은 기억해 두었다가 다음 요청에서 그 호출에 붙인다
    nxt = HISTORY + [Message(role="assistant", tool_calls=[tc])]
    await collect(llm.stream("s", nxt, TOOLS))
    last = models.calls[1]["contents"][-1]
    assert last.parts[0].thought_signature == b"S1"


async def test_gemini_llm_no_tools_omits_tools():
    from agent.providers.gemini_llm import GeminiLLM

    def stream(model, contents, config, n):
        async def g():
            yield gem_chunk(gpart(text="네"))
        return g()

    client, models = fake_gen_client(stream=stream)
    events = await collect(GeminiLLM(client=client).stream("s", [Message("user", "안녕")], []))
    assert [e.type for e in events] == ["text", "done"]
    assert not models.calls[0]["config"].tools


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
    assert kw["reasoning"] == {"effort": "medium"}   # AG-T50 기본 medium
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
def test_provider_config_from_env_defaults():
    from agent.providers.factory import ProviderConfig

    cfg = ProviderConfig.from_env({"GEMINI_API_KEY": "g"})
    assert cfg.gemini_api_key == "g" and cfg.soniox_api_key == "" and cfg.openai_api_key == ""
    assert cfg.stt == "auto" and cfg.llm == "auto"
    assert cfg.tts_model == "gemini-3.8-flash-tts" and cfg.tts_voice == "Kore"
    assert cfg.tts == "edge" and cfg.edge_voice == "ko-KR-SunHiNeural"
    assert cfg.stt_live_model == "gemini-3.5-transcribe-live"
    assert cfg.openai_model == "gpt-5.4-mini"
    cfg2 = ProviderConfig.from_env({"GEMINI_API_KEY": "g", "GEMINI_LLM_MODEL": "gemini-3.5-flash",
                                    "STT_PROVIDER": "gemini_batch"})
    assert cfg2.gemini_llm_model == "gemini-3.5-flash" and cfg2.stt == "gemini_batch"


def test_factory_picks_gemini_without_other_keys():
    from agent.providers.factory import ProviderConfig, make_llm, make_stt, make_tts
    from agent.providers.gemini_llm import GeminiLLM
    from agent.providers.gemini_stt import GeminiLiveSTT
    from agent.providers.gemini_tts import GeminiTTS

    fake = NS()
    cfg = ProviderConfig.from_env({"GEMINI_API_KEY": "g"})
    stt = make_stt(cfg, gemini_client=fake)
    primary = getattr(stt, "primary", stt)
    assert isinstance(primary, GeminiLiveSTT)
    llm = make_llm(cfg, gemini_client=fake)
    assert isinstance(llm, GeminiLLM) and llm.model == cfg.gemini_llm_model
    tts = make_tts(cfg, gemini_client=fake)
    from agent.providers.edge_tts import EdgeTTS
    assert isinstance(tts, EdgeTTS) and tts.voice == "ko-KR-SunHiNeural"    # AD6: 기본 edge
    cfg_g = ProviderConfig.from_env({"GEMINI_API_KEY": "g", "DIRECTON_TTS_PROVIDER": "gemini"})
    tts = make_tts(cfg_g, gemini_client=fake)
    assert isinstance(tts, GeminiTTS) and tts.model == "gemini-3.8-flash-tts"
    cfg_e = ProviderConfig.from_env({"DIRECTON_TTS_PROVIDER": "edge", "EDGE_TTS_VOICE": "ko-KR-InJoonNeural"})
    assert make_tts(cfg_e).voice == "ko-KR-InJoonNeural"


def test_factory_picks_soniox_and_openai_with_keys():
    from agent.providers.factory import ProviderConfig, make_llm, make_stt
    from agent.providers.openai_llm import OpenAILLM
    from agent.providers.soniox_stt import SonioxSTT

    cfg = ProviderConfig.from_env({"GEMINI_API_KEY": "g", "SONIOX_API_KEY": "s", "OPENAI_API_KEY": "o"})
    assert isinstance(make_stt(cfg), SonioxSTT)
    llm = make_llm(cfg, openai_client=NS())
    assert isinstance(llm, OpenAILLM) and llm.model == "gpt-5.4-mini"


def test_factory_explicit_choice_overrides_env_and_dict_config():
    from agent.providers.factory import make_llm, make_stt, make_tts
    from agent.providers.gemini_llm import GeminiLLM
    from agent.providers.gemini_stt import GeminiBatchSTT

    fake = NS()
    cfg = {"GEMINI_API_KEY": "g", "OPENAI_API_KEY": "o", "SONIOX_API_KEY": "s",
           "llm": "gemini", "stt": "gemini_batch", "gemini_llm_model": "gemini-3.1-flash-lite",
           "stt_batch_model": "gemini-3.5-flash-lite", "tts": "gemini", "tts_model": "gemini-2.5-flash-preview-tts",
           "tts_voice": "Leda"}
    llm = make_llm(cfg, gemini_client=fake)
    assert isinstance(llm, GeminiLLM) and llm.model == "gemini-3.1-flash-lite"
    stt = make_stt(cfg, gemini_client=fake)
    assert isinstance(stt, GeminiBatchSTT) and stt.model == "gemini-3.5-flash-lite"
    tts = make_tts(cfg, gemini_client=fake)
    assert tts.model == "gemini-2.5-flash-preview-tts" and tts.voice == "Leda"


def test_factory_soniox_requested_without_key_errors():
    from agent.providers.factory import make_stt

    with pytest.raises(ValueError):
        make_stt({"GEMINI_API_KEY": "g", "stt": "soniox"})


async def test_factory_live_falls_back_to_batch_when_connect_fails():
    from agent.providers.factory import ProviderConfig, make_stt
    from agent.providers.gemini_stt import GeminiBatchStream

    client = FakeLiveClient(fail=True)
    client.aio.models = FakeModels(gen=lambda *a: None)
    stt = make_stt(ProviderConfig.from_env({"GEMINI_API_KEY": "g"}), gemini_client=client)
    stream = await stt.open_stream()
    assert isinstance(stream, GeminiBatchStream)


# ══════════════════════════════════════════════════════
# BAZE 게이트웨이 (AG-T16, AD9)
# ══════════════════════════════════════════════════════
BAZE = "https://api.baze.co/v1/gateway"


def test_baze_key_routes_soniox_openai_gemini(monkeypatch):
    from agent.providers import factory
    from agent.providers.factory import ProviderConfig, make_llm, make_stt, make_tts
    from agent.providers.gemini_llm import GeminiLLM
    from agent.providers.openai_llm import OpenAILLM
    from agent.providers.soniox_stt import SonioxSTT

    cfg = ProviderConfig.from_env({"BAZE_API_KEY": "bz"})
    assert cfg.baze_api_key == "bz" and cfg.baze_base_url == BAZE
    stt = make_stt(cfg)
    assert isinstance(stt, SonioxSTT) and stt.api_key == "bz" and stt.model == "stt-rt-v6"
    assert stt.url == "wss://api.baze.co/v1/gateway/soniox/transcribe-websocket"
    llm = make_llm(cfg)
    assert isinstance(llm, OpenAILLM) and llm.model == "gpt-5.4-mini"
    assert llm.api_key == "bz" and llm.base_url == BAZE
    assert str(llm._client().base_url).rstrip("/") == BAZE
    calls = []
    monkeypatch.setattr("agent.providers._gemini.make_gemini_client",
                        lambda api_key="", base_url="": calls.append((api_key, base_url)) or NS())
    g = make_llm(ProviderConfig.from_env({"BAZE_API_KEY": "bz", "LLM_PROVIDER": "gemini"}))
    assert isinstance(g, GeminiLLM) and calls == [("bz", BAZE + "/gemini")]
    assert make_tts(cfg).__class__.__name__ == "EdgeTTS"                 # AD9: 기본 edge 유지
    custom = ProviderConfig.from_env({"BAZE_API_KEY": "bz", "BAZE_BASE_URL": "https://x.test/gw/"})
    assert make_stt(custom).url == "wss://x.test/gw/soniox/transcribe-websocket"
    assert factory.baze_url(custom, "/gemini") == "https://x.test/gw/gemini"


def test_direct_keys_win_over_baze():
    from agent.providers.factory import ProviderConfig, make_llm, make_stt
    from agent.providers.soniox_stt import SONIOX_URL

    cfg = ProviderConfig.from_env({"BAZE_API_KEY": "bz", "SONIOX_API_KEY": "s", "OPENAI_API_KEY": "o"})
    stt = make_stt(cfg)
    assert stt.api_key == "s" and stt.url == SONIOX_URL
    llm = make_llm(cfg)
    assert llm.api_key == "o" and llm.base_url == ""


def test_baze_mode_fallback_chain_drops_missing_models():
    from agent.providers.factory import FallbackLLM, ProviderConfig, make_llm_with_fallback

    cfg = ProviderConfig.from_env({"BAZE_API_KEY": "bz", "LLM_PROVIDER": "gemini"})
    llm = make_llm_with_fallback(cfg, gemini_client=NS())
    assert isinstance(llm, FallbackLLM)
    assert [x.model for x in llm.llms] == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    # OpenAI 기본 + BAZE: Gemini 폴백 없이 같은 모델 사본만(AG-T41, AD17 대회 규정 LLM)
    llm2 = make_llm_with_fallback(ProviderConfig.from_env({"BAZE_API_KEY": "bz"}), gemini_client=NS())
    assert [x.model for x in llm2.llms] == ["gpt-5.4-mini", "gpt-5.4-mini"]
    # 명시한 폴백은 그대로
    cfg3 = ProviderConfig.from_env({"BAZE_API_KEY": "bz", "LLM_PROVIDER": "gemini",
                                    "GEMINI_LLM_FALLBACK_MODELS": "gemini-3.5-flash"})
    assert [x.model for x in make_llm_with_fallback(cfg3, gemini_client=NS()).llms] ==         ["gemini-3.5-flash-lite", "gemini-3.5-flash"]


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


async def test_gemini_live_finalize_waits_full_when_no_result_yet():
    """오디오를 실시간보다 빨리 보내 가설이 아직 없을 때도 확정을 끝까지 기다린다."""
    from agent.providers.gemini_stt import GeminiLiveSTT

    client = FakeLiveClient(final_on_end=False)
    stream = await GeminiLiveSTT(client=client).open_stream()
    sess = client.sessions[0]
    sess.words = []                      # 송신 중 가설 없음
    await stream.send_audio(b"\x01\x00" * 16000)

    async def late_final():
        await asyncio.sleep(0.5)
        sess.q.put_nowait(live_msg(final="늦은 결과"))
    t = asyncio.create_task(late_final())
    assert await stream.finalize(timeout=2.0) == "늦은 결과"
    await t
    await stream.close()
