"""Gemini STT.

- GeminiLiveSTT: `gemini-3.5-transcribe-live` bidi 스트리밍(`client.aio.live.connect`).
  20ms 프레임을 묶어 chunk_ms(기본 100ms) 단위로 보낸다. 서버 메시지
  `server_content.interim_input_transcription`(현재 세그먼트 가설, 교체)과
  `server_content.input_transcription`(확정 세그먼트, 이어붙임)을 모은다.
  finalize는 `audio_stream_end=True`를 보내 서버의 무음 대기를 건너뛰고 확정을 받는다.
  같은 세션에서 stream_end 뒤에도 오디오를 계속 보낼 수 있다(2026-11-19 실측).
- GeminiBatchSTT: 오디오를 모았다가 finalize에서 generateContent 한 번으로 전사.
  `*-transcribe` 모델은 `audio_transcription_config`, 그 밖의 모델은 받아쓰기 프롬프트.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import wave
from typing import Any

from agent.core.types import SAMPLE_RATE

log = logging.getLogger(__name__)

LIVE_MODEL = "gemini-3.5-transcribe-live"
BATCH_MODEL = "gemini-3.5-transcribe"
MIME = f"audio/pcm;rate={SAMPLE_RATE}"
BATCH_PROMPT = "이 오디오의 한국어 발화를 들리는 그대로 받아써라. 숫자는 아라비아 숫자로 쓴다. 받아쓴 텍스트만 출력한다."


def _terms(context: str) -> list[str]:
    return [t.strip() for t in re.split(r"[,\n]", context or "") if t.strip()][:1000]


def _client(client: Any, api_key: str) -> Any:
    if client is not None:
        return client
    from agent.providers._gemini import make_gemini_client
    return make_gemini_client(api_key)


# ── Live ─────────────────────────────────────────────
class GeminiLiveStream:
    def __init__(self, cm: Any, session: Any, chunk_bytes: int, short_wait: float = 0.3):
        self._cm = cm
        self._session = session
        self._chunk = chunk_bytes
        self._short_wait = short_wait
        self._buf = b""
        self._finals: list[str] = []
        self._interim = ""
        self._seq = 0                    # 받은 확정 세그먼트 수
        self._seq_at_send = 0            # 마지막 오디오 송신 시점의 _seq
        self._dirty = False              # 마지막 finalize 이후 오디오가 있었나
        self._dead = False
        self._closed = False
        self._changed = asyncio.Event()
        self._lock = asyncio.Lock()
        self._rx = asyncio.create_task(self._recv_loop())

    # 수신
    async def _recv_loop(self) -> None:
        try:
            while not self._closed:
                got = False
                async for msg in self._session.receive():
                    got = True
                    self._handle(msg)
                if not got:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:  # 연결 끊김 등
            if not self._closed:
                log.warning("gemini live recv error: %s", e)
        finally:
            self._dead = True
            self._changed.set()

    def _handle(self, msg: Any) -> None:
        sc = getattr(msg, "server_content", None)
        if sc is None:
            return
        fin = getattr(sc, "input_transcription", None)
        if fin is not None:
            t = (getattr(fin, "text", None) or "").strip()
            if t:
                self._finals.append(t)
            self._interim = ""
            self._seq += 1
        interim = getattr(sc, "interim_input_transcription", None)
        if interim is not None:
            self._interim = (getattr(interim, "text", None) or "").strip()
        self._changed.set()

    # 송신
    async def _send(self, **kw: Any) -> None:
        if self._dead:
            return
        try:
            async with self._lock:
                await self._session.send_realtime_input(**kw)
        except Exception as e:
            log.warning("gemini live send error: %s", e)
            self._dead = True
            self._changed.set()

    async def _flush(self) -> None:
        if self._buf:
            from google.genai import types
            data, self._buf = self._buf, b""
            self._seq_at_send = self._seq
            await self._send(audio=types.Blob(data=data, mime_type=MIME))

    async def send_audio(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        self._buf += pcm
        self._dirty = True
        if len(self._buf) >= self._chunk:
            await self._flush()

    def latest_text(self) -> str:
        parts = self._finals + ([self._interim] if self._interim else [])
        return " ".join(parts)

    async def finalize(self, timeout: float = 1.0) -> str:
        await self._flush()
        if not self._dirty or self._dead:
            return self.latest_text()
        self._dirty = False
        seq0 = self._seq
        # 마지막 송신 뒤에 확정이 왔고 가설이 비어 있으면 서버가 이미 확정한 것 → 짧게만 기다린다
        settled = not self._interim and self._seq > self._seq_at_send
        wait = min(timeout, self._short_wait) if settled else timeout
        self._changed.clear()
        await self._send(audio_stream_end=True)
        loop = asyncio.get_running_loop()
        end = loop.time() + wait
        while not (self._seq > seq0 and not self._interim) and not self._dead:
            remaining = end - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except asyncio.TimeoutError:
                break
            self._changed.clear()
        return self.latest_text()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._rx.cancel()
        try:
            await self._rx
        except BaseException:
            pass
        try:
            await self._cm.__aexit__(None, None, None)
        except Exception as e:
            log.debug("gemini live close: %s", e)


class GeminiLiveSTT:
    def __init__(self, client: Any = None, model: str = LIVE_MODEL, language: str = "ko-KR",
                 chunk_ms: int = 100, api_key: str = "", connect_timeout: float = 5.0):
        self.client = client
        self.model = model
        self.language = language
        self.chunk_bytes = SAMPLE_RATE * 2 * chunk_ms // 1000
        self.api_key = api_key
        self.connect_timeout = connect_timeout

    def _config(self, context: str) -> Any:
        from google.genai import types
        return types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=[self.language], custom_vocabulary=_terms(context) or None),
        )

    async def open_stream(self, context: str = "") -> GeminiLiveStream:
        self.client = _client(self.client, self.api_key)
        cm = self.client.aio.live.connect(model=self.model, config=self._config(context))
        session = await asyncio.wait_for(cm.__aenter__(), self.connect_timeout)
        return GeminiLiveStream(cm, session, self.chunk_bytes)


# ── Batch ────────────────────────────────────────────
def _wav(pcm: bytes) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return b.getvalue()


def _response_text(resp: Any) -> str:
    out = []
    for cand in (getattr(resp, "candidates", None) or [])[:1]:
        content = getattr(cand, "content", None)
        for p in (getattr(content, "parts", None) or []):
            at = getattr(p, "audio_transcription", None)
            if at is not None and getattr(at, "text", None):
                out.append(at.text)
            elif getattr(p, "text", None) and not getattr(p, "thought", None):
                out.append(p.text)
    return " ".join(t.strip() for t in out if t.strip())


class GeminiBatchStream:
    def __init__(self, stt: "GeminiBatchSTT", context: str):
        self._stt = stt
        self._context = context
        self._buf = bytearray()
        self._finals: list[str] = []

    async def send_audio(self, pcm: bytes) -> None:
        self._buf += pcm

    def latest_text(self) -> str:
        return " ".join(self._finals)

    async def finalize(self, timeout: float = 1.0) -> str:
        # 일괄 전사는 수 초가 걸린다. 계약의 짧은 timeout 대신 request_timeout을 쓴다.
        if not self._buf:
            return self.latest_text()
        pcm, self._buf = bytes(self._buf), bytearray()
        try:
            text = await asyncio.wait_for(self._stt.transcribe(pcm, self._context),
                                          max(timeout, self._stt.request_timeout))
        except Exception as e:
            log.warning("gemini batch stt failed: %s", e)
            text = ""
        if text:
            self._finals.append(text)
        return self.latest_text()

    async def close(self) -> None:
        self._buf = bytearray()


class GeminiBatchSTT:
    def __init__(self, client: Any = None, model: str = BATCH_MODEL, language: str = "ko-KR",
                 prompt: str = BATCH_PROMPT, thinking_level: str | None = None,
                 request_timeout: float = 20.0, api_key: str = ""):
        self.client = client
        self.model = model
        self.language = language
        self.prompt = prompt
        self.thinking_level = thinking_level
        self.request_timeout = request_timeout
        self.api_key = api_key

    async def open_stream(self, context: str = "") -> GeminiBatchStream:
        self.client = _client(self.client, self.api_key)
        return GeminiBatchStream(self, context)

    async def transcribe(self, pcm: bytes, context: str = "") -> str:
        from google.genai import types
        part = types.Part.from_bytes(data=_wav(pcm), mime_type="audio/wav")
        if "transcribe" in self.model:
            contents: list = [part]
            config = types.GenerateContentConfig(
                audio_transcription_config=types.AudioTranscriptionConfig(
                    language_codes=[self.language], custom_vocabulary=_terms(context) or None))
        else:
            prompt = self.prompt + (f"\n참고 용어: {context}" if context else "")
            contents = [part, prompt]
            kw = {}
            if self.thinking_level:
                kw["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
            config = types.GenerateContentConfig(**kw)
        resp = await self.client.aio.models.generate_content(model=self.model, contents=contents,
                                                             config=config)
        return _response_text(resp)
