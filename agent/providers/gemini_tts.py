"""Gemini TTS (스트리밍).

2026-11-19 실측 결과:
- `models.generate_content_stream`(response_modalities=["AUDIO"])이 Interactions API 스트림보다
  첫 덩어리가 훨씬 빠르다(약 1.0~1.1s vs 3.6s). 그래서 generateContent 스트림을 쓴다.
- 16kHz 직접 출력은 지원되지 않는다(Interactions `response_format.sample_rate`=16000 → 400,
  generateContent `response_mime_type` audio/* → 400). 서버는 `audio/l16; rate=24000`을 준다.
  → soxr 스트림 리샘플러로 16k 변환(청크 경계 홀수 바이트는 이월).
- 실패하면 1회 재시도(첫 오디오를 내보낸 뒤의 실패는 재시도하지 않고 올린다).
"""
from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator

import numpy as np
import soxr

from agent.core.types import SAMPLE_RATE

log = logging.getLogger(__name__)

TTS_MODEL = "gemini-3.8-flash-lite-tts"


def _is_quota(e: Exception) -> bool:
    """429 할당량 초과는 즉시 재시도해도 소용없다(무료 티어 TTS는 모델당 하루 10회, 2026-11-19 실측)."""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code == 429 or "RESOURCE_EXHAUSTED" in str(e)


def _rate(mime: str | None) -> int:
    m = re.search(r"rate=(\d+)", mime or "")
    return int(m.group(1)) if m else 24000


class StreamResampler:
    """s16le 바이트 스트림 → 16k s16le. 같은 레이트면 통과."""

    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE):
        self.in_rate, self.out_rate = in_rate, out_rate
        self._rs = None if in_rate == out_rate else soxr.ResampleStream(
            in_rate, out_rate, 1, dtype="int16", quality="HQ")
        self._carry = b""

    def feed(self, data: bytes, last: bool = False) -> bytes:
        buf = self._carry + data
        n = len(buf) // 2 * 2
        self._carry = buf[n:]
        if self._rs is None:
            return buf[:n]
        x = np.frombuffer(buf[:n], dtype=np.int16)
        return self._rs.resample_chunk(x, last=last).tobytes()


class GeminiTTS:
    def __init__(self, client: Any = None, model: str = TTS_MODEL, voice: str = "Kore",
                 style: str = "", retries: int = 1, api_key: str = ""):
        self.client = client
        self.model = model
        self.voice = voice
        self.style = style
        self.retries = retries
        self.api_key = api_key

    def _config(self) -> Any:
        from google.genai import types
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice))),
        )

    def _prompt(self, text: str) -> str:
        return f"{self.style}: {text}" if self.style else text

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = (text or "").strip()
        if not text:
            return
        if self.client is None:
            from agent.providers._gemini import make_gemini_client
            self.client = make_gemini_client(self.api_key)
        attempt = 0
        while True:
            yielded = False
            try:
                async for chunk in self._once(text):
                    yielded = True
                    yield chunk
                if not yielded:
                    raise RuntimeError("Gemini TTS returned no audio")
                return
            except Exception as e:
                if yielded or attempt >= self.retries or _is_quota(e):
                    raise
                attempt += 1
                log.warning("gemini tts retry %d: %s", attempt, e)

    async def _once(self, text: str) -> AsyncIterator[bytes]:
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model, contents=self._prompt(text), config=self._config())
        rs: StreamResampler | None = None
        async for chunk in stream:
            for cand in (getattr(chunk, "candidates", None) or [])[:1]:
                content = getattr(cand, "content", None)
                for p in (getattr(content, "parts", None) or []):
                    inline = getattr(p, "inline_data", None)
                    if inline is None or not getattr(inline, "data", None):
                        continue
                    if rs is None:
                        rs = StreamResampler(_rate(getattr(inline, "mime_type", None)))
                    out = rs.feed(inline.data)
                    if out:
                        yield out
        if rs is not None:
            tail = rs.feed(b"", last=True)
            if tail:
                yield tail
