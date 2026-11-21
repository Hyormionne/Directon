"""BAZE 게이트웨이 TTS (AG-T16, AD9).

- `POST {BAZE_BASE_URL}/audio/speech/`, `Authorization: Bearer`, JSON {model, input, voice, instructions}.
- 응답은 헤더 없는 raw PCM(24kHz s16le mono)이고 문서상 스트리밍이 없어 한 번에 온다.
  그래도 본문을 받는 대로 읽어 16k로 변환해 yield한다(서버가 나눠 보내면 그만큼 빨라진다).
- 입력 4000자 제한. 상담원 문장은 짧으므로 나누지 않는다.
- 첫 오디오 전에 실패하면 `retries`회 재시도. 4xx(키·권한·입력 오류)는 재시도하지 않는다.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import httpx

from agent.providers.gemini_tts import TTS_MODEL, StreamResampler

log = logging.getLogger(__name__)

BAZE_BASE_URL = "https://api.baze.co/v1/gateway"
BAZE_TTS_RATE = 24000
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT_S = 1.0


class BazeTTS:
    def __init__(self, api_key: str = "", base_url: str = BAZE_BASE_URL, model: str = TTS_MODEL,
                 voice: str = "Kore", style: str = "", retries: int = 1, timeout_s: float = 30.0,
                 transport: Any = None):
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/audio/speech/"
        self.model = model
        self.voice = voice
        self.style = style
        self.retries = retries
        self.timeout_s = timeout_s
        self.transport = transport

    def _body(self, text: str) -> dict:
        body = {"model": self.model, "input": text, "voice": self.voice}
        if self.style:
            body["instructions"] = self.style
        return body

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = (text or "").strip()
        if not text:
            return
        attempt = 0
        while True:
            yielded = False
            try:
                async for chunk in self._once(text):
                    yielded = True
                    yield chunk
                if not yielded:
                    raise RuntimeError("BAZE TTS returned no audio")
                return
            except Exception as e:
                code = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else 0
                limited = code == 429                       # 게이트웨이 분당 요청 한도(120회)
                client_err = 400 <= code < 500 and not limited
                if yielded or client_err or attempt >= (RATE_LIMIT_RETRIES if limited else self.retries):
                    raise
                attempt += 1
                log.warning("baze tts retry %d: %s", attempt, str(e)[:120])
                if limited:
                    await asyncio.sleep(RATE_LIMIT_WAIT_S * attempt)

    async def _once(self, text: str) -> AsyncIterator[bytes]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout_s) as client:
            async with client.stream("POST", self.url, json=self._body(text), headers=headers) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    resp.raise_for_status()
                rs = StreamResampler(BAZE_TTS_RATE)
                async for b in resp.aiter_bytes():
                    out = rs.feed(b)
                    if out:
                        yield out
                tail = rs.feed(b"", last=True)
                if tail:
                    yield tail
