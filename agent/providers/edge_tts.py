"""edge-tts 기반 상담원 TTS (AD6: Gemini TTS 무료 티어가 모델당 하루 10요청이라 개발 중 대체).

- edge-tts `Communicate.stream()`으로 MP3 덩어리(24kHz 48kbps)를 받는다.
  출력 형식은 라이브러리에 고정되어 있고, 서비스도 raw PCM·16k MP3·webm 요청을 거부한다
  (2026-11-19 실측: raw-16khz-16bit-mono-pcm 등 → "No audio was received").
- MP3는 ffmpeg 프로세스 파이프로 스트리밍 디코드해 16k s16le mono로 바꾼다.
  ffmpeg 기동은 edge 연결과 동시에 시작해 지연을 숨긴다. 디코드된 PCM은 오는 대로 yield.
- 첫 오디오 전에 실패하면 1회 재시도.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

from agent.core.types import SAMPLE_RATE

log = logging.getLogger(__name__)

EDGE_VOICE = "ko-KR-SunHiNeural"


class Decoder(Protocol):
    async def write(self, data: bytes) -> None: ...
    async def close_input(self) -> None: ...
    def output(self) -> AsyncIterator[bytes]: ...
    async def aclose(self) -> None: ...


class FfmpegMp3Decoder:
    """MP3 바이트 → ffmpeg(stdin/stdout 파이프) → 16k s16le."""

    ARGS = ["-hide_banner", "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0", "-f", "mp3", "-i", "pipe:0",
            "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]

    def __init__(self, proc: Any):
        self.proc = proc
        self._carry = b""

    @classmethod
    async def start(cls, ffmpeg: str = "ffmpeg") -> "FfmpegMp3Decoder":
        proc = await asyncio.create_subprocess_exec(
            ffmpeg, *cls.ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        return cls(proc)

    async def write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def close_input(self) -> None:
        if not self.proc.stdin.is_closing():
            self.proc.stdin.close()

    async def output(self) -> AsyncIterator[bytes]:
        while True:
            b = await self.proc.stdout.read(4096)
            if not b:
                break
            buf = self._carry + b
            n = len(buf) // 2 * 2
            self._carry = buf[n:]
            if n:
                yield buf[:n]

    async def aclose(self) -> None:
        if self.proc.returncode is None:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
            await self.proc.wait()


DecoderFactory = Callable[[], Awaitable[Decoder]]


class EdgeTTS:
    def __init__(self, voice: str = EDGE_VOICE, rate: str = "+0%", pitch: str = "+0Hz",
                 communicate_cls: Any = None, decoder_factory: DecoderFactory | None = None,
                 ffmpeg: str = "ffmpeg", retries: int = 1):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self._comm = communicate_cls
        self._decoder_factory = decoder_factory or (lambda: FfmpegMp3Decoder.start(ffmpeg))
        self.retries = retries

    def _communicate(self, text: str) -> Any:
        cls = self._comm
        if cls is None:
            import edge_tts
            cls = edge_tts.Communicate
        return cls(text, self.voice, rate=self.rate, pitch=self.pitch)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = (text or "").strip()
        if not text:
            return
        attempt = 0
        while True:
            yielded = False
            try:
                async for pcm in self._once(text):
                    yielded = True
                    yield pcm
                if not yielded:
                    raise RuntimeError("edge-tts returned no audio")
                return
            except Exception as e:
                if yielded or attempt >= self.retries:
                    raise
                attempt += 1
                log.warning("edge tts retry %d: %s", attempt, e)

    async def _once(self, text: str) -> AsyncIterator[bytes]:
        dec_task = asyncio.ensure_future(self._decoder_factory())   # edge 연결과 병렬로 기동
        comm = self._communicate(text)
        dec: Decoder | None = None

        async def pump() -> None:
            nonlocal dec
            try:
                async for ch in comm.stream():
                    if ch.get("type") == "audio" and ch.get("data"):
                        if dec is None:
                            dec = await dec_task
                        await dec.write(ch["data"])
            finally:
                if dec is None:
                    dec = await dec_task
                await dec.close_input()

        pump_task = asyncio.create_task(pump())
        try:
            decoder = await dec_task
            async for pcm in decoder.output():
                yield pcm
            await pump_task          # edge 쪽 오류를 올린다
        finally:
            if not pump_task.done():
                pump_task.cancel()
                try:
                    await pump_task
                except BaseException:
                    pass
            if dec_task.done() and not dec_task.cancelled() and dec_task.exception() is None:
                await dec_task.result().aclose()
            else:
                dec_task.cancel()
