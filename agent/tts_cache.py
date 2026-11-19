"""고정 문구 → PCM 캐시(assets/tts_cache/<hash>.pcm). 첫 마디·필러·사과 문구용."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from agent.config import Settings
from agent.core.types import FRAME_BYTES, SessionInfo, TTSProvider

log = logging.getLogger(__name__)

TRIM_RMS = 200.0     # 앞쪽 무음 자르기 기준(20ms 프레임 RMS)


def _rms(b: bytes) -> float:
    a = np.frombuffer(b[: len(b) - len(b) % 2], dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def trim_leading_silence(pcm: bytes, threshold: float = TRIM_RMS) -> bytes:
    """첫 프레임부터 유성음이 되게 앞 무음 프레임을 자른다. 전부 무음이면 원본."""
    for off in range(0, len(pcm), FRAME_BYTES):
        if _rms(pcm[off:off + FRAME_BYTES]) >= threshold:
            return pcm[off:]
    return pcm


async def synthesize_all(tts: TTSProvider, text: str) -> bytes:
    chunks = []
    async for c in tts.synthesize(text):
        chunks.append(c)
    return b"".join(chunks)


class TTSCache:
    def __init__(self, cache_dir: str | Path = "assets/tts_cache", voice: str = "default"):
        self.dir = Path(cache_dir)
        self.voice = voice
        self._mem: dict[str, bytes] = {}

    def key(self, text: str) -> str:
        return hashlib.sha1(f"{self.voice}\n{text}".encode("utf-8")).hexdigest()[:16]

    def path(self, text: str) -> Path:
        return self.dir / f"{self.key(text)}.pcm"

    def get(self, text: str) -> bytes | None:
        if text in self._mem:
            return self._mem[text]
        p = self.path(text)
        if p.exists():
            data = p.read_bytes()
            self._mem[text] = data
            return data
        return None

    def put(self, text: str, pcm: bytes) -> bytes:
        pcm = trim_leading_silence(pcm)
        self._mem[text] = pcm
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path(text).write_bytes(pcm)
        return pcm

    async def get_or_synth(self, text: str, tts: TTSProvider) -> bytes:
        cached = self.get(text)
        if cached is not None:
            return cached
        return self.put(text, await synthesize_all(tts, text))

    async def warm(self, phrases: list[str] | tuple[str, ...], tts: TTSProvider) -> int:
        """없는 문구만 합성해 저장한다. 합성한 개수를 반환(실패는 경고만)."""
        n = 0
        for text in phrases:
            if self.get(text) is not None:
                continue
            try:
                self.put(text, await synthesize_all(tts, text))
                n += 1
            except Exception as e:  # noqa: BLE001
                log.warning("tts cache warm failed for %r: %r", text, e)
        return n


def select_opener(info: SessionInfo, settings: Settings) -> str:
    """첫 마디 선택. 기본은 이름 없는 캐시 문구."""
    name = ((info.caller or {}).get("name") or "").strip()
    if settings.opener_use_name and name:
        return f"네, {name} 고객님."
    if info.history and len(settings.openers) > 1:
        return settings.openers[1]
    return settings.openers[0]
