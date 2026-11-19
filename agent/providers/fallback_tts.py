"""TTS 대체(AG-T18). edge-tts는 비공식 서비스라 당일 막힐 수 있다 → 문장 단위로 2차 TTS(Gemini)로 넘긴다.

- 문장마다 1차에 먼저 요청한다. 첫 덩어리가 first_chunk_timeout_s 안에 오지 않거나, 첫 덩어리 전에
  예외가 나거나, 아무 소리 없이 끝나면 1차 스트림을 닫고 같은 문장을 2차로 합성해 이어 보낸다.
- 1차가 실패하면 cooldown_s 동안 1차를 건너뛰고 바로 2차를 쓴다(문장마다 타임아웃을 다시 기다리지 않게).
  쿨다운이 지나면 다시 1차를 시도한다.
- 첫 덩어리를 보낸 뒤의 실패는 2차로 다시 말하지 않는다(같은 말 반복 방지). 그대로 올리고 1차만 쿨다운.
  상위(Responder/Session)가 그 턴을 캐시 사과로 마무리한다.
- 계약은 TTSProvider 그대로: 16k s16le mono 덩어리를 순서대로. 두 Provider 모두 이미 16k로 변환해 준다.
  레벨 정규화는 바깥 LevelTTS가 한다(1차·2차 목소리 레벨이 달라도 같은 목표 RMS로 맞춘다).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable

log = logging.getLogger(__name__)

DEFAULT_FIRST_CHUNK_TIMEOUT_S = 2.0   # edge 첫 바이트 p95 약 1.1~1.3초(baseline_smoke.md) + 여유
DEFAULT_COOLDOWN_S = 30.0


async def _aclose(it: Any) -> None:
    aclose = getattr(it, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except BaseException:  # noqa: BLE001
            pass


class FallbackTTS:
    """TTSProvider 계약. primary 실패 시 secondary로 문장 단위 대체."""

    def __init__(self, primary: Any, secondary: Any,
                 first_chunk_timeout_s: float = DEFAULT_FIRST_CHUNK_TIMEOUT_S,
                 cooldown_s: float = DEFAULT_COOLDOWN_S, clock: Callable[[], float] = time.monotonic):
        self.primary, self.secondary = primary, secondary
        self.first_chunk_timeout_s = float(first_chunk_timeout_s)
        self.cooldown_s = float(cooldown_s)
        self._clock = clock
        self._down_until = 0.0
        self.stats: dict[str, int] = {}       # "served:primary|secondary", "fallback:<사유>"

    @property
    def voice(self) -> str:
        """캐시 네임스페이스는 1차 목소리 기준(server._cache_voice_key)."""
        return getattr(self.primary, "voice", "") or ""

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    def primary_healthy(self) -> bool:
        return self._clock() >= self._down_until

    def _mark_down(self, why: str, err: BaseException | None) -> None:
        self._down_until = self._clock() + self.cooldown_s
        self._count(f"fallback:{why}")
        log.warning("TTS 1차(%s) 실패(%s: %r) → %.0f초 동안 2차(%s)", type(self.primary).__name__, why, err,
                    self.cooldown_s, type(self.secondary).__name__)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        text = (text or "").strip()
        if not text:
            return
        if self.primary_healthy():
            it = self.primary.synthesize(text).__aiter__()
            try:
                try:
                    first = await asyncio.wait_for(it.__anext__(), self.first_chunk_timeout_s)
                except StopAsyncIteration:
                    self._mark_down("empty", None)
                except (Exception, TimeoutError) as e:  # noqa: BLE001 — 첫 덩어리 전 실패는 2차로
                    why = "timeout" if isinstance(e, (TimeoutError, asyncio.TimeoutError)) else "error"
                    self._mark_down(why, e)
                else:
                    self._count("served:primary")
                    yield first
                    try:
                        async for chunk in it:
                            yield chunk
                    except Exception as e:  # noqa: BLE001 — 말하던 중 실패: 반복 없이 올린다
                        self._mark_down("mid", e)
                        raise
                    return
            finally:
                await _aclose(it)
        self._count("served:secondary")
        async for chunk in self.secondary.synthesize(text):
            yield chunk
