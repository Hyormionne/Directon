"""BAZE 지연 시뮬레이션: 직접 API로 측정하되 BAZE 경유의 추가 지연만큼 첫 출력을 늦춘다.

- 개발 측정은 공식 직접 API(빠름)로 하고, 제출본(BAZE 경유)의 속도를 보수적으로 보기 위함.
- `DIRECTON_SIM_LLM_DELAY_S`: LLM 스트림 첫 이벤트 전 대기(초). 직접 vs BAZE 첫 토큰 p50 차이.
- `DIRECTON_SIM_TTS_DELAY_S`: TTS 첫 오디오 전 대기(초).
- 0이거나 없으면 감싸지 않는다.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator


def _delay(name: str, env: Any = None) -> float:
    env = os.environ if env is None else env
    try:
        return max(0.0, float(env.get(name) or 0))
    except ValueError:
        return 0.0


class _Delegate:
    def __init__(self, inner: Any, delay_s: float):
        self.inner = inner
        self.delay_s = delay_s

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class DelayedLLM(_Delegate):
    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        first = True
        async for ev in self.inner.stream(*args, **kwargs):
            if first:
                await asyncio.sleep(self.delay_s)
                first = False
            yield ev


class DelayedTTS(_Delegate):
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        first = True
        async for chunk in self.inner.synthesize(text):
            if first:
                await asyncio.sleep(self.delay_s)
                first = False
            yield chunk


def wrap_llm(llm: Any, env: Any = None) -> Any:
    d = _delay("DIRECTON_SIM_LLM_DELAY_S", env)
    return DelayedLLM(llm, d) if d > 0 else llm


def wrap_tts(tts: Any, env: Any = None) -> Any:
    d = _delay("DIRECTON_SIM_TTS_DELAY_S", env)
    return DelayedTTS(tts, d) if d > 0 else tts
