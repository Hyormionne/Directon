"""LLM 폴백 체인(AG-T09). Gemini Flash 계열은 503·429·시간 초과가 잦다(docs/decisions/bench_providers.md).

- 체인의 각 모델에 차례로 요청한다. 첫 이벤트가 first_event_timeout_s 안에 오지 않거나, 첫 이벤트 전에
  오류(503/429/시간 초과/연결 오류 등)가 나면 그 스트림을 닫고 다음 모델로 즉시 재시도한다.
- 첫 이벤트를 하나라도 내보낸 뒤에는 폴백하지 않는다(텍스트 중복 발화·툴 중복 호출 방지). 그 뒤 오류는 그대로 올린다.
- 마지막 모델은 첫 이벤트 제한 없이 기다린다(상위 Responder의 llm_timeout_s가 상한).
- 400(잘못된 요청)도 폴백한다: 다른 모델이 받아 줄 수 있고, 같은 모델 재시도는 의미가 없다.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from agent.core.types import LLMEvent, Message

log = logging.getLogger(__name__)

DEFAULT_FIRST_EVENT_TIMEOUT_S = 2.5


def _name(llm: Any) -> str:
    return str(getattr(llm, "model", None) or type(llm).__name__)


async def _aclose(it: Any) -> None:
    aclose = getattr(it, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except BaseException:  # noqa: BLE001
            pass


class FallbackLLM:
    """LLMProvider 계약. llms[0]이 기본, 나머지가 대체 순서."""

    def __init__(self, llms: list[Any], first_event_timeout_s: float = DEFAULT_FIRST_EVENT_TIMEOUT_S):
        if not llms:
            raise ValueError("FallbackLLM에는 모델이 하나 이상 필요하다")
        self.llms = list(llms)
        self.first_event_timeout_s = float(first_event_timeout_s)
        self.last_model: str = ""
        self.stats: dict[str, int] = {}          # "served:<model>", "fallback:<model>:<사유>"

    @property
    def model(self) -> str:
        return _name(self.llms[0])

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    async def stream(self, system: str, messages: list[Message],
                     tools: list[dict]) -> AsyncIterator[LLMEvent]:
        last_err: BaseException | None = None
        n = len(self.llms)
        for i, llm in enumerate(self.llms):
            is_last = i == n - 1
            it = llm.stream(system, list(messages), tools).__aiter__()
            try:
                try:
                    if is_last:
                        first = await it.__anext__()
                    else:
                        first = await asyncio.wait_for(it.__anext__(), self.first_event_timeout_s)
                except StopAsyncIteration:
                    self.last_model = _name(llm)
                    self._count(f"served:{self.last_model}")
                    return
                except (Exception, TimeoutError) as e:  # noqa: BLE001 — 첫 이벤트 전 실패는 다음 모델로
                    last_err = e
                    if is_last:
                        raise
                    why = "timeout" if isinstance(e, (TimeoutError, asyncio.TimeoutError)) else "error"
                    self._count(f"fallback:{_name(llm)}:{why}")
                    log.warning("LLM %s 첫 이벤트 실패(%s: %r) → %s", _name(llm), why, e,
                                _name(self.llms[i + 1]))
                    continue
                self.last_model = _name(llm)
                self._count(f"served:{self.last_model}")
                if i:
                    log.info("LLM fallback served by %s", self.last_model)
                yield first
                async for ev in it:                  # 이 뒤로는 폴백하지 않는다
                    yield ev
                return
            finally:
                await _aclose(it)
        if last_err is not None:  # pragma: no cover — 위에서 마지막 모델 오류는 이미 올렸다
            raise last_err
