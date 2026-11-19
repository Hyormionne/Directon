"""LLM 동시 요청(hedging, AG-T15c). docs/decisions/llm_latency_r2.md.

- mode="hedge": llms[0](기본)에 요청하고 hedge_delay_s 안에 첫 이벤트가 없으면 llms[1](대체)에도 동시에
  요청한다. **먼저 첫 이벤트(텍스트·툴 호출·done 모두)를 낸 쪽**을 채택하고 다른 쪽은 취소한다.
  기본 모델이 h 전에 첫 이벤트 없이 오류를 내면 기다리지 않고 바로 대체 모델을 시작한다.
- mode="race": 처음부터 둘 다 보낸다(입력 비용 2배).
- 첫 이벤트를 낸 스트림만 끝까지 소비한다(이후 전환 없음: 텍스트 중복 발화·툴 중복 호출 방지).
- 둘 다 첫 이벤트 전에 실패하면 llms[2:]를 FallbackLLM처럼 차례로 시도하고, 그래도 안 되면 마지막 예외를 올린다.
- 툴 후속 라운드 고정: 채택된 스트림이 낸 tool_call id → 모델을 기억한다. 마지막 메시지가 툴 결과이고 그
  호출을 낸 모델을 알면 그 모델에만 요청한다(Gemini thought_signature는 GeminiLLM 인스턴스가 모델별로
  기억하므로 다른 모델로 넘기면 서명이 없다). 고정 모델이 첫 이벤트 전에 실패하거나 first_event_timeout_s(기본 2.5초)
  안에 첫 이벤트가 없으면 나머지로 폴백한다(GeminiLLM은 모르는 호출 id에 공식 우회 서명을 붙인다).
- FallbackLLM을 상속한다: 서버 예열(`llms` 속성 순회·404 제외)과 기존 isinstance 검사를 그대로 쓴다.
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, AsyncIterator

from agent.core.types import LLMEvent, Message
from agent.providers.fallback_llm import DEFAULT_FIRST_EVENT_TIMEOUT_S, FallbackLLM, _aclose, _name

log = logging.getLogger(__name__)

DEFAULT_HEDGE_DELAY_S = 0.5             # AG-T15c 실측: 3.5-flash-lite 첫 이벤트 p50 ~1.1s라 0.8s도 거의 항상 발동 → 꼬리만 길어짐
MODES = ("hedge", "race")
_PIN_MAX = 2048


class _Lane:
    """한 모델의 스트림과 '다음 이벤트' 태스크."""

    def __init__(self, idx: int, llm: Any, system: str, messages: list[Message], tools: list[dict]):
        self.idx = idx
        self.llm = llm
        self.it = llm.stream(system, list(messages), tools).__aiter__()
        self.task: asyncio.Task = asyncio.ensure_future(self.it.__anext__())

    async def close(self) -> None:
        if not self.task.done():
            self.task.cancel()
        try:
            await self.task
        except BaseException:  # noqa: BLE001 — 취소·오류 모두 무시(이미 진 쪽)
            pass
        await _aclose(self.it)


class HedgedLLM(FallbackLLM):
    """LLMProvider 계약. llms[0] 기본, llms[1] 동시 요청 대상, llms[2:] 순차 폴백."""

    def __init__(self, llms: list[Any], hedge_delay_s: float = DEFAULT_HEDGE_DELAY_S, mode: str = "hedge",
                 first_event_timeout_s: float = DEFAULT_FIRST_EVENT_TIMEOUT_S):
        if not llms:
            raise ValueError("HedgedLLM에는 모델이 하나 이상 필요하다")
        if mode not in MODES:
            raise ValueError(f"알 수 없는 hedge mode: {mode}")
        # first_event_timeout_s: 순차 경로(툴 후속 고정 모델, llms[2:])에서 마지막이 아닌 모델의 첫 이벤트 제한
        super().__init__(llms, first_event_timeout_s=first_event_timeout_s)
        self.hedge_delay_s = float(hedge_delay_s)
        self.mode = mode
        self._pins: OrderedDict[str, str] = OrderedDict()     # tool_call id → 모델 이름

    # ── 툴 후속 라운드 고정 ──
    def _remember(self, ev: LLMEvent, llm: Any) -> None:
        if ev.type == "tool_call" and ev.tool_call is not None and ev.tool_call.id:
            self._pins[ev.tool_call.id] = _name(llm)
            self._pins.move_to_end(ev.tool_call.id)
            while len(self._pins) > _PIN_MAX:
                self._pins.popitem(last=False)

    def _pinned(self, messages: list[Message]) -> int | None:
        if not messages or messages[-1].role != "tool":
            return None
        for m in reversed(messages):
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    name = self._pins.get(tc.id)
                    if name is not None:
                        for i, llm in enumerate(self.llms):
                            if _name(llm) == name:
                                return i
                return None
        return None

    # ── 스트림 ──
    async def stream(self, system: str, messages: list[Message],
                     tools: list[dict]) -> AsyncIterator[LLMEvent]:
        llms = list(self.llms)
        pin = self._pinned(messages)
        if pin is not None:
            self._count(f"pinned:{_name(llms[pin])}")
            order = [pin] + [i for i in range(len(llms)) if i != pin]
            async for ev in self._sequential(order, llms, system, messages, tools):
                yield ev
            return

        lanes: list[_Lane] = []
        winner: _Lane | None = None
        last_err: BaseException | None = None
        try:
            lanes.append(_Lane(0, llms[0], system, messages, tools))
            if len(llms) > 1:
                if self.mode == "race":
                    lanes.append(_Lane(1, llms[1], system, messages, tools))
                else:
                    done, _ = await asyncio.wait({lanes[0].task}, timeout=self.hedge_delay_s)
                    err = lanes[0].task.exception() if done else None
                    if not done or (err is not None and not isinstance(err, StopAsyncIteration)):
                        self._count(f"hedge:{_name(llms[1])}")
                        lanes.append(_Lane(1, llms[1], system, messages, tools))
            winner, first, last_err = await self._first_of(lanes)
            if winner is not None:
                for lane in lanes:
                    if lane is not winner:
                        await lane.close()
                self.last_model = _name(winner.llm)
                self._count(f"served:{self.last_model}")
                if first is None:                         # 빈 스트림
                    return
                self._remember(first, winner.llm)
                yield first
                async for ev in winner.it:                # 이 뒤로는 전환하지 않는다
                    self._remember(ev, winner.llm)
                    yield ev
                return
        finally:
            for lane in lanes:
                await lane.close()

        rest = list(range(len(lanes), len(llms)))
        if not rest:
            assert last_err is not None
            raise last_err
        async for ev in self._sequential(rest, llms, system, messages, tools):
            yield ev

    async def _first_of(self, lanes: list[_Lane]) -> tuple[_Lane | None, LLMEvent | None, BaseException | None]:
        """첫 이벤트(또는 빈 스트림 종료)를 먼저 낸 lane. 모두 실패하면 (None, None, 마지막 예외)."""
        pending = {lane.task: lane for lane in lanes}
        last_err: BaseException | None = None
        while pending:
            done, _ = await asyncio.wait(set(pending), return_when=asyncio.FIRST_COMPLETED)
            # 같은 순간에 끝났으면 기본 모델(앞 lane)을 우선한다
            for task in sorted(done, key=lambda t: pending[t].idx):
                lane = pending.pop(task)
                err = task.exception()
                if err is None:
                    return lane, task.result(), None
                if isinstance(err, StopAsyncIteration):
                    return lane, None, None
                last_err = err
                self._count(f"error:{_name(lane.llm)}")
                log.warning("LLM %s 첫 이벤트 전 실패: %r", _name(lane.llm), err)
        return None, None, last_err

    async def _sequential(self, order: list[int], llms: list[Any], system: str, messages: list[Message],
                          tools: list[dict]) -> AsyncIterator[LLMEvent]:
        last_err: BaseException | None = None
        for k, i in enumerate(order):
            llm = llms[i]
            it = llm.stream(system, list(messages), tools).__aiter__()
            try:
                try:
                    if k == len(order) - 1 or self.first_event_timeout_s <= 0:
                        first = await it.__anext__()
                    else:
                        first = await asyncio.wait_for(it.__anext__(), self.first_event_timeout_s)
                except StopAsyncIteration:
                    self.last_model = _name(llm)
                    self._count(f"served:{self.last_model}")
                    return
                except (Exception, TimeoutError) as e:  # noqa: BLE001 — 첫 이벤트 전 실패·시간 초과는 다음 모델로
                    last_err = e
                    self._count(f"error:{_name(llm)}")
                    if k == len(order) - 1:
                        raise
                    log.warning("LLM %s 첫 이벤트 실패(%r) → %s", _name(llm), e, _name(llms[order[k + 1]]))
                    continue
                self.last_model = _name(llm)
                self._count(f"served:{self.last_model}")
                self._remember(first, llm)
                yield first
                async for ev in it:
                    self._remember(ev, llm)
                    yield ev
                return
            finally:
                await _aclose(it)
        if last_err is not None:  # pragma: no cover
            raise last_err
