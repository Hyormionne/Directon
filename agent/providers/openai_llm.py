"""OpenAI Responses API 스트리밍 + 툴 호출 (docs/research/02_TTS_LLM.md B절).

openai 패키지는 지연 import한다(키가 없으면 설치·사용하지 않는다). 테스트는 fake client.
store=False로 매번 전체 입력을 보낸다.

AG-T50 reasoning:
- reasoning_effort(none|minimal|low|medium|high|xhigh, 기본 medium)를 `reasoning.effort`로 보낸다.
- effort가 none이 아니고 include_reasoning이면 `include=["reasoning.encrypted_content"]`를 요청하고, 응답의
  reasoning item(암호화 내용이 있는 것만)을 그 응답이 낸 function_call id에 묶어 reasoning_store에 둔다.
  다음 라운드 입력을 만들 때 그 호출이 든 assistant 메시지 앞에 같은 item을 한 번만 되돌린다
  (store=False라 서버에 상태가 없으므로, 툴 라운드 사이 추론을 이어 가려면 OpenAI가 권장하는 방식).
  암호화 내용이 없는 item은 store=False에서 id로 찾을 수 없어(404) 되돌리지 않는다.
  hedge 사본은 factory가 reasoning_store를 공유시킨다(HedgedLLM은 툴 후속을 모델 이름으로 고정해 기본 쪽으로 보낸다).
- 스트림의 reasoning 이벤트(response.reasoning_*·reasoning output item)는 LLMEvent로 내보내지 않는다:
  말하지 않고, HedgedLLM·FallbackLLM의 '첫 이벤트'(hedge 발동·첫 이벤트 제한)도 첫 텍스트·툴 호출·done 기준이다.
  summary는 요청하지 않는다(이벤트·토큰만 늘어난다).
"""
from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import Any, AsyncIterator, Mapping

from agent.core.types import LLMEvent, Message, ToolCall

log = logging.getLogger(__name__)

OPENAI_MODEL = "gpt-5.4-mini"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_REASONING_EFFORT = "medium"      # AG-T50 사용자 요청(2026-11-21): "LLM 엔진 추론 레벨을 미디움으로 올려"
REASONING_INCLUDE = ["reasoning.encrypted_content"]
_STORE_MAX = 2048


def normalize_effort(effort: Any) -> str:
    e = str(effort if effort not in (None, "") else DEFAULT_REASONING_EFFORT).strip().lower()
    if e not in REASONING_EFFORTS:
        raise ValueError(f"알 수 없는 reasoning effort: {effort} (가능: {', '.join(REASONING_EFFORTS)})")
    return e


def _g(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_responses_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools or []:
        fn = t.get("function", t)
        out.append({"type": "function", "name": fn["name"], "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


def to_responses_input(messages: list[Message],
                       reasoning: Mapping[str, list[dict]] | None = None) -> list[dict]:
    """reasoning: function_call id → 그 호출을 낸 응답의 reasoning item들(같은 목록은 한 번만 넣는다)."""
    items: list[dict] = []
    used: set[int] = set()
    for m in messages:
        if m.role == "user":
            items.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if reasoning:
                for tc in m.tool_calls:
                    r = reasoning.get(tc.id)
                    if r and id(r) not in used:
                        used.add(id(r))
                        items.extend(dict(x) for x in r)
            if m.content:
                items.append({"role": "assistant", "content": m.content})
            for tc in m.tool_calls:
                items.append({"type": "function_call", "call_id": tc.id, "name": tc.name,
                              "arguments": json.dumps(tc.arguments, ensure_ascii=False)})
        elif m.role == "tool":
            items.append({"type": "function_call_output", "call_id": m.tool_call_id,
                          "output": m.content})
    return items


def _reasoning_item(item: Any) -> dict | None:
    """되돌릴 수 있는 reasoning item(dict). 암호화 내용이 없으면 None."""
    enc = _g(item, "encrypted_content")
    if not enc:
        return None
    out: dict = {"type": "reasoning"}
    rid = _g(item, "id")
    if rid:
        out["id"] = rid
    out["summary"] = [{"type": "summary_text", "text": _g(p, "text", "")} for p in (_g(item, "summary") or [])]
    out["encrypted_content"] = enc
    return out


def _parse_args(s: str) -> dict:
    try:
        v = json.loads(s or "{}")
    except ValueError:
        from agent.providers._privacy import redact
        log.warning("bad tool arguments json: %s", redact(s))
        return {}
    return v if isinstance(v, dict) else {}


class OpenAILLM:
    def __init__(self, client: Any = None, model: str = OPENAI_MODEL,
                 reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                 max_output_tokens: int | None = None, api_key: str = "", base_url: str = "",
                 include_reasoning: bool = True,
                 reasoning_store: "OrderedDict[str, list[dict]] | None" = None):
        self.client = client
        self.base_url = base_url   # AG-T16: BAZE 게이트웨이(OpenAI 호환 Responses)
        self.model = model
        self.reasoning_effort = normalize_effort(reasoning_effort)
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key
        self.include_reasoning = bool(include_reasoning)
        # function_call id → reasoning item들. hedge 사본과 공유한다(factory).
        self.reasoning_store: OrderedDict[str, list[dict]] = (
            reasoning_store if reasoning_store is not None else OrderedDict())

    @property
    def keep_reasoning(self) -> bool:
        return self.reasoning_effort != "none" and self.include_reasoning

    def _store(self, call_id: str, items: list[dict]) -> None:
        if not call_id or not items:
            return
        self.reasoning_store[call_id] = items
        self.reasoning_store.move_to_end(call_id)
        while len(self.reasoning_store) > _STORE_MAX:
            self.reasoning_store.popitem(last=False)

    def _client(self) -> Any:
        if self.client is None:
            from openai import AsyncOpenAI  # 지연 import
            self.client = AsyncOpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                                      base_url=self.base_url or None)
        return self.client

    async def stream(self, system: str, messages: list[Message],
                     tools: list[dict]) -> AsyncIterator[LLMEvent]:
        kw: dict = {
            "model": self.model,
            "instructions": system,
            "input": to_responses_input(messages, self.reasoning_store if self.keep_reasoning else None),
            "reasoning": {"effort": self.reasoning_effort},
            "store": False,
            "stream": True,
        }
        if self.keep_reasoning:
            kw["include"] = list(REASONING_INCLUDE)
        rtools = to_responses_tools(tools)
        if rtools:
            kw["tools"] = rtools
        if self.max_output_tokens:
            kw["max_output_tokens"] = self.max_output_tokens
        stream = await self._client().responses.create(**kw)
        pending: dict[Any, dict] = {}
        reasoning: list[dict] = []      # 이 응답의 reasoning item(같은 목록 객체를 이 응답의 호출 id들에 묶는다)
        async for ev in stream:
            t = _g(ev, "type", "")
            if t == "response.output_text.delta":
                d = _g(ev, "delta", "")
                if d:
                    yield LLMEvent(type="text", text=d)
            elif t == "response.output_item.added":
                item = _g(ev, "item")
                if _g(item, "type") == "function_call":
                    pending[_g(ev, "output_index")] = {"name": _g(item, "name"),
                                                       "call_id": _g(item, "call_id"), "args": ""}
            elif t == "response.function_call_arguments.delta":
                p = pending.get(_g(ev, "output_index"))
                if p is not None:
                    p["args"] += _g(ev, "delta", "")
            elif t == "response.output_item.done":
                item = _g(ev, "item")
                if _g(item, "type") == "reasoning":
                    r = _reasoning_item(item) if self.keep_reasoning else None
                    if r is not None:
                        reasoning.append(r)
                elif _g(item, "type") == "function_call":
                    p = pending.pop(_g(ev, "output_index"), {})
                    args = _g(item, "arguments") or p.get("args", "")
                    call_id = _g(item, "call_id") or p.get("call_id", "")
                    self._store(call_id, reasoning)
                    yield LLMEvent(type="tool_call", tool_call=ToolCall(
                        id=call_id,
                        name=_g(item, "name") or p.get("name", ""),
                        arguments=_parse_args(args)))
            elif t in ("error", "response.failed", "response.incomplete"):
                if t == "response.incomplete":
                    log.warning("openai response incomplete")
                    break
                msg = _g(ev, "message") or _g(_g(_g(ev, "response"), "error"), "message") or t
                raise RuntimeError(f"OpenAI stream error: {msg}")
            elif t == "response.completed":
                break
        yield LLMEvent(type="done")
