"""OpenAI Responses API 스트리밍 + 툴 호출 (docs/research/02_TTS_LLM.md B절).

openai 패키지는 지연 import한다(키가 없으면 설치·사용하지 않는다). 테스트는 fake client.
store=False로 매번 전체 입력을 보낸다(reasoning none이라 reasoning item 보존 불필요).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

from agent.core.types import LLMEvent, Message, ToolCall

log = logging.getLogger(__name__)

OPENAI_MODEL = "gpt-5.4-mini"


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


def to_responses_input(messages: list[Message]) -> list[dict]:
    items: list[dict] = []
    for m in messages:
        if m.role == "user":
            items.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if m.content:
                items.append({"role": "assistant", "content": m.content})
            for tc in m.tool_calls:
                items.append({"type": "function_call", "call_id": tc.id, "name": tc.name,
                              "arguments": json.dumps(tc.arguments, ensure_ascii=False)})
        elif m.role == "tool":
            items.append({"type": "function_call_output", "call_id": m.tool_call_id,
                          "output": m.content})
    return items


def _parse_args(s: str) -> dict:
    try:
        v = json.loads(s or "{}")
    except ValueError:
        from agent.providers._privacy import redact
        log.warning("bad tool arguments json: %s", redact(s))
        return {}
    return v if isinstance(v, dict) else {}


class OpenAILLM:
    def __init__(self, client: Any = None, model: str = OPENAI_MODEL, reasoning_effort: str = "none",
                 max_output_tokens: int | None = None, api_key: str = "", base_url: str = ""):
        self.client = client
        self.base_url = base_url   # AG-T16: BAZE 게이트웨이(OpenAI 호환 Responses)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key
        self.cache_key = os.environ.get("OPENAI_PROMPT_CACHE_KEY", "directon")   # 같은 앞부분 요청을 한 캐시로 모은다
        self.last_usage: dict = {}      # 마지막 응답의 토큰 수(입력, 캐시된 입력, 출력)

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
            "input": to_responses_input(messages),
            "reasoning": {"effort": self.reasoning_effort},
            "store": False,
            "stream": True,
        }
        rtools = to_responses_tools(tools)
        if rtools:
            kw["tools"] = rtools
        if self.max_output_tokens:
            kw["max_output_tokens"] = self.max_output_tokens
        if self.cache_key:
            kw["prompt_cache_key"] = self.cache_key
        stream = await self._client().responses.create(**kw)
        pending: dict[Any, dict] = {}
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
                if _g(item, "type") == "function_call":
                    p = pending.pop(_g(ev, "output_index"), {})
                    args = _g(item, "arguments") or p.get("args", "")
                    yield LLMEvent(type="tool_call", tool_call=ToolCall(
                        id=_g(item, "call_id") or p.get("call_id", ""),
                        name=_g(item, "name") or p.get("name", ""),
                        arguments=_parse_args(args)))
            elif t in ("error", "response.failed", "response.incomplete"):
                if t == "response.incomplete":
                    log.warning("openai response incomplete")
                    break
                msg = _g(ev, "message") or _g(_g(_g(ev, "response"), "error"), "message") or t
                raise RuntimeError(f"OpenAI stream error: {msg}")
            elif t == "response.completed":
                self._note_usage(_g(_g(ev, "response"), "usage"))
                break
        yield LLMEvent(type="done")

    def _note_usage(self, usage: Any) -> None:
        """입력 토큰 중 캐시에서 읽은 양을 기록한다(프롬프트 앞부분 캐시가 걸리는지 확인용)."""
        if usage is None:
            return
        try:
            inp = int(_g(usage, "input_tokens", 0) or 0)
            cached = int(_g(_g(usage, "input_tokens_details"), "cached_tokens", 0) or 0)
            out = int(_g(usage, "output_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        self.last_usage = {"input": inp, "cached": cached, "output": out}
        log.info("llm usage: input=%d cached=%d output=%d", inp, cached, out)
