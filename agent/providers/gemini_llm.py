"""Gemini 텍스트 LLM: 스트리밍 + function calling.

- 시스템 프롬프트는 system_instruction, 자동 함수 호출은 끈다(실행은 상위 계층).
- thinking은 모델이 허용하는 최저로: 후보 [level MINIMAL → budget 0 → level LOW → 미지정]을
  차례로 시도하고, 첫 출력 전에 실패하면 다음 후보로 넘어간다. 성공한 후보는 모델별로 기억한다.
  (2026-11-19 실측: gemini-3.8-flash는 MINIMAL 400·budget 0 허용, gemini-3.5-flash-lite는
  MINIMAL 허용·budget 0 400, gemini-3.1-flash-lite/3.5-flash는 둘 다 허용)
- Gemini 3 계열은 이전 턴의 function_call 파트에 thought_signature를 되돌려야 한다.
  ToolCall에는 그 자리가 없으므로 호출 id → 서명을 이 객체가 기억해 두었다가 변환할 때 붙인다.
  모르는 id에는 공식 우회값(skip_thought_signature_validator)을 쓴다.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

from agent.core.types import LLMEvent, Message, ToolCall

log = logging.getLogger(__name__)

LLM_MODEL = "gemini-3.5-flash-lite"   # 2026-11-19 실측(docs/decisions/bench_providers.md)
DUMMY_SIGNATURE = b"skip_thought_signature_validator"
THINKING_CANDIDATES: list[dict | None] = [
    {"thinking_level": "MINIMAL"}, {"thinking_budget": 0}, {"thinking_level": "LOW"}, None]


def to_function_declarations(tools: list[dict]) -> list[Any]:
    """OpenAI 형식 [{"type":"function","function":{name,description,parameters}}] → FunctionDeclaration."""
    from google.genai import types
    out = []
    for t in tools or []:
        fn = t.get("function", t)
        kw: dict = {"name": fn["name"], "description": fn.get("description", "")}
        params = fn.get("parameters")
        if params and params.get("properties"):
            kw["parameters_json_schema"] = params
        out.append(types.FunctionDeclaration(**kw))
    return out


def _tool_response(content: str) -> dict:
    try:
        val = json.loads(content)
    except (TypeError, ValueError):
        val = content
    return val if isinstance(val, dict) else {"result": val}


def to_gemini_contents(messages: list[Message], signatures: dict[str, bytes] | None = None) -> list[Any]:
    from google.genai import types
    signatures = signatures or {}
    contents: list[Any] = []
    for m in messages:
        if m.role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=m.content)]))
        elif m.role == "assistant":
            parts = [types.Part(text=m.content)] if m.content else []
            for i, tc in enumerate(m.tool_calls):
                sig = signatures.get(tc.id, DUMMY_SIGNATURE) if i == 0 else signatures.get(tc.id)
                parts.append(types.Part(
                    function_call=types.FunctionCall(id=tc.id, name=tc.name, args=dict(tc.arguments)),
                    thought_signature=sig))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif m.role == "tool":
            part = types.Part(function_response=types.FunctionResponse(
                id=m.tool_call_id or None, name=m.name, response=_tool_response(m.content)))
            prev = contents[-1] if contents else None
            if prev is not None and prev.role == "user" and prev.parts and all(
                    p.function_response is not None for p in prev.parts):
                prev.parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
    return contents


class GeminiLLM:
    def __init__(self, client: Any = None, model: str = LLM_MODEL, temperature: float | None = None,
                 max_output_tokens: int | None = None, thinking: dict | None | str = "auto",
                 api_key: str = ""):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking = thinking
        self.api_key = api_key
        self._signatures: dict[str, bytes] = {}
        self._thinking_idx: dict[str, int] = {}

    def _candidates(self) -> list[dict | None]:
        if self.thinking != "auto":
            return [self.thinking]  # type: ignore[list-item]
        start = self._thinking_idx.get(self.model, 0)
        return THINKING_CANDIDATES[start:]

    def _config(self, system: str, tools: list[dict], thinking: dict | None) -> Any:
        from google.genai import types
        kw: dict = {
            "system_instruction": system or None,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            "thinking_config": types.ThinkingConfig(**thinking) if thinking else types.ThinkingConfig(),
        }
        decls = to_function_declarations(tools)
        if decls:
            kw["tools"] = [types.Tool(function_declarations=decls)]
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.max_output_tokens:
            kw["max_output_tokens"] = self.max_output_tokens
        return types.GenerateContentConfig(**kw)

    async def stream(self, system: str, messages: list[Message],
                     tools: list[dict]) -> AsyncIterator[LLMEvent]:
        if self.client is None:
            from agent.providers._gemini import make_gemini_client
            self.client = make_gemini_client(self.api_key)
        contents = to_gemini_contents(messages, self._signatures)
        cands = self._candidates()
        for i, thinking in enumerate(cands):
            emitted = False
            try:
                stream = await self.client.aio.models.generate_content_stream(
                    model=self.model, contents=contents, config=self._config(system, tools, thinking))
                async for chunk in stream:
                    for ev in self._events(chunk):
                        emitted = True
                        yield ev
            except Exception as e:
                if emitted or i == len(cands) - 1 or not _is_bad_request(e):
                    raise
                log.info("gemini %s thinking %s rejected, next: %s", self.model, thinking, e)
                continue
            if self.thinking == "auto":
                self._thinking_idx[self.model] = THINKING_CANDIDATES.index(thinking)
            yield LLMEvent(type="done")
            return

    def _events(self, chunk: Any) -> list[LLMEvent]:
        out: list[LLMEvent] = []
        for cand in (getattr(chunk, "candidates", None) or [])[:1]:
            content = getattr(cand, "content", None)
            for p in (getattr(content, "parts", None) or []):
                if getattr(p, "thought", None):
                    continue
                fc = getattr(p, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    cid = getattr(fc, "id", None) or f"call_{uuid.uuid4().hex[:12]}"
                    sig = getattr(p, "thought_signature", None)
                    if sig:
                        self._signatures[cid] = sig
                    out.append(LLMEvent(type="tool_call", tool_call=ToolCall(
                        id=cid, name=fc.name, arguments=dict(getattr(fc, "args", None) or {}))))
                elif getattr(p, "text", None):
                    out.append(LLMEvent(type="text", text=p.text))
        return out


def _is_bad_request(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code == 400 or "400" in str(e)[:40] or "INVALID_ARGUMENT" in str(e)
