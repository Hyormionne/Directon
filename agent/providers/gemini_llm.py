"""Gemini 텍스트 LLM: 스트리밍 + function calling.

- 시스템 프롬프트는 system_instruction, 자동 함수 호출은 끈다(실행은 상위 계층).
- thinking은 모델이 허용하는 최저로: 후보 [level MINIMAL → budget 0 → level LOW → 미지정]을
  차례로 시도하고, 첫 출력 전에 실패하면 다음 후보로 넘어간다. 성공한 후보는 모델별로 기억한다.
  (2026-11-19 실측: gemini-3.8-flash는 MINIMAL 400·budget 0 허용, gemini-3.5-flash-lite는
  MINIMAL 허용·budget 0 400, gemini-3.1-flash-lite/3.5-flash는 둘 다 허용)
- Gemini 3 계열은 이전 턴의 function_call 파트에 thought_signature를 되돌려야 한다.
  ToolCall에는 그 자리가 없으므로 호출 id → 서명을 이 객체가 기억해 두었다가 변환할 때 붙인다.
  모르는 id에는 공식 우회값(skip_thought_signature_validator)을 쓴다.
- 컨텍스트 캐시(AG-T15c, 선택): context_cache(llm_cache.ContextCache)를 주면 system의 고정 접두부+툴 정의를
  cachedContent로 만들어 재사용한다. 캐시를 쓰는 요청은 system_instruction·tools 없이 cached_content를 주고,
  가변 접미부는 contents 앞의 별도 user 턴(+model 확인 턴)으로 옮긴다. 캐시 관련 오류(만료·삭제)는 캐시를 버리고 캐시 없이 재시도한다.
- usage(AG-T15c): 스트림 끝의 usage_metadata를 on_usage(dict) 콜백과 usage_totals에 남긴다.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Callable

from agent.core.types import LLMEvent, Message, ToolCall
from agent.providers.llm_cache import is_cache_error, split_system

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
                 api_key: str = "", context_cache: Any = None,
                 on_usage: Callable[[dict], None] | None = None):
        self.client = client
        self.context_cache = context_cache
        self.on_usage = on_usage
        self.usage_totals: dict[str, int] = {"calls": 0, "prompt": 0, "cached": 0, "output": 0, "thoughts": 0}
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

    def _config(self, system: str, tools: list[dict], thinking: dict | None, cached: str | None = None) -> Any:
        from google.genai import types
        kw: dict = {
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            "thinking_config": types.ThinkingConfig(**thinking) if thinking else types.ThinkingConfig(),
        }
        if cached:
            kw["cached_content"] = cached
        else:
            kw["system_instruction"] = system or None
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
        split = None
        cached: str | None = None
        if self.context_cache is not None and system:
            split = split_system(system, self.context_cache.markers)
            if split is not None:
                cached = self.context_cache.lookup(self.model, split[0], tools)
        cands = self._candidates()
        i = 0
        while i < len(cands):
            thinking = cands[i]
            emitted = False
            usage: Any = None
            try:
                req = _with_suffix(contents, split[1]) if cached and split else contents
                stream = await self.client.aio.models.generate_content_stream(
                    model=self.model, contents=req, config=self._config(system, tools, thinking, cached))
                async for chunk in stream:
                    usage = getattr(chunk, "usage_metadata", None) or usage
                    for ev in self._events(chunk):
                        emitted = True
                        yield ev
            except Exception as e:
                if not emitted and cached and split and is_cache_error(e):
                    log.info("gemini %s cached content rejected, retry without cache: %s", self.model, e)
                    self.context_cache.invalidate(self.model, split[0], tools)
                    cached = None
                    continue
                if emitted or i == len(cands) - 1 or not _is_bad_request(e):
                    raise
                log.info("gemini %s thinking %s rejected, next: %s", self.model, thinking, e)
                i += 1
                continue
            if self.thinking == "auto":
                self._thinking_idx[self.model] = THINKING_CANDIDATES.index(thinking)
            self._record_usage(usage, bool(cached))
            yield LLMEvent(type="done")
            return

    def _record_usage(self, um: Any, cache_used: bool) -> None:
        u = {"model": self.model, "cache_used": cache_used,
             "prompt": int(getattr(um, "prompt_token_count", 0) or 0),
             "cached": int(getattr(um, "cached_content_token_count", 0) or 0),
             "output": int(getattr(um, "candidates_token_count", 0) or 0),
             "thoughts": int(getattr(um, "thoughts_token_count", 0) or 0)}
        self.usage_totals["calls"] += 1
        for k in ("prompt", "cached", "output", "thoughts"):
            self.usage_totals[k] += u[k]
        if self.on_usage is not None:
            try:
                self.on_usage(u)
            except Exception:  # noqa: BLE001
                log.debug("on_usage failed", exc_info=True)

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


SUFFIX_HEADER = "[상담 시스템 안내 — 고객 발화 아님]\n"
SUFFIX_ACK = "네, 이번 통화 정보를 확인했습니다."


def _with_suffix(contents: list[Any], suffix: str) -> list[Any]:
    """캐시 요청: 가변 접미부를 별도 user 턴(안내 머리말) + model 확인 턴으로 대화 앞에 둔다.
    고객 첫 발화와 같은 user 파트에 붙이면 '발신 고객 정보: 없음 … 번호를 물어본다' 지시가 지나치게 두드러져
    재고 문의에도 번호부터 묻는 일이 늘었다(3/8 vs 8/8, docs/decisions/llm_latency_r2.md)."""
    from google.genai import types
    head = types.Content(role="user", parts=[types.Part(text=SUFFIX_HEADER + suffix)])
    ack = types.Content(role="model", parts=[types.Part(text=SUFFIX_ACK)])
    return [head, ack] + list(contents)


def _is_bad_request(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code == 400 or "400" in str(e)[:40] or "INVALID_ARGUMENT" in str(e)
