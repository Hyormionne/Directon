"""Gemini 명시적 컨텍스트 캐시(cachedContents, AG-T15c). docs/decisions/llm_latency_r2.md.

- 시스템 프롬프트(agent.prompts.build_system)는 "고정부(규칙·매뉴얼) + `## 이번 통화`(시각·고객·이력·첫 마디)"다.
  마커 앞의 **고정 접두부 + 툴 정의**를 캐시로 만들고, 접미부는 요청 contents 앞의 별도 user 턴(+model 확인 턴)으로 옮긴다(gemini_llm._with_suffix).
  (cached_content를 쓰는 요청에는 system_instruction·tools·tool_config를 줄 수 없다 — 400, 2026-11-19 실측)
- 키 = sha256(모델, 접두부, 툴 정의). 접미부가 달라도 같은 캐시를 쓴다. 캐시는 모델별이다.
- 첫 요청은 기다리지 않는다: 캐시가 없으면 백그라운드로 만들고 이번 요청은 캐시 없이 보낸다.
- 최소 크기: flash-lite 명시적 캐시 최소 1024토큰(실측 400 "min_total_token_count=1024").
  추정 토큰(UTF-8 바이트/4, 한국어 실측 약 4.1바이트/토큰)이 min_tokens 미만이면 만들지 않는다.
  생성이 실패하면 fail_retry_s 동안 다시 시도하지 않는다.
- TTL(기본 1시간) 만료 refresh_margin_s 전부터 새 캐시를 만든다(이전 캐시는 TTL로 자연 만료).
- 비용(2026-11 가격표): 캐시 입력 3.5-flash-lite $0.03/1M(일반 $0.30), 3.1-flash-lite $0.025/1M,
  저장 $1.00/1M토큰/시간. 접두부 약 5.8천 토큰이면 모델당 시간당 약 $0.006.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MARKERS: tuple[str, ...] = ("## 이번 통화",)
MIN_TOKENS = 1024
DEFAULT_TTL_S = 3600
REFRESH_MARGIN_S = 120.0
FAIL_RETRY_S = 600.0


def estimate_tokens(text: str) -> int:
    return len((text or "").encode("utf-8")) // 4


def split_system(system: str, markers: tuple[str, ...] = DEFAULT_MARKERS) -> tuple[str, str] | None:
    """(고정 접두부, 가변 접미부). 마커가 없거나 접두부가 비면 None."""
    best = -1
    for mk in markers:
        if system.startswith(mk):
            return None
        i = system.find("\n" + mk)
        if i >= 0 and (best < 0 or i < best):
            best = i + 1
    if best <= 0:
        return None
    return system[:best], system[best:]


def cache_key(model: str, prefix: str, tools: list[dict] | None) -> str:
    raw = json.dumps([model, prefix, tools or []], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ContextCache:
    """여러 GeminiLLM(모델)이 공유할 수 있다. lookup은 동기·즉시 반환(실행 중 이벤트 루프 필요)."""

    def __init__(self, client: Any, min_tokens: int = MIN_TOKENS, ttl_s: int = DEFAULT_TTL_S,
                 refresh_margin_s: float = REFRESH_MARGIN_S, fail_retry_s: float = FAIL_RETRY_S,
                 markers: tuple[str, ...] = DEFAULT_MARKERS, clock: Any = time.monotonic):
        self.client = client
        self.min_tokens = int(min_tokens)
        self.ttl_s = int(ttl_s)
        self.refresh_margin_s = float(refresh_margin_s)
        self.fail_retry_s = float(fail_retry_s)
        self.markers = tuple(markers)
        self.clock = clock
        self._entries: dict[str, tuple[str, float]] = {}      # key → (cache name, 만료 시각)
        self._failed: dict[str, float] = {}                    # key → 재시도 가능 시각
        self._pending: dict[str, asyncio.Task] = {}
        self.created: list[str] = []
        self.stats: dict[str, int] = {}

    def _count(self, k: str) -> None:
        self.stats[k] = self.stats.get(k, 0) + 1

    def lookup(self, model: str, prefix: str, tools: list[dict] | None) -> str | None:
        tok = estimate_tokens(prefix) + estimate_tokens(json.dumps(tools or [], ensure_ascii=False))
        if tok < self.min_tokens:
            self._count("too_small")
            return None
        key = cache_key(model, prefix, tools)
        now = self.clock()
        ent = self._entries.get(key)
        if ent is not None and now < ent[1] - self.refresh_margin_s:
            self._count("hit")
            return ent[0]
        if self._failed.get(key, 0.0) > now:
            self._count("skip_failed")
            return ent[0] if ent is not None and now < ent[1] else None
        if key not in self._pending:
            self._pending[key] = asyncio.ensure_future(self._create(key, model, prefix, tools))
        self._count("miss")
        return ent[0] if ent is not None and now < ent[1] else None

    async def _create(self, key: str, model: str, prefix: str, tools: list[dict] | None) -> None:
        from google.genai import types

        from agent.providers.gemini_llm import to_function_declarations
        try:
            kw: dict = {"system_instruction": prefix, "ttl": f"{self.ttl_s}s"}
            decls = to_function_declarations(tools or [])
            if decls:
                kw["tools"] = [types.Tool(function_declarations=decls)]
            t0 = self.clock()
            cc = await self.client.aio.caches.create(model=model, config=types.CreateCachedContentConfig(**kw))
            self._entries[key] = (cc.name, t0 + self.ttl_s)
            self._failed.pop(key, None)
            self.created.append(cc.name)
            self._count("created")
            um = getattr(cc, "usage_metadata", None)
            log.info("context cache %s for %s (%s tokens)", cc.name, model, getattr(um, "total_token_count", "?"))
        except Exception as e:  # noqa: BLE001 — 캐시는 최적화일 뿐, 실패해도 요청은 캐시 없이 간다
            self._failed[key] = self.clock() + self.fail_retry_s
            self._count("create_failed")
            log.warning("context cache create failed for %s: %r", model, e)
        finally:
            self._pending.pop(key, None)

    def invalidate(self, model: str, prefix: str, tools: list[dict] | None) -> None:
        if self._entries.pop(cache_key(model, prefix, tools), None) is not None:
            self._count("invalidated")

    async def wait_pending(self) -> None:
        tasks = list(self._pending.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        """만든 캐시를 지운다(저장 비용 정지). 측정 스크립트 정리용."""
        await self.wait_pending()
        for name in self.created:
            try:
                await self.client.aio.caches.delete(name=name)
            except Exception as e:  # noqa: BLE001
                log.info("context cache delete failed %s: %r", name, e)
        self.created.clear()
        self._entries.clear()


def is_cache_error(e: BaseException) -> bool:
    s = str(e).lower()
    return "cachedcontent" in s or "cached content" in s or "cached_content" in s
