"""AG-T15c: 동시 요청(hedging) LLM + Gemini 컨텍스트 캐시. 네트워크 없음(fake LLM·fake client)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from agent.core.types import LLMEvent, Message, ToolCall


class ScriptLLM:
    """script: float=대기(초), Exception=발생, LLMEvent=내보냄. 호출마다 같은 스크립트(scripts면 차례로)."""

    def __init__(self, model: str, script=None, scripts=None):
        self.model = model
        self.scripts = list(scripts) if scripts is not None else None
        self.script = script or []
        self.calls = 0
        self.started_at: list[float] = []
        self.closed = 0
        self.cancelled = 0
        self.seen: list[list[Message]] = []

    async def stream(self, system, messages, tools):
        self.calls += 1
        self.started_at.append(asyncio.get_running_loop().time())
        self.seen.append(list(messages))
        script = self.scripts.pop(0) if self.scripts else self.script
        try:
            for item in script:
                if isinstance(item, (int, float)):
                    await asyncio.sleep(item)
                elif isinstance(item, BaseException):
                    raise item
                else:
                    yield item
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.closed += 1


def text(t: str) -> LLMEvent:
    return LLMEvent(type="text", text=t)


def tool(cid: str, name: str = "track_delivery") -> LLMEvent:
    return LLMEvent(type="tool_call", tool_call=ToolCall(id=cid, name=name, arguments={"order_id": "1"}))


DONE = LLMEvent(type="done")
Q = [Message("user", "배송 어디쯤?")]


async def collect(agen):
    return [e async for e in agen]


def run(llm, messages=Q, timeout=3.0):
    return asyncio.wait_for(collect(llm.stream("sys", messages, [])), timeout)


# ── hedging ───────────────────────────────────────────
async def test_fast_primary_never_calls_secondary():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [text("빠른 답."), DONE])
    b = ScriptLLM("b", [text("대체."), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=0.2)
    evs = await run(llm)
    assert [e.text for e in evs if e.type == "text"] == ["빠른 답."]
    assert b.calls == 0
    assert llm.last_model == "a"
    assert llm.stats.get("served:a") == 1


async def test_slow_primary_starts_secondary_after_delay_and_adopts_first():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [1.0, text("늦은 답."), DONE])
    b = ScriptLLM("b", [text("대체 답."), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=0.05)
    t0 = asyncio.get_running_loop().time()
    evs = await run(llm)
    assert [e.text for e in evs if e.type == "text"] == ["대체 답."]
    assert b.calls == 1
    assert b.started_at[0] - t0 >= 0.04                     # h ms 뒤에 시작
    assert a.cancelled == 1 and a.closed == 1               # 진 쪽은 취소
    assert llm.last_model == "b"
    assert llm.stats.get("hedge:b") == 1


async def test_primary_can_still_win_after_hedge_started():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [0.1, text("기본 답."), DONE])
    b = ScriptLLM("b", [1.0, text("대체 답."), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=0.03)
    evs = await run(llm)
    assert [e.text for e in evs if e.type == "text"] == ["기본 답."]
    assert b.calls == 1 and b.cancelled == 1 and b.closed == 1
    assert llm.last_model == "a"


async def test_no_switch_after_first_event():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [text("첫 문장."), 0.3, text("둘째 문장."), DONE])
    b = ScriptLLM("b", [text("대체."), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=0.05)
    evs = await run(llm)
    assert [e.text for e in evs if e.type == "text"] == ["첫 문장.", "둘째 문장."]
    assert evs[-1].type == "done"
    assert b.calls == 0


async def test_tool_call_event_counts_as_first_event():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [1.0, text("늦음."), DONE])
    b = ScriptLLM("b", [tool("c1"), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=0.05)
    evs = await run(llm)
    assert [e.type for e in evs] == ["tool_call", "done"]
    assert evs[0].tool_call.id == "c1"
    assert a.cancelled == 1


async def test_primary_error_before_delay_starts_secondary_immediately():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [RuntimeError("503 UNAVAILABLE")])
    b = ScriptLLM("b", [text("대체."), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=1.0)
    t0 = asyncio.get_running_loop().time()
    evs = await run(llm)
    assert [e.text for e in evs if e.type == "text"] == ["대체."]
    assert b.started_at[0] - t0 < 0.5
    assert llm.stats.get("error:a") == 1


async def test_one_side_error_waits_for_other():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [0.2, text("기본."), DONE])
    b = ScriptLLM("b", [RuntimeError("429")])
    llm = HedgedLLM([a, b], hedge_delay_s=0.02)
    evs = await run(llm)
    assert [e.text for e in evs if e.type == "text"] == ["기본."]


async def test_both_fail_raises():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [0.05, RuntimeError("503 a")])
    b = ScriptLLM("b", [RuntimeError("503 b")])
    llm = HedgedLLM([a, b], hedge_delay_s=0.01)
    with pytest.raises(RuntimeError):
        await run(llm)


async def test_rest_of_chain_used_sequentially_when_both_fail():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [RuntimeError("503 a")])
    b = ScriptLLM("b", [RuntimeError("503 b")])
    c = ScriptLLM("c", [text("셋째."), DONE])
    llm = HedgedLLM([a, b, c], hedge_delay_s=0.01)
    evs = await run(llm)
    assert [e.text for e in evs if e.type == "text"] == ["셋째."]
    assert llm.last_model == "c"


async def test_race_mode_starts_both_at_once():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [0.3, text("기본."), DONE])
    b = ScriptLLM("b", [0.05, text("대체."), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=5.0, mode="race")
    t0 = asyncio.get_running_loop().time()
    evs = await run(llm)
    assert b.started_at[0] - t0 < 0.05
    assert [e.text for e in evs if e.type == "text"] == ["대체."]
    assert a.cancelled == 1


async def test_tool_followup_round_pinned_to_same_model():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", scripts=[[1.0, tool("ca"), DONE], [text("기본 후속."), DONE]])
    b = ScriptLLM("b", scripts=[[tool("cb"), DONE], [0.2, text("대체 후속."), DONE]])
    llm = HedgedLLM([a, b], hedge_delay_s=0.05)
    evs = await run(llm)
    tc = evs[0].tool_call
    assert tc.id == "cb"
    a_calls = a.calls
    follow = Q + [Message("assistant", tool_calls=[tc]),
                  Message("tool", content='{"status":"ok"}', tool_call_id="cb", name="track_delivery")]
    evs2 = await run(llm, follow)
    assert [e.text for e in evs2 if e.type == "text"] == ["대체 후속."]   # h를 넘겨도 대체 모델(같은 모델) 유지
    assert a.calls == a_calls                                             # 기본 모델 미호출
    assert llm.stats.get("pinned:b") == 1


async def test_pinned_model_error_falls_back_to_other():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", scripts=[[tool("ca"), DONE], [RuntimeError("503")]])
    b = ScriptLLM("b", [text("대체."), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=0.5)
    evs = await run(llm)
    tc = evs[0].tool_call
    follow = Q + [Message("assistant", tool_calls=[tc]),
                  Message("tool", content="{}", tool_call_id="ca", name="track_delivery")]
    evs2 = await run(llm, follow)
    assert [e.text for e in evs2 if e.type == "text"] == ["대체."]


async def test_pinned_model_slow_beyond_first_event_timeout_falls_back():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", scripts=[[tool("ca"), DONE], [5.0, text("늦음."), DONE]])
    b = ScriptLLM("b", [text("대체."), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=0.5, first_event_timeout_s=0.05)
    evs = await run(llm)
    follow = Q + [Message("assistant", tool_calls=[evs[0].tool_call]),
                  Message("tool", content="{}", tool_call_id="ca", name="track_delivery")]
    evs2 = await run(llm, follow)
    assert [e.text for e in evs2 if e.type == "text"] == ["대체."]
    assert a.cancelled == 1


async def test_new_user_turn_is_not_pinned():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", scripts=[[1.0, text("x"), DONE], [text("기본."), DONE]])
    b = ScriptLLM("b", scripts=[[tool("cb"), DONE], [text("대체."), DONE]])
    llm = HedgedLLM([a, b], hedge_delay_s=0.05)
    evs = await run(llm)
    tc = evs[0].tool_call
    nxt = Q + [Message("assistant", tool_calls=[tc]),
               Message("tool", content="{}", tool_call_id="cb", name="track_delivery"),
               Message("assistant", content="배송 중입니다."), Message("user", "고마워요")]
    evs2 = await run(llm, nxt)
    assert [e.text for e in evs2 if e.type == "text"] == ["기본."]


async def test_aclose_midstream_cancels_everything():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [1.0, text("늦음."), DONE])
    b = ScriptLLM("b", [text("첫."), 5.0, text("둘."), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=0.02)
    agen = llm.stream("sys", Q, [])
    first = await asyncio.wait_for(agen.__anext__(), 2.0)
    assert first.text == "첫."
    await agen.aclose()
    await asyncio.sleep(0.01)
    assert a.closed == 1 and b.closed == 1
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert pending == []


async def test_single_model_chain_passthrough_and_llms_attr():
    from agent.providers.hedged_llm import HedgedLLM

    a = ScriptLLM("a", [text("네."), DONE])
    b = ScriptLLM("b", [text("x"), DONE])
    llm = HedgedLLM([a, b], hedge_delay_s=0.01)
    assert llm.model == "a" and [m.model for m in llm.llms] == ["a", "b"]
    llm.llms = [a]                                  # 서버 예열이 404 모델을 빼는 방식
    evs = await run(llm)
    assert [e.text for e in evs if e.type == "text"] == ["네."]


def test_hedged_requires_models():
    from agent.providers.hedged_llm import HedgedLLM

    with pytest.raises(ValueError):
        HedgedLLM([])
    with pytest.raises(ValueError):
        HedgedLLM([ScriptLLM("a")], mode="nope")


# ── factory ───────────────────────────────────────────
def test_factory_llm_mode_hedge_and_fallback():
    from agent.providers.factory import ProviderConfig, make_llm_hedged, make_llm_with_fallback
    from agent.providers.fallback_llm import FallbackLLM
    from agent.providers.hedged_llm import HedgedLLM

    env = {"GEMINI_API_KEY": "g", "LLM_PROVIDER": "gemini", "GEMINI_LLM_FALLBACK_MODELS": "gemini-3.1-flash-lite"}
    h = make_llm_with_fallback(ProviderConfig.from_env({**env, "LLM_MODE": "hedge", "LLM_HEDGE_DELAY_MS": "700"}),
                               gemini_client=NS())
    assert isinstance(h, HedgedLLM)
    assert [m.model for m in h.llms] == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    assert h.hedge_delay_s == pytest.approx(0.7) and h.mode == "hedge"
    assert h.first_event_timeout_s == 2.5
    r = make_llm_with_fallback(ProviderConfig.from_env({**env, "LLM_MODE": "race"}), gemini_client=NS())
    assert isinstance(r, HedgedLLM) and r.mode == "race"
    f = make_llm_with_fallback(ProviderConfig.from_env({**env, "LLM_MODE": "fallback"}), gemini_client=NS())
    assert isinstance(f, FallbackLLM) and not isinstance(f, HedgedLLM)
    single = make_llm_with_fallback(ProviderConfig.from_env({**env, "LLM_MODE": "single"}), gemini_client=NS())
    assert single.model == "gemini-3.5-flash-lite" and not hasattr(single, "llms")
    direct = make_llm_hedged(ProviderConfig.from_env(env), gemini_client=NS())
    assert isinstance(direct, HedgedLLM)
    # 기본값(AG-T15c 실측): hedge 500ms + 컨텍스트 캐시
    dflt = make_llm_with_fallback(ProviderConfig.from_env(env), gemini_client=NS())
    assert isinstance(dflt, HedgedLLM) and dflt.mode == "hedge" and dflt.hedge_delay_s == pytest.approx(0.5)
    assert all(m.context_cache is not None for m in dflt.llms)


def test_factory_context_cache_switch():
    from agent.providers.factory import ProviderConfig, make_llm_with_fallback

    env = {"GEMINI_API_KEY": "g", "LLM_PROVIDER": "gemini", "GEMINI_LLM_FALLBACK_MODELS": "gemini-3.1-flash-lite",
           "LLM_MODE": "hedge"}
    on = make_llm_with_fallback(ProviderConfig.from_env({**env, "LLM_CONTEXT_CACHE": "1"}), gemini_client=NS())
    assert all(m.context_cache is not None for m in on.llms)
    assert on.llms[0].context_cache is on.llms[1].context_cache        # 공유(모델별 키)
    off = make_llm_with_fallback(ProviderConfig.from_env({**env, "LLM_CONTEXT_CACHE": "0"}), gemini_client=NS())
    assert all(m.context_cache is None for m in off.llms)


# ── 컨텍스트 캐시 ─────────────────────────────────────
MARK = "## 이번 통화"
PREFIX = "너는 상담원이다.\n" + "규칙 " * 3000
SYS1 = PREFIX + f"\n\n{MARK}\n- 현재 시각: 2026-11-20 14:50"
SYS2 = PREFIX + f"\n\n{MARK}\n- 현재 시각: 2026-11-21 09:00"
TOOLS = [{"type": "function", "function": {"name": "track_delivery", "description": "배송 조회",
          "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                         "required": ["order_id"]}}}]


def test_split_system_fixed_prefix():
    from agent.providers.llm_cache import split_system

    pre, suf = split_system(SYS1)
    assert pre == PREFIX + "\n\n" and suf.startswith(MARK)
    assert split_system("마커 없는 프롬프트") is None


def test_cache_key_depends_on_prefix_tools_model_not_suffix():
    from agent.providers.llm_cache import cache_key, split_system

    p1, _ = split_system(SYS1)
    p2, _ = split_system(SYS2)
    k = cache_key("m", p1, TOOLS)
    assert k == cache_key("m", p2, TOOLS)                 # 접미부(시각)만 다르면 같은 키
    assert k != cache_key("m2", p1, TOOLS)
    assert k != cache_key("m", p1 + "x", TOOLS)
    assert k != cache_key("m", p1, [])
    assert len(k) == 64


class FakeCaches:
    def __init__(self, fail: Exception | None = None, delay: float = 0.0):
        self.created: list[dict] = []
        self.fail = fail
        self.delay = delay

    async def create(self, *, model, config):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.created.append({"model": model, "config": config})
        if self.fail:
            raise self.fail
        return NS(name=f"cachedContents/c{len(self.created)}", usage_metadata=NS(total_token_count=4000))


async def test_context_cache_below_min_never_creates():
    from agent.providers.llm_cache import ContextCache

    caches = FakeCaches()
    cc = ContextCache(client=NS(aio=NS(caches=caches)), min_tokens=1024)
    assert cc.lookup("m", "짧은 접두부", TOOLS) is None
    await asyncio.sleep(0.01)
    assert cc.lookup("m", "짧은 접두부", TOOLS) is None
    assert caches.created == []
    assert cc.stats.get("too_small") == 2


async def test_context_cache_creates_in_background_then_reuses():
    from agent.providers.llm_cache import ContextCache, split_system

    caches = FakeCaches()
    cc = ContextCache(client=NS(aio=NS(caches=caches)), min_tokens=1024, ttl_s=600)
    pre, _ = split_system(SYS1)
    assert cc.lookup("m", pre, TOOLS) is None             # 첫 요청은 기다리지 않고 캐시 없이
    await cc.wait_pending()
    name = cc.lookup("m", pre, TOOLS)
    assert name == "cachedContents/c1"
    assert cc.lookup("m", split_system(SYS2)[0], TOOLS) == name
    assert len(caches.created) == 1
    cfg = caches.created[0]["config"]
    assert cfg.system_instruction == pre and cfg.tools and cfg.ttl == "600s"
    cc.invalidate("m", pre, TOOLS)
    assert cc.lookup("m", pre, TOOLS) is None
    await cc.wait_pending()
    assert len(caches.created) == 2


async def test_context_cache_create_failure_is_remembered():
    from agent.providers.llm_cache import ContextCache, split_system

    caches = FakeCaches(fail=RuntimeError("400 Cached content is too small. min_total_token_count=4096"))
    cc = ContextCache(client=NS(aio=NS(caches=caches)), min_tokens=1024)
    pre, _ = split_system(SYS1)
    cc.lookup("m", pre, TOOLS)
    await cc.wait_pending()
    for _ in range(3):
        assert cc.lookup("m", pre, TOOLS) is None
    await cc.wait_pending()
    assert len(caches.created) == 1


def _gem_stream(models_calls, fail_first: Exception | None = None):
    def gen(model, contents, config, n):
        async def g():
            if fail_first is not None and n == 1:
                raise fail_first
            yield NS(candidates=[NS(content=NS(parts=[NS(text="네.", function_call=None, thought=None,
                                                         thought_signature=None)]))],
                     usage_metadata=NS(prompt_token_count=5000, cached_content_token_count=4000,
                                       candidates_token_count=3, thoughts_token_count=0))
        return g()
    return gen


class FakeModels:
    def __init__(self, stream):
        self._stream = stream
        self.calls: list[dict] = []

    async def generate_content_stream(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._stream(model, contents, config, len(self.calls))


async def test_gemini_llm_uses_cached_content_and_moves_suffix_to_contents():
    from agent.providers.gemini_llm import GeminiLLM
    from agent.providers.llm_cache import ContextCache

    caches = FakeCaches()
    models = FakeModels(_gem_stream(None))
    client = NS(aio=NS(models=models, caches=caches))
    usage: list[dict] = []
    llm = GeminiLLM(client=client, model="m", context_cache=ContextCache(client=client, min_tokens=1024),
                    on_usage=usage.append)
    await collect(llm.stream(SYS1, Q, TOOLS))               # 캐시 없음(백그라운드 생성 시작)
    first = models.calls[0]["config"]
    assert first.system_instruction == SYS1 and first.tools and not first.cached_content
    await llm.context_cache.wait_pending()
    await collect(llm.stream(SYS2, Q, TOOLS))
    call = models.calls[1]
    cfg = call["config"]
    assert cfg.cached_content == "cachedContents/c1"
    assert cfg.system_instruction is None and not cfg.tools
    # 접미부는 고객 발화와 섞지 않고 별도 user 턴(안내 머리말) + model 확인 턴으로 앞에 둔다
    # (같은 user 파트에 붙이면 '번호부터 묻기'가 늘었다 — llm_latency_r2.md)
    head, ack = call["contents"][0], call["contents"][1]
    assert head.role == "user" and MARK in head.parts[0].text and "09:00" in head.parts[0].text
    assert ack.role == "model" and ack.parts[0].text
    assert call["contents"][-1].role == "user" and call["contents"][-1].parts[-1].text == "배송 어디쯤?"
    assert len(call["contents"][-1].parts) == 1
    assert usage[-1]["cached"] == 4000 and usage[-1]["prompt"] == 5000 and usage[-1]["model"] == "m"
    assert llm.usage_totals["calls"] == 2


async def test_gemini_llm_cache_error_retries_without_cache():
    from agent.providers.gemini_llm import GeminiLLM
    from agent.providers.llm_cache import ContextCache

    caches = FakeCaches()
    err = RuntimeError("403 PERMISSION_DENIED CachedContent not found (or permission denied)")
    models = FakeModels(_gem_stream(None, fail_first=None))
    client = NS(aio=NS(models=models, caches=caches))
    cc = ContextCache(client=client, min_tokens=1024)
    llm = GeminiLLM(client=client, model="m", context_cache=cc)
    cc.lookup("m", split_system_prefix(SYS1), TOOLS)
    await cc.wait_pending()
    models._stream = _gem_stream(None, fail_first=err)
    models.calls.clear()
    evs = await collect(llm.stream(SYS1, Q, TOOLS))
    assert [e.type for e in evs] == ["text", "done"]
    assert models.calls[0]["config"].cached_content
    assert not models.calls[1]["config"].cached_content and models.calls[1]["config"].system_instruction == SYS1
    assert cc.stats.get("invalidated") == 1


def split_system_prefix(s):
    from agent.providers.llm_cache import split_system
    return split_system(s)[0]


async def test_gemini_llm_without_marker_or_cache_unchanged():
    from agent.providers.gemini_llm import GeminiLLM
    from agent.providers.llm_cache import ContextCache

    caches = FakeCaches()
    models = FakeModels(_gem_stream(None))
    client = NS(aio=NS(models=models, caches=caches))
    llm = GeminiLLM(client=client, model="m", context_cache=ContextCache(client=client, min_tokens=1024))
    await collect(llm.stream("한 단어로만 답하세요.", Q, []))
    await llm.context_cache.wait_pending()
    assert caches.created == []
    assert models.calls[0]["config"].system_instruction == "한 단어로만 답하세요."
