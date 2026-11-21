"""AG-T41: 대회 규정 LLM(gpt-5.4-mini) 고정. 다른 모델로 새는 경로(폴백·hedge) 차단. 네트워크 없음."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest


def _cfg(**env):
    from agent.providers.factory import ProviderConfig
    return ProviderConfig.from_env(env)


# ── 체인: OpenAI 기본이면 Gemini를 붙이지 않는다 ──────────────
@pytest.mark.parametrize("keys", [
    {"OPENAI_API_KEY": "o", "GEMINI_API_KEY": "g"},
    {"BAZE_API_KEY": "b"},
    {"BAZE_API_KEY": "b", "GEMINI_API_KEY": "g"},
])
@pytest.mark.parametrize("mode", ["hedge", "race", "fallback", "single"])
def test_openai_chain_has_no_gemini(keys, mode):
    from agent.providers.factory import llm_models, make_llm_with_fallback
    from agent.providers.gemini_llm import GeminiLLM

    llm = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", LLM_MODE=mode, **keys),
                                 gemini_client=NS(), openai_client=NS())
    chain = list(getattr(llm, "llms", None) or [llm])
    assert not any(isinstance(m, GeminiLLM) for m in chain)
    assert set(llm_models(llm)) == {"gpt-5.4-mini"}


def test_openai_hedge_uses_same_model_replica():
    from agent.providers.factory import make_llm_with_fallback
    from agent.providers.hedged_llm import HedgedLLM
    from agent.providers.openai_llm import OpenAILLM

    client = NS()
    h = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o", GEMINI_API_KEY="g",
                                    LLM_HEDGE_DELAY_MS="700"), openai_client=client)
    assert isinstance(h, HedgedLLM) and h.mode == "hedge" and h.hedge_delay_s == pytest.approx(0.7)
    assert len(h.llms) == 2 and all(isinstance(m, OpenAILLM) for m in h.llms)
    assert h.llms[0] is not h.llms[1]
    assert [m.model for m in h.llms] == ["gpt-5.4-mini", "gpt-5.4-mini"]
    assert h.llms[1].client is client


def test_openai_replica_keeps_baze_route():
    from agent.providers.factory import make_llm_with_fallback

    h = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", BAZE_API_KEY="b"))
    assert [m.base_url for m in h.llms] == [h.llms[0].base_url] * 2 and h.llms[0].base_url
    assert all(m.api_key == "b" for m in h.llms)


def test_same_model_hedge_switch_off_gives_single():
    from agent.providers.factory import make_llm_with_fallback
    from agent.providers.openai_llm import OpenAILLM

    llm = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o", LLM_SAME_MODEL_HEDGE="0"),
                                 openai_client=NS())
    assert isinstance(llm, OpenAILLM)


def test_explicit_openai_fallback_models_are_still_openai():
    """GEMINI_LLM_FALLBACK_MODELS를 줘도 OpenAI 기본이면 무시한다(이름 그대로 Gemini 전용)."""
    from agent.providers.factory import llm_models, make_llm_with_fallback

    llm = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o", GEMINI_API_KEY="g",
                                      GEMINI_LLM_FALLBACK_MODELS="gemini-3.1-flash-lite"),
                                 gemini_client=NS(), openai_client=NS())
    assert set(llm_models(llm)) == {"gpt-5.4-mini"}


# ── 정책 검사 ───────────────────────────────────────────
def test_check_llm_policy_accepts_allowed_chain():
    from agent.providers.factory import check_llm_policy, make_llm_with_fallback

    llm = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o"), openai_client=NS())
    check_llm_policy(llm, "gpt-5.4-mini")


def test_check_llm_policy_rejects_gemini_chain():
    from agent.providers.factory import check_llm_policy, make_llm_with_fallback

    llm = make_llm_with_fallback(_cfg(LLM_PROVIDER="gemini", GEMINI_API_KEY="g"), gemini_client=NS())
    with pytest.raises(ValueError) as e:
        check_llm_policy(llm, "gpt-5.4-mini")
    assert "gemini-3.5-flash-lite" in str(e.value)


def test_check_llm_policy_star_disables_and_list_parses():
    from agent.providers.factory import check_llm_policy

    gem = NS(model="gemini-3.5-flash-lite")
    check_llm_policy(gem, "*")
    check_llm_policy(gem, " gpt-5.4-mini , gemini-3.5-flash-lite ")
    with pytest.raises(ValueError):
        check_llm_policy(NS(llms=[NS(model="gpt-5.4-mini"), gem]), "gpt-5.4-mini")


def test_allowed_models_default_is_contest_model():
    from agent.providers.factory import allowed_llm_models

    assert allowed_llm_models({}) == "gpt-5.4-mini"
    assert allowed_llm_models({"LLM_ALLOWED_MODELS": "*"}) == "*"


def test_auto_provider_prefers_openai_with_baze_key():
    from agent.providers.factory import make_llm
    from agent.providers.openai_llm import OpenAILLM

    assert isinstance(make_llm(_cfg(BAZE_API_KEY="b", GEMINI_API_KEY="g")), OpenAILLM)


# ── 서버: 직접 조립하면 정책 위반 시 시작하지 않는다 ─────────────
def test_serve_refuses_disallowed_llm(monkeypatch):
    import asyncio

    from agent import server

    deps = NS(llm=NS(model="gemini-3.5-flash-lite"))
    monkeypatch.setattr(server, "build_deps", lambda: deps)
    monkeypatch.delenv("LLM_ALLOWED_MODELS", raising=False)

    async def boom(_deps):  # 예열까지 가면 안 된다
        raise AssertionError("warmup reached")
    monkeypatch.setattr(server, "warmup", boom)
    with pytest.raises(SystemExit) as e:
        asyncio.run(server.serve(0))
    assert "gemini-3.5-flash-lite" in str(e.value)


def test_serve_with_injected_deps_skips_policy(monkeypatch):
    """테스트·벤치가 deps를 직접 넣는 경로는 검사하지 않는다(가짜 LLM)."""
    import asyncio

    from agent import server

    async def stop(_deps):
        raise RuntimeError("stop-after-policy")
    monkeypatch.setattr(server, "warmup", stop)
    with pytest.raises(RuntimeError, match="stop-after-policy"):
        asyncio.run(server.serve(0, deps=NS(llm=NS(model="fake"))))
