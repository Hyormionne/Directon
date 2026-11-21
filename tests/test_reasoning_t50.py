"""AG-T50: gpt-5.4-mini reasoning effort 설정(기본 medium), effort별 hedge·첫 이벤트 타이밍,
store=False 툴 라운드 사이 reasoning item 되돌리기, reasoning 스트림 이벤트 무시. 네트워크 없음(fake client)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from agent.core.types import Message, ToolCall

TOOLS = [{"type": "function", "function": {"name": "get_order_info", "description": "주문 조회",
                                           "parameters": {"type": "object", "properties": {}}}}]


class FakeResponses:
    """create 호출마다 events_list에서 하나씩 꺼내 스트림으로 돌려준다."""

    def __init__(self, *events_list):
        self.events_list = list(events_list)
        self.calls: list[dict] = []

    async def create(self, **kw):
        self.calls.append(kw)
        events = self.events_list.pop(0) if self.events_list else [NS(type="response.completed")]

        async def g():
            for e in events:
                yield e
        return g()


async def collect(ait):
    return [e async for e in ait]


def _cfg(**env):
    from agent.providers.factory import ProviderConfig
    return ProviderConfig.from_env(env)


def _tool_round_events(call_id="call_1", encrypted="ENC1", rid="rs_1"):
    """reasoning 이벤트 + 함수 호출 1개."""
    r_item = NS(type="reasoning", id=rid, summary=[], encrypted_content=encrypted)
    fc = NS(type="function_call", name="get_order_info", call_id=call_id, id="fc_1", arguments="{}")
    return [
        NS(type="response.created"),
        NS(type="response.output_item.added", output_index=0, item=NS(type="reasoning", id=rid, summary=[])),
        NS(type="response.reasoning_summary_part.added", output_index=0, part=NS(type="summary_text", text="")),
        NS(type="response.reasoning_summary_text.delta", output_index=0, delta="고객이 주문을 묻는다"),
        NS(type="response.reasoning_text.delta", output_index=0, delta="생각 중"),
        NS(type="response.reasoning_summary_text.done", output_index=0, text="고객이 주문을 묻는다"),
        NS(type="response.output_item.done", output_index=0, item=r_item),
        NS(type="response.output_item.added", output_index=1,
           item=NS(type="function_call", name="get_order_info", call_id=call_id, id="fc_1", arguments="")),
        NS(type="response.output_item.done", output_index=1, item=fc),
        NS(type="response.completed", response=NS()),
    ]


def _followup_messages(call_id="call_1"):
    return [Message("user", "주문 어디예요?"),
            Message("assistant", "", tool_calls=[ToolCall(call_id, "get_order_info", {})]),
            Message("tool", '{"status":"배송중"}', tool_call_id=call_id, name="get_order_info")]


# ── OpenAILLM: effort·include ─────────────────────────────
async def test_default_effort_is_medium_and_requests_encrypted_reasoning():
    from agent.providers.openai_llm import OpenAILLM

    resp = FakeResponses([NS(type="response.completed")])
    llm = OpenAILLM(client=NS(responses=resp))
    assert llm.reasoning_effort == "medium"
    await collect(llm.stream("s", [Message("user", "x")], []))
    kw = resp.calls[0]
    assert kw["reasoning"] == {"effort": "medium"}
    assert kw["store"] is False
    assert kw["include"] == ["reasoning.encrypted_content"]


async def test_effort_none_sends_no_include():
    from agent.providers.openai_llm import OpenAILLM

    resp = FakeResponses([NS(type="response.completed")])
    await collect(OpenAILLM(client=NS(responses=resp), reasoning_effort="none").stream("s", [Message("user", "x")], []))
    kw = resp.calls[0]
    assert kw["reasoning"] == {"effort": "none"}
    assert "include" not in kw


async def test_include_switch_off():
    from agent.providers.openai_llm import OpenAILLM

    resp = FakeResponses([NS(type="response.completed")])
    llm = OpenAILLM(client=NS(responses=resp), reasoning_effort="medium", include_reasoning=False)
    await collect(llm.stream("s", [Message("user", "x")], []))
    assert "include" not in resp.calls[0]


def test_invalid_effort_rejected():
    from agent.providers.openai_llm import OpenAILLM

    with pytest.raises(ValueError):
        OpenAILLM(reasoning_effort="ultra")
    assert OpenAILLM(reasoning_effort=" High ").reasoning_effort == "high"


# ── 스트림: reasoning 이벤트는 말하지 않는다 ─────────────────
async def test_reasoning_events_are_never_voiced():
    from agent.providers.openai_llm import OpenAILLM

    events = _tool_round_events()
    events.insert(-1, NS(type="response.output_text.delta", output_index=2, delta="확인할게요."))
    resp = FakeResponses(events)
    out = await collect(OpenAILLM(client=NS(responses=resp)).stream("s", [Message("user", "x")], TOOLS))
    assert [e.type for e in out] == ["tool_call", "text", "done"]
    assert "".join(e.text for e in out if e.type == "text") == "확인할게요."


# ── 툴 라운드 사이 reasoning item 되돌리기 ───────────────────
async def test_reasoning_items_fed_back_after_tool_call():
    from agent.providers.openai_llm import OpenAILLM

    resp = FakeResponses(_tool_round_events(), [NS(type="response.output_text.delta", delta="배송중입니다."),
                                                NS(type="response.completed")])
    llm = OpenAILLM(client=NS(responses=resp))
    out = await collect(llm.stream("s", [Message("user", "주문 어디예요?")], TOOLS))
    assert out[0].tool_call.id == "call_1"
    await collect(llm.stream("s", _followup_messages(), TOOLS))
    inp = resp.calls[1]["input"]
    types = [i.get("type", i.get("role")) for i in inp]
    assert types == ["user", "reasoning", "function_call", "function_call_output"]
    r = inp[1]
    assert r["id"] == "rs_1" and r["encrypted_content"] == "ENC1" and r["summary"] == []


async def test_reasoning_without_encrypted_content_not_fed_back():
    from agent.providers.openai_llm import OpenAILLM

    resp = FakeResponses(_tool_round_events(encrypted=None), [NS(type="response.completed")])
    llm = OpenAILLM(client=NS(responses=resp))
    await collect(llm.stream("s", [Message("user", "x")], TOOLS))
    await collect(llm.stream("s", _followup_messages(), TOOLS))
    assert all(i.get("type") != "reasoning" for i in resp.calls[1]["input"])


async def test_reasoning_not_fed_back_when_include_off():
    from agent.providers.openai_llm import OpenAILLM

    resp = FakeResponses(_tool_round_events(), [NS(type="response.completed")])
    llm = OpenAILLM(client=NS(responses=resp), include_reasoning=False)
    await collect(llm.stream("s", [Message("user", "x")], TOOLS))
    await collect(llm.stream("s", _followup_messages(), TOOLS))
    assert all(i.get("type") != "reasoning" for i in resp.calls[1]["input"])


async def test_reasoning_items_once_for_parallel_calls_and_unknown_ids_skipped():
    from agent.providers.openai_llm import OpenAILLM

    events = _tool_round_events("call_a")
    fc_b = NS(type="function_call", name="get_order_info", call_id="call_b", id="fc_2", arguments="{}")
    events.insert(-1, NS(type="response.output_item.done", output_index=2, item=fc_b))
    resp = FakeResponses(events, [NS(type="response.completed")])
    llm = OpenAILLM(client=NS(responses=resp))
    await collect(llm.stream("s", [Message("user", "x")], TOOLS))
    msgs = [Message("user", "x"),
            Message("assistant", "", tool_calls=[ToolCall("call_a", "get_order_info", {}),
                                                 ToolCall("call_b", "get_order_info", {})]),
            Message("tool", "{}", tool_call_id="call_a"), Message("tool", "{}", tool_call_id="call_b"),
            Message("assistant", "", tool_calls=[ToolCall("call_zz", "get_order_info", {})]),
            Message("tool", "{}", tool_call_id="call_zz")]
    await collect(llm.stream("s", msgs, TOOLS))
    types = [i.get("type", i.get("role")) for i in resp.calls[1]["input"]]
    assert types.count("reasoning") == 1
    assert types[:4] == ["user", "reasoning", "function_call", "function_call"]


def test_to_responses_input_without_store_unchanged():
    from agent.providers.openai_llm import to_responses_input

    items = to_responses_input(_followup_messages())
    assert [i.get("type", i.get("role")) for i in items] == ["user", "function_call", "function_call_output"]


# ── factory: 설정·사본 ──────────────────────────────────────
def test_factory_default_effort_medium_and_env_override():
    from agent.providers.factory import make_llm

    llm = make_llm(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o"), openai_client=NS())
    assert llm.reasoning_effort == "medium" and llm.include_reasoning is True
    llm = make_llm(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o", OPENAI_REASONING_EFFORT="none"),
                   openai_client=NS())
    assert llm.reasoning_effort == "none"
    with pytest.raises(ValueError):
        make_llm(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o", OPENAI_REASONING_EFFORT="ultra"),
                 openai_client=NS())


def test_factory_baze_include_defaults_off_and_switchable():
    from agent.providers.factory import make_llm

    llm = make_llm(_cfg(LLM_PROVIDER="openai", BAZE_API_KEY="b"))
    assert llm.base_url and llm.reasoning_effort == "medium" and llm.include_reasoning is False
    llm = make_llm(_cfg(LLM_PROVIDER="openai", BAZE_API_KEY="b", OPENAI_REASONING_INCLUDE="1"))
    assert llm.include_reasoning is True
    llm = make_llm(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o", OPENAI_REASONING_INCLUDE="0"),
                   openai_client=NS())
    assert llm.include_reasoning is False


@pytest.mark.parametrize("keys", [{"OPENAI_API_KEY": "o"}, {"BAZE_API_KEY": "b", "OPENAI_REASONING_INCLUDE": "1"}])
def test_replica_inherits_effort_include_and_shares_reasoning_store(keys):
    from agent.providers.factory import make_llm_with_fallback

    h = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_REASONING_EFFORT="low", **keys),
                               openai_client=NS())
    a, b = h.llms
    assert a is not b
    assert a.reasoning_effort == b.reasoning_effort == "low"
    assert a.include_reasoning == b.include_reasoning is True
    assert a.reasoning_store is b.reasoning_store


async def test_replica_reasoning_reaches_primary_followup():
    """hedge 사본이 첫 라운드를 이겨도 툴 후속(이름으로 기본에 고정)에서 reasoning item이 실린다."""
    from agent.providers.factory import make_llm_with_fallback

    resp = FakeResponses(_tool_round_events(), [NS(type="response.completed")])
    h = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o"),
                               openai_client=NS(responses=resp))
    primary, replica = h.llms
    await collect(replica.stream("s", [Message("user", "x")], TOOLS))
    await collect(primary.stream("s", _followup_messages(), TOOLS))
    assert [i.get("type") for i in resp.calls[1]["input"]].count("reasoning") == 1


# ── factory: effort별 타이밍 ───────────────────────────────
@pytest.mark.parametrize("effort,hedge_s,first_s", [
    ("none", 0.5, 2.5), ("minimal", 1.0, 3.5), ("low", 1.5, 4.0), ("medium", 3.0, 6.0), ("high", 6.0, 12.0),
])
def test_timing_defaults_scale_with_effort(effort, hedge_s, first_s):
    from agent.providers.factory import make_llm_with_fallback

    h = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o", OPENAI_REASONING_EFFORT=effort),
                               openai_client=NS())
    assert h.hedge_delay_s == pytest.approx(hedge_s)
    assert h.first_event_timeout_s == pytest.approx(first_s)
    f = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o", OPENAI_REASONING_EFFORT=effort,
                                    LLM_MODE="fallback"), openai_client=NS())
    assert f.first_event_timeout_s == pytest.approx(first_s)


def test_default_openai_timing_is_medium():
    from agent.providers.factory import make_llm_with_fallback

    h = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o"), openai_client=NS())
    assert h.hedge_delay_s == pytest.approx(3.0) and h.first_event_timeout_s == pytest.approx(6.0)


def test_timing_env_overrides_win_over_effort_defaults():
    from agent.providers.factory import make_llm_with_fallback

    h = make_llm_with_fallback(_cfg(LLM_PROVIDER="openai", OPENAI_API_KEY="o", OPENAI_REASONING_EFFORT="medium",
                                    LLM_HEDGE_DELAY_MS="700", LLM_FIRST_EVENT_TIMEOUT_S="2.0"),
                               openai_client=NS())
    assert h.hedge_delay_s == pytest.approx(0.7) and h.first_event_timeout_s == pytest.approx(2.0)


def test_gemini_primary_keeps_reasoning_none_timing():
    from agent.providers.factory import make_llm_with_fallback

    h = make_llm_with_fallback(_cfg(GEMINI_API_KEY="g", LLM_PROVIDER="gemini", OPENAI_REASONING_EFFORT="high",
                                    GEMINI_LLM_FALLBACK_MODELS="gemini-3.1-flash-lite"), gemini_client=NS())
    assert h.hedge_delay_s == pytest.approx(0.5) and h.first_event_timeout_s == pytest.approx(2.5)
