import time

from agent.providers.latency_sim import DelayedLLM, DelayedTTS, wrap_llm, wrap_tts


class FakeLLM:
    model = "gpt-5.4-mini"

    async def stream(self, system, messages, tools=None):
        yield "a"
        yield "b"


class FakeTTS:
    async def synthesize(self, text):
        yield b"\x01\x00"
        yield b"\x02\x00"


def test_no_env_no_wrap():
    llm, tts = FakeLLM(), FakeTTS()
    assert wrap_llm(llm, {}) is llm
    assert wrap_tts(tts, {"DIRECTON_SIM_TTS_DELAY_S": "0"}) is tts


async def test_llm_delay_before_first_event_only():
    w = wrap_llm(FakeLLM(), {"DIRECTON_SIM_LLM_DELAY_S": "0.05"})
    assert isinstance(w, DelayedLLM) and w.model == "gpt-5.4-mini"
    t0 = time.perf_counter()
    out = [e async for e in w.stream("s", [])]
    assert out == ["a", "b"]
    assert time.perf_counter() - t0 >= 0.045


async def test_tts_delay():
    w = wrap_tts(FakeTTS(), {"DIRECTON_SIM_TTS_DELAY_S": "0.05"})
    assert isinstance(w, DelayedTTS)
    t0 = time.perf_counter()
    out = [c async for c in w.synthesize("x")]
    assert len(out) == 2 and time.perf_counter() - t0 >= 0.045
