"""AG-T10b 숫자 재인식 래퍼(DigitRecheckSTT) 테스트(네트워크 금지)."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace as NS

import pytest

from agent.providers import stt_recheck as R


# ── 자릿수 판정 ──────────────────────────────────────
def test_digit_runs():
    assert R.digit_runs("주문번호 2611150231 배송") == ["2611150231"]
    assert R.digit_runs("주문번호 2611 150 231이요") == ["2611150231"]
    assert R.digit_runs("010-1234-5678로 해 주세요") == ["01012345678"]
    assert R.digit_runs("공일공 일이삼사 오육칠팔") == ["01012345678"]
    assert R.digit_runs("배송 언제 와요?") == []


@pytest.mark.parametrize("text,expected", [
    ("주문번호 261023 배송 언제 와요?", True),        # 6자리
    ("주문번호 261200517 취소해 주세요", True),       # 9자리
    ("12345678901로 조회", True),                     # 11자리인데 01로 시작하지 않음
    ("주문번호 2611150231 배송 언제 와요?", False),
    ("제 번호는 01043210987이에요", False),
    ("2611 150 231", False),
    ("배송 언제 와요?", False),
    ("3개 주문했어요 11월 21일에", False),              # 짧은 수·단위 붙은 수는 무시
    ("", False),
])
def test_needs_recheck(text, expected):
    assert R.needs_recheck(text) is expected


# ── 래퍼 ─────────────────────────────────────────────
class FakeStream:
    def __init__(self, final):
        self.final = final
        self.sent = bytearray()
        self.closed = False
        self.finalize_calls = []

    async def send_audio(self, pcm):
        self.sent += pcm

    def latest_text(self):
        return "부분 " + self.final

    async def finalize(self, timeout=1.0):
        self.finalize_calls.append(timeout)
        return self.final

    async def close(self):
        self.closed = True


class FakeSTT:
    def __init__(self, final):
        self.final = final
        self.contexts = []

    async def open_stream(self, context=""):
        self.contexts.append(context)
        self.stream = FakeStream(self.final)
        return self.stream


class Rechecker:
    def __init__(self, result="주문번호 2611200517 취소해 주세요", delay=0.0, exc=None):
        self.result, self.delay, self.exc = result, delay, exc
        self.calls = []

    async def __call__(self, pcm, context=""):
        self.calls.append((bytes(pcm), context))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.result


async def _run(final, rechecker, timeout=0.5, chunks=(b"\x01\x00" * 320, b"\x02\x00" * 320)):
    inner = FakeSTT(final)
    stt = R.DigitRecheckSTT(inner, rechecker, timeout=timeout)
    s = await stt.open_stream("ctx")
    for c in chunks:
        await s.send_audio(c)
    t0 = time.perf_counter()
    text = await s.finalize(1.0)
    return s, inner, text, time.perf_counter() - t0


async def test_mismatch_triggers_recheck_and_replaces():
    rc = Rechecker()
    s, inner, text, _ = await _run("주문번호 261200517 취소해 주세요", rc)
    assert text == "주문번호 2611200517 취소해 주세요"
    assert s.latest_text() == "주문번호 2611200517 취소해 주세요"       # 교체 결과가 남는다
    assert len(rc.calls) == 1
    pcm, ctx = rc.calls[0]
    assert pcm == b"\x01\x00" * 320 + b"\x02\x00" * 320                 # 발화 오디오 전체
    assert ctx == "ctx"
    assert inner.contexts == ["ctx"] and bytes(inner.stream.sent) == pcm
    assert s.last_recheck["triggered"] and s.last_recheck["replaced"]


async def test_match_keeps_original_without_recheck():
    rc = Rechecker()
    s, _, text, _ = await _run("주문번호 2611150231 배송 언제 와요?", rc)
    assert text == "주문번호 2611150231 배송 언제 와요?"
    assert rc.calls == []
    assert not s.last_recheck["triggered"]


async def test_no_digits_no_recheck():
    rc = Rechecker()
    _, _, text, _ = await _run("배송 언제 와요?", rc)
    assert text == "배송 언제 와요?" and rc.calls == []


async def test_recheck_failure_keeps_original():
    rc = Rechecker(exc=RuntimeError("503"))
    s, _, text, _ = await _run("주문번호 261023 배송", rc)
    assert text == "주문번호 261023 배송"
    assert s.last_recheck["triggered"] and not s.last_recheck["replaced"]
    assert s.last_recheck["error"]
    assert s.latest_text() == "부분 주문번호 261023 배송"               # 내부 스트림 그대로


async def test_recheck_timeout_keeps_original_quickly():
    rc = Rechecker(delay=2.0)
    s, _, text, dt = await _run("주문번호 261023 배송", rc, timeout=0.05)
    assert text == "주문번호 261023 배송"
    assert dt < 0.5
    assert not s.last_recheck["replaced"]


async def test_recheck_result_still_wrong_keeps_original():
    rc = Rechecker(result="주문번호 26102 배송")
    _, _, text, _ = await _run("주문번호 261023 배송", rc)
    assert text == "주문번호 261023 배송"
    rc2 = Rechecker(result="")
    _, _, text2, _ = await _run("주문번호 261023 배송", rc2)
    assert text2 == "주문번호 261023 배송"


async def test_second_turn_audio_and_close_passthrough():
    rc = Rechecker()
    inner = FakeSTT("주문번호 261200517 취소")
    s = await R.DigitRecheckSTT(inner, rc, timeout=0.5).open_stream()
    await s.send_audio(b"\x01\x00" * 10)
    await s.finalize(1.0)
    await s.send_audio(b"\x03\x00" * 10)             # 확정 뒤 계속 들어온 오디오도 누적(같은 발화 스트림)
    await s.finalize(1.0)
    assert rc.calls[-1][0] == b"\x01\x00" * 10 + b"\x03\x00" * 10
    await s.close()
    assert inner.stream.closed


async def test_wrapper_does_not_wrap_batch_streams():
    from agent.providers.gemini_stt import GeminiBatchSTT, GeminiBatchStream

    stt = R.DigitRecheckSTT(GeminiBatchSTT(client=NS()), Rechecker())
    s = await stt.open_stream()
    assert isinstance(s, GeminiBatchStream)


# ── factory ─────────────────────────────────────────
def test_factory_recheck_switch():
    from agent.providers.factory import ProviderConfig, make_stt
    from agent.providers.gemini_stt import GeminiLiveSTT

    on = make_stt(ProviderConfig.from_env({"GEMINI_API_KEY": "g", "STT_DIGIT_RECHECK": "1",
                                           "STT_RECHECK_MODEL": "m-x", "STT_RECHECK_TIMEOUT_S": "1.7"}),
                  gemini_client=NS())
    assert isinstance(on, R.DigitRecheckSTT)
    assert isinstance(on.primary, GeminiLiveSTT)        # 기존 테스트의 getattr(stt, "primary") 경로 유지
    assert on.timeout == pytest.approx(1.7)
    assert on.recheck_model == "m-x"
    off = make_stt(ProviderConfig.from_env({"GEMINI_API_KEY": "g", "STT_DIGIT_RECHECK": "0"}),
                   gemini_client=NS())
    assert getattr(off, "recheck", None) is None          # batch 재인식 끔(부분 결과 복원은 별도 스위치)
    assert on.recheck is not None
    # batch만 쓰는 구성은 감싸지 않는다
    b = make_stt(ProviderConfig.from_env({"GEMINI_API_KEY": "g", "STT_PROVIDER": "gemini_batch",
                                          "STT_DIGIT_RECHECK": "1"}), gemini_client=NS())
    assert not isinstance(b, R.DigitRecheckSTT)


# ── 부분 결과(interim) 숫자 복원: 추가 지연 없음 ─────────
class InterimStream(FakeStream):
    """finalize 직전 부분 결과에는 10자리가 있었는데 확정에서 자릿수가 틀어진 경우."""

    def __init__(self, interim, final):
        super().__init__(final)
        self.interim = interim

    def latest_text(self):
        return self.interim


class InterimSTT(FakeSTT):
    def __init__(self, interim, final):
        super().__init__(final)
        self.interim = interim

    async def open_stream(self, context=""):
        self.stream = InterimStream(self.interim, self.final)
        return self.stream


async def test_interim_digits_restore_without_batch_call():
    rc = Rechecker()
    inner = InterimSTT("2611 180042 취소해", "26111800042 취소해 주세요.")
    s = await R.DigitRecheckSTT(inner, rc, timeout=0.5).open_stream()
    await s.send_audio(b"\x01\x00" * 10)
    text = await s.finalize(1.0)
    assert text == "2611180042 취소해 주세요."            # 숫자열만 부분 결과의 것으로 교체, 꼬리는 확정 결과
    assert rc.calls == []                                   # batch 재인식 안 함
    assert s.last_recheck["source"] == "interim"


async def test_interim_restore_only_when_final_invalid_and_single_run():
    rc = Rechecker(result="")
    # 확정이 이미 맞으면 손대지 않는다
    inner = InterimSTT("2611 180042 취소해", "2611180043 취소해 주세요.")
    s = await R.DigitRecheckSTT(inner, rc, timeout=0.5).open_stream()
    await s.send_audio(b"\x01\x00" * 10)
    assert await s.finalize(1.0) == "2611180043 취소해 주세요."
    # 부분 결과에도 맞는 숫자열이 없으면 그대로(→ batch 재인식 경로)
    inner = InterimSTT("2611 18004 취소해", "26111800042 취소해 주세요.")
    s = await R.DigitRecheckSTT(inner, rc, timeout=0.5).open_stream()
    await s.send_audio(b"\x01\x00" * 10)
    assert await s.finalize(1.0) == "26111800042 취소해 주세요."
    assert len(rc.calls) == 1


async def test_batch_recheck_can_be_disabled():
    inner = FakeSTT("주문번호 261023 배송")
    s = await R.DigitRecheckSTT(inner, None, timeout=0.5).open_stream()
    await s.send_audio(b"\x01\x00" * 10)
    assert await s.finalize(1.0) == "주문번호 261023 배송"
    assert s.last_recheck["triggered"] and not s.last_recheck["replaced"]


def test_factory_default_wraps_with_interim_fix_only():
    from agent.providers.factory import ProviderConfig, make_stt

    stt = make_stt(ProviderConfig.from_env({"GEMINI_API_KEY": "g"}), gemini_client=NS())
    assert isinstance(stt, R.DigitRecheckSTT) and stt.recheck is None
    off = make_stt(ProviderConfig.from_env({"GEMINI_API_KEY": "g", "STT_DIGIT_INTERIM_FIX": "0"}),
                   gemini_client=NS())
    assert not isinstance(off, R.DigitRecheckSTT)
