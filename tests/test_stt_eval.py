"""AG-T10b 오프라인 STT 평가기 테스트(네트워크 금지: fake STT 주입)."""
from __future__ import annotations

import asyncio

import pytest

from eval import stt_eval as E


def case(cid="T1", utt="", tools=None, tags=("digits",), channel="phone", noise="none"):
    return {"case_id": cid, "customer_utterance": utt, "tags": list(tags),
            "expected_tools": tools or [], "env": {"channel": channel, "noise": noise}}


# ── CER ─────────────────────────────────────────────
def test_cer_basic():
    assert E.cer("배송 언제 와요?", "배송 언제 와요") == 0.0          # 공백·문장부호 무시
    assert E.cer("가나다라", "가나다") == pytest.approx(0.25)
    assert E.cer("가나다라", "가나마라") == pytest.approx(0.25)
    assert E.cer("가나", "가나다라") == pytest.approx(1.0)              # 삽입 2 / 길이 2
    assert E.cer("", "") == 0.0
    assert E.cer("", "가") == 1.0


def test_cer_digit_normalization():
    ref = "주문번호는 이육일일 일오공 이삼일이요."
    hyp = "주문번호는 2611150231이요."
    assert E.cer(ref, hyp) > 0.3                          # 낭독 vs 아라비아 숫자: 원문 비교는 크게 틀림
    assert E.cer(ref, hyp, norm_digits=True) == 0.0
    assert E.cer("공일공 일이삼사 오육칠팔", "010-1234-5678", norm_digits=True) == 0.0
    # 숫자 하나 빠지면 오류로 남는다
    assert E.cer(ref, "주문번호는 261115023이요.", norm_digits=True) > 0


# ── 숫자 정확도 ─────────────────────────────────────
def test_digit_targets_from_expected_tools_key_args():
    c = case(utt="이육일일 일오공 이삼일 취소요", tools=[
        {"name": "get_order_list", "arguments": {"phone_number": "01012345678"}},
        {"name": "cancel_order", "arguments": {"order_id": "2611150231", "reason": "변심"}}])
    assert E.digit_targets(c) == ["01012345678", "2611150231"]


def test_digit_correct_order_id():
    c = case(utt="주문번호 이육일일 일오공 이삼일 배송 언제 와요?",
             tools=[{"name": "track_delivery", "arguments": {"order_id": "2611150231"}}])
    assert E.digit_correct(c, "주문번호 2611150231 배송 언제 와요?") is True
    assert E.digit_correct(c, "주문번호 2611 150 231 배송 언제 와요?") is True
    assert E.digit_correct(c, "주문번호 이육일일 일오공 이삼일 배송") is True   # 한글 낭독도 정규화
    assert E.digit_correct(c, "주문번호 261023 배송 언제 와요?") is False
    assert E.digit_correct(c, "주문번호 26111502310 배송") is False          # 자릿수 초과
    assert E.digit_correct(c, "배송 언제 와요?") is False


def test_digit_correct_phone_and_self_correction():
    c = case(utt="공일공 사삼이일, 아 죄송해요 다시 할게요, 공일공 사삼이일 공구팔칠이요.",
             tools=[{"name": "get_member_info", "arguments": {"phone_number": "01043210987"}}])
    assert E.digit_correct(c, "010 4321 아 죄송해요 다시 할게요 010-4321-0987이요") is True
    assert E.digit_correct(c, "010 4321 아 죄송해요 다시 할게요 010 4321 087이요") is False


def test_digit_correct_fallback_when_target_not_contiguous_or_absent():
    # 정정으로 정답이 이어 읽히지 않음 → 정답 텍스트의 숫자열 전체를 재현했는지로 판정
    c = case(utt="주문번호 이육일일 일팔공, 아 잠깐만요, 일육공 사오칠이요.",
             tools=[{"name": "track_delivery", "arguments": {"order_id": "2611160457"}}])
    assert E.digit_correct(c, "주문번호 2611180 아 잠깐만요 160457이요.") is True
    assert E.digit_correct(c, "주문번호 2611180 아 잠깐만요 16457이요.") is False
    # 기대 함수 없음(번호를 덜 말함) → 읽은 숫자 그대로
    c2 = case(utt="주문번호는 이육일일 일오공 이삼, 이렇게요.")
    assert E.digit_correct(c2, "주문번호는 2611150 23 이렇게요") is True
    assert E.digit_correct(c2, "주문번호는 261023 이렇게요") is False


def test_digit_correct_none_for_non_digit_cases():
    c = case(utt="배송 언제 와요?", tags=("short",))
    assert E.digit_correct(c, "배송 언제 와요?") is None


# ── 집계 ────────────────────────────────────────────
def test_summarize_groups_and_metrics():
    rows = [
        {"case_id": "a", "channel": "phone", "noise": "none", "cer": 0.0, "cer_norm": 0.0,
         "digit_ok": True, "finalize_ms": 100, "text_at_end": "x", "error": None},
        {"case_id": "b", "channel": "phone", "noise": "pink", "cer": 0.5, "cer_norm": 0.2,
         "digit_ok": False, "finalize_ms": 300, "text_at_end": "", "error": None},
        {"case_id": "c", "channel": "wideband", "noise": "none", "cer": 0.1, "cer_norm": 0.1,
         "digit_ok": None, "finalize_ms": 200, "text_at_end": "", "error": None},
        {"case_id": "d", "channel": "wideband", "noise": "none", "error": "boom"},
    ]
    s = E.summarize(rows)
    o = s["overall"]
    assert o["n"] == 3 and o["errors"] == 1
    assert o["cer"] == pytest.approx(0.2)
    assert o["cer_norm"] == pytest.approx(0.1)
    assert o["digit_n"] == 2 and o["digit_acc"] == pytest.approx(0.5)
    assert o["empty_at_end"] == pytest.approx(2 / 3)
    assert o["fin_p50"] == pytest.approx(200)
    assert 200 < o["fin_p95"] <= 300
    assert s["by_channel"]["phone"]["n"] == 2
    assert s["by_noise"]["none"]["n"] == 2
    assert s["by_channel"]["wideband"]["digit_acc"] is None
    md = E.render_markdown(s, title="t")
    assert "phone" in md and "숫자" in md


# ── 한 문항 재생(fake STT, 가속) ─────────────────────
class FakeStream:
    def __init__(self):
        self.sent = 0
        self.sent_at_finalize = None
        self.closed = False

    async def send_audio(self, pcm):
        self.sent += len(pcm)

    def latest_text(self):
        return "부분" if self.sent >= 32000 else ""   # 1초 뒤부터 부분 결과

    async def finalize(self, timeout=1.0):
        self.sent_at_finalize = self.sent
        await asyncio.sleep(0.01)
        return "최종 텍스트"

    async def close(self):
        self.closed = True


class FakeSTT:
    def __init__(self):
        self.streams = []
        self.contexts = []

    async def open_stream(self, context=""):
        self.contexts.append(context)
        s = FakeStream()
        self.streams.append(s)
        return s


async def test_run_case_finalizes_200ms_after_speech_end():
    stt = FakeSTT()
    pcm = b"\x00\x00" * 16000 * 3                    # 3초
    r = await E.run_case(stt, pcm, speech_end=1.5, speed=0, prep_ms=200, context="ctx")
    s = stt.streams[0]
    assert s.sent_at_finalize == int((1.5 + 0.2) * 16000) * 2
    assert r["final"] == "최종 텍스트"
    assert r["text_at_end"] == "부분"                  # 1.5초 시점
    assert r["text_at_prep"] == "부분"
    assert r["finalize_ms"] >= 5
    assert s.closed and stt.contexts == ["ctx"]


async def test_run_case_empty_partial_recorded():
    stt = FakeSTT()
    pcm = b"\x00\x00" * 16000 * 2
    r = await E.run_case(stt, pcm, speech_end=0.5, speed=0, prep_ms=200)
    assert r["text_at_end"] == "" and r["final"] == "최종 텍스트"


async def test_evaluate_rows_with_retry_and_concurrency():
    calls = {"n": 0}

    class Flaky(FakeSTT):
        async def open_stream(self, context=""):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("503")
            return await super().open_stream(context)

    cases = [case(cid="A", utt="주문번호 이육일일 일오공 이삼일",
                  tools=[{"name": "track_delivery", "arguments": {"order_id": "2611150231"}}]),
             case(cid="B", utt="최종 텍스트", tags=("short",), channel="wideband")]
    audio = {"A": (b"\x00\x00" * 16000, 0.5), "B": (b"\x00\x00" * 16000, 0.5)}
    rows = await E.evaluate(Flaky(), cases, lambda cid: audio[cid], speed=0, concurrency=2,
                            retries=2, retry_wait=0.0)
    by = {r["case_id"]: r for r in rows}
    assert by["A"]["error"] is None and by["A"]["digit_ok"] is False
    assert by["B"]["cer"] == 0.0 and by["B"]["digit_ok"] is None
    assert by["A"]["channel"] == "phone" and by["B"]["channel"] == "wideband"
