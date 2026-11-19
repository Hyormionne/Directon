"""AG-T15b 턴 결정기 2차 보정: 자기 정정·잠깐 멈춤 보류, 숫자 그룹 미완 보류, '에'로 이어 부른 완결 번호 인정,
쉼표 끝 보류, make_detector 계약 유지. 근거: docs/decisions/turn_calibration_r2.md
"""
from __future__ import annotations

import pytest

from agent.core.types import TurnDecision, TurnSignals
from agent.turn import detector as dm
from agent.turn.rules import analyze


def kind(text: str) -> str:
    return dm.table_kind(text, dm.CALIBRATED_CONFIG).kind


def wait(text: str, speech_ms: int = 2000) -> int:
    return dm.make_detector().wait_ms(TurnSignals(silence_ms=0, speech_ms=speech_ms, text=text))[0]


# ── 자기 정정·잠깐 멈춤 ──
@pytest.mark.parametrize("text", [
    "취소요. 아 아니다,",
    "취소요. 아, 아니다.",
    "31 아니",
    "삼일, 아니",
    "주문번호 2611 150, 아니",
    "공일공 1234 5678, 다시 할게요.",
    "01012345678 다시 할게요",
    "배송지 바꿔 주세요. 아니고요.",
    "교환해 주세요. 아니에요.",
    "환불해 줘. 아 아니야",
    "주소가 부산 해운대구, 아 잘못 말했어요.",
    "반품, 아 잘못 말했다.",
    "검은색으로, 아니 그게 아니라",
    "잠시만요, 번호 찾아볼게요.",
    "아 잠깐만요.",
    "제가 한번 찾아볼게요.",
])
def test_self_correction_is_held(text):
    assert dm.self_correction_end(text), text
    assert kind(text) == "self_correction"


@pytest.mark.parametrize("text", [
    "아니요", "아니", "아니에요", "아니요 괜찮아요", "네 아니요",      # 단답은 그대로
    "결제 다시 할게요.",                                             # 정정이 아닌 완결 요청
    "제품에 문제 있는 거 아니에요?",
    "그건 제 주문이 아니에요.",
    "주문 취소해 주세요.",
    "",
])
def test_not_self_correction(text):
    assert not dm.self_correction_end(text), text
    assert kind(text) != "self_correction"


def test_self_correction_waits_long():
    w_corr = wait("취소요. 아 아니다,")
    w_plain = wait("취소요.")
    assert w_corr >= 1200 and w_corr > w_plain
    d = dm.make_detector()
    s = TurnSignals(silence_ms=1300, speech_ms=2000, text="공일공 1234 5678, 다시 할게요.")
    assert not d.decide(s).end                                 # 1.3초 쉼에서도 끊지 않는다
    assert d.cap_ms(s) > dm.CALIBRATED_CONFIG.hard_max_ms      # 상한 예외 묶음


# ── 숫자 그룹 미완 ──
@pytest.mark.parametrize("text,expected", [
    ("주문번호 2611 150", "digits_partial"),
    ("이육일일 일오공", "digits_partial"),
    ("010-1234", "digits_partial"),
    ("010에 1234", "digits_partial"),
    ("공일공에 일이삼사에", "digits_incomplete"),
    ("2611, 150,", "digits_partial"),                          # 쉼표가 있어도 미완 숫자는 숫자 보류
])
def test_digit_group_partial_held(text, expected):
    assert kind(text) == expected
    assert wait(text) >= 1000
    assert wait(text) > wait("주문번호 2611 150 231이요.")


# ── '에'로 이어 부른 완결 번호 ──
@pytest.mark.parametrize("text", [
    "공일공에 1234 5678",
    "010에 1234에 5678",
    "공일공에 일이삼사 오육칠팔",
    "핸드폰 번호 공일공에 일이삼사에 오육칠팔요.",
    "010에 8411 3910이요",
    "주문번호는 이육일일에 일오공에 이삼일입니다.",
])
def test_joined_number_complete(text):
    assert dm.digits_join_complete(text), text
    assert kind(text) == "digits_complete"


@pytest.mark.parametrize("text", ["공일공에 1234", "010에 1234에", "이육일일에 일오공"])
def test_joined_number_still_partial(text):
    assert not dm.digits_join_complete(text)
    assert kind(text) in ("digits_partial", "digits_incomplete")


# ── 쉼표 끝 ──
@pytest.mark.parametrize("text", ["러닝화가 좀 작아서요,", "배송이 너무 늦어요,", "엠이요,", "이육일일 일삼공 일사칠,",
                                  "2611130147,"])
def test_comma_end_held(text):
    assert kind(text) == "comma"
    assert wait(text) > wait(text.rstrip(",") + ".")


def test_comma_does_not_override_question_or_holds():
    assert kind("배송 언제 와요?,") == "question"
    assert kind("주문을 했는데,") == "connective"
    assert kind("주문번호가,") == "particle"


# ── 문장 끝 + 짧은 발화 길이 ──
def test_final_text_short_speech_waits_longer_than_long_speech():
    # 짧은 발화의 완결 텍스트는 "엠이요. [쉼] 아 근데…"처럼 이어질 때가 많다(round1 T0152)
    assert wait("엠이요.", speech_ms=500) >= wait("엠이요.", speech_ms=2600)


# ── make_detector 계약 ──
def test_make_detector_contract():
    d1, d2 = dm.make_detector(), dm.make_detector()
    assert d1 is not d2 and d1.config is dm.CALIBRATED_CONFIG
    c = dm.CALIBRATED_CONFIG
    assert c.self_corr_hold and c.comma_hold and c.digit_join_fix and c.digit_partial_hold
    assert c.hard_max_ms <= 1400 and c.exempt_max_ms <= 2000
    for text in ("네", "언제 와요?", "", "주문번호가", "취소요. 아 아니다,", "엠이요,"):
        for p in (None, 0.0, 0.9):
            s = TurnSignals(silence_ms=180, speech_ms=1000, text=text, smart_turn_p=p)
            assert not d1.decide(s).end
            w, why = d1.wait_ms(TurnSignals(silence_ms=0, speech_ms=1000, text=text, smart_turn_p=p))
            assert c.min_silence_ms <= w <= max(c.hard_max_ms, c.exempt_max_ms) and isinstance(why, str)
            assert d1.cap_ms(s) in (c.hard_max_ms, c.exempt_max_ms)
    assert isinstance(d1.decide(TurnSignals(silence_ms=500, speech_ms=1000, text="네")), TurnDecision)
    assert d1.decide(TurnSignals(silence_ms=c.hard_max_ms, speech_ms=1000, text="언제 와요?", smart_turn_p=0.0)).end
    assert not d1.decide(TurnSignals(silence_ms=3000, speech_ms=100, text="")).end      # 잡음성
    # 빈 텍스트 짧은 발화도 2초 안에는 끝 판정(세션 빈 텍스트 경로)
    assert d1.decide(TurnSignals(silence_ms=2000, speech_ms=800, text="")).end


def test_previous_calibration_kept_for_comparison():
    old = dm.CALIBRATED_CONFIG_T12
    assert not old.self_corr_hold and not old.comma_hold and not old.digit_join_fix
    assert old.hard_max_ms == 1100
    # 예전 설정에서는 정정 표현이 완결로 보인다(회귀 비교용)
    assert dm.table_kind("배송지 바꿔 주세요. 아니고요.", old).kind == analyze("배송지 바꿔 주세요. 아니고요.").kind


def test_new_flags_off_by_default_keep_table_kind():
    base = dm.TableTurnConfig()
    for text in ("취소요. 아 아니다,", "공일공에 1234 5678", "엠이요,"):
        assert dm.table_kind(text, base).kind != "self_correction"
        assert dm.table_kind(text, base).kind != "comma"
