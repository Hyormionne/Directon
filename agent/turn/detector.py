"""말 끝 결정기. 계약: agent.core.types.TurnDetector

목표: 잘못 끊기 최소 + 무음 1.1초 상한(말 끝 후 1.5초 안에 첫 소리).
순수 함수 decide(signals, config). 상태 없음. 세션이 20ms마다 불러도 된다.

판정 순서
 1. 잡음성 발화(speech_ms < min_speech_ms 이고 STT 텍스트 없음) → 끝 아님(상한도 적용 안 함)
 2. silence < min_silence_ms(200) → 끝 아님
 3. silence ≥ hard_max_ms(1100) → 끝
 4. 대기 시간 wait를 고르고 silence ≥ wait면 끝
    - 텍스트가 연결어미/간투사/숫자 미완                         → hold_ms 1100
    - p 있음: p ≥ 0.7 and end_score ≥ 0                       → fast_ms 400
              단답(네/아니요…) and p ≥ 0.3                     → fast_ms 400
              p < 0.3 or end_score < 0                        → slow_ms 900
              그 밖(애매)                                      → mid_ms 600
    - p 없음(텍스트만): 단답·완결 숫자열 → 400, end_score < 0 → 900, 그 밖 → 600
    - expect_short면 wait −100ms(최소 300ms)
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.core.types import TurnDecision, TurnSignals
from agent.turn.rules import STRONG_HOLD, analyze

TEXT_ONLY_FAST_KINDS = frozenset({"short_answer", "digits_complete"})


@dataclass(frozen=True)
class TurnConfig:
    min_silence_ms: int = 200
    hard_max_ms: int = 1100
    fast_ms: int = 400
    mid_ms: int = 600
    slow_ms: int = 900
    hold_ms: int = 1100
    short_step_ms: int = 100       # expect_short일 때 줄이는 양
    short_floor_ms: int = 300      # expect_short 최솟값
    min_speech_ms: int = 250       # 이보다 짧고 텍스트가 없으면 잡음으로 본다
    p_high: float = 0.7
    p_low: float = 0.3


DEFAULT_CONFIG = TurnConfig()


def _fmt_p(p: float | None) -> str:
    return "none" if p is None else f"{p:.2f}"


def wait_ms(s: TurnSignals, c: TurnConfig = DEFAULT_CONFIG) -> tuple[int, str]:
    """이 신호에서 말 끝으로 보려면 필요한 무음 길이(ms)와 근거."""
    r = analyze(s.text)
    p = s.smart_turn_p
    if r.kind in STRONG_HOLD:
        w, why = c.hold_ms, "hold"
    elif p is None:
        if r.kind in TEXT_ONLY_FAST_KINDS:
            w, why = c.fast_ms, "text_fast"
        elif r.score < 0:
            w, why = c.slow_ms, "text_slow"
        else:
            w, why = c.mid_ms, "text_mid"
    elif p >= c.p_high and r.score >= 0:
        w, why = c.fast_ms, "fast"
    elif r.kind == "short_answer" and p >= c.p_low:
        w, why = c.fast_ms, "fast_short_answer"
    elif p < c.p_low or r.score < 0:
        w, why = c.slow_ms, "slow"
    else:
        w, why = c.mid_ms, "mid"
    if s.expect_short:
        w = max(c.short_floor_ms, w - c.short_step_ms)
        why += "+short"
    return w, f"{why}(wait={w} p={_fmt_p(p)} text={r.kind}:{r.score:+.1f})"


def decide(s: TurnSignals, c: TurnConfig = DEFAULT_CONFIG) -> TurnDecision:
    if s.speech_ms < c.min_speech_ms and not (s.text or "").strip():
        return TurnDecision(False, f"noise_short_speech(speech={s.speech_ms}ms)")
    if s.silence_ms < c.min_silence_ms:
        return TurnDecision(False, f"silence<{c.min_silence_ms}")
    if s.silence_ms >= c.hard_max_ms:
        return TurnDecision(True, f"hard_max({c.hard_max_ms}) silence={s.silence_ms}")
    w, why = wait_ms(s, c)
    if s.silence_ms >= w:
        return TurnDecision(True, f"end {why} silence={s.silence_ms}")
    return TurnDecision(False, f"wait {why} silence={s.silence_ms}")


class RuleTurnDetector:
    """TurnDetector 구현. 설정만 들고 있는 얇은 래퍼."""

    def __init__(self, config: TurnConfig = DEFAULT_CONFIG):
        self.config = config

    def decide(self, s: TurnSignals) -> TurnDecision:
        return decide(s, self.config)

    def wait_ms(self, s: TurnSignals) -> tuple[int, str]:
        return wait_ms(s, self.config)
