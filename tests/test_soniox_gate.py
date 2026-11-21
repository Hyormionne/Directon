"""AG-T30: Soniox <end> 게이트(규칙·Jev) 판정과 이벤트 재생."""
import pytest

from agent.turn.jev_turn import JevTurnResult
from agent.turn.soniox_gate import GatePolicy, GateVerdict, SonioxEndpointGate, combine, rule_verdict, simulate_gate


def tok(text, final=True, start=0):
    return {"text": text, "is_final": final, "start_ms": start}


END = {"text": "<end>", "is_final": True}


def test_rule_holds_connective_and_accepts_final():
    assert not rule_verdict("신어 봤는데").accept
    assert rule_verdict("재고 있어요?").accept


def test_combine_jev_threshold_and_fallback():
    pol = GatePolicy("jev", jev_threshold=0.6)
    assert combine("재고 있어요?", 0.7, 150, pol).accept
    assert not combine("재고 있어요?", 0.4, 150, pol).accept
    # Jev 실패면 규칙으로
    v = combine("신어 봤는데", None, 0, pol)
    assert not v.accept and v.reason.startswith("jev_none")


def test_combine_rule_jev_rule_hold_wins():
    v = combine("신어 봤는데", 0.99, 150, GatePolicy("rule_jev"))
    assert not v.accept and v.reason.startswith("rule_hold")


def _rule(pol):
    return lambda text: combine(text, None, 0.0, pol)


def vad(*spans, until=6.0):
    """(시작초, 끝초) 음성 구간 → 20ms 프레임 bool 목록."""
    n = int(round(until / 0.02))
    return [any(a <= f * 0.02 < b for a, b in spans) for f in range(n)]


def test_simulate_rejects_mid_end_then_accepts_real_end():
    pol = GatePolicy("rule", hold_ms=1000)
    sp = vad((0.2, 1.0), (1.5, 2.2))
    ev = [{"t": 1.4, "toks": [tok("신어 봤는데"), END]},          # 연결어미 → 거절(침묵 400ms, 1.5초에 음성 재개)
          {"t": 2.7, "toks": [tok(" 발이 아파요."), END]}]       # 수락
    g = simulate_gate(ev, _rule(pol), pol, sp, 5.0)
    assert g.decision_t == 2.7 and g.reason.startswith("rule_end")
    assert [x["accept"] for x in g.trace] == [False, True]


def test_simulate_hold_timeout_uses_vad_silence():
    pol = GatePolicy("rule", hold_ms=800)
    sp = vad((0.2, 1.0))
    ev = [{"t": 1.5, "toks": [tok("주문번호가"), END]}]
    g = simulate_gate(ev, _rule(pol), pol, sp, 4.0)
    assert g.reason == "hold_timeout" and g.decision_t == pytest.approx(1.8)


def test_simulate_ignores_end_arriving_while_vad_speech():
    pol = GatePolicy("rule")
    sp = vad((0.2, 1.0), (1.3, 2.4))                  # 1.3초에 다음 말 시작
    ev = [{"t": 1.6, "toks": [tok("배송 조회요."), END]},       # 늦게 온 <end>: 지금은 음성 → 무시
          {"t": 2.9, "toks": [tok(" 주소도 바꿔 주세요."), END]}]
    g = simulate_gate(ev, _rule(pol), pol, sp, 5.0)
    assert g.decision_t == 2.9 and g.trace[0]["reason"].startswith("stale")


def test_simulate_jev_latency_and_cancel_on_vad_speech():
    pol = GatePolicy("jev", jev_threshold=0.5)
    sp = vad((0.2, 1.0), (1.55, 2.4))
    ev = [{"t": 1.5, "toks": [tok("배송 조회 좀 하려고요."), END]},  # Jev 0.2초 → 1.7초 도착, 그 전 1.55초 음성 → 버림
          {"t": 2.9, "toks": [tok(" 그리고 주소도요."), END]}]
    g = simulate_gate(ev, lambda t: GateVerdict(True, "jev_end", p=0.9, ms=200.0), pol, sp, 5.0)
    assert g.decision_t == pytest.approx(3.1)


def test_simulate_cap_without_any_end():
    pol = GatePolicy("rule", cap_ms=1500)
    g = simulate_gate([{"t": 1.2, "toks": [tok("네")]}], _rule(pol), pol, vad((0.2, 1.0)), 4.0)
    assert g.reason == "cap_timeout" and g.decision_t == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_gate_judge_calls_jev_only_when_needed():
    calls = []

    async def jev(text, history, known):
        calls.append(text)
        return JevTurnResult(0.9, 0.1, 0.1, 180.0)

    g = SonioxEndpointGate(GatePolicy("rule_jev"), jev=jev)
    v = await g.judge("신어 봤는데")
    assert not v.accept and calls == []
    v = await g.judge("재고 있어요?")
    assert v.accept and v.p == 0.9 and v.ms == 180.0 and calls == ["재고 있어요?"]
    assert not (await g.judge("  ")).accept


# ── 기존 결정기 Soniox 재생(eval.soniox_gate_eval.simulate_detector) ──
class _Det:
    """텍스트가 '요.'로 끝나고 침묵 500ms면 끝. 받은 텍스트를 기록한다."""

    def __init__(self):
        self.texts = []

    def decide(self, s):
        from agent.core.types import TurnDecision
        self.texts.append((s.silence_ms, s.text))
        return TurnDecision(s.text.endswith("요.") and s.silence_ms >= 500, "")


def test_simulate_detector_uses_finalize_text_after_delay():
    from eval.soniox_gate_eval import simulate_detector
    # 음성 0.2~1.0초. 확정 토큰은 1.6초에야 도착(Soniox 지연). finalize(침묵 200ms=1.2초 요청, +240ms)로 먼저 얻는다
    ev = [{"t": 0.9, "toks": [tok("배송", final=False, start=200)]},
          {"t": 1.6, "toks": [tok("배송 조회요.", start=200), END]}]
    d = _Det()
    t, _ = simulate_detector(ev, vad((0.2, 1.0)), 3.0, d, finalize_ms=240)
    assert t == pytest.approx(1.5)             # 침묵 500ms 시점(1.5초)에 finalize 텍스트로 확정
    assert (200, "배송") in d.texts           # finalize 전에는 미확정 텍스트


def test_simulate_detector_drops_finalize_on_resume_and_ignores_short_noise():
    from eval.soniox_gate_eval import simulate_detector
    ev = [{"t": 1.2, "toks": [tok("신어 봤는데", start=200)]},
          {"t": 2.6, "toks": [tok(" 아파요.", start=1500)]}]
    d = _Det()
    # 0.05초짜리 잡음(60ms 미만)은 발화로 보지 않는다. 진짜 음성 0.2~1.0, 1.5~2.0
    t, _ = simulate_detector(ev, vad((0.0, 0.04), (0.2, 1.0), (1.5, 2.0)), 4.0, d)
    # 첫 쉼("신어 봤는데")에서는 안 끊고, 음성 재개로 finalize를 버린 뒤 2.2초 요청 finalize(+240ms)로 침묵 500ms에 확정
    assert t == pytest.approx(2.5)
    assert all(not x.endswith("요.") for s_, x in d.texts if s_ <= 500 and x.startswith("신어") and "아파" not in x)


# ── 기존 결정기 + Jev 결합(simulate_detector combo) ──
def test_combo_silence_extend_holds_soft_final_until_cap():
    from agent.turn.detector import make_detector
    from agent.turn.jev_turn import JevTurnPolicy
    from eval.soniox_gate_eval import Combo, simulate_detector
    # "주문했는데요"(soft_final) 뒤 0.9초 쉼, 이어서 말한다. 표만 쓰면 쉼에서 끊고, Jev 늘리기(p 0.1)면 상한까지 기다린다
    ev = [{"t": 1.1, "toks": [tok("오메가3 주문했는데요", start=200), END]},
          {"t": 2.9, "toks": [tok(" 아직 안 와서요.", start=1900), END]}]
    sp = vad((0.2, 1.0), (1.9, 2.6))
    base, _ = simulate_detector(ev, sp, 5.0, make_detector())
    assert base is not None and base < 1.9            # 표만: 말 끝 전 확정
    t, why = simulate_detector(ev, sp, 5.0, make_detector(), jev=lambda text: (0.1 if "는데요" in text else 0.9, 150.0),
                               combo=Combo("silence", JevTurnPolicy(hi=1.01, lo=0.3)))
    assert t is not None and t > 2.6                   # 다음 말 뒤에 확정


def test_combo_end_trigger_decides_immediately_on_high_p():
    from agent.turn.detector import make_detector
    from agent.turn.jev_turn import JevTurnPolicy
    from eval.soniox_gate_eval import Combo, simulate_detector
    ev = [{"t": 1.5, "toks": [tok("배송 조회해 주세요.", start=200), END]}]
    t, why = simulate_detector(ev, vad((0.2, 1.0)), 4.0, make_detector(), jev=lambda text: (0.95, 150.0),
                               combo=Combo("end", JevTurnPolicy(hi=1.01, lo=0.3), end_hi=0.9))
    assert t == pytest.approx(1.66) and why.startswith("end_jev")   # <end> 1.5초 + Jev 150ms → 다음 프레임


def test_combo_latest_match_uses_p_even_after_text_changes():
    from agent.turn.detector import make_detector
    from agent.turn.jev_turn import JevTurnPolicy
    from eval.soniox_gate_eval import Combo, simulate_detector
    # 침묵 중 텍스트가 "주문" → "주문했는데요"로 바뀐다(Soniox 늦은 부분 결과). exact면 p를 못 쓰고, latest면 늘린다
    ev = [{"t": 1.0, "toks": [tok("오메가3 주문", final=False, start=200)]},
          {"t": 1.9, "toks": [tok("오메가3 주문했는데요", start=200), END]},
          {"t": 3.3, "toks": [tok(" 아직 안 와서요.", start=2300), END]}]
    sp = vad((0.2, 1.2), (2.3, 3.0))
    fn = lambda text: (0.05, 150.0)
    ex, _ = simulate_detector(ev, sp, 5.0, make_detector(), jev=fn,
                              combo=Combo("silence", JevTurnPolicy(hi=1.01, lo=0.3), max_requests=1))
    la, _ = simulate_detector(ev, sp, 5.0, make_detector(), jev=fn,
                              combo=Combo("silence", JevTurnPolicy(hi=1.01, lo=0.3), max_requests=1, match="latest"))
    assert ex is not None and ex < 2.3 and la is not None and la > 3.0
