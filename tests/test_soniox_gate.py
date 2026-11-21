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
