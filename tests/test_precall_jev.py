"""AG-T31: 말하는 중 Jev 선호출 재생·정책·정합성 판정(네트워크 없음)."""
import pytest

from agent.core.types import TurnDecision
from eval.precall_jev_eval import (JevStore, Precall, Replay, SpeechFire, SpeechPolicy, equivalent, existing_outcome,
                                   replay, same_body, speech_fire, speech_outcome, tick_body)


def tok(text, final=True, start=0):
    return {"text": text, "is_final": final, "start_ms": start}


def vad(*spans, until=6.0):
    n = int(round(until / 0.02))
    return [any(a <= f * 0.02 < b for a, b in spans) for f in range(n)]


class _Det:
    def decide(self, s):
        return TurnDecision(s.text.endswith("요.") and s.silence_ms >= 600, "")


def test_replay_ticks_only_while_speaking_and_records_silence_precall():
    ev = [{"t": 0.9, "toks": [tok("배송", final=False, start=200)]},
          {"t": 1.6, "toks": [tok("배송 조회요.", start=200)]}]
    rp = replay(ev, vad((0.2, 1.0)), 3.0, _Det(), tick_ms=300)
    assert all(t <= 1.2 for t, _ in rp.ticks)                    # 침묵 200ms 이후로는 tick 없음
    assert [x for _, x in rp.ticks][-1] == "배송"
    assert rp.commit_t == pytest.approx(1.6)                     # 침묵 600ms
    assert rp.survivor.src == "finalize" and rp.survivor.t == pytest.approx(1.44)
    assert rp.silence_fires == 2 and rp.final_t == pytest.approx(1.44)


def test_replay_resume_cancels_silence_precall():
    ev = [{"t": 1.1, "toks": [tok("신어 봤는데", start=200)]},
          {"t": 2.4, "toks": [tok(" 아파요.", start=1500)]}]
    rp = replay(ev, vad((0.2, 1.0), (1.5, 2.0)), 4.0, _Det())
    assert rp.survivor.text == "신어 봤는데 아파요."
    assert rp.survivor.t > 2.0


def _pc(table):
    return lambda text: table[text]


def test_speech_fire_first_and_latest():
    ticks = [(0.4, "배송"), (0.8, "배송 조회"), (1.2, "배송 조회 좀"), (1.6, "배송 조회 좀요")]
    p = _pc({"배송": (0.9, 100), "배송 조회": (0.2, 100), "배송 조회 좀": (0.1, 100), "배송 조회 좀요": (0.1, 900)})
    f = speech_fire(ticks, 3.0, p, SpeechPolicy(tau=0.3, mode="first", min_chars=2))
    assert f.fire.text == "배송 조회" and f.fire.t == pytest.approx(0.9) and f.jev_calls == 2 and f.fires == 1
    g = speech_fire(ticks, 2.4, p, SpeechPolicy(tau=0.3, mode="latest", min_chars=2))
    assert g.fire.text == "배송 조회 좀" and g.fires == 2 and g.jev_calls == 4   # 마지막은 도착이 확정 뒤


def _rp(commit_t=3.0, text="배송 조회 좀 해 주세요.", survivor_t=2.6):
    return Replay(commit_t, text, "", Precall(survivor_t, text, "finalize"), 2, 2.6, [])


def test_existing_outcome_head_and_wait():
    o = existing_outcome(_rp(), 1000.0)
    assert o.used == "silence" and o.head_ms == pytest.approx(400) and o.llm_wait_ms == pytest.approx(600)


def test_speech_outcome_accept_reject_and_check_wait():
    rp = _rp()
    sf = SpeechFire(Precall(1.0, "배송 조회", "speech"), 3, 1)
    acc = speech_outcome(rp, sf, 1000.0, SpeechPolicy(sigma=0.5), lambda e, f: (0.9, 500.0), lambda e, f: True)
    # 판정 요청 2.6초 + 0.5초 = 3.1초 → 확정 뒤 100ms 대기, 선행 2000ms라 LLM은 끝남
    assert acc.used == "speech" and acc.check_wait_ms == pytest.approx(100) and acc.llm_wait_ms == pytest.approx(100)
    assert not acc.wrong and acc.jev_calls == 4 and acc.llm_calls == 3
    rej = speech_outcome(rp, sf, 1000.0, SpeechPolicy(sigma=0.5), lambda e, f: (0.2, 100.0), lambda e, f: True)
    assert rej.used == "silence" and rej.missed and rej.llm_wait_ms == pytest.approx(600)
    bad = speech_outcome(rp, sf, 1000.0, SpeechPolicy(check="none"), lambda e, f: (None, 0), lambda e, f: False)
    assert bad.used == "speech" and bad.wrong


def test_equivalent_tool_and_text():
    a = {"kind": "tool", "calls": [{"name": "track_delivery", "arguments": {"order_id": "1"}}]}
    b = {"kind": "tool", "calls": [{"name": "track_delivery", "arguments": {"order_id": "2"}}]}
    assert equivalent(a, a, None) is True and equivalent(a, b, None) is False
    t = {"kind": "text", "reply": "네"}
    assert equivalent(a, t, True) is False and equivalent(t, t, True) is True and equivalent(t, t, None) is None


def test_jev_store_fills_with_injected_ask(tmp_path):
    seen = []

    def ask(body):
        seen.append(body)
        q = next(iter(body["questions"]))
        return {"answers": {q: {"noul": 0.25}}}, 123.0

    st = JevStore(tmp_path / "c.jsonl", ask)
    b1, b2 = tick_body("배송 조회", [], True), same_body("배송", "배송 조회요", [], True)
    st.fill([b1, b2, b1])
    assert st.calls == 2 and st.get(b1, "meaning_will_change") == (0.25, 123.0)
    again = JevStore(tmp_path / "c.jsonl", None)
    assert again.get(b2, "same_reply") == (0.25, 123.0)
