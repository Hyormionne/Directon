"""AG-T12 오프라인 턴 시뮬레이터(eval/turn_sim.py)와 보정 결정기(make_detector) 테스트.

가짜 VAD 확률(프레임별 0/1)과 가짜 Smart Turn으로 세션과 같은 판정·지표를 확인한다. 네트워크 없음.
"""
from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from agent.core.types import TurnDecision, TurnSignals
from agent.turn.detector import DEFAULT_CONFIG, RuleTurnDetector, TurnConfig, decide
from eval import turn_sim as ts

F = 0.02


def make_item(pattern, texts, smart_p=0.9, uid="u", meta=None, cut_types=None):
    """pattern: [(kind, ms), ...] kind=speech|sil. 첫 sil은 lead, 마지막 sil은 tail.
    texts: 각 쉼 지점·최종 끝의 STT 텍스트(순서대로)."""
    probs = []
    points = []
    t = 0.0
    speech_seen = 0
    n_speech = sum(1 for k, _ in pattern if k == "speech")
    ti = 0
    for i, (k, ms) in enumerate(pattern):
        n = ms // 20
        if k == "speech":
            probs += [1.0] * n
            t += ms / 1000
            speech_seen += 1
            if speech_seen < n_speech:
                dur = pattern[i + 1][1]
                ct = (cut_types or ["connective"] * 10)[ti]
                points.append(ts.SimPoint(t=round(t, 3), kind="pause", cut_type=ct, dur_ms=dur, text=texts[ti]))
            else:
                points.append(ts.SimPoint(t=round(t, 3), kind="end", text=texts[ti]))
            ti += 1
        else:
            probs += [0.0] * n
            t += ms / 1000
    smart = smart_p if callable(smart_p) else (lambda i, p=smart_p: p)
    return ts.SimItem(uid=uid, probs=np.array(probs, dtype=np.float32), points=points,
                      smart_fn=smart, meta=meta or {"type": "connective", "channel": "wideband", "noise": "none"})


def reference_end_frame(item, detector, params=ts.SessionParams()):
    """세션(_on_frame)을 손으로 옮긴 기준 구현."""
    speech = silence = 0
    prepared = False
    smart = None
    prep_frame = None
    for i, p in enumerate(item.probs):
        if p >= params.vad_threshold:
            if silence > 0 and prepared:
                prepared, smart = False, None
            speech += 20
            silence = 0
            continue
        if speech < params.min_speech_ms:
            if speech and silence + 20 >= params.precall_ms:
                speech = silence = 0
            elif speech:
                silence += 20
            continue
        silence += 20
        if not prepared and silence >= params.precall_ms:
            prepared, prep_frame = True, i
        if prepared and smart is None and i >= prep_frame + params.smart_delay_frames:
            smart = item.smart_fn(prep_frame)
        text = ts.text_at(item, i, silence, params)
        d = detector.decide(TurnSignals(silence_ms=silence, speech_ms=speech, text=text,
                                        smart_turn_p=smart, expect_short=params.expect_short))
        if d.end:
            return i
    return None


def test_false_cut_in_pause():
    # 1초 말 → 800ms 쉼(텍스트 완결처럼, p 높음) → 1초 말 → 꼬리
    it = make_item([("sil", 400), ("speech", 1000), ("sil", 800), ("speech", 1000), ("sil", 1500)],
                   ["배송 조회하려고요.", "그리고 주소도 바꿀 수 있나요?"], smart_p=0.9, cut_types=["trap"])
    r = ts.simulate(it, RuleTurnDetector())
    assert r.outcome == "false_cut"
    assert r.where == "trap"
    # 쉼 시작 1.4s, fast 400ms → 침묵 400ms가 되는 프레임 끝 = 1.8s
    assert r.decision_t == pytest.approx(1.8, abs=1e-6)


def test_latency_at_final_end():
    it = make_item([("sil", 400), ("speech", 1000), ("sil", 1500)], ["배송 언제 와요"], smart_p=0.5)
    r = ts.simulate(it, RuleTurnDetector())      # final, p 애매 → mid 600
    assert r.outcome == "ok"
    assert r.latency_ms == pytest.approx(600, abs=1e-6)


def test_hold_text_waits_hard_max():
    it = make_item([("sil", 200), ("speech", 800), ("sil", 1500)], ["주문을 했는데"], smart_p=0.9)
    r = ts.simulate(it, RuleTurnDetector())
    assert r.latency_ms == pytest.approx(DEFAULT_CONFIG.hard_max_ms, abs=1e-6)
    assert r.hard_max


def test_short_blip_is_ignored():
    # 40ms 잡음 → 무음 300ms → 진짜 말 600ms → 꼬리
    it = make_item([("sil", 200), ("speech", 40), ("sil", 300), ("speech", 600), ("sil", 1500)],
                   ["", "네"], smart_p=0.9, cut_types=["filler"])
    r = ts.simulate(it, RuleTurnDetector())
    assert r.outcome == "ok"
    assert r.latency_ms == pytest.approx(400, abs=1e-6)     # 단답 fast


def test_no_end_is_miss():
    it = make_item([("sil", 200), ("speech", 1000), ("sil", 300)], ["주문을 했는데"], smart_p=0.1)
    r = ts.simulate(it, RuleTurnDetector())
    assert r.outcome == "miss"


def test_smart_turn_called_once_per_long_pause_with_last_8s():
    calls = []

    def smart(i):
        calls.append(i)
        return 0.1

    it = make_item([("sil", 400), ("speech", 1000), ("sil", 800), ("speech", 600), ("sil", 150),
                    ("speech", 600), ("sil", 1500)],
                   ["주문번호가", "어제", "언제 와요?"], smart_p=smart, cut_types=["particle", "particle"])
    ts.simulate(it, RuleTurnDetector())
    # 쉼 800ms(≥200) 1회, 150ms 쉼은 준비 전, 최종 끝 1회
    assert len(calls) == 2
    # 준비 프레임 = 침묵 200ms가 되는 프레임 = 쉼 시작 프레임 + 9
    assert calls[0] == int(round(1.4 / F)) + 9


def test_text_at_uses_point_text_and_lag():
    it = make_item([("sil", 400), ("speech", 1000), ("sil", 800), ("speech", 1000), ("sil", 1500)],
                   ["주문을 했는데", "언제 와요?"])
    i_pause = int(round(1.4 / F)) + 15          # 쉼 안, 침묵 320ms
    assert ts.text_at(it, i_pause, 320, ts.SessionParams()) == "주문을 했는데"
    assert ts.text_at(it, i_pause, 320, ts.SessionParams(text_lag_ms=400)) == ""
    assert ts.text_at(it, 5, 0, ts.SessionParams()) == ""
    assert ts.text_at(it, i_pause, 320, ts.SessionParams(empty_text_rate=1.0)) == ""
    # 비율 근사: 많은 구간에서 약 35%가 빈 텍스트
    empt = [ts.text_empty(it, k, ts.SessionParams(empty_text_rate=0.35)) for k in range(2000)]
    assert 0.3 < sum(empt) / len(empt) < 0.4


@pytest.mark.parametrize("seed", range(6))
def test_simulate_matches_reference_and_fast(seed):
    rng = random.Random(seed)
    texts_pool = ["주문번호가", "주문을 했는데", "배송 조회하려고요.", "어", "공일공에 일이삼사", "네", "언제 와요?",
                  "취소해 주세요", "반품 접수"]
    configs = [DEFAULT_CONFIG, TurnConfig(fast_ms=300, mid_ms=500, slow_ms=700, hold_ms=1000),
               TurnConfig(p_high=0.5, p_low=0.1, fast_ms=240)]
    from agent.turn.detector import make_detector
    dets = [RuleTurnDetector(c) for c in configs] + [make_detector()]
    for k in range(25):
        pat = [("sil", rng.choice([200, 600, 1000]))]
        n = rng.randint(1, 3)
        for j in range(n):
            pat.append(("speech", rng.choice([40, 100, 600, 1200])))
            pat.append(("sil", rng.choice([100, 260, 420, 700, 1000, 1300]) if j < n - 1 else 1500))
        texts = [rng.choice(texts_pool) for _ in range(n)]
        p = rng.random()
        it = make_item(pat, texts, smart_p=p, uid=f"{seed}-{k}")
        for params in (ts.SessionParams(), ts.SessionParams(smart_delay_frames=2, text_lag_ms=300),
                       ts.SessionParams(smart_delay_frames=1, empty_text_rate=0.35)):
            for det in dets:
                ref = reference_end_frame(it, det, params)
                r = ts.simulate(it, det, params)
                assert r.end_frame == ref, (pat, texts, p, det)
                fast = ts.simulate_fast(it, det, params)
                assert fast.end_frame == ref, (pat, texts, p, det, "fast")


def test_summarize_metrics():
    R = ts.SimResult
    res = [
        R(uid="a", end_frame=1, decision_t=1.0, outcome="false_cut", where="trap", latency_ms=None),
        R(uid="b", end_frame=1, decision_t=1.0, outcome="ok", latency_ms=400.0),
        R(uid="c", end_frame=1, decision_t=1.0, outcome="ok", latency_ms=600.0),
        R(uid="d", end_frame=1, decision_t=1.0, outcome="ok", latency_ms=1100.0, hard_max=True),
        R(uid="e", end_frame=None, decision_t=None, outcome="miss", latency_ms=None),
    ]
    meta = {"a": {"type": "trap", "channel": "phone", "noise": "none"},
            "b": {"type": "complete", "channel": "phone", "noise": "none"},
            "c": {"type": "trap", "channel": "wideband", "noise": "white"},
            "d": {"type": "connective", "channel": "wideband", "noise": "white"},
            "e": {"type": "complete", "channel": "phone", "noise": "none"}}
    s = ts.summarize(res, meta, group_keys=("type", "channel"))
    assert s["n"] == 5
    assert s["false_cut_rate"] == pytest.approx(0.2)
    assert s["miss_rate"] == pytest.approx(0.2)
    assert s["latency_mean"] == pytest.approx(700.0)
    assert s["latency_p50"] == pytest.approx(600.0)
    assert s["latency_p95"] == pytest.approx(np.percentile([400, 600, 1100], 95))
    assert s["hard_max_rate"] == pytest.approx(1 / 3)
    assert s["by"]["type"]["trap"]["false_cut_rate"] == pytest.approx(0.5)
    assert s["by"]["channel"]["phone"]["n"] == 3


def test_extract_features_with_fakes():
    class FakeVAD:
        def __init__(self):
            self.n = 0

        def process(self, frame):
            x = np.frombuffer(frame, np.int16).astype(np.float32)
            return 1.0 if np.sqrt((x ** 2).mean()) > 500 else 0.0

    class FakeSmart:
        def __init__(self):
            self.lens = []

        def predict(self, pcm):
            self.lens.append(len(pcm))
            return 0.42

    sr = 16000
    sil = np.zeros(int(0.5 * sr), np.int16)
    sp = (3000 * np.sin(np.arange(int(9.0 * sr)) * 0.05)).astype(np.int16)
    audio = np.concatenate([sil, sp, np.zeros(int(0.6 * sr), np.int16), sp[: sr], np.zeros(int(1.5 * sr), np.int16)])
    sm = FakeSmart()
    probs, smart = ts.extract_features(audio, FakeVAD(), sm)
    assert len(probs) == int(np.ceil(len(audio) / 320))
    assert len(smart) == 2 and set(smart.values()) == {0.42}
    assert max(sm.lens) <= 8 * sr * 2
    first = min(smart)
    assert first == int(round(9.5 / F)) + 9


# ── 보정 결정기 ──
def test_make_detector_uses_calibrated_config():
    from agent.turn import detector as dm

    d1, d2 = dm.make_detector(), dm.make_detector()
    assert d1 is not d2
    assert d1.config is dm.CALIBRATED_CONFIG
    assert isinstance(d1.decide(TurnSignals(silence_ms=100, speech_ms=1000, text="네")), TurnDecision)
    for text in ("네", "언제 와요?", "", "주문번호가"):
        for p in (None, 0.0, 0.5, 1.0):
            assert not d1.decide(TurnSignals(silence_ms=180, speech_ms=1000, text=text, smart_turn_p=p)).end
    assert d1.decide(TurnSignals(silence_ms=dm.CALIBRATED_CONFIG.hard_max_ms, speech_ms=1000,
                                 text="언제 와요?", smart_turn_p=0.0)).end
    assert not d1.decide(TurnSignals(silence_ms=3000, speech_ms=100, text="")).end   # 잡음성
    w, why = d1.wait_ms(TurnSignals(silence_ms=0, speech_ms=1000, text="언제 와요?", smart_turn_p=0.9))
    assert dm.CALIBRATED_CONFIG.min_silence_ms <= w <= dm.CALIBRATED_CONFIG.hard_max_ms
    assert isinstance(why, str)


def test_default_detector_unchanged():
    s = TurnSignals(silence_ms=450, speech_ms=1000, text="배송 언제 와요?", smart_turn_p=0.9)
    assert decide(s).end
    assert RuleTurnDetector().config is DEFAULT_CONFIG


@pytest.mark.parametrize("text,kind", [
    ("주문번호 이육일일", "digits_partial"),          # 숫자 그룹 사이 쉼(rules는 weak_final)
    ("2611", "digits_partial"),
    ("2611, 150", "digits_partial"),
    ("010에 1234", "digits_partial"),
    ("공일공에", "digits_incomplete"),
    ("주문번호 2611 150 231이요", "digits_complete"),
    ("010에 8411 3910이요", "digits_complete"),       # rules는 digits_incomplete(마지막 덩어리만 셈)
    ("제 번호는 공일공에 팔사일일 삼구일영이요.", "digits_complete"),
    ("3", "weak_final"),                               # 한 자리 답은 그대로
    ("네", "short_answer"),
    ("배송 언제 와요?", "question"),
])
def test_table_kind_digit_fixes(text, kind):
    from agent.turn.detector import TableTurnConfig, table_kind

    assert table_kind(text, TableTurnConfig()).kind == kind
    assert table_kind(text, TableTurnConfig(digit_partial_hold=False)).kind == __import__(
        "agent.turn.rules", fromlist=["analyze"]).analyze(text).kind


def test_calibrated_holds_digit_group_pause():
    from agent.turn import detector as dm

    d = dm.make_detector()
    w_partial, _ = d.wait_ms(TurnSignals(silence_ms=0, speech_ms=2000, text="주문번호 이육일일", smart_turn_p=0.95))
    w_done, _ = d.wait_ms(TurnSignals(silence_ms=0, speech_ms=2000, text="주문번호 이육일일 일오공 이삼일이요",
                                      smart_turn_p=0.95))
    assert w_partial >= 900 and w_done < w_partial
