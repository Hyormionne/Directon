"""AG-T12 턴 보정 데이터셋: 원문 생성 규칙, 라벨(쉼 시작·최종 끝) 정확도, train/test 비중복 분할.

생성기: scripts/gen_turnset.py. 네트워크 없이 돈다(가짜 TTS = 사인파 버스트).
"""
from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "gen_turnset.py"
TURNSET = ROOT / "data" / "turnset"
LABELS = TURNSET / "labels.jsonl"
SR = 16000


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_turnset", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 가짜 TTS: 앞뒤 100ms 무음 + 문자 수에 비례한 사인파 버스트 ──
def burst_len(text: str) -> int:
    return int(0.08 * SR * max(3, len(text)))


def fake_synth(text: str) -> np.ndarray:
    n = burst_len(text)
    t = np.arange(n) / SR
    f0 = 180 + 15 * (len(text) % 7)
    x = 8000 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2)
    pad = np.zeros(SR // 10)
    return np.concatenate([pad, x, pad]).astype(np.int16)


SEGS = [
    {"text": "주문번호가", "pause_after_ms": 700},
    {"text": "이육일일 일오공", "pause_after_ms": 1200},
    {"text": "이삼일이요 언제 와요?", "pause_after_ms": 0},
]


def expected_positions(segs, lead_ms: int, speed: float):
    """가짜 TTS에서 기대하는 쉼 시작·최종 끝(초)."""
    t = 0.0
    pauses = []
    for i, s in enumerate(segs):
        t += burst_len(s["text"]) / SR
        if i < len(segs) - 1:
            pauses.append(lead_ms / 1000 + t / speed)
            t += s["pause_after_ms"] / 1000
    return pauses, lead_ms / 1000 + t / speed


@pytest.mark.parametrize("env", [
    {"channel": "wideband", "noise": "none", "speed": 1.0, "lead_ms": 400},
    {"channel": "phone_ulaw", "noise": "white", "snr_db": 10, "gain_db": -12, "speed": 1.0, "lead_ms": 900},
    {"channel": "phone", "noise": "pink", "snr_db": 20, "speed": 1.2, "lead_ms": 300},
    {"channel": "wideband", "noise": "hum", "snr_db": 20, "speed": 0.85, "lead_ms": 1200, "packet_loss": 0.02},
])
def test_render_labels_within_20ms(gen, env):
    r = gen.render(SEGS, fake_synth, env, seed=7)
    exp_pauses, exp_end = expected_positions(SEGS, env["lead_ms"], env["speed"])
    assert len(r.pauses) == 2
    for (t, dur_ms), e, seg in zip(r.pauses, exp_pauses, SEGS):
        assert abs(t - e) <= 0.02, (t, e)
        assert abs(dur_ms - seg["pause_after_ms"] / env["speed"]) <= 40
    assert abs(r.end_t - exp_end) <= 0.02
    assert r.audio.dtype == np.int16
    assert len(r.audio) / SR >= r.end_t + 1.0          # 뒤 무음(상한 1.1초 판정이 가능한 길이)


def test_render_single_segment_has_no_pause(gen):
    r = gen.render([{"text": "네 맞아요", "pause_after_ms": 0}], fake_synth,
                   {"channel": "wideband", "noise": "none", "speed": 1.0, "lead_ms": 500}, seed=1)
    assert r.pauses == []
    assert abs(r.end_t - (0.5 + burst_len("네 맞아요") / SR)) <= 0.02


def test_render_pitch_shift_keeps_timing(gen):
    env = {"channel": "wideband", "noise": "none", "speed": 1.0, "lead_ms": 400}
    r0 = gen.render(SEGS, fake_synth, env, seed=3, pitch_st=0)
    r1 = gen.render(SEGS, fake_synth, env, seed=3, pitch_st=3)
    for (a, _), (b, _) in zip(r0.pauses, r1.pauses):
        assert abs(a - b) <= 0.02
    assert abs(r0.end_t - r1.end_t) <= 0.02


# ── 원문 규칙 검사 ──
@pytest.mark.parametrize("text,cut,ok", [
    ("제가 어제 주문을 했는데", "connective", True),
    ("사이즈가 안 맞아서", "connective", True),
    ("반품하면", "connective", True),
    ("배송 조회하려고요.", "connective", False),
    ("주문번호가", "particle", True),
    ("어제 주문한 거를", "particle", True),
    ("주문번호가요", "particle", False),
    ("공일공에 일이삼사", "digits", True),
    ("주문번호 이육일일", "digits", True),
    ("주문번호 2611", "digits", False),
    ("어…", "filler", True),
    ("음 그러니까", "filler", True),
    ("배송 조회하려고요.", "trap", True),
    ("주문 취소했는데요?", "trap", True),
    ("배송 조회하려고", "trap", False),
])
def test_check_segment(gen, text, cut, ok):
    assert gen.check_segment(text, cut) is ok


@pytest.mark.parametrize("text,ok", [
    ("언제 와요?", True), ("네", True), ("아니요", True), ("오육칠팔이요", True),
    ("취소해 주세요.", True), ("취소해 줘", True), ("배송이 늦어서", False), ("주문 3개요", False),
])
def test_check_final(gen, text, ok):
    assert gen.check_final(text) is ok


def test_build_sources_templates_only(gen):
    utts = gen.build_sources(seed=1, gemini_drafts=[])
    assert len(utts) >= 600
    types = Counter(u["type"] for u in utts)
    for t in ("connective", "particle", "digits", "filler", "trap", "complete"):
        assert types[t] >= 50, types
    assert types["complete"] / len(utts) >= 0.25
    for u in utts:
        segs = u["segments"]
        assert segs[-1]["pause_after_ms"] == 0
        for s in segs[:-1]:
            assert 300 <= s["pause_after_ms"] <= 2000
            assert gen.check_segment(s["text"], s["cut"]), s
        assert gen.check_final(segs[-1]["text"]), segs[-1]
        assert not re.search(r"[0-9]", " ".join(s["text"] for s in segs))
    assert gen.build_sources(seed=1, gemini_drafts=[]) == utts       # 결정적


def test_sources_do_not_overlap_testset(gen):
    utts = gen.build_sources(seed=1, gemini_drafts=[])
    test_texts = gen.testset_texts()
    assert len(test_texts) > 50
    for u in utts:
        for s in u["segments"]:
            n = gen.norm_text(s["text"])
            assert len(n) <= 4 or n not in test_texts, s["text"]


def assert_split_disjoint(train, test):
    """분할 단위는 (원 음성, 피치) 가상 화자와 원문 템플릿. 원 음성 하나 이상은 통째로 test에만."""
    assert {r["voice"] for r in train}.isdisjoint({r["voice"] for r in test})
    assert {r["template_id"] for r in train}.isdisjoint({r["template_id"] for r in test})
    test_only = {r["base_voice"] for r in test} - {r["base_voice"] for r in train}
    assert test_only, "test 전용 원 음성이 없다"


def test_voice_is_base_plus_pitch(gen):
    assert gen.voice_id("edge:ko-KR-SunHiNeural", 0) == "edge:ko-KR-SunHiNeural"
    assert gen.voice_id("sapi:Heami", -3) == "sapi:Heami|-3"
    assert "sapi:Heami" in gen.BASE_VOICES
    assert not any(v.startswith("gemini:Kore") for v in gen.BASE_VOICES)


def test_split_disjoint_voices_and_templates(gen):
    utts = gen.build_sources(seed=1, gemini_drafts=[])
    rows = gen.assign_split_voice_env(utts, seed=1)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    assert 0.15 <= len(test) / len(rows) <= 0.35
    assert_split_disjoint(train, test)
    assert len({r["base_voice"] for r in rows}) >= 4
    assert len({r["voice"] for r in rows}) >= 11                       # 원 음성 × 피치 = 가상 화자
    assert not any(v.startswith("gemini:Kore") for v in (r["base_voice"] for r in rows))
    ch = Counter(r["env"]["channel"] for r in rows)
    assert set(ch) == {"wideband", "phone", "phone_ulaw"}
    assert len(Counter(r["env"]["noise"] for r in rows)) == 5


# ── 생성된 데이터셋 ──
def load_labels():
    assert LABELS.exists(), f"{LABELS} 없음 — scripts/gen_turnset.py로 생성"
    return [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_labels_file_shape():
    rows = load_labels()
    assert len(rows) >= 600
    types = Counter(r["type"] for r in rows)
    assert types["complete"] / len(rows) >= 0.25
    for r in rows:
        assert r["points"][-1]["kind"] == "end"
        ts = [p["t"] for p in r["points"]]
        assert ts == sorted(ts)
        n_pause = sum(1 for p in r["points"] if p["kind"] == "pause")
        assert n_pause == len(r["segments"]) - 1
        for p in r["points"]:
            assert "text_gt" in p and "text_stt" in p
            assert not re.search(r"[0-9]", p["text_gt"])


def test_labels_file_split_disjoint():
    rows = load_labels()
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    assert train and test
    assert_split_disjoint(train, test)
    assert len({r["voice"] for r in rows}) >= 11
