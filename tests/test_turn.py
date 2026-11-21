"""AG-T03 턴 감지 테스트: Silero VAD, Smart Turn v3.2, 한국어 규칙, 결정기."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from agent import audio
from agent.core.types import FRAME_BYTES, SAMPLE_RATE, TurnSignals

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "tests/fixtures/starter_kit_demo/audio"
MODELS = ROOT / "models"


@pytest.fixture(scope="session")
def model_paths():
    """모델이 없으면 skip하지 않고 다운로드 스크립트로 먼저 확보한다."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("download_models", ROOT / "scripts/download_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ensure_all(MODELS)


def demo_pcm(name: str) -> bytes:
    pcm, sr = audio.wav_read(DEMO / f"{name}.wav")
    assert sr == SAMPLE_RATE
    return pcm


# ══════════════════════════════════════════════════════
# Silero VAD
# ══════════════════════════════════════════════════════
@pytest.fixture()
def vad(model_paths):
    from agent.turn.vad import SileroVAD

    return SileroVAD(model_paths["silero_vad.onnx"])


def run_vad(vad, pcm: bytes) -> np.ndarray:
    return np.array([vad.process(f) for f in audio.frames(pcm)])


def test_vad_silence_low(vad):
    probs = run_vad(vad, b"\x00" * FRAME_BYTES * 50)
    assert probs.max() < 0.2


def test_vad_demo_speech_high(vad):
    pcm = demo_pcm("DEMO1")
    fr = audio.frames(pcm)
    probs = np.array([vad.process(f) for f in fr])
    loud = np.array([audio.rms(f) > 1000 for f in fr])
    quiet = np.array([audio.rms(f) < 50 for f in fr])
    assert loud.sum() > 100
    assert probs[loud].mean() > 0.5
    assert probs[quiet].mean() < probs[loud].mean()


def test_vad_rechunks_20ms_to_512(vad):
    f = b"\x00" * FRAME_BYTES  # 320샘플
    assert vad.process(f) == 0.0          # 아직 512샘플 미만 → 추론 없음
    assert vad.n_chunks == 0
    vad.process(f)                        # 640샘플 → 1청크(512) + 128 남음
    assert vad.n_chunks == 1
    for _ in range(6):                    # 총 8프레임 = 2560샘플 = 5청크
        vad.process(f)
    assert vad.n_chunks == 5


def test_vad_returns_latest_prob_between_chunks(vad):
    pcm = demo_pcm("DEMO1")
    fr = audio.frames(pcm)[:60]
    probs = [vad.process(f) for f in fr]
    # 추론이 없는 프레임은 직전 확률을 그대로 반환
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert vad.last_prob == probs[-1]


def test_vad_state_and_reset(vad):
    pcm = demo_pcm("DEMO1")[: SAMPLE_RATE * 2 * 2]
    a = run_vad(vad, pcm)
    b = run_vad(vad, pcm)                 # 상태가 이어지므로 결과가 다를 수 있음
    vad.reset()
    assert vad.n_chunks == 0 and vad.last_prob == 0.0
    c = run_vad(vad, pcm)
    np.testing.assert_allclose(a, c, atol=1e-6)
    assert b.shape == a.shape


def test_vad_matches_reference_chunk_loop(vad, model_paths):
    """원본 OnnxWrapper(64샘플 컨텍스트 + state)와 같은 결과."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_paths["silero_vad.onnx"]))
    x = audio.pcm_to_float(demo_pcm("DEMO1"))[: 512 * 40]
    state = np.zeros((2, 1, 128), np.float32)
    ctx = np.zeros((1, 64), np.float32)
    ref = []
    for i in range(0, len(x), 512):
        inp = np.concatenate([ctx, x[None, i:i + 512]], axis=1)
        out, state = sess.run(None, {"input": inp, "state": state, "sr": np.array(16000, np.int64)})
        ctx = inp[:, -64:]
        ref.append(float(out[0, 0]))
    got = []
    for f in audio.frames(audio.float_to_pcm(x)):
        n0 = vad.n_chunks
        p = vad.process(f)
        if vad.n_chunks > n0:
            got.append(p)
    np.testing.assert_allclose(got, ref[: len(got)], atol=2e-3)
    assert len(got) >= 39


def test_vad_white_noise_low(vad):
    """저잡음(통화 배경 수준, RMS≈30) 백색잡음만 있는 구간은 낮은 확률."""
    rng = np.random.default_rng(1)
    noise = np.clip(rng.normal(0, 30, SAMPLE_RATE * 3), -32768, 32767).astype(np.int16).tobytes()
    probs = run_vad(vad, noise)
    assert probs.mean() < 0.2
    assert probs.max() < 0.5


def test_vad_speech_with_noise_snr20(vad):
    """음성 + 백색잡음(SNR 20dB): 음성 구간은 높게, 잡음만 있는 구간은 낮게."""
    rng = np.random.default_rng(2)
    sp = audio.pcm_to_float(demo_pcm("DEMO1"))
    fr_rms = np.array([audio.rms(f) for f in audio.frames(demo_pcm("DEMO1"))])
    speech_rms = np.mean(fr_rms[fr_rms > 1000]) / 32768.0
    noise_rms = speech_rms / (10 ** (20 / 20))
    lead = rng.normal(0, noise_rms, SAMPLE_RATE * 2).astype(np.float32)   # 잡음만 2초
    mix = np.concatenate([lead, sp + rng.normal(0, noise_rms, len(sp)).astype(np.float32)])
    fr = audio.frames(audio.float_to_pcm(mix))
    probs = np.array([vad.process(f) for f in fr])
    n_lead = len(lead) // 320
    assert probs[5:n_lead].mean() < 0.2
    loud = np.array([audio.rms(f) > 1000 for f in audio.frames(demo_pcm("DEMO1"))])
    sp_probs = probs[n_lead:n_lead + len(loud)]
    assert sp_probs[loud[: len(sp_probs)]].mean() > 0.5


# ══════════════════════════════════════════════════════
# Smart Turn v3.2
# ══════════════════════════════════════════════════════
# 원본 파이프라인(transformers WhisperFeatureExtractor(chunk_length=8) + inference.py의
# truncate_audio_to_last_n_seconds + smart-turn-v3.2-cpu.onnx)으로 미리 계산한 기준값.
# (mean, std, f[0,10,400], f[0,40,799], f[0,70,100], prob)
REF = {
    "DEMO1": (0.353712, 0.517765, 1.532732, -0.201116, 0.317279, 0.038061),
    "DEMO2": (0.233348, 0.358546, 0.057099, 0.057099, 0.057099, 0.990507),
    "DEMO3": (0.473817, 0.499394, 1.092759, -0.200370, -0.200370, 0.029157),   # 9.3초 → 뒤 8초
    "DEMO1_half": (0.054299, 0.344775, -0.067847, -0.067847, -0.067847, 0.519683),  # 1.5초 → 앞 0패딩
}


def ref_audio(key: str) -> np.ndarray:
    if key == "DEMO1_half":
        return audio.pcm_to_float(demo_pcm("DEMO1"))[: int(1.5 * SAMPLE_RATE)]
    return audio.pcm_to_float(demo_pcm(key))


@pytest.fixture(scope="module")
def smart_turn(model_paths):
    from agent.turn.smart_turn import SmartTurnV3

    st = SmartTurnV3(model_paths["smart-turn-v3.2-cpu.onnx"], threads=16)
    st.predict(b"\x00\x00" * SAMPLE_RATE)  # 워밍업
    return st
















def test_vad_latency_report(vad):
    fr = audio.frames(demo_pcm("DEMO1"))
    t0 = time.perf_counter()
    for f in fr:
        vad.process(f)
    per_chunk = (time.perf_counter() - t0) * 1000 / max(vad.n_chunks, 1)
    print(f"\n[vad] {per_chunk:.3f}ms/512-chunk over {vad.n_chunks} chunks")
    assert per_chunk < 10


# ══════════════════════════════════════════════════════
# 한국어 텍스트 규칙
# ══════════════════════════════════════════════════════
END_STRONG = [            # end_score >= 0.5
    "배송 언제 와요?",
    "주문 취소하고 싶어요",
    "환불해 주세요",
    "주문한 거 확인했습니다",
    "배송이 너무 늦다",
    "이거 교환 가능한가요",
    "취소되는 거죠",
    "언제 도착합니까",
    "반품할 수 있을까",
    "어제 주문했는데요",
    "사이즈가 안 맞거든요",
    "주소 바꿔 주세요.",
    "네",
    "아니요",
    "예",
    "맞아요",
    "네.",
    "01012345678",
    "010 1234 5678",
    "010-1234-5678",
    "2611150231",
    "주문번호는 2611150231이요",
    "공일공 일이삼사 오육칠팔",
    "이게 맞아?",
]
END_WEAK = [              # 0 < end_score < 0.5
    "주문번호가요",
    "저는요",
]
HOLD = [                  # end_score < 0
    "주문하고",
    "사이즈가 작아서",
    "취소하면",
    "어제 주문했는데",
    "물건이 왔으니까",
    "좋긴 하지만",
    "주문번호가",
    "택배를",
    "제 전화번호는",
    "부산에서",
    "어",
    "음",
    "그",
    "저기",
    "그리고",
    "그러니까 어",
    "010 1234",
    "공일공 일이삼사",
    "261115",
    "010에",
]
STRONG_HOLD_KINDS = {"connective", "filler", "digits_incomplete"}


@pytest.mark.parametrize("text", END_STRONG)
def test_rules_end_strong(text):
    from agent.turn.rules import end_score

    assert end_score(text) >= 0.5, text


@pytest.mark.parametrize("text", END_WEAK)
def test_rules_end_weak(text):
    from agent.turn.rules import end_score

    s = end_score(text)
    assert 0.0 < s < 0.5, (text, s)


@pytest.mark.parametrize("text", HOLD)
def test_rules_hold(text):
    from agent.turn.rules import end_score

    assert end_score(text) < 0.0, text


def test_rules_case_count():
    assert len(END_STRONG) + len(END_WEAK) + len(HOLD) >= 30


@pytest.mark.parametrize("text,kind", [
    ("주문하고", "connective"),
    ("사이즈가 작아서", "connective"),
    ("어제 주문했는데", "connective"),
    ("그리고", "connective"),
    ("어", "filler"),
    ("저기", "filler"),
    ("010 1234", "digits_incomplete"),
    ("010에", "digits_incomplete"),
    ("주문번호가", "particle"),
    ("네", "short_answer"),
    ("배송 언제 와요?", "question"),
    ("01012345678", "digits_complete"),
    ("주문번호가요", "weak_final"),
])
def test_rules_kind(text, kind):
    from agent.turn.rules import analyze

    assert analyze(text).kind == kind


def test_rules_ordering():
    from agent.turn.rules import end_score

    assert end_score("주문번호가요") < end_score("주문번호 알려 드릴게요")
    assert end_score("주문번호가") < end_score("주문번호가요")
    assert end_score("어") <= end_score("주문번호가")


def test_rules_range_and_empty():
    from agent.turn.rules import end_score

    assert end_score("") == 0.0
    assert end_score("   ") == 0.0
    for t in END_STRONG + END_WEAK + HOLD + ["배송 조회", "ㅋㅋ", "?!"]:
        assert -1.0 <= end_score(t) <= 1.0


# ══════════════════════════════════════════════════════
# 결정기
# ══════════════════════════════════════════════════════
def sig(silence_ms, text="배송 언제 와요?", p=0.9, speech_ms=1500, short=False):
    return TurnSignals(silence_ms=silence_ms, speech_ms=speech_ms, text=text,
                       smart_turn_p=p, expect_short=short)


@pytest.fixture()
def det():
    from agent.turn.detector import RuleTurnDetector

    return RuleTurnDetector()


def test_detector_defaults():
    from agent.turn.detector import TurnConfig

    c = TurnConfig()
    assert (c.min_silence_ms, c.hard_max_ms) == (200, 1100)
    assert (c.fast_ms, c.mid_ms, c.slow_ms, c.hold_ms) == (400, 600, 900, 1100)
    assert (c.short_step_ms, c.short_floor_ms, c.min_speech_ms) == (100, 300, 250)
    assert (c.p_high, c.p_low) == (0.7, 0.3)


@pytest.mark.parametrize("silence", [0, 100, 199])
def test_detector_below_200_never(det, silence):
    for text, p, short in [("네", 1.0, True), ("01012345678", 0.99, True), ("", None, False)]:
        assert det.decide(sig(silence, text, p, short=short)).end is False


@pytest.mark.parametrize("text,p", [("어", 0.0), ("주문하고", 0.1), ("010 1234", None), ("", None), ("배송", 0.5)])
def test_detector_hard_max_always(det, text, p):
    assert det.decide(sig(1100, text, p)).end is True
    assert det.decide(sig(5000, text, p)).end is True


def test_detector_fast_400(det):
    assert det.decide(sig(399, "배송 언제 와요?", 0.7)).end is False
    assert det.decide(sig(400, "배송 언제 와요?", 0.7)).end is True
    assert det.decide(sig(400, "배송 조회", 0.95)).end is True     # end_score == 0 도 fast


def test_detector_mid_600(det):
    s = "배송 언제 와요?"
    assert det.decide(sig(599, s, 0.5)).end is False
    assert det.decide(sig(600, s, 0.5)).end is True
    assert det.decide(sig(599, s, 0.3)).end is False             # p_low 경계: 0.3은 애매
    assert det.decide(sig(600, s, 0.3)).end is True
    assert det.decide(sig(599, s, 0.69)).end is False


def test_detector_slow_900(det):
    assert det.decide(sig(899, "배송 언제 와요?", 0.29)).end is False   # p < 0.3
    assert det.decide(sig(900, "배송 언제 와요?", 0.29)).end is True
    assert det.decide(sig(899, "주문번호가", 0.95)).end is False       # end_score < 0
    assert det.decide(sig(900, "주문번호가", 0.95)).end is True


@pytest.mark.parametrize("text", ["주문하고", "어", "그리고", "010 1234", "사이즈가 작아서"])
def test_detector_hold_1100(det, text):
    assert det.decide(sig(1099, text, 0.99)).end is False
    assert det.decide(sig(1100, text, 0.99)).end is True


def test_detector_expect_short_minus_100(det):
    s = "배송 언제 와요?"
    assert det.decide(sig(299, s, 0.9, short=True)).end is False
    assert det.decide(sig(300, s, 0.9, short=True)).end is True       # 400-100
    assert det.decide(sig(500, s, 0.5, short=True)).end is True       # 600-100
    assert det.decide(sig(499, s, 0.5, short=True)).end is False
    assert det.decide(sig(800, s, 0.1, short=True)).end is True       # 900-100
    assert det.decide(sig(999, "어", 0.1, short=True)).end is False
    assert det.decide(sig(1000, "어", 0.1, short=True)).end is True   # 1100-100


def test_detector_expect_short_floor_300():
    from agent.turn.detector import RuleTurnDetector, TurnConfig

    d = RuleTurnDetector(TurnConfig(fast_ms=350))
    assert d.decide(sig(299, "네", 0.9, short=True)).end is False
    assert d.decide(sig(300, "네", 0.9, short=True)).end is True


def test_detector_short_noise_not_end(det):
    r = det.decide(sig(1500, "", None, speech_ms=120))
    assert r.end is False and "noise" in r.reason
    r = det.decide(sig(700, "", 0.9, speech_ms=249))
    assert r.end is False and "noise" in r.reason
    assert det.decide(sig(0, "", None, speech_ms=0)).end is False
    # 짧아도 STT가 단어를 잡았으면(빠른 "네") 정상 규칙
    assert det.decide(sig(400, "네", 0.9, speech_ms=200)).end is True
    assert det.decide(sig(400, "", 0.9, speech_ms=250)).end is True


def test_detector_text_only_when_p_none(det):
    assert det.decide(sig(400, "네", None)).end is True                # 단답 → fast
    assert det.decide(sig(400, "01012345678", None)).end is True
    assert det.decide(sig(599, "배송 언제 와요?", None)).end is False   # 텍스트만이면 신중히 600
    assert det.decide(sig(600, "배송 언제 와요?", None)).end is True
    assert det.decide(sig(899, "주문번호가", None)).end is False
    assert det.decide(sig(900, "주문번호가", None)).end is True
    assert det.decide(sig(1099, "주문하고", None)).end is False
    assert det.decide(sig(599, "", None)).end is False
    assert det.decide(sig(600, "", None)).end is True


def test_detector_short_answer_with_ambiguous_p(det):
    # 단답("네")은 오디오 모델이 망설여도(0.3≤p<0.7) fast
    assert det.decide(sig(400, "네", 0.4)).end is True
    assert det.decide(sig(400, "네", 0.2)).end is False              # p가 미완료 쪽이면 slow


def test_detector_reason_and_wait_ms(det):
    r = det.decide(sig(450, "배송 언제 와요?", 0.9))
    assert r.end and "400" in r.reason
    w, why = det.wait_ms(sig(0, "어", 0.9))
    assert w == 1100 and why
    assert det.wait_ms(sig(0, "배송 언제 와요?", 0.9))[0] == 400


def test_detector_pure_function_and_injectable():
    from agent.turn.detector import TurnConfig, decide

    c = replace(TurnConfig(), fast_ms=500, hard_max_ms=1000)
    s = sig(450, "배송 언제 와요?", 0.9)
    assert decide(s, c).end is False
    assert decide(s, c) == decide(s, c)
    assert decide(sig(1000, "어", 0.0), c).end is True
    assert decide(s).end is True                                       # 기본 설정


def test_detector_satisfies_protocol(det):
    from agent.core.types import TurnDecision, TurnDetector

    d: TurnDetector = det
    assert isinstance(d.decide(sig(300)), TurnDecision)
