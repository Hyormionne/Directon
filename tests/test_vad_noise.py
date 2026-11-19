"""AG-T10c 잡음 강건 VAD: 배경 잡음 바닥을 추정해 정상 상태 잡음(white·babble 대용)을 음성으로 보지 않는다.

- NoiseGate: 프레임 에너지 + 내부 VAD 확률 → 음성 여부. 잡음 바닥을 최소 통계로 추정하고, 잡음이 출렁이거나
  (babble) 내부 VAD가 고른 잡음에 속으면 (a) 확률 임계 상향, (b) 에너지 ≥ 바닥 + margin, (c) 연속 프레임 조건을
  함께 건다. 그 밖에는 내부 VAD 그대로(white·pink·hum은 Silero가 이미 잘 거른다).
- NoiseRobustVAD(inner): 임의의 VAD를 감싼다. NoiseRobustSileroVAD: SileroVAD 하위 클래스(서버 make_vad 기본).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent import audio
from agent.core.types import FRAME_BYTES, SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "DIRECT_ON 2026 Starter Kit/data/audio/demo"
MODELS = ROOT / "models"
FR = FRAME_BYTES // 2          # 320 샘플 = 20ms
SR = SAMPLE_RATE


# ── 합성 신호 ─────────────────────────────────────────
def _unit(x: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(x * x)))
    return x / r if r > 0 else x


def white(sec: float, rms: float, seed: int = 0) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(int(sec * SR))) * rms


def speech_shaped(sec: float, rms: float, seed: int = 0) -> np.ndarray:
    """babble 대용: 200~4000Hz 대역 핑크 잡음 × 음절 리듬(2.5~5.5Hz) 변조."""
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    band = (f >= 200) & (f <= 4000)
    spec = np.where(band, spec / np.sqrt(np.maximum(f, 1.0)), 0)
    x = np.fft.irfft(spec, n)
    t = np.arange(n) / SR
    mod = np.ones(n)
    for _ in range(4):
        mod += 0.5 * np.abs(np.sin(2 * np.pi * rng.uniform(2.5, 5.5) * t + rng.uniform(0, np.pi)))
    return _unit(x * mod) * rms


def to_pcm(x: np.ndarray) -> bytes:
    return np.clip(np.round(x), -32768, 32767).astype(np.int16).tobytes()


def frames_of(x: np.ndarray) -> list[bytes]:
    return audio.frames(to_pcm(x))


def run_gate(vad, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.array([vad.process(f) >= threshold for f in frames_of(x)])


def first_silence_reach(speech: np.ndarray, start: int, need: int = 10) -> int | None:
    """start 프레임부터 보아 비음성 프레임이 need개 연속된 첫 시점(프레임 번호). 없으면 None."""
    run = 0
    for i in range(start, len(speech)):
        run = 0 if speech[i] else run + 1
        if run >= need:
            return i
    return None


class FooledVAD:
    """잡음에도 속는 VAD 흉내: 소리가 조금이라도 있으면 0.9."""

    def __init__(self):
        self.resets = 0

    def process(self, frame: bytes) -> float:
        return 0.9 if audio.rms(frame) > 10 else 0.0

    def reset(self) -> None:
        self.resets += 1


# ── NoiseRobustVAD(가짜 내부 VAD) ─────────────────────
def test_steady_white_noise_not_speech():
    from agent.turn.vad import NoiseRobustVAD

    vad = NoiseRobustVAD(FooledVAD())
    sp = run_gate(vad, white(3.0, 300))
    assert sp[50:].mean() == 0.0                       # 1초 뒤(바닥·속음 추정 후)엔 음성 판정 없음


def test_babble_substitute_not_speech():
    from agent.turn.vad import NoiseRobustVAD

    vad = NoiseRobustVAD(FooledVAD())
    sp = run_gate(vad, speech_shaped(4.0, 300, seed=1))
    assert sp[30:].mean() <= 0.03
    reach = first_silence_reach(sp, 30)
    assert reach is not None and reach <= 30 + 25      # 침묵 200ms가 0.5초 안에


def test_loud_speech_over_noise_detected_and_tail_noise_is_silence():
    from agent.turn.vad import NoiseRobustVAD

    noise = speech_shaped(5.0, 300, seed=2)
    speech = np.zeros_like(noise)
    a, b = int(2.0 * SR), int(3.5 * SR)
    speech[a:b] = speech_shaped(1.5, 1700, seed=3)     # 잡음보다 +15dB
    vad = NoiseRobustVAD(FooledVAD())
    sp = run_gate(vad, noise + speech)
    fa, fb = a // FR, b // FR
    assert sp[fa + 3:fb].mean() >= 0.8                 # 말 구간은 음성
    reach = first_silence_reach(sp, fb)
    assert reach is not None and reach - fb <= 15      # 말 끝 뒤 잡음은 300ms 안에 '침묵 200ms'


def test_clean_input_passes_inner_probability_through():
    from agent.turn.vad import NoiseRobustVAD

    x = np.concatenate([np.zeros(int(0.5 * SR)), speech_shaped(1.0, 3000, seed=4), np.zeros(int(0.5 * SR))])
    inner, wrapped = FooledVAD(), NoiseRobustVAD(FooledVAD())
    for f in frames_of(x):
        assert wrapped.process(f) == inner.process(f)  # 잡음 없으면 원래 VAD 그대로


def test_reset_keeps_noise_floor_and_resets_inner():
    from agent.turn.vad import NoiseRobustVAD

    inner = FooledVAD()
    vad = NoiseRobustVAD(inner)
    run_gate(vad, white(2.0, 300, seed=5))
    vad.reset()
    assert inner.resets == 1
    sp = run_gate(vad, white(1.0, 300, seed=6))
    assert sp.mean() == 0.0                            # 다음 턴도 처음부터 잡음으로 안다
    assert vad.noisy


def test_noise_gate_config_thresholds():
    from agent.turn.vad import NoiseGateConfig

    c = NoiseGateConfig()
    assert c.p_noisy > c.p_base >= 0.5
    assert c.margin_db > 0 and c.onset_frames >= 2


# ── 실제 Silero ───────────────────────────────────────
needs_model = pytest.mark.skipif(not (MODELS / "silero_vad.onnx").exists(), reason="models/ 없음")


def demo_speech(name: str, rms: float = 3000.0) -> np.ndarray:
    pcm, sr = audio.wav_read(DEMO / f"{name}.wav")
    x = np.frombuffer(pcm, np.int16).astype(np.float64)
    fr = np.array([np.sqrt(np.mean(x[i:i + FR] ** 2)) for i in range(0, len(x) - FR, FR)])
    on = np.flatnonzero(fr > 300)
    x = x[on[0] * FR:(on[-1] + 1) * FR]
    lv = np.sqrt(np.mean(fr[fr > 300] ** 2))
    return x * (rms / lv)


def demo_babble(sec: float, rms: float, seed: int = 0) -> np.ndarray:
    """데모 음성 여러 개를 어긋나게 겹친 웅성거림."""
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    out = np.zeros(n)
    for name in ("DEMO1", "DEMO3") * 4:
        s = _unit(demo_speech(name))
        tiled = np.tile(s, int(np.ceil(n / len(s))) + 1)
        off = int(rng.integers(0, len(s)))
        out += tiled[off:off + n] * rng.uniform(0.5, 1.0)
    return _unit(out) * rms


@needs_model
def test_make_vad_default_plain_silero_and_gate_opt_in(monkeypatch):
    # AD8: 기본은 원래 SileroVAD, DIRECTON_VAD_NOISE_GATE=1일 때만 잡음 래퍼
    from agent.turn.vad import NoiseRobustSileroVAD, SileroVAD, make_vad

    monkeypatch.delenv("DIRECTON_VAD_NOISE_GATE", raising=False)
    v = make_vad()
    assert isinstance(v, SileroVAD) and not isinstance(v, NoiseRobustSileroVAD)
    monkeypatch.setenv("DIRECTON_VAD_NOISE_GATE", "1")
    v = make_vad()
    assert isinstance(v, NoiseRobustSileroVAD)


@needs_model
@pytest.mark.parametrize("kind", ["white", "babble"])
def test_silero_noise_robust_on_noise_and_speech(kind):
    from agent.turn.vad import NoiseRobustSileroVAD

    speech = demo_speech("DEMO2")
    lead, tail = int(1.5 * SR), int(1.5 * SR)
    n = lead + len(speech) + tail
    noise = white(n / SR, 600, seed=7) if kind == "white" else demo_babble(n / SR, 600, seed=8)
    clean = np.concatenate([np.zeros(lead), speech, np.zeros(tail)])
    vad = NoiseRobustSileroVAD()
    sp = run_gate(vad, clean + noise)
    loud = np.array([np.sqrt(np.mean(f ** 2)) > 1500 for f in clean[: len(sp) * FR].reshape(-1, FR)])
    assert sp[loud].mean() >= 0.6                      # 실제 음성은 음성으로
    assert sp[30:lead // FR].mean() <= 0.2             # 말 전 잡음은 대부분 비음성(원래 Silero는 babble에서 1.0)
    end = (lead + len(speech)) // FR
    reach = first_silence_reach(sp, end)
    assert reach is not None and reach - end <= 25     # 말 끝 뒤 잡음은 0.5초 안에 '침묵 200ms'


@needs_model
def test_silero_noise_robust_reset_keeps_silero_state_contract():
    from agent.turn.vad import NoiseRobustSileroVAD

    v = NoiseRobustSileroVAD()
    for f in frames_of(white(1.0, 600, seed=9)):
        v.process(f)
    v.reset()
    assert v.n_chunks == 0 and v.last_prob == 0.0      # SileroVAD.reset 계약 유지
    assert v.noisy                                     # 잡음 바닥은 유지
