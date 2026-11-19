"""AG-T03 오디오 유틸 테스트."""
from __future__ import annotations

import numpy as np
import pytest

from agent import audio
from agent.core.types import FRAME_BYTES, SAMPLE_RATE


def tone(sec: float, freq: float = 440.0, amp: float = 0.5, sr: int = SAMPLE_RATE) -> bytes:
    t = np.arange(int(sec * sr)) / sr
    return audio.float_to_pcm(amp * np.sin(2 * np.pi * freq * t))


def silence(sec: float, sr: int = SAMPLE_RATE) -> bytes:
    return b"\x00\x00" * int(sec * sr)


# ── rms ──────────────────────────────────────────────
def test_rms_zero_and_empty():
    assert audio.rms(silence(0.1)) == 0.0
    assert audio.rms(b"") == 0.0


def test_rms_sine_amplitude():
    # 진폭 A 사인파의 RMS = A/sqrt(2) (int16 단위)
    pcm = tone(1.0, amp=0.5)
    assert audio.rms(pcm) == pytest.approx(0.5 * 32767 / np.sqrt(2), rel=0.01)


def test_rms_constant():
    pcm = np.full(320, 1000, dtype=np.int16).tobytes()
    assert audio.rms(pcm) == pytest.approx(1000.0)


# ── pcm <-> float ────────────────────────────────────
def test_pcm_float_roundtrip():
    x = np.array([0, 1, -1, 32767, -32768, 1234], dtype=np.int16)
    f = audio.pcm_to_float(x.tobytes())
    assert f.dtype == np.float32
    assert f.min() >= -1.0 and f.max() <= 1.0
    back = np.frombuffer(audio.float_to_pcm(f), dtype=np.int16)
    assert np.abs(back.astype(int) - x.astype(int)).max() <= 1


def test_float_to_pcm_clips():
    out = np.frombuffer(audio.float_to_pcm(np.array([2.0, -2.0, 0.0])), dtype=np.int16)
    assert out[0] == 32767 and out[1] == -32768 and out[2] == 0


def test_pcm_to_float_scale():
    f = audio.pcm_to_float(np.array([16384, -16384], dtype=np.int16).tobytes())
    assert f[0] == pytest.approx(0.5) and f[1] == pytest.approx(-0.5)


# ── frames ───────────────────────────────────────────
def test_frames_exact_split():
    pcm = tone(0.1)  # 1600 샘플 = 5 프레임
    fr = audio.frames(pcm)
    assert len(fr) == 5
    assert all(len(f) == FRAME_BYTES for f in fr)
    assert b"".join(fr) == pcm


def test_frames_last_zero_padded():
    pcm = tone(0.03)  # 480 샘플 = 1.5 프레임
    fr = audio.frames(pcm)
    assert len(fr) == 2
    assert len(fr[-1]) == FRAME_BYTES
    assert fr[-1][320:] == b"\x00" * 320
    assert (fr[0] + fr[1])[: len(pcm)] == pcm


def test_frames_empty():
    assert audio.frames(b"") == []


def test_frames_custom_size():
    fr = audio.frames(silence(0.1), frame_bytes=1024)
    assert all(len(f) == 1024 for f in fr)
    assert len(fr) == 4  # 3200 bytes → 3.125 → 4


# ── trim_silence ─────────────────────────────────────
def test_trim_silence_both_ends():
    pcm = silence(0.3) + tone(0.5) + silence(0.4)
    out = audio.trim_silence(pcm, thresh=100)
    assert len(out) == pytest.approx(len(tone(0.5)), abs=FRAME_BYTES)
    assert audio.rms(out) > 10000


def test_trim_silence_all_silent():
    assert audio.trim_silence(silence(0.5), thresh=100) == b""


def test_trim_silence_noise_below_threshold():
    rng = np.random.default_rng(0)
    noise = (rng.normal(0, 30, 8000)).astype(np.int16).tobytes()
    pcm = noise + tone(0.2) + noise
    out = audio.trim_silence(pcm, thresh=200)
    assert len(out) <= len(tone(0.2)) + 2 * FRAME_BYTES
    assert len(out) >= len(tone(0.2)) - FRAME_BYTES


def test_trim_silence_keeps_inner_silence():
    pcm = tone(0.2) + silence(0.3) + tone(0.2)
    out = audio.trim_silence(pcm, thresh=100)
    assert len(out) == pytest.approx(len(pcm), abs=FRAME_BYTES)


# ── resample ─────────────────────────────────────────
def test_resample_24k_to_16k_length():
    pcm24 = tone(1.0, sr=24000)
    out = audio.resample(pcm24, 24000, 16000)
    n_in = len(pcm24) // 2
    n_out = len(out) // 2
    assert n_out == pytest.approx(n_in * 2 / 3, abs=2)


def test_resample_preserves_frequency_and_level():
    pcm24 = tone(1.0, freq=1000, amp=0.5, sr=24000)
    out = audio.pcm_to_float(audio.resample(pcm24, 24000, 16000))
    spec = np.abs(np.fft.rfft(out))
    peak_hz = np.argmax(spec) * 16000 / len(out)
    assert peak_hz == pytest.approx(1000, abs=5)
    assert audio.rms(audio.float_to_pcm(out[1000:-1000])) == pytest.approx(0.5 * 32767 / np.sqrt(2), rel=0.03)


def test_resample_same_rate_is_identity():
    pcm = tone(0.1)
    assert audio.resample(pcm, 16000, 16000) == pcm


def test_resample_empty():
    assert audio.resample(b"", 24000, 16000) == b""


# ── wav ──────────────────────────────────────────────
def test_wav_roundtrip(tmp_path):
    pcm = tone(0.25)
    p = tmp_path / "a.wav"
    audio.wav_write(p, pcm)
    got, sr = audio.wav_read(p)
    assert sr == SAMPLE_RATE
    assert got == pcm


def test_wav_write_custom_rate(tmp_path):
    pcm = tone(0.1, sr=24000)
    p = tmp_path / "b.wav"
    audio.wav_write(p, pcm, rate=24000)
    got, sr = audio.wav_read(p)
    assert sr == 24000 and got == pcm


def test_wav_read_demo():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "DIRECT_ON 2026 Starter Kit/data/audio/demo/DEMO1.wav"
    pcm, sr = audio.wav_read(p)
    assert sr == 16000
    assert len(pcm) / 2 / sr == pytest.approx(8.3, abs=0.05)
