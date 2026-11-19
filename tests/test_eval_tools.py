"""AG-T06 테스트 음성 도구(증강·합성·조립) + 측정기 요약 + 채점기 테스트.

네트워크 금지: TTS·LLM 심사는 모두 fake를 주입한다.
"""
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from eval import augment as A
from eval import build_testset as B
from eval import runner as R
from eval import score as S
from eval import synth as Y

SR = 16000


# ───────────────────────── 신호 도우미 ─────────────────────────
def tone(sec, freq=440.0, amp=8000.0, sr=SR):
    t = np.arange(int(round(sec * sr))) / sr
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def silence(sec, sr=SR):
    return np.zeros(int(round(sec * sr)), dtype=np.float32)


def speechy(sec, seed=0, amp=6000.0):
    """음성 흉내: 여러 톤 + 잡음, 음절처럼 진폭 변조."""
    rng = np.random.default_rng(seed)
    n = int(round(sec * SR))
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f * t) for f in (180, 420, 900, 1800, 3000))
    x = x / 5 + 0.2 * rng.standard_normal(n)
    env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 4 * t))
    return (x * env * amp).astype(np.float32)


def dither(sec, seed=1, amp=15.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(round(sec * SR))) * amp).astype(np.float32)


def band_energy(x, lo, hi, sr=SR):
    spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return spec[(f >= lo) & (f < hi)].sum()


def dominant_freq(x, sr=SR):
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)]


def snr_db(clean, mixed, mask):
    noise = mixed.astype(np.float64) - clean.astype(np.float64)
    return 10 * np.log10(np.mean(clean[mask].astype(np.float64) ** 2) / np.mean(noise[mask] ** 2))


BASE = dict(lead_ms=0, tail_ms=0)


# ───────────────────────── 증강: 공통 ─────────────────────────
class TestEnv:
    def test_defaults_and_from_dict(self):
        e = A.Env.from_dict({"channel": "phone", "noise": "pink", "snr_db": 10})
        assert e.channel == "phone" and e.noise == "pink" and e.snr_db == 10
        assert e.speed == 1.0 and e.gain_db == 0.0 and e.packet_loss == 0.0

    @pytest.mark.parametrize("bad", [{"channel": "radio"}, {"noise": "rain"}, {"speed": 3.0},
                                     {"packet_loss": 1.5}])
    def test_invalid_values(self, bad):
        with pytest.raises(ValueError):
            A.Env.from_dict(bad)


@pytest.mark.parametrize("env", [
    {"channel": "wideband"}, {"channel": "phone"}, {"channel": "phone_ulaw"},
    {"noise": "white", "snr_db": 10}, {"noise": "pink", "snr_db": 10}, {"noise": "hum", "snr_db": 20},
    {"noise": "babble", "snr_db": 15}, {"gain_db": -12}, {"gain_db": 6}, {"packet_loss": 0.1},
])
def test_length_preserved_and_deterministic(env):
    x = np.concatenate([silence(0.2), speechy(0.8), silence(0.2)])
    pool = [speechy(0.7, seed=s) for s in range(2, 7)]
    e = {**BASE, **env}
    r1 = A.augment(x, e, seed=7, babble_pool=pool)
    r2 = A.augment(x, e, seed=7, babble_pool=pool)
    assert r1.audio.dtype == np.int16
    assert len(r1.audio) == len(x)
    assert np.array_equal(r1.audio, r2.audio)


@pytest.mark.parametrize("noise", ["white", "pink", "babble", "hum"])
def test_noise_seed_changes_output(noise):
    x = np.concatenate([silence(0.2), speechy(0.8)])
    pool = [speechy(0.7, seed=s) for s in range(2, 7)]
    e = {**BASE, "noise": noise, "snr_db": 10}
    a = A.augment(x, e, seed=1, babble_pool=pool).audio
    b = A.augment(x, e, seed=2, babble_pool=pool).audio
    assert not np.array_equal(a, b)


# ───────────────────────── 증강: 잡음·SNR ─────────────────────────
@pytest.mark.parametrize("noise", ["white", "pink", "babble", "hum"])
@pytest.mark.parametrize("target", [30, 20, 10, 5])
def test_snr_within_1db_on_speech_region(noise, target):
    lead, sp = 0.3, 1.0
    x = np.concatenate([silence(lead), speechy(sp), silence(0.3)])
    mask = np.zeros(len(x), bool)
    mask[int(lead * SR): int((lead + sp) * SR)] = True
    pool = [speechy(0.6, seed=s) for s in range(10, 16)]
    r = A.augment(x, {**BASE, "noise": noise, "snr_db": target}, seed=3, babble_pool=pool)
    got = snr_db(x, r.audio, mask)
    assert abs(got - target) <= 1.0, (noise, target, got)
    # 잡음은 음성 구간 밖(앞 무음)에도 깔린다
    assert np.abs(r.audio[: int(lead * SR)]).max() > 0


def test_noise_none_is_identity_without_other_changes():
    x = np.concatenate([silence(0.1), speechy(0.5)])
    r = A.augment(x, {**BASE}, seed=0)
    assert np.array_equal(r.audio, np.round(x).astype(np.int16))
    assert np.all(r.noise == 0)


def test_pink_noise_has_more_low_than_high_energy():
    n = A.make_noise("pink", SR * 2, np.random.default_rng(0))
    w = A.make_noise("white", SR * 2, np.random.default_rng(0))
    assert band_energy(n, 50, 500) / band_energy(n, 4000, 7500) > 5 * (
        band_energy(w, 50, 500) / band_energy(w, 4000, 7500))


def test_hum_noise_has_60hz_family():
    h = A.make_noise("hum", SR * 2, np.random.default_rng(0))
    spec_lo = band_energy(h, 55, 65) + band_energy(h, 115, 125) + band_energy(h, 175, 185)
    assert spec_lo / band_energy(h, 0, 8000) > 0.5


def test_babble_uses_injected_pool():
    pool = [tone(0.5, f, 5000) for f in (500, 1100, 1700, 2300, 2900)]
    n = A.make_noise("babble", SR * 2, np.random.default_rng(0), pool=pool)
    # 풀에 있는 주파수 근처 에너지가 대부분
    near = sum(band_energy(n, f - 30, f + 30) for f in (500, 1100, 1700, 2300, 2900))
    assert near / band_energy(n, 0, 8000) > 0.7
    # 여러 화자가 겹친다(최소 3개 주파수가 뚜렷)
    strong = [f for f in (500, 1100, 1700, 2300, 2900) if band_energy(n, f - 30, f + 30) > 0.03 * near]
    assert len(strong) >= 3


def test_babble_without_pool_falls_back_and_flags():
    x = np.concatenate([silence(0.1), speechy(0.5)])
    r = A.augment(x, {**BASE, "noise": "babble", "snr_db": 10}, seed=0, babble_pool=[])
    assert r.info.get("noise_fallback") is True
    assert len(r.audio) == len(x)


# ───────────────────────── 증강: 채널 ─────────────────────────
def test_phone_cuts_energy_above_3400hz():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 2) * 3000).astype(np.float32)
    for ch in ("phone", "phone_ulaw"):
        y = A.augment(x, {**BASE, "channel": ch}, seed=0).audio.astype(np.float32)
        hi_in, hi_out = band_energy(x, 3600, 8000), band_energy(y, 3600, 8000)
        assert 10 * np.log10(hi_out / hi_in) < -20, ch
        assert 10 * np.log10(band_energy(y, 4200, 8000) / band_energy(x, 4200, 8000)) < -40, ch
        # 통과 대역(500~3000Hz)은 대체로 유지(±3dB)
        mid = 10 * np.log10(band_energy(y, 500, 3000) / band_energy(x, 500, 3000))
        assert -3 < mid < 3, (ch, mid)
        # 저역(300Hz 미만)도 줄어든다
        assert band_energy(y, 20, 150) < 0.3 * band_energy(x, 20, 150), ch


def test_ulaw_codec_known_values_and_roundtrip():
    assert A.ulaw_encode(np.array([0], np.int16))[0] == 0xFF
    assert A.ulaw_encode(np.array([32767], np.int16))[0] == 0x80
    assert A.ulaw_encode(np.array([-32768], np.int16))[0] == 0x00
    assert A.ulaw_decode(np.array([0xFF], np.uint8))[0] == 0
    x = np.linspace(-30000, 30000, 2001).astype(np.int16)
    y = A.ulaw_decode(A.ulaw_encode(x))
    err = np.abs(y.astype(np.int32) - x.astype(np.int32))
    # μ-law 상대 오차 ≤ 약 3% (+ 작은 값 절대오차)
    assert np.all(err <= 0.035 * np.abs(x.astype(np.int32)) + 40)
    assert len(np.unique(A.ulaw_encode(x))) <= 256


def test_phone_ulaw_differs_from_phone_by_quantization_only():
    x = np.concatenate([silence(0.1), speechy(1.0, amp=4000)])
    p = A.augment(x, {**BASE, "channel": "phone"}, seed=0).audio.astype(np.float64)
    u = A.augment(x, {**BASE, "channel": "phone_ulaw"}, seed=0).audio.astype(np.float64)
    assert not np.array_equal(p, u)
    diff = u - p
    assert 10 * np.log10(np.mean(p ** 2) / np.mean(diff ** 2)) > 25   # 양자화 잡음은 작다


# ───────────────────────── 증강: 이득·패킷 손실 ─────────────────────────
def test_gain_minus12_lowers_level():
    x = speechy(0.5)
    y = A.augment(x, {**BASE, "gain_db": -12}, seed=0).audio.astype(np.float64)
    d = 20 * np.log10(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    assert abs(d + 12) < 0.2


def test_gain_plus6_clips_without_wraparound():
    x = tone(0.3, 300, 20000)
    r = A.augment(x, {**BASE, "gain_db": 6}, seed=0)
    y = r.audio.astype(np.int32)
    assert y.max() == 32767 and y.min() >= -32768
    nz = np.abs(x) > 100
    assert np.all(np.sign(y[nz]) == np.sign(x[nz]))        # 부호 뒤집힘(오버플로) 없음
    assert r.info["clipped_frac"] > 0.1


def test_packet_loss_zeroes_whole_20ms_frames():
    x = np.full(SR * 2, 5000, np.float32)
    r = A.augment(x, {**BASE, "packet_loss": 0.3}, seed=11)
    frames = r.audio.reshape(-1, 320)
    dropped = np.all(frames == 0, axis=1)
    kept = np.all(frames == 5000, axis=1)
    assert np.all(dropped | kept)                           # 프레임 중간에서 끊기지 않음
    assert 0.15 < dropped.mean() < 0.45
    r2 = A.augment(x, {**BASE, "packet_loss": 0.3}, seed=11)
    assert np.array_equal(r.audio, r2.audio)
    r3 = A.augment(x, {**BASE, "packet_loss": 0.3}, seed=12)
    assert not np.array_equal(r.audio, r3.audio)
    assert r.info["dropped_frames"] == int(dropped.sum())


def test_packet_loss_zero_drops_nothing():
    x = np.full(SR, 5000, np.float32)
    assert np.all(A.augment(x, {**BASE, "packet_loss": 0.0}, seed=0).audio == 5000)


# ───────────────────────── 증강: 속도·lead ─────────────────────────
@pytest.mark.parametrize("method", ["ffmpeg", "wsola"])
@pytest.mark.parametrize("rate", [0.85, 1.2])
def test_time_stretch_keeps_pitch(method, rate):
    if method == "ffmpeg" and not A.ffmpeg_available():
        pytest.skip("ffmpeg 없음")
    x = tone(1.0, 440, 8000)
    y = A.time_stretch(x, rate, method=method)
    assert abs(len(y) - len(x) / rate) <= 0.02 * len(x)
    mid = y[len(y) // 4: 3 * len(y) // 4]
    assert abs(dominant_freq(mid) - 440) < 10


def test_time_stretch_identity():
    x = speechy(0.3)
    assert np.array_equal(A.time_stretch(x, 1.0), x)


def test_lead_ms_prepends_silence_and_shifts_speech_end():
    x = np.concatenate([speechy(1.0), silence(0.1)])
    r0 = A.augment(x, {"lead_ms": 0, "tail_ms": 0}, seed=0)
    r4 = A.augment(x, {"lead_ms": 400, "tail_ms": 0}, seed=0)
    assert len(r4.audio) == len(r0.audio) + int(0.4 * SR)
    assert np.all(r4.audio[: int(0.4 * SR)] == 0)
    assert abs((r4.speech_end - r0.speech_end) - 0.4) < 0.002


def test_tail_ms_appends_silence():
    x = speechy(0.5)
    r = A.augment(x, {"lead_ms": 0, "tail_ms": 700}, seed=0)
    assert len(r.audio) == len(x) + int(0.7 * SR)


# ───────────────────────── speech_end ─────────────────────────
def test_detect_speech_end_with_lead_and_mid_pause():
    x = np.concatenate([dither(0.5, 1), speechy(1.0, 2), dither(0.4, 3), speechy(0.6, 4), dither(0.8, 5)])
    assert abs(A.detect_speech_end(x) - 2.5) <= 0.02


def test_detect_speech_end_none_for_silence():
    assert A.detect_speech_end(dither(1.0)) is None


def test_speech_end_from_clean_track_not_noisy_mix():
    """잡음은 끝까지 깔리지만 speech_end는 깨끗한 트랙 기준."""
    x = np.concatenate([speechy(1.0), silence(1.0)])
    r = A.augment(x, {"lead_ms": 300, "tail_ms": 500, "noise": "white", "snr_db": 5}, seed=0)
    assert abs(r.speech_end - 1.3) <= 0.02
    assert np.abs(r.audio[-1000:]).max() > 0                  # 꼬리에도 잡음


@pytest.mark.parametrize("speed", [0.85, 1.2])
def test_speech_end_after_speed_and_lead(speed):
    x = np.concatenate([speechy(0.8, 1), silence(0.5), speechy(0.7, 2)])     # 2.0초 음성
    r = A.augment(x, {"lead_ms": 400, "tail_ms": 300, "speed": speed}, seed=0)
    expected = 0.4 + 2.0 / speed
    assert abs(r.speech_end - expected) <= 0.04, (r.speech_end, expected)
    assert r.info["speed_method"] in ("ffmpeg", "wsola")


def test_normalize_level_sets_speech_rms_and_caps_peak():
    x = np.concatenate([silence(0.5), speechy(1.0, amp=20000)])
    y = A.normalize_level(x, target_rms=3000)
    seg = y[int(0.5 * SR):].astype(np.float64)
    assert abs(20 * np.log10(np.sqrt(np.mean(seg ** 2)) / 3000)) < 0.5
    spiky = np.concatenate([speechy(1.0, amp=1000), np.full(10, 30000, np.float32)])
    z = A.normalize_level(spiky, target_rms=3000, peak_limit=20000)
    assert np.abs(z).max() <= 20000 + 1


def test_trim_silence():
    x = np.concatenate([dither(0.3), speechy(0.5), dither(0.2)])
    y = A.trim_silence(x)
    assert abs(len(y) / SR - 0.5) <= 0.02


# ───────────────────────── 세그먼트 조립 ─────────────────────────
def fake_tts(text, voice=None, voice_style=None):
    """앞 200ms + 음성(글자 수 비례) + 뒤 150ms 무음."""
    return np.concatenate([dither(0.2), speechy(0.1 * len(text), seed=len(text)), dither(0.15)]
                          ).astype(np.int16)


def voiced_runs(x, thr=500, frame=160):
    f = x[: len(x) // frame * frame].astype(np.float64).reshape(-1, frame)
    v = np.sqrt((f ** 2).mean(1)) > thr
    runs, start = [], None
    for i, b in enumerate(v):
        if b and start is None:
            start = i
        if not b and start is not None:
            runs.append((start * frame / SR, i * frame / SR)); start = None
    if start is not None:
        runs.append((start * frame / SR, len(v) * frame / SR))
    return runs


def test_assemble_segments_exact_pauses():
    segs = [{"text": "주문번호가요", "pause_after_ms": 700},       # 6자 → 0.6초
            {"text": "일이삼", "pause_after_ms": 300},              # 3자 → 0.3초
            {"text": "언제와요", "pause_after_ms": 0}]              # 4자 → 0.4초
    y = A.assemble_segments(segs, lambda t: fake_tts(t))
    assert y.dtype == np.int16
    assert abs(len(y) / SR - (0.6 + 0.7 + 0.3 + 0.3 + 0.4)) <= 0.02
    runs = voiced_runs(y)
    assert len(runs) == 3
    assert abs((runs[1][0] - runs[0][1]) - 0.7) <= 0.02
    assert abs((runs[2][0] - runs[1][1]) - 0.3) <= 0.02


def test_assemble_single_segment_trimmed():
    y = A.assemble_segments([{"text": "네", "pause_after_ms": 500}], lambda t: fake_tts(t))
    assert abs(len(y) / SR - 0.1) <= 0.02            # 마지막 세그먼트 뒤 pause는 붙이지 않는다


# ───────────────────────── 합성(synth) ─────────────────────────
class FakeBackend:
    def __init__(self, rate=24000, fail_times=0, exc=None, delay=0.0):
        self.calls, self.rate, self.fail_times, self.exc, self.delay = [], rate, fail_times, exc, delay
        self.active = self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, text, name, *rest):
        with self.lock:
            self.calls.append((text, name, *rest))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if len(self.calls) <= self.fail_times:
                raise (self.exc or RuntimeError("429 RESOURCE_EXHAUSTED"))
            sec = 0.05 * len(text)
            t = np.arange(int(sec * self.rate)) / self.rate
            return (np.sin(2 * np.pi * 300 * t) * 6000).astype(np.int16), self.rate
        finally:
            with self.lock:
                self.active -= 1


def make_synth(tmp_path, **kw):
    g = kw.pop("gemini", FakeBackend())
    e = kw.pop("edge", FakeBackend(rate=16000))
    s = Y.Synthesizer(cache_dir=tmp_path / "cache", gemini_backend=g, edge_backend=e,
                      sleep=lambda s: None, **kw)
    return s, g, e


class TestSynth:
    def test_parse_voice(self):
        assert Y.parse_voice("gemini:Kore") == ("gemini", "Kore", None)
        assert Y.parse_voice("edge:ko-KR-SunHiNeural") == ("edge", "ko-KR-SunHiNeural", None)
        assert Y.parse_voice("gemini:Puck@gemini-3.8-flash-lite-tts") == ("gemini", "Puck", "gemini-3.8-flash-lite-tts")
        for bad in ("Kore", "azure:x", "gemini:", "edge:"):
            with pytest.raises(ValueError):
                Y.parse_voice(bad)

    def test_gemini_resampled_to_16k_and_cached(self, tmp_path):
        s, g, _ = make_synth(tmp_path)
        a = s.synthesize("안녕하세요", "gemini:Kore")
        assert a.dtype == np.int16
        assert abs(len(a) - int(0.25 * SR)) <= 2           # 24k 0.25초 → 16k 4000샘플
        files = list((tmp_path / "cache").glob("*.pcm"))
        assert len(files) == 1 and files[0].stat().st_size == len(a) * 2
        b = s.synthesize("안녕하세요", "gemini:Kore")
        assert np.array_equal(a, b) and len(g.calls) == 1
        # 새 인스턴스도 디스크 캐시를 쓴다
        s2, g2, _ = make_synth(tmp_path)
        s2.synthesize("안녕하세요", "gemini:Kore")
        assert len(g2.calls) == 0

    def test_cache_key_varies(self, tmp_path):
        s, g, e = make_synth(tmp_path)
        s.synthesize("네", "gemini:Kore")
        s.synthesize("네", "gemini:Puck")
        s.synthesize("네", "gemini:Kore", voice_style="급하게")
        s.synthesize("네", "edge:ko-KR-SunHiNeural")
        assert len(g.calls) == 3 and len(e.calls) == 1
        assert len(list((tmp_path / "cache").glob("*.pcm"))) == 4

    def test_style_passed_only_to_gemini(self, tmp_path):
        s, g, e = make_synth(tmp_path)
        s.synthesize("네", "gemini:Kore", voice_style="작은 목소리로")
        assert g.calls[0][2] == "작은 목소리로"
        s.synthesize("네", "edge:ko-KR-SunHiNeural", voice_style="작은 목소리로")
        s.synthesize("네", "edge:ko-KR-SunHiNeural")       # 스타일은 edge에서 무시 → 같은 캐시
        assert len(e.calls) == 1

    def test_retry_on_rate_limit(self, tmp_path):
        g = FakeBackend(fail_times=2)
        s, _, _ = make_synth(tmp_path, gemini=g, retries=4)
        a = s.synthesize("다시", "gemini:Kore")
        assert len(a) > 0 and len(g.calls) == 3

    def test_retry_gives_up(self, tmp_path):
        g = FakeBackend(fail_times=99)
        s, _, _ = make_synth(tmp_path, gemini=g, retries=2)
        with pytest.raises(RuntimeError):
            s.synthesize("실패", "gemini:Kore")
        assert len(g.calls) == 3

    def test_bad_request_not_retried(self, tmp_path):
        g = FakeBackend(fail_times=99, exc=RuntimeError("400 INVALID_ARGUMENT Unknown name speechConfig"))
        s, _, _ = make_synth(tmp_path, gemini=g, retries=4)
        with pytest.raises(RuntimeError):
            s.synthesize("x", "gemini:Kore")
        assert len(g.calls) == 1
        assert Y.is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED"))
        assert Y.is_retryable(RuntimeError("503 UNAVAILABLE"))
        assert not Y.is_retryable(RuntimeError("404 NOT_FOUND"))

    def test_suspiciously_long_output_is_regenerated(self, tmp_path):
        """말투 지시문까지 읽어 버린 듯한 과도한 길이는 다시 합성한다."""
        class LongOnce(FakeBackend):
            def __call__(self, text, name, *rest):
                pcm, rate = super().__call__(text, name, *rest)
                if len(self.calls) == 1:
                    return np.tile(pcm, 200), rate
                return pcm, rate
        g = LongOnce()
        s, _, _ = make_synth(tmp_path, gemini=g, retries=3)
        a = s.synthesize("네 알겠어요", "gemini:Kore", voice_style="천천히")
        assert len(g.calls) == 2 and len(a) / SR < 2
        assert Y.max_duration("주문 좀 취소해 주세요") < Y.max_duration("주문 좀 취소해 주세요", style="천천히")

    def test_bad_cached_entry_is_regenerated(self, tmp_path):
        s, g, _ = make_synth(tmp_path)
        p = s.cache_path("네 알겠어요", "gemini:Kore", "천천히")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.zeros(SR * 30, np.int16).tofile(p)                    # 30초짜리 잘못된 캐시
        a = s.synthesize("네 알겠어요", "gemini:Kore", voice_style="천천히")
        assert len(g.calls) == 1 and len(a) / SR < 2

    def test_max_duration_bounds(self):
        assert Y.max_duration("주문 좀 취소해 주세요.", style="노인 화자처럼 천천히") < 8.5   # 지시문 낭독(8.7초) 거부
        assert Y.max_duration("주문번호 이육일일일오공이삼일이요. 언제 와요?") > 7.0          # 정상 5.5초 허용

    def test_empty_audio_is_error(self, tmp_path):
        s, g, _ = make_synth(tmp_path, retries=1)
        with pytest.raises(RuntimeError):
            s.synthesize("", "gemini:Kore")

    def test_gemini_concurrency_limited(self, tmp_path):
        g = FakeBackend(delay=0.05)
        s, _, _ = make_synth(tmp_path, gemini=g, max_gemini_concurrency=2)
        ths = [threading.Thread(target=s.synthesize, args=(f"문장{i}", "gemini:Kore")) for i in range(6)]
        [t.start() for t in ths]; [t.join() for t in ths]
        assert len(g.calls) == 6 and g.max_active <= 2

    def test_gemini_backend_parses_response_with_fake_client(self):
        class Inline:  # inline_data
            def __init__(self):
                self.data = (np.ones(2400, np.int16) * 1000).tobytes()
                self.mime_type = "audio/L16;codec=pcm;rate=24000"

        class Resp:
            def __init__(self):
                part = type("P", (), {"inline_data": Inline()})()
                content = type("C", (), {"parts": [part]})()
                self.candidates = [type("Cand", (), {"content": content})()]

        seen = {}

        class Models:
            def generate_content(self, model, contents, config):
                seen.update(model=model, contents=contents, config=config)
                return Resp()

        client = type("Client", (), {"models": Models()})()
        be = Y.GeminiTTS(client=client)
        pcm, rate = be("주문 취소요", "Kore", "급하게", None)
        assert rate == 24000 and len(pcm) == 2400 and pcm.dtype == np.int16
        assert seen["model"] == Y.GEMINI_DEFAULT_MODEL
        assert "주문 취소요" in seen["contents"] and "급하게" in seen["contents"]
        pcm, _ = be("주문 취소요", "Kore", None, "gemini-2.5-flash-preview-tts")
        assert seen["model"] == "gemini-2.5-flash-preview-tts" and seen["contents"] == "주문 취소요"

    def test_gemini_backend_parses_wav_with_trailing_metadata(self):
        """실제 응답은 audio/wav: RIFF 헤더 + data + 뒤에 출처 메타데이터 청크가 붙는다."""
        import struct
        pcm = (np.arange(1200, dtype=np.int16) % 50 * 100).astype(np.int16).tobytes()
        fmt = struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
        trailer = b"c2pa" + struct.pack("<I", 12) + b"trainedAlgo!"
        body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm + trailer
        wav = b"RIFF" + struct.pack("<I", len(body)) + body

        class Resp:
            def __init__(self):
                inline = type("I", (), {"data": wav, "mime_type": "audio/wav"})()
                part = type("P", (), {"inline_data": inline})()
                self.candidates = [type("Cand", (), {"content": type("C", (), {"parts": [part]})()})()]

        client = type("Client", (), {"models": type("M", (), {
            "generate_content": lambda self, model, contents, config: Resp()})()})()
        out, rate = Y.GeminiTTS(client=client)("네", "Kore", None, None)
        assert rate == 24000 and out.tobytes() == pcm

    def test_daily_quota_falls_back_to_next_model(self):
        quota = ("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier "
                 "'retryDelay': '83659s'")
        assert not Y.is_retryable(RuntimeError(quota))
        assert not Y.is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED 'retryDelay': '600s'"))
        assert Y.is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED 'retryDelay': '20s'"))
        used = []
        ok = type("I", (), {"data": np.zeros(240, np.int16).tobytes(), "mime_type": "audio/L16;rate=24000"})()
        resp = type("R", (), {"candidates": [type("Cand", (), {"content": type("C", (), {
            "parts": [type("P", (), {"inline_data": ok})()]})()})()]})()

        class Models:
            def generate_content(self, model, contents, config):
                used.append(model)
                if model == Y.GEMINI_DEFAULT_MODEL:
                    raise RuntimeError(quota)
                return resp

        be = Y.GeminiTTS(client=type("Cl", (), {"models": Models()})())
        pcm, rate = be("네", "Kore", None, None)
        assert used == [Y.GEMINI_DEFAULT_MODEL, Y.GEMINI_FALLBACK_MODELS[0]] and len(pcm) == 240
        be2 = Y.GeminiTTS(client=type("Cl", (), {"models": Models()})())
        with pytest.raises(RuntimeError):                  # 모델을 명시하면 대체하지 않는다
            be2("네", "Kore", None, Y.GEMINI_DEFAULT_MODEL)

    def test_styled_request_skips_lite_model(self):
        """flash-lite-tts는 말투 지시문을 소리 내어 읽는다(실측) → 말투 지정 시 건너뛴다."""
        used = []
        ok = type("I", (), {"data": np.zeros(240, np.int16).tobytes(), "mime_type": "audio/L16;rate=24000"})()
        resp = type("R", (), {"candidates": [type("Cand", (), {"content": type("C", (), {
            "parts": [type("P", (), {"inline_data": ok})()]})()})()]})()

        class Models:
            def generate_content(self, model, contents, config):
                used.append(model)
                if model == Y.GEMINI_DEFAULT_MODEL:
                    raise RuntimeError("429 RESOURCE_EXHAUSTED PerDay")
                return resp

        be = Y.GeminiTTS(client=type("Cl", (), {"models": Models()})())
        be("네", "Kore", "천천히", None)
        assert used == [Y.GEMINI_DEFAULT_MODEL, "gemini-2.5-flash-preview-tts"]

    def test_style_prompt_separates_notes_from_transcript(self):
        """AG-T15d: 'Say in this style (X): "..."'는 3.8-flash-tts가 지시문까지 낭독했다.
        말투는 연출 메모 칸에만, 대본 칸에는 발화 원문만(따옴표·괄호·지시 없이) 둔다."""
        text, style = "계속 이런 식이면 소비자원에 신고할 거예요.", "단호하고 화난 목소리로"
        p = Y.style_prompt(text, style)
        notes, sep, transcript = p.partition("### TRANSCRIPT\n")
        assert sep and transcript == text                  # 대본 칸 = 원문 그대로
        assert style in notes and style not in transcript
        assert "Say in this style" not in p and f'"{text}"' not in p and f"({style})" not in p
        assert "Do not read" in notes
        assert p.count(text) == 1
        assert Y.style_prompt(text, None) == text          # 말투 없으면 원문만
        assert Y.STYLE_PROMPT_VERSION >= 3                 # 형식 변경 → 말투 지정 캐시 무효화

    def test_style_prompt_version_changes_only_styled_cache_key(self, monkeypatch):
        k_plain = Y.Synthesizer.cache_key("네", "gemini:Kore")
        k_style = Y.Synthesizer.cache_key("네", "gemini:Kore", "화난 목소리로")
        monkeypatch.setattr(Y, "STYLE_PROMPT_VERSION", Y.STYLE_PROMPT_VERSION + 1)
        assert Y.Synthesizer.cache_key("네", "gemini:Kore") == k_plain
        assert Y.Synthesizer.cache_key("네", "gemini:Kore", "화난 목소리로") != k_style

    def test_gemini_backend_sends_separated_prompt(self):
        sent = []
        ok = type("I", (), {"data": np.zeros(240, np.int16).tobytes(), "mime_type": "audio/L16;rate=24000"})()
        resp = type("R", (), {"candidates": [type("Cand", (), {"content": type("C", (), {
            "parts": [type("P", (), {"inline_data": ok})()]})()})()]})()

        class Models:
            def generate_content(self, model, contents, config):
                sent.append(contents)
                return resp

        be = Y.GeminiTTS(client=type("Cl", (), {"models": Models()})())
        be("사람 바꿔 줘.", "Aoede", "짜증 섞인 목소리로", None)
        assert sent == [Y.style_prompt("사람 바꿔 줘.", "짜증 섞인 목소리로")]
        assert sent[0].endswith("### TRANSCRIPT\n사람 바꿔 줘.")

    def test_daily_quota_detected_from_per_day_metric(self):
        e = RuntimeError("429 RESOURCE_EXHAUSTED Quota exceeded for metric: "
                         "generativelanguage.googleapis.com/generate_requests_per_model_per_day, limit: 100")
        assert Y.is_daily_quota(e) and not Y.is_retryable(e)

    def test_edge_backend_with_fake_communicate(self):
        class FakeComm:
            def __init__(self, text, voice, **kw):
                self.text, self.voice = text, voice

            async def stream(self):
                yield {"type": "WordBoundary"}
                yield {"type": "audio", "data": b"AB"}
                yield {"type": "audio", "data": b"CD"}

        got = {}

        def decoder(mp3):
            got["mp3"] = mp3
            return np.zeros(160, np.int16), 16000

        be = Y.EdgeTTS(communicate_cls=FakeComm, decoder=decoder)
        pcm, rate = be("네", "ko-KR-SunHiNeural")
        assert got["mp3"] == b"ABCD" and rate == 16000 and len(pcm) == 160


# ───────────────────────── build_testset ─────────────────────────
class FakeSynth:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice, voice_style=None):
        self.calls.append((text, voice, voice_style))
        return fake_tts(text)


def write_cases(tmp_path, cases):
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")
    return p


CASES = [
    {"case_id": "T1", "now": "2026-11-21 11:00", "history": [], "customer_profile": None,
     "customer_utterance": "주문번호가요 일이삼",
     "segments": [{"text": "주문번호가요", "pause_after_ms": 700}, {"text": "일이삼", "pause_after_ms": 0}],
     "voice": "gemini:Kore",
     "env": {"channel": "phone_ulaw", "noise": "babble", "snr_db": 15, "gain_db": -6, "speed": 1.0,
             "lead_ms": 400, "packet_loss": 0.0},
     "category": "tool_lookup", "tags": ["mid_pause"]},
    {"case_id": "T2", "now": "2026-11-21 11:00", "history": [], "customer_profile": None,
     "customer_utterance": "영업시간이요", "voice": "edge:ko-KR-SunHiNeural", "voice_style": "급하게",
     "env": {"channel": "wideband", "noise": "none", "lead_ms": 200},
     "category": "faq", "tags": ["short"]},
    {"case_id": "T3", "now": "2026-11-21 11:00", "history": [], "customer_profile": None,
     "customer_utterance": "반품하고 싶어요", "voice": "gemini:Puck",
     "env": {"channel": "phone", "noise": "white", "snr_db": 20, "lead_ms": 300, "speed": 1.2},
     "category": "tool_action", "tags": []},
]


def read_wav(p):
    with wave.open(str(p)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


class TestBuild:
    def test_build_writes_wav_and_meta(self, tmp_path):
        out = tmp_path / "audio"
        synth = FakeSynth()
        meta = B.build(B.load_cases(write_cases(tmp_path, CASES)), out, synth=synth, workers=2)
        assert set(meta) == {"T1", "T2", "T3"}
        disk = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert set(disk) == {"T1", "T2", "T3"}
        # T1: lead 0.4 + 0.6 + pause 0.7 + 0.3 = 2.0
        assert abs(disk["T1"]["speech_end"] - 2.0) <= 0.03
        # T2: 세그먼트 없음 → customer_utterance 한 덩어리(6자=0.6초), lead 0.2
        assert abs(disk["T2"]["speech_end"] - 0.8) <= 0.03
        # T3: speed 1.2 → 0.8/1.2 + lead 0.3
        assert abs(disk["T3"]["speech_end"] - (0.3 + 0.8 / 1.2)) <= 0.04
        for cid in ("T1", "T2", "T3"):
            a = read_wav(out / f"{cid}.wav")
            assert len(a) / SR > disk[cid]["speech_end"]
        assert ("영업시간이요", "edge:ko-KR-SunHiNeural", "급하게") in synth.calls

    def test_build_is_deterministic(self, tmp_path):
        cases = B.load_cases(write_cases(tmp_path, CASES))
        B.build(cases, tmp_path / "a", synth=FakeSynth(), workers=1)
        B.build(cases, tmp_path / "b", synth=FakeSynth(), workers=3)
        for cid in ("T1", "T2", "T3"):
            assert np.array_equal(read_wav(tmp_path / "a" / f"{cid}.wav"), read_wav(tmp_path / "b" / f"{cid}.wav"))

    def test_skip_existing_force_and_only(self, tmp_path):
        cases = B.load_cases(write_cases(tmp_path, CASES))
        out = tmp_path / "audio"
        B.build(cases, out, synth=FakeSynth())
        s2 = FakeSynth()
        B.build(cases, out, synth=s2)
        assert s2.calls == []
        s3 = FakeSynth()
        B.build(cases, out, synth=s3, only={"T2"}, force=True)
        assert {c[0] for c in s3.calls} == {"영업시간이요"}
        assert set(json.loads((out / "meta.json").read_text(encoding="utf-8"))) == {"T1", "T2", "T3"}

    def test_one_case_failure_does_not_abort_build(self, tmp_path):
        class Flaky(FakeSynth):
            def synthesize(self, text, voice, voice_style=None):
                if text == "영업시간이요":
                    raise RuntimeError("합성 실패")
                return super().synthesize(text, voice, voice_style)
        out = tmp_path / "audio"
        meta = B.build(B.load_cases(write_cases(tmp_path, CASES)), out, synth=Flaky())
        assert set(meta) == {"T1", "T3"}
        assert not (out / "T2.wav").exists()

    def test_cli_parses(self, tmp_path):
        p = write_cases(tmp_path, CASES)
        a = B.parse_args(["--cases", str(p), "--out", str(tmp_path / "o"), "--only", "T1,T3",
                          "--workers", "3", "--force"])
        assert a.only == "T1,T3" and a.workers == 3 and a.force


# ───────────────────────── runner (요약 함수) ─────────────────────────
def frame(amp):
    return (np.ones(320, np.int16) * amp).tobytes()


class TestRunner:
    def test_summarize_basic(self):
        frames = [(-0.2, frame(0)), (0.3, frame(100)), (0.9, frame(2000)), (1.0, frame(2000)), (2.5, frame(3000))]
        events = [{"t": 0.85, "type": "agent.transcript", "text": "네, 고객님."},
                  {"t": 2.4, "type": "agent.transcript", "text": "배송중입니다."},
                  {"t": 3.0, "type": "turn.end"}]
        wall_se = 1000.0
        calls = [{"name": "get_order_info", "arguments": {"order_id": "1"}, "matched": True, "t_response": 1002.0}]
        out = R.summarize(frames, events, calls=calls, wall_speech_end=wall_se)
        assert out["first_sound"] == pytest.approx(0.9)
        assert out["spoke_before_end"] is False
        assert out["turn_end"] == pytest.approx(3.0)
        assert out["agent_text"] == "네, 고객님. 배송중입니다."
        assert out["events"] == events
        assert out["tool_calls"][0]["t_response"] == pytest.approx(2.0)
        assert out["tool_calls"][0]["matched"] is True
        assert out["after_tool_sound"] == pytest.approx(0.5)
        assert out["agent_audio_sec"] == pytest.approx(5 * 0.02)

    def test_summarize_spoke_before_end_and_silence(self):
        out = R.summarize([(-0.1, frame(1000))], [], calls=[], wall_speech_end=0)
        assert out["spoke_before_end"] is True
        out = R.summarize([(0.5, frame(10))], [], calls=[], wall_speech_end=0)
        assert out["first_sound"] is None and out["spoke_before_end"] is False
        assert out["turn_end"] is None and "after_tool_sound" not in out

    def test_attach_case_fields(self):
        row = {"case_id": "T1", "first_sound": 1.0}
        R.attach_case(row, {**CASES[0], "expected_tools": [{"name": "x", "arguments": {}}],
                            "expected_points": ["a"], "must_not": ["b"]})
        for k in ("category", "tags", "env", "expected_tools", "expected_points", "must_not",
                  "customer_utterance", "voice"):
            assert k in row
        assert row["first_sound"] == 1.0

    def test_cli_args(self):
        a = R.parse_args(["--server", "ws://x", "--cases", "c", "--audio", "a", "--out", "o",
                          "--concurrency", "2", "--limit", "3"])
        assert a.concurrency == 2 and a.limit == 3 and a.mock == "http://localhost:8100"


# ───────────────────────── score ─────────────────────────
class TestSpeed:
    @pytest.mark.parametrize("t,pts", [(0.0, 100), (1.5, 100), (1.51, 90), (2.0, 90), (2.5, 80), (3.0, 70),
                                       (3.5, 60), (4.0, 50), (4.5, 30), (5.0, 30), (6.0, 10), (6.01, 0),
                                       (None, 0), (-0.1, 0), (float("nan"), 0)])
    def test_table(self, t, pts):
        assert S.speed_points(t) == pts

    def test_row_zero_conditions(self):
        ok = {"case_id": "a", "first_sound": 1.0, "turn_end": 5.0, "spoke_before_end": False}
        assert S.speed_score(ok) == 100
        assert S.speed_score({**ok, "spoke_before_end": True}) == 0
        assert S.speed_score({**ok, "error": "x"}) == 0
        assert S.speed_score({**ok, "turn_end": None}) == 0
        assert S.speed_score({**ok, "turn_end": 25.5}) == 0
        assert S.speed_score({**ok, "first_sound": None}) == 0


def call(name, **args):
    return {"name": name, "arguments": args}


class TestToolMatch:
    def test_exact(self):
        m = S.tool_match([call("get_order_info", order_id="2611")], [call("get_order_info", order_id="2611")])
        assert m["score"] == 100 and m["exact"]

    def test_wrong_arg(self):
        m = S.tool_match([call("get_order_info", order_id="2611")], [call("get_order_info", order_id="2612")])
        assert m["score"] == 0 and not m["exact"] and m["name_match"]

    def test_expected_none_but_called(self):
        m = S.tool_match([], [call("cancel_order", order_id="1")])
        assert m["score"] == 0 and not m["exact"]
        assert S.tool_match([], [])["score"] == 100

    def test_missing_call(self):
        assert S.tool_match([call("a", x=1)], [])["score"] == 0

    def test_partial_and_extra(self):
        exp = [call("a", x=1), call("b", y=2)]
        act = [call("a", x=1), call("c", z=3)]
        m = S.tool_match(exp, act)
        assert m["score"] == pytest.approx(100 / 3)   # 1 일치 / (기대 2 + 여분 1)
        assert len(m["missing"]) == 1 and len(m["extra"]) == 1

    def test_duplicate_identical_calls_not_penalized(self):
        m = S.tool_match([call("a", x=1)], [call("a", x=1), call("a", x=1)])
        assert m["exact"]

    def test_free_text_excluded_by_option(self):
        exp = [call("cancel_order", order_id="1", reason="단순 변심")]
        act = [call("cancel_order", order_id="1", reason="마음이 바뀌어서")]
        assert S.tool_match(exp, act)["exact"]
        assert not S.tool_match(exp, act, exclude_free_text=False)["exact"]
        assert {"reason", "new_address", "preferred_time"} <= S.FREE_TEXT_KEYS

    def test_extra_non_free_arg_mismatch(self):
        assert not S.tool_match([call("a", x=1)], [call("a", x=1, y=2)])["exact"]


def rows_fixture():
    return [
        {"case_id": "A", "first_sound": 1.2, "turn_end": 4.0, "spoke_before_end": False,
         "category": "faq", "tags": ["short"], "env": {"channel": "phone", "noise": "none"},
         "expected_tools": [], "tool_calls": [], "agent_text": "10시에 엽니다",
         "expected_points": ["영업시간 안내"], "must_not": []},
        {"case_id": "B", "first_sound": 2.2, "turn_end": 6.0, "spoke_before_end": False,
         "category": "tool_lookup", "tags": ["digits", "mid_pause"],
         "env": {"channel": "phone_ulaw", "noise": "babble"},
         "expected_tools": [call("get_order_info", order_id="1")],
         "tool_calls": [{**call("get_order_info", order_id="1"), "matched": True, "t_response": 1.0}],
         "agent_text": "배송중", "expected_points": ["배송 상태"], "must_not": ["취소"]},
        {"case_id": "C", "first_sound": -0.3, "turn_end": 3.0, "spoke_before_end": True,
         "category": "tool_lookup", "tags": ["digits"], "env": {"channel": "wideband", "noise": "white"},
         "expected_tools": [call("get_order_info", order_id="9")],
         "tool_calls": [call("get_order_info", order_id="8")], "agent_text": "",
         "expected_points": ["배송 상태"], "must_not": []},
    ]


class TestScoreRows:
    def test_scores_and_aggregate(self):
        sc = S.score_rows(rows_fixture())
        by = {r["case_id"]: r["scores"] for r in sc}
        assert by["A"]["speed"] == 100 and by["A"]["tool"] == 100
        assert by["B"]["speed"] == 80 and by["B"]["tool"] == 100
        assert by["C"]["speed"] == 0 and by["C"]["tool"] == 0
        assert by["A"]["judge"] is None
        tags = S.aggregate(sc, "tags")
        assert tags["digits"]["n"] == 2 and tags["digits"]["speed"] == pytest.approx(40)
        assert tags["short"]["n"] == 1
        cats = S.aggregate(sc, "category")
        assert cats["tool_lookup"]["n"] == 2 and cats["tool_lookup"]["tool_exact"] == pytest.approx(0.5)
        assert set(S.aggregate(sc, "channel")) == {"phone", "phone_ulaw", "wideband"}
        assert S.aggregate(sc, "noise")["babble"]["n"] == 1
        summ = S.summarize(sc)
        assert summ["n"] == 3 and summ["speed"] == pytest.approx(60)
        assert summ["spoke_before_end"] == 1 and summ["tool_exact"] == pytest.approx(2 / 3)

    def test_judge_injected(self):
        seen = []

        def judge(row):
            seen.append(row["case_id"])
            if row["case_id"] == "B":
                return {"points": [True], "violations": [True], "comment": "취소 언급"}
            return {"points": [True], "violations": [False] * len(row.get("must_not") or [])}

        sc = S.score_rows(rows_fixture(), judge_fn=judge)
        by = {r["case_id"]: r["scores"] for r in sc}
        assert by["A"]["judge"] == 100 and by["B"]["judge"] == 0
        assert sorted(seen) == ["A", "B", "C"]
        # 총점 추정 = 응대 50% + 함수 20% + 속도 30%
        assert by["A"]["total"] == pytest.approx(100)
        assert by["B"]["total"] == pytest.approx(0.2 * 100 + 0.3 * 80)

    def test_judge_partial_points(self):
        sc = S.score_rows([{**rows_fixture()[0], "expected_points": ["a", "b"]}],
                          judge_fn=lambda r: {"points": [True, False], "violations": []})
        assert sc[0]["scores"]["judge"] == 50

    def test_gemini_judge_retries_503_with_fake_client(self):
        n = {"calls": 0}

        class Models:
            def generate_content(self, model, contents, config):
                n["calls"] += 1
                if n["calls"] == 1:
                    raise RuntimeError("503 UNAVAILABLE high demand")
                assert "10시에 엽니다" in contents and "영업시간 안내" in contents
                return type("R", (), {"text": '{"points": [true], "violations": [], "comment": "ok"}'})()

        judge = S.make_gemini_judge(client=type("C", (), {"models": Models()})(), sleep=lambda s: None)
        sc = S.score_rows(rows_fixture()[:1], judge_fn=judge)
        assert sc[0]["scores"]["judge"] == 100 and n["calls"] == 2

    def test_judge_error_is_recorded(self):
        def boom(row):
            raise RuntimeError("quota")
        sc = S.score_rows(rows_fixture()[:1], judge_fn=boom)
        assert sc[0]["scores"]["judge"] is None and "quota" in sc[0]["scores"]["judge_error"]

    def test_score_run_writes_report(self, tmp_path, capsys):
        run = tmp_path / "run"
        run.mkdir()
        (run / "results.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_fixture()), encoding="utf-8")
        S.main(["--run", str(run)])
        rep = (run / "report.md").read_text(encoding="utf-8")
        for word in ("속도", "함수", "tool_lookup", "phone_ulaw", "babble", "digits"):
            assert word in rep
        assert "C" in rep                                   # 실패 문항 목록
        out = capsys.readouterr().out
        assert "속도" in out and "3" in out

    def test_score_run_merges_cases_when_missing(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        bare = [{"case_id": "T1", "first_sound": 1.0, "turn_end": 2.0, "spoke_before_end": False,
                 "tool_calls": []}]
        (run / "results.jsonl").write_text(json.dumps(bare[0]), encoding="utf-8")
        cases = write_cases(tmp_path, [{**CASES[0], "expected_tools": []}])
        sc, _ = S.score_run(run, cases_path=cases)
        assert sc[0]["category"] == "tool_lookup" and sc[0]["scores"]["tool"] == 100
