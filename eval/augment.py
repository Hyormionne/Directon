"""통화 환경 증강 + speech_end 계산 + 세그먼트 조립 (docs/AGENT_PLAN.md 3.3).

내부 표현은 float32, int16 눈금(-32768..32767). 16kHz mono 고정.

처리 순서 (augment):
  1. speed: 피치 유지 시간 늘이기(ffmpeg atempo, 없으면 WSOLA)
  2. lead_ms / tail_ms 무음 붙이기 → 깨끗한 트랙 확정
  3. speech_end = 깨끗한 트랙의 마지막 음성 샘플 위치(잡음 섞기 전)
  4. noise: 깨끗한 트랙의 음성 구간 전력 기준으로 snr_db에 맞춰 섞는다(전 구간에 깔림)
  5. channel: phone = 8kHz + 300~3400Hz 대역, phone_ulaw = phone + G.711 μ-law 왕복
  6. gain_db: 곱한 뒤 int16 범위로 하드 클리핑(랩어라운드 없음)
  7. packet_loss: 20ms 프레임 단위 무작위 무음화
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field, fields
from typing import Callable, Sequence

import numpy as np
import soxr
from scipy.signal import butter, sosfiltfilt

SR = 16000
PACKET = 320                  # 20ms
DETECT_FRAME = 160            # 10ms — speech_end 검출 프레임
ABS_FLOOR_RMS = 50.0          # 이보다 작은 프레임은 무조건 무음
REL_THRESH_DB = -35.0         # 가장 큰 프레임 대비 이보다 작으면 무음

CHANNELS = ("wideband", "phone", "phone_ulaw")
NOISES = ("none", "white", "pink", "babble", "hum")


# ───────────────────────── 환경 ─────────────────────────
@dataclass
class Env:
    channel: str = "wideband"
    noise: str = "none"
    snr_db: float = 20.0
    gain_db: float = 0.0
    packet_loss: float = 0.0
    speed: float = 1.0
    lead_ms: int = 0
    tail_ms: int = 1500
    unknown: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | "Env" | None) -> "Env":
        if isinstance(d, Env):
            return d
        d = dict(d or {})
        known = {f.name for f in fields(cls)} - {"unknown"}
        kw = {k: v for k, v in d.items() if k in known and v is not None}
        env = cls(**kw)
        env.unknown = {k: v for k, v in d.items() if k not in known and k != "seed"}
        env.validate()
        return env

    def validate(self):
        if self.channel not in CHANNELS:
            raise ValueError(f"channel={self.channel!r} (허용: {CHANNELS})")
        if self.noise not in NOISES:
            raise ValueError(f"noise={self.noise!r} (허용: {NOISES})")
        if not 0.5 <= float(self.speed) <= 2.0:
            raise ValueError(f"speed={self.speed} (0.5~2.0)")
        if not 0.0 <= float(self.packet_loss) < 1.0:
            raise ValueError(f"packet_loss={self.packet_loss} (0~1)")
        if self.lead_ms < 0 or self.tail_ms < 0:
            raise ValueError("lead_ms/tail_ms는 0 이상")


@dataclass
class AugResult:
    audio: np.ndarray            # int16, 최종 음성
    speech_end: float | None     # 초, 깨끗한 트랙 기준
    clean: np.ndarray            # float32, 최종 길이에 맞춘 깨끗한 트랙(speed·lead·tail 반영)
    noise: np.ndarray            # float32, 섞인 잡음(채널 전, 이득 전)
    info: dict


# ───────────────────────── 음성 구간 검출 ─────────────────────────
def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = int(np.ceil(len(x) / frame))
    pad = np.zeros(n * frame, np.float64)
    pad[: len(x)] = x
    f = pad.reshape(n, frame)
    return np.sqrt((f * f).mean(axis=1))


def _threshold(rms: np.ndarray) -> float:
    peak = float(rms.max()) if len(rms) else 0.0
    return max(ABS_FLOOR_RMS, peak * 10 ** (REL_THRESH_DB / 20))


def _voiced_frames(x: np.ndarray, frame: int = DETECT_FRAME):
    rms = _frame_rms(np.asarray(x, np.float64), frame)
    thr = _threshold(rms)
    return rms > thr, thr


def speech_bounds(x: np.ndarray, sr: int = SR) -> tuple[int, int] | None:
    """[첫 음성 샘플, 마지막 음성 샘플+1]. 음성이 없으면 None."""
    x = np.asarray(x, np.float64)
    voiced, thr = _voiced_frames(x)
    idx = np.flatnonzero(voiced)
    if len(idx) == 0:
        return None
    fr = DETECT_FRAME
    a0 = max(0, (idx[0] - 1) * fr)
    a1 = min(len(x), (idx[0] + 1) * fr)
    over = np.flatnonzero(np.abs(x[a0:a1]) > thr)
    start = a0 + (over[0] if len(over) else fr)
    b0 = idx[-1] * fr
    b1 = min(len(x), (idx[-1] + 2) * fr)
    over = np.flatnonzero(np.abs(x[b0:b1]) > thr)
    end = b0 + (over[-1] + 1 if len(over) else fr)
    return int(start), int(min(end, len(x)))


def detect_speech_end(x: np.ndarray, sr: int = SR) -> float | None:
    """마지막 음성 샘플 끝 시각(초). 음성이 없으면 None."""
    b = speech_bounds(x, sr)
    return None if b is None else b[1] / sr


def speech_mask(x: np.ndarray, frame: int = PACKET) -> np.ndarray:
    """프레임 단위 음성 구간 마스크(샘플 길이)."""
    voiced, _ = _voiced_frames(x, frame)
    return np.repeat(voiced, frame)[: len(x)]


def trim_silence(x: np.ndarray) -> np.ndarray:
    b = speech_bounds(x)
    if b is None:
        return np.asarray(x)[:0]
    return np.asarray(x)[b[0]: b[1]]


def normalize_level(x: np.ndarray, target_rms: float = 3000.0, peak_limit: float = 30000.0) -> np.ndarray:
    """음성 구간 RMS를 target_rms(기본 약 -21dBFS)로 맞춘다. 피크가 peak_limit를 넘으면 그만큼 낮춘다.
    엔진마다 출력 레벨이 달라(Gemini는 거의 0dBFS, edge는 약 -4dBFS) gain_db 의미를 통일하려는 것."""
    x = np.asarray(x, np.float32)
    m = speech_mask(x)
    if not m.any():
        return x
    r = float(np.sqrt(np.mean(x[m].astype(np.float64) ** 2)))
    g = target_rms / r if r > 0 else 1.0
    peak = float(np.abs(x).max()) * g
    if peak > peak_limit:
        g *= peak_limit / peak
    return (x * np.float32(g)).astype(np.float32)


# ───────────────────────── 세그먼트 조립 ─────────────────────────
def assemble_segments(segments: Sequence[dict], synth_fn: Callable[[str], np.ndarray]) -> np.ndarray:
    """세그먼트를 각각 합성 → 앞뒤 무음 trim → pause_after_ms 무음을 정확히 끼워 잇는다.
    마지막 세그먼트의 pause는 붙이지 않는다(뒤 무음은 augment의 tail_ms가 맡는다)."""
    parts = []
    for i, seg in enumerate(segments):
        a = np.asarray(synth_fn(seg["text"]))
        t = trim_silence(a.astype(np.float32))
        if len(t) == 0:
            raise RuntimeError(f"세그먼트 {i} 합성 결과에 음성이 없다: {seg['text']!r}")
        parts.append(t)
        if i < len(segments) - 1:
            parts.append(np.zeros(int(round(int(seg.get("pause_after_ms", 0) or 0) * SR / 1000)), np.float32))
    y = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


# ───────────────────────── 속도 ─────────────────────────
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _stretch_ffmpeg(x: np.ndarray, rate: float) -> np.ndarray:
    pcm = np.clip(np.round(x), -32768, 32767).astype(np.int16).tobytes()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1",
           "-i", "pipe:0", "-filter:a", f"atempo={rate:.6f}", "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1"]
    p = subprocess.run(cmd, input=pcm, capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.int16).astype(np.float32)


def _stretch_wsola(x: np.ndarray, rate: float, win: int = 640, tol: int = 160) -> np.ndarray:
    """간단 WSOLA(40ms 창, 50% 겹침, ±10ms 탐색). 피치 유지."""
    x = np.asarray(x, np.float32)
    hop = win // 2
    n_out = int(round(len(x) / rate))
    w = np.hanning(win).astype(np.float32)
    xp = np.concatenate([np.zeros(tol, np.float32), x, np.zeros(win + tol + int(hop * rate) + 2, np.float32)])
    out = np.zeros(n_out + win, np.float32)
    wsum = np.zeros(n_out + win, np.float32)
    delta = 0
    k = 0
    while k * hop < n_out:
        ana = tol + int(round(k * hop * rate)) + delta
        seg = xp[ana: ana + win]
        if len(seg) < win:
            break
        o = k * hop
        out[o: o + win] += seg * w
        wsum[o: o + win] += w
        nat = xp[ana + hop: ana + hop + win]          # 자연스러운 다음 조각
        nxt = tol + int(round((k + 1) * hop * rate))
        lo, hi = max(0, nxt - tol), nxt + tol
        region = xp[lo: hi + win]
        if len(region) >= win and len(nat) == win:
            corr = np.correlate(region, nat, mode="valid")
            delta = lo + int(np.argmax(corr)) - nxt
        else:
            delta = 0
        k += 1
    out = out[:n_out] / np.maximum(wsum[:n_out], 1e-3)
    return out


def time_stretch(x: np.ndarray, rate: float, method: str = "auto") -> np.ndarray:
    """rate>1이면 빨라진다(길이 1/rate). method: auto|ffmpeg|wsola."""
    if rate == 1.0:
        return x
    if method == "auto":
        method = "ffmpeg" if ffmpeg_available() else "wsola"
    if method == "ffmpeg":
        return _stretch_ffmpeg(x, rate)
    if method == "wsola":
        return _stretch_wsola(x, rate)
    raise ValueError(method)


# ───────────────────────── 잡음 ─────────────────────────
def _unit(x: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))) if len(x) else 0.0
    return (x / r).astype(np.float32) if r > 0 else np.asarray(x, np.float32)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1] if len(f) > 1 else 1.0
    return np.fft.irfft(spec / np.sqrt(f), n)


def _speech_shaped(n: int, rng: np.random.Generator) -> np.ndarray:
    """풀이 없을 때 babble 대용: 음성 대역 핑크 잡음 + 음절 리듬 변조."""
    sos = butter(4, [200, 4000], btype="band", fs=SR, output="sos")
    x = sosfiltfilt(sos, _pink(n, rng))
    t = np.arange(n) / SR
    mod = np.ones(n)
    for _ in range(4):
        mod += 0.5 * np.abs(np.sin(2 * np.pi * rng.uniform(2.5, 5.5) * t + rng.uniform(0, np.pi)))
    return x * mod


def _babble(n: int, rng: np.random.Generator, pool: Sequence[np.ndarray]) -> np.ndarray:
    srcs = [t for t in (trim_silence(np.asarray(p, np.float32)) for p in pool) if len(t) > SR // 10]
    if not srcs:
        raise LookupError("babble pool 비어 있음")
    k = int(min(len(srcs), rng.integers(4, 7)))
    pick = rng.choice(len(srcs), size=k, replace=False)
    out = np.zeros(n, np.float64)
    for i in pick:
        s = _unit(srcs[int(i)])
        reps = int(np.ceil(n / len(s))) + 1
        tiled = np.tile(s, reps)
        off = int(rng.integers(0, len(s)))
        out += tiled[off: off + n] * rng.uniform(0.5, 1.0)
    return out


def _hum(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    amps = [1.0, 0.6, 0.45, 0.35, 0.3, 0.25, 0.2, 0.18, 0.16, 0.15]
    x = sum(a * np.sin(2 * np.pi * 60 * (i + 1) * t + rng.uniform(0, 2 * np.pi)) for i, a in enumerate(amps))
    return x + 0.02 * rng.standard_normal(n)


def make_noise(kind: str, n: int, rng: np.random.Generator, pool: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """RMS 1로 정규화한 잡음. babble은 pool(다른 문항의 깨끗한 음성)이 필요하다."""
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        x = _pink(n, rng)
    elif kind == "hum":
        x = _hum(n, rng)
    elif kind == "babble":
        x = _babble(n, rng, pool or [])
    else:
        raise ValueError(kind)
    return _unit(x)


def mix_at_snr(clean: np.ndarray, noise_unit: np.ndarray, snr_db: float, mask: np.ndarray) -> np.ndarray:
    """음성 구간(mask) 전력 기준으로 SNR을 맞춘 잡음을 돌려준다."""
    c = np.asarray(clean, np.float64)
    if not mask.any():
        mask = np.ones(len(c), bool)
    p_speech = float(np.mean(c[mask] ** 2))
    p_noise = float(np.mean(np.asarray(noise_unit, np.float64)[mask] ** 2)) or 1.0
    scale = np.sqrt(p_speech / (10 ** (snr_db / 10)) / p_noise)
    return (noise_unit * scale).astype(np.float32)


# ───────────────────────── 채널 ─────────────────────────
_BIAS, _CLIP = 0x84, 32635


def ulaw_encode(x: np.ndarray) -> np.ndarray:
    """G.711 μ-law 인코딩(int16 → uint8)."""
    v = np.asarray(x, np.int32)
    sign = (v < 0).astype(np.int32)
    mag = np.minimum(np.abs(v), _CLIP) + _BIAS
    exp = np.clip(np.floor(np.log2(mag)).astype(np.int32) - 7, 0, 7)
    mant = (mag >> (exp + 3)) & 0x0F
    return (~((sign << 7) | (exp << 4) | mant) & 0xFF).astype(np.uint8)


def ulaw_decode(b: np.ndarray) -> np.ndarray:
    u = (~np.asarray(b, np.int32)) & 0xFF
    sign = u & 0x80
    exp = (u >> 4) & 0x07
    mant = u & 0x0F
    mag = (((mant << 3) + _BIAS) << exp) - _BIAS
    return np.where(sign != 0, -mag, mag).astype(np.int16)


_PHONE_SOS = butter(6, [300, 3400], btype="band", fs=8000, output="sos")


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), x.dtype)])


def phone_channel(x: np.ndarray, ulaw: bool = False) -> np.ndarray:
    n = len(x)
    y = soxr.resample(np.asarray(x, np.float64), SR, 8000, quality="HQ")
    y = sosfiltfilt(_PHONE_SOS, y)
    if ulaw:
        y = ulaw_decode(ulaw_encode(np.clip(np.round(y), -32768, 32767))).astype(np.float64)
    y = soxr.resample(y, 8000, SR, quality="HQ")
    return _fit(y.astype(np.float32), n)


# ───────────────────────── 이득·패킷 손실 ─────────────────────────
def apply_gain(x: np.ndarray, gain_db: float) -> tuple[np.ndarray, float]:
    y = np.asarray(x, np.float32) * np.float32(10 ** (gain_db / 20)) if gain_db else np.asarray(x, np.float32)
    clipped = float(np.mean((y > 32767) | (y < -32768))) if len(y) else 0.0
    return np.clip(y, -32768, 32767), clipped


def apply_packet_loss(x: np.ndarray, p: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    if p <= 0:
        return x, 0
    n = int(np.ceil(len(x) / PACKET))
    drop = rng.random(n) < p
    m = np.repeat(~drop, PACKET)[: len(x)]
    return x * m, int(drop.sum())


# ───────────────────────── 전체 ─────────────────────────
def augment(clean: np.ndarray, env: dict | Env | None, seed: int = 0,
            babble_pool: Sequence[np.ndarray] | None = None) -> AugResult:
    e = Env.from_dict(env)
    info: dict = {}
    if e.unknown:
        info["unknown_env_keys"] = sorted(e.unknown)
    rng = lambda k: np.random.default_rng([int(seed) & 0xFFFFFFFF, k])  # noqa: E731 단계별 독립 난수

    x = np.asarray(clean, np.float32)
    if e.speed != 1.0:
        method = "ffmpeg" if ffmpeg_available() else "wsola"
        x = time_stretch(x, float(e.speed), method=method)
        info["speed_method"] = method
    lead = int(round(e.lead_ms * SR / 1000))
    tail = int(round(e.tail_ms * SR / 1000))
    x = np.concatenate([np.zeros(lead, np.float32), x, np.zeros(tail, np.float32)])
    speech_end = detect_speech_end(x)

    noise = np.zeros(len(x), np.float32)
    if e.noise != "none" and len(x):
        try:
            nu = make_noise(e.noise, len(x), rng(1), pool=babble_pool)
        except LookupError:
            nu = _unit(_speech_shaped(len(x), rng(1)))
            info["noise_fallback"] = True
        noise = mix_at_snr(x, nu, float(e.snr_db), speech_mask(x))
    y = x + noise

    if e.channel != "wideband":
        y = phone_channel(y, ulaw=(e.channel == "phone_ulaw"))
    y, info["clipped_frac"] = apply_gain(y, float(e.gain_db))
    y, info["dropped_frames"] = apply_packet_loss(y, float(e.packet_loss), rng(2))
    audio = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return AugResult(audio=audio, speech_end=speech_end, clean=x, noise=noise, info=info)
