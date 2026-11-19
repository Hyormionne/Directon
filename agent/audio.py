"""PCM 오디오 유틸. 기본 규약은 16kHz s16le mono bytes(agent.core.types)."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import soxr

from agent.core.types import FRAME_BYTES, SAMPLE_RATE


def pcm_to_float(pcm: bytes) -> np.ndarray:
    """s16le bytes → float32 [-1, 1)."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm(x: np.ndarray) -> bytes:
    """float [-1, 1] → s16le bytes(범위 밖은 자른다)."""
    y = np.round(np.asarray(x, dtype=np.float64) * 32768.0)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


def rms(pcm: bytes) -> float:
    """int16 단위 RMS. 빈 입력은 0."""
    if len(pcm) < 2:
        return 0.0
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def frames(pcm: bytes, frame_bytes: int = FRAME_BYTES) -> list[bytes]:
    """frame_bytes(기본 20ms=640B) 단위로 자른다. 마지막 조각은 0으로 채운다."""
    out = [pcm[i:i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
    if out and len(out[-1]) < frame_bytes:
        out[-1] = out[-1] + b"\x00" * (frame_bytes - len(out[-1]))
    return out


def trim_silence(pcm: bytes, thresh: float = 100.0, frame_bytes: int = FRAME_BYTES) -> bytes:
    """앞뒤에서 RMS < thresh인 프레임을 걷어낸다(가운데 쉼은 둔다). 전부 무음이면 b""."""
    n = len(pcm) // 2 * 2
    pcm = pcm[:n]
    starts = list(range(0, n, frame_bytes))
    loud = [rms(pcm[s:s + frame_bytes]) >= thresh for s in starts]
    if not any(loud):
        return b""
    first = loud.index(True)
    last = len(loud) - 1 - loud[::-1].index(True)
    return pcm[starts[first]:min(starts[last] + frame_bytes, n)]


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """soxr(HQ)로 샘플레이트 변환. 24k→16k면 길이 2/3."""
    if src_rate == dst_rate or not pcm:
        return pcm
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16)
    y = soxr.resample(x, src_rate, dst_rate, quality="HQ")
    return np.asarray(y, dtype=np.int16).tobytes()


def wav_read(path: str | Path) -> tuple[bytes, int]:
    """16-bit PCM wav → (mono s16le bytes, sample_rate). 다채널은 평균해 mono로."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"16-bit PCM wav만 지원: {path} (sampwidth={w.getsampwidth()})")
        ch, sr = w.getnchannels(), w.getframerate()
        data = w.readframes(w.getnframes())
    if ch > 1:
        x = np.frombuffer(data, dtype=np.int16).reshape(-1, ch).astype(np.int32).mean(axis=1)
        data = np.round(x).astype(np.int16).tobytes()
    return data, sr


def wav_write(path: str | Path, pcm: bytes, rate: int = SAMPLE_RATE) -> None:
    """mono s16le bytes → wav."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
