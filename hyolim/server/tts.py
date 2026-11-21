"""음성합성(Gemini TTS 스트리밍) → 16kHz PCM 조각.

Gemini 스트리밍 출력은 헤더 없는 audio/l16 24kHz mono. 받는 즉시 16kHz 로 바꿔 내보낸다.
응답 JSON 위치가 API 버전에 따라 달라서(inlineData.data / delta.data) base64 오디오를 찾아 읽는다.
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import os
from typing import AsyncIterator

import httpx
import numpy as np
from scipy.signal import resample_poly

from . import config as C

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
STYLE = "차분하고 친절한 콜센터 상담원 말투로, 또박또박 조금 빠르게 읽어 주세요: "


def to16k(pcm24: bytes) -> bytes:
    a = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32)
    if C.TTS_RATE == C.RATE or not len(a):
        return pcm24
    g = np.gcd(C.RATE, C.TTS_RATE)
    b = resample_poly(a, C.RATE // g, C.TTS_RATE // g)
    return np.clip(b, -32768, 32767).astype(np.int16).tobytes()


def _find_audio(obj) -> list[str]:
    out = []
    if isinstance(obj, dict):
        for k in ("inlineData", "inline_data"):
            d = obj.get(k)
            if isinstance(d, dict) and d.get("data"):
                out.append(d["data"])
        if obj.get("type") == "audio" and isinstance(obj.get("data"), str):
            out.append(obj["data"])
        for v in obj.values():
            if isinstance(v, (dict, list)):
                out += _find_audio(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _find_audio(v)
    return out


def fake_pcm(text: str) -> bytes:
    """가짜 TTS: 글자 수에 비례한 길이의 낮은 음."""
    sec = min(6.0, 0.08 * len(text) + 0.2)
    t = np.arange(int(C.RATE * sec)) / C.RATE
    return (np.sin(2 * np.pi * 300 * t) * 6000).astype(np.int16).tobytes()


CACHE_DIR = os.path.join(C.ROOT, "server", "cache", "tts")
CACHE_MAX_CHARS = 300                     # 이보다 긴 문장은 다시 나올 일이 적어 저장하지 않는다
STATS = {"chars": 0, "cached_chars": 0}   # 서버 전체 누적
# 턴별 사용량: 동시에 여러 문항이 돌아도 섞이지 않게 턴(연결)마다 따로 센다
TURN = contextvars.ContextVar("tts_turn_stats", default=None)


def _count(key: str, n: int):
    STATS[key] += n
    t = TURN.get()
    if t is not None:
        t[key] += n


def _cache_path(text: str) -> str:
    key = f"{C.TTS_BACKEND}|{C.TTS_MODEL if C.TTS_BACKEND == 'gemini' else C.OPENAI_TTS_MODEL}|" \
          f"{C.TTS_VOICE if C.TTS_BACKEND == 'gemini' else C.OPENAI_TTS_VOICE}|{text.strip()}"
    return os.path.join(CACHE_DIR, hashlib.sha1(key.encode("utf-8")).hexdigest() + ".pcm")


async def synth_stream(http: httpx.AsyncClient, text: str) -> AsyncIterator[bytes]:
    """같은 문장(템플릿 답·채움말 등)은 디스크에 저장해 두고 다시 쓴다 → TTS 크레딧·지연 절약."""
    if not text.strip():
        return
    if C.FAKE:
        yield fake_pcm(text)
        return
    path = _cache_path(text)
    if os.path.exists(path):
        _count("cached_chars", len(text))
        yield open(path, "rb").read()
        return
    _count("chars", len(text))
    buf = []
    async for chunk in _synth_stream_raw(http, text):
        buf.append(chunk)
        yield chunk
    if buf and len(text) <= CACHE_MAX_CHARS:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = path + ".tmp"
            open(tmp, "wb").write(b"".join(buf))
            os.replace(tmp, path)
        except OSError:
            pass


async def _synth_stream_raw(http: httpx.AsyncClient, text: str) -> AsyncIterator[bytes]:
    if C.TTS_BACKEND == "openai":
        async for c in _openai_stream(http, text):
            yield c
        return
    body = {
        "contents": [{"role": "user", "parts": [{"text": STYLE + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": C.TTS_VOICE}}},
        },
    }
    carry = b""
    async with http.stream("POST", C.TTS_URL.format(model=C.TTS_MODEL), json=body, timeout=15,
                           headers={"x-goog-api-key": C.GEMINI_API_KEY, "Authorization": f"Bearer {C.GEMINI_API_KEY}"}) as r:
        if r.status_code != 200:
            raise RuntimeError(f"TTS {r.status_code}: {(await r.aread())[:300]!r}")
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            for b64 in _find_audio(ev):
                raw = carry + base64.b64decode(b64)
                n = len(raw) // 6 * 6                          # 24k→16k(3:2) 경계 맞추기 (샘플 3개 단위)
                carry = raw[n:]
                if n:
                    yield to16k(raw[:n])
    if carry:
        yield to16k(carry[: len(carry) // 2 * 2])


async def synth(http: httpx.AsyncClient, text: str) -> bytes:
    return b"".join([c async for c in synth_stream(http, text)])


# 임시: OpenAI TTS (Gemini 키가 없을 때 지연 측정용). TTS_BACKEND=openai — pcm 24kHz 스트리밍
async def _openai_stream(http, text: str):
    body = {"model": C.OPENAI_TTS_MODEL, "voice": C.OPENAI_TTS_VOICE, "input": text, "response_format": "pcm",
            "instructions": "한국어 콜센터 상담원처럼 차분하고 친절하게, 또박또박 조금 빠르게 말하세요."}
    carry = b""
    async with http.stream("POST", "https://api.openai.com/v1/audio/speech", json=body, timeout=15,
                           headers={"Authorization": f"Bearer {C.OPENAI_API_KEY}"}) as r:
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI TTS {r.status_code}: {(await r.aread())[:300]!r}")
        async for chunk in r.aiter_bytes():
            raw = carry + chunk
            n = len(raw) // 6 * 6
            carry = raw[n:]
            if n:
                yield to16k(raw[:n])
    if carry:
        yield to16k(carry[: len(carry) // 2 * 2])
