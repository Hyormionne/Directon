"""테스트 음성 합성: Gemini TTS / edge-tts + 해시 캐시 (AD3=A).

voice 형식:
  "gemini:Kore"                               — Gemini TTS 기본 모델
  "gemini:Puck@gemini-3.8-flash-lite-tts"     — 모델 지정(@ 뒤)
  "edge:ko-KR-SunHiNeural"                    — edge-tts(무료, 키 불필요)

결과는 16kHz s16le mono. `<cache_dir>/<sha256>.pcm`(+ 같은 이름 .json 메모)에 캐시한다.
voice_style(말투 지시, 예: "급하게", "작은 목소리로")은 Gemini에만 적용, edge에서는 무시한다.

백엔드 계약(주입 가능 — 테스트는 fake):
  gemini_backend(text, voice_name, style|None, model|None) -> (int16 ndarray, sample_rate)
  edge_backend(text, voice_name) -> (int16 ndarray, sample_rate)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soxr

SR = 16000
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data" / "testset" / "cache"
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash-tts"
GEMINI_FALLBACK_MODELS = ("gemini-3.8-flash-lite-tts", "gemini-2.5-flash-preview-tts")
# 말투 지시를 소리 내어 읽어 버리는 모델(2026-11-19 실측: "세인 이 스타일 노인 화자처럼…"으로 낭독)
STYLE_UNSAFE_MODELS = {"gemini-3.8-flash-lite-tts"}
CACHE_VERSION = 2   # 2: Gemini audio/wav 헤더·메타 청크 제거
STYLE_PROMPT_VERSION = 2   # style_prompt 형식이 바뀌면 올린다(말투 지정 항목만 무효화)


def parse_voice(voice: str) -> tuple[str, str, str | None]:
    if not isinstance(voice, str) or ":" not in voice:
        raise ValueError(f"voice 형식 오류: {voice!r} (예: gemini:Kore, edge:ko-KR-SunHiNeural)")
    engine, rest = voice.split(":", 1)
    if engine not in ("gemini", "edge"):
        raise ValueError(f"지원하지 않는 엔진: {engine!r}")
    name, _, model = rest.partition("@")
    if not name:
        raise ValueError(f"voice 이름이 비었다: {voice!r}")
    if engine == "edge" and model:
        raise ValueError("edge 음성은 모델 지정(@)을 지원하지 않는다")
    return engine, name, (model or None)


def count_syllables(text: str) -> float:
    return (len(re.findall(r"[가-힣]", text)) + 1.5 * len(re.findall(r"[0-9]", text))
            + 0.5 * len(re.findall(r"[A-Za-z]", text)))


def max_duration(text: str, style: str | None = None) -> float:
    """합성 길이 상한(초). 넘으면 말투 지시문을 읽었거나 반복한 것으로 보고 다시 합성한다."""
    return 1.5 + count_syllables(text) * (0.6 if style else 0.45)


def style_prompt(text: str, style: str | None) -> str:
    """Gemini TTS 말투 지시. 지시문을 읽어 버리지 않도록 발화를 분리한다."""
    if not style:
        return text
    # 2026-11-19 실측: "Read the following ... Utterance: ..." 형식은 flash-lite-tts가 지시문까지 읽어
    # 10음절 문장이 17초가 됐다. 문서식 'Say ...: "..."' 형식은 4.4초로 정상.
    return f'Say in this style ({style}): "{text}"'


# ───────────────────────── 백엔드 ─────────────────────────
_NON_RETRYABLE = re.compile(r"\b(400|401|403|404)\b|INVALID_ARGUMENT|PERMISSION_DENIED|UNAUTHENTICATED|NOT_FOUND")
_RETRY_DELAY = re.compile(r"retryDelay'?\"?\s*:\s*'?\"?(\d+(?:\.\d+)?)s")
MAX_RETRY_DELAY = 120.0


def is_daily_quota(e: Exception) -> bool:
    """일일 한도 소진(무료 등급 TTS는 모델당 하루 10회) 또는 긴 대기 요구 → 기다려도 소용없다."""
    msg = str(e)
    if "PerDay" in msg:
        return True
    m = _RETRY_DELAY.search(msg)
    return bool(m and float(m.group(1)) > MAX_RETRY_DELAY)


def is_retryable(e: Exception) -> bool:
    """429(짧은 대기)·5xx·네트워크 오류는 재시도. 잘못된 요청(4xx)·일일 한도 소진은 재시도하지 않는다."""
    msg = str(e)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return not is_daily_quota(e)
    return not _NON_RETRYABLE.search(msg)


def parse_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """RIFF/WAVE를 청크 단위로 읽는다(data 청크만 PCM, 뒤에 붙은 메타데이터 청크는 버린다)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("WAV 아님")
    pos, rate, bits, ch, pcm = 12, 24000, 16, 1, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8: pos + 8 + size]
        if cid == b"fmt ":
            _, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if pcm is None:
        raise RuntimeError("WAV에 data 청크가 없다")
    if bits != 16:
        raise RuntimeError(f"16비트 PCM만 지원(받은 값 {bits})")
    a = np.frombuffer(pcm[: len(pcm) // 2 * 2], "<i2")
    if ch > 1:
        a = a[: len(a) // ch * ch].reshape(-1, ch).mean(axis=1).astype(np.int16)
    return a.copy(), rate


def decode_audio_bytes(data: bytes, mime_type: str | None) -> tuple[np.ndarray, int]:
    """Gemini TTS inline_data → (int16, rate). audio/wav(헤더+메타 청크)와 audio/L16(raw) 모두 처리."""
    if data[:4] == b"RIFF":
        return parse_wav_bytes(data)
    m = re.search(r"rate=(\d+)", mime_type or "")
    rate = int(m.group(1)) if m else 24000
    return np.frombuffer(data[: len(data) // 2 * 2], "<i2").copy(), rate


class GeminiTTS:
    """google-genai 기반 Gemini TTS. 24kHz s16le PCM을 돌려준다."""

    def __init__(self, client=None, default_model: str = GEMINI_DEFAULT_MODEL,
                 fallback_models: tuple[str, ...] = GEMINI_FALLBACK_MODELS):
        self._client = client
        self.default_model = default_model
        self.fallback_models = fallback_models
        self.exhausted: set[str] = set()              # 이번 실행에서 일일 한도가 소진된 모델

    @property
    def client(self):
        if self._client is None:
            from google import genai
            # TTS의 speechConfig는 v1beta에만 있다(SDK 기본 v1이면 400 "Unknown name speechConfig").
            self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
                                        http_options={"api_version": "v1beta"})
        return self._client

    def _config(self, voice_name: str):
        from google.genai import types
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name))))

    def _once(self, model: str, contents: str, voice_name: str):
        try:
            config = self._config(voice_name)
        except ImportError:  # pragma: no cover
            config = None
        resp = self.client.models.generate_content(model=model, contents=contents, config=config)
        for part in resp.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                return decode_audio_bytes(inline.data, inline.mime_type)
        raise RuntimeError("Gemini TTS 응답에 오디오가 없다")

    def __call__(self, text: str, voice_name: str, style: str | None, model: str | None):
        contents = style_prompt(text, style)
        if model:                                     # 명시한 모델은 대체하지 않는다
            return self._once(model, contents, voice_name)
        last = None
        for m in (self.default_model, *self.fallback_models):
            if m in self.exhausted or (style and m in STYLE_UNSAFE_MODELS):
                continue
            try:
                return self._once(m, contents, voice_name)
            except Exception as e:  # 모델 미지원(404)·일일 한도 소진이면 다음 모델, 그 외는 위로(재시도 대상)
                msg = str(e)
                if ("404" in msg or "NOT_FOUND" in msg or "not supported" in msg.lower()
                        or is_daily_quota(e)):
                    if is_daily_quota(e):
                        self.exhausted.add(m)
                    last = e
                    continue
                raise
        raise RuntimeError(f"Gemini TTS 모델 모두 실패: {last}")


def decode_with_ffmpeg(data: bytes) -> tuple[np.ndarray, int]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
           "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"]
    p = subprocess.run(cmd, input=data, capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.int16).copy(), SR


class EdgeTTS:
    """edge-tts(MP3 스트림) → ffmpeg로 16kHz PCM 디코드."""

    def __init__(self, communicate_cls=None, decoder: Callable[[bytes], tuple[np.ndarray, int]] | None = None):
        self._comm = communicate_cls
        self.decoder = decoder or decode_with_ffmpeg

    async def _collect(self, text: str, voice_name: str) -> bytes:
        cls = self._comm
        if cls is None:
            import edge_tts
            cls = edge_tts.Communicate
        buf = bytearray()
        async for chunk in cls(text, voice_name).stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                buf += chunk["data"]
        return bytes(buf)

    def __call__(self, text: str, voice_name: str):
        data = asyncio.run(self._collect(text, voice_name))
        if not data:
            raise RuntimeError("edge-tts 응답에 오디오가 없다")
        return self.decoder(data)


# ───────────────────────── 합성기 ─────────────────────────
class Synthesizer:
    def __init__(self, cache_dir: str | os.PathLike = DEFAULT_CACHE, gemini_backend=None, edge_backend=None,
                 max_gemini_concurrency: int = 2, max_edge_concurrency: int = 4, retries: int = 5,
                 backoff: float = 2.0, sleep: Callable[[float], None] = time.sleep):
        self.cache_dir = Path(cache_dir)
        self._gemini = gemini_backend
        self._edge = edge_backend
        self._sem = {"gemini": threading.BoundedSemaphore(max_gemini_concurrency),
                     "edge": threading.BoundedSemaphore(max_edge_concurrency)}
        self.retries, self.backoff, self.sleep = retries, backoff, sleep
        self.stats = {"hit": 0, "miss": 0, "retry": 0}
        self._lock = threading.Lock()

    @property
    def gemini(self):
        if self._gemini is None:
            self._gemini = GeminiTTS()
        return self._gemini

    @property
    def edge(self):
        if self._edge is None:
            self._edge = EdgeTTS()
        return self._edge

    @staticmethod
    def cache_key(text: str, voice: str, voice_style: str | None = None) -> str:
        engine, name, model = parse_voice(voice)
        style = voice_style if engine == "gemini" else None
        d = {"v": CACHE_VERSION, "engine": engine, "voice": name, "model": model,
             "style": style or None, "text": text}
        if style:
            d["style_prompt_v"] = STYLE_PROMPT_VERSION
        blob = json.dumps(d, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def cache_path(self, text: str, voice: str, voice_style: str | None = None) -> Path:
        return self.cache_dir / f"{self.cache_key(text, voice, voice_style)}.pcm"

    def synthesize(self, text: str, voice: str, voice_style: str | None = None) -> np.ndarray:
        engine, name, model = parse_voice(voice)
        path = self.cache_path(text, voice, voice_style)
        limit = max_duration(text, voice_style if engine == "gemini" else None)
        if path.exists() and 0 < path.stat().st_size <= limit * SR * 2:   # 길이 의심 캐시는 다시 합성
            with self._lock:
                self.stats["hit"] += 1
            return np.fromfile(path, dtype=np.int16)
        with self._lock:
            self.stats["miss"] += 1
        pcm = self._call_with_retry(engine, name, model, text, voice_style)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{threading.get_ident()}")
        pcm.astype("<i2").tofile(tmp)
        os.replace(tmp, path)
        path.with_suffix(".json").write_text(json.dumps(
            {"text": text, "voice": voice, "voice_style": voice_style if engine == "gemini" else None,
             "samples": int(len(pcm))}, ensure_ascii=False), encoding="utf-8")
        return pcm

    def _call_with_retry(self, engine, name, model, text, style) -> np.ndarray:
        last = None
        for attempt in range(self.retries + 1):
            try:
                with self._sem[engine]:
                    if engine == "gemini":
                        raw, rate = self.gemini(text, name, style, model)
                    else:
                        raw, rate = self.edge(text, name)
                raw = np.asarray(raw, np.int16)
                if len(raw) == 0:
                    raise RuntimeError("합성 결과가 비었다")
                if rate != SR:
                    raw = soxr.resample(raw.astype(np.float32), rate, SR, quality="HQ")
                    raw = np.clip(np.round(raw), -32768, 32767).astype(np.int16)
                limit = max_duration(text, style if engine == "gemini" else None)
                if len(raw) / SR > limit:
                    raise RuntimeError(f"합성 길이 의심: {len(raw) / SR:.1f}s > 상한 {limit:.1f}s")
                return raw
            except (ValueError, TypeError):
                raise
            except Exception as e:  # 레이트리밋·일시 오류 → 지수 백오프
                if not is_retryable(e):
                    raise RuntimeError(f"{engine}:{name} 합성 실패(재시도 안 함): {e!r}") from e
                last = e
                if attempt < self.retries:
                    with self._lock:
                        self.stats["retry"] += 1
                    delay = min(60.0, self.backoff * (2 ** attempt)) * (0.75 + 0.5 * random.random())
                    self.sleep(delay)
        raise RuntimeError(f"{engine}:{name} 합성 실패({self.retries + 1}회): {last!r}")


_default: Synthesizer | None = None


def synthesize(text: str, voice: str, voice_style: str | None = None) -> np.ndarray:
    """기본 합성기(data/testset/cache)로 합성. 16kHz int16 배열."""
    global _default
    if _default is None:
        _default = Synthesizer()
    return _default.synthesize(text, voice, voice_style)
