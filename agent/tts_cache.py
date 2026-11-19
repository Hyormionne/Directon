"""고정 문구 → PCM 캐시(assets/tts_cache/<hash>.pcm). 첫 마디·필러·사과 문구용.

레벨(AG-T09): 측정기는 바이너리 메시지 단위로 RMS ≥ 300을 '소리'로 본다. 그래서
- 합성해 캐시에 넣는 PCM은 음성 구간 RMS를 TARGET_RMS(3000, 약 -21dBFS)로 맞추고(피크 한도로 클리핑 방지),
  앞 무음을 자른 뒤 첫 송신 청크(100ms)의 RMS가 FIRST_CHUNK_MIN_RMS 이상이 되도록 느린 시작을 더 자른다.
- 실시간 문장 TTS는 LevelTTS로 감싸 같은 레벨로 맞춘다(문장마다 잰 레벨의 이동 평균으로 고정 이득).
put()은 받은 바이트를 그대로 저장한다(앞 무음 자르기만). 정규화는 합성 경로(warm·get_or_synth)에서 한다.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from agent.config import Settings
from agent.core.types import FRAME_BYTES, SessionInfo, TTSProvider

log = logging.getLogger(__name__)

TRIM_RMS = 200.0     # 앞쪽 무음 자르기 기준(20ms 프레임 RMS)
TARGET_RMS = 3000.0  # 음성 구간 목표 RMS(테스트 음성과 같은 기준, eval/build_testset.py)
PEAK_LIMIT = 30000.0
MEASURE_RMS = 300.0  # 측정기 무음 기준
FIRST_CHUNK_MIN_RMS = 2 * MEASURE_RMS   # 여유를 두고 첫 청크가 이 이상이게
MAX_ONSET_TRIM_MS = 300                 # 느린 시작을 더 자르는 최대 길이


def _rms(b: bytes) -> float:
    a = np.frombuffer(b[: len(b) - len(b) % 2], dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def trim_leading_silence(pcm: bytes, threshold: float = TRIM_RMS) -> bytes:
    """첫 프레임부터 유성음이 되게 앞 무음 프레임을 자른다. 전부 무음이면 원본."""
    for off in range(0, len(pcm), FRAME_BYTES):
        if _rms(pcm[off:off + FRAME_BYTES]) >= threshold:
            return pcm[off:]
    return pcm


def _i16(pcm: bytes) -> tuple[np.ndarray, bytes]:
    n = len(pcm) // 2 * 2
    return np.frombuffer(pcm[:n], dtype=np.int16), pcm[n:]


def _frame_rms(a: np.ndarray, n: int = FRAME_BYTES // 2) -> np.ndarray:
    m = len(a) // n
    if m == 0:
        return np.zeros(0)
    f = a[: m * n].astype(np.float64).reshape(m, n)
    return np.sqrt(np.mean(f * f, axis=1))


def speech_rms(pcm: bytes) -> float:
    """음성 구간(20ms 프레임 RMS가 상위 레벨의 10% 이상이고 30 이상)의 RMS. 음성이 없으면 0."""
    a, _ = _i16(pcm)
    fr = _frame_rms(a)
    if fr.size == 0:
        return _rms(pcm)
    ref = float(np.percentile(fr, 95))
    voiced = fr >= max(30.0, 0.1 * ref)
    if not voiced.any():
        return 0.0
    return float(np.sqrt(np.mean(fr[voiced] ** 2)))


def apply_gain(pcm: bytes, gain: float, peak_limit: float = PEAK_LIMIT) -> bytes:
    a, rest = _i16(pcm)
    if gain == 1.0 or a.size == 0:
        return pcm
    y = np.clip(np.round(a.astype(np.float64) * gain), -peak_limit, peak_limit).astype(np.int16)
    return y.tobytes() + rest


def normalize_pcm(pcm: bytes, target_rms: float = TARGET_RMS, peak_limit: float = PEAK_LIMIT) -> bytes:
    """음성 구간 RMS를 target_rms로. 피크가 peak_limit를 넘으면 그만큼 이득을 낮춘다. 무음이면 그대로."""
    r = speech_rms(pcm)
    a, _ = _i16(pcm)
    if r <= 0 or a.size == 0:
        return pcm
    g = target_rms / r
    peak = float(np.abs(a.astype(np.float64)).max())
    if peak * g > peak_limit:
        g = peak_limit / peak
    return apply_gain(pcm, g, peak_limit)


def ensure_first_chunk(pcm: bytes, chunk_bytes: int = 3200, min_rms: float = FIRST_CHUNK_MIN_RMS,
                       max_trim_ms: int = MAX_ONSET_TRIM_MS) -> bytes:
    """앞 무음을 자르고, 첫 송신 청크 RMS가 min_rms 미만이면 느린 시작을 프레임 단위로 더 자른다."""
    pcm = trim_leading_silence(pcm)
    max_cut = max_trim_ms * 32                     # 16kHz × 2바이트
    cut = 0
    while cut < max_cut and len(pcm) - cut > chunk_bytes and _rms(pcm[cut:cut + chunk_bytes]) < min_rms:
        cut += FRAME_BYTES
    return pcm[cut:]


def prepare_cached_pcm(pcm: bytes, chunk_bytes: int = 3200, target_rms: float = TARGET_RMS) -> bytes:
    """합성 결과 → 캐시용 PCM(레벨 정규화 + 첫 청크 유성음 보장)."""
    if target_rms and target_rms > 0:
        pcm = normalize_pcm(pcm, target_rms)
    return ensure_first_chunk(pcm, chunk_bytes)


class LevelTTS:
    """TTSProvider 래퍼: 실시간 문장 합성 출력의 음성 레벨을 target_rms로 맞춘다.

    아직 레벨을 모르면 처음 probe_bytes(300ms)를 모아 이득을 정하고, 문장이 끝나면 그 문장 전체 레벨을
    이동 평균(est)에 반영한다. 그 뒤 문장은 est로 이득을 정해 덩어리를 바로 흘려보낸다(TTFB 유지).
    서버 시작 때 캐시 warm이 이 래퍼를 거치므로 첫 실시간 문장부터 버퍼링이 없다.
    """

    def __init__(self, inner: Any, target_rms: float = TARGET_RMS, peak_limit: float = PEAK_LIMIT,
                 probe_bytes: int = 9600, alpha: float = 0.3, min_gain: float = 1 / 16, max_gain: float = 16.0):
        self.inner = inner
        self.target_rms = target_rms
        self.peak_limit = peak_limit
        self.probe_bytes = probe_bytes
        self.alpha = alpha
        self.min_gain, self.max_gain = min_gain, max_gain
        self.est: float | None = None

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _gain(self, r: float) -> float:
        if r <= 0:
            return 1.0
        return min(self.max_gain, max(self.min_gain, self.target_rms / r))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        raw: list[bytes] = []
        gain = None if self.est is None else self._gain(self.est)
        pending = b""
        carry = b""

        def out(b: bytes) -> bytes:
            nonlocal carry
            b = carry + b
            n = len(b) // 2 * 2
            carry = b[n:]
            return apply_gain(b[:n], gain or 1.0, self.peak_limit)

        async for chunk in self.inner.synthesize(text):
            raw.append(chunk)
            if gain is None:
                pending += chunk
                if len(pending) < self.probe_bytes:
                    continue
                r = speech_rms(pending)
                if r <= 0 and len(pending) < 4 * self.probe_bytes:
                    continue                          # 아직 무음뿐이면 조금 더 모은다
                gain = self._gain(r)
                chunk, pending = pending, b""
            data = out(chunk)
            if data:
                yield data
        if pending:
            if gain is None:
                gain = self._gain(speech_rms(pending))
            data = out(pending)
            if data:
                yield data
        if carry:
            yield carry
        r_all = speech_rms(b"".join(raw))
        if r_all > 0:
            self.est = r_all if self.est is None else (1 - self.alpha) * self.est + self.alpha * r_all


async def synthesize_all(tts: TTSProvider, text: str) -> bytes:
    chunks = []
    async for c in tts.synthesize(text):
        chunks.append(c)
    return b"".join(chunks)


class TTSCache:
    def __init__(self, cache_dir: str | Path = "assets/tts_cache", voice: str = "default",
                 chunk_bytes: int = 3200, target_rms: float = TARGET_RMS):
        self.dir = Path(cache_dir)
        self.voice = voice
        self.chunk_bytes = chunk_bytes
        self.target_rms = target_rms
        self._mem: dict[str, bytes] = {}

    def key(self, text: str) -> str:
        return hashlib.sha1(f"{self.voice}\n{text}".encode("utf-8")).hexdigest()[:16]

    def path(self, text: str) -> Path:
        return self.dir / f"{self.key(text)}.pcm"

    def get(self, text: str) -> bytes | None:
        if text in self._mem:
            return self._mem[text]
        p = self.path(text)
        if p.exists():
            data = p.read_bytes()
            self._mem[text] = data
            return data
        return None

    def put(self, text: str, pcm: bytes) -> bytes:
        pcm = trim_leading_silence(pcm)
        self._mem[text] = pcm
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path(text).write_bytes(pcm)
        return pcm

    async def get_or_synth(self, text: str, tts: TTSProvider) -> bytes:
        cached = self.get(text)
        if cached is not None:
            return cached
        return self.put(text, self._prepare(await synthesize_all(tts, text)))

    def _prepare(self, pcm: bytes) -> bytes:
        return prepare_cached_pcm(pcm, self.chunk_bytes, self.target_rms)

    async def warm(self, phrases: list[str] | tuple[str, ...], tts: TTSProvider) -> int:
        """없는 문구만 합성해 저장한다. 합성한 개수를 반환(실패는 경고만)."""
        n = 0
        for text in phrases:
            if self.get(text) is not None:
                continue
            try:
                self.put(text, self._prepare(await synthesize_all(tts, text)))
                n += 1
            except Exception as e:  # noqa: BLE001
                log.warning("tts cache warm failed for %r: %r", text, e)
        return n


def select_opener(info: SessionInfo, settings: Settings) -> str:
    """첫 마디 선택. 기본은 이름 없는 캐시 문구."""
    name = ((info.caller or {}).get("name") or "").strip()
    if settings.opener_use_name and name:
        return f"네, {name} 고객님."
    if info.history and len(settings.openers) > 1:
        return settings.openers[1]
    return settings.openers[0]


# ── 텍스트 기반 첫 마디(AG-T10c) ───────────────────────
# 확정 시점 STT 텍스트의 키워드로 고른다. 사과가 먼저(불만이면서 조회 요청인 말은 사과가 낫다).
APOLOGY_KEYWORDS = (
    "불편", "화나", "화가", "짜증", "실망", "최악", "엉망", "어이없", "황당", "너무하", "답답",
    "파손", "깨져", "깨졌", "깨진", "찢어", "찢겨", "고장", "불량", "하자", "망가", "부서",
    "다쳤", "다쳐", "데었", "데여", "상처", "오배송", "잘못 왔", "잘못 배송", "잘못 보내", "누락", "빠져 있",
    "아직도", "안 왔", "안왔", "안 와요", "안와요", "안 오", "못 받", "늦어", "늦게", "몇 번", "항의", "책임",
)
LOOKUP_KEYWORDS = (
    "조회", "확인해", "확인 좀", "확인하고", "알아봐", "언제 와", "언제 도착", "언제 오", "어디쯤", "배송 상태",
    "배송 조회", "주문 내역", "주문한", "주문했", "취소", "반품", "교환", "환불", "재고", "쿠폰", "주소",
    "배송지", "포인트", "적립", "콜백", "전화 주",
)
_LONG_NUMBER = re.compile(r"\d{4,}|[공영일이삼사오육칠팔구]{4,}")


def select_opener_for_text(text: str, settings: Settings, default: str | None = None) -> str:
    """확정 시점 텍스트 → 첫 마디. 불만·사과 필요 → opener_apology, 조회·처리 요청 → opener_lookup, 그 밖 → default."""
    base = default if default is not None else settings.openers[0]
    t = (text or "").strip()
    if not settings.opener_by_text or not t:
        return base
    if settings.opener_apology and any(k in t for k in APOLOGY_KEYWORDS):
        return settings.opener_apology
    if settings.opener_lookup and (any(k in t for k in LOOKUP_KEYWORDS) or _LONG_NUMBER.search(t.replace(" ", ""))):
        return settings.opener_lookup
    return base
